from transformers import GenerationConfig, LogitsProcessorList, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification, StoppingCriteriaList
from transformers.data.data_collator import DataCollatorWithPadding
import torch
from trl.trainer.utils import pad
from trl import get_peft_config, create_reference_model, AutoModelForCausalLMWithValueHead
from peft import PeftModel

def score(self, hidden_states):
    """
    Function to extract value logits from the final hidden states.

    The TRL PPO trainer expects a .score() method to exist on the model.
    """
    # The value head is typically a single linear layer named 'v_head'
    return self.v_head(hidden_states)

def load_model(cache_dir, model_args, adapter_path : str = None):

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                 trust_remote_code=model_args.trust_remote_code,
                                                 dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 cache_dir=cache_dir
                                                 # device_map="auto"
                                                 )
    critic = AutoModelForCausalLMWithValueHead.from_pretrained(model_args.model_name_or_path,
                                                                trust_remote_code=model_args.trust_remote_code,
                                                                dtype=torch.bfloat16,
                                                                low_cpu_mem_usage=True,
                                                                cache_dir=cache_dir,
                                                                )

    if getattr(critic, 'base_model_prefix', None) is None:
        critic.base_model_prefix = 'pretrained_model'

    if not hasattr(critic, 'score'):
        import types
        critic.score = types.MethodType(score, critic)

    if adapter_path is not None:
        peft_config = None
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    else:
        peft_config = get_peft_config(model_args)

    if peft_config is None and adapter_path is None:
        ref_policy = create_reference_model(model, num_shared_layers=4)
    else:
        ref_policy = None

    return ref_policy, model, critic, peft_config

@torch.no_grad()
def adjusted_batch_generation(model: torch.nn.Module,
                      queries: torch.Tensor,
                      local_rollout_forward_batch_size: int,
                      pad_token_id: int,
                      generation_config: GenerationConfig,
                      logits_processor: LogitsProcessorList = None,
                      stopping_criteria: StoppingCriteriaList = None,
                      labels: torch.Tensor = None
                      ):
    query_responses = []
    logitss = []
    syncode_scores = []
    batch_size = queries.shape[0]

    for i in range(0, batch_size, local_rollout_forward_batch_size):
        query = queries[i: i + local_rollout_forward_batch_size]

        if logits_processor is not None and labels is not None:
            label = labels[i: i + local_rollout_forward_batch_size]
            logits_processor[0].set_labels(label)

        query_response, logits = generate(
            model,
            query,
            pad_token_id,
            generation_config,
            logits_processor,
            stopping_criteria,
        )
        query_responses.append(query_response)
        logitss.append(logits)
        if logits_processor is not None:
            syncode_scores.append(logits_processor[0].get_scores())

    # padding tensorseval_dataloader
    padded_query_responses = pad(query_responses, padding_value=pad_token_id, padding_side="right")
    padded_logitss = pad(logitss, padding_value=0, padding_side="right")

    # reshaping
    padded_query_responses = padded_query_responses.view(-1, padded_query_responses.shape[-1])[:batch_size]
    padded_logitss = padded_logitss.view(-1, *padded_logitss.shape[2:])[:batch_size]

    padded_syncode_scores = None
    if logits_processor is not None:
        padded_syncode_scores = pad(syncode_scores, padding_value=0, padding_side="right")
        padded_syncode_scores = padded_syncode_scores.view(-1, padded_syncode_scores.shape[-1])[:batch_size]

    return padded_query_responses, padded_logitss, padded_syncode_scores


def generate(lm_backbone: torch.nn.Module, queries: torch.Tensor, pad_token_id: int,
              generation_config: GenerationConfig, logits_processor: LogitsProcessorList = None, stopping_criteria: StoppingCriteriaList = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates sequences from the language model backbone in a way that does not affect padding tokens.

    Args:
        lm_backbone (`torch.nn.Module`):
            The language model backbone used for generation.
        queries (`torch.Tensor`):
            The tensor containing the input queries.
        pad_token_id (`int`):
            The token ID representing the pad token.
        generation_config (`GenerationConfig`):
            The configuration for the generation process.

    Returns:
        tuple:
            - `generated_sequences` (`torch.Tensor`):
                The concatenated tensor of input queries and generated sequences.
            - `logits` (`torch.Tensor`):
                The logits output from the generation process.
    """
    context_length = queries.shape[1]
    attention_mask = queries != pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)

    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        logits_processor=logits_processor,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria = stopping_criteria,
    )
    logits = torch.stack(output.scores, 1)
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1), logits


def decode_batch(batch, processing_class):
    return processing_class.batch_decode(batch, skip_special_tokens=True, clean_up_tokenization_spaces=False)


class StringAdaptedDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, padding=True):  # dynamic padding
        super().__init__(tokenizer, padding=padding)

    def __call__(self, features):
        unit_tests = [f.pop("unit_tests", None) for f in features]
        batch = super().__call__(features)  # pads to longest in batch
        if any(ut is not None for ut in unit_tests):
            batch["unit_tests"] = unit_tests
        return batch

