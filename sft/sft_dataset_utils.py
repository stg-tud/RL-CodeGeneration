from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from rl_datasets.dataset_special_functions import robo_instruct_tokenization_function, open_code_instruct_special_tokenization_function


def get_robo_instruct_sft_dataset(model_or_tokenizer_name: str, cache_dir: str, seed: int = 42) -> DatasetDict:
    """
    Builds a single-column ("text") causal-LM SFT dataset from zichao22/robo-instruct.

    Reuses robo_instruct_tokenization_function (rl_datasets/dataset_special_functions.py)
    as-is to render the few-shot chat-template prompt and clean the target code - the same
    function the PPO training path uses, so this matches PPO's training-time prompt format.

    NOTE: this does NOT match eval/roboeval.py's own prompt format (its update_prompt()
    uses 2 few-shot examples at indices [0,3,4,7,8] and a 1024 max_length, vs this
    function's 1 example / indices [0,1,2] / 512 max_length inherited from
    robo_instruct_tokenization_function). That train/eval prompt mismatch is pre-existing
    in the framework and affects PPO too, so it is kept here to stay comparable with PPO.
    """
    raw_dataset = load_dataset("zichao22/robo-instruct", cache_dir=cache_dir)["train"]
    split = raw_dataset.train_test_split(test_size=0.2, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_or_tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def format_split(dataset):
        input_sequences = list(dataset["text"])
        target_sequences = list(dataset["program"])

        # Mutates input_sequences/target_sequences in place: renders the few-shot
        # chat-template prompt and strips NEW_LINE/INDENT/DEDENT from the target code.
        robo_instruct_tokenization_function(input_sequences, target_sequences, tokenizer)

        # Newline after ```python matches the few-shot demonstration's fencing
        # style (its opening fence has one baked into the source example text);
        # no newline before the closing ``` matches that same precedent too.
        texts = [
            prompt + "```python\n" + code + "```" + tokenizer.eos_token
            for prompt, code in zip(input_sequences, target_sequences)
        ]
        return Dataset.from_dict({"text": texts})

    return DatasetDict({
        "train": format_split(split["train"]),
        "test": format_split(split["test"]),
    })


def get_open_code_instruct_sft_dataset(model_or_tokenizer_name: str, cache_dir: str, seed: int = 42,
                                        train_size: int = 5000, test_size: int = 500) -> DatasetDict:
    """
    Builds a single-column ("text") causal-LM SFT dataset from nvidia/OpenCodeInstruct.

    Reuses open_code_instruct_special_tokenization_function (rl_datasets/dataset_special_functions.py)
    as-is - the same function PPO's own training path uses (rl_datasets/dataset_utils.py::
    get_open_code_instruct_dataset) - so this matches PPO's training-time prompt format
    (see get_robo_instruct_sft_dataset's docstring above for the same caveat re: eval's own,
    separately-defined prompt format in eval/evalplus_pass_k.py::update_prompt).

    Also reuses that same loader's domain=='generic'/non-empty-unit_tests filter (a data-quality
    filter; `unit_tests` itself is PPO-reward-specific and unused here). OpenCodeInstruct is
    loaded with streaming=True, matching the PPO-side loader, since materializing the whole
    dataset isn't necessary - only train_size+test_size rows (after filtering) are taken.
    """
    filter_fn = lambda x: x['domain'] == 'generic' and x['unit_tests'] is not None and len(x['unit_tests']) > 0

    raw_dataset = load_dataset("nvidia/OpenCodeInstruct", cache_dir=cache_dir, streaming=True)["train"]
    raw_dataset = raw_dataset.filter(filter_fn)
    raw_dataset = raw_dataset.shuffle(seed=seed, buffer_size=10000)

    rows = list(raw_dataset.take(train_size + test_size))
    train_rows, test_rows = rows[:train_size], rows[train_size:]

    tokenizer = AutoTokenizer.from_pretrained(model_or_tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    def format_rows(rows):
        input_sequences = [row["input"] for row in rows]
        target_sequences = [row["output"] for row in rows]

        # Mutates input_sequences/target_sequences in place: renders the chat-template
        # prompt and strips NEW_LINE/INDENT/DEDENT from the target code.
        open_code_instruct_special_tokenization_function(input_sequences, target_sequences, tokenizer)

        # Unlike robo-instruct's "program" field (raw, unfenced code), OpenCodeInstruct's
        # "output" field already comes wrapped in its own ```python...``` fence as part of
        # the dataset's natural response format (confirmed on real data - wrapping it again
        # here produced visible double-fencing, e.g. "```python```pythondef f():...``````").
        # Only add a fence for the rare row that doesn't already have one.
        def ensure_fenced(code):
            return code if code.strip().startswith("```") else "```python\n" + code + "```"

        texts = [
            prompt + ensure_fenced(code) + tokenizer.eos_token
            for prompt, code in zip(input_sequences, target_sequences)
        ]
        return Dataset.from_dict({"text": texts})

    return DatasetDict({
        "train": format_rows(train_rows),
        "test": format_rows(test_rows),
    })
