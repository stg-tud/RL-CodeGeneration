from unsloth import FastLanguageModel
from transformers import TrainingArguments, logging
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer, SFTConfig
import os 
import torch 
import hydra 
from omegaconf import DictConfig, OmegaConf
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    training_args = SFTConfig(**cfg.training_args)
    os.environ["WANDB_PROJECT"] = cfg.misc_args.wandb_project_name  # name your W&B project
    if cfg.misc_args.verbose:
        logging.set_verbosity_debug()
    if cfg.data_args.load_from_disk:
        print(cfg.data_args.train_data_name_or_path)
        train_dataset = load_from_disk(cfg.data_args.train_data_name_or_path)
    else:
        train_dataset = load_dataset(cfg.data_args.train_data_name_or_path, split="train")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.model_args.model_name_or_path,
        max_seq_length = cfg.misc_args.seq_length,
        load_in_4bit=False
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.model_args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = cfg.model_args.lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = cfg.misc_args.seq_length,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    dataset_text_field = cfg.data_args.text_column
    data_collator =None

    tokenizer.pad_token = tokenizer.eos_token
    # Must add EOS_TOKEN
    def add_eos_to_batch(examples):
        examples[cfg.data_args.text_column] = [
            text + tokenizer.eos_token for text in examples[cfg.data_args.text_column]
        ]
        return examples
    train_dataset = train_dataset.map(add_eos_to_batch, batched=True)
    
    training_args.packing = cfg.misc_args.packing
    training_args.max_seq_length = cfg.misc_args.seq_length
    training_args.dataset_text_field = dataset_text_field
    trainer = SFTTrainer(
        model=model,
        args = training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()