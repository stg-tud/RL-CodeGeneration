import argparse
import json
import os

from transformers import AutoTokenizer
from trl import ModelConfig, ScriptArguments, SFTConfig

from sft.sft_dataset_utils import get_robo_instruct_sft_dataset, get_open_code_instruct_sft_dataset
from sft.sft_wrapper import SFTWrapper

# Cache location for HuggingFace models/datasets, same convention as main.py.
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Maps a variant's ScriptArguments.dataset_name to the dataset-prep function to use -
# mirrors rl_datasets/dataset_utils.py::get_dataset's own dataset_name-based dispatch.
DATASET_BUILDERS = {
    "zichao22/robo-instruct": get_robo_instruct_sft_dataset,
    "nvidia/OpenCodeInstruct": get_open_code_instruct_sft_dataset,
}


def parse_sft_hyperparams(path: str, variant: str = None):
    all_params = json.load(open(path))
    if variant is None:
        # Default/original recipe: top-level ScriptArguments/ModelConfig/SFTConfig,
        # unchanged since before variants existed - never nest or rename these.
        params = all_params
    else:
        if variant not in all_params:
            raise ValueError(f"Unknown --variant {variant!r}; available: "
                              f"{[k for k in all_params if isinstance(all_params[k], dict) and 'ModelConfig' in all_params[k]]}")
        params = all_params[variant]

    script_args = ScriptArguments(**params["ScriptArguments"])
    model_args = ModelConfig(**params["ModelConfig"])

    # Optional extra path segment (e.g. "1.5b"/"7b") so checkpoints for different model
    # sizes of the same domain/dataset don't collide (evaluate_passk's save_dir has no
    # model-name segment). Absent for the default recipe to keep its existing layout unchanged.
    checkpoint_subdir = params.get("checkpoint_subdir")
    output_dir_parts = ["savings", model_args.model_name_or_path, script_args.dataset_name, "sft"]
    logging_dir_parts = ["savings", "logs", model_args.model_name_or_path, script_args.dataset_name, "sft"]
    if checkpoint_subdir:
        output_dir_parts.append(checkpoint_subdir)
        logging_dir_parts.append(checkpoint_subdir)

    sft_args = SFTConfig(
        output_dir=os.path.join(*output_dir_parts),
        logging_dir=os.path.join(*logging_dir_parts),
        **params["SFTConfig"],
    )
    return script_args, model_args, sft_args


if __name__ == "__main__":
    os.environ["TRANSFORMERS_CACHE"] = HF_HOME
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_CACHE"] = HF_HOME

    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, default=os.path.join("sft", "sft_hyperparams.json"))
    parser.add_argument("--variant", type=str, default=None,
                         help="Named block in --hyperparams to use, e.g. code_gen_1.5b. Leave unset for the "
                              "default/original recipe (top-level ScriptArguments/ModelConfig/SFTConfig).")
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Override total training steps, for smoke-testing (e.g. --max_steps 1). "
                              "Leave unset for a real run driven by num_train_epochs.")
    parser.add_argument("--adapter_path", type=str, default=None,
                         help="Resume training from an existing adapter checkpoint's policy/ directory.")
    args = parser.parse_args()

    script_args, model_args, sft_args = parse_sft_hyperparams(args.hyperparams, args.variant)
    if args.max_steps is not None:
        sft_args.max_steps = args.max_steps

    if script_args.dataset_name not in DATASET_BUILDERS:
        raise ValueError(f"No SFT dataset builder registered for dataset_name={script_args.dataset_name!r}; "
                          f"available: {list(DATASET_BUILDERS.keys())}")
    dataset = DATASET_BUILDERS[script_args.dataset_name](model_args.model_name_or_path, cache_dir=HF_HOME)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=HF_HOME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    trainer = SFTWrapper(
        model_args=model_args,
        sft_args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        cache_dir=HF_HOME,
        adapter_path=args.adapter_path,
    )
    trainer.train()
    # Save under a checkpoint-N subfolder (matching Trainer's own mid-training
    # checkpoint layout) so the final adapter is reachable via the same
    # `--checkpoint sft/checkpoint-N` convention as any intermediate checkpoint.
    final_checkpoint_dir = os.path.join(sft_args.output_dir, f"checkpoint-{trainer.state.global_step}")
    trainer.save_model(final_checkpoint_dir)
