import json
import os
from trl import ModelConfig, RLOOConfig, ScriptArguments, PPOConfig
from rewards.extra_rewards import *

from huggingface_hub import snapshot_download

def manual_download(model_or_tokenizer_name, cache_dir):
    snapshot_download(
        repo_id=model_or_tokenizer_name,
        cache_dir=cache_dir,  # optional    # optional
        local_dir_use_symlinks=False              # if you want full copies
    )

def get_deepspeed(path="config/deepspeed.json"):
    return json.load(open(path))

def parse_json_file(path="config/hyperparams.json", params="ppo", framework_params="code_gen"):
    json_file = json.load(open(path))

    params = json_file[params]
    script_args = ScriptArguments(**params["ScriptArguments"])
    model_args = ModelConfig(**params["ModelConfig"])

    framework_params = json_file["framework"][framework_params]
    # "token": rewards are attributed to the tokens they concern (default).
    # "sequence": the summed non-KL reward of a response is placed on its last token.
    # The REWARD_GRANULARITY environment variable overrides the config file.
    granularity = os.environ.get("REWARD_GRANULARITY", framework_params.get("reward_granularity", "token"))
    if granularity not in ("token", "sequence"):
        raise ValueError(f"reward_granularity must be 'token' or 'sequence', got '{granularity}'")
    framework_params["reward_granularity"] = granularity

    # PPOWrapper deletes output_dir at startup, so each granularity needs its own directory.
    # The sequence-level run uses a sibling directory (not a subdirectory) so that neither
    # run can delete the other's checkpoints; the token-level paths stay unchanged.
    run_suffix = "" if granularity == "token" else "__sequence_level"
    output_dir = os.path.join("savings/", model_args.model_name_or_path, script_args.dataset_name + run_suffix)
    logging_dir = os.path.join("savings/logs/", model_args.model_name_or_path, script_args.dataset_name + run_suffix)

    training_args = None
    if "RLOOConfig" in params :
        training_args = RLOOConfig(
            output_dir=output_dir,
            logging_dir=logging_dir,
            **params["RLOOConfig"]
            )
    elif "PPOConfig" in params :
        training_args = PPOConfig(
            output_dir=output_dir,
            logging_dir=logging_dir,
            **params["PPOConfig"]
        )

    # TRL builds run_name from exp_name in PPOTrainer.__init__, so the sequence-level run
    # is tagged there to keep the two runs distinguishable in wandb.
    if training_args is not None and granularity == "sequence":
        training_args.exp_name = f"{training_args.exp_name}__sequence_level"
    use_extra_rewards = framework_params["use_extra_rewards"]
    reward_functions = []
    if use_extra_rewards:
        reward_functions = [globals()[reward_name] for reward_name in framework_params["reward_functions"]]

    return training_args, model_args, script_args,framework_params, reward_functions
