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
    training_args = None
    if "RLOOConfig" in params :
        training_args = RLOOConfig(
            output_dir=os.path.join("savings/", model_args.model_name_or_path, script_args.dataset_name),
            logging_dir=os.path.join("savings/logs/", model_args.model_name_or_path, script_args.dataset_name),
            **params["RLOOConfig"]
            )
    elif "PPOConfig" in params :
        training_args = PPOConfig(
            output_dir=os.path.join("savings/", model_args.model_name_or_path, script_args.dataset_name),
            logging_dir=os.path.join("savings/logs/", model_args.model_name_or_path, script_args.dataset_name),
            **params["PPOConfig"]
        )

    framework_params = json_file["framework"][framework_params]
    use_extra_rewards = framework_params["use_extra_rewards"]
    reward_functions = []
    if use_extra_rewards:
        reward_functions = [globals()[reward_name] for reward_name in framework_params["reward_functions"]]

    return training_args, model_args, script_args,framework_params, reward_functions
