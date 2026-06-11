import argparse
import os
from eval.roboeval import generate_evaluate as roboEval
from eval.evalplus_pass_k import generate_evaluate as evalplusPassK
from wrappers.ppo_wrapper import PPOWrapper
from rl_datasets.dataset_utils import *
from utils import *

# Cache location for HuggingFace models/datasets.
# if set, otherwise fall back to the standard HuggingFace default (~/.cache/huggingface).
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

if __name__ == '__main__':
    os.environ["TRANSFORMERS_CACHE"] = HF_HOME
    os.environ['HF_HOME'] = HF_HOME
    os.environ['HF_HUB_CACHE'] = HF_HOME

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default="fine_tune", choices=["fine_tune", "evaluate_roboeval", "evaluate_passk", "generate_completion"])
    parser.add_argument('--param', default="ppo_code_gen", choices=["ppo", "ppo_code_gen"])
    parser.add_argument('--framework_params', default="code_gen", choices=["code_gen", "robo"])
    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()
    training_args, model_args, script_args, framework_params, reward_functions = parse_json_file(params=args.param, framework_params=args.framework_params)

    if not args.checkpoint is None:
        adapter_path = "savings/" + model_args.model_name_or_path + "/" + script_args.dataset_name + "/" + args.checkpoint + "/policy"
    else:
        adapter_path = None

    if args.mode == "fine_tune":
        dataset_helper = get_dataset(dataset_name=script_args.dataset_name, model_or_tokenizer_name=model_args.model_name_or_path, tokenize_targets=True, cache_dir=HF_HOME)
        fine_tuner = PPOWrapper(script_args=script_args, model_args=model_args, ppo_args=training_args, dataset_helper=dataset_helper,
                                framework_params=framework_params, reward_functions=reward_functions, cache_dir=HF_HOME)
        fine_tuner.train()

    elif args.mode == "generate_completion":
        dataset_helper = get_dataset(dataset_name=script_args.dataset_name, model_or_tokenizer_name=model_args.model_name_or_path, tokenize_targets=True, cache_dir=HF_HOME)
        fine_tuner = PPOWrapper(script_args=script_args, model_args=model_args, ppo_args=training_args, dataset_helper=dataset_helper,
                                framework_params=framework_params, reward_functions=reward_functions, cache_dir=HF_HOME, adapter_path=adapter_path)
        fine_tuner.generate_completions(sampling=True)

    elif args.mode == "evaluate_roboeval":
        dataset_helper = get_dataset(dataset_name=script_args.dataset_name,
                                     model_or_tokenizer_name=model_args.model_name_or_path, tokenize_targets=True,
                                     cache_dir=HF_HOME, eval_mode=True)
        ppo_wrapper = PPOWrapper(script_args=script_args, model_args=model_args, ppo_args=training_args,
                                dataset_helper=dataset_helper,
                                framework_params=framework_params, reward_functions=reward_functions, cache_dir=HF_HOME, adapter_path=adapter_path)

        if adapter_path is not None:
            save_dir = "roboeval/" + model_args.model_name_or_path + "/" + args.checkpoint
        else:
            save_dir = "roboeval/" + model_args.model_name_or_path + "/original"

        roboEval(ppo_wrapper.processing_class, save_dir, ppo_wrapper.generate_eval)

    elif args.mode == "evaluate_passk":
        dataset_helper = get_dataset(dataset_name=script_args.dataset_name,
                                     model_or_tokenizer_name=model_args.model_name_or_path, tokenize_targets=True,
                                     cache_dir=HF_HOME, eval_mode=True)
        ppo_wrapper = PPOWrapper(script_args=script_args, model_args=model_args, ppo_args=training_args,
                                 dataset_helper=dataset_helper,
                                 framework_params=framework_params, reward_functions=reward_functions,
                                 cache_dir=HF_HOME, adapter_path=adapter_path)
        if adapter_path is not None:
            save_dir = "evalplus_passk/" + args.checkpoint
        else:
            save_dir = "evalplus_passk/original/"
        evalplusPassK(ppo_wrapper.processing_class, save_dir, ppo_wrapper.generate_eval)
