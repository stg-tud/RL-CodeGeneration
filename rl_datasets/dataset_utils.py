
from rl_datasets.dataset_helper import BaseDatasetHelper
from rl_datasets.dataset_special_functions import *

def get_dataset(dataset_name, model_or_tokenizer_name, cache_dir, tokenize_targets=True, eval_mode = False):
    if dataset_name == 'code_search_net':
        return get_code_search_net_dataset(model_or_tokenizer_name, cache_dir, tokenize_targets, eval_mode = eval_mode)
    elif dataset_name == 'codeparrot/xlcost-text-to-code':
        return get_xlcost_dataset(model_or_tokenizer_name, cache_dir, tokenize_targets, eval_mode = eval_mode)
    elif dataset_name == 'codeparrot/apps':
        return get_apps_dataset(model_or_tokenizer_name, cache_dir, tokenize_targets, eval_mode = eval_mode)
    elif dataset_name == 'zichao22/robo-instruct':
        return get_robo_instruct_dataset(model_or_tokenizer_name, cache_dir, tokenize_targets, eval_mode = eval_mode)
    elif dataset_name == 'nvidia/OpenCodeInstruct':
        return get_open_code_instruct_dataset(model_or_tokenizer_name, cache_dir, tokenize_targets, eval_mode = eval_mode)


def get_code_search_net_dataset(tokenizer_or_model, cache_dir, tokekize_targets=True, eval_mode = False):
    return BaseDatasetHelper("code_search_net", tokenizer_name=tokenizer_or_model,
                             input_feature_name="func_documentation_string", cache_dir=cache_dir,
                             target_feature_name="whole_func_string", load_subset_feature="python", tokenize_targets=tokekize_targets, eval_mode = eval_mode)


def get_xlcost_dataset(tokenizer_or_model, cache_dir, tokekize_targets=True, eval_mode = False):
    return BaseDatasetHelper("codeparrot/xlcost-text-to-code", tokenizer_name=tokenizer_or_model,
                             input_feature_name="text", special_tokenization_function=xlcost_special_tokenization_function, cache_dir=cache_dir,
                             target_feature_name="code", load_subset_feature="Python-program-level", tokenize_targets=tokekize_targets, eval_mode = eval_mode)

def get_robo_instruct_dataset(tokenizer_or_model, cache_dir, tokekize_targets=True, eval_mode = False):
    return BaseDatasetHelper("zichao22/robo-instruct", tokenizer_name=tokenizer_or_model,
                             input_feature_name="text", special_tokenization_function=robo_instruct_tokenization_function, cache_dir=cache_dir,
                             target_feature_name="program", tokenize_targets=tokekize_targets, eval_mode = eval_mode)

def get_apps_dataset(tokenizer_or_model, cache_dir, tokekize_targets=True, eval_mode = False):
    return BaseDatasetHelper(dataset_name="codeparrot/apps", tokenizer_name=tokenizer_or_model, load_subset_feature='all', cache_dir=cache_dir,
                                                input_feature_name="question", target_feature_name="solutions", tokenize_targets=tokekize_targets, eval_mode = eval_mode)

def get_open_code_instruct_dataset(tokenizer_or_model, cache_dir, tokekize_targets=True, eval_mode = False):
    filter_fn = lambda x: x['domain'] == 'generic' and x['unit_tests'] is not None and len(x['unit_tests']) > 0

    return BaseDatasetHelper(dataset_name="nvidia/OpenCodeInstruct", tokenizer_name=tokenizer_or_model,
                             cache_dir=cache_dir, special_tokenization_function = open_code_instruct_special_tokenization_function,
                             input_feature_name="input", target_feature_name="output", filter_fn=filter_fn, streaming=True,
                             tokenize_targets=tokekize_targets, eval_mode=eval_mode)
