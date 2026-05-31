from rewards.robo_instruct.robo_instruct.misc.llm_generation_utils import post_process_llama3_instruction
from Levenshtein import distance

import os
import pandas as pd
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np 

#### Evol Instruct Prompts ####
def add_new_constraint():
    return "Add new constraints and requirements to the original robot task, adding approximately 10 additional words."
    
def replace_requirement():
    return "Replace a commonly used requirement in the robot task with a less common and more specific one."
    
def more_actions():
    return "If the original robot task can be solved with only a few logical steps, please add more reasoning steps."
    
def higher_complexity():
    return "Propose higher time or space complexity requirements, but please refrain from doing so frequently."
#### Evol Instruct Prompts ####
def evol_instruct(system_inst, df):
    df = [inst[0] for inst in df]
    inst = np.random.choice(df)
    method_funcs = [add_new_constraint, replace_requirement, more_actions, higher_complexity]
    idx = np.random.randint(0, 4)
    method_prompt = method_funcs[idx]()
    content = """Please increase the difficulty of the given robot task instruction a
bit.
You can increase the difficulty using, but not limited to, the following
methods:
{}

### original robot task: {}

Only give me the new robot task and do not add any other extra information.
""".format(method_prompt, inst)

    prompt = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": content},
    ]
    return prompt

def self_instruct(system_inst, df):
    prompt = [
        {"role": "system", "content": system_inst},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[0][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[1][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[2][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[3][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[4][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
        {"role": "assistant", "content": df[5][0]},
        {"role": "user", "content": "Generate an interesting robot task that can be accomplished using above capabilities."},
    ]
    return prompt

def gather_prompts_vllm(llm, s_param, tokenizer, args, generate_n=20):
    existing_prompts = pd.read_csv(args.prompt_path, header=None)
    if os.path.exists(args.save_name):
        new_prompts = pd.read_csv(args.save_name, header=None)
    else:
        new_prompts = pd.DataFrame()
    if len(new_prompts) > args.reuse_prompt_threshold:
        df1 = new_prompts.sample(n=2).values.tolist()
        df2 = existing_prompts.sample(n=4).values.tolist()
        df = df1 + df2
        print("using Self Instruct")
    else:
        df = existing_prompts.sample(n=6).values.tolist()
    
    system_inst = """
You are a helpful assistant. Here is a robot that has the following capabilities:
- def get_current_location() -> str:
- def get_all_rooms() -> list[str]:
- def is_in_room(object : str) -> bool:
- def go_to(location : str) -> None:
- def ask(person : str, question : str, options: list[str]) -> str:
- def say(message : str) -> None:
- def pick(obj: str) -> None:
- def place(obj: str) -> None:
"""
    if args.instruct_type == "evol":
        prompt = evol_instruct(system_inst, df)
    elif args.instruct_type == "self_instruct":
        prompt = self_instruct(system_inst, df)
    else:
        raise ValueError("Unknown instruct type: {}".format(args.instruct_type))
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    outputs = llm.generate([prompt] * generate_n, s_param)
    result = []
    for output in outputs:
        instruction = output.outputs[0].text
        instruction = post_process_llama3_instruction(instruction)
        result.append(instruction)
    return result

def deduplicate_instructions(args, df):
    roboeval_df = pd.read_csv("roboeval_data.csv")
    roboeval_prompts = roboeval_df["prompt_program"].map(lambda x: x.split("\n")[0])

    print("Initial Size: ", len(df))
    prompts = list(df["prompt"])
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
    duplicated_idx = set()
    encoded_prompts = []
    encoded_roboeval_prompts = []
    for p in prompts:
        encoded = tokenizer.encode(p)
        encoded_prompts.append(encoded)
        
    for p in roboeval_prompts:
        encoded = tokenizer.encode(p)
        encoded_roboeval_prompts.append(encoded)

    for i in range(len(encoded_prompts)):
        encoded_prompt = encoded_prompts[i]
        # check if it is similar to roboeval
        too_similar = False
        for encoded_roboeval_prompt in encoded_roboeval_prompts:
            dist = distance(encoded_prompt, encoded_roboeval_prompt)
            edit_sim = 1 - dist / max(len(encoded_prompt), len(encoded_roboeval_prompt))
            if edit_sim > 0.6:
                too_similar = True
                break
            
        if too_similar:
            duplicated_idx.add(i)
            continue
        
        for j in range(i+1, len(encoded_prompts)):
            dist = distance(encoded_prompt, encoded_prompts[j])
            edit_sim = 1 - dist / max(len(encoded_prompt), len(encoded_prompts[j]))
            if edit_sim > 0.6:
                duplicated_idx.add(j)
    df.drop(list(duplicated_idx), inplace=True)
    print("Final Size: ", len(df))
    df.to_csv(args.save_name, index=False, header=["prompt"])

def main_vllm_llama3_inst_8b(args):
    llama3_llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    sampling_params = SamplingParams(
        temperature=0.999, 
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=stop_token_ids)
    final_result = []
    for i in range(args.num_batches):
        print("generating batch: ", i)
        result = gather_prompts_vllm(llama3_llm, sampling_params, tokenizer, args, generate_n=args.batch_size)
        final_result.extend(result)
    
    df = pd.DataFrame({"prompt": final_result})
    deduplicate_instructions(args, df)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt_path", type=str, default="seed_instructions.csv")
    parser.add_argument("-s", "--save_name", type=str)
    parser.add_argument("-r", "--reuse_prompt_threshold", type=int, default=50)
    parser.add_argument("-i", "--instruct_type", type=str, default="self_instruct", choices=["evol", "self_instruct"])
    parser.add_argument("-n", "--num_batches", type=int, default=2500)
    parser.add_argument("-b", "--batch_size", type=int, default=5)
    args = parser.parse_args()
    main_vllm_llama3_inst_8b(args)
