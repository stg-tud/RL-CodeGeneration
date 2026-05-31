from rewards.robo_instruct.roboeval.misc.utils import load_module
from rewards.robo_instruct.robo_instruct.misc.llm_generation_utils import post_process_vllm_generation, construct_text_field

import argparse
import pandas as pd
import numpy as np 
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_gen_instruction(args, tokenizer):
    instructions = pd.read_csv(args.input_name)
    
    messages = load_module("", "../../roboeval/code_generation/openai_chat_completion_prefix.py").__dict__["messages"]
    for msg in messages:
        if msg["role"] == "user":
            msg["content"] = "# Instruction: " + msg["content"]
    
    prompts = []
    raw_instructions = []
    for index, row in instructions.iterrows():
        if index < args.skip_n:
            continue
        instruction = row[0]
        if type(instruction) == type(np.nan):
            continue
        prompt = messages + [{"role": "user", "content": "# Instruction: " + instruction}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

        for _ in range(args.max_rejection_sampling_count):
            prompts.append(prompt)
            raw_instructions.append(instruction)
    return prompts, raw_instructions

def gen_program(args):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    prompts, instructions = setup_gen_instruction(args, tokenizer)
    print(f"Generating {len(prompts)} prompts")
    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
    sampling_params = SamplingParams(
        temperature=0.999, 
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=stop_token_ids)
    outputs = llm.generate(prompts, sampling_params)
    
    programs = post_process_vllm_generation(outputs)
    texts = []

    if args.construct_text:
        for inst, p in zip(instructions, programs):
            text = construct_text_field(inst, p)
            texts.append(text)

    pd.DataFrame({'prompt': instructions, 'program': programs, "text": texts}).to_csv(args.save_name, index=False) # temp
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_name", type=str)
    parser.add_argument("-s", "--save_name", type=str)
    parser.add_argument("-max", "--max_rejection_sampling_count", type=int, default=3, help="rejection sampling repeat num")
    parser.add_argument("-t", "--construct_text", action="store_true")
    parser.add_argument("--skip_n", type=int, default=0)
    args = parser.parse_args()
    gen_program(args)

