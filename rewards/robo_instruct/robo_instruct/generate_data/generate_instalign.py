from rewards.robo_instruct.robo_instruct.misc.llm_generation_utils import post_process_llama3_instruction, construct_text_field
from rewards.robo_instruct.robo_instruct.instalign.utils import setup_instruction_revision_prompts, setup_decision_prompts, post_process_revision, post_process_decision

import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def revise_instruction(df):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    prompts = setup_instruction_revision_prompts(df, tokenizer)
    print("Correct Prompt Number of prompts: ", len(prompts))
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, 
        top_p=0.95,
        max_tokens=2048,
        best_of=3,
        stop_token_ids=stop_token_ids)
    outputs = llm.generate(prompts, sampling_params)
    revisions = []
    for i in range(len(outputs)):
        original = df["prompt"].iloc[i]
        revision = outputs[i].outputs[0].text
        revision = post_process_llama3_instruction(revision)
        instruction = post_process_revision(revision, original)
        revisions.append(instruction)
    return revisions

def decision(df, revisions):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    prompts = setup_decision_prompts(df, revisions, tokenizer)

    sampling_params = SamplingParams(
        temperature=0.3, 
        top_p=0.2,
        best_of=3,
        max_tokens=2048,
        stop_token_ids=stop_token_ids)
    outputs = llm.generate(prompts, sampling_params)
    result_texts = []
    for i in range(len(outputs)):
        original = df["prompt"].iloc[i]
        program = df["program"].iloc[i]
        revision = revisions[i]
        
        decision = outputs[i].outputs[0].text.strip()
        instruction = post_process_decision(decision, original, revision)
        text = construct_text_field(instruction, program)
        result_texts.append(text)
    df["text"] = result_texts
    return df 

def main(args):
    df = pd.read_csv(args.input_name)
    revisions = revise_instruction(df)
    df = decision(df, revisions)
    df.to_csv(args.save_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_name", type=str)
    parser.add_argument("-s", "--save_name", type=str)
    args = parser.parse_args()

    llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", swap_space=5)
    TEMPERATURE = 0.3
    main(args)
