from unsloth import FastLanguageModel
import torch 
import argparse

parser = argparse.ArgumentParser(description="Merge PEFT model with base model")
parser.add_argument('--model_name', type=str, required=True, help='Path to the trained PEFT model')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the merged model')
args = parser.parse_args()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.model_name,
    max_seq_length = 2048,
    dtype = torch.float16,
    load_in_4bit = False,
)

model.save_pretrained_merged(args.save_path, tokenizer, save_method = "merged_16bit")