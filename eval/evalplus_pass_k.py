import re
import json
import os
import shutil
import sys
from pathlib import Path
from accelerate import Accelerator
from evalplus.data import get_human_eval_plus, write_jsonl, get_mbpp_plus
import subprocess

MAX_LENGTH = 1024
N_SAMPLES = 1

def extract_python_code(text):
    match = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else ""

def update_prompt(prompt, tokenizer=None):
    messages = [
        {"role": "system", "content": """You are a helpful Python assistant.
        you follow the instruction exactly."""},
    ]

    content_humanEval = ("""Create a python Program with following instructions and include type hints for all parameters and the return values.
                   # Instruction: You are given a function with docstring in it. Your task is to complete the function exactly as the docstring describes. You are not allowed to add any other function to the given code.\n```python\n""" + prompt + "\n```")
    content_mbpp = ("""Create a python Program with following instructions. Only create the asked function with no other output.
                   # Instruction:""" + prompt)
    prompt = messages + [{"role": "user", "content": content_mbpp}]

    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                                         max_length=MAX_LENGTH)
    return prompt

def get_all_generation(tokenizer, problems):
    prompts = []
    task_ids = []
    n_samples_per_task = N_SAMPLES
    for task_id, task in problems.items():
        prompt = task["prompt"]
        prompt = update_prompt(prompt, tokenizer)
        prompts.extend([prompt] * n_samples_per_task)
        task_ids.extend([task_id] * n_samples_per_task)

    return prompts, task_ids

def generate_evaluate(tokenizer, save_dir, generation_func):
    acc = Accelerator()

    project_path = str(Path(__file__).resolve().parent.parent) + "/"
    samples_path = project_path + save_dir + "/evalplus_samples.jsonl"
    results_path = project_path + save_dir + "/evalplus_samples_eval_results.json"
    log_path = project_path + save_dir + "/evalplus_run_log.json"

    print("Sample_Path: ", samples_path)
    if acc.is_main_process:
        if not os.path.exists(samples_path):
            problems = get_mbpp_plus()
            prompts, task_ids = get_all_generation(tokenizer, problems)

            # Lowered from 32: at response_length=1024 (see config/hyperparams.json's
            # sft_code_gen_1.5b/7b), per-sequence KV cache is big enough at batch 32 to
            # OOM a single GPU (ZeRO stage 2 in config/deepspeed.json shards optimizer/
            # gradient state, not model params, so it doesn't help during pure generation
            # - the only lever here is batch size). Pure throughput/memory knob, does not
            # change which completions get generated or scored. (8 confirmed to use only
            # ~52/82GB on a 7B model - raised to 16 for speed; watch nvidia-smi and drop
            # back down if it climbs close to the GPU's ceiling.)
            batch_size = 16
            unprocessed_programs, _ = generation_func(prompts, batch_size, MAX_LENGTH)

            samples = []
            raw_samples = []
            for raw_program, task_id in zip(unprocessed_programs, task_ids):
                program = extract_python_code(raw_program)
                samples.append({
                    "task_id": task_id,
                    "completion": program
                })
                # Debug-only: the raw, pre-extraction text, so an empty "completion"
                # above (extract_python_code found no "```python" marker) can be
                # diagnosed - e.g. the model answering without a fence, or with a
                # bare "```" - without needing to re-run generation. Not read by
                # evalplus itself, purely for inspection.
                raw_samples.append({"task_id": task_id, "raw": raw_program})

            os.makedirs(save_dir, exist_ok=True)
            with open(project_path + save_dir + "/evalplus_raw_completions.jsonl", "w", encoding="utf-8") as f:
                for row in raw_samples:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            write_jsonl(samples_path, samples)

        if os.path.exists(results_path):
            os.remove(results_path)

        print("\n--- Starting EvalPlus Evaluation ---")

        env = os.environ.copy()
        env["EVALPLUS_SKIP_GUARD"] = "1"

        cmd = [
            sys.executable,
            "-c",
            (
                "import sys, resource; "
                "resource.setrlimit = lambda *args, **kwargs: None; "  # 🔥 Mock guard
                "from evalplus.evaluate import main; "
                "sys.argv = ['evalplus.evaluate'] + sys.argv[1:]; "
                "main()"
            ),
            "--dataset", "mbpp",
            "--samples", samples_path,
        ]

        result = subprocess.run(
            cmd,
            env=env,
            cwd=save_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print("=== EvalPlus STDOUT ===")
        print(result.stdout)

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result.stdout, f, indent=2, ensure_ascii=False)

        print("=== EvalPlus STDERR ===")
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError("EvalPlus failed")

    acc.wait_for_everyone()
