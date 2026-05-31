
import os

import numpy as np
import pandas as pd
import ast
import subprocess
import tempfile
from evaluate import load
from datasets import load_dataset, DatasetDict, Dataset

code_eval = load("code_eval")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

MAX_LENGTH = 1024

def extract_python_code(text):
    if "```python" in text:
        start_idx = text.find("```python")
        if start_idx != -1:
            start_idx = start_idx + len("```python")
        else:
            start_idx = 0
        end_idx = text.find("```", start_idx)
        if end_idx == -1:
            end_idx = len(text)
        code = text[start_idx:end_idx]
    else:
        code = ""
    return code

def update_prompt(prompt, tokenizer=None):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    ]

    content = ("""Create a python Program with following instructions and include type hints for all parameters and the return values.
                   # Instruction: """ + prompt)
    prompt = messages + [{"role": "user", "content": content}]

    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                                         max_length=MAX_LENGTH)
    return prompt

def get_all_generation(tokenizer, tasks):
    samples = []
    test_cases = []

    for data in tasks:
        prompts = data["input"]
        unit_tests_batch = data["unit_tests"]
        for prompt, unit_tests in zip(prompts, unit_tests_batch):
            samples.append(update_prompt(prompt, tokenizer))
            test_cases.append(ast.literal_eval(unit_tests))

    return samples, test_cases

def save_results(results, save_dir):
    pass1_result = {}
    error_breakdown = {
        "PythonError": 0,
        "CompletionError": 0,
        "Success": 0
    }
    for result in results:
        for key, value in result.items():
            if key == "error_names":
                for error_name in value:
                    error_breakdown[error_name] += 1
            else:
                pass1_result[key] = value

    os.makedirs(os.path.join(save_dir, "pass1"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "error_breakdown"), exist_ok=True)
    pass1_save_path = os.path.join(save_dir, "pass1", "result.csv")
    error_breakdown_save_path = os.path.join(save_dir, "error_breakdown", "result.csv")
    pd.DataFrame(pass1_result.items(), columns=["name", "pass_1"]).to_csv(pass1_save_path, index=False)
    pd.DataFrame(error_breakdown.items(), columns=["name", "error_count"]).to_csv(error_breakdown_save_path,
                                                                                  index=False)
def run_code_subprocess(code_string, test_case, timeout=2):
    """
    Run Python code in a subprocess safely with a timeout.
    Returns True if code executes without exception, False otherwise.
    """
    code_to_run = f"{code_string}\n{test_case}"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "-c", code_to_run],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            if result.returncode== 0:
                print("Program Successful")
                return "success", True
            else:
                print("CompletionError", result.returncode)
                return "CompletionError", False
    except subprocess.TimeoutExpired:
        print("PYTHON TIMED_OUT")
        return "PythonError", False
    except Exception as e:
        print("PYTHON RUNTIME ERROR:", e)
        return "PythonError", False

def evaluate(programs, unit_tests_batch):
    end_results = []
    for idx, (program, unit_tests) in enumerate(zip(programs, unit_tests_batch)):
        results = {
            f"pass_tests_{idx}": [],
            "error_names": []
        }
        for _, test in enumerate(unit_tests):
            error_name, res = run_code_subprocess(program, test)
            results["error_names"].append(error_name)
            results[f"pass_tests_{idx}"].append(int(res))

        results[f"pass_tests_{idx}"] = np.mean(results[f"pass_tests_{idx}"])
        end_results.append(results)
    return end_results



def generate_evaluate(tokenizer, save_dir, generation_func):
    dataset = load_dataset("nvidia/OpenCodeInstruct", streaming=True)
    dataset.shuffle(seed=41, buffer_size=10000)
    base_stream = dataset["train"]

    finite_rows = list(base_stream.take(500))
    raw_dataset = DatasetDict({
        "test": Dataset.from_list(finite_rows)
    })
    tasks = raw_dataset["test"]

    prompts, tests_batch = get_all_generation(tokenizer, tasks)

    batch_size = 32
    unprocessed_programs, accelerator_main_process = generation_func(prompts, batch_size, MAX_LENGTH)

    if accelerator_main_process:
        programs = []
        for program in unprocessed_programs:
            program = extract_python_code(program)
            programs.append(program)

        results = evaluate(programs, tests_batch)
        save_results(results, save_dir)
