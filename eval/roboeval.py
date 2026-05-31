from rewards.robo_instruct.roboeval.misc.utils import read_benchmark, load_module
from rewards.robo_instruct.roboeval.benchmark.simple_tracer import evaluate_task
import os
from pathlib import Path
import pandas as pd
from rl_datasets.dataset_special_functions import get_content
from joblib import Parallel, delayed
import json

os.environ["TOKENIZERS_PARALLELISM"] = "true"
BENCHMARK_FILE = Path("rewards/robo_instruct/roboeval/benchmark/tasks")
BENCHMARK_TASKS = read_benchmark(BENCHMARK_FILE, "*")
PROMPT_VARIATION = 5
NUM_COMPLETION = 1
MAX_LENGTH = 1024

def extract_python_code(text):
    if "def task_program():" in text:
        start_idx = text.find("def task_program():")
        end_idx = text.find("```", start_idx)
        if end_idx == -1:
            end_idx = len(text)
        code = text[start_idx:end_idx]
    else:
        code = ""
    return code

def update_prompt(prompt, tokenizer=None):
    messages = load_module("", "rewards/robo_instruct/roboeval/code_generation/openai_chat_completion_prefix.py").__dict__["messages"]
    # we have an input length of 512 so i need to use less examples here only 2
    #indices = [0, 1, 2]
    indices = [0, 3, 4, 7, 8]
    messages = [messages[i] for i in indices]

    for msg in messages:
        if msg["role"] == "user":
            content = "** Example:\n" + get_content(msg["content"])
            msg["content"] = content

        if msg["role"] == "assistant":
            msg["content"] = "```python" + msg["content"] + "```"

    content = get_content(prompt)
    prompt = messages + [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, max_length=MAX_LENGTH)
    return prompt

def get_all_generation(num_completions, tokenizer):
    result = []
    tasknames = []
    completion_num = num_completions
    for _, task in BENCHMARK_TASKS.iterrows():
        prompts = task["prompts"]
        tasknames.extend([task["name"]] * completion_num * len(prompts))
        assert len(prompts) == PROMPT_VARIATION, f"Prompt variation mismatch: {len(prompts)} vs {PROMPT_VARIATION}"
        for prompt in prompts:
            prompt = update_prompt(prompt, tokenizer)
            result.extend([prompt] * completion_num)

    assert len(result) == len(tasknames), f"Length mismatch: {len(result)} vs {len(tasknames)}"
    return result, tasknames

def save_results(results, save_dir):
    pass1_result = {}
    error_breakdown = {
        "RobotExecutionError": 0,
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


def evaluate(tasknames, programs, benchmark_file):
    # use joblib to speed up evaluation process
    num_completions_task = NUM_COMPLETION
    tasknames_dedup = tasknames[0:len(tasknames):num_completions_task]
    total_len = len(tasknames_dedup)

    results = Parallel(n_jobs=total_len)(delayed(evaluate_task)(
        benchmark_file,
        programs[i * num_completions_task:(i + 1) * num_completions_task],
        tasknames_dedup[i],
        i % PROMPT_VARIATION
    ) for i in range(total_len))
    return results


def generate_evaluate(tokenizer, save_dir, generation_func):
    prompts, tasknames = get_all_generation(NUM_COMPLETION, tokenizer)

    batch_size = NUM_COMPLETION * PROMPT_VARIATION
    unprocessed_programs, accelerator_main_process = generation_func(prompts, batch_size, MAX_LENGTH)

    if accelerator_main_process:
        programs = []
        for program in unprocessed_programs:
            program = extract_python_code(program)
            programs.append(program)

        results = evaluate(tasknames, programs, BENCHMARK_FILE)
        save_results(results, save_dir)

        program_results = {}
        for i, program in enumerate(programs):
            program_results[tasknames[i] + f"_{i}"] = program
        with open(f"{save_dir}/programs.json", "w") as f:
            json.dump(program_results, f)
