"""
Add your rewards directly here to be found by the framework
"""
import ast
import subprocess
import tempfile

from codebleu import calc_codebleu
from rewards.reward_helper import *

def robo_instruct_sim_reward(postprocessed_responses, processing_class: PreTrainedTokenizerBase, scoress, labels, **kwargs):
    rewardss = []
    for postprocessed_response, scores, label in zip(postprocessed_responses, scoress, labels):

        attention_mask = scores > 0.0
        rewards = torch.zeros_like(attention_mask, dtype=torch.bfloat16).to(postprocessed_responses.device)

        if attention_mask.any():
            attention_mask = attention_mask.to(postprocessed_response.device)

            selected_output = torch.masked_fill(postprocessed_response, ~attention_mask,
                                                processing_class.pad_token_id)

            string = processing_class.decode(selected_output, skip_special_tokens=True)

            code_string = extract_python_code(string)

            status = False
            passed_count = 0
            if "def task_program():\n" in code_string:
                (trace_elements, status, passed_count) = rejection_sampling_simulation(
                    program=code_string,
                    simulation_timeout=2,
                    resampling_count=5,
                    allowed_timeout_count=2,
                    VERBOSE=False
                )
            if status:
                start_index, end_index = get_related_indicies(code_string, selected_output, processing_class, attention_mask)

                reward = passed_count / 5.0
                rewards[start_index:end_index] = reward
            else:
                rewards = torch.where(attention_mask, -0.1, rewards)

        rewardss.append(rewards)
    return torch.stack(rewardss)


def calculate_dfg_match(postprocessed_responses, processing_class: PreTrainedTokenizerBase, scoress, labels, **kwargs):
    eval = kwargs.get("is_eval", False)
    rewardss = []
    for postprocessed_response, scores, label in zip(postprocessed_responses, scoress, labels):

        attention_mask = scores > 0.0
        rewards = torch.zeros_like(attention_mask, dtype=torch.bfloat16).to(postprocessed_responses.device)

        if attention_mask.any():
            attention_mask = attention_mask.to(postprocessed_response.device)
            selected_output = torch.masked_fill(postprocessed_response, ~attention_mask,
                                                processing_class.pad_token_id)

            code_string = processing_class.decode(selected_output, skip_special_tokens=True)
            code_string = extract_python_code(code_string)
            label_code = processing_class.decode(label, skip_special_tokens=True)
            label_code = extract_python_code(label_code)

            if eval:
                print("code_string", code_string)
                print("label_code", label_code)
            # Normalize dfg to -0.5 and 0.5
            dfg = calc_codebleu([label_code], [code_string], "python").get("dataflow_match_score") - 0.5
            if dfg > 0:
                start_index, end_index = get_related_indicies(code_string, selected_output, processing_class,
                                                              attention_mask)

                reward = dfg
                rewards[start_index:end_index] = reward
            #else:
                #rewards = torch.where(attention_mask, -0.1, rewards)

        rewardss.append(rewards)
    return torch.stack(rewardss)

def calculate_pass_at_k(postprocessed_responses, processing_class: PreTrainedTokenizerBase, scoress, labels, **kwargs):

    rewardss = []
    batch_unit_tests = kwargs.get("unit_tests", [])

    for postprocessed_response, scores, unit_tests in zip(postprocessed_responses, scoress, batch_unit_tests):

        attention_mask = scores > 0.0
        rewards = torch.zeros_like(attention_mask, dtype=torch.bfloat16).to(postprocessed_responses.device)

        if attention_mask.any():
            attention_mask = attention_mask.to(postprocessed_response.device)
            selected_output = torch.masked_fill(postprocessed_response, ~attention_mask,
                                                processing_class.pad_token_id)

            code_string = processing_class.decode(selected_output, skip_special_tokens=True)
            code_string = extract_python_code(code_string)

            if code_string != "":
                #candidates = [[code_string]]
                test_cases = ast.literal_eval(unit_tests)
                pass_at_k = 0

                for test_case in test_cases:
                    code_to_run = f"{code_string}\n{test_case}"
                    if run_code_subprocess(code_to_run, timeout=2):
                        pass_at_k += 1

                if pass_at_k > 0 and len(test_cases) > 0:
                    start_index, end_index = get_related_indicies(code_string, selected_output, processing_class,
                                                              attention_mask)

                    reward = pass_at_k / len(test_cases)
                    rewards[start_index:end_index] = reward
                else:
                    rewards = torch.where(attention_mask, -0.3, rewards)
            else:
                rewards = torch.where(attention_mask, -0.3, rewards)

        rewardss.append(rewards)
    return torch.stack(rewardss)

def run_code_subprocess(code_string, timeout=2):
    """
    Run Python code in a subprocess safely with a timeout.
    Returns True if code executes without exception, False otherwise.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["python", "-c", code_string],
                cwd=tmpdir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
