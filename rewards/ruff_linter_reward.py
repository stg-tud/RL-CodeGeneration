import subprocess
import os
import tempfile
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from transformers import PreTrainedTokenizerBase
from trl.trainer.utils import first_true_indices
import torch

# --- Rule Categorization Mapping (from step 1) ---
RUFF_RULE_CATEGORIES = {
    # Compilable (Syntactical/Fundamental Errors)
    "E9": "compilable",    # SyntaxError, IndentationError, etc.
    "PLE": "compilable",   # Pylint errors (many indicate fundamental issues)

    # Runnable (Runtime Errors / Logic Bugs / Undefined Behavior)
    "F": "runnable",       # Pyflakes (unused imports/vars, undefined names)
    "B": "runnable",       # Bugbear (common bug-prone patterns, e.g., mutable defaults)
    "TID": "runnable",     # flake8-tidy-imports (import restrictions leading to errors)

    # Safe (Security Vulnerabilities)
    "S": "safety",         # flake8-bandit (direct security checks)
    "B30": "safety",       # Bugbear B30x rules (e.g., insecure pickle usage)
}

def get_ruff_category(rule_code: str) -> str:
    """Categorizes a Ruff rule code based on the predefined mapping."""
    # Check for direct matches first (e.g., "E902")
    if rule_code in RUFF_RULE_CATEGORIES:
        return RUFF_RULE_CATEGORIES[rule_code]

    # Check for prefix matches (e.g., "E9" for "E902", "B30" for "B301")
    for prefix, category in RUFF_RULE_CATEGORIES.items():
        if rule_code.startswith(prefix):
            return category

    # Default category for rules not explicitly mapped
    return "others"

def save_code_to_temp_file(code_string: str, suffix: str = ".py") -> str:
    """Saves the given code string to a temporary file and returns its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'w') as tmp:
        tmp.write(code_string)
    return path

def run_single_ruff_check_and_categorize(code_string: str) -> List[Dict[str, Any]]:
    """
    Runs Ruff once on the code, collects all errors, and categorizes them.
    Returns a dictionary where keys are categories and values are lists of issues.
    """
    temp_file_path = save_code_to_temp_file(code_string)
    #print(f"Running comprehensive Ruff check on: {temp_file_path}")

    categorized_issues: List[Dict[str, Any]] = []

    try:
        # Run Ruff with all relevant rules enabled (as per pyproject.toml)
        # Use --output-format json for easy parsing
        command = ["ruff", "check", "--output-format", "json", temp_file_path]
        #print(f"  Command: {' '.join(command)}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        ruff_raw_output = result.stdout
        ruff_errors_stderr = result.stderr

        if ruff_errors_stderr:
            print(f"  Ruff Check (stderr):\n{ruff_errors_stderr}")

        violations = []
        if ruff_raw_output.strip():
            try:
                violations = json.loads(ruff_raw_output)
                #print(violations)
            except json.JSONDecodeError as e:
                print(f"  Error decoding Ruff JSON output: {e}. Raw: {ruff_raw_output}")
                # Add a synthetic error to the compilable category if parsing fails
                categorized_issues.append({
                    "category": "others",
                    "code": "JSON_PARSE_ERROR",
                    "message": f"Ruff output could not be parsed: {e}",
                    "start_location": {"row": 0, "column": 0},
                    "end_location": {"row": 0, "column": 0}
                })
                return categorized_issues

        # Categorize each violation
        for violation in violations:
            code = violation.get("code", "UNKNOWN")
            if not code is None:
                category = get_ruff_category(code)
                categorized_issues.append({
                    "category": category,
                    "code": code,
                    "message": violation.get("message", "UNKNOWN"),
                    "start_location": violation.get("location", "UNKNOWN"),
                    "end_location": violation.get("end_location", "UNKNOWN")
                })
            else:
                categorized_issues.append({
                    "category": "compilable",
                    "code": "Syntax error",
                    "message": violation.get("message", "UNKNOWN"),
                    "start_location": violation.get("location", "UNKNOWN"),
                    "end_location": violation.get("end_location", "UNKNOWN")
                })


        #print(f"Ruff detected a total of {len(violations)} issues.")

    except FileNotFoundError:
        print("Error: Ruff command not found. Make sure Ruff is installed and in your PATH.")
        categorized_issues.append({
            "category": "others",
            "code": "RUFF_NOT_FOUND",
            "message": "Ruff command not found.",
            "start_location": {"row": 0, "column": 0},
            "end_location": {"row": 0, "column": 0}
        })
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return categorized_issues

def find_row_change_token(selected_tokens, processing_class: PreTrainedTokenizerBase):
    sequence_start_idx : int  = first_true_indices(selected_tokens != processing_class.pad_token_id)
    sequence_end_idx : int = first_true_indices(selected_tokens[sequence_start_idx:] == processing_class.pad_token_id) + sequence_start_idx
    rows = [sequence_start_idx.item()]
    for idx, _ in enumerate(selected_tokens):
        if idx > sequence_start_idx and idx <= sequence_end_idx:
            sequence = processing_class.batch_decode([selected_tokens[sequence_start_idx: idx]])[0]
            num_new_line_char = sequence.count("\n")
            if num_new_line_char >= 1:
                sequence_start_idx = idx
                for count in range(num_new_line_char):
                    rows.append(idx)

    return rows

def get_rewards(violations, row_change_tokens, ref_tensor):
    #rewards = syncode_rewards.clone()
    rewards = torch.zeros_like(ref_tensor, dtype=torch.float)
    for violation in violations:
        start_row = violation["start_location"]["row"] - 1
        end_row = violation["end_location"]["row"]

        penalty = -1.0
        """if violation["category"] == "compilable":
            penalty = -2.0
        elif violation["category"] == "runnable" or violation["category"] == "safety":
            penalty = -1.0
        elif violation["category"] == "others":
            penalty = -1.0"""

        if start_row >= len(row_change_tokens):
            start_row = len(row_change_tokens) - 1

        start_token_idx = row_change_tokens[start_row]
        if end_row >= len(row_change_tokens):
            rewards[start_token_idx:] += penalty
        else:
            end_token_idx = row_change_tokens[end_row]
            rewards[start_token_idx: end_token_idx] += penalty

    return rewards

def ruff_linter_reward(postprocessed_responses, processing_class: PreTrainedTokenizerBase):
    violationss = []
    rewardss = []

    for postprocessed_response in postprocessed_responses:
        code_string = processing_class.decode(postprocessed_response, skip_special_tokens=True)

        violations = run_single_ruff_check_and_categorize(code_string)
        row_change_tokens = find_row_change_token(postprocessed_response, processing_class)
        rewards = get_rewards(violations, row_change_tokens, postprocessed_response)

        attention_mask = rewards == 0.0
        rewards = torch.where(attention_mask, 1.0, rewards)

        attention_mask = rewards < 0.0
        rewards = torch.where(attention_mask, -1.0, rewards)

        rewardss.append(rewards)
        violationss.append(violations)

    return torch.stack(rewardss), violationss
