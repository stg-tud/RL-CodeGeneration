import subprocess
import os
import sys
import shutil
import tempfile
import json
import re
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

# Ruff discovers configuration relative to the checked file. The code is written to a
# temporary file outside the project, so the config has to be passed explicitly.
RUFF_CONFIG_PATH = os.environ.get(
    "RUFF_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"),
)

if not os.path.isfile(RUFF_CONFIG_PATH):
    raise FileNotFoundError(f"Ruff configuration not found: {RUFF_CONFIG_PATH}")


def _resolve_ruff_command() -> List[str]:
    """Locates the ruff executable (PATH, then the active interpreter's bin dir)."""
    ruff_path = shutil.which("ruff")
    if ruff_path is None:
        candidate = os.path.join(os.path.dirname(sys.executable), "ruff")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            ruff_path = candidate
    if ruff_path is not None:
        return [ruff_path]
    try:
        subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, check=True)
        return [sys.executable, "-m", "ruff"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "Ruff is not installed. Install it (e.g. `pip install ruff==0.12.2`) "
            "or add it to PATH."
        )


RUFF_COMMAND = _resolve_ruff_command()


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
        command = RUFF_COMMAND + [
            "check",
            "--config", RUFF_CONFIG_PATH,
            "--no-cache",
            "--output-format", "json",
            temp_file_path,
        ]
        #print(f"  Command: {' '.join(command)}")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        ruff_raw_output = result.stdout
        ruff_errors_stderr = result.stderr

        # Exit code 0: no violations, 1: violations found, 2: ruff itself failed
        # (e.g. invalid configuration). Treating 2 as "no violations" would hand out
        # full reward, so fail loudly instead.
        if result.returncode not in (0, 1):
            raise RuntimeError(
                f"Ruff failed with exit code {result.returncode}:\n{ruff_errors_stderr}"
            )

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

# Same fence pattern as extract_python_code in eval/evalplus_pass_k.py.
CODE_BLOCK_PATTERN = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Reward for tokens outside the fenced code block (fences, explanations). Kept positive:
# the extra rewards select tokens with score > 0 and then look for the fences again,
# so a non-positive value here would drop the fences from that selection.
OUTSIDE_CODE_REWARD = 1.0


def extract_code_block(text: str) -> Tuple[str, int, int]:
    """
    Returns (code, first_line, num_lines) of the first fenced code block, where
    first_line is the 0-based line of the response on which the code starts.
    Without a fenced block the whole response is treated as code.
    """
    match = CODE_BLOCK_PATTERN.search(text)
    if match is None:
        return text, 0, text.count("\n") + 1
    code = match.group(1)
    first_line = text[:match.start(1)].count("\n")
    num_lines = code.count("\n") + (0 if code.endswith("\n") else 1)
    return code, first_line, num_lines


def shift_violation_rows(violations: List[Dict[str, Any]], offset: int) -> None:
    """Moves violation rows from code-block coordinates to response coordinates."""
    if offset == 0:
        return
    for violation in violations:
        for key in ("start_location", "end_location"):
            location = violation.get(key)
            if isinstance(location, dict) and isinstance(location.get("row"), int):
                location["row"] += offset


def ruff_linter_reward(postprocessed_responses, processing_class: PreTrainedTokenizerBase):
    violationss = []
    rewardss = []

    for postprocessed_response in postprocessed_responses:
        response_text = processing_class.decode(postprocessed_response, skip_special_tokens=True)
        code_string, first_line, num_lines = extract_code_block(response_text)

        violations = run_single_ruff_check_and_categorize(code_string)
        shift_violation_rows(violations, first_line)

        row_change_tokens = find_row_change_token(postprocessed_response, processing_class)
        rewards = get_rewards(violations, row_change_tokens, postprocessed_response)

        # Token span of the code block: from the start of its first line up to the
        # start of the line after its last line.
        code_mask = torch.zeros_like(rewards, dtype=torch.bool)
        if first_line < len(row_change_tokens):
            start_token_idx = row_change_tokens[first_line]
            end_line = first_line + num_lines
            end_token_idx = row_change_tokens[end_line] if end_line < len(row_change_tokens) else len(rewards)
            code_mask[start_token_idx:end_token_idx] = True

        rewards = torch.where(rewards < 0.0, -1.0, 1.0)
        rewards = torch.where(code_mask, rewards, OUTSIDE_CODE_REWARD)

        rewardss.append(rewards)
        violationss.append(violations)

    return torch.stack(rewardss), violationss
