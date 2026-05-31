
from transformers import PreTrainedTokenizer
from rewards.robo_instruct.roboeval.misc.utils import load_module
import re


def remove_special_tokens(code_string):
    """
        Removes special tokens like "NEW_LINE", "INDENT", and "DEDENT" from a code string.
        The function processes the code string line by line, adjusting indentation based on the "INDENT" and "DEDENT" tokens.

        Parameters:
        ----------
        code_string : str
            A string representing code, possibly containing special tokens like "NEW_LINE", "INDENT", and "DEDENT".

        Returns:
        --------
        str
            A cleaned-up version of the code string with special tokens removed and proper indentation restored.
    """
    lines = code_string.split("NEW_LINE")
    lines = [item.strip() for item in lines]

    curr_indent = 0
    new_lines = []
    for line in lines:
        indent_count = line.count('INDENT')
        dedent_count = line.count('DEDENT')
        curr_indent += indent_count - dedent_count
        wo_indent = re.sub('INDENT\s?', '', line)
        wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
        new_lines.append('\t' * curr_indent + wo_dedent)
    return ("\n").join(new_lines)

def xlcost_special_tokenization_function(input_sequences, target_sequences, tokenizer : PreTrainedTokenizer):

    messages = [
        {"role": "system", "content": """
        You are Qwen, created by Alibaba Cloud. You are a helpful Python assistant.
        Always include Python code inside Markdown code blocks using triple backticks
        and specify the language as 'python'
        """},
        {"role": "user", "content": """
        Create a python Program with following instructions and include type hints for all parameters and the return values.
        # Instruction: create a hello world function
        """},
        {"role": "assistant", "content": """
        ```python
            def hello_world() -> None:
                print('Hello World')
        ```
        """}
    ]

    for idx, (input_seq, code) in enumerate(zip(input_sequences, target_sequences)):
        content = ("""Create a python Program with following instructions and include type hints for all parameters and the return values.
                   # Instruction: """ + input_seq)
        prompt = messages + [{"role": "user", "content": content}]

        input_sequences[idx] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, max_length=512)
        target_sequences[idx] = remove_special_tokens(code)

def open_code_instruct_special_tokenization_function(input_sequences, target_sequences, tokenizer : PreTrainedTokenizer):
    messages = [
        {"role": "system", "content": """You are a helpful Python assistant.
        you follow the instruction exactly."""},
    ]
    for idx, (input_seq, code) in enumerate(zip(input_sequences, target_sequences)):
        content = ("""Create a python Program with following instructions. Only create the asked function with no other output.
                   # Instruction: """ + input_seq)
        prompt = messages + [{"role": "user", "content": content}]

        input_sequences[idx] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, max_length=512)
        target_sequences[idx] = remove_special_tokens(code)

def robo_instruct_tokenization_function(input_sequences, target_sequences, tokenizer : PreTrainedTokenizer):
    messages = load_module("", "rewards/robo_instruct/roboeval/code_generation/openai_chat_completion_prefix.py").__dict__["messages"]
    # we have an input length of 512 so i need to use less examples
    indices = [0, 1, 2]
    messages = [messages[i] for i in indices]

    for msg in messages:
        if msg["role"] == "user":
            content = "** Example:\n" + get_content(msg["content"])
            msg["content"] = content

        if msg["role"] == "assistant":
            msg["content"] = "```python" + msg["content"] + "```"


    for idx, (input_seq, code) in enumerate(zip(input_sequences, target_sequences)):
        input_seq = input_seq.split("# Instruction:", 1)[1]
        input_seq = input_seq.split("def task_program():", 1)[0]

        content = get_content(input_seq)

        prompt = messages + [{"role": "user", "content": content}]
        input_sequences[idx] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, max_length=512)

        target_sequences[idx] = remove_special_tokens(code)

def get_content(instruction):
    content = (
    "Create one python program using only the given functions for following instruction.\n"
    "Instruction: " + instruction + "\n")

    return content
