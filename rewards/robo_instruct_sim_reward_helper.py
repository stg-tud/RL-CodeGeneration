from rewards.robo_instruct.robo_instruct.robosim.robosim_tracer import rejection_sampling_simulation
from transformers import PreTrainedTokenizerBase
import torch
import re

def extract_python_code(text):
    match = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else ""

def get_related_indicies(code, selected_tokens, processing_class, attention_mask):
    non_zero_indcies = torch.nonzero(attention_mask)
    start_index = non_zero_indcies[0]
    end_index = non_zero_indcies[-1]
    code_extended = "```python" + code + "```"

    normalized_code = re.sub(r'\s+', '', code)
    normalized_code_extended = re.sub(r'\s+', '', code_extended)

    for i in range(start_index, selected_tokens.size(0)):
        text = processing_class.decode(selected_tokens[start_index : i], skip_special_tokens=True)
        normalized_text = re.sub(r'\s+', '', text)
        if normalized_code in normalized_text or normalized_code_extended in normalized_text:
            end_index = i
            break

    return start_index, end_index
