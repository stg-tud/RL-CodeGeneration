import numpy as np
import re 

system = """
### Role: You are an expert at understanding robot programs. You will be given a task instruction and robot program pair. However, the instruction may not align with the program well. You need to correct the task instruction to match the given robot program. 

### Context: The robot only has access to the following 8 APIs and standard Python functions
- def get_current_location() -> str:
- def get_all_rooms() -> list[str]:
- def is_in_room(object : str) -> bool:
- def go_to(location : str) -> None:
- ask(person : str, question : str, options: list[str]) -> str:
- say(message : str) -> None:
- def pick(obj: str) -> None:
- def place(obj: str) -> None:

### Inputs 
Original Instruction: This is a task instruction that may not align with the robot program
Robot Program: This is a python function starting with `def task_program():`

### Task:
1. Write down all the provided APIs used in the program and explain the effect of each API in this program
2. Examine these APIs and write down step by step what the program does
3. Combine all the results above and rewrite the instruction under # Final Corrected Instruction:
You need to be specific and clear in your final corrected instruction.
"""

example1_task = """
# Original Instruction: 
the task of querying Arjun about his readiness for an event or departure
# Robot Program:
def task_program():
    start_loc = get_current_location()
    go_to("Arjun's office")
    response = ask("Arjun", "Are you ready to go?", ["Yes", "No"])
    go_to(start_loc)
    say("Arjun said: " + response)
"""

example1_response = """
# API Usage:
start_loc = get_current_location()
    - Get the current location as the starting location
go_to("Arjun's office")
    - Go to Arjun's office
response = ask("Arjun", "Are you ready to go?", ["Yes", "No"])
    - Ask if arjun is ready to go or not and get the response
go_to(start_loc)
    - Come back to the starting location
say("Arjun said: " + response)
    - Say Arjun's response

# The Given Task:
The robot gets its current location as the starting location
The robot goes to Arjun's office
The robot asks if Arjun is ready to go and gets the response
The robot returns to the starting location

# Final Corrected Instruction:
Go to Arjun's office, ask him if he is ready to head out, and come back and tell me what he said
"""

example2_task = """
# Original Instruction: 
1. Pick boxes from storage room 
2. place boxes to Alice's office 
3. Otherwise say you cannot get the boxes
# Robot Program:
def task_program():
    go_to("Alice's office")
    num_boxes = ask("Alice", "How many boxes do you need?", ["1", "2", "3"])
    go_to("storage room")
    response = ask("", "Do you have" + num_boxes + " boxes?", ["Yes", "No"])
    if response == "Yes":
        for _ in range(int(num_boxes)):
            pick("box")
            go_to("Alice's office")
            place("box")
            go_to("storage room")
    else:
        go_to("Alice's office")
        say("I could not get the boxes")
"""

example2_response = """
# API Usage:
go_to("Alice's office")
    - go to Alice's office
num_boxes = ask("Alice", "How many boxes do you need?", ["1", "2", "3"])
    - ask Alice how many boxes she needs. The options are 1, 2, or 3
go_to("storage room")
    - go to the storage room
response = ask("", "Do you have" + num_boxes + " boxes?", ["Yes", "No"])
    - ask if they have the number of boxes Alice needs
if response == "Yes":
    - if the response is Yes
for _ in range(int(num_boxes)):
    - loop for the number of boxes Alice needs
pick("box")
    - pick up a box
go_to("Alice's office")
    - go to Alice's office
place("box")
    - place the box in Alice's office
go_to("storage room")
    - go to the storage room
else:
    - if the response is No
go_to("Alice's office")
    - go to Alice's office
say("I could not get the boxes")
    - say that the robot could not get the boxes

# The Given Task:
The robot goes to Alice's office
The robot asks Alice how many boxes she needs with options 1, 2, or 3
The robot goes to the storage room
The robot asks if they have the number of boxes Alice needs
If they have the boxes, the robot picks up the boxes and places them in Alice's office
If they don't have the boxes, the robot tells Alice it could not get the boxes

# Final Corrected Instruction:
Ask Alice if she needs 1, 2, or 3 boxes. Go to the storage room and ask if they have that many boxes. If so, go place the boxes in Alice's office. Otherwise, tell Alice you could not get the boxes.
"""

def setup_instruction_revision_prompts(df, tokenizer):
    prompts = []
    for index, row in df.iterrows():
        prompt = row["prompt"]
        program = row["program"]
        if type(prompt) == type(np.nan) or type(program) == type(np.nan):
            continue
        user_task_new = f"# Original Instruction:\n{prompt}\n# Robot Program:\n{program}"
        prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": example1_task},
            {"role": "assistant", "content": example1_response},
            {"role": "user", "content": example2_task},
            {"role": "assistant", "content": example2_response},
            {"role": "user", "content": user_task_new}
        ]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        prompts.append(prompt)

    return prompts

def setup_decision_prompts(df, revisions, tokenizer):
    prompts = []
    for idx, (_, row)in enumerate(df.iterrows()):
        prompt = row["prompt"]
        program = row["program"]
        explanation_new = revisions[idx]
        
        question = f"""Which of the following is the best instruction for the given program?
### Program
{program}

### Instruction 1
{prompt}

### Instruction 2
{explanation_new}

### Reasoning and Final Result
1. Think step by step first. 
2. Then choose the best instruction for the given program between ### Instruction 1 and ### Instruction 2
"""
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. You are helping a user to determine the best instruction for the given program. You need to first reason about each instruction and then output either ### Instruction 1 or ### Instruction 2."},
            {"role": "user", "content": question}
        ]
        
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
        prompts.append(prompt)
    
    return prompts

def post_process_revision(revision, original):
    revision = revision.strip()
    if "final corrected instruction:" not in revision.lower():
        return original
    else:
        revision = re.split("final corrected instruction:", revision, flags=re.IGNORECASE)
        revision = revision[-1].strip()
        return revision

def parse_decision(decision):
    idx1 = decision.rfind("instruction 1")
    idx2 = decision.rfind("instruction 2")
    if idx1 == -1:
        return True 
    elif idx2 == -1:
        return False
    elif idx1 < idx2:
        return True 
    elif idx2 < idx1:
        return False

def post_process_decision(decision, original, revision):
    decision = decision.strip().lower()
    if "instruction 1" not in decision and "instruction 2" not in decision:
        return original
    
    status = parse_decision(decision)
    if status:
        return revision
    else:
        return original