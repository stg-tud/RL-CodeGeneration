from rewards.robo_instruct.roboeval.benchmark.simple_tracer import code_replace, bounded_run, construct_trace_element_from_str

import random 
import re
import sys

def replace_with_true_or_false(text):
    # Pattern explanation remains the same as before.
    should_replace = random.choices([True, False], weights=[0.8, 0.2], k=1)[0]
    if not should_replace:
        return text
    # change get_all_rooms
    pattern1 = r'(["\'])(?:(?!\1).)*\1\s+(not\s+)?in\s+(\w+)'
    
    pattern2 = r'(["\'])(?:(?!\1).)*\1\s+(not\s+)?in\s+get_all_rooms\(\)'
    pattern3 = r'(["\'])(?:(?!\1).)*\1\s+(not\s+)?in\s+get_current_location\(\)'
    
    # Replacement function: Randomly returns 'True' or 'False'
    def replacement1(match):
        if "ask" in match.group(0):
            return match.group(0)
        if "get_all_rooms" in match.group(0):
            return match.group(0)
        if "get_current_location" in match.group(0):
            return match.group(0)
        return str(random.choice(['True', 'False', match.group(0)]))
    
    def replacement2(match):
        return str(random.choice(['True', 'False', match.group(0)]))
      
    # Use re.sub with a replacement function
    replaced_text = re.sub(pattern1, replacement1, text, flags=re.IGNORECASE)
    replaced_text = re.sub(pattern2, replacement2, replaced_text, flags=re.IGNORECASE)
    replaced_text = re.sub(pattern3, replacement2, replaced_text, flags=re.IGNORECASE)
    
    return replaced_text


def rejection_sampling_simulation(program : str, simulation_timeout: int, resampling_count: int, 
                                  allowed_timeout_count: int, VERBOSE: bool, replace_w_tf = True):   
    timeout_count = 0
    passed_test_count = 0
    trace_elements_list = []
    for _ in range(resampling_count):
        if replace_w_tf:
          program_modified = replace_with_true_or_false(program)
        else:
          program_modified = program
        program_modified = code_replace(program_modified)
        project_path = sys.path[0]
        p = f"""
import sys
import time
sys.path.insert(0, "{project_path}")
from rewards.robo_instruct.robo_instruct.robosim.robosim_simulator import RobotExecution
from rewards.robo_instruct.robo_instruct.robosim.robosim_initial_state import initial_state
robot = RobotExecution(initial_state)
{program_modified}
"""
        sys.stdout.flush()
        ret = bounded_run(["python", "-c", p], timeout_seconds=simulation_timeout, max_output_size=1000000)

        if ret.exit_code == -1:
            if VERBOSE:
                print("PYTHON TIMED_OUT:", ret.exit_code)
                timeout_count += 1
            if timeout_count >= allowed_timeout_count:
                py_error = "TimeOutError: Program did not terminate"
                if VERBOSE:
                    print(ret.stdout)
                #return py_error, False
        elif ret.exit_code == 0:
            passed_test_count += 1
            if VERBOSE:
                print("Program Successful")
                trace_elements = construct_trace_element_from_str(ret.stdout)
                trace_elements_list.append(trace_elements)
        else:
            py_error = ret.stderr.strip("\n").split("\n")[-1].strip()
            py_error = ret.stderr
            if VERBOSE:
                print("PYTHON RUNTIME ERROR: ", ret.exit_code)
                print(py_error)
                print(ret.stdout)
            #return py_error, False

    if passed_test_count > 0:
        return "Success", True, passed_test_count
    else:
        return "failed", False, passed_test_count
