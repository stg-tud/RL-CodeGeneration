from rewards.robo_instruct.robo_instruct.robosim.robosim_tracer import rejection_sampling_simulation

import pandas as pd
import argparse
from joblib import Parallel, delayed
 
def drop_nan(data):
    print("Dropping NaN values...")
    print("Original length:", len(data))
    columns = ["prompt", "program", "revision", "text"]
    for col in columns:
        if col not in data.columns:
            print(col)
            continue
        data = data.dropna(subset=col)
    print("Dropna New length:", len(data))
    print()
    return data

def deduplicate_text_program_col(data):
    print("Deduplicating all columns...")
    print("Original length:", len(data))
    columns = ["program", "text"]
    for col in columns:
        if col not in data.columns:
            print(col)
            continue
        data = data.drop_duplicates(subset=col)
    print("Dedup New length:", len(data))
    print()
    return data

def strip_all_four_cols(data):
    print("Stripping all columns...")
    columns = ["prompt", "program", "revision", "text"]
    for col in columns:
        if col not in data.columns:
            print(col)
            continue
        data[col] = data[col].apply(lambda x: x.strip())
    return data

APIS = [
    "get_current_location(", 
    "get_all_rooms(",
    "is_in_room(",
    "go_to(",
    "ask(",
    "say(",
    "pick(",
    "place("
    ]

def remove_unused_apis(data, apis):
    # Create a mask that is True for rows where any API is found in the 'program' column
    mask = data['program'].apply(lambda x: any(api in x for api in apis))
    # Filter the DataFrame using the mask to keep only rows where the mask is True
    filtered_data = data[mask]
    return filtered_data

def remove_ellipsis(data):
    # remove ellipsis that affect program generations
    print("Removing ellipsis...")
    print("Before removing ellipsis", len(data))
    data.reset_index(drop=True, inplace=True)

    count = 0
    # preprocess
    tmp_programs = []
    for idx in range(len(data["program"])):
        code = data.iloc[idx]["program"]
        if "..." in code:
            tmp_code = code.replace('...', '')
            tmp_programs.append([tmp_code, idx])            
    results = Parallel(n_jobs=-2)(delayed(rejection_sampling_simulation)(tmp_program, 2, 100, 2, False) for tmp_program, idx in tmp_programs)
    idx_to_remove = []
    for idx, (trace, status) in enumerate(results):
        if not status:
            idx_to_remove.append(tmp_programs[idx][1])
            id = tmp_programs[idx][1]
            if "..." in data.iloc[id]["program"] and "\"...\"" not in data.iloc[id]["program"]:
                print("Ellipsis in program")
            count += 1
    print(idx_to_remove)
    data.drop(idx_to_remove, inplace=True)
    print("After removing ellipsis", len(data))
    return data

if __name__ == "__main__":
    overwrite_data = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite_data", action="store_true")
    parser.add_argument("-i", "--input_data", type=str)
    parser.add_argument("-s", "--save_data", type=str)
    args = parser.parse_args()
    
    original_data = pd.read_csv(args.input_data)

    data = drop_nan(original_data)
    data = strip_all_four_cols(data)
    data = deduplicate_text_program_col(data)
    data = remove_unused_apis(data, APIS)
    data = remove_ellipsis(data)
    data = remove_ellipsis(data)
    
    if args.overwrite_data:
        df_subset = data.iloc[:5000]
        df_subset.to_csv(args.input_data, index=False)
    else:
        data.to_csv(args.save_data, index=False)
    
