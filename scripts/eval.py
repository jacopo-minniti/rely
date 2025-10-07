import os
import json
import numpy as np
from rely.utils import normalize_answer
import argparse

def process_json_file(path):
    """
    Processes a single JSON file to check for correctness based on two methods:
    1. Majority Vote: Compares 'majority_vote' with 'ground_truth'.
    2. Best of N: If available, finds the solution with the highest 'value'
       and compares its 'answer' with 'ground_truth'.
    
    Returns:
        A tuple containing:
        - is_majority_correct (bool or None): True if correct, False if incorrect, None if should be skipped.
        - best_of_n_applicable (bool): True if 'best of n' logic could be applied.
        - is_best_of_n_correct (bool): True if the 'best of n' answer is correct.
        - total_tokens (int): The total tokens from the file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data.get('ground_truth')
        
        # Get total tokens
        total_tokens = int(data.get('total_tokens', 0))

        # --- 1. Majority Vote Calculation (Original Logic) ---
        majority_vote = data.get('majority_vote')
        accuracy = data.get('accuracy')
        
        # Skip entries where accuracy or majority_vote is "N/A"
        if accuracy == "N/A" or majority_vote == "N/A":
            return None, False, False, total_tokens
            
        is_majority_correct = False
        if ground_truth is not None and majority_vote is not None:
            # Skip if majority_vote is empty string
            if str(majority_vote).strip():
                normalized_gt = normalize_answer(str(ground_truth))
                normalized_mv = normalize_answer(str(majority_vote))
                if (normalized_gt == normalized_mv or ground_truth == majority_vote) and normalized_gt != "":
                    is_majority_correct = True

        # --- 2. Best of N Calculation ---
        best_of_n_applicable = False
        is_best_of_n_correct = False
        
        solutions = data.get('solutions')
        # Check if 'solutions' is a non-empty list
        if isinstance(solutions, list) and solutions and ground_truth is not None:
            best_solution = None
            max_value = float('-inf')
            
            # Find the solution with the highest 'value'
            best_solution_answer_key = None
            for sol in solutions:
                # Accept either 'answer' or 'final_answer' as the answer key
                answer_key = None
                if isinstance(sol, dict) and 'value' in sol:
                    if 'answer' in sol:
                        answer_key = 'answer'
                    elif 'final_answer' in sol:
                        answer_key = 'final_answer'
                # Only consider solutions with non-empty answers
                if answer_key is not None and sol.get(answer_key) and str(sol[answer_key]).strip():
                    try:
                        current_value = float(sol['value'])
                        if current_value > max_value:
                            max_value = current_value
                            best_solution = sol
                            best_solution_answer_key = answer_key
                    except (ValueError, TypeError):
                        # Ignore solutions with non-numeric 'value'
                        continue
            # If a best solution was found, compare its answer to the ground truth
            if best_solution is not None and best_solution_answer_key is not None:
                best_of_n_applicable = True
                normalized_gt = normalize_answer(str(ground_truth))
                normalized_best_answer = normalize_answer(str(best_solution[best_solution_answer_key]))
                if (normalized_gt == normalized_best_answer) and normalized_gt != "":
                    is_best_of_n_correct = True

        return is_majority_correct, best_of_n_applicable, is_best_of_n_correct, total_tokens

    except Exception as e:
        # For any file reading/parsing error, print warning and skip file
        print(f"Warning: Failed to process file {path}: {e}")
        return None, False, False, 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process JSON files and compute evaluation statistics.')
    parser.add_argument('input_dir', help='Directory containing JSON files to process')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"❌ Error: The directory does not exist: '{input_dir}'")
        exit(1)

    # Recursively find all .json files
    json_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.endswith('.json')]

    if not json_files:
        print(f"❌ No JSON files found in '{input_dir}' or its subfolders.")
        exit(1)

    # --- Initialize counters ---
    total_files = 0
    # Counters for majority vote
    majority_correct_count = 0
    # Counters for best of n
    best_of_n_applicable_count = 0
    best_of_n_correct_count = 0
    # List to store all token counts for percentile calculation
    token_counts = []

    for path in sorted(json_files):
        is_majority_correct, best_of_n_applicable, is_best_of_n_correct, total_tokens = process_json_file(path)
        
        # Skip entries that should not be considered (accuracy or majority_vote is "N/A")
        if is_majority_correct is None:
            continue
            
        total_files += 1
        
        # Accumulate majority vote stats
        if is_majority_correct:
            majority_correct_count += 1
        
        # Accumulate best of n stats
        if best_of_n_applicable:
            best_of_n_applicable_count += 1
            if is_best_of_n_correct:
                best_of_n_correct_count += 1
        
        # Store token count for percentile calculation
        token_counts.append(total_tokens)

    # Calculate statistics
    majority_percent_correct = (majority_correct_count / total_files) if total_files > 0 else 0.0
    
    # Best-of-N is only applicable when solutions have meaningful value scores
    # For self-consistency, this will be 0 since all solutions have value=1.0
    if best_of_n_applicable_count > 0:
        best_of_n_percent_correct = (best_of_n_correct_count / best_of_n_applicable_count)
    else:
        best_of_n_percent_correct = 0.0
    
    mean_tokens = np.mean(token_counts) if token_counts else 0.0
    percentile_95_tokens = np.percentile(token_counts, 95) if token_counts else 0.0

    # Extract B1 and B3 from directory name (assuming format contains "max_B1_B3")
    dir_name = os.path.basename(input_dir.rstrip('/'))
    b1, b3 = 0, 0  # default values
    if "max_" in dir_name:
        parts = dir_name.split("max_")[1].split("_")
        if len(parts) >= 2:
            try:
                b1 = int(parts[0])
                b3 = int(parts[1])
            except ValueError:
                pass

    # Print the number of samples evaluated
    print(f"Number of samples evaluated: {total_files}")
    
    # Print results in dictionary format
    result = {
        "B1": b1,
        "B3": b3,
        "tokens generated": round(mean_tokens),
        "bon": round(best_of_n_percent_correct, 4),
        "maj": round(majority_percent_correct, 4)
    }
    print(json.dumps(result))