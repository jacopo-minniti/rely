import os
import json
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
        - is_majority_correct (bool): True if the majority vote is correct.
        - sample_accuracy (float): The accuracy value from the JSON.
        - best_of_n_applicable (bool): True if 'best of n' logic could be applied.
        - is_best_of_n_correct (bool): True if the 'best of n' answer is correct.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data.get('ground_truth')

        # --- 1. Majority Vote Calculation (Original Logic) ---
        majority_vote = data.get('majority_vote')
        is_majority_correct = False
        if ground_truth is not None and majority_vote is not None:
            normalized_gt = normalize_answer(str(ground_truth))
            normalized_mv = normalize_answer(str(majority_vote))
            if (normalized_gt == normalized_mv) and normalized_gt != "":
                is_majority_correct = True
        
        try:
            sample_accuracy = float(str(data.get('accuracy', '0.00%')).strip('%'))
        except (ValueError, TypeError):
            sample_accuracy = 0.0

        # --- 2. Best of N Calculation (New Logic) ---
        best_of_n_applicable = False
        is_best_of_n_correct = False
        
        solutions = data.get('solutions')
        # Check if 'solutions' is a non-empty list
        if isinstance(solutions, list) and solutions and ground_truth is not None:
            best_solution = None
            max_value = float('-inf')
            
            # Find the solution with the highest 'value'
            for sol in solutions:
                # Accept either 'answer' or 'final_answer' as the answer key
                answer_key = None
                if isinstance(sol, dict) and 'value' in sol:
                    if 'answer' in sol:
                        answer_key = 'answer'
                    elif 'final_answer' in sol:
                        answer_key = 'final_answer'
                if answer_key is not None:
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
            if best_solution is not None:
                best_of_n_applicable = True
                normalized_gt = normalize_answer(str(ground_truth))
                normalized_best_answer = normalize_answer(str(best_solution[best_solution_answer_key]))
                if (normalized_gt == normalized_best_answer) and normalized_gt != "":
                    is_best_of_n_correct = True

        return is_majority_correct, sample_accuracy, best_of_n_applicable, is_best_of_n_correct

    except Exception:
        # For any file reading/parsing error, count as incorrect
        return False, 0.0, False, False

def main():
    parser = argparse.ArgumentParser(description='Process JSON files and compute evaluation statistics.')
    parser.add_argument('input_dir', help='Directory containing JSON files to process')
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"❌ Error: The directory does not exist: '{input_dir}'")
        return

    # Recursively find all .json files
    json_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.endswith('.json')]

    if not json_files:
        print(f"❌ No JSON files found in '{input_dir}' or its subfolders.")
        return

    # --- Initialize counters ---
    total_files = 0
    # Counters for majority vote
    majority_correct_count = 0
    acc_sum = 0.0
    # Counters for best of n
    best_of_n_applicable_count = 0
    best_of_n_correct_count = 0

    for path in sorted(json_files):
        total_files += 1
        is_majority_correct, accuracy, best_of_n_applicable, is_best_of_n_correct = process_json_file(path)
        
        # Accumulate majority vote stats
        acc_sum += accuracy
        if is_majority_correct:
            majority_correct_count += 1
        
        # Accumulate best of n stats
        if best_of_n_applicable:
            best_of_n_applicable_count += 1
            if is_best_of_n_correct:
                best_of_n_correct_count += 1

    print(f"\n✅ Processed {total_files} JSON files in '{input_dir}'")
    print("-" * 40)
    
    # --- Print Majority Vote Statistics ---
    percent_correct = (majority_correct_count / total_files) * 100 if total_files > 0 else 0.0
    mean_accuracy = (acc_sum / total_files) if total_files > 0 else 0.0

    print("📊 Majority Vote Statistics")
    print(f"   Correct: {majority_correct_count} / {total_files} ({percent_correct:.2f}%)")
    print(f"   Mean sample accuracy: {mean_accuracy:.2f}%")
    print("-" * 40)

    # --- Print Best of N Statistics (if applicable) ---
    if best_of_n_applicable_count > 0:
        best_of_n_percent_correct = (best_of_n_correct_count / best_of_n_applicable_count) * 100
        print("🏆 Best of N (by value) Statistics")
        print(f"   Applicable in: {best_of_n_applicable_count} / {total_files} files")
        print(f"   Correct: {best_of_n_correct_count} / {best_of_n_applicable_count} ({best_of_n_percent_correct:.2f}%)")
        print("-" * 40)

if __name__ == '__main__':
    main()