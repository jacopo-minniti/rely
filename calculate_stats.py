import json
import glob
from collections import Counter
import os

def calculate_metrics():
    """
    Calculates and prints performance metrics from JSON result files.
    """
    # Define the path to search for result files.
    # This pattern finds all results.json files within any subdirectory 
    # that follows the structure 'question_*/run_*'.
    path_pattern = os.path.join('uats_results', 'question_*', 'run_*', 'results.json')
    
    # Find all files matching the pattern
    result_files = glob.glob(path_pattern)
    
    if not result_files:
        print("Error: No result files found.")
        print(f"Please ensure the script is in the correct directory and that files exist at the path: {path_pattern}")
        return

    # Initialize counters
    total_questions = len(result_files)
    most_consistent_correct = 0
    at_least_one_correct = 0
    total_correct_answers = 0
    total_possible_answers = 0

    # Process each result file
    for file_path in result_files:
        print(file_path)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract necessary data from JSON
            correct_answer = data.get("correct_answer")
            all_answers = data.get("all_answers", [])
            
            # Ensure there are answers to process
            if not all_answers:
                continue

            # --- Metric 1: Most consistent answer is correct ---
            # Find the most common answer using collections.Counter
            answer_counts = Counter(all_answers)
            most_common_answer = answer_counts.most_common(1)[0][0]
            if most_common_answer == correct_answer:
                most_consistent_correct += 1

            # --- Metric 2: At least one answer is correct ---
            if correct_answer in all_answers:
                at_least_one_correct += 1
                
            # --- Metric 3: Mean accuracy ---
            # Sum the counts provided directly in the JSON for mean accuracy calculation
            total_correct_answers += data.get("correct_count", 0)
            total_possible_answers += data.get("total_answers", 0)

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            total_questions -= 1 # Adjust total if a file is malformed

    # --- Calculate final percentages ---
    if total_questions > 0:
        most_consistent_pct = (most_consistent_correct / total_questions) * 100
        at_least_one_pct = (at_least_one_correct / total_questions) * 100
    else:
        most_consistent_pct = 0
        at_least_one_pct = 0

    if total_possible_answers > 0:
        mean_accuracy_pct = (total_correct_answers / total_possible_answers) * 100
    else:
        mean_accuracy_pct = 0

    # --- Print the final results ---
    print("\n--- Aggregated Results ---")
    print(f"- Most consistent answer is correct: {most_consistent_correct} out of {total_questions} ({most_consistent_pct:.2f}%)")
    print(f"- At least one answer is correct: {at_least_one_correct} out of {total_questions} ({at_least_one_pct:.2f}%)")
    print(f"- Mean accuracy (total correct / total generated): {mean_accuracy_pct:.2f}% ({total_correct_answers}/{total_possible_answers})")
    print("--------------------------\n")

if __name__ == "__main__":
    calculate_metrics()