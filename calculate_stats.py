import json
import glob
from collections import Counter
import os

def calculate_metrics(base_path):
    """
    Calculates and prints performance metrics from JSON result files,
    supporting both 'results.json' and 'summary.json' formats across
    two different directory structures.
    """
    # --- File Discovery ---
    # Define the base directory for the results.

    # Define patterns for all supported directory structures and filenames.
    # This finds files both with and without the intermediate 'run_*' directory.
    patterns = [
        # Structure with 'run_*' subdirectory
        os.path.join(base_path, 'question_*', 'run_*', 'results.json'),
        os.path.join(base_path, 'question_*', 'run_*', 'summary.json'),
        # Structure without 'run_*' subdirectory
        os.path.join(base_path, 'question_*', 'results.json'),
        os.path.join(base_path, 'question_*', 'summary.json')
    ]

    # Collect all files that match any of the defined patterns.
    all_found_files = []
    for pattern in patterns:
        all_found_files.extend(glob.glob(pattern))

    # Use a set to get a unique list of file paths, then convert back to a list.
    result_files = list(set(all_found_files))

    if not result_files:
        print("Error: No result files ('results.json' or 'summary.json') found.")
        print(f"Please ensure results exist in '{os.path.join(base_path, 'question_*')}' or '{os.path.join(base_path, 'question_*', 'run_*')}' directories.")
        return

    # --- Initialize counters ---
    # This counter tracks successfully processed files.
    processed_questions = 0
    most_consistent_correct = 0
    at_least_one_correct = 0
    total_correct_answers = 0
    total_possible_answers = 0

    # --- Process each result file ---
    for file_path in sorted(result_files): # Sorting for consistent output order
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # These variables will be populated based on the JSON format.
            is_most_consistent_correct = False
            at_least_one_is_correct = False
            correct_count_in_file = 0
            total_answers_in_file = 0

            # --- Check format and extract data ---
            # The 'summary.json' format has a nested 'evaluation' key.
            if "evaluation" in data:
                # --- New format ('summary.json') processing ---
                evaluation_data = data.get("evaluation", {})
                is_most_consistent_correct = evaluation_data.get("is_most_consistent_correct", False)
                correct_count_in_file = evaluation_data.get("correct_count", 0)
                at_least_one_is_correct = correct_count_in_file > 0
                total_answers_in_file = data.get("num_samples", 0)

            else:
                # --- Original format ('results.json') processing ---
                correct_answer = data.get("correct_answer")
                all_answers = data.get("all_answers", [])

                # Skip if essential data is missing
                if not all_answers or correct_answer is None:
                    print(f"Warning: Skipping file {file_path} due to missing 'all_answers' or 'correct_answer' field.")
                    continue

                # Calculate metrics on the fly for the original format
                answer_counts = Counter(all_answers)
                most_common_answer = answer_counts.most_common(1)[0][0]
                is_most_consistent_correct = (most_common_answer == correct_answer)
                at_least_one_is_correct = (correct_answer in all_answers)

                correct_count_in_file = data.get("correct_count", 0)
                total_answers_in_file = data.get("total_answers", 0)

            # --- Update aggregate counters after successful data extraction ---
            processed_questions += 1
            if is_most_consistent_correct:
                most_consistent_correct += 1
            if at_least_one_is_correct:
                at_least_one_correct += 1

            total_correct_answers += correct_count_in_file
            total_possible_answers += total_answers_in_file

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            # Do not increment processed_questions count if a file is malformed

    # --- Calculate final percentages ---
    if processed_questions > 0:
        most_consistent_pct = (most_consistent_correct / processed_questions) * 100
        at_least_one_pct = (at_least_one_correct / processed_questions) * 100
    else:
        print("Warning: No valid result files were processed.")
        most_consistent_pct = 0
        at_least_one_pct = 0

    if total_possible_answers > 0:
        mean_accuracy_pct = (total_correct_answers / total_possible_answers) * 100
    else:
        mean_accuracy_pct = 0

    # --- Print the final results ---
    print(f"\n--- Aggregated Results from {processed_questions} questions ---")
    print(f"- Most consistent answer is correct: {most_consistent_correct} out of {processed_questions} ({most_consistent_pct:.2f}%)")
    print(f"- At least one answer is correct: {at_least_one_correct} out of {processed_questions} ({at_least_one_pct:.2f}%)")
    print(f"- Mean accuracy (total correct / total generated): {mean_accuracy_pct:.2f}% ({total_correct_answers}/{total_possible_answers})")
    print("----------------------------------------------------\n")

if __name__ == "__main__":
    base = "/Users/jacopominniti/Desktop/results/uats/classification_05_all_8000"
    calculate_metrics(base)