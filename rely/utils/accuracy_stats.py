import json
import glob
from collections import Counter
import os

def _find_result_files(base_path: str) -> list[str]:
    """
    Discovers all unique result files ('results.json', 'summary.json')
    within the specified base directory structure.

    Args:
        base_path: The root directory to search from.

    Returns:
        A sorted list of unique file paths.
    """
    patterns = [
        os.path.join(base_path, 'question_*', 'run_*', 'results.json'),
        os.path.join(base_path, 'question_*', 'run_*', 'summary.json'),
        os.path.join(base_path, 'question_*', 'results.json'),
        os.path.join(base_path, 'question_*', 'summary.json')
    ]
    
    all_found_files = []
    for pattern in patterns:
        all_found_files.extend(glob.glob(pattern))
        
    return sorted(list(set(all_found_files)))

def _parse_file_data(file_path: str) -> dict | None:
    """
    Parses a single JSON result file and extracts key metrics.
    Supports both 'summary.json' and 'results.json' formats.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A dictionary containing extracted metrics or None if parsing fails.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # New format ('summary.json') with a nested 'evaluation' key
        if "evaluation" in data:
            evaluation_data = data.get("evaluation", {})
            correct_count = evaluation_data.get("correct_count", 0)
            return {
                "is_most_consistent_correct": evaluation_data.get("is_most_consistent_correct", False),
                "at_least_one_is_correct": correct_count > 0,
                "correct_count_in_file": correct_count,
                "total_answers_in_file": data.get("num_samples", 0),
            }
        
        # Original format ('results.json')
        else:
            correct_answer = data.get("correct_answer")
            all_answers = data.get("all_answers", [])

            if not all_answers or correct_answer is None:
                print(f"Warning: Skipping file {file_path} due to missing data.")
                return None

            answer_counts = Counter(all_answers)
            most_common_answer = answer_counts.most_common(1)[0][0]
            
            return {
                "is_most_consistent_correct": most_common_answer == correct_answer,
                "at_least_one_is_correct": correct_answer in all_answers,
                "correct_count_in_file": data.get("correct_count", 0),
                "total_answers_in_file": data.get("total_answers", 0),
            }
            
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Warning: Could not process file {file_path}. Error: {e}")
        return None

# --- API Functions ---

def process_and_calculate_metrics(base_path: str) -> dict | None:
    """
    Calculates performance metrics from JSON result files. This is the main
    API function that processes data and returns it in a structured format.

    Args:
        base_path: The base directory containing the result files.

    Returns:
        A dictionary containing aggregated metrics, or None if no files are found.
    """
    result_files = _find_result_files(base_path)
    if not result_files:
        print("Error: No result files ('results.json' or 'summary.json') found.")
        print(f"Please ensure results exist in '{os.path.join(base_path, 'question_*')}' directories.")
        return None

    # Initialize counters
    processed_questions = 0
    most_consistent_correct = 0
    at_least_one_correct = 0
    total_correct_answers = 0
    total_possible_answers = 0

    # Process each result file
    for file_path in result_files:
        metrics = _parse_file_data(file_path)
        if metrics:
            processed_questions += 1
            if metrics["is_most_consistent_correct"]:
                most_consistent_correct += 1
            if metrics["at_least_one_is_correct"]:
                at_least_one_correct += 1
            
            total_correct_answers += metrics["correct_count_in_file"]
            total_possible_answers += metrics["total_answers_in_file"]

    # --- Prepare final results ---
    if processed_questions == 0:
        print("Warning: No valid result files were processed.")
        return None

    most_consistent_pct = (most_consistent_correct / processed_questions) * 100
    at_least_one_pct = (at_least_one_correct / processed_questions) * 100
    mean_accuracy_pct = (total_correct_answers / total_possible_answers) * 100 if total_possible_answers > 0 else 0

    return {
        "processed_questions": processed_questions,
        "most_consistent_correct_count": most_consistent_correct,
        "at_least_one_correct_count": at_least_one_correct,
        "total_correct_answers": total_correct_answers,
        "total_possible_answers": total_possible_answers,
        "most_consistent_pct": most_consistent_pct,
        "at_least_one_pct": at_least_one_pct,
        "mean_accuracy_pct": mean_accuracy_pct
    }

def display_metrics(metrics: dict):
    """
    Prints the calculated metrics to the console in a human-readable format.

    Args:
        metrics: A dictionary of results from process_and_calculate_metrics.
    """
    if not metrics:
        print("No metrics to display.")
        return

    print(f"\n--- Aggregated Results from {metrics['processed_questions']} questions ---")
    print(f"- Most consistent answer is correct: {metrics['most_consistent_correct_count']} out of {metrics['processed_questions']} ({metrics['most_consistent_pct']:.2f}%)")
    print(f"- At least one answer is correct: {metrics['at_least_one_correct_count']} out of {metrics['processed_questions']} ({metrics['at_least_one_pct']:.2f}%)")
    print(f"- Mean accuracy (total correct / total generated): {metrics['mean_accuracy_pct']:.2f}% ({metrics['total_correct_answers']}/{metrics['total_possible_answers']})")
    print("----------------------------------------------------\n")


# calculated_metrics = process_and_calculate_metrics(base_dir)
# display_metrics(calculated_metrics)