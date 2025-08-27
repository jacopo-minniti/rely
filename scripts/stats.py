import os
import json
import glob

def parse_results_json(data: dict):
    """
    Parse a results.json file format and extract relevant statistics.
    
    Args:
        data (dict): The loaded JSON data from results.json
        
    Returns:
        tuple: (is_majority_correct, accuracy_percentage)
    """
    correct_answer = data.get('correct_answer')
    all_answers = data.get('all_answers', [])
    
    if not all_answers:
        return False, 0.0
    
    # Calculate majority vote
    answer_counts = {}
    for answer in all_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    if not answer_counts:
        return False, 0.0
        
    majority_answer = max(answer_counts, key=lambda x: answer_counts[x])
    is_majority_correct = majority_answer == correct_answer
    
    # Calculate accuracy as percentage of correct answers
    correct_count = data.get('correct_count', 0)
    total_answers = data.get('total_answers', len(all_answers))
    accuracy_percentage = (correct_count / total_answers * 100) if total_answers > 0 else 0.0
    
    return is_majority_correct, accuracy_percentage

def calculate_sbs_statistics(base_path: str):
    """
    Analyzes self-consistency results from a directory of JSON summaries.

    This function supports two formats:
    1. Folders matching 'q_*' with 'summary.json' files
    2. Folders matching 'question_*' with 'run_*' subdirectories containing 'results.json' files
    
    Computes aggregate statistics about the correctness of the generated answers.

    Args:
        base_path (str): The path to the run directory containing question folders.
    """
    # Check for both patterns: q_* and question_*
    q_folders = sorted(glob.glob(os.path.join(base_path, 'q_*')))
    question_folders = sorted(glob.glob(os.path.join(base_path, 'question_*')))
    
    total_processed = 0
    majority_vote_correct_count = 0
    total_accuracy_sum = 0.0
    total_tokens_sum = 0
    total_tokens_count = 0

    if q_folders:
        print(f"🔍 Found {len(q_folders)} 'q_*' folders, processing with summary.json format...")
        for folder in q_folders:
            result = process_summary_json(folder)
            if result is not None:
                is_correct, accuracy, tokens = result
                if is_correct:
                    majority_vote_correct_count += 1
                total_accuracy_sum += accuracy
                total_processed += 1
                if tokens is not None:
                    total_tokens_sum += tokens
                    total_tokens_count += 1
    
    if question_folders:
        print(f"🔍 Found {len(question_folders)} 'question_*' folders, processing with results.json format...")
        for folder in question_folders:
            # Look for run_* subdirectories
            run_folders = sorted(glob.glob(os.path.join(folder, 'run_*')))
            if not run_folders:
                print(f"⚠️ Warning: No 'run_*' subdirectories found in '{folder}'. Skipping.")
                continue
            # Process the first run folder (or you could aggregate across all runs)
            for run_folder in run_folders:
                result = process_results_json(run_folder)
                if result is not None:
                    is_correct, accuracy, tokens = result
                    if is_correct:
                        majority_vote_correct_count += 1
                    total_accuracy_sum += accuracy
                    total_processed += 1
                    if tokens is not None:
                        total_tokens_sum += tokens
                        total_tokens_count += 1
                break  # Only process the first run folder for each question
    
    if total_processed == 0:
        print(f"❌ Error: No valid question folders found in the path: '{base_path}'")
        print("Looking for either 'q_*' folders with 'summary.json' or 'question_*' folders with 'run_*/results.json'")
        return

    print(f"🔍 Processed {total_processed} questions from '{base_path}'...")

    # --- Calculate Final Percentages ---
    majority_vote_correct_percent = (majority_vote_correct_count / total_processed) * 100
    mean_accuracy_percent = total_accuracy_sum / total_processed
    mean_tokens = (total_tokens_sum / total_tokens_count) if total_tokens_count > 0 else 0.0

    # --- Display Results ---
    print("\n" + "="*50)
    print(f"📊 Overall Statistics for: {os.path.basename(base_path)}")
    print("="*50)
    print(
        f"Majority vote is correct: {majority_vote_correct_count} out of {total_processed} "
        f"({majority_vote_correct_percent:.2f}%)"
    )
    print(
        f"Mean accuracy per dataset: {mean_accuracy_percent:.2f}%"
    )
    print(
        f"Mean total_generated_tokens: {mean_tokens:.2f}"
    )
    print("="*50)

def process_summary_json(folder: str):
    """Process a folder with summary.json format."""
    summary_file = os.path.join(folder, 'summary.json')

    if not os.path.exists(summary_file):
        print(f"⚠️ Warning: 'summary.json' not found in '{folder}'. Skipping.")
        return None

    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if the majority vote answer was correct
        is_correct = data.get('majority_vote') == data.get('ground_truth')

        # Get accuracy
        accuracy_str = data.get('accuracy', '0.00%')
        accuracy = float(accuracy_str.strip('%'))

        # Get total_generated_tokens if present
        tokens = data.get('total_generated_tokens')
        if tokens is not None:
            try:
                tokens = int(tokens)
            except Exception:
                tokens = None

        return is_correct, accuracy, tokens

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"❌ Error processing '{summary_file}': {e}. Skipping.")
        return None

def process_results_json(folder: str):
    """Process a folder with results.json format."""
    results_file = os.path.join(folder, 'results.json')

    if not os.path.exists(results_file):
        print(f"⚠️ Warning: 'results.json' not found in '{folder}'. Skipping.")
        return None

    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # parse_results_json returns (is_majority_correct, accuracy_percentage)
        is_correct, accuracy = parse_results_json(data)
        tokens = data.get('total_generated_tokens')
        if tokens is not None:
            try:
                tokens = int(tokens)
            except Exception:
                tokens = None
        return is_correct, accuracy, tokens

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"❌ Error processing '{results_file}': {e}. Skipping.")
        return None


if __name__ == '__main__':
    run_path = 'sbs_results'

    if os.path.isdir(run_path):
        calculate_sbs_statistics(run_path)
    else:
        print(f"❌ Error: The base path does not exist: '{run_path}'")
        print("Please ensure the path is correct and you are running the script from the correct location.")