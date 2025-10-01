import math
import random
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from rely.utils import load_dataset, save_dataset, extract_final_answer, normalize_answer


def calculate_cwe(completions: List[str], ground_truth: str) -> float:
    """
    Calculate the Correctness-Weighted Entropy (CWE) metric.
    CWE = P(Correct) × H(X)
    Where:
    - P(Correct) is the proportion of correct answers
    - H(X) is the entropy of the full answer distribution
    
    A high CWE indicates high correctness with high diversity in answers.
    A low CWE indicates either low correctness or low diversity in answers.
    """
    if not completions:
        return 0.0
    
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # Collect all answers
    all_answers = []
    correct_count = 0
    
    for completion in completions:
        extracted = extract_final_answer(completion)
        if extracted:
            normalized_extracted = normalize_answer(extracted)
            all_answers.append(normalized_extracted)
            if normalized_extracted == normalized_ground_truth:
                correct_count += 1
        else:
            # If no answer is extracted, treat as a specific answer
            all_answers.append("<no_answer>")
    
    total_answers = len(completions)
    
    # If no answers, return 0
    if total_answers == 0:
        return 0.0
    
    # Calculate correctness rate
    correctness_rate = correct_count / total_answers
    
    # If no diversity (all same answer), return 0
    if len(set(all_answers)) <= 1:
        return 0.0
    
    # Calculate entropy of all answers
    answer_counts = {}
    for answer in all_answers:
        answer_counts[answer] = answer_counts.get(answer, 0) + 1
    
    entropy = 0.0
    for count in answer_counts.values():
        probability = count / total_answers
        if probability > 0:
            entropy -= probability * math.log(probability)
    
    # Calculate CWE
    cwe = correctness_rate * entropy
    
    return cwe


def format_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single item from completer format to process-reward format.
    Returns a single formatted item with full CoT and raw CWE values for each step.
    
    Args:
        item: The dataset item to format
    """
    original_item = item["original_item"]
    samples = item["samples"]

    # Extract the question and ground truth solution
    question = original_item.get("question", "")
    solution = original_item.get("solution", "")
    
    # Get the full CoT from the original item
    full_cot = original_item.get("attempt", "")
    if not full_cot:
        # If no attempt field, try to get the longest cut_cot from samples
        full_cot = max((sample["cut_cot"] for sample in samples), key=len, default="")
    
    # Split the full CoT into steps
    cot_steps = full_cot.split("\n\n") if full_cot else []
    
    # Create a mapping from step index to CEP value
    step_variances = {}
    
    for sample in samples:
        cut_cot = sample["cut_cot"]
        completions = sample["completions"]
        
        # Find which step this sample corresponds to
        if cut_cot:
            cut_steps = cut_cot.split("\n\n")
            step_index = len(cut_steps) - 1  # Index of the last step in this cut
            
            # Calculate CWE of completion final answers
            cwe_score = calculate_cwe(completions, solution)
            
            # Store the CWE for this step (use maximum if multiple samples for same step)
            if step_index not in step_variances:
                step_variances[step_index] = cwe_score
            else:
                step_variances[step_index] = max(step_variances[step_index], cwe_score)
    
    # Create variance values list matching the number of steps
    variance_values = []
    for i in range(len(cot_steps)):
        variance_values.append(step_variances.get(i, 0.0))  # Default to 0.0 if no sample for this step
    
    formatted_item = {
        "prompt": question.strip(),
        "completions": cot_steps,  # List of reasoning steps
        "variance_values": variance_values  # List of raw CWE values for each step
    }
    
    return formatted_item


def convert_dataset(input_file: str) -> List[Dict[str, Any]]:
    """
    Convert the entire dataset from completer format to process-reward format.
    Normalizes CWE values to [0, 1] range across the entire dataset.
    
    Args:
        input_file: Path to the input JSONL file
    """
    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    
    if not data:
        raise ValueError(f"Could not load data from {input_file}")
    
    print(f"Loaded {len(data)} items from the dataset")
    print("Using CWE-based evaluation with continuous values")
    
    # First pass: convert all items and collect all variance values
    all_formatted_items = []
    all_variance_values = []
    
    for item in data:
        formatted_item = format_dataset_item(item)
        all_formatted_items.append(formatted_item)
        all_variance_values.extend(formatted_item["variance_values"])
    
    # Normalize CWE scores to [0, 1] range
    if all_variance_values:
        min_variance = min(all_variance_values)
        max_variance = max(all_variance_values)
        
        print(f"CWE statistics: min={min_variance:.4f}, max={max_variance:.4f}")
        
        # Normalize variance values to [0, 1] range
        variance_range = max_variance - min_variance
        for item in all_formatted_items:
            if variance_range > 0:
                item["labels"] = [(val - min_variance) / variance_range for val in item["variance_values"]]
            else:
                item["labels"] = [0.0] * len(item["variance_values"])  # All values are the same
            del item["variance_values"]  # Remove the temporary field
    else:
        print("Warning: No variance values found in the dataset")
        for item in all_formatted_items:
            item["labels"] = [0.0] * len(item.get("variance_values", []))
            if "variance_values" in item:
                del item["variance_values"]
    
    print(f"Converted to {len(all_formatted_items)} formatted items with normalized CWE labels")
    return all_formatted_items


def remove_outliers_by_steps(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes outliers from the dataset based on the number of steps per item.
    Uses the 1.5 * IQR rule to identify and remove items with an unusually high number of steps.
    """
    if not data:
        return []

    print("\n--- Removing Outliers based on Step Count ---")
    
    step_lengths = [len(item['labels']) for item in data]
    
    if not step_lengths:
        print("No items with labels to process for outlier removal.")
        return data

    q1, q3 = np.percentile(step_lengths, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    print(f"Step count stats for outlier detection:")
    print(f"  - Q1 (25th percentile): {q1:.1f} steps")
    print(f"  - Q3 (75th percentile): {q3:.1f} steps")
    print(f"  - IQR: {iqr:.1f} steps")
    print(f"  - Upper bound for outliers (Q3 + 1.5*IQR): {upper_bound:.2f} steps")
    
    original_count = len(data)
    filtered_data = [item for item in data if len(item['labels']) <= upper_bound]
    removed_count = original_count - len(filtered_data)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers (items with more than {int(upper_bound)} steps).")
    else:
        print("No outliers were detected based on the number of steps.")
    
    print(f"Dataset size changed from {original_count} to {len(filtered_data)} items.")
    
    return filtered_data


def print_label_statistics(data: List[Dict[str, Any]], dataset_name: str = "Dataset") -> None:
    """
    Print detailed statistics about the continuous variance labels in the dataset.
    """
    if not data:
        print(f"{dataset_name} is empty")
        return
    
    print(f"\n--- {dataset_name} Statistics ---")
    
    # Overall stats
    total_items = len(data)
    steps_per_item = [len(item['labels']) for item in data]
    total_steps = sum(steps_per_item)
    print(f"Overall Info:")
    print(f"  - Total items: {total_items}")
    print(f"  - Total steps: {total_steps}")
    
    # Per-item step stats
    steps_np = np.array(steps_per_item)
    print(f"\nSteps Per Item:")
    print(f"  - Mean: {np.mean(steps_np):.2f}")
    print(f"  - Median: {np.median(steps_np):.2f}")
    print(f"  - Std Dev: {np.std(steps_np):.2f}")
    print(f"  - Range: [{np.min(steps_np)}, {np.max(steps_np)}]")
    
    # Collect all labels for detailed stats
    all_labels = [label for item in data for label in item['labels']]
    if not all_labels:
        print("\nDataset has no labels to analyze.")
        return
    
    labels_np = np.array(all_labels)
    p25, p75 = np.percentile(labels_np, [25, 75])
    
    print(f"\nLabel Values (per step):")
    print(f"  - Mean: {np.mean(labels_np):.4f}")
    print(f"  - Median: {np.median(labels_np):.4f}")
    print(f"  - Std Dev: {np.std(labels_np):.4f}")
    print(f"  - Range: [{np.min(labels_np):.4f}, {np.max(labels_np):.4f}]")
    print(f"  - IQR (25th-75th percentile): [{p25:.4f}, {p75:.4f}]")
    
    # Per-item average label stats
    avg_labels_per_item = [np.mean(item['labels']) if item['labels'] else 0 for item in data]
    avg_labels_np = np.array(avg_labels_per_item)
    
    print(f"\nAverage Label Value (per item):")
    print(f"  - Mean: {np.mean(avg_labels_np):.4f}")
    print(f"  - Median: {np.median(avg_labels_np):.4f}")
    print(f"  - Std Dev: {np.std(avg_labels_np):.4f}")
    print(f"  - Range: [{np.min(avg_labels_np):.4f}, {np.max(avg_labels_np):.4f}]")


def split_dataset_without_contamination(
    formatted_data: List[Dict[str, Any]], 
    train_ratio: float = 0.85
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split the dataset ensuring no prompt contamination between train and test sets.
    """
    print("\n--- Splitting Dataset (No Contamination) ---")
    
    # Group items by prompt to avoid contamination
    prompt_groups = defaultdict(list)
    for item in formatted_data:
        prompt_groups[item['prompt']].append(item)
    
    print(f"Splitting {len(formatted_data)} items from {len(prompt_groups)} unique prompts.")
    
    # Get unique prompts and shuffle them
    unique_prompts = list(prompt_groups.keys())
    random.seed(42) # for reproducibility
    random.shuffle(unique_prompts)
    
    # Split prompts based on the specified ratio
    split_point = int(len(unique_prompts) * train_ratio)
    train_prompts = set(unique_prompts[:split_point])
    
    # Create new train and test sets
    new_train, new_test = [], []
    for prompt, items in prompt_groups.items():
        if prompt in train_prompts:
            new_train.extend(items)
        else:
            new_test.extend(items)
    
    print(f"Split results:")
    print(f"  - Train: {len(new_train)} items ({len(train_prompts)} prompts)")
    print(f"  - Test:  {len(new_test)} items ({len(prompt_groups) - len(train_prompts)} prompts)")
    print(f"  - Train percentage: {len(new_train) / len(formatted_data) * 100:.1f}%")
    
    # Verify no contamination
    train_prompts_check = set(item['prompt'] for item in new_train)
    test_prompts_check = set(item['prompt'] for item in new_test)
    contamination = train_prompts_check.intersection(test_prompts_check)
    if contamination:
        print(f"WARNING: Found {len(contamination)} overlapping prompts between train and test sets!")
    else:
        print("Contamination check passed: No overlapping prompts found.")
        
    return new_train, new_test


if __name__ == "__main__":
    
    # Wait for file operations to complete
    print("Waiting for file operations to complete...")
    time.sleep(30)
    
    # Step 1: Convert dataset format with CWE-based continuous labels
    print("\n--- Step 1: Converting Dataset Format ---")
    formatted_data = convert_dataset("data/math_completions.jsonl")
    
    # Step 2: Remove outliers with excessively long reasoning chains
    cleaned_data = remove_outliers_by_steps(formatted_data)
    
    # Step 3: Print detailed statistics on the cleaned dataset
    print_label_statistics(cleaned_data, "Cleaned Dataset")
    
    # Wait before splitting
    print("\nWaiting before dataset split...")
    time.sleep(10)
    
    # Step 4: Split the cleaned data without contamination
    train_data, test_data = split_dataset_without_contamination(
        cleaned_data, 
        train_ratio=0.85
    )
    
    # Wait before saving
    print("\nWaiting before saving datasets...")
    time.sleep(2)
    
    # Step 5: Save the datasets
    print(f"\n--- Step 5: Saving Datasets ---")
    save_dataset(train_data, "data/math_cwe_train.jsonl")
    save_dataset(test_data, "data/math_cwe_test.jsonl")

    print(f"Training set saved to: data/math_cwe_train.jsonl")
    print(f"Test set saved to: data/math_cwe_test.jsonl")
    print("\nPipeline complete! ✅")