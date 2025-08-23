import argparse
import math
import random
from collections import defaultdict, Counter
from typing import List, Dict, Any

from rely.utils import load_dataset, save_dataset, extract_final_answer, normalize_answer, merge

def extract_question_from_prompt(prompt: str) -> str:
    """
    Extract the raw question from a formatted prompt.
    This removes the system prompt and assistant formatting, keeping only the question.
    """
    return prompt.strip()


def is_completion_correct(completion: str, correct_answer: str) -> bool:
    """
    Check if a completion leads to the correct final answer.
    Uses extract_answer to get the latest \\boxed{} content.
    """
    if not correct_answer:
        return False
        
    extracted = extract_final_answer(completion)
    if not extracted:
        return False
        
    # Clean and compare answers (case insensitive, stripped)
    return normalize_answer(extracted) == normalize_answer(correct_answer)


def calculate_entropy(completions: List[str]) -> float:
    """
    Calculate the entropy of completion final answers.
    Returns the natural logarithm (e-based) entropy of the distribution.
    """
    if not completions:
        return 0.0
    
    # Extract final answers from all completions
    final_answers = []
    for completion in completions:
        extracted = extract_final_answer(completion)
        if extracted:
            final_answers.append(normalize_answer(extracted))
        else:
            final_answers.append("")  # Use empty string for no answer
    
    # If no answers were extracted, return 0 entropy
    if not final_answers:
        return 0.0
    
    # Count occurrences of each unique answer
    answer_counts = Counter(final_answers)
    total_answers = len(final_answers)
    
    # Calculate entropy using natural logarithm (e-based)
    entropy = 0.0
    for count in answer_counts.values():
        probability = count / total_answers
        if probability > 0:  # Avoid log(0)
            entropy -= probability * math.log(probability)
    
    return entropy


def format_dataset_item(item: Dict[str, Any], evaluation_method: str = "hard_label", entropy_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Convert a single item from completer format to process-reward format.
    Returns a single formatted item with full CoT and labels for each step.
    
    Args:
        item: The dataset item to format
        evaluation_method: Either "hard_label" or "entropy"
        entropy_threshold: Threshold for entropy-based labeling (default: 0.1)
    """
    original_item = item["original_item"]
    samples = item["samples"]
    
    # Extract the clean question
    question = original_item.get("question", "")
    clean_question = extract_question_from_prompt(question)
    
    # Get the correct answer from the original item (needed for hard_label evaluation)
    correct_answer = original_item.get("solution", "")
    
    # Get the full CoT from the original item
    full_cot = original_item.get("attempt", "")
    if not full_cot:
        # If no attempt field, try to get the longest cut_cot from samples
        full_cot = max((sample["cut_cot"] for sample in samples), key=len, default="")
    
    # Split the full CoT into steps
    cot_steps = full_cot.split("\n\n") if full_cot else []
    
    # Create a mapping from step index to evaluation result
    step_labels = {}
    
    for sample in samples:
        cut_cot = sample["cut_cot"]
        completions = sample["completions"]
        
        # Find which step this sample corresponds to
        if cut_cot:
            cut_steps = cut_cot.split("\n\n")
            step_index = len(cut_steps) - 1  # Index of the last step in this cut
            
            if evaluation_method == "hard_label":
                # Check if ANY of the completions leads to the correct answer
                any_correct = any(is_completion_correct(comp, correct_answer) for comp in completions)
                
                # Store the result for this step (use OR logic if multiple samples for same step)
                if step_index not in step_labels:
                    step_labels[step_index] = any_correct
                else:
                    step_labels[step_index] = step_labels[step_index] or any_correct
                    
            elif evaluation_method == "entropy":
                # Calculate entropy of completion final answers
                entropy_score = calculate_entropy(completions)
                
                # Label is 1 if entropy > entropy_threshold, otherwise 0
                entropy_label = True if entropy_score > entropy_threshold else False
                
                # Store the result for this step (use OR logic if multiple samples for same step)
                if step_index not in step_labels:
                    step_labels[step_index] = entropy_label
                else:
                    step_labels[step_index] = max(step_labels[step_index], entropy_label)
    
    # Create labels list matching the number of steps
    labels = []
    for i in range(len(cot_steps)):
        labels.append(step_labels.get(i, 0))  # Default to 0 if no sample for this step
    
    formatted_item = {
        "prompt": clean_question,
        "completions": cot_steps,  # List of reasoning steps
        "labels": labels  # List of 0/1s for each step
    }
    
    return formatted_item


def convert_dataset(input_file: str, evaluation_method: str = "hard_label", entropy_threshold: float = 0.1) -> List[Dict[str, Any]]:
    """
    Convert the entire dataset from completer format to process-reward format.
    
    Args:
        input_file: Path to the input JSONL file
        evaluation_method: Either "hard_label" or "entropy"
        entropy_threshold: Threshold for entropy-based labeling (default: 0.1)
    """
    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    
    if not data:
        raise ValueError(f"Could not load data from {input_file}")
    
    print(f"Loaded {len(data)} items from the dataset")
    print(f"Using evaluation method: {evaluation_method}")
    if evaluation_method == "entropy":
        print(f"Entropy threshold: {entropy_threshold}")
    
    all_formatted_items = []
    
    for item in data:
        formatted_item = format_dataset_item(item, evaluation_method, entropy_threshold)
        all_formatted_items.append(formatted_item)
    
    print(f"Converted to {len(all_formatted_items)} formatted items")
    return all_formatted_items


def print_label_statistics(data: List[Dict[str, Any]], dataset_name: str = "Dataset") -> None:
    """
    Print statistics about the labels in the dataset.
    
    Args:
        data: List of formatted dataset items
        dataset_name: Name of the dataset for display purposes
    """
    if not data:
        print(f"{dataset_name} is empty")
        return
    
    total_items = len(data)
    total_steps = sum(len(item['labels']) for item in data)
    
    # Count labels
    true_labels = sum(sum(item['labels']) for item in data)
    false_labels = total_steps - true_labels
    
    # Calculate percentages
    true_percentage = (true_labels / total_steps * 100) if total_steps > 0 else 0
    false_percentage = (false_labels / total_steps * 100) if total_steps > 0 else 0
    
    # Calculate average steps per item
    avg_steps_per_item = total_steps / total_items if total_items > 0 else 0
    
    print(f"\n{dataset_name} Label Statistics:")
    print(f"  Total items: {total_items}")
    print(f"  Total steps: {total_steps}")
    print(f"  Average steps per item: {avg_steps_per_item:.2f}")
    print(f"  True labels (1): {true_labels} ({true_percentage:.1f}%)")
    print(f"  False labels (0): {false_labels} ({false_percentage:.1f}%)")
    
    # Additional statistics: distribution of steps per item
    steps_per_item = [len(item['labels']) for item in data]
    min_steps = min(steps_per_item) if steps_per_item else 0
    max_steps = max(steps_per_item) if steps_per_item else 0
    
    print(f"  Steps per item - Min: {min_steps}, Max: {max_steps}")
    
    # Distribution of true labels per item
    true_per_item = [sum(item['labels']) for item in data]
    avg_true_per_item = sum(true_per_item) / len(true_per_item) if true_per_item else 0
    min_true_per_item = min(true_per_item) if true_per_item else 0
    max_true_per_item = max(true_per_item) if true_per_item else 0
    
    print(f"  True labels per item - Avg: {avg_true_per_item:.2f}, Min: {min_true_per_item}, Max: {max_true_per_item}")


def balance_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Balance the dataset by removing samples from the majority class at the step level.
    """
    print("Balancing dataset...")
    
    # Count total labels
    total_true = sum(sum(item['labels']) for item in data)
    total_false = sum(len(item['labels']) - sum(item['labels']) for item in data)
    total_steps = total_true + total_false
    
    print(f"Before balancing: True={total_true}, False={total_false}, Total={total_steps}")
    
    if total_true == total_false:
        print("Dataset is already balanced.")
        return data
    
    # Determine majority and minority counts
    majority_count = max(total_true, total_false)
    minority_count = min(total_true, total_false)
    majority_is_true = total_true > total_false
    
    # Calculate how many majority samples to remove
    samples_to_remove = majority_count - minority_count
    
    print(f"Majority class ({'True' if majority_is_true else 'False'}): {majority_count}")
    print(f"Minority class ({'False' if majority_is_true else 'True'}): {minority_count}")
    print(f"Need to remove {samples_to_remove} majority samples")
    
    # Create a list of all step indices with their labels
    step_indices = []
    for item_idx, item in enumerate(data):
        for step_idx, label in enumerate(item['labels']):
            step_indices.append((item_idx, step_idx, bool(label)))
    
    # Filter to majority class steps and shuffle
    majority_steps = [(item_idx, step_idx) for item_idx, step_idx, label in step_indices 
                      if label == majority_is_true]
    random.shuffle(majority_steps)
    
    # Select steps to remove
    steps_to_remove = set(majority_steps[:samples_to_remove])
    
    # Create new balanced dataset
    balanced_data = []
    for item_idx, item in enumerate(data):
        new_completions = []
        new_labels = []
        
        for step_idx, (completion, label) in enumerate(zip(item['completions'], item['labels'])):
            if (item_idx, step_idx) not in steps_to_remove:
                new_completions.append(completion)
                new_labels.append(label)
        
        # Only keep items that still have steps
        if new_completions:
            balanced_item = {
                'prompt': item['prompt'],
                'completions': new_completions,
                'labels': new_labels
            }
            balanced_data.append(balanced_item)
    
    # Verify balancing
    new_total_true = sum(sum(item['labels']) for item in balanced_data)
    new_total_false = sum(len(item['labels']) - sum(item['labels']) for item in balanced_data)
    new_total_steps = new_total_true + new_total_false
    
    print(f"After balancing: True={new_total_true}, False={new_total_false}, Total={new_total_steps}")
    print(f"Removed {total_steps - new_total_steps} steps from the majority class")
    print(f"Removed {len(data) - len(balanced_data)} items with no remaining steps")
    
    return balanced_data


def split_dataset_without_contamination(
    formatted_data: List[Dict[str, Any]], 
    train_ratio: float = 0.85
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split the dataset ensuring no prompt contamination between train and test sets.
    """
    print("Splitting dataset without contamination...")
    
    # Group items by prompt to avoid contamination
    prompt_groups = defaultdict(list)
    for item in formatted_data:
        prompt_groups[item['prompt']].append(item)
    
    print(f"Total items: {len(formatted_data)}")
    print(f"Unique prompts: {len(prompt_groups)}")
    print(f"Average items per prompt: {len(formatted_data) / len(prompt_groups):.2f}")
    
    # Get unique prompts and shuffle them
    unique_prompts = list(prompt_groups.keys())
    random.shuffle(unique_prompts)
    
    # Split prompts based on the specified ratio
    split_point = int(len(unique_prompts) * train_ratio)
    train_prompts = set(unique_prompts[:split_point])
    test_prompts = set(unique_prompts[split_point:])
    
    # Create new train and test sets
    new_train = []
    new_test = []
    
    for prompt, items in prompt_groups.items():
        if prompt in train_prompts:
            new_train.extend(items)
        else:
            new_test.extend(items)
    
    print(f"\nSplit results:")
    print(f"Train prompts: {len(train_prompts)}")
    print(f"Test prompts: {len(test_prompts)}")
    print(f"Train items: {len(new_train)}")
    print(f"Test items: {len(new_test)}")
    print(f"Train percentage: {len(new_train) / len(formatted_data) * 100:.1f}%")
    print(f"Test percentage: {len(new_test) / len(formatted_data) * 100:.1f}%")
    
    # Verify no contamination
    train_prompts_check = set(item['prompt'] for item in new_train)
    test_prompts_check = set(item['prompt'] for item in new_test)
    contamination = train_prompts_check.intersection(test_prompts_check)
    print(f"Data contamination check: {len(contamination)} overlapping prompts")
    
    if contamination:
        print("WARNING: Found contamination between train and test sets!")
        for prompt in list(contamination)[:5]:  # Show first 5 contaminated prompts
            print(f"  - {prompt[:100]}...")
    
    return new_train, new_test


def main():
    parser = argparse.ArgumentParser(
        description="Convert completer dataset to process-reward format and split without contamination"
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file from the completer"
    )
    parser.add_argument(
        "output_file",
        help="Base name for output files (train and test will be appended)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Ratio of data to use for training (default: 0.85)"
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        choices=["hard_label", "entropy"],
        default="hard_label",
        help="Evaluation method: 'hard_label' checks final answer correctness, 'entropy' uses answer distribution entropy (default: hard_label)"
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.1,
        help="Threshold for entropy-based labeling. Labels are True if entropy > threshold (default: 0.1)"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance the two classes in the training set only by removing samples from the majority class (test set keeps original distribution)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    
    if args.entropy_threshold < 0:
        raise ValueError("entropy_threshold must be non-negative")
    
    # Generate output file names
    base_name = args.output_file
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]  # Remove .jsonl extension
    
    train_output = f"{base_name}_train.jsonl"
    test_output = f"{base_name}_test.jsonl"
    
    try:
        # Step 1: Convert dataset format
        formatted_data = convert_dataset(args.input_file, args.evaluation, args.entropy_threshold)
        
        # Print overall dataset statistics
        print_label_statistics(formatted_data, "Overall Dataset")
        
        # Step 2: Split without contamination
        train_data, test_data = split_dataset_without_contamination(
            formatted_data, 
            train_ratio=args.train_ratio
        )
        
        # Step 3: Balance only the training data if requested
        if args.balance:
            train_data = balance_dataset(train_data)
            print_label_statistics(train_data, "Balanced Training Set")
        
        # Print statistics for train and test sets
        print_label_statistics(train_data, "Training Set")
        print_label_statistics(test_data, "Test Set (Original Distribution)")
        
        # Step 4: Save the datasets
        print(f"\nSaving datasets...")
        save_dataset(train_data, train_output)
        save_dataset(test_data, test_output)
        
        print(f"Training set saved to: {train_output}")
        print(f"Test set saved to: {test_output}")
        print("Conversion complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    
    # python test.py data/math_completions.jsonl math_dataset.jsonl --evaluation entropy --entropy-threshold 0.01 --balance
    exit(main())