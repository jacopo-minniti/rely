#!/usr/bin/env python3
"""
Convert dataset script that:
1. Formats the dataset from completer output to process-reward model format
2. Splits the data to avoid contamination between train and test sets
"""

import argparse
import math
import random
import re
from collections import defaultdict, Counter
from typing import List, Dict, Any

from rely.utils import load_dataset, save_dataset, extract_final_answer, normalize_answer


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


def format_dataset_item(item: Dict[str, Any], evaluation_method: str = "hard_label") -> Dict[str, Any]:
    """
    Convert a single item from completer format to process-reward format.
    Returns a single formatted item with full CoT and labels for each step.
    
    Args:
        item: The dataset item to format
        evaluation_method: Either "hard_label" or "entropy"
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
                
                # Label is 1 if entropy > 0.1, otherwise 0
                entropy_label = 1 if entropy_score > 0.1 else 0
                
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


def convert_dataset(input_file: str, evaluation_method: str = "hard_label") -> List[Dict[str, Any]]:
    """
    Convert the entire dataset from completer format to process-reward format.
    
    Args:
        input_file: Path to the input JSONL file
        evaluation_method: Either "hard_label" or "entropy"
    """
    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    
    if not data:
        raise ValueError(f"Could not load data from {input_file}")
    
    print(f"Loaded {len(data)} items from the dataset")
    print(f"Using evaluation method: {evaluation_method}")
    
    all_formatted_items = []
    
    for item in data:
        formatted_item = format_dataset_item(item, evaluation_method)
        all_formatted_items.append(formatted_item)
    
    print(f"Converted to {len(all_formatted_items)} formatted items")
    return all_formatted_items


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
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    
    # Generate output file names
    base_name = args.output_file
    if base_name.endswith('.jsonl'):
        base_name = base_name[:-6]  # Remove .jsonl extension
    
    train_output = f"{base_name}_train.jsonl"
    test_output = f"{base_name}_test.jsonl"
    
    try:
        # Step 1: Convert dataset format
        formatted_data = convert_dataset(args.input_file, args.evaluation)
        
        # Step 2: Split without contamination
        train_data, test_data = split_dataset_without_contamination(
            formatted_data, 
            train_ratio=args.train_ratio
        )
        
        # Step 3: Save the datasets
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
    exit(main())
