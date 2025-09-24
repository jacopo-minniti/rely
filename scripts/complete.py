import math
import random
import time
from collections import defaultdict, Counter
from typing import List, Dict, Any

from rely.utils import load_dataset, save_dataset, extract_final_answer, normalize_answer
from rely.generate import Completer, CompleterConfig


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


def format_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single item from completer format to process-reward format.
    Returns a single formatted item with full CoT and raw entropy values for each step.
    
    Args:
        item: The dataset item to format
    """
    original_item = item["original_item"]
    samples = item["samples"]

    # Extract the question
    question = original_item.get("question", "")
    
    # Get the full CoT from the original item
    full_cot = original_item.get("attempt", "")
    if not full_cot:
        # If no attempt field, try to get the longest cut_cot from samples
        full_cot = max((sample["cut_cot"] for sample in samples), key=len, default="")
    
    # Split the full CoT into steps
    cot_steps = full_cot.split("\n\n") if full_cot else []
    
    # Create a mapping from step index to entropy value
    step_entropies = {}
    
    for sample in samples:
        cut_cot = sample["cut_cot"]
        completions = sample["completions"]
        
        # Find which step this sample corresponds to
        if cut_cot:
            cut_steps = cut_cot.split("\n\n")
            step_index = len(cut_steps) - 1  # Index of the last step in this cut
            
            # Calculate entropy of completion final answers
            entropy_score = calculate_entropy(completions)
            
            # Store the entropy for this step (use maximum if multiple samples for same step)
            if step_index not in step_entropies:
                step_entropies[step_index] = entropy_score
            else:
                step_entropies[step_index] = max(step_entropies[step_index], entropy_score)
    
    # Create entropy values list matching the number of steps
    entropy_values = []
    for i in range(len(cot_steps)):
        entropy_values.append(step_entropies.get(i, 0.0))  # Default to 0.0 if no sample for this step
    
    formatted_item = {
        "prompt": question.strip(),
        "completions": cot_steps,  # List of reasoning steps
        "entropy_values": entropy_values  # List of raw entropy values for each step
    }
    
    return formatted_item


def convert_dataset(input_file: str) -> List[Dict[str, Any]]:
    """
    Convert the entire dataset from completer format to process-reward format.
    Normalizes entropy values to [0, 1] range across the entire dataset.
    
    Args:
        input_file: Path to the input JSONL file
    """
    print(f"Loading dataset from {input_file}...")
    data = load_dataset(input_file)
    
    if not data:
        raise ValueError(f"Could not load data from {input_file}")
    
    print(f"Loaded {len(data)} items from the dataset")
    print("Using entropy-based evaluation with continuous values")
    
    # First pass: convert all items and collect all entropy values
    all_formatted_items = []
    all_entropy_values = []
    
    for item in data:
        formatted_item = format_dataset_item(item)
        all_formatted_items.append(formatted_item)
        all_entropy_values.extend(formatted_item["entropy_values"])
    
    # Find min and max entropy values for normalization
    if all_entropy_values:
        min_entropy = min(all_entropy_values)
        max_entropy = max(all_entropy_values)
        entropy_range = max_entropy - min_entropy
        
        print(f"Entropy statistics: min={min_entropy:.4f}, max={max_entropy:.4f}, range={entropy_range:.4f}")
        
        # Normalize entropy values to [0, 1] range
        if entropy_range > 0:
            for item in all_formatted_items:
                normalized_entropies = []
                for entropy_val in item["entropy_values"]:
                    normalized_val = (entropy_val - min_entropy) / entropy_range
                    normalized_entropies.append(normalized_val)
                item["labels"] = normalized_entropies  # Use "labels" for compatibility with existing code
                del item["entropy_values"]  # Remove raw entropy values
        else:
            # If all entropy values are the same, set all labels to 0.5
            for item in all_formatted_items:
                item["labels"] = [0.5] * len(item["entropy_values"])
                del item["entropy_values"]
    else:
        print("Warning: No entropy values found in the dataset")
        for item in all_formatted_items:
            item["labels"] = [0.0] * len(item.get("entropy_values", []))
            if "entropy_values" in item:
                del item["entropy_values"]
    
    print(f"Converted to {len(all_formatted_items)} formatted items with normalized entropy labels")
    return all_formatted_items


def print_label_statistics(data: List[Dict[str, Any]], dataset_name: str = "Dataset") -> None:
    """
    Print statistics about the continuous entropy labels in the dataset.
    
    Args:
        data: List of formatted dataset items
        dataset_name: Name of the dataset for display purposes
    """
    if not data:
        print(f"{dataset_name} is empty")
        return
    
    total_items = len(data)
    total_steps = sum(len(item['labels']) for item in data)
    
    # Collect all label values for statistics
    all_labels = []
    for item in data:
        all_labels.extend(item['labels'])
    
    if not all_labels:
        print(f"{dataset_name} has no labels")
        return
    
    # Calculate statistics for continuous values
    min_label = min(all_labels)
    max_label = max(all_labels)
    avg_label = sum(all_labels) / len(all_labels)
    
    # Calculate average steps per item
    avg_steps_per_item = total_steps / total_items if total_items > 0 else 0
    
    print(f"\n{dataset_name} Label Statistics:")
    print(f"  Total items: {total_items}")
    print(f"  Total steps: {total_steps}")
    print(f"  Average steps per item: {avg_steps_per_item:.2f}")
    print(f"  Label range: [{min_label:.4f}, {max_label:.4f}]")
    print(f"  Average label value: {avg_label:.4f}")
    
    # Additional statistics: distribution of steps per item
    steps_per_item = [len(item['labels']) for item in data]
    min_steps = min(steps_per_item) if steps_per_item else 0
    max_steps = max(steps_per_item) if steps_per_item else 0
    
    print(f"  Steps per item - Min: {min_steps}, Max: {max_steps}")
    
    # Distribution of average labels per item
    avg_per_item = [sum(item['labels']) / len(item['labels']) if item['labels'] else 0 for item in data]
    avg_avg_per_item = sum(avg_per_item) / len(avg_per_item) if avg_per_item else 0
    min_avg_per_item = min(avg_per_item) if avg_per_item else 0
    max_avg_per_item = max(avg_per_item) if avg_per_item else 0
    
    print(f"  Average label per item - Mean: {avg_avg_per_item:.4f}, Min: {min_avg_per_item:.4f}, Max: {max_avg_per_item:.4f}")


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
    """
    Main function that runs the complete pipeline:
    1. Generate completions using the Completer
    2. Convert the dataset format with entropy-based continuous labels
    3. Split into train/test sets without contamination
    4. Save the results
    """    


    input_dataset = "data/math_generations_v1.jsonl"  # Input dataset path
    output_base = "math_completions_100_v1.jsonl"  # Base name for output files
    completions_file = "math_completions_100_v1.jsonl"  # Intermediate completions file
    train_ratio = 0.85
    
    # Completer configuration - adjust these settings as needed
    completer_config = CompleterConfig(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        # tp_size=1,
        dp_size=20,
        # max_num_seqs=128,
        forking_strategy="newline",
        completion_type="short",
        dataset=input_dataset,
        question_field="question"
    )
    
    # Generate output file names
    train_output = f"{output_base}_train.jsonl"
    test_output = f"{output_base}_test.jsonl"
    
    try:
        # Step 1: Generate completions using the Completer
        print("Step 1: Generating completions...")
        completer = Completer(completer_config)
        completer.generate(
            output_file=completions_file,
            n_completions_per_item=50,
            max_new_tokens=20,
            temperature=1.0,
            cot_percentage=1.0
        )
        print(f"Completions saved to: {completions_file}")
        
        # Wait for file operations to complete
        print("Waiting for file operations to complete...")
        time.sleep(30)
        
        # Step 2: Convert dataset format with entropy-based continuous labels
        print("\nStep 2: Converting dataset format...")
        formatted_data = convert_dataset(completions_file)
        
        # Print overall dataset statistics
        print_label_statistics(formatted_data, "Overall Dataset")
        
        # Wait before splitting
        print("Waiting before dataset split...")
        time.sleep(10)
        
        # Step 3: Split without contamination
        print("\nStep 3: Splitting dataset...")
        train_data, test_data = split_dataset_without_contamination(
            formatted_data, 
            train_ratio=train_ratio
        )
        
        # Print statistics for train and test sets
        print_label_statistics(train_data, "Training Set")
        print_label_statistics(test_data, "Test Set")
        
        # Wait before saving
        print("Waiting before saving datasets...")
        time.sleep(2)
        
        # Step 4: Save the datasets
        print(f"\nStep 4: Saving datasets...")
        save_dataset(train_data, train_output)
        save_dataset(test_data, test_output)
        
        print(f"Training set saved to: {train_output}")
        print(f"Test set saved to: {test_output}")
        print("Pipeline complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

# def split_into_four(input_file: str, output_prefix: str = "math_generations_v"):
#     data = load_dataset(input_file)
#     n = len(data)
#     indices = list(range(n))
#     random.shuffle(indices)
#     part_size = n // 4

#     parts = [
#         [data[i] for i in indices[:part_size]],
#         [data[i] for i in indices[part_size:2*part_size]],
#         [data[i] for i in indices[2*part_size:3*part_size]],
#         [data[i] for i in indices[3*part_size:]]
#     ]

#     for idx, part in enumerate(parts):
#         out_file = f"{output_prefix}{idx+1}.jsonl"
#         save_dataset(part, out_file)
#         print(f"Saved {len(part)} items to {out_file}")

# split_into_four("math_generations.jsonl")