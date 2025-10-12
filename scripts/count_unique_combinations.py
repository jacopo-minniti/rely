from rely.utils import load_dataset, normalize_answer, extract_final_answer, save_dataset
from collections import Counter
import copy

# Load both datasets
data1 = load_dataset("data/math_completions.jsonl")
data2 = load_dataset("data/math_completions_v2.jsonl")

print(f"Dataset 1: {len(data1)} items")
print(f"Dataset 2: {len(data2)} items")

# Merge datasets by combining completions for identical CoTs
def merge_datasets(dataset1, dataset2):
    """Merge two datasets by combining completions for identical CoTs"""
    merged_data = []
    
    # Create a mapping from original item + cut_cot to samples for dataset2
    dataset2_map = {}
    for item in dataset2:
        item_key = str(item['original_item'])  # Use string representation as key
        if item_key not in dataset2_map:
            dataset2_map[item_key] = {}
        
        for sample in item['samples']:
            cot_key = sample['cut_cot']
            dataset2_map[item_key][cot_key] = sample
    
    # Process dataset1 and merge with dataset2 where possible
    for item1 in dataset1:
        item_key = str(item1['original_item'])
        merged_item = copy.deepcopy(item1)
        
        # Check if this item exists in dataset2
        if item_key in dataset2_map:
            item2_samples = dataset2_map[item_key]
            
            # Process each sample in the merged item
            for sample_idx, sample in enumerate(merged_item['samples']):
                cot_key = sample['cut_cot']
                
                # If matching CoT found in dataset2, merge completions
                if cot_key in item2_samples:
                    sample2 = item2_samples[cot_key]
                    # Combine completions from both datasets
                    sample['completions'].extend(sample2['completions'])
                    print(f"Merged CoT: N={len(sample['completions'])} (was {len(item1['samples'][sample_idx]['completions'])} + {len(sample2['completions'])})")
        
        merged_data.append(merged_item)
    
    # Add items from dataset2 that weren't in dataset1
    dataset1_keys = {str(item['original_item']) for item in dataset1}
    for item2 in dataset2:
        item_key = str(item2['original_item'])
        if item_key not in dataset1_keys:
            merged_data.append(copy.deepcopy(item2))
            print(f"Added new item from dataset2 with {len(item2['samples'])} samples")
    
    return merged_data

print("\nMerging datasets...")
data = merge_datasets(data1, data2)
print(f"Merged dataset: {len(data)} items")
save_dataset(data, "data/merged_math_completions.jsonl")

# Analyze the structure and count 0s and 1s for each step
print(f"Total items in dataset: {len(data)}")

# For tracking counts across all steps
all_step_counts = []  # List of (num_0s, num_1s) for each step

# Analyze each item
for item_idx, item in enumerate(data):
    print(f"\nItem {item_idx + 1}:")
    print(f"  Original item keys: {list(item['original_item'].keys())}")
    print(f"  Number of samples: {len(item['samples'])}")
    
    # Look at each sample (each represents a different step)
    for sample_idx, sample in enumerate(item['samples']):
        print(f"    Sample {sample_idx + 1}:")
        print(f"      Cut CoT length: {len(sample['cut_cot'])}")
        print(f"      Number of completions: {len(sample['completions'])}")
        
        # Count correct vs incorrect completions for this step
        completions = sample['completions']
        if len(completions) > 0:  # Process any number of completions
            # Get ground truth from original item
            ground_truth = item['original_item'].get('solution', '')
            normalized_ground_truth = normalize_answer(ground_truth)
            
            # Count correct (1) and incorrect (0) completions
            correct_count = 0
            incorrect_count = 0
            
            for completion in completions:
                extracted = extract_final_answer(completion)
                if extracted:
                    normalized_extracted = normalize_answer(extracted)
                    if normalized_extracted == normalized_ground_truth:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                else:
                    # If no answer is extracted, treat as incorrect
                    incorrect_count += 1
            
            print(f"        Step {sample_idx + 1}: {incorrect_count} incorrect, {correct_count} correct (N={len(completions)} completions)")
            all_step_counts.append((incorrect_count, correct_count))
        else:
            print(f"        Warning: No completions found for this step")
    
    # Only analyze first few items for initial inspection
    if item_idx >= 2:
        print("\n... (showing first 3 items for inspection)")
        break

# Count unique combinations of (incorrect, correct)
unique_counts = Counter(all_step_counts)
print(f"\nUnique (incorrect, correct) combinations across entire dataset:")
for (incorrect, correct), frequency in sorted(unique_counts.items()):
    print(f"  ({incorrect}, {correct}): appears {frequency} times")

print(f"\nTotal number of unique (incorrect, correct) combinations: {len(unique_counts)}")
print(f"Total steps analyzed: {len(all_step_counts)}")

# Now process the entire dataset for complete analysis
print("\n" + "="*60)
print("PROCESSING ENTIRE DATASET")
print("="*60)

all_counts_full = []
total_items = len(data)
processed_items = 0

for item_idx, item in enumerate(data):
    processed_items += 1
    if processed_items % 100 == 0:  # Progress indicator
        print(f"Progress: {processed_items}/{total_items} items processed")
    
    # Process all samples in this item
    for sample in item['samples']:
        completions = sample['completions']
        
        if len(completions) > 0:  # Process any number of completions
            # Get ground truth from original item
            ground_truth = item['original_item'].get('solution', '')
            normalized_ground_truth = normalize_answer(ground_truth)
            
            # Count correct (1) and incorrect (0) completions
            correct_count = 0
            incorrect_count = 0
            
            for completion in completions:
                extracted = extract_final_answer(completion)
                if extracted:
                    normalized_extracted = normalize_answer(extracted)
                    if normalized_extracted == normalized_ground_truth:
                        correct_count += 1
                    else:
                        incorrect_count += 1
                else:
                    # If no answer is extracted, treat as incorrect
                    incorrect_count += 1
            
            all_counts_full.append((incorrect_count, correct_count))

# Final analysis
unique_counts_full = Counter(all_counts_full)
print(f"\nFINAL RESULTS:")
print(f"Total steps analyzed across entire dataset: {len(all_counts_full)}")
print(f"Total unique (incorrect, correct) combinations: {len(unique_counts_full)}")

print(f"\nAll unique combinations and their frequencies:")
for (incorrect, correct), frequency in sorted(unique_counts_full.items()):
    print(f"  ({incorrect}, {correct}): appears {frequency} times")

# Additional statistics
if all_counts_full:
    incorrect_list = [incorrect for incorrect, correct in all_counts_full]
    correct_list = [correct for incorrect, correct in all_counts_full]

    print(f"\nStatistics:")
    print(f"  Incorrect - Min: {min(incorrect_list)}, Max: {max(incorrect_list)}, Avg: {sum(incorrect_list)/len(incorrect_list):.2f}")
    print(f"  Correct - Min: {min(correct_list)}, Max: {max(correct_list)}, Avg: {sum(correct_list)/len(correct_list):.2f}")
else:
    print(f"\nNo data found to analyze statistics.")
