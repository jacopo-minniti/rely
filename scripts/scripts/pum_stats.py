import numpy as np
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_dataset_statistics():
    """Analyze statistics of the MATH-PUM dataset"""
    
    print("Loading dataset...")
    dataset = load_dataset("jacopo-minniti/MATH-PUM-qwen2.5-1.5B", "pp-1", split="test")
    
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Statistics to collect
    all_true_samples = 0  # Samples where all labels are True
    all_false_samples = 0  # Samples where all labels are False
    mixed_samples = 0  # Samples with both True and False labels
    
    label_lengths = []  # Number of steps per sample
    total_true_labels = 0
    total_false_labels = 0
    
    true_ratio_per_sample = []  # Ratio of True labels in each sample
    
    print("\nAnalyzing samples...")
    for sample in tqdm(dataset):
        labels = sample['labels']
        label_lengths.append(len(labels))
        
        # Count True and False labels
        true_count = sum(labels)
        false_count = len(labels) - true_count
        
        total_true_labels += true_count
        total_false_labels += false_count
        
        # Calculate ratio of True labels
        true_ratio = true_count / len(labels) if len(labels) > 0 else 0
        true_ratio_per_sample.append(true_ratio)
        
        # Categorize samples
        if all(labels):
            all_true_samples += 1
        elif not any(labels):
            all_false_samples += 1
        else:
            mixed_samples += 1
    
    # Calculate statistics
    avg_steps = np.mean(label_lengths)
    median_steps = np.median(label_lengths)
    min_steps = min(label_lengths)
    max_steps = max(label_lengths)
    
    avg_true_ratio = np.mean(true_ratio_per_sample)
    median_true_ratio = np.median(true_ratio_per_sample)
    
    # Print results
    print(f"\n{'='*60}")
    print("DATASET STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nSample Distribution by Label Pattern:")
    print(f"  All True samples:  {all_true_samples:6d} ({all_true_samples/len(dataset)*100:.2f}%)")
    print(f"  All False samples: {all_false_samples:6d} ({all_false_samples/len(dataset)*100:.2f}%)")
    print(f"  Mixed samples:     {mixed_samples:6d} ({mixed_samples/len(dataset)*100:.2f}%)")
    print(f"  Total:             {len(dataset):6d}")
    
    print(f"\nSteps per Sample:")
    print(f"  Average:  {avg_steps:.2f}")
    print(f"  Median:   {median_steps:.1f}")
    print(f"  Min:      {min_steps}")
    print(f"  Max:      {max_steps}")
    
    print(f"\nOverall Label Distribution:")
    print(f"  Total True labels:  {total_true_labels:8d} ({total_true_labels/(total_true_labels+total_false_labels)*100:.2f}%)")
    print(f"  Total False labels: {total_false_labels:8d} ({total_false_labels/(total_true_labels+total_false_labels)*100:.2f}%)")
    print(f"  Total labels:       {total_true_labels+total_false_labels:8d}")
    
    print(f"\nTrue Label Ratio per Sample:")
    print(f"  Average ratio:  {avg_true_ratio:.3f}")
    print(f"  Median ratio:   {median_true_ratio:.3f}")
    
    # Additional insights
    print(f"\n{'='*60}")
    print("INSIGHTS")
    print(f"{'='*60}")
    
    # Samples with extreme ratios
    very_high_true_ratio = sum(1 for ratio in true_ratio_per_sample if ratio >= 0.9)
    very_low_true_ratio = sum(1 for ratio in true_ratio_per_sample if ratio <= 0.1)
    balanced_samples = sum(1 for ratio in true_ratio_per_sample if 0.4 <= ratio <= 0.6)
    
    print(f"\nSamples by True Label Ratio:")
    print(f"  Very high (≥90% True):  {very_high_true_ratio:6d} ({very_high_true_ratio/len(dataset)*100:.2f}%)")
    print(f"  Very low (≤10% True):   {very_low_true_ratio:6d} ({very_low_true_ratio/len(dataset)*100:.2f}%)")
    print(f"  Balanced (40-60% True): {balanced_samples:6d} ({balanced_samples/len(dataset)*100:.2f}%)")
    
    # Step length distribution
    length_counter = Counter(label_lengths)
    print(f"\nMost Common Step Counts:")
    for length, count in length_counter.most_common(10):
        print(f"  {length:2d} steps: {count:6d} samples ({count/len(dataset)*100:.2f}%)")
    
    # Check for potential issues
    print(f"\n{'='*60}")
    print("POTENTIAL ISSUES")
    print(f"{'='*60}")
    
    if all_true_samples > len(dataset) * 0.1:
        print(f"⚠️  High number of all-True samples ({all_true_samples/len(dataset)*100:.1f}%)")
    
    if all_false_samples > len(dataset) * 0.1:
        print(f"⚠️  High number of all-False samples ({all_false_samples/len(dataset)*100:.1f}%)")
    
    if abs(avg_true_ratio - 0.5) > 0.2:
        print(f"⚠️  Dataset is imbalanced (avg True ratio: {avg_true_ratio:.3f})")
    
    if max_steps > 3 * avg_steps:
        print(f"⚠️  Some samples are very long ({max_steps} vs avg {avg_steps:.1f})")
    
    # Create visualizations
    create_visualizations(label_lengths, true_ratio_per_sample, length_counter)
    
    return {
        'total_samples': len(dataset),
        'all_true_samples': all_true_samples,
        'all_false_samples': all_false_samples,
        'mixed_samples': mixed_samples,
        'avg_steps': avg_steps,
        'median_steps': median_steps,
        'min_steps': min_steps,
        'max_steps': max_steps,
        'total_true_labels': total_true_labels,
        'total_false_labels': total_false_labels,
        'avg_true_ratio': avg_true_ratio,
        'median_true_ratio': median_true_ratio,
        'label_lengths': label_lengths,
        'true_ratio_per_sample': true_ratio_per_sample
    }

def create_visualizations(label_lengths, true_ratio_per_sample, length_counter):
    """Create visualization plots for the dataset statistics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Distribution of steps per sample
    axes[0, 0].hist(label_lengths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.mean(label_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(label_lengths):.1f}')
    axes[0, 0].axvline(np.median(label_lengths), color='orange', linestyle='--', label=f'Median: {np.median(label_lengths):.1f}')
    axes[0, 0].set_xlabel('Number of Steps per Sample')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Steps per Sample')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution of True label ratios
    axes[0, 1].hist(true_ratio_per_sample, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(np.mean(true_ratio_per_sample), color='red', linestyle='--', label=f'Mean: {np.mean(true_ratio_per_sample):.3f}')
    axes[0, 1].axvline(np.median(true_ratio_per_sample), color='orange', linestyle='--', label=f'Median: {np.median(true_ratio_per_sample):.3f}')
    axes[0, 1].set_xlabel('Ratio of True Labels per Sample')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of True Label Ratios')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Most common step counts (bar plot)
    common_lengths = dict(length_counter.most_common(15))
    axes[1, 0].bar(common_lengths.keys(), common_lengths.values(), alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Number of Steps')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Most Common Step Counts')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: True ratio vs number of steps (scatter plot)
    axes[1, 1].scatter(label_lengths, true_ratio_per_sample, alpha=0.5, s=20, color='purple')
    axes[1, 1].set_xlabel('Number of Steps')
    axes[1, 1].set_ylabel('Ratio of True Labels')
    axes[1, 1].set_title('True Ratio vs Number of Steps')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/rely/assets/figures/dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualizations saved to: /workspace/rely/assets/figures/dataset_statistics.png")
    plt.show()

if __name__ == "__main__":
    stats = analyze_dataset_statistics()
