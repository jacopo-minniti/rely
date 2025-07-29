import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import seaborn as sns
from collections import defaultdict
from .load import load_dataset
import re
from collections import Counter
import math


def entropy_from_completions(completions):
    """
    Calculates the entropy of the categorical distribution of the last uppercase letter in parentheses (e.g., (A))
    from a list of completion strings.
    Args:
        completions: List of strings.
    Returns:
        Entropy (float) of the distribution. Returns -1 if no valid letters are found or if 5+ invalid letters.
    """
    if not isinstance(completions, list) or len(completions) == 0:
        return -1
    # Regex to find the last (A), (B), ... in each string
    pattern = re.compile(r"\(([A-Z])\)")
    letters = []
    invalid_count = 0
    for c in completions:
        matches = pattern.findall(c)
        if matches:
            letters.append(matches[-1])
        else:
            invalid_count += 1
            # Return -1 if we have 5 or more invalid letters
            if invalid_count >= 5:
                return -1
    if not letters:
        return -1
    counts = Counter(letters)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy


def validate_entropy_field(data: List[Dict[str, Any]]) -> bool:
    """
    Validate that all items in the dataset have an entropy field.
    
    Args:
        data: List of dictionaries containing the data
        
    Returns:
        True if all items have entropy field, False otherwise
    """
    missing_entropy = [i for i, item in enumerate(data) if 'entropy' not in item]
    
    if missing_entropy:
        print(f"Warning: {len(missing_entropy)} items missing entropy field at indices: {missing_entropy[:10]}{'...' if len(missing_entropy) > 10 else ''}")
        return False
    
    return True


def basic_entropy_stats(data: List[Dict[str, Any]], epsilon: float = 0.1) -> Dict[str, Any]:
    """
    Compute basic statistics on entropy values.
    
    Args:
        data: List of dictionaries containing the data
        epsilon: Threshold for high entropy classification
        
    Returns:
        Dictionary containing various entropy statistics
    """
    if not validate_entropy_field(data):
        return {}
    
    entropies = [item["entropy"] for item in data]
    entropies = np.array(entropies)
    
    # Basic stats
    total = len(entropies)
    above_epsilon = entropies[entropies > epsilon]
    below_epsilon = entropies[entropies <= epsilon]
    
    stats = {
        'total_items': total,
        'above_epsilon': len(above_epsilon),
        'below_epsilon': len(below_epsilon),
        'pct_above_epsilon': 100.0 * len(above_epsilon) / total if total > 0 else 0,
        'pct_below_epsilon': 100.0 * len(below_epsilon) / total if total > 0 else 0,
        'all_entropy': {
            'min': float(np.min(entropies)),
            'max': float(np.max(entropies)),
            'mean': float(np.mean(entropies)),
            'median': float(np.median(entropies)),
            'std': float(np.std(entropies)),
            'q25': float(np.percentile(entropies, 25)),
            'q75': float(np.percentile(entropies, 75))
        }
    }
    
    if len(above_epsilon) > 0:
        stats['above_epsilon_stats'] = {
            'min': float(np.min(above_epsilon)),
            'max': float(np.max(above_epsilon)),
            'mean': float(np.mean(above_epsilon)),
            'median': float(np.median(above_epsilon)),
            'std': float(np.std(above_epsilon))
        }
    
    if len(below_epsilon) > 0:
        stats['below_epsilon_stats'] = {
            'min': float(np.min(below_epsilon)),
            'max': float(np.max(below_epsilon)),
            'mean': float(np.mean(below_epsilon)),
            'median': float(np.median(below_epsilon)),
            'std': float(np.std(below_epsilon))
        }
    
    return stats


def print_entropy_stats(data: List[Dict[str, Any]], epsilon: float = 0.1) -> None:
    """
    Print formatted entropy statistics.
    
    Args:
        data: List of dictionaries containing the data
        epsilon: Threshold for high entropy classification
    """
    stats = basic_entropy_stats(data, epsilon)
    
    if not stats:
        return
    
    print(f"=== Entropy Statistics (ε = {epsilon}) ===")
    print(f"Total items: {stats['total_items']}")
    print(f"Items with entropy > {epsilon}: {stats['above_epsilon']} ({stats['pct_above_epsilon']:.2f}%)")
    print(f"Items with entropy ≤ {epsilon}: {stats['below_epsilon']} ({stats['pct_below_epsilon']:.2f}%)")
    print()
    
    print("All entropy stats:")
    all_stats = stats['all_entropy']
    print(f"  Min: {all_stats['min']:.4f}")
    print(f"  Max: {all_stats['max']:.4f}")
    print(f"  Mean: {all_stats['mean']:.4f}")
    print(f"  Median: {all_stats['median']:.4f}")
    print(f"  Std: {all_stats['std']:.4f}")
    print(f"  Q25: {all_stats['q25']:.4f}")
    print(f"  Q75: {all_stats['q75']:.4f}")
    print()
    
    if 'above_epsilon_stats' in stats:
        print(f"Entropy > {epsilon} stats:")
        above_stats = stats['above_epsilon_stats']
        print(f"  Min: {above_stats['min']:.4f}")
        print(f"  Max: {above_stats['max']:.4f}")
        print(f"  Mean: {above_stats['mean']:.4f}")
        print(f"  Median: {above_stats['median']:.4f}")
        print(f"  Std: {above_stats['std']:.4f}")
        print()


def plot_entropy_distribution(data: List[Dict[str, Any]], epsilon: float = 0.1, 
                            save_path: Optional[Union[str, Path]] = None, bins: int = 50) -> None:
    """
    Plot histogram of entropy values with threshold line.
    
    Args:
        data: List of dictionaries containing the data
        epsilon: Threshold for high entropy classification
        save_path: Path to save the plot (optional)
        bins: Number of histogram bins
    """
    if not validate_entropy_field(data):
        return
    
    entropies = [item["entropy"] for item in data]
    
    plt.figure(figsize=(10, 6))
    
    # Main histogram
    plt.hist(entropies, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    
    # Threshold line
    plt.axvline(epsilon, color='red', linestyle='--', linewidth=2, label=f'ε = {epsilon}')
    
    # Add mean and median lines
    mean_entropy = np.mean(entropies)
    median_entropy = np.median(entropies)
    plt.axvline(mean_entropy, color='green', linestyle='-', linewidth=2, label=f'Mean = {mean_entropy:.3f}')
    plt.axvline(median_entropy, color='orange', linestyle='-', linewidth=2, label=f'Median = {median_entropy:.3f}')
    
    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.title("Distribution of Entropy Values")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Always save to figures directory
    figures_dir = ensure_figures_directory()
    
    if save_path is None:
        save_path = figures_dir / f"entropy_distribution_eps{epsilon}.png"
    else:
        save_path = figures_dir / Path(save_path).name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved entropy distribution plot to {save_path}")
    
    plt.show()


def analyze_entropy_by_category(data: List[Dict[str, Any]], category_field: str, 
                              epsilon: float = 0.1) -> Dict[str, Any]:
    """
    Analyze entropy statistics grouped by a categorical field.
    
    Args:
        data: List of dictionaries containing the data
        category_field: Field name to group by
        epsilon: Threshold for high entropy classification
        
    Returns:
        Dictionary containing statistics grouped by category
    """
    if not validate_entropy_field(data):
        return {}
    
    # Check if category field exists
    missing_category = [i for i, item in enumerate(data) if category_field not in item]
    if missing_category:
        print(f"Warning: {len(missing_category)} items missing {category_field} field")
        return {}
    
    # Group data by category
    categories = defaultdict(list)
    for item in data:
        categories[item[category_field]].append(item['entropy'])
    
    results = {}
    for category, entropies in categories.items():
        entropies = np.array(entropies)
        above_epsilon = entropies[entropies > epsilon]
        
        results[category] = {
            'count': len(entropies),
            'above_epsilon': len(above_epsilon),
            'pct_above_epsilon': 100.0 * len(above_epsilon) / len(entropies),
            'mean': float(np.mean(entropies)),
            'median': float(np.median(entropies)),
            'std': float(np.std(entropies)),
            'min': float(np.min(entropies)),
            'max': float(np.max(entropies))
        }
    
    return results


def print_category_analysis(data: List[Dict[str, Any]], category_field: str, 
                          epsilon: float = 0.1) -> None:
    """
    Print formatted category-based entropy analysis.
    
    Args:
        data: List of dictionaries containing the data
        category_field: Field name to group by
        epsilon: Threshold for high entropy classification
    """
    results = analyze_entropy_by_category(data, category_field, epsilon)
    
    if not results:
        return
    
    print(f"=== Entropy Analysis by {category_field} (ε = {epsilon}) ===")
    print()
    
    # Sort categories by mean entropy
    sorted_categories = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    for category, stats in sorted_categories:
        print(f"{category}:")
        print(f"  Count: {stats['count']}")
        print(f"  Above ε: {stats['above_epsilon']} ({stats['pct_above_epsilon']:.1f}%)")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()


def plot_category_analysis(data: List[Dict[str, Any]], category_field: str, 
                          epsilon: float = 0.1, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot category-based entropy analysis.
    
    Args:
        data: List of dictionaries containing the data
        category_field: Field name to group by
        epsilon: Threshold for high entropy classification
        save_path: Path to save the plot (optional)
    """
    results = analyze_entropy_by_category(data, category_field, epsilon)
    
    if not results:
        return
    
    # Sort categories by mean entropy
    sorted_categories = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    categories = [cat for cat, _ in sorted_categories]
    means = [stats['mean'] for _, stats in sorted_categories]
    pct_above = [stats['pct_above_epsilon'] for _, stats in sorted_categories]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mean entropy by category
    bars1 = ax1.bar(categories, means, color='lightblue', alpha=0.7)
    ax1.axhline(epsilon, color='red', linestyle='--', label=f'ε = {epsilon}')
    ax1.set_ylabel('Mean Entropy')
    ax1.set_title(f'Mean Entropy by {category_field}')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Percentage above epsilon by category
    bars2 = ax2.bar(categories, pct_above, color='lightcoral', alpha=0.7)
    ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.set_ylabel('Percentage Above ε (%)')
    ax2.set_title(f'Percentage Above Threshold by {category_field}')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, pct in zip(bars2, pct_above):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Always save to figures directory
    figures_dir = ensure_figures_directory()
    
    if save_path is None:
        save_path = figures_dir / f"category_analysis_{category_field}_eps{epsilon}.png"
    else:
        save_path = figures_dir / Path(save_path).name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved category analysis plot to {save_path}")
    
    plt.show()


def compare_datasets(datasets: Dict[str, Union[str, Path]], epsilon: float = 0.1) -> Dict[str, Any]:
    """
    Compare entropy statistics across multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to file paths
        epsilon: Threshold for high entropy classification
        
    Returns:
        Dictionary containing comparison statistics
    """
    results = {}
    
    for name, file_path in datasets.items():
        print(f"Loading dataset: {name}")
        try:
            data = load_dataset(file_path)
            stats = basic_entropy_stats(data, epsilon)
            if stats:
                results[name] = stats
            else:
                print(f"Warning: Could not compute stats for {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return results


def print_dataset_comparison(datasets: Dict[str, Union[str, Path]], epsilon: float = 0.1) -> None:
    """
    Print formatted comparison of multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to file paths
        epsilon: Threshold for high entropy classification
    """
    results = compare_datasets(datasets, epsilon)
    
    if not results:
        return
    
    print(f"=== Dataset Comparison (ε = {epsilon}) ===")
    print()
    
    # Print summary table
    print(f"{'Dataset':<20} {'Total':<8} {'Above ε':<8} {'% Above':<8} {'Mean':<8} {'Median':<8}")
    print("-" * 70)
    
    for name, stats in results.items():
        print(f"{name:<20} {stats['total_items']:<8} {stats['above_epsilon']:<8} "
              f"{stats['pct_above_epsilon']:<8.1f} {stats['all_entropy']['mean']:<8.4f} "
              f"{stats['all_entropy']['median']:<8.4f}")
    
    print()


def plot_dataset_comparison(datasets: Dict[str, Union[str, Path]], epsilon: float = 0.1,
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Create comparison plots for multiple datasets.
    
    Args:
        datasets: Dictionary mapping dataset names to file paths
        epsilon: Threshold for high entropy classification
        save_path: Path to save the plot (optional)
    """
    results = compare_datasets(datasets, epsilon)
    
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    all_entropies = []
    labels = []
    for name, stats in results.items():
        data = load_dataset(datasets[name])
        entropies = [item["entropy"] for item in data]
        all_entropies.append(entropies)
        labels.append(name)
    
    ax1.boxplot(all_entropies, labels=labels)
    ax1.axhline(epsilon, color='red', linestyle='--', label=f'ε = {epsilon}')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Entropy Distribution Comparison')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Bar plot of percentage above epsilon
    names = list(results.keys())
    pct_above = [results[name]['pct_above_epsilon'] for name in names]
    
    bars = ax2.bar(names, pct_above, color='lightcoral', alpha=0.7)
    ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax2.set_ylabel('Percentage Above ε (%)')
    ax2.set_title('Percentage of Items Above Threshold')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, pct in zip(bars, pct_above):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Always save to figures directory
    figures_dir = ensure_figures_directory()
    
    if save_path is None:
        dataset_names = "_".join(list(results.keys())[:3])  # Use first 3 dataset names
        save_path = figures_dir / f"dataset_comparison_{dataset_names}_eps{epsilon}.png"
    else:
        save_path = figures_dir / Path(save_path).name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved dataset comparison plot to {save_path}")
    
    plt.show()


def entropy_threshold_analysis(data: List[Dict[str, Any]], thresholds: List[float] = None) -> Dict[str, Any]:
    """
    Analyze how entropy statistics change across different thresholds.
    
    Args:
        data: List of dictionaries containing the data
        thresholds: List of thresholds to analyze (default: [0.01, 0.05, 0.1, 0.2, 0.5])
        
    Returns:
        Dictionary containing statistics for each threshold
    """
    if not validate_entropy_field(data):
        return {}
    
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    entropies = np.array([item["entropy"] for item in data])
    results = {}
    
    for threshold in thresholds:
        above_threshold = entropies[entropies > threshold]
        
        results[threshold] = {
            'above_threshold': len(above_threshold),
            'pct_above': 100.0 * len(above_threshold) / len(entropies),
            'mean_above': float(np.mean(above_threshold)) if len(above_threshold) > 0 else 0,
            'median_above': float(np.median(above_threshold)) if len(above_threshold) > 0 else 0
        }
    
    return results


def print_threshold_analysis(data: List[Dict[str, Any]], thresholds: List[float] = None) -> None:
    """
    Print formatted threshold analysis.
    
    Args:
        data: List of dictionaries containing the data
        thresholds: List of thresholds to analyze
    """
    results = entropy_threshold_analysis(data, thresholds)
    
    if not results:
        return
    
    print("=== Entropy Threshold Analysis ===")
    print()
    print(f"{'Threshold':<12} {'Above':<8} {'% Above':<10} {'Mean Above':<12} {'Median Above':<12}")
    print("-" * 60)
    
    for threshold, stats in sorted(results.items()):
        print(f"{threshold:<12.2f} {stats['above_threshold']:<8} {stats['pct_above']:<10.1f} "
              f"{stats['mean_above']:<12.4f} {stats['median_above']:<12.4f}")
    
    print()


def plot_threshold_analysis(data: List[Dict[str, Any]], thresholds: List[float] = None,
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot threshold analysis results.
    
    Args:
        data: List of dictionaries containing the data
        thresholds: List of thresholds to analyze
        save_path: Path to save the plot (optional)
    """
    results = entropy_threshold_analysis(data, thresholds)
    
    if not results:
        return
    
    thresholds_list = sorted(results.keys())
    pct_above = [results[t]['pct_above'] for t in thresholds_list]
    mean_above = [results[t]['mean_above'] for t in thresholds_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Percentage above threshold
    ax1.plot(thresholds_list, pct_above, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Percentage Above Threshold (%)')
    ax1.set_title('Percentage of Items Above Different Thresholds')
    ax1.grid(True, alpha=0.3)
    
    # Mean entropy above threshold
    ax2.plot(thresholds_list, mean_above, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Mean Entropy Above Threshold')
    ax2.set_title('Mean Entropy of Items Above Different Thresholds')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Always save to figures directory
    figures_dir = ensure_figures_directory()
    
    if save_path is None:
        threshold_range = f"{min(thresholds_list):.2f}-{max(thresholds_list):.2f}"
        save_path = figures_dir / f"threshold_analysis_{threshold_range}.png"
    else:
        save_path = figures_dir / Path(save_path).name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved threshold analysis plot to {save_path}")
    
    plt.show()


def ensure_figures_directory() -> Path:
    """
    Ensure the figures directory exists and return its path.
    
    Returns:
        Path to the figures directory
    """
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    return figures_dir


def list_saved_figures() -> List[str]:
    """
    List all saved figures in the figures directory.
    
    Returns:
        List of figure filenames
    """
    figures_dir = ensure_figures_directory()
    return [f.name for f in figures_dir.glob("*.png")]


def clear_figures_directory() -> None:
    """
    Remove all PNG files from the figures directory.
    """
    figures_dir = ensure_figures_directory()
    for png_file in figures_dir.glob("*.png"):
        png_file.unlink()
    print(f"Cleared {figures_dir} directory")


# Example usage function
def analyze_entropy_dataset(file_path: Union[str, Path], epsilon: float = 0.1, 
                          category_field: Optional[str] = None,
                          save_plots: bool = True) -> None:
    """
    Comprehensive entropy analysis for a single dataset.
    
    Args:
        file_path: Path to the dataset file
        epsilon: Threshold for high entropy classification
        category_field: Optional field to group analysis by
        save_plots: Whether to save plots to files (always saves to figures/ directory)
    """
    print(f"Analyzing entropy for dataset: {file_path}")
    print("=" * 50)
    
    # Load data
    data = load_dataset(file_path)
    
    # Basic stats
    print_entropy_stats(data, epsilon)
    
    # Distribution plot (always saves to figures/)
    dataset_name = Path(file_path).stem
    plot_entropy_distribution(data, epsilon, f"entropy_distribution_{dataset_name}_eps{epsilon}.png")
    
    # Category analysis if field provided
    if category_field:
        print_category_analysis(data, category_field, epsilon)
        plot_category_analysis(data, category_field, epsilon, save_path=f"category_analysis_{category_field}_{dataset_name}.png")
    
    # Threshold analysis (always saves to figures/)
    plot_threshold_analysis(data, save_path=f"threshold_analysis_{dataset_name}.png")


def comprehensive_entropy_analysis(datasets: Dict[str, Union[str, Path]], 
                                 epsilon: float = 0.1,
                                 category_field: Optional[str] = None) -> None:
    """
    Perform comprehensive entropy analysis on multiple datasets.
    All plots are automatically saved to the figures/ directory.
    
    Args:
        datasets: Dictionary mapping dataset names to file paths
        epsilon: Threshold for high entropy classification
        category_field: Optional field to group analysis by
    """
    print("=== COMPREHENSIVE ENTROPY ANALYSIS ===")
    print(f"Analyzing {len(datasets)} datasets with ε = {epsilon}")
    print("=" * 60)
    
    # Ensure figures directory exists
    ensure_figures_directory()
    
    # Individual dataset analysis
    for name, file_path in datasets.items():
        print(f"\n--- Analyzing {name} ---")
        analyze_entropy_dataset(file_path, epsilon, category_field)
    
    # Dataset comparison
    print(f"\n--- Dataset Comparison ---")
    print_dataset_comparison(datasets, epsilon)
    plot_dataset_comparison(datasets, epsilon)
    
    # List all saved figures
    saved_figures = list_saved_figures()
    print(f"\n=== Saved Figures ({len(saved_figures)} total) ===")
    for figure in sorted(saved_figures):
        print(f"  {figure}")
    print(f"\nAll figures saved to: {ensure_figures_directory().absolute()}") 