"""
Calculate entropy scores from model completions.

This module provides functions for calculating entropy scores and semantic isotropy
from model completions and embeddings.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
from collections import Counter
import math
from typing import List, Dict, Any, Optional


def calculate_semantic_isotropy(embeddings: torch.Tensor) -> float:
    """
    Calculates the semantic isotropy score for a given set of embeddings.
    
    The score is the von Neumann entropy of the cosine similarity matrix
    of the normalized embeddings.
    
    Args:
        embeddings: A tensor of shape [K, D], where K is the number of
                    completions and D is the embedding dimension.
    
    Returns:
        The calculated semantic isotropy score as a float.
    """
    # Ensure calculations are done without tracking gradients
    with torch.no_grad():
        # Step 0: Handle edge cases where a score cannot be computed
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape[0] < 2:
            return -1

        # Step 1: Normalize the embeddings to have unit L2 norm
        # This projects the embeddings onto the unit sphere
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Step 2: Compute the Cosine Kernel Matrix
        # For normalized vectors, this is equivalent to a matrix multiplication
        kernel_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Step 3: Calculate the von Neumann Entropy of the kernel matrix
        # This requires finding the eigenvalues of the matrix.
        # We use .eigvalsh() as the kernel matrix is symmetric, which is more stable.
        try:
            # Clamp the matrix to be perfectly symmetric and add a small identity
            # matrix for numerical stability before finding eigenvalues.
            k = (kernel_matrix + kernel_matrix.T) / 2
            k = k + 1e-6 * torch.eye(k.shape[0], device=k.device)
            eigenvalues = torch.linalg.eigvalsh(k)
        except Exception as e:
            print(f"Error calculating eigenvalues: {e}")
            # If eigenvalue computation fails, return a default score
            return -1

        # Filter out non-positive eigenvalues that can result from numerical instability
        positive_eigenvalues = eigenvalues[eigenvalues > 1e-6]
        if positive_eigenvalues.numel() == 0:
            return -1

        # The von Neumann entropy formula is -Tr(p * log(p)), where p is the density
        # matrix. In terms of eigenvalues, this is -sum(eig * log(eig)).
        entropy = -torch.sum(positive_eigenvalues * torch.log(positive_eigenvalues))
        
        return entropy.item()


def calculate_entropy_from_completions(completions: List[str]) -> float:
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


def calculate_entropy_scores(
    input_file: str,
    output_file: str,
    dataset_percentage: float = 1.0
) -> None:
    """
    Calculate entropy scores from completions in a dataset.
    
    Args:
        input_file: Path to input PT file containing completions
        output_file: Path to output PT file with entropy scores
        dataset_percentage: Percentage of dataset to process (0.0-1.0)
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load the dataset
    try:
        data = torch.load(input_file)
        print(f"Loaded {len(data)} items from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Take only a percentage of the dataset
    num_items_to_process = int(len(data) * dataset_percentage)
    data = data[:num_items_to_process]
    print(f"Processing {len(data)} items ({dataset_percentage*100:.1f}% of dataset)")
    
    # Process each item and add entropy
    processed_data = []
    
    for i, item in enumerate(tqdm(data, desc="Processing items")):
        # Create a copy of the item to avoid modifying the original
        processed_item = item.copy()
        
        # Calculate entropy from completions
        if 'completions' in item:
            entropy_score = calculate_entropy_from_completions(item['completions'])
            processed_item['entropy'] = entropy_score
            
            # Only add items with valid entropy scores (not -1)
            if entropy_score != -1:
                processed_data.append(processed_item)
        else:
            print(f"Warning: Item {i} has no 'completions' field")
            continue
    
    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    try:
        torch.save(processed_data, output_file)
        print(f"Successfully saved {len(processed_data)} items with entropy scores")
    except Exception as e:
        print(f"Error saving dataset: {e}")
        raise
    
    # Print some statistics
    valid_entropies = [item['entropy'] for item in processed_data if item['entropy'] != -1]
    if valid_entropies:
        print(f"Entropy statistics:")
        print(f"  Mean: {sum(valid_entropies) / len(valid_entropies):.4f}")
        print(f"  Min: {min(valid_entropies):.4f}")
        print(f"  Max: {max(valid_entropies):.4f}")
        print(f"  Valid items: {len(valid_entropies)}/{len(processed_data)}")
    else:
        print("No valid entropy scores found")


def main():
    """Command-line interface for calculate_entropy_scores."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate entropy scores from completions")
    parser.add_argument("--input-file", type=str, default="nn-short-100-v2.pt", help="Input PT file path")
    parser.add_argument("--output-file", type=str, default="nn-short-100-v2-scores.pt", help="Output PT file path")
    parser.add_argument("--dataset-percentage", type=float, default=1.0, help="Percentage of dataset to process (0.0-1.0)")
    
    args = parser.parse_args()
    
    calculate_entropy_scores(
        input_file=args.input_file,
        output_file=args.output_file,
        dataset_percentage=args.dataset_percentage
    )


if __name__ == "__main__":
    main() 