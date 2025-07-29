"""
Entropy Aggregator for combining entropy results.

This module provides the EntropyAggregator class for aggregating entropy
results from multiple files and computing statistics.
"""

import glob
import numpy as np
from typing import List, Dict, Union
from pathlib import Path

from .utils import calculate_entropy_statistics


class EntropyAggregator:
    """
    A class for aggregating entropy results from multiple files and computing statistics.
    
    This class handles loading multiple entropy files, concatenating them,
    and computing comprehensive statistics.
    """
    
    def __init__(self):
        """Initialize the EntropyAggregator."""
        pass
    
    def aggregate_entropies(
        self, 
        file_paths: Union[List[str], str],
        auto_discover: bool = False,
        pattern: str = "entropy_outputs/entropies_*.npy"
    ) -> Dict[str, float]:
        """
        Aggregate entropies from multiple files and compute statistics.
        
        Parameters
        ----------
        file_paths : List[str] | str
            List of file paths to aggregate, or a single path.
            If auto_discover is True, this can be a directory path.
        auto_discover : bool, default=False
            Whether to automatically discover .npy files using the pattern.
        pattern : str, default="entropy_outputs/entropies_*.npy"
            Pattern to use for auto-discovery of entropy files.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing aggregated entropy statistics.
        """
        # Handle file path discovery
        if auto_discover:
            if isinstance(file_paths, str):
                # Use the string as a directory and apply pattern
                pattern = str(Path(file_paths) / "entropies_*.npy")
            file_paths = glob.glob(pattern)
        
        elif isinstance(file_paths, str):
            file_paths = [file_paths]
        
        if not file_paths:
            raise ValueError("No entropy files found. Check paths or enable auto_discover.")
        
        # Load and concatenate all entropy arrays
        arrays = []
        for file_path in file_paths:
            try:
                arr = np.load(file_path)
                arrays.append(arr)
                print(f"Loaded {len(arr)} entropies from {file_path}")
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        if not arrays:
            raise ValueError("No valid entropy files could be loaded.")
        
        # Concatenate all arrays
        all_entropies = np.concatenate(arrays)
        
        # Calculate statistics
        stats = calculate_entropy_statistics(all_entropies.tolist())
        
        # Add additional metadata
        stats["num_files"] = len(arrays)
        stats["file_paths"] = file_paths
        
        return stats
    
    def load_single_file(self, file_path: str) -> Dict[str, float]:
        """
        Load and compute statistics for a single entropy file.
        
        Parameters
        ----------
        file_path : str
            Path to the entropy file.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing entropy statistics.
        """
        try:
            entropies = np.load(file_path)
            stats = calculate_entropy_statistics(entropies.tolist())
            stats["file_path"] = file_path
            return stats
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")
    
    def compare_files(self, file_paths: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compare statistics across multiple entropy files.
        
        Parameters
        ----------
        file_paths : List[str]
            List of file paths to compare.
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary mapping file paths to their statistics.
        """
        results = {}
        
        for file_path in file_paths:
            try:
                results[file_path] = self.load_single_file(file_path)
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                results[file_path] = {"error": str(e)}
        
        return results
    
    def find_entropy_threshold(
        self, 
        file_paths: Union[List[str], str],
        percentile: float = 80.0,
        auto_discover: bool = False
    ) -> float:
        """
        Find the entropy threshold at a specific percentile.
        
        Parameters
        ----------
        file_paths : List[str] | str
            File paths to aggregate.
        percentile : float, default=80.0
            Percentile to use for threshold calculation.
        auto_discover : bool, default=False
            Whether to auto-discover files.
            
        Returns
        -------
        float
            The entropy threshold at the specified percentile.
        """
        stats = self.aggregate_entropies(file_paths, auto_discover=auto_discover)
        
        # Calculate the specific percentile
        all_entropies = np.concatenate([
            np.load(fp) for fp in stats["file_paths"]
        ])
        
        threshold = np.percentile(all_entropies, percentile)
        return float(threshold)
    
    def print_summary(self, stats: Dict[str, float]):
        """
        Print a formatted summary of entropy statistics.
        
        Parameters
        ----------
        stats : Dict[str, float]
            Statistics dictionary from aggregate_entropies or load_single_file.
        """
        print("\n--- Entropy Statistics Summary ---")
        print(f"Total tokens: {stats.get('total_tokens', 'N/A')}")
        print(f"Number of files: {stats.get('num_files', 1)}")
        print(f"Mean entropy: {stats.get('mean', 'N/A'):.4f}")
        print(f"Median entropy: {stats.get('median', 'N/A'):.4f}")
        print(f"Standard deviation: {stats.get('std', 'N/A'):.4f}")
        print(f"Min entropy: {stats.get('min', 'N/A'):.4f}")
        print(f"Max entropy: {stats.get('max', 'N/A'):.4f}")
        print(f"80th percentile: {stats.get('80th_percentile', 'N/A'):.4f}")
        print(f"90th percentile: {stats.get('90th_percentile', 'N/A'):.4f}")
        print(f"95th percentile: {stats.get('95th_percentile', 'N/A'):.4f}")
        
        if "file_paths" in stats:
            print(f"\nFiles processed:")
            for fp in stats["file_paths"]:
                print(f"  - {fp}") 