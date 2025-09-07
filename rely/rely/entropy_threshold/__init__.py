"""
Entropy Threshold Module

This module provides functionality for calculating token-level entropies from language model outputs
and determining entropy thresholds for uncertainty quantification.

Main Components:
- EntropyCalculator: Core class for calculating token entropies
- EntropyAggregator: Utility for aggregating entropy results
- ParallelProcessor: Helper for parallel processing across multiple GPUs

Example Usage:
    from rely.entropy_threshold import EntropyCalculator, EntropyAggregator
    
    # Initialize calculator
    calculator = EntropyCalculator(
        model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length=30000
    )
    
    # Calculate entropies for a dataset
    entropies = calculator.calculate_entropies(
        dataset_path="generations-mmlu-qwen3-8B.jsonl",
        start_idx=0,
        end_idx=1000
    )
    
    # Get statistics
    stats = calculator.get_entropy_statistics(entropies)
    threshold = stats["80th_percentile"]
"""

from .entropy_calculator import EntropyCalculator
from .entropy_aggregator import EntropyAggregator
from .parallel_processor import ParallelProcessor
from .utils import load_prompts, calculate_token_entropies, calculate_entropy_statistics

__all__ = [
    "EntropyCalculator",
    "EntropyAggregator", 
    "ParallelProcessor",
    "load_prompts",
    "calculate_token_entropies",
    "calculate_entropy_statistics"
]

__version__ = "1.0.0" 