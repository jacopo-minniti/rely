"""
Complete module for generating model completions.

This module contains functions for generating completions from various data sources
using vLLM for efficient inference.
"""

from .generate import generate_from_dataset
from .complete import complete_from_forks
from .complete_nn import complete_from_jsonl

__all__ = [
    'generate_from_dataset',
    'complete_from_forks', 
    'complete_from_jsonl'
] 