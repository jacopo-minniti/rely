"""
Utility functions for entropy threshold calculations.

This module contains the core functions for loading prompts and calculating
token-level entropies from language model outputs.
"""

import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from ..utils import (
    MMLU_SYSTEM_PROMPT,
    format_system_prompt,
    ensure_think_ending,
    basic_entropy_stats
)

# Use the system prompt from the main utils module
SYSTEM_PROMPT = MMLU_SYSTEM_PROMPT


def load_prompts(
    dataset_path: str,
    start_idx: Optional[int] = None, 
    end_idx: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Loads and formats prompts from the dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the JSONL file containing the dataset.
    start_idx : int | None
        Inclusive start index of the line in the dataset to process. If ``None`` starts from 0.
    end_idx : int | None
        Exclusive end index. If ``None`` reads until the end of file.

    Returns
    -------
    List[Dict[str, str]]
        List of prompt/completion pairs formatted for entropy calculation.
    """
    prompts: List[Dict[str, str]] = []

    # Stream the file line-by-line so that we can cheaply skip to the start_idx.
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if start_idx is not None and i < start_idx:
                continue  # Skip until we reach the first relevant line
            if end_idx is not None and i >= end_idx:
                break     # We have read all requested lines

            item = json.loads(line)

            # Build the prompt/response pair using the utility function
            user_prompt_part = format_system_prompt(SYSTEM_PROMPT, item['question'])
            completion_text = ensure_think_ending(item['attempt']) + "<|im_end|>"
            prompts.append({
                "prompt": user_prompt_part,
                "completion": completion_text,
            })

    return prompts


def calculate_token_entropies(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[Dict[str, str]],
    temperature: float = 1.0,
    device: str = "cuda"
) -> List[float]:
    """
    Calculates the entropy for each token in the completion part of the prompts.
    This follows the paper's methodology.

    Parameters
    ----------
    model : torch.nn.Module
        The language model for inference.
    tokenizer
        The tokenizer for the model.
    prompts : List[Dict[str, str]]
        List of prompt/completion pairs.
    temperature : float, default=1.0
        Temperature for entropy calculation.
    device : str, default="cuda"
        Device to run the model on.

    Returns
    -------
    List[float]
        List of entropy values for each token in the completions.
    """
    all_entropies = []

    print(f"Calculating token entropies for {len(prompts)} prompts...")
    # Use tqdm for a progress bar
    for item in tqdm(prompts):
        full_text = item['prompt'] + item['completion']
        
        # Tokenize the full sequence
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        
        # We need to find where the completion part starts to only calculate entropy for those tokens
        prompt_tokens = tokenizer(item['prompt'], return_tensors="pt")
        start_index = prompt_tokens.input_ids.shape[1]

        with torch.no_grad():
            # Get logits for the entire sequence in one forward pass
            outputs = model(**inputs)
            logits = outputs.logits

        # We are interested in the entropy of predicting token `i` given tokens `0..i-1`.
        # The logits at index `i-1` are used to predict token `i`.
        # So we iterate from the start of the completion to the end of the sequence.
        for i in range(start_index, inputs.input_ids.shape[1]):
            # Get the logits for the current token's prediction (logits at position i-1)
            token_logits = logits[0, i - 1, :]

            # Apply temperature (T=1.0 doesn't change logits but is included for correctness)
            scaled_logits = token_logits / temperature
            
            # Calculate probabilities using softmax
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Calculate the entropy for this token's distribution
            # Use torch.distributions.Categorical for a stable and efficient implementation
            entropy = torch.distributions.Categorical(probs=probabilities).entropy()
            
            all_entropies.append(entropy.item())
            
    return all_entropies


def calculate_entropy_statistics(entropies: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics from a list of entropy values.

    Parameters
    ----------
    entropies : List[float]
        List of entropy values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean, median, and percentile statistics.
    """
    # Convert to the format expected by basic_entropy_stats
    data = [{"entropy": float(entropy)} for entropy in entropies]
    
    # Use the existing utility function
    stats = basic_entropy_stats(data)
    
    # Convert to our expected format
    entropies_arr = np.array(entropies, dtype=np.float32)
    
    return {
        "total_tokens": len(entropies_arr),
        "mean": stats.get("all_entropy", {}).get("mean", 0.0),
        "median": stats.get("all_entropy", {}).get("median", 0.0),
        "std": stats.get("all_entropy", {}).get("std", 0.0),
        "min": stats.get("all_entropy", {}).get("min", 0.0),
        "max": stats.get("all_entropy", {}).get("max", 0.0),
        "80th_percentile": float(np.percentile(entropies_arr, 80)),
        "90th_percentile": float(np.percentile(entropies_arr, 90)),
        "95th_percentile": float(np.percentile(entropies_arr, 95))
    } 