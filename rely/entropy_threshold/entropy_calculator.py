"""
Entropy Calculator for Language Models.

This module provides the EntropyCalculator class for calculating token-level
entropies from language model outputs.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional
from unsloth import FastLanguageModel

from .utils import load_prompts, calculate_token_entropies, calculate_entropy_statistics


class EntropyCalculator:
    """
    A class for calculating token-level entropies from language model outputs.
    
    This class handles model loading, prompt processing, and entropy calculation
    in a clean, composable interface.
    """
    
    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length: int = 30000,
        load_in_4bit: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the EntropyCalculator.
        
        Parameters
        ----------
        model_name : str, default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
            Name of the model to load.
        max_seq_length : int, default=30000
            Maximum sequence length for the model.
        load_in_4bit : bool, default=True
            Whether to load the model in 4-bit quantization.
        device : str | None, default=None
            Device to load the model on. If None, will use CUDA if available.
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Set device='cpu' for CPU-only mode.")
        
        if self.device == "cpu":
            print("Warning: Running on CPU. This will be very slow.")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
            )
            FastLanguageModel.for_inference(self.model)
            self.model.eval()
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def calculate_entropies(
        self,
        dataset_path: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        temperature: float = 1.0,
        save_path: Optional[str] = None
    ) -> List[float]:
        """
        Calculate token entropies for a dataset.
        
        Parameters
        ----------
        dataset_path : str
            Path to the JSONL dataset file.
        start_idx : int | None, default=None
            Start index for processing (inclusive).
        end_idx : int | None, default=None
            End index for processing (exclusive).
        temperature : float, default=1.0
            Temperature for entropy calculation.
        save_path : str | None, default=None
            Path to save the entropies as a numpy array. If None, won't save.
            
        Returns
        -------
        List[float]
            List of entropy values for each token.
        """
        # Load prompts
        prompts = load_prompts(dataset_path, start_idx, end_idx)
        
        if not prompts:
            raise ValueError("No prompts were loaded. Check dataset path and indices.")
        
        # Calculate entropies
        entropies = calculate_token_entropies(
            self.model, 
            self.tokenizer, 
            prompts, 
            temperature=temperature,
            device=self.device
        )
        
        # Save if requested
        if save_path:
            self._save_entropies(entropies, save_path)
        
        return entropies
    
    def calculate_entropies_from_prompts(
        self,
        prompts: List[Dict[str, str]],
        temperature: float = 1.0,
        save_path: Optional[str] = None
    ) -> List[float]:
        """
        Calculate token entropies from pre-loaded prompts.
        
        Parameters
        ----------
        prompts : List[Dict[str, str]]
            List of prompt/completion pairs.
        temperature : float, default=1.0
            Temperature for entropy calculation.
        save_path : str | None, default=None
            Path to save the entropies as a numpy array. If None, won't save.
            
        Returns
        -------
        List[float]
            List of entropy values for each token.
        """
        if not prompts:
            raise ValueError("No prompts provided.")
        
        # Calculate entropies
        entropies = calculate_token_entropies(
            self.model, 
            self.tokenizer, 
            prompts, 
            temperature=temperature,
            device=self.device
        )
        
        # Save if requested
        if save_path:
            self._save_entropies(entropies, save_path)
        
        return entropies
    
    def get_entropy_statistics(self, entropies: List[float]) -> Dict[str, float]:
        """
        Calculate statistics from entropy values.
        
        Parameters
        ----------
        entropies : List[float]
            List of entropy values.
            
        Returns
        -------
        Dict[str, float]
            Dictionary containing entropy statistics.
        """
        return calculate_entropy_statistics(entropies)
    
    def _save_entropies(self, entropies: List[float], save_path: str):
        """Save entropies to a numpy file."""
        if not entropies:
            print("Warning: No entropies to save.")
            return
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Convert to numpy array and save
        entropies_arr = np.array(entropies, dtype=np.float32)
        np.save(save_path, entropies_arr)
        
        print(f"Saved {len(entropies_arr)} entropies to {save_path}")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer 