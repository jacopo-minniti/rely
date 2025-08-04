from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class UATSConfig:
    """Configuration for UATS inference."""

    model_name: str = "unsloth/Qwen3-1.7B-unsloth-bnb-4"
    uncertainty_probe_path: Optional[str] = None
    value_probe_path: Optional[str] = None
    beam_width: int = 3
    budget: int = 1024  # Total token budget across all branches during reasoning
    max_step_tokens: int = 256
    device: str = "auto"
    probe_device: str = "cuda"
    temperature: float = 1.0
    top_p: float = 0.95 


@dataclass
class Branch:
    """Represents a single branch in the search tree."""

    text: str
    ids: torch.Tensor
    step_count: int
    score: float
    uncertainty: Optional[float]
    value: float
    total_tokens: int
    final_answer: Optional[str] = None 