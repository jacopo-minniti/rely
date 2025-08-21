from dataclasses import dataclass
from typing import Optional, Union
import torch


@dataclass
class UATSConfig:
    """Configuration for UATS inference."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    uncertainty_probe_path: Optional[str] = None
    value_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B"  # HF model with classification head
    value_scoring_method: str = "product"  # "product", "minimum", "average", or "last_step"
    beam_width: int = 3
    budget: int = 1024 
    uncertainty_threshold: Union[float, None] = 0.8
    max_step_tokens: int = 256
    device: str = "auto"
    probe_device: str = "cuda"
    value_device: str = "cuda:1"  # Separate device for value model
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
    id: int = -1
    parent_id: Optional[int] = None
    
    def __eq__(self, other):
        """Compare branches by their unique id to avoid tensor comparison issues."""
        if not isinstance(other, Branch):
            return False
        return self.id == other.id
    
    def __hash__(self):
        """Hash based on id for consistent behavior with __eq__."""
        return hash(self.id) 