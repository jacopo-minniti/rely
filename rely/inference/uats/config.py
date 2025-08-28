from dataclasses import dataclass, field
from typing import Optional, Union
import torch


@dataclass
class UATSConfig:
    """Configuration for UATS inference."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    uncertainty_model_path: str = "path/to/uncertainty/model"  # HF model with classification head
    uncertainty_scoring_method: str = "last_step"  # "product", "minimum", "average", or "last_step"
    value_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B"  # HF model with classification head
    value_scoring_method: str = "product"  # "product", "minimum", "average", or "last_step"
    beam_width: int = 3
    budget: int = 1024 
    uncertainty_threshold: Union[float, None] = 0.8
    max_step_tokens: int = 256
    # Device configuration - supports hosting each model on different devices
    device: str = "cuda:1"  # Policy/generation model device
    uncertainty_device: str = "cuda:0"  # Uncertainty model device  
    value_device: str = "cuda:0"  # Value model device
    temperature: float = 1.0
    top_p: float = 0.95 


@dataclass
class Branch:
    text: str
    ids: torch.Tensor
    step_count: int
    score: float
    uncertainty: Optional[float]
    value: float
    total_tokens: int
    id: int
    parent_id: Optional[int]
    final_answer: Optional[str] = None

    def to_dict(self):
        """Converts the Branch object to a JSON-serializable dictionary."""
        return {
            "text": self.text,
            "step_count": self.step_count,
            "score": self.score,
            "uncertainty": self.uncertainty,
            "value": self.value,
            "total_tokens": self.total_tokens,
            "id": self.id,
            "parent_id": self.parent_id,
            "final_answer": self.final_answer,
        }
    
    @classmethod
    def from_dict(cls, data):
        """Creates a Branch object from a dictionary."""
        # 'ids' are not stored, so we create a placeholder tensor.
        # This is okay because the primary use of from_dict is in the main process
        # after the search is complete, where 'ids' are not needed for further generation.
        ids_tensor = torch.tensor([])

        return cls(
            text=data["text"],
            ids=ids_tensor,
            step_count=data["step_count"],
            score=data["score"],
            uncertainty=data["uncertainty"],
            value=data["value"],
            total_tokens=data["total_tokens"],
            id=data["id"],
            parent_id=data["parent_id"],
            final_answer=data.get("final_answer"),
        )