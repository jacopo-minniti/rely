from dataclasses import dataclass, field
from typing import Optional, Union
import torch


@dataclass
class UATSConfig:
    """Configuration for UATS inference."""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    uncertainty_model_path: str = "path/to/uncertainty/model"
    value_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B"
    uncertainty_scoring_method: str = "last_step"  # "product", "average", "minimum", or "last_step"
    value_scoring_method: str = "product"  # "product", "average", "minimum", or "last_step"
    beam_width: int = 3
    max_branches: int = 2 
    budget: int = 1024 
    uncertainty_threshold: Union[float, None] = 0.5
    max_step_tokens: int = 256
    device: str = "cuda:1"
    uncertainty_device: str = "cuda:0"
    value_device: str = "cuda:0"
    temperature: float = 0.9
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
    is_final: bool = False

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
            "is_final": self.is_final,
        }
    
    @classmethod
    def from_dict(cls, data):
        """Creates a Branch object from a dictionary."""
        return cls(
            text=data["text"],
            ids=torch.tensor([]),
            step_count=data["step_count"],
            score=data["score"],
            uncertainty=data["uncertainty"],
            value=data["value"],
            total_tokens=data["total_tokens"],
            id=data["id"],
            parent_id=data["parent_id"],
            final_answer=data.get("final_answer"),
            is_final=data.get("is_final", False),
        )