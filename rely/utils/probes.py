import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class UncertaintyProbe(nn.Module):
    """
    Simple linear probe to predict an uncertainty score from hidden states.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear(x)


class ValueProbe(nn.Module):
    """
    MLP probe to predict a value score for a given sequence.
    """
    def __init__(self, hidden_size, hidden_dims=(512, 128), dropout_p=0.3):
        super().__init__()
        layers = []
        current_dim = hidden_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            current_dim = h_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


def load_probes(
    hidden_size: int, 
    model_dtype: torch.dtype,
    uncertainty_probe_path: Optional[str] = None,
    value_probe_path: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[UncertaintyProbe, ValueProbe]:
    """
    Load uncertainty and value probes from saved checkpoints.
    
    Args:
        hidden_size: Hidden size of the model
        model_dtype: Data type of the model
        uncertainty_probe_path: Path to uncertainty probe checkpoint
        value_probe_path: Path to value probe checkpoint
        device: Device to load probes on
    
    Returns:
        Tuple of (uncertainty_probe, value_probe)
    """
    # Load uncertainty probe
    uncertainty_probe = UncertaintyProbe(hidden_size).to(device).to(model_dtype)
    if uncertainty_probe_path and Path(uncertainty_probe_path).exists():
        try:
            uncertainty_probe.load_state_dict(
                torch.load(uncertainty_probe_path, map_location=device)
            )
        except Exception as e:
            print(f"Warning: Could not load uncertainty probe from {uncertainty_probe_path}: {e}")
    uncertainty_probe.eval()

    # Load value probe
    value_probe = ValueProbe(hidden_size).to(device).to(model_dtype)
    if value_probe_path and Path(value_probe_path).exists():
        try:
            value_probe.load_state_dict(
                torch.load(value_probe_path, map_location=device)
            )
        except Exception as e:
            print(f"Warning: Could not load value probe from {value_probe_path}: {e}")
    value_probe.eval()
    
    return uncertainty_probe, value_probe


def convert_isotropy_to_branches(approximated_I_score: float, n: int, w: int) -> int:
    """
    Converts the approximated semantic isotropy score to a number of branches.
    
    Args:
        approximated_I_score: Uncertainty score from the probe
        n: Number of approximation steps
        w: Beam width
    
    Returns:
        Number of branches to generate
    """
    if n <= 1:
        return 1
    normalized_score = approximated_I_score / np.log(n)
    num_branches = int(np.ceil(normalized_score * w))
    return max(1, min(num_branches, w))  # Ensure between 1 and w 