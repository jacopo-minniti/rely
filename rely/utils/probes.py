import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

class MLPProbe(nn.Module):
    """A more advanced MLP probe with configurable depth, width, dropout, and batch norm."""
    def __init__(self, input_dim, hidden_dims=[512, 128], dropout_p=0.3):
        super(MLPProbe, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # Batch norm before activation
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            current_dim = h_dim
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, 1)  # Final layer to produce logits

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)



def load_probes(
    hidden_size: int, 
    model_dtype: torch.dtype,
    uncertainty_probe_path: Optional[str] = None,
    value_probe_path: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
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
        
    Raises:
        FileNotFoundError: If either probe file is not found
        RuntimeError: If probe loading fails
    """
    # Check if uncertainty probe path exists
    if not uncertainty_probe_path or not Path(uncertainty_probe_path).exists():
        raise FileNotFoundError(f"Uncertainty probe file not found: {uncertainty_probe_path}")
    
    # Check if value probe path exists
    if not value_probe_path or not Path(value_probe_path).exists():
        raise FileNotFoundError(f"Value probe file not found: {value_probe_path}")
    
    uncertainty_probe = None
    try:
        uncertainty_probe = MLPProbe(hidden_size, hidden_dims=[256, 128]).to(device).to(model_dtype)
        uncertainty_probe.load_state_dict(
            torch.load(uncertainty_probe_path, map_location=device)
        )
        print(f"Successfully loaded uncertainty probe as MLPProbe from {uncertainty_probe_path}")
    except Exception as e:
            raise RuntimeError(f"Failed to load uncertainty probe from {uncertainty_probe_path}: {e}")
    
    uncertainty_probe.eval()

    value_probe = None
    try:
        value_probe = MLPProbe(hidden_size, hidden_dims=[256, 128]).to(device).to(model_dtype)
        value_probe.load_state_dict(
            torch.load(value_probe_path, map_location=device)
        )
        print(f"Successfully loaded value probe as MLPProbe from {value_probe_path}")
    except Exception as e:
            raise RuntimeError(f"Failed to load value probe from {value_probe_path}: {e}")
    
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