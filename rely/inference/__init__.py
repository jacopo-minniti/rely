"""
Inference modules for the rely package.

This package contains inference-related functionality including UATS (Uncertainty-guided Approximate Tree Search)
and Budget Forcing inference strategies.
"""

from .uats import (
    UATSConfig,
    Branch,
    GuidedTreeSearch,
    save_results,
    run_uats
)

from .majority_voting import (
    SelfConsistencyConfig,
    SelfConsistencyResult,
    SelfConsistencyInference,
    run_self_consistency,
    save_self_consistency_result
)

__all__ = [
    # UATS exports
    "UATSConfig",
    "Branch",
    "GuidedTreeSearch", 
    "run_uats",
    "save_results",
    # Self-Consistency exports
    "SelfConsistencyConfig",
    "SelfConsistencyResult",
    "SelfConsistencyInference",
    "run_self_consistency",
    "save_self_consistency_result",
]

__version__ = "1.0.0" 