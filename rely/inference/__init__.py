"""
Inference modules for the rely package.

This package contains inference-related functionality including UATS (Uncertainty-guided Approximate Tree Search)
and Budget Forcing inference strategies.
"""

from .uats import (
    UATSConfig,
    Branch,
    GuidedTreeSearch,
    create_uats_searcher,
    save_branches,
    load_model_and_tokenizer,
    run_uats
)

from .budget_forcing import (
    BudgetForcingConfig,
    BudgetForcingResult,
    BudgetForcingInference,
    run_budget_forcing_inference,
    create_budget_forcing_inference,
    save_budget_forcing_result
)

from .self_consistency import (
    SelfConsistencyConfig,
    SelfConsistencyResult,
    SelfConsistencyInference,
    run_self_consistency,
    save_self_consistency_result
)

from .sbs import (
    SBSConfig,
    StepBeamSearch
)

__all__ = [
    # UATS exports
    "UATSConfig",
    "Branch",
    "GuidedTreeSearch", 
    "run_uats",
    "create_uats_searcher",
    "save_branches",
    "load_model_and_tokenizer",
    # Budget Forcing exports
    "BudgetForcingConfig",
    "BudgetForcingResult",
    "BudgetForcingInference",
    "run_budget_forcing_inference",
    "create_budget_forcing_inference",
    "save_budget_forcing_result",
    # Self-Consistency exports
    "SelfConsistencyConfig",
    "SelfConsistencyResult",
    "SelfConsistencyInference",
    "run_self_consistency",
    "save_self_consistency_result",
    # SBS exports
    "SBSConfig",
    "StepBeamSearch"
]

__version__ = "1.0.0" 