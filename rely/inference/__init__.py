"""
Inference modules for the rely package.

This package contains inference-related functionality including AUTS (Approximate Uncertainty-guided Tree Search)
and Budget Forcing inference strategies.
"""

from .auts import (
    AUTSConfig,
    Branch,
    GuidedTreeSearch,
    run_auts_search,
    create_auts_searcher,
    save_branches,
    load_model_and_tokenizer
)

from .auts.budget_forcing import (
    BudgetForcingConfig,
    BudgetForcingResult,
    BudgetForcingInference,
    run_budget_forcing_inference,
    create_budget_forcing_inference,
    save_budget_forcing_result
)

__all__ = [
    # AUTS exports
    "AUTSConfig",
    "Branch",
    "GuidedTreeSearch", 
    "run_auts_search",
    "create_auts_searcher",
    "save_branches",
    "load_model_and_tokenizer",
    # Budget Forcing exports
    "BudgetForcingConfig",
    "BudgetForcingResult",
    "BudgetForcingInference",
    "run_budget_forcing_inference",
    "create_budget_forcing_inference",
    "save_budget_forcing_result"
]

__version__ = "1.0.0" 