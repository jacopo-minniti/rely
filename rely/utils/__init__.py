"""
Utility modules for the rely package.

This package contains various utility functions and classes for data loading,
text processing, probe management, dataset analysis, and more.
"""

# Import from load.py
from .load import (
    load_dataset,
    save_dataset,
    validate_file_format
)

# Import from merge.py
from .merge import merge

# Import from show.py
from .show import (
    show_fields,
    show_first_n,
    show_summary
)

# Import from entropy-stats.py
from .entropy_stats import (
    validate_entropy_field,
    basic_entropy_stats,
    print_entropy_stats,
    plot_entropy_distribution,
    analyze_entropy_by_category,
    print_category_analysis,
    plot_category_analysis,
    compare_datasets,
    print_dataset_comparison,
    plot_dataset_comparison,
    entropy_threshold_analysis,
    print_threshold_analysis,
    plot_threshold_analysis,
    ensure_figures_directory,
    list_saved_figures,
    clear_figures_directory,
    analyze_entropy_dataset,
    comprehensive_entropy_analysis
)

# Probes utilities moved to rely.inference.uats.probes but keep backward compatibility
try:
    from .probes import (
        load_probes,
        convert_isotropy_to_branches,
        MLPProbe,
    )
except ModuleNotFoundError:  # probes module relocated
    from importlib import import_module

    _probes_module = import_module("rely.inference.uats.probes")

    load_probes = _probes_module.load_probes
    convert_isotropy_to_branches = _probes_module.convert_isotropy_to_branches
    MLPProbe = _probes_module.MLPProbe

# Import from text_utils.py
from .text_utils import (
    get_last_step_pos,
    count_tokens_after_marker,
    format_system_prompt,
    ensure_think_ending,
    MMLU_SYSTEM_PROMPT
)

__all__ = [
    # Load utilities
    "load_dataset",
    "save_dataset", 
    "validate_file_format",
    
    # Merge utilities
    "merge",
    
    # Show utilities
    "show_fields",
    "show_first_n",
    "show_summary",
    
    # Entropy statistics utilities
    "validate_entropy_field",
    "basic_entropy_stats",
    "print_entropy_stats",
    "plot_entropy_distribution",
    "analyze_entropy_by_category",
    "print_category_analysis",
    "plot_category_analysis",
    "compare_datasets",
    "print_dataset_comparison",
    "plot_dataset_comparison",
    "entropy_threshold_analysis",
    "print_threshold_analysis",
    "plot_threshold_analysis",
    "ensure_figures_directory",
    "list_saved_figures",
    "clear_figures_directory",
    "analyze_entropy_dataset",
    "comprehensive_entropy_analysis",
    
    # Probe utilities
    "UncertaintyProbe",
    "ValueProbe",
    "load_probes",
    "convert_isotropy_to_branches",
    
    # Text utilities
    "get_last_step_pos",
    "count_tokens_after_marker",
    "format_system_prompt",
    "ensure_think_ending",
    "MMLU_SYSTEM_PROMPT"
]

__version__ = "1.0.0" 