# Expose main API for rely.complete
from .completer_offline import Completer, CompleterConfig
from .generate import generate_from_dataset

__all__ = [
    "Completer",
    "generate_from_dataset",
    "CompleterConfig",
] 