"""Blockchain integration for trustless validation."""

# Optional blockchain imports
try:
    from .validator import ModelValidator, create_model_validator
    __all__ = ["ModelValidator", "create_model_validator"]
except ImportError:
    __all__ = []