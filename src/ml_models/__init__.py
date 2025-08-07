"""Lightweight ML models for edge deployment."""

from .base import EdgeMLModel

# Optional model imports
try:
    from .lstm import LightweightLSTM, AttentionLSTM
    __all__ = ["EdgeMLModel", "LightweightLSTM", "AttentionLSTM"]
except ImportError:
    __all__ = ["EdgeMLModel"]

try:
    from .xgboost_model import LightweightXGBoost
    __all__.append("LightweightXGBoost")
except ImportError:
    pass