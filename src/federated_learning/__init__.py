"""Federated learning infrastructure."""

# Optional FL imports
try:
    from .client import EVChargingClient
    from .server import TrustlessStrategy, FederatedLearningServer
    __all__ = ["EVChargingClient", "TrustlessStrategy", "FederatedLearningServer"]
except ImportError:
    __all__ = []