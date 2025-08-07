#!/usr/bin/env python3
"""Health check script for EV Charging Optimization system."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run health check for all system components."""
    
    # Test core components
    try:
        from src.metrics.collector import MetricsCollector
        print('✅ Core components: OK')
    except Exception as e:
        print(f'❌ Core components: {e}')

    # Test PyTorch
    try:
        import torch
        print('✅ PyTorch: Available')
    except ImportError:
        print('⚠️ PyTorch: Not available')

    # Test XGBoost
    try:
        import xgboost
        print('✅ XGBoost: Available')
    except ImportError:
        print('⚠️ XGBoost: Not available')

    # Test Flower FL
    try:
        import flwr
        print('✅ Flower FL: Available')
    except ImportError:
        print('⚠️ Flower FL: Not available')

    # Test Web3.py
    try:
        from web3 import Web3
        print('✅ Web3.py: Available')
    except ImportError:
        print('⚠️ Web3.py: Not available')

if __name__ == "__main__":
    main()