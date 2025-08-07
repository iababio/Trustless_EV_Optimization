#!/usr/bin/env python3
"""
Installation script for EV Charging Optimization research system.
Handles optional dependencies gracefully for maximum compatibility.
"""

import subprocess
import sys
import importlib
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors gracefully."""
    print(f"📦 {description if description else cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Warning: {description} failed - {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please use Python 3.8 or higher")
        return False

def install_core_requirements():
    """Install core requirements that should always work."""
    core_requirements = [
        "torch>=1.13.0,<3.0.0",
        "numpy>=1.21.0,<2.0.0", 
        "pandas>=1.5.0,<3.0.0",
        "scikit-learn>=1.1.0,<2.0.0",
        "xgboost>=1.6.0,<3.0.0",
        "matplotlib>=3.5.0,<4.0.0",
        "psutil>=5.8.0,<6.0.0",
        "python-dotenv>=0.19.0,<2.0.0",
        "structlog>=21.0.0,<24.0.0",
    ]
    
    print("🔧 Installing core ML and data processing libraries...")
    for req in core_requirements:
        if not run_command(f"pip install '{req}'", f"Installing {req.split('>=')[0]}"):
            print(f"⚠️  Failed to install {req}, continuing with others...")
    
    return True

def install_federated_learning():
    """Install federated learning dependencies."""
    print("🌐 Installing federated learning libraries...")
    
    # Try Flower first
    if run_command("pip install 'flwr>=1.4.0,<2.0.0'", "Installing Flower FL framework"):
        print("✅ Federated learning support installed")
        return True
    else:
        print("⚠️  Flower installation failed - FL features may be limited")
        return False

def install_blockchain_optional():
    """Try to install blockchain dependencies (optional)."""
    print("🔗 Installing blockchain integration (optional)...")
    
    blockchain_requirements = [
        "web3>=6.0.0,<7.0.0",
        "eth-account>=0.8.0,<1.0.0", 
        "eth-utils>=2.0.0,<3.0.0",
    ]
    
    blockchain_success = True
    for req in blockchain_requirements:
        if not run_command(f"pip install '{req}'", f"Installing {req.split('>=')[0]}"):
            blockchain_success = False
            break
    
    if blockchain_success:
        print("✅ Blockchain integration installed")
        return True
    else:
        print("⚠️  Blockchain dependencies failed - will use mock validator")
        return False

def install_development_tools():
    """Install development and testing tools."""
    print("🛠️  Installing development tools...")
    
    dev_requirements = [
        "pytest>=6.2.0,<8.0.0",
        "black>=22.0.0,<24.0.0", 
        "mypy>=0.991,<2.0.0",
    ]
    
    for req in dev_requirements:
        run_command(f"pip install '{req}'", f"Installing {req.split('>=')[0]}")

def verify_installation():
    """Verify that key components can be imported."""
    print("🔍 Verifying installation...")
    
    core_imports = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"), 
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("matplotlib", "Matplotlib"),
    ]
    
    optional_imports = [
        ("flwr", "Flower FL"),
        ("web3", "Web3 blockchain"),
    ]
    
    # Test core imports
    all_core_success = True
    for module, name in core_imports:
        try:
            importlib.import_module(module)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ {name} import failed")
            all_core_success = False
    
    # Test optional imports
    for module, name in optional_imports:
        try:
            importlib.import_module(module)
            print(f"✅ {name} imported successfully (optional)")
        except ImportError:
            print(f"⚠️  {name} not available (optional)")
    
    return all_core_success

def create_demo_config():
    """Create configuration file for the demo."""
    config_content = '''# EV Charging Optimization Demo Configuration

# Core Settings
EXPERIMENT_ID=ev_demo_001
ENABLE_METRICS=true
ENABLE_PROMETHEUS=false

# Federated Learning Settings  
NUM_FL_CLIENTS=5
NUM_FL_ROUNDS=10
LOCAL_EPOCHS=3

# Model Settings
MODEL_TYPE=lstm  # or 'xgboost' or 'both'
ENABLE_EDGE_OPTIMIZATION=true

# Privacy Settings
ENABLE_PRIVACY=true
PRIVACY_EPSILON=1.0

# Blockchain Settings (optional)
ENABLE_BLOCKCHAIN=false
# ETHEREUM_NODE_URL=http://localhost:8545
# MODEL_VALIDATOR_CONTRACT_ADDRESS=
'''
    
    config_file = Path(".env.example")
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created configuration template: {config_file}")
    print("   Copy to .env and customize as needed")

def main():
    """Main installation process."""
    print("🚗⚡ EV Charging Optimization - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install components
    install_core_requirements()
    install_federated_learning()
    install_blockchain_optional()
    install_development_tools()
    
    # Verify installation
    if verify_installation():
        print("\n✅ Installation completed successfully!")
    else:
        print("\n⚠️  Installation completed with some issues")
        print("   Core functionality should still work")
    
    # Create demo configuration
    create_demo_config()
    
    print("\n🚀 Next steps:")
    print("   1. Copy .env.example to .env and customize settings")
    print("   2. Run: python examples/complete_demo.py")
    print("   3. Check results in the 'results/' directory")
    
    return True

if __name__ == "__main__":
    main()