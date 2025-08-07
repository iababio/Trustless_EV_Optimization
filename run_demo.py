#!/usr/bin/env python3
"""
Simple launcher for the EV Charging Optimization demo.
Automatically handles dependencies and runs the demonstration.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("🔍 Checking dependencies...")
    
    # Core dependencies that should always work
    core_deps = {
        'numpy': 'numpy>=1.21.0',
        'pandas': 'pandas>=1.5.0', 
        'matplotlib': 'matplotlib>=3.5.0',
        'psutil': 'psutil>=5.8.0',
    }
    
    # Optional dependencies
    optional_deps = {
        'torch': 'torch>=1.13.0',
        'xgboost': 'xgboost>=1.6.0',
        'sklearn': 'scikit-learn>=1.1.0',
    }
    
    missing_core = []
    missing_optional = []
    
    # Check core dependencies
    for package, pip_name in core_deps.items():
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_core.append(pip_name)
            print(f"❌ {package} is missing")
    
    # Check optional dependencies
    for package, pip_name in optional_deps.items():
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_optional.append(pip_name)
            print(f"⚠️  {package} is missing (optional)")
    
    # Install missing core dependencies
    if missing_core:
        print(f"\n📦 Installing core dependencies: {', '.join(missing_core)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_core)
            print("✅ Core dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install core dependencies")
            return False
    
    # Try to install optional dependencies
    if missing_optional:
        print(f"\n📦 Attempting to install optional dependencies...")
        for dep in missing_optional:
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', dep
                ], timeout=60)
                print(f"✅ {dep} installed successfully")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(f"⚠️  {dep} installation failed - continuing without it")
    
    return True

def create_sample_data():
    """Create sample EV charging data if it doesn't exist."""
    data_file = Path("local_charging_data.csv")
    
    if data_file.exists():
        print("✅ Sample data file already exists")
        return True
    
    print("📊 Creating sample EV charging data...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample data
        np.random.seed(42)
        
        # Create sample charging events
        start_date = datetime(2018, 8, 24, 9, 50)
        data = []
        
        for i in range(200):  # Generate 200 sample records
            # Random charging session
            meter_start = np.random.randint(50, 1000)
            energy_consumed = np.random.exponential(500)  # Exponential distribution for energy
            meter_end = meter_start + energy_consumed
            meter_total = energy_consumed
            
            # Duration roughly correlated with energy but with noise
            base_duration = energy_consumed * 2 + np.random.normal(0, 200)
            duration = max(int(abs(base_duration)), 10)  # Minimum 10 seconds
            
            # Timestamp
            timestamp = start_date + timedelta(hours=i*2 + np.random.uniform(-1, 1))
            
            # Charger name
            charger_names = ['charger_1', 'charger_2', 'charger_3', 'NA']
            charger = np.random.choice(charger_names, p=[0.3, 0.3, 0.3, 0.1])
            
            data.append({
                'Start Time': timestamp.strftime('%d.%m.%Y %H:%M'),
                'Meter Start (Wh)': meter_start,
                'Meter End(Wh)': meter_end,
                'Meter Total(Wh)': meter_total,
                'Total Duration (s)': duration,
                'Charger_name': charger
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(data_file, index=False)
        
        print(f"✅ Sample data created: {data_file} ({len(data)} records)")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def run_demo():
    """Run the main demonstration."""
    print("\n🚗⚡ Starting EV Charging Optimization Demo")
    print("=" * 50)
    
    try:
        # Import and run the demo
        import sys
        from pathlib import Path
        
        # Add examples directory to path
        examples_dir = Path(__file__).parent / "examples"
        sys.path.insert(0, str(examples_dir))
        
        from complete_demo import main
        import asyncio
        
        # Run the async demo
        result = asyncio.run(main())
        return result
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔧 Trying simplified version...")
        
        # Fallback to simplified demo
        return run_simplified_demo()

def run_simplified_demo():
    """Run a simplified version of the demo with minimal dependencies."""
    print("🔧 Running simplified demo with available components...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import sys
        from pathlib import Path
        
        # Add project to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        # Load and process data
        data_file = "local_charging_data.csv"
        if not Path(data_file).exists():
            print("❌ No data file found")
            return False
        
        print("📊 Loading EV charging data...")
        data = pd.read_csv(data_file)
        print(f"✅ Loaded {len(data)} charging records")
        
        # Basic data analysis
        print("\n📈 Basic Data Analysis:")
        print(f"   • Total records: {len(data)}")
        print(f"   • Average energy per session: {data['Meter Total(Wh)'].mean():.2f} Wh")
        print(f"   • Average session duration: {data['Total Duration (s)'].mean():.0f} seconds")
        
        # Show unique chargers
        unique_chargers = data['Charger_name'].value_counts()
        print(f"   • Unique chargers: {len(unique_chargers)}")
        for charger, count in unique_chargers.head(3).items():
            print(f"     - {charger}: {count} sessions")
        
        print("\n✅ Simplified demo completed successfully!")
        print("   📁 For full functionality, install missing dependencies:")
        print("      pip install torch xgboost scikit-learn flwr")
        
        return True
        
    except Exception as e:
        print(f"❌ Simplified demo failed: {e}")
        return False

def main():
    """Main entry point."""
    print("🚗⚡ EV Charging Optimization - Demo Launcher")
    print("=" * 55)
    
    # Check Python version
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    else:
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   Please use Python 3.8 or higher")
        return False
    
    # Check and install dependencies
    if not check_and_install_dependencies():
        print("❌ Dependency setup failed")
        return False
    
    # Create sample data if needed
    if not create_sample_data():
        print("❌ Sample data creation failed")
        return False
    
    # Run the demonstration
    success = run_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("   📁 Check the 'results/' directory for detailed results")
    else:
        print("\n❌ Demo encountered issues")
        print("   💡 Try running: python install.py")
    
    return success

if __name__ == "__main__":
    main()