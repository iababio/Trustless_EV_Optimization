#!/usr/bin/env python3
"""
Simplified standalone demo for EV Charging Optimization research system.
This script can run with minimal dependencies and provides immediate results.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_data_if_needed():
    """Create sample EV charging data if it doesn't exist."""
    data_file = project_root / "local_charging_data.csv"
    
    if data_file.exists():
        print(f"✅ Found existing data file: {data_file}")
        return data_file
    
    print("📊 Creating sample EV charging data...")
    
    # Generate realistic sample data
    np.random.seed(42)
    
    # Create sample charging events with realistic patterns
    start_date = datetime(2018, 8, 24, 9, 50)
    data = []
    
    charger_names = ['charger_1', 'charger_2', 'charger_3', 'charger_4', 'charger_5', 'NA']
    
    for i in range(300):  # Generate 300 sample records
        # Realistic charging session parameters
        # Typical EV battery capacity: 20-100 kWh
        # Most sessions are partial charges
        base_energy = np.random.lognormal(mean=8, sigma=1)  # Log-normal distribution
        energy_consumed = max(50, min(base_energy, 50000))  # 50Wh to 50kWh
        
        meter_start = np.random.randint(0, 10000)
        meter_end = meter_start + energy_consumed
        meter_total = energy_consumed
        
        # Duration roughly correlated with energy (charging rate varies)
        # Typical charging rates: 3-22 kW for AC, 50-150 kW for DC
        charging_power = np.random.uniform(3000, 22000)  # 3-22 kW
        base_duration = (energy_consumed / charging_power) * 3600  # Convert to seconds
        duration = max(60, int(base_duration * np.random.uniform(0.8, 2.0)))  # Add variability
        
        # Timestamps with daily patterns (more usage during day)
        day_offset = i // 10  # Spread across multiple days
        hour_bias = np.random.choice([9, 10, 11, 14, 15, 16, 17, 18], p=[0.1, 0.15, 0.1, 0.15, 0.15, 0.15, 0.1, 0.1])
        minute_offset = np.random.randint(0, 60)
        
        from datetime import timedelta
        timestamp = start_date + timedelta(days=day_offset, hours=hour_bias-9, minutes=minute_offset)
        
        # Charger distribution (some chargers more popular)
        charger = np.random.choice(charger_names, p=[0.25, 0.25, 0.25, 0.15, 0.05, 0.05])
        
        data.append({
            'Start Time': timestamp.strftime('%d.%m.%Y %H:%M'),
            'Meter Start (Wh)': round(meter_start, 2),
            'Meter End(Wh)': round(meter_end, 2),
            'Meter Total(Wh)': round(meter_total, 2),
            'Total Duration (s)': duration,
            'Charger_name': charger
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(data_file, index=False)
    
    print(f"✅ Sample data created: {data_file} ({len(data)} records)")
    return data_file


def analyze_charging_data(data_file):
    """Perform basic analysis of EV charging data."""
    print("\n📊 Loading and analyzing EV charging data...")
    
    # Load data
    try:
        data = pd.read_csv(data_file)
        print(f"✅ Loaded {len(data)} charging records")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return {}
    
    # Basic statistics
    analysis = {}
    
    try:
        # Convert timestamp
        data['timestamp'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M', errors='coerce')
        
        # Basic metrics
        analysis['total_records'] = len(data)
        analysis['total_energy_kwh'] = data['Meter Total(Wh)'].sum() / 1000  # Convert to kWh
        analysis['avg_energy_per_session_kwh'] = data['Meter Total(Wh)'].mean() / 1000
        analysis['avg_duration_minutes'] = data['Total Duration (s)'].mean() / 60
        analysis['unique_chargers'] = data['Charger_name'].nunique()
        
        # Peak hours analysis
        data['hour'] = data['timestamp'].dt.hour
        hourly_usage = data.groupby('hour').size()
        peak_hours = hourly_usage.nlargest(3).index.tolist()
        analysis['peak_hours'] = peak_hours
        
        # Charger utilization
        charger_usage = data['Charger_name'].value_counts().head(5)
        analysis['top_chargers'] = charger_usage.to_dict()
        
        # Energy distribution
        energy_stats = data['Meter Total(Wh)'].describe()
        analysis['energy_distribution'] = {
            'min_kwh': energy_stats['min'] / 1000,
            'max_kwh': energy_stats['max'] / 1000,
            'median_kwh': energy_stats['50%'] / 1000,
            'std_kwh': energy_stats['std'] / 1000,
        }
        
        # Duration vs Energy correlation
        correlation = data['Meter Total(Wh)'].corr(data['Total Duration (s)'])
        analysis['energy_duration_correlation'] = correlation
        
    except Exception as e:
        print(f"⚠️  Analysis error: {e}")
    
    return analysis


def test_optional_components():
    """Test which optional components are available."""
    print("\n🔧 Testing optional ML components...")
    
    components = {}
    
    # Test PyTorch
    try:
        import torch
        components['torch'] = {
            'available': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
        print("✅ PyTorch available - Neural networks enabled")
    except ImportError:
        components['torch'] = {'available': False}
        print("⚠️  PyTorch not available - Neural networks disabled")
    
    # Test XGBoost
    try:
        import xgboost
        components['xgboost'] = {
            'available': True,
            'version': xgboost.__version__
        }
        print("✅ XGBoost available - Gradient boosting enabled")
    except ImportError:
        components['xgboost'] = {'available': False}
        print("⚠️  XGBoost not available - Gradient boosting disabled")
    
    # Test Scikit-learn
    try:
        import sklearn
        components['sklearn'] = {
            'available': True,
            'version': sklearn.__version__
        }
        print("✅ Scikit-learn available - Classical ML enabled")
    except ImportError:
        components['sklearn'] = {'available': False}
        print("⚠️  Scikit-learn not available - Classical ML disabled")
    
    # Test Federated Learning
    try:
        import flwr
        components['flwr'] = {
            'available': True,
            'version': flwr.__version__
        }
        print("✅ Flower available - Federated learning enabled")
    except ImportError:
        components['flwr'] = {'available': False}
        print("⚠️  Flower not available - Federated learning disabled")
    
    return components


def run_basic_ml_demo(data_file, components):
    """Run basic ML demonstrations with available components."""
    print("\n🧠 Running ML demonstrations...")
    
    results = {}
    
    # Load and prepare data
    try:
        data = pd.read_csv(data_file)
        
        # Simple feature engineering
        data['timestamp'] = pd.to_datetime(data['Start Time'], format='%d.%m.%Y %H:%M', errors='coerce')
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Select numeric features for ML
        features = ['hour', 'day_of_week', 'Meter Start (Wh)', 'Total Duration (s)']
        target = 'Meter Total(Wh)'
        
        # Remove rows with missing values
        ml_data = data[features + [target]].dropna()
        
        if len(ml_data) < 10:
            print("⚠️  Insufficient data for ML demonstrations")
            return results
        
        X = ml_data[features].values
        y = ml_data[target].values
        
        print(f"📊 ML dataset: {len(ml_data)} samples, {len(features)} features")
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return results
    
    # Test XGBoost if available
    if components.get('xgboost', {}).get('available', False):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            import xgboost as xgb
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)
            
            results['xgboost'] = {
                'rmse': rmse,
                'r2_score': r2,
                'feature_importance': dict(zip(features, model.feature_importances_))
            }
            
            print(f"✅ XGBoost model trained - R² = {r2:.3f}, RMSE = {rmse:.2f}")
            
        except Exception as e:
            print(f"⚠️  XGBoost demo failed: {e}")
    
    # Test scikit-learn if available
    if components.get('sklearn', {}).get('available', False):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mse)
            
            results['random_forest'] = {
                'rmse': rmse,
                'r2_score': r2,
                'feature_importance': dict(zip(features, model.feature_importances_))
            }
            
            print(f"✅ Random Forest model trained - R² = {r2:.3f}, RMSE = {rmse:.2f}")
            
        except Exception as e:
            print(f"⚠️  Random Forest demo failed: {e}")
    
    return results


def save_demo_results(analysis, components, ml_results):
    """Save demonstration results to files."""
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Compile all results
    demo_results = {
        'timestamp': timestamp,
        'data_analysis': analysis,
        'available_components': components,
        'ml_results': ml_results,
        'system_info': {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
        }
    }
    
    # Save JSON results
    results_file = results_dir / f"demo_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    # Save readable summary
    summary_file = results_dir / f"demo_summary_{timestamp}.md"
    with open(summary_file, 'w') as f:
        f.write(f"# EV Charging Optimization Demo Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data Analysis Summary
        f.write("## Data Analysis\n\n")
        if analysis:
            f.write(f"- **Total charging sessions:** {analysis.get('total_records', 'N/A')}\n")
            f.write(f"- **Total energy consumed:** {analysis.get('total_energy_kwh', 0):.2f} kWh\n")
            f.write(f"- **Average energy per session:** {analysis.get('avg_energy_per_session_kwh', 0):.2f} kWh\n")
            f.write(f"- **Average session duration:** {analysis.get('avg_duration_minutes', 0):.1f} minutes\n")
            f.write(f"- **Number of chargers:** {analysis.get('unique_chargers', 'N/A')}\n")
            f.write(f"- **Energy-duration correlation:** {analysis.get('energy_duration_correlation', 0):.3f}\n\n")
        
        # ML Results Summary
        if ml_results:
            f.write("## Machine Learning Results\n\n")
            for model_name, results in ml_results.items():
                f.write(f"### {model_name.title()}\n")
                f.write(f"- **R² Score:** {results.get('r2_score', 0):.3f}\n")
                f.write(f"- **RMSE:** {results.get('rmse', 0):.2f}\n")
                
                if 'feature_importance' in results:
                    f.write("- **Feature Importance:**\n")
                    for feature, importance in results['feature_importance'].items():
                        f.write(f"  - {feature}: {importance:.3f}\n")
                f.write("\n")
        
        # Component Availability
        f.write("## Available Components\n\n")
        for component, info in components.items():
            status = "✅" if info.get('available', False) else "❌"
            version = info.get('version', 'N/A')
            f.write(f"- {status} **{component.title()}**: {version}\n")
    
    print(f"✅ Results saved to:")
    print(f"   📄 {results_file}")
    print(f"   📄 {summary_file}")


def main():
    """Main demonstration function."""
    print("🚗⚡ EV Charging Optimization - Research Demo")
    print("=" * 55)
    
    try:
        # Step 1: Create or load data
        data_file = create_sample_data_if_needed()
        
        # Step 2: Analyze data
        analysis = analyze_charging_data(data_file)
        
        # Step 3: Test available components
        components = test_optional_components()
        
        # Step 4: Run ML demonstrations
        ml_results = run_basic_ml_demo(data_file, components)
        
        # Step 5: Save results
        save_demo_results(analysis, components, ml_results)
        
        # Summary
        print("\n🎉 Demo completed successfully!")
        
        if analysis:
            print(f"\n📊 Key Findings:")
            print(f"   • Analyzed {analysis.get('total_records', 0)} charging sessions")
            print(f"   • Total energy: {analysis.get('total_energy_kwh', 0):.2f} kWh")
            print(f"   • Average session: {analysis.get('avg_energy_per_session_kwh', 0):.2f} kWh")
            print(f"   • Peak hours: {analysis.get('peak_hours', [])}")
        
        if ml_results:
            print(f"\n🧠 ML Performance:")
            for model, results in ml_results.items():
                print(f"   • {model.title()}: R² = {results.get('r2_score', 0):.3f}")
        
        # Installation recommendations
        missing_components = [name for name, info in components.items() if not info.get('available', False)]
        if missing_components:
            print(f"\n💡 To unlock more features, install:")
            print(f"   pip install {' '.join(missing_components)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)