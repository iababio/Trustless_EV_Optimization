#!/usr/bin/env python3
"""
Experiment 2: Baseline Model Development and Evaluation

This script develops and evaluates baseline forecasting models for EV charging demand prediction.
"""

import sys
import os
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_analysis.preprocessing.data_loader import EVChargingDataLoader
from federated_learning.models.baseline_models import BaselineModelSuite


def main():
    """Run baseline models experiment"""
    print("🤖 Experiment 2: Baseline Model Development")
    print("=" * 50)

    # Load and preprocess data
    data_path = '/Users/ababio/Lab/Research/EV_Optimization/updated_vehicle_data.csv'
    loader = EVChargingDataLoader(data_path)
    
    print("📊 Loading and preprocessing data...")
    raw_data = loader.load_raw_data()
    cleaned_data = loader.clean_data()
    processed_data = loader.engineer_features(cleaned_data)
    
    # Prepare data for modeling
    features, targets = loader.get_feature_target_split(processed_data)
    print(f"📊 Features matrix: {features.shape}")
    print(f"🎯 Targets matrix: {targets.shape}")

    # Initialize baseline model suite
    baseline_suite = BaselineModelSuite(random_state=42)

    # Train all baseline models
    print("\n🏋️ Training baseline models...")
    baseline_results = baseline_suite.train_all_baselines(
        processed_data, 
        target_col='Meter Total(Wh)',
        train_split=0.7,
        val_split=0.2
    )

    # Display model comparison
    if baseline_results and 'models' in baseline_results:
        print("\n📊 Baseline Model Performance:")
        
        # Machine Learning models
        ml_models = baseline_results['models'].get('machine_learning', {})
        for model_name, model_info in ml_models.items():
            if 'error' not in model_info:
                train_rmse = model_info.get('train_rmse', 0)
                val_rmse = model_info.get('val_rmse', 0)
                print(f"   - {model_name}: Train RMSE={train_rmse:.3f}, Val RMSE={val_rmse:.3f}")
        
        # Deep Learning models
        dl_models = baseline_results['models'].get('deep_learning', {})
        lstm_info = dl_models.get('lstm', {})
        if 'error' not in lstm_info:
            train_rmse = lstm_info.get('train_rmse', 0)
            val_rmse = lstm_info.get('val_rmse', 0)
            print(f"   - LSTM: Train RMSE={train_rmse:.3f}, Val RMSE={val_rmse:.3f}")

        # Generate and display comparison table
        comparison_df = baseline_suite.get_model_comparison()
        print("\n📋 Model Comparison Table:")
        print(comparison_df.to_string(index=False))

    print("\n✅ Baseline model experiment completed!")
    
    return baseline_results


if __name__ == "__main__":
    main()