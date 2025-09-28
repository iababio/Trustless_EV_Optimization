#!/usr/bin/env python3
"""
Experiment 1: Data Analysis and Exploratory Data Analysis

This script performs comprehensive data analysis and visualization of the EV charging dataset.
"""

import sys
import os
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from data_analysis.preprocessing.data_loader import EVChargingDataLoader
from data_analysis.eda.visualization_suite import EVChargingVisualizer


def main():
    """Run data analysis experiment"""
    print("📊 Experiment 1: Data Analysis and EDA")
    print("=" * 50)

    # Initialize data loader
    data_path = '/Users/ababio/Lab/Research/EV_Optimization/updated_vehicle_data.csv'
    loader = EVChargingDataLoader(data_path)

    # Load and analyze data
    print("📈 Loading dataset...")
    raw_data = loader.load_raw_data()
    print(f"Dataset loaded: {raw_data.shape[0]} records, {raw_data.shape[1]} features")

    # Analyze missing values
    print("\n🔍 Analyzing missing values...")
    missing_analysis = loader.analyze_missing_values()
    print("Top 10 features with missing values:")
    for col, pct in list(missing_analysis.items())[:10]:
        if pct > 0:
            print(f"  - {col}: {pct:.1f}% missing")

    # Clean and process data
    print("\n🧹 Cleaning and preprocessing data...")
    cleaned_data = loader.clean_data()
    processed_data = loader.engineer_features(cleaned_data)

    print(f"✅ Data preprocessing completed")
    print(f"   - Original shape: {raw_data.shape}")
    print(f"   - Processed shape: {processed_data.shape}")
    print(f"   - Features engineered: {processed_data.shape[1] - raw_data.shape[1]}")

    # Generate comprehensive EDA visualizations
    print("\n📊 Generating comprehensive EDA visualizations...")
    visualizer = EVChargingVisualizer(processed_data)
    eda_figures = visualizer.generate_all_visualizations()

    print(f"Generated {len(eda_figures)} visualization categories:")
    for category in eda_figures.keys():
        print(f"   - {category}")

    # Display summary statistics
    summary_stats = visualizer.generate_summary_statistics()
    print("\n📈 Dataset Summary:")
    print(f"   - Total records: {summary_stats['dataset_overview']['total_records']:,}")
    print(f"   - Unique vehicles: {summary_stats['dataset_overview']['unique_vehicles']:,}")
    print(f"   - Unique sessions: {summary_stats['dataset_overview']['unique_sessions']:,}")

    # Show first visualization if available
    if 'temporal_overview' in eda_figures:
        eda_figures['temporal_overview']['plotly_figure'].show()
        print("👆 Temporal Overview - showing charging patterns over time")

    print("\n✅ Data analysis experiment completed!")
    
    return processed_data, eda_figures


if __name__ == "__main__":
    main()