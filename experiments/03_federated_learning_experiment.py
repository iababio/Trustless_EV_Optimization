#!/usr/bin/env python3
"""
Experiment 3: Federated Learning Simulation

This script implements and evaluates federated learning for EV charging demand prediction.
"""

import sys
import os
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from data_analysis.preprocessing.data_loader import EVChargingDataLoader
from federated_learning.models.baseline_models import LightweightLSTM
from federated_learning.simulation.federated_simulator import (
    FederatedChargingSimulator, ClientConfig, ClientType
)


def main():
    """Run federated learning experiment"""
    print("🔗 Experiment 3: Federated Learning Simulation")
    print("=" * 50)

    # Load and preprocess data
    data_path = '/Users/ababio/Lab/Research/EV_Optimization/updated_vehicle_data.csv'
    loader = EVChargingDataLoader(data_path)
    
    print("📊 Loading and preprocessing data...")
    raw_data = loader.load_raw_data()
    cleaned_data = loader.clean_data()
    processed_data = loader.engineer_features(cleaned_data)
    
    features, targets = loader.get_feature_target_split(processed_data)

    # Create federated data splits
    print("📡 Creating federated client data splits...")
    federated_data = loader.create_federated_splits(processed_data, n_clients=10)

    print(f"🏢 Created {len(federated_data)} federated clients:")
    for client_id, data in federated_data.items():
        manufacturers = data['Manufacturer'].nunique() if 'Manufacturer' in data.columns else 0
        print(f"   - Client {client_id}: {len(data)} samples, {manufacturers} manufacturers")

    # Initialize federated learning simulator
    print("\n🎯 Setting up federated learning environment...")

    # Create model architecture for federated learning
    feature_dim = len([col for col in features.columns if col not in ['Session_ID', 'Vehicle ID']])
    model_architecture = LightweightLSTM(input_size=feature_dim, hidden_size=128, num_layers=3)  # Increased capacity
    print(f"   - Model: LSTM with {feature_dim} input features, 128 hidden units, 3 layers")

    # Initialize simulator
    fl_simulator = FederatedChargingSimulator(
        model_architecture=model_architecture,
        aggregation_strategy="fedavg",
        random_seed=42
    )

    # Create clients from data
    numeric_features = [col for col in features.columns if col not in ['Session_ID', 'Vehicle ID']]
    client_configs = fl_simulator.create_clients_from_data(
        federated_data, 
        target_col='Meter Total(Wh)',
        feature_cols=numeric_features
    )

    print(f"✅ Federated learning setup completed with {len(client_configs)} clients")

    # Run federated learning simulation
    print("\n🚀 Running Federated Learning Simulation")
    print("=" * 50)

    # Configure realistic network conditions
    print("🌐 Configuring realistic network conditions...")
    fl_simulator.simulate_network_conditions(dropout_rate=0.1, slow_client_ratio=0.2)

    # Run simulation with proper training time
    print("🏃 Starting federated training simulation...")
    n_rounds = 50  # Balanced between performance and time
    client_fraction = 0.8  # 80% client participation per round

    print(f"   - Training for {n_rounds} rounds with {client_fraction*100}% client participation")
    print("   - This will take longer for better model performance...")
    print("   - Estimated time: 3-5 minutes for meaningful results")

    federated_results = fl_simulator.run_simulation(
        n_rounds=n_rounds,
        client_fraction=client_fraction
    )

    # Display results
    print("\n📊 Federated Learning Results:")
    final_metrics = federated_results['final_metrics']
    print(f"   - Final Loss: {final_metrics['final_loss']:.4f}")
    print(f"   - Final Accuracy: {final_metrics['final_accuracy']:.4f}")
    print(f"   - Total Communication Cost: {final_metrics['total_communication_cost']:.0f} MB")
    print(f"   - Average Round Duration: {final_metrics['avg_round_duration']:.2f} seconds")

    convergence_info = federated_results['convergence_analysis']
    if convergence_info['converged']:
        print(f"   - Converged at round: {convergence_info['convergence_round']}")
    else:
        print("   - Did not converge within the simulation period")

    # Display communication and privacy metrics
    comm_metrics = federated_results.get('communication_metrics', {})
    if comm_metrics:
        print(f"\n📡 Communication Metrics:")
        print(f"   - Total Bytes Transmitted: {comm_metrics.get('total_bytes', 0):,}")
        print(f"   - Average Round Communication: {comm_metrics.get('avg_round_bytes', 0):.0f} bytes")

    privacy_metrics = federated_results.get('privacy_metrics', {})
    if privacy_metrics:
        print(f"\n🔒 Privacy Metrics:")
        print(f"   - Privacy Budget Used: ε = {privacy_metrics.get('epsilon_used', 0):.2f}")
        print(f"   - Data Samples Protected: {privacy_metrics.get('samples_protected', 0):,}")

    print("\n✅ Federated learning experiment completed!")
    
    return fl_simulator, federated_results, federated_data


if __name__ == "__main__":
    main()