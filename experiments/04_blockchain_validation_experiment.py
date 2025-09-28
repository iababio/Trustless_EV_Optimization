#!/usr/bin/env python3
"""
Experiment 4: Blockchain-Based Model Validation

This script implements and evaluates blockchain-based validation for federated learning models.
"""

import sys
import os
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_analysis.preprocessing.data_loader import EVChargingDataLoader
from federated_learning.models.baseline_models import LightweightLSTM
from federated_learning.simulation.federated_simulator import FederatedChargingSimulator
from blockchain.validation.blockchain_validator import (
    FederatedBlockchainIntegration, MockBlockchainValidator
)


def run_federated_simulation():
    """Setup and run a basic federated learning simulation for blockchain validation"""
    print("🔗 Setting up federated learning for blockchain validation...")
    
    # Load and preprocess data
    data_path = '/Users/ababio/Lab/Research/EV_Optimization/updated_vehicle_data.csv'
    loader = EVChargingDataLoader(data_path)
    
    raw_data = loader.load_raw_data()
    cleaned_data = loader.clean_data()
    processed_data = loader.engineer_features(cleaned_data)
    features, targets = loader.get_feature_target_split(processed_data)

    # Create federated data splits
    federated_data = loader.create_federated_splits(processed_data, n_clients=5)
    
    # Initialize federated learning simulator
    feature_dim = len([col for col in features.columns if col not in ['Session_ID', 'Vehicle ID']])
    model_architecture = LightweightLSTM(input_size=feature_dim, hidden_size=64, num_layers=2)  # Improved capacity

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

    # Run proper simulation for meaningful results
    print("   - Training for 50 rounds for meaningful model performance...")
    federated_results = fl_simulator.run_simulation(n_rounds=50, client_fraction=0.8)
    
    return fl_simulator, federated_results, federated_data


def main():
    """Run blockchain validation experiment"""
    print("⛓️ Experiment 4: Blockchain-Based Model Validation")
    print("=" * 50)

    # Setup federated learning simulation
    fl_simulator, federated_results, federated_data = run_federated_simulation()
    
    print(f"✅ Federated simulation completed with {len(federated_results['round_history'])} rounds")

    # Initialize blockchain validator (using mock for demonstration)
    print("\n🔗 Setting up blockchain validation system...")
    blockchain_integration = FederatedBlockchainIntegration(use_mock=True)

    # Register clients in blockchain
    print("📝 Registering federated clients...")
    for client_id in federated_data.keys():
        result = blockchain_integration.validator.register_client(f"client_{client_id}")
        print(f"   - Client {client_id}: {'✅ Registered' if result else '❌ Failed'}")

    # Simulate blockchain validation for federated rounds
    print("\n⚡ Simulating blockchain validation...")
    blockchain_validations = []

    for i, round_result in enumerate(federated_results['round_history']):
        # Validate round with blockchain
        success, message = blockchain_integration.validate_federated_round(
            fl_simulator.global_model,
            [{'client_id': cid} for cid in round_result.participating_clients],
            {
                'accuracy': round_result.global_accuracy,
                'loss': round_result.global_loss
            }
        )
        
        blockchain_validations.append({
            'round': i + 1,
            'success': success,
            'message': message,
            'accuracy': round_result.global_accuracy,
            'participants': len(round_result.participating_clients)
        })

    # Display blockchain validation results
    print("\n📋 Blockchain Validation Results:")
    successful_validations = 0
    for validation in blockchain_validations:
        status = "✅ VALIDATED" if validation['success'] else "❌ REJECTED"
        print(f"   - Round {validation['round']}: {status} (Accuracy: {validation['accuracy']:.3f})")
        if validation['success']:
            successful_validations += 1

    print(f"\n📊 Validation Summary:")
    print(f"   - Total rounds validated: {len(blockchain_validations)}")
    print(f"   - Successful validations: {successful_validations}")
    print(f"   - Success rate: {successful_validations/len(blockchain_validations)*100:.1f}%")

    # Get system security metrics
    security_metrics = blockchain_integration.get_system_security_metrics()
    print("\n🛡️ System Security Status:")
    print(f"   - Recent Validations: {security_metrics['recent_validations']}")
    print(f"   - Average Trust Score: {security_metrics['avg_trust_score']:.3f}")
    print(f"   - System Integrity: {'✅ Secure' if security_metrics['avg_trust_score'] > 0.8 else '⚠️ Monitor'}")

    # Test different attack scenarios
    print("\n🔍 Testing Attack Scenarios:")
    
    # Simulate malicious model update
    print("   - Testing malicious model detection...")
    malicious_metrics = {'accuracy': 0.1, 'loss': 10.0}  # Clearly malicious
    success, message = blockchain_integration.validate_federated_round(
        fl_simulator.global_model,
        [{'client_id': 'malicious_client'}],
        malicious_metrics
    )
    print(f"     Malicious update: {'❌ BLOCKED' if not success else '⚠️ PASSED'}")
    
    # Test consensus mechanism
    print("   - Testing consensus mechanism...")
    consensus_results = blockchain_integration.validator.validate_consensus(
        [0.85, 0.87, 0.86, 0.84, 0.88],  # Normal accuracies
        threshold=0.8
    )
    print(f"     Consensus validation: {'✅ PASSED' if consensus_results else '❌ FAILED'}")

    print("\n✅ Blockchain validation experiment completed!")
    
    return blockchain_integration, blockchain_validations


if __name__ == "__main__":
    main()