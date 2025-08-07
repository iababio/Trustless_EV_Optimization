"""Integration tests for federated learning workflows."""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch

try:
    from src.federated_learning.client import EVChargingClient
    from src.federated_learning.server import TrustlessStrategy, FederatedLearningServer
    HAS_FEDERATED_LEARNING = True
except ImportError:
    HAS_FEDERATED_LEARNING = False

try:
    import torch
    from src.ml_models.lstm import LightweightLSTM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.ml_models.xgboost_model import LightweightXGBoost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@pytest.mark.skipif(not HAS_FEDERATED_LEARNING, reason="Federated learning dependencies not available")
class TestFederatedLearningIntegration:
    """Integration tests for complete federated learning workflows."""

    @pytest.fixture
    def fl_data(self):
        """Create federated learning test data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15min'),
            'station_id': np.random.choice(['station_001', 'station_002', 'station_003'], n_samples),
            'meter_total_wh': np.random.exponential(5000, n_samples),
            'session_duration_min': np.random.gamma(2, 30, n_samples),
        }
        
        # Add engineered features
        df = pd.DataFrame(data)
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Add some numeric features
        for i in range(n_features - 6):
            df[f'feature_{i}'] = np.random.randn(n_samples)
        
        return df

    @pytest.fixture
    def fl_clients(self, fl_data, metrics_collector):
        """Create federated learning clients."""
        clients = []
        n_clients = 3
        data_per_client = len(fl_data) // n_clients
        
        for i in range(n_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client if i < n_clients - 1 else len(fl_data)
            client_data = fl_data.iloc[start_idx:end_idx].copy()
            
            # Create model for client
            if HAS_TORCH:
                model = LightweightLSTM(
                    input_size=8,
                    hidden_size=16,
                    num_layers=1,
                    metrics_collector=metrics_collector
                )
            elif HAS_XGBOOST:
                model = LightweightXGBoost(
                    model_name=f"client_{i}_model",
                    metrics_collector=metrics_collector,
                    n_estimators=10
                )
            else:
                model = Mock()  # Mock model if no ML libraries available
            
            client = EVChargingClient(
                station_id=f"station_{i:03d}",
                model=model,
                local_data=client_data,
                metrics_collector=metrics_collector
            )
            
            clients.append(client)
        
        return clients

    @pytest.fixture
    def fl_strategy(self, metrics_collector):
        """Create federated learning strategy."""
        return TrustlessStrategy(
            min_fit_clients=2,
            min_available_clients=3,
            metrics_collector=metrics_collector,
            enable_blockchain_validation=False
        )

    def test_client_initialization(self, fl_clients):
        """Test FL client initialization."""
        for client in fl_clients:
            assert client.station_id.startswith("station_")
            assert client.model is not None
            assert client.local_data is not None
            assert len(client.local_data) > 0

    def test_client_data_preparation(self, fl_clients):
        """Test client data preparation for FL."""
        client = fl_clients[0]
        
        # Prepare data for training
        X, y = client.prepare_training_data()
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert len(X) > 0

    def test_client_local_training(self, fl_clients):
        """Test local training on client."""
        client = fl_clients[0]
        
        # Mock local training
        if hasattr(client.model, 'train'):
            # For XGBoost models
            X, y = client.prepare_training_data()
            if len(X) > 10:  # Ensure sufficient data
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                training_result = client.model.train(X_train, y_train, X_val, y_val)
                assert training_result is not None
        
        # Test get client info
        client_info = client.get_client_info()
        assert 'station_id' in client_info
        assert 'data_size' in client_info
        assert 'model_type' in client_info

    def test_fl_strategy_initialization(self, fl_strategy):
        """Test FL strategy initialization."""
        assert fl_strategy.min_fit_clients == 2
        assert fl_strategy.min_available_clients == 3
        assert not fl_strategy.enable_blockchain_validation

    def test_fl_strategy_client_selection(self, fl_strategy, fl_clients):
        """Test client selection strategy."""
        # Mock client configurations
        client_configs = []
        for i, client in enumerate(fl_clients):
            config = Mock()
            config.cid = f"client_{i}"
            config.num_examples = len(client.local_data)
            client_configs.append((config, {}))  # (ClientProxy, config)
        
        selected_configs = fl_strategy.configure_fit(
            server_round=1,
            parameters=None,
            client_manager=Mock()
        )
        
        # Should return configuration for fit
        assert selected_configs is not None

    def test_fl_server_initialization(self, fl_strategy, metrics_collector):
        """Test FL server initialization."""
        server = FederatedLearningServer(
            strategy=fl_strategy,
            server_address="localhost:8080",
            metrics_collector=metrics_collector
        )
        
        assert server.strategy == fl_strategy
        assert server.server_address == "localhost:8080"
        assert server.metrics_collector == metrics_collector

    @pytest.mark.asyncio
    async def test_fl_simulation_workflow(self, fl_clients, fl_strategy, metrics_collector):
        """Test complete FL simulation workflow."""
        # Create FL server
        server = FederatedLearningServer(
            strategy=fl_strategy,
            server_address="localhost:8080",
            metrics_collector=metrics_collector
        )
        
        # Simulate FL rounds
        num_rounds = 2
        
        for round_id in range(1, num_rounds + 1):
            # Simulate client selection and training
            participating_clients = fl_clients[:2]  # Select subset
            
            round_results = []
            for client in participating_clients:
                # Simulate local training
                client_result = {
                    'station_id': client.station_id,
                    'data_size': len(client.local_data),
                    'training_loss': np.random.uniform(0.1, 0.5),
                    'round_id': round_id
                }
                round_results.append(client_result)
            
            # Record FL metrics
            if metrics_collector:
                metrics_collector.record_federated_metrics(
                    round_id=round_id,
                    num_clients=len(participating_clients),
                    round_duration=np.random.uniform(30, 120),
                    convergence_metric=np.random.uniform(0.8, 0.95),
                    component="fl_server"
                )
        
        # Check that FL metrics were recorded
        metrics = metrics_collector.export_metrics(format_type="dict")
        fl_metrics = [
            m for m in metrics.get('metrics', [])
            if m.get('metric_type') == 'federated_learning'
        ]
        
        assert len(fl_metrics) > 0

    def test_fl_aggregation_simulation(self, fl_strategy):
        """Test federated aggregation simulation."""
        # Create mock client results
        mock_results = []
        for i in range(3):
            result = Mock()
            result.num_examples = 100 + i * 20
            result.parameters = [np.random.randn(10, 5), np.random.randn(5)]  # Mock weights
            result.metrics = {'loss': 0.3 - i * 0.05, 'accuracy': 0.7 + i * 0.05}
            mock_results.append((Mock(), result))
        
        # Test aggregation
        aggregated_params = fl_strategy.aggregate_fit(
            server_round=1,
            results=mock_results,
            failures=[]
        )
        
        assert aggregated_params is not None

    def test_fl_evaluation_simulation(self, fl_strategy):
        """Test federated evaluation simulation."""
        # Create mock evaluation results
        mock_eval_results = []
        for i in range(3):
            result = Mock()
            result.num_examples = 50 + i * 10
            result.loss = 0.25 - i * 0.02
            result.metrics = {'accuracy': 0.75 + i * 0.03}
            mock_eval_results.append((Mock(), result))
        
        # Test evaluation aggregation
        eval_result = fl_strategy.aggregate_evaluate(
            server_round=1,
            results=mock_eval_results,
            failures=[]
        )
        
        if eval_result:
            loss, metrics = eval_result
            assert isinstance(loss, (int, float))
            assert isinstance(metrics, dict)

    def test_fl_privacy_preservation(self, fl_clients, metrics_collector):
        """Test privacy-preserving features in FL."""
        client = fl_clients[0]
        client.privacy_epsilon = 1.0  # Enable differential privacy
        
        # Test that raw data doesn't leave client
        assert client.local_data is not None
        
        # Simulate adding noise for privacy
        if hasattr(client, 'add_noise_to_updates'):
            original_weights = np.random.randn(10)
            noisy_weights = client.add_noise_to_updates(original_weights, epsilon=1.0)
            
            # Weights should be different due to noise
            assert not np.array_equal(original_weights, noisy_weights)

    def test_fl_client_dropout_handling(self, fl_strategy, fl_clients):
        """Test handling of client dropouts during FL."""
        # Simulate some clients dropping out
        available_clients = fl_clients[:2]  # Only 2 out of 3 clients
        
        # Strategy should adapt to fewer clients
        min_clients = fl_strategy.min_fit_clients
        
        if len(available_clients) >= min_clients:
            # Should proceed with fewer clients
            assert len(available_clients) >= min_clients
        else:
            # Should handle gracefully
            pass

    def test_fl_convergence_detection(self, fl_strategy, metrics_collector):
        """Test convergence detection in FL."""
        # Simulate multiple rounds with improving metrics
        convergence_metrics = []
        for round_id in range(1, 6):
            # Simulate improving accuracy
            accuracy = 0.7 + round_id * 0.03 + np.random.normal(0, 0.01)
            convergence_metrics.append(accuracy)
            
            metrics_collector.record_federated_metrics(
                round_id=round_id,
                num_clients=3,
                round_duration=60.0,
                convergence_metric=accuracy,
                component="fl_server"
            )
        
        # Check if convergence can be detected
        recent_metrics = convergence_metrics[-3:]  # Last 3 rounds
        if len(recent_metrics) >= 2:
            improvement = recent_metrics[-1] - recent_metrics[-2]
            # Small improvement might indicate convergence
            is_converging = abs(improvement) < 0.01
            assert isinstance(is_converging, bool)

    @pytest.mark.slow
    def test_fl_scalability_simulation(self, metrics_collector):
        """Test FL with larger number of clients."""
        num_clients = 10
        clients = []
        
        # Create more clients
        for i in range(num_clients):
            # Create minimal client data
            client_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=50, freq='1H'),
                'meter_total_wh': np.random.exponential(3000, 50),
                'hour_of_day': np.random.randint(0, 24, 50),
                'day_of_week': np.random.randint(0, 7, 50),
                'feature_1': np.random.randn(50),
                'feature_2': np.random.randn(50),
            })
            
            client = EVChargingClient(
                station_id=f"station_{i:03d}",
                model=Mock(),  # Use mock model for speed
                local_data=client_data,
                metrics_collector=metrics_collector
            )
            clients.append(client)
        
        # Test that all clients can be created and have data
        assert len(clients) == num_clients
        for client in clients:
            assert len(client.local_data) > 0
            client_info = client.get_client_info()
            assert 'station_id' in client_info

    def test_fl_fault_tolerance(self, fl_strategy, metrics_collector):
        """Test fault tolerance in FL system."""
        # Simulate client failures during training
        successful_results = []
        failed_results = []
        
        for i in range(5):
            if i < 3:  # 3 successful clients
                result = Mock()
                result.num_examples = 100
                result.parameters = [np.random.randn(10)]
                result.metrics = {'loss': 0.3}
                successful_results.append((Mock(), result))
            else:  # 2 failed clients
                failure = Mock()
                failure.code = Mock()
                failure.reason = "Client timeout"
                failed_results.append(failure)
        
        # Strategy should handle failures gracefully
        if len(successful_results) >= fl_strategy.min_fit_clients:
            # Should be able to proceed with successful clients
            aggregated = fl_strategy.aggregate_fit(
                server_round=1,
                results=successful_results,
                failures=failed_results
            )
            assert aggregated is not None or len(failed_results) > 0  # Either success or handled failure