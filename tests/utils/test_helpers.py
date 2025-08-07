"""Test helper functions and utilities."""

import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

# Import optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def generate_ev_charging_data(
        n_samples: int = 1000,
        n_stations: int = 5,
        start_date: str = '2023-01-01',
        seed: int = 42
    ) -> pd.DataFrame:
        """Generate realistic EV charging data."""
        np.random.seed(seed)
        
        stations = [f'station_{i:03d}' for i in range(n_stations)]
        
        data = {
            'timestamp': pd.date_range(start_date, periods=n_samples, freq='15min'),
            'station_id': np.random.choice(stations, n_samples),
            'meter_total_wh': np.random.exponential(4000, n_samples),
            'session_duration_min': np.random.gamma(2, 25, n_samples),
            'connector_type': np.random.choice(['Type1', 'Type2', 'CCS', 'CHAdeMO'], n_samples),
            'temperature': 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 96) + np.random.normal(0, 2, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add temporal features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Add peak usage patterns
        peak_hours = [8, 9, 17, 18, 19]
        df['is_peak'] = df['hour_of_day'].isin(peak_hours)
        
        # Adjust energy consumption based on patterns
        df.loc[df['is_peak'], 'meter_total_wh'] *= np.random.uniform(1.2, 1.8, sum(df['is_peak']))
        df.loc[df['is_weekend'], 'meter_total_wh'] *= np.random.uniform(0.8, 1.2, sum(df['is_weekend']))
        
        return df
    
    @staticmethod
    def generate_federated_data(
        n_clients: int = 5,
        samples_per_client: int = 200,
        n_features: int = 10,
        iid: bool = True,
        seed: int = 42
    ) -> List[pd.DataFrame]:
        """Generate data distributed across federated clients."""
        np.random.seed(seed)
        
        client_datasets = []
        
        for client_id in range(n_clients):
            if not iid:
                # Non-IID: each client has different data distribution
                client_seed = seed + client_id * 100
                np.random.seed(client_seed)
                
                # Shift the data distribution for each client
                shift_factor = client_id * 0.5
                
                X = np.random.randn(samples_per_client, n_features) + shift_factor
                y = np.sum(X, axis=1) + np.random.randn(samples_per_client) * 0.1 + shift_factor * 10
            else:
                # IID: similar distribution across clients
                X = np.random.randn(samples_per_client, n_features)
                y = np.sum(X, axis=1) + np.random.randn(samples_per_client) * 0.1
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
            df['target'] = y
            df['client_id'] = f'client_{client_id:03d}'
            df['timestamp'] = pd.date_range(
                '2023-01-01', 
                periods=samples_per_client, 
                freq=f'{client_id+1}h'  # Different sampling frequencies
            )
            
            client_datasets.append(df)
        
        return client_datasets
    
    @staticmethod
    def generate_adversarial_data(
        base_data: pd.DataFrame,
        attack_type: str = 'poisoning',
        severity: float = 0.1
    ) -> pd.DataFrame:
        """Generate adversarial data for security testing."""
        adversarial_data = base_data.copy()
        
        if attack_type == 'poisoning':
            # Model poisoning: corrupt random samples
            n_corrupt = int(len(adversarial_data) * severity)
            corrupt_indices = np.random.choice(len(adversarial_data), n_corrupt, replace=False)
            
            # Flip targets for poisoning
            if 'meter_total_wh' in adversarial_data.columns:
                adversarial_data.loc[corrupt_indices, 'meter_total_wh'] *= -1
            
        elif attack_type == 'evasion':
            # Evasion attack: add noise to features
            numeric_columns = adversarial_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['timestamp']:
                    noise = np.random.normal(0, severity * adversarial_data[col].std(), len(adversarial_data))
                    adversarial_data[col] += noise
        
        elif attack_type == 'inference':
            # Privacy inference attack: try to extract sensitive info
            # Add synthetic sensitive attributes that could be inferred
            adversarial_data['inferred_income'] = adversarial_data['meter_total_wh'] * 0.001 + np.random.normal(50, 10, len(adversarial_data))
            adversarial_data['inferred_location_sensitivity'] = np.random.beta(2, 5, len(adversarial_data))
        
        return adversarial_data


class MockModels:
    """Mock ML models for testing without heavy dependencies."""
    
    class MockLSTM:
        """Mock LSTM model."""
        
        def __init__(self, input_size: int, hidden_size: int = 32, **kwargs):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.is_trained = False
            self.weights = np.random.randn(input_size * hidden_size)
        
        def forward(self, x):
            if HAS_TORCH:
                if isinstance(x, torch.Tensor):
                    batch_size, seq_len, _ = x.shape
                    return torch.randn(batch_size, 1)
                else:
                    return np.random.randn(x.shape[0], 1)
            else:
                return np.random.randn(x.shape[0], 1)
        
        def train_model(self, train_loader, val_loader=None, **kwargs):
            self.is_trained = True
            return {
                'total_epochs': 10,
                'final_train_loss': 0.1,
                'final_val_loss': 0.15,
                'total_training_time_seconds': 5.0
            }
        
        def predict(self, x):
            return self.forward(x)
        
        def get_model_summary(self):
            return {
                'model_name': 'MockLSTM',
                'parameters': {'total_parameters': len(self.weights)},
                'architecture': 'LSTM'
            }
    
    class MockXGBoost:
        """Mock XGBoost model."""
        
        def __init__(self, **kwargs):
            self.model = None
            self.is_trained = False
            self.feature_importance = {}
        
        def train(self, X, y, X_val=None, y_val=None):
            self.is_trained = True
            self.model = "trained_model"  # Mock trained model
            
            # Mock feature importance
            n_features = X.shape[1] if hasattr(X, 'shape') else 10
            self.feature_importance = {
                f'feature_{i}': np.random.random() 
                for i in range(n_features)
            }
            
            return {
                'training_time': 2.0,
                'final_train_score': 0.85,
                'final_val_score': 0.82 if X_val is not None else None
            }
        
        def predict(self, X, return_metrics=False):
            if not self.is_trained:
                raise ValueError("Model not trained")
            
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
            predictions = np.random.randn(n_samples) * 1000 + 5000  # Mock predictions
            
            if return_metrics:
                return predictions, {
                    'inference_time': 0.01,
                    'throughput': n_samples / 0.01
                }
            return predictions
        
        def evaluate(self, X, y):
            predictions = self.predict(X)
            
            # Mock evaluation metrics
            return {
                'r2_score': 0.75,
                'rmse': 1500.0,
                'mae': 1200.0
            }
        
        def get_feature_importance(self):
            return self.feature_importance


class TestEnvironment:
    """Test environment management utilities."""
    
    @staticmethod
    @contextmanager
    def temporary_directory():
        """Create and cleanup temporary directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            yield Path(temp_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @staticmethod
    @contextmanager
    def temporary_file(content: str = "", suffix: str = ".csv"):
        """Create and cleanup temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            yield temp_file
        finally:
            os.unlink(temp_file)
    
    @staticmethod
    @contextmanager
    def environment_variables(**env_vars):
        """Temporarily set environment variables."""
        old_env = {}
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
        
        try:
            yield
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
    
    @staticmethod
    def get_test_config() -> Dict[str, Any]:
        """Get standardized test configuration."""
        return {
            'batch_size': 16,
            'learning_rate': 0.001,
            'num_epochs': 3,  # Reduced for testing
            'patience': 2,
            'test_data_samples': 100,
            'fl_clients': 3,
            'fl_rounds': 2,
            'privacy_epsilon': 1.0,
            'random_seed': 42
        }


class AssertionHelpers:
    """Custom assertion helpers for testing."""
    
    @staticmethod
    def assert_dataframe_shape(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1):
        """Assert DataFrame has minimum shape requirements."""
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
        assert len(df) >= min_rows, f"DataFrame has {len(df)} rows, expected >= {min_rows}"
        assert len(df.columns) >= min_cols, f"DataFrame has {len(df.columns)} columns, expected >= {min_cols}"
    
    @staticmethod
    def assert_numeric_range(values, min_val=None, max_val=None):
        """Assert numeric values are within range."""
        if hasattr(values, 'values'):
            values = values.values
        
        values = np.asarray(values)
        finite_values = values[np.isfinite(values)]
        
        if min_val is not None:
            assert np.all(finite_values >= min_val), f"Found values below {min_val}: {finite_values.min()}"
        
        if max_val is not None:
            assert np.all(finite_values <= max_val), f"Found values above {max_val}: {finite_values.max()}"
    
    @staticmethod
    def assert_dict_contains(dictionary: Dict, required_keys: List[str]):
        """Assert dictionary contains required keys."""
        missing_keys = [key for key in required_keys if key not in dictionary]
        assert not missing_keys, f"Dictionary missing required keys: {missing_keys}"
    
    @staticmethod
    def assert_execution_time(duration: float, max_seconds: float):
        """Assert execution completed within time limit."""
        assert duration <= max_seconds, f"Execution took {duration:.2f}s, expected <= {max_seconds}s"
    
    @staticmethod
    def assert_model_performance(metrics: Dict, min_accuracy: float = 0.5):
        """Assert model performance meets minimum requirements."""
        if 'accuracy' in metrics:
            assert metrics['accuracy'] >= min_accuracy, f"Accuracy {metrics['accuracy']} below minimum {min_accuracy}"
        
        if 'r2_score' in metrics:
            # R² can be negative, but should be reasonable for test data
            assert metrics['r2_score'] > -2.0, f"R² score {metrics['r2_score']} is unreasonably low"
        
        if 'loss' in metrics:
            assert metrics['loss'] >= 0, f"Loss should be non-negative, got {metrics['loss']}"


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.benchmarks = {}
    
    @contextmanager
    def measure(self, operation_name: str):
        """Measure execution time of an operation."""
        import time
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.benchmarks[operation_name] = duration
    
    def get_benchmark(self, operation_name: str) -> Optional[float]:
        """Get benchmark time for an operation."""
        return self.benchmarks.get(operation_name)
    
    def get_all_benchmarks(self) -> Dict[str, float]:
        """Get all benchmark results."""
        return self.benchmarks.copy()
    
    def assert_performance(self, operation_name: str, max_seconds: float):
        """Assert operation completed within time limit."""
        duration = self.get_benchmark(operation_name)
        assert duration is not None, f"No benchmark found for {operation_name}"
        assert duration <= max_seconds, f"{operation_name} took {duration:.2f}s, expected <= {max_seconds}s"