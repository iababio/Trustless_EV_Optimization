"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import sys
import os
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.collector import MetricsCollector
from src.data_pipeline.processor import EVChargingDataProcessor


@pytest.fixture
def sample_ev_data():
    """Create sample EV charging data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic EV charging data
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15min'),
        'station_id': np.random.choice(['station_001', 'station_002', 'station_003'], n_samples),
        'meter_total_wh': np.random.exponential(5000, n_samples),  # Energy consumption
        'session_duration_min': np.random.gamma(2, 30, n_samples),  # Session duration
        'connector_type': np.random.choice(['Type1', 'Type2', 'CCS'], n_samples),
        'temperature': 20 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 96),  # Daily temperature cycle
    }
    
    df = pd.DataFrame(data)
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df


@pytest.fixture
def temp_data_file(sample_ev_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_ev_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def metrics_collector():
    """Create a metrics collector for testing."""
    return MetricsCollector(experiment_id="test_experiment", enable_prometheus=False)


@pytest.fixture
def data_processor(metrics_collector):
    """Create a data processor for testing."""
    return EVChargingDataProcessor(metrics_collector)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_model_weights():
    """Generate mock model weights for testing."""
    return {
        'layer1.weight': np.random.randn(32, 10).astype(np.float32),
        'layer1.bias': np.random.randn(32).astype(np.float32),
        'layer2.weight': np.random.randn(16, 32).astype(np.float32),
        'layer2.bias': np.random.randn(16).astype(np.float32),
        'output.weight': np.random.randn(1, 16).astype(np.float32),
        'output.bias': np.random.randn(1).astype(np.float32),
    }


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 5,
        'patience': 3,
        'test_data_samples': 100,
        'fl_clients': 3,
        'fl_rounds': 2,
    }