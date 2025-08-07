"""Unit tests for LSTM models."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

try:
    from src.ml_models.lstm import LightweightLSTM, AttentionLSTM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestLightweightLSTM:
    """Test suite for LightweightLSTM."""

    @pytest.fixture
    def sample_model(self, metrics_collector):
        """Create a sample LSTM model for testing."""
        return LightweightLSTM(
            input_size=10,
            hidden_size=16,
            num_layers=2,
            metrics_collector=metrics_collector
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        batch_size, seq_len, input_size = 4, 8, 10
        X = torch.randn(batch_size, seq_len, input_size)
        y = torch.randn(batch_size, 1)
        return X, y

    def test_model_initialization(self, sample_model):
        """Test LSTM model initialization."""
        assert sample_model.input_size == 10
        assert sample_model.hidden_size == 16
        assert sample_model.num_layers == 2
        assert hasattr(sample_model, 'lstm')
        assert hasattr(sample_model, 'output_layers')

    def test_forward_pass(self, sample_model, sample_data):
        """Test forward pass through the model."""
        X, _ = sample_data
        
        with torch.no_grad():
            output = sample_model.forward(X)
        
        assert output.shape == (4, 1)  # batch_size, output_size
        assert not torch.isnan(output).any()

    def test_hidden_state_initialization(self, sample_model):
        """Test hidden state initialization."""
        batch_size = 4
        device = torch.device('cpu')
        
        h0, c0 = sample_model.init_hidden(batch_size, device)
        
        expected_shape = (sample_model.num_layers, batch_size, sample_model.hidden_size)
        assert h0.shape == expected_shape
        assert c0.shape == expected_shape
        assert h0.device == device
        assert c0.device == device

    def test_predict_sequence(self, sample_model, sample_data):
        """Test multi-step sequence prediction."""
        X, _ = sample_data
        
        with torch.no_grad():
            results = sample_model.predict_sequence(X, future_steps=3)
        
        assert 'predictions' in results
        predictions = results['predictions']
        assert predictions.shape == (4, 3, 1)  # batch_size, future_steps, output_size

    def test_predict_sequence_with_attention(self, sample_model, sample_data):
        """Test sequence prediction with attention weights."""
        X, _ = sample_data
        
        with torch.no_grad():
            results = sample_model.predict_sequence(X, future_steps=2, return_attention=True)
        
        assert 'predictions' in results
        assert 'attention_weights' in results
        
        attention = results['attention_weights']
        assert attention.shape == (4, 2)  # batch_size, future_steps

    def test_model_config(self, sample_model):
        """Test model configuration retrieval."""
        config = sample_model.get_model_config()
        
        expected_keys = ['input_size', 'hidden_size', 'num_layers', 'output_size', 'dropout', 'bidirectional']
        for key in expected_keys:
            assert key in config
        
        assert config['input_size'] == 10
        assert config['hidden_size'] == 16

    def test_parameter_count(self, sample_model):
        """Test parameter counting."""
        param_info = sample_model.count_parameters()
        
        assert 'total_parameters' in param_info
        assert 'trainable_parameters' in param_info
        assert param_info['total_parameters'] > 0
        assert param_info['trainable_parameters'] > 0

    def test_model_summary(self, sample_model):
        """Test model summary generation."""
        summary = sample_model.get_model_summary()
        
        assert 'model_name' in summary
        assert 'architecture' in summary
        assert 'parameters' in summary
        assert summary['model_name'] == 'LightweightLSTM'

    def test_training_with_data_loaders(self, sample_model, sample_data, test_config):
        """Test model training with data loaders."""
        X, y = sample_data
        
        # Create simple data loaders
        train_dataset = torch.utils.data.TensorDataset(X, y)
        val_dataset = torch.utils.data.TensorDataset(X, y)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
        
        # Train for a few epochs
        training_summary = sample_model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=0.01
        )
        
        assert 'total_epochs' in training_summary
        assert 'final_train_loss' in training_summary
        assert 'total_training_time_seconds' in training_summary
        assert training_summary['total_epochs'] <= 2

    def test_edge_optimization(self, sample_model):
        """Test edge deployment optimization."""
        original_params = sample_model.count_parameters()['total_parameters']
        
        # Apply optimization
        optimization_report = sample_model.optimize_for_edge(
            pruning_ratio=0.3,
            quantize=False  # Skip quantization for testing
        )
        
        assert 'pruning_applied' in optimization_report
        assert 'original_size' in optimization_report
        assert 'optimized_size' in optimization_report

    def test_inference_time_measurement(self, sample_model, sample_data):
        """Test inference time measurement."""
        X, _ = sample_data
        sample_input = X[:1]  # Single sample
        
        stats = sample_model.measure_inference_time(sample_input, num_runs=10)
        
        assert 'mean_inference_time' in stats
        assert 'std_inference_time' in stats
        assert 'min_inference_time' in stats
        assert 'max_inference_time' in stats
        assert stats['mean_inference_time'] > 0

    def test_gradient_clipping(self, sample_model, sample_data):
        """Test gradient clipping during training."""
        X, y = sample_data
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(sample_model.parameters(), lr=0.01)
        
        # Forward pass
        output = sample_model.forward(X)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        grad_norms_before = []
        for param in sample_model.parameters():
            if param.grad is not None:
                grad_norms_before.append(param.grad.norm().item())
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(sample_model.parameters(), max_norm=1.0)
        
        # Check gradients are clipped
        grad_norms_after = []
        for param in sample_model.parameters():
            if param.grad is not None:
                grad_norms_after.append(param.grad.norm().item())
        
        assert len(grad_norms_after) > 0

    def test_device_handling(self, sample_model, sample_data):
        """Test device handling (CPU/GPU)."""
        X, _ = sample_data
        
        # Test CPU
        assert sample_model.device == torch.device('cpu')
        output_cpu = sample_model.forward(X)
        assert output_cpu.device == torch.device('cpu')
        
        # Test GPU if available
        if torch.cuda.is_available():
            sample_model.to(torch.device('cuda'))
            X_gpu = X.cuda()
            output_gpu = sample_model.forward(X_gpu)
            assert output_gpu.device.type == 'cuda'

    def test_weight_initialization(self, metrics_collector):
        """Test weight initialization."""
        model = LightweightLSTM(
            input_size=5,
            hidden_size=8,
            num_layers=1,
            metrics_collector=metrics_collector
        )
        
        # Check that weights are initialized (not all zeros)
        for name, param in model.named_parameters():
            assert not torch.allclose(param, torch.zeros_like(param))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestAttentionLSTM:
    """Test suite for AttentionLSTM."""

    @pytest.fixture
    def attention_model(self, metrics_collector):
        """Create a sample Attention LSTM model."""
        return AttentionLSTM(
            input_size=10,
            hidden_size=16,
            attention_dim=8,
            metrics_collector=metrics_collector
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data for attention model tests."""
        # Create synthetic time series data
        batch_size = 4
        sequence_length = 8
        input_features = 10
        
        X = torch.randn(batch_size, sequence_length, input_features)
        y = torch.randn(batch_size, 1)
        
        return X, y

    def test_attention_initialization(self, attention_model):
        """Test Attention LSTM initialization."""
        assert hasattr(attention_model, 'attention')
        assert hasattr(attention_model, 'output_projection')
        assert attention_model.attention_dim == 8

    def test_attention_forward_pass(self, attention_model, sample_data):
        """Test forward pass with attention."""
        X, _ = sample_data
        
        with torch.no_grad():
            prediction, attention_weights = attention_model.forward(X)
        
        assert prediction.shape == (4, 1)
        assert attention_weights.shape == (4, 8)  # batch_size, seq_len
        
        # Attention weights should sum to 1 for each sample
        attention_sums = attention_weights.sum(dim=1)
        assert torch.allclose(attention_sums, torch.ones(4), atol=1e-6)

    def test_attention_config(self, attention_model):
        """Test attention model configuration."""
        config = attention_model.get_model_config()
        
        assert 'attention_dim' in config
        assert config['attention_dim'] == 8