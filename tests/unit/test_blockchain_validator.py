"""Unit tests for blockchain validator."""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock

from src.blockchain.validator import ModelValidator, MockModelValidator, create_model_validator


class TestMockModelValidator:
    """Test suite for MockModelValidator."""

    @pytest.fixture
    def mock_validator(self, metrics_collector):
        """Create a mock validator for testing."""
        return MockModelValidator(metrics_collector=metrics_collector)

    def test_initialization(self, mock_validator):
        """Test mock validator initialization."""
        assert mock_validator.validation_count == 0
        assert mock_validator.successful_validations == 0
        assert isinstance(mock_validator.validation_history, list)

    def test_validate_model_update(self, mock_validator):
        """Test model update validation."""
        model_hash = "0x" + "a" * 64
        metadata = json.dumps({"accuracy": 0.95, "loss": 0.05})
        
        result = mock_validator.validate_model_update(
            model_hash=model_hash,
            metadata=metadata,
            round_id=1
        )
        
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'model_hash')
        assert hasattr(result, 'transaction_hash')
        assert hasattr(result, 'validation_time')
        
        assert result.model_hash == model_hash
        assert result.is_valid in [True, False]
        assert mock_validator.validation_count == 1

    def test_validation_with_invalid_hash(self, mock_validator):
        """Test validation with invalid model hash."""
        invalid_hash = "invalid_hash"
        metadata = json.dumps({"accuracy": 0.95})
        
        result = mock_validator.validate_model_update(
            model_hash=invalid_hash,
            metadata=metadata,
            round_id=1
        )
        
        assert not result.is_valid

    def test_validation_with_invalid_metadata(self, mock_validator):
        """Test validation with invalid metadata."""
        model_hash = "0x" + "a" * 64
        invalid_metadata = "not json"
        
        result = mock_validator.validate_model_update(
            model_hash=model_hash,
            metadata=invalid_metadata,
            round_id=1
        )
        
        assert not result.is_valid

    def test_get_validator_stats(self, mock_validator):
        """Test getting validator statistics."""
        # Perform some validations
        for i in range(5):
            model_hash = "0x" + str(i) * 64
            metadata = json.dumps({"accuracy": 0.9 + i * 0.01})
            mock_validator.validate_model_update(model_hash, metadata, i + 1)
        
        stats = mock_validator.get_validator_stats()
        
        assert 'total_validations' in stats
        assert 'successful_validations' in stats
        assert 'success_rate' in stats
        assert 'average_validation_time' in stats
        
        assert stats['total_validations'] == 5
        assert 0 <= stats['success_rate'] <= 1

    def test_validation_history(self, mock_validator):
        """Test validation history tracking."""
        model_hash = "0x" + "a" * 64
        metadata = json.dumps({"accuracy": 0.95})
        
        mock_validator.validate_model_update(model_hash, metadata, 1)
        
        assert len(mock_validator.validation_history) == 1
        history_item = mock_validator.validation_history[0]
        
        assert 'model_hash' in history_item
        assert 'metadata' in history_item
        assert 'round_id' in history_item
        assert 'validation_result' in history_item

    def test_metrics_recording(self, mock_validator):
        """Test that blockchain metrics are recorded."""
        model_hash = "0x" + "a" * 64
        metadata = json.dumps({"accuracy": 0.95})
        
        mock_validator.validate_model_update(model_hash, metadata, 1)
        
        # Check that metrics were recorded
        if mock_validator.metrics_collector:
            metrics = mock_validator.metrics_collector.export_metrics(format_type="dict")
            blockchain_metrics = [
                m for m in metrics.get('metrics', [])
                if m.get('metric_type') == 'blockchain'
            ]
            assert len(blockchain_metrics) > 0


class TestModelValidator:
    """Test suite for ModelValidator (Web3 version)."""

    @pytest.fixture
    def web3_mock(self):
        """Create a mock Web3 instance."""
        web3_mock = Mock()
        web3_mock.is_connected.return_value = True
        web3_mock.eth.get_block.return_value = {'timestamp': 1640995200}
        return web3_mock

    @pytest.fixture
    def contract_mock(self):
        """Create a mock smart contract."""
        contract_mock = Mock()
        contract_mock.address = "0x1234567890123456789012345678901234567890"
        
        # Mock contract functions
        contract_mock.functions.validateModelUpdate.return_value.call.return_value = True
        contract_mock.functions.getContractStats.return_value.call.return_value = [10, 8, 5, 100]
        
        return contract_mock

    @pytest.mark.skipif(True, reason="Requires Web3 and blockchain connection")
    def test_web3_validator_initialization(self, web3_mock, contract_mock, metrics_collector):
        """Test Web3 validator initialization."""
        with patch('src.blockchain.validator.Web3', return_value=web3_mock):
            validator = ModelValidator(
                web3_instance=web3_mock,
                contract_address="0x1234567890123456789012345678901234567890",
                metrics_collector=metrics_collector
            )
            
            assert validator.web3 == web3_mock
            assert validator.contract_address == "0x1234567890123456789012345678901234567890"

    @pytest.mark.skipif(True, reason="Requires Web3 and blockchain connection")  
    def test_web3_validate_model_update(self, web3_mock, contract_mock, metrics_collector):
        """Test Web3 model validation."""
        with patch('src.blockchain.validator.Web3', return_value=web3_mock):
            validator = ModelValidator(
                web3_instance=web3_mock,
                contract_address="0x1234567890123456789012345678901234567890",
                metrics_collector=metrics_collector
            )
            validator.contract = contract_mock
            
            model_hash = "0x" + "a" * 64
            metadata = json.dumps({"accuracy": 0.95})
            
            result = validator.validate_model_update(model_hash, metadata, 1)
            
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'transaction_hash')


class TestValidatorFactory:
    """Test suite for validator factory function."""

    def test_create_validator_mock(self, metrics_collector):
        """Test creating mock validator."""
        validator = create_model_validator(
            metrics_collector=metrics_collector,
            use_blockchain=False
        )
        
        assert isinstance(validator, MockModelValidator)

    @patch('src.blockchain.validator.Web3')
    def test_create_validator_web3_unavailable(self, mock_web3_class, metrics_collector):
        """Test creating validator when Web3 is unavailable."""
        # Mock Web3 to raise ImportError
        mock_web3_class.side_effect = ImportError("Web3 not available")
        
        validator = create_model_validator(
            metrics_collector=metrics_collector,
            use_blockchain=True
        )
        
        # Should fallback to mock validator
        assert isinstance(validator, MockModelValidator)

    def test_create_validator_with_config(self, metrics_collector):
        """Test creating validator with configuration."""
        config = {
            'contract_address': '0x1234567890123456789012345678901234567890',
            'rpc_url': 'http://localhost:8545',
            'private_key': '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890'
        }
        
        validator = create_model_validator(
            metrics_collector=metrics_collector,
            use_blockchain=False,
            config=config
        )
        
        assert isinstance(validator, MockModelValidator)


class TestValidatorIntegration:
    """Integration tests for validator components."""

    def test_validator_with_fl_workflow(self, metrics_collector):
        """Test validator integration with federated learning workflow."""
        validator = MockModelValidator(metrics_collector=metrics_collector)
        
        # Simulate FL rounds
        for round_id in range(1, 4):
            model_hash = f"0x{'a' * 60}{round_id:04d}"
            metadata = json.dumps({
                "round": round_id,
                "participants": 5,
                "accuracy": 0.8 + round_id * 0.05
            })
            
            result = validator.validate_model_update(model_hash, metadata, round_id)
            assert result is not None
        
        stats = validator.get_validator_stats()
        assert stats['total_validations'] == 3

    def test_validator_error_handling(self, metrics_collector):
        """Test validator error handling."""
        validator = MockModelValidator(metrics_collector=metrics_collector)
        
        # Test with None values
        result = validator.validate_model_update(None, None, 1)
        assert not result.is_valid
        
        # Test with empty values
        result = validator.validate_model_update("", "", 0)
        assert not result.is_valid

    def test_validator_concurrent_validations(self, metrics_collector):
        """Test validator handling concurrent validations."""
        import threading
        import time
        
        validator = MockModelValidator(metrics_collector=metrics_collector)
        results = []
        
        def validate_model(round_id):
            model_hash = f"0x{'b' * 60}{round_id:04d}"
            metadata = json.dumps({"accuracy": 0.9, "round": round_id})
            result = validator.validate_model_update(model_hash, metadata, round_id)
            results.append(result)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=validate_model, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert validator.validation_count == 5

    def test_validator_performance_metrics(self, metrics_collector):
        """Test validator performance metrics collection."""
        validator = MockModelValidator(metrics_collector=metrics_collector)
        
        # Perform multiple validations
        validation_times = []
        for i in range(10):
            start_time = time.time()
            model_hash = f"0x{'c' * 60}{i:04d}"
            metadata = json.dumps({"accuracy": 0.9})
            validator.validate_model_update(model_hash, metadata, i + 1)
            validation_times.append(time.time() - start_time)
        
        stats = validator.get_validator_stats()
        
        # Check performance metrics
        assert 'average_validation_time' in stats
        assert stats['average_validation_time'] > 0
        assert stats['total_validations'] == 10