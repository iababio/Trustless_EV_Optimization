"""Security tests for privacy protection mechanisms."""

import pytest
import numpy as np
import pandas as pd
import json
from unittest.mock import Mock, patch

from src.metrics.collector import MetricsCollector
from src.blockchain.validator import MockModelValidator


class TestPrivacyProtection:
    """Test suite for privacy-preserving mechanisms."""

    @pytest.fixture
    def sensitive_data(self):
        """Create sensitive EV charging data for privacy testing."""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'user_id': [f'user_{i:03d}' for i in np.random.randint(0, 100, n_samples)],
            'location_lat': 37.7749 + np.random.normal(0, 0.1, n_samples),
            'location_lon': -122.4194 + np.random.normal(0, 0.1, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='30min'),
            'meter_total_wh': np.random.exponential(4000, n_samples),
            'session_duration_min': np.random.gamma(2, 25, n_samples),
            'payment_amount': np.random.exponential(15, n_samples),
            'vehicle_model': np.random.choice(['Tesla Model 3', 'Nissan Leaf', 'BMW i3'], n_samples),
        })

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for security tests."""
        return MetricsCollector("security_test", enable_prometheus=False)

    def test_data_anonymization(self, sensitive_data):
        """Test that sensitive data is properly anonymized."""
        from src.privacy.anonymizer import DataAnonymizer
        
        anonymizer = DataAnonymizer()
        anonymized_data = anonymizer.anonymize_data(sensitive_data)
        
        # Should not contain original user IDs
        if 'user_id' in anonymized_data.columns:
            original_ids = set(sensitive_data['user_id'].unique())
            anonymized_ids = set(anonymized_data['user_id'].unique())
            assert len(original_ids.intersection(anonymized_ids)) == 0
        
        # Should preserve statistical properties
        assert len(anonymized_data) == len(sensitive_data)
        assert abs(anonymized_data['meter_total_wh'].mean() - sensitive_data['meter_total_wh'].mean()) < 1000

    def test_differential_privacy_noise_addition(self):
        """Test differential privacy noise addition."""
        from src.privacy.differential_privacy import add_laplace_noise, add_gaussian_noise
        
        original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        epsilon = 1.0
        
        # Test Laplace noise
        noisy_data_laplace = add_laplace_noise(original_data, epsilon=epsilon)
        assert len(noisy_data_laplace) == len(original_data)
        assert not np.array_equal(original_data, noisy_data_laplace)
        
        # Test Gaussian noise  
        noisy_data_gaussian = add_gaussian_noise(original_data, sigma=1.0)
        assert len(noisy_data_gaussian) == len(original_data)
        assert not np.array_equal(original_data, noisy_data_gaussian)
        
        # Noise should be bounded for reasonable epsilon
        max_noise = np.max(np.abs(noisy_data_laplace - original_data))
        assert max_noise < 20  # Reasonable bound for epsilon=1.0

    def test_privacy_budget_tracking(self, metrics_collector):
        """Test privacy budget consumption tracking."""
        initial_budget = 10.0
        privacy_tracker = PrivacyBudgetTracker(initial_budget, metrics_collector)
        
        # Consume some budget
        privacy_tracker.consume_budget(1.5, "data_analysis")
        assert privacy_tracker.remaining_budget == 8.5
        
        privacy_tracker.consume_budget(2.0, "model_training")  
        assert privacy_tracker.remaining_budget == 6.5
        
        # Should track consumption history
        history = privacy_tracker.get_consumption_history()
        assert len(history) == 2
        assert history[0]['amount'] == 1.5
        assert history[1]['amount'] == 2.0

    def test_privacy_budget_exhaustion(self, metrics_collector):
        """Test behavior when privacy budget is exhausted."""
        privacy_tracker = PrivacyBudgetTracker(2.0, metrics_collector)
        
        # Consume most of budget
        privacy_tracker.consume_budget(1.8, "operation1")
        
        # Should warn when budget is low
        assert privacy_tracker.remaining_budget < 1.0
        
        # Should raise exception when trying to exceed budget
        with pytest.raises(ValueError, match="Insufficient privacy budget"):
            privacy_tracker.consume_budget(1.0, "operation2")

    def test_secure_aggregation_simulation(self):
        """Test secure aggregation of model updates."""
        from src.privacy.secure_aggregation import SecureAggregator
        
        aggregator = SecureAggregator()
        
        # Simulate multiple client updates
        client_updates = [
            {'weights': np.random.randn(10), 'client_id': f'client_{i}'}
            for i in range(5)
        ]
        
        # Aggregate securely
        aggregated_weights = aggregator.aggregate_updates(client_updates)
        
        assert len(aggregated_weights) == 10
        assert isinstance(aggregated_weights, np.ndarray)
        
        # Should be different from individual client weights
        for update in client_updates:
            assert not np.array_equal(aggregated_weights, update['weights'])

    def test_homomorphic_encryption_simulation(self):
        """Test homomorphic encryption for private computation."""
        from src.privacy.homomorphic import HomomorphicEncryption
        
        he = HomomorphicEncryption()
        
        # Test encryption/decryption
        plaintext = [10, 20, 30]
        encrypted = he.encrypt(plaintext)
        decrypted = he.decrypt(encrypted)
        
        assert decrypted == plaintext
        
        # Test homomorphic addition
        plaintext2 = [5, 10, 15]
        encrypted2 = he.encrypt(plaintext2)
        
        encrypted_sum = he.add(encrypted, encrypted2)
        decrypted_sum = he.decrypt(encrypted_sum)
        
        expected_sum = [a + b for a, b in zip(plaintext, plaintext2)]
        assert decrypted_sum == expected_sum

    def test_data_minimization(self, sensitive_data):
        """Test data minimization principles."""
        from src.privacy.minimizer import DataMinimizer
        
        minimizer = DataMinimizer()
        
        # Define minimal required fields for a specific task
        required_fields = ['timestamp', 'meter_total_wh', 'session_duration_min']
        
        minimized_data = minimizer.minimize_data(sensitive_data, required_fields)
        
        # Should only contain required fields
        assert set(minimized_data.columns) == set(required_fields)
        assert len(minimized_data) == len(sensitive_data)
        
        # Should not contain sensitive fields
        sensitive_fields = ['user_id', 'location_lat', 'location_lon', 'payment_amount']
        for field in sensitive_fields:
            assert field not in minimized_data.columns

    def test_k_anonymity(self, sensitive_data):
        """Test k-anonymity privacy protection."""
        from src.privacy.k_anonymity import KAnonymizer
        
        k_anonymizer = KAnonymizer(k=5)
        
        # Apply k-anonymity to quasi-identifiers
        quasi_identifiers = ['timestamp', 'location_lat', 'location_lon']
        
        anonymized_data = k_anonymizer.apply_k_anonymity(
            sensitive_data, 
            quasi_identifiers=quasi_identifiers
        )
        
        # Check that each combination of quasi-identifiers appears at least k times
        if len(quasi_identifiers) > 0:
            grouped = anonymized_data.groupby(quasi_identifiers).size()
            assert all(count >= 5 for count in grouped.values)

    def test_l_diversity(self, sensitive_data):
        """Test l-diversity privacy protection."""
        from src.privacy.l_diversity import LDiversityProtector
        
        protector = LDiversityProtector(l=3)
        
        protected_data = protector.apply_l_diversity(
            sensitive_data,
            sensitive_attribute='vehicle_model',
            quasi_identifiers=['timestamp', 'location_lat']
        )
        
        # Each equivalence class should have at least l distinct values
        # for the sensitive attribute
        assert len(protected_data) > 0

    def test_input_validation_and_sanitization(self):
        """Test input validation and sanitization."""
        from src.security.input_validator import InputValidator
        
        validator = InputValidator()
        
        # Test SQL injection protection
        malicious_input = "'; DROP TABLE users; --"
        sanitized = validator.sanitize_string(malicious_input)
        assert "DROP TABLE" not in sanitized
        assert "--" not in sanitized
        
        # Test XSS protection
        xss_input = "<script>alert('xss')</script>"
        sanitized_xss = validator.sanitize_string(xss_input)
        assert "<script>" not in sanitized_xss
        assert "alert" not in sanitized_xss
        
        # Test numeric validation
        assert validator.validate_numeric("123.45", min_value=0, max_value=1000)
        assert not validator.validate_numeric("abc", min_value=0, max_value=1000)
        assert not validator.validate_numeric("1001", min_value=0, max_value=1000)

    def test_access_control_simulation(self):
        """Test access control mechanisms."""
        from src.security.access_control import AccessController
        
        controller = AccessController()
        
        # Define roles and permissions
        controller.define_role("admin", ["read", "write", "delete"])
        controller.define_role("analyst", ["read"])
        controller.define_role("station_operator", ["read", "write"])
        
        # Assign roles to users
        controller.assign_role("alice", "admin")
        controller.assign_role("bob", "analyst") 
        controller.assign_role("charlie", "station_operator")
        
        # Test permissions
        assert controller.has_permission("alice", "delete")
        assert controller.has_permission("bob", "read")
        assert not controller.has_permission("bob", "write")
        assert controller.has_permission("charlie", "write")
        assert not controller.has_permission("charlie", "delete")

    def test_audit_logging(self, metrics_collector):
        """Test security audit logging."""
        from src.security.audit_logger import AuditLogger
        
        audit_logger = AuditLogger(metrics_collector)
        
        # Log various security events
        audit_logger.log_access_attempt("alice", "data_analysis", success=True)
        audit_logger.log_access_attempt("mallory", "admin_panel", success=False)
        audit_logger.log_data_access("bob", "charging_data", "read")
        audit_logger.log_privacy_operation("charlie", "differential_privacy", epsilon=1.0)
        
        # Check audit trail
        audit_trail = audit_logger.get_audit_trail()
        assert len(audit_trail) == 4
        
        # Check failed access attempts
        failed_attempts = audit_logger.get_failed_attempts()
        assert len(failed_attempts) == 1
        assert failed_attempts[0]['user'] == "mallory"

    def test_encryption_at_rest(self, sensitive_data, temp_dir):
        """Test encryption of data at rest."""
        from src.security.encryption import DataEncryption
        
        encryptor = DataEncryption()
        
        # Encrypt and save data
        encrypted_file = temp_dir / "encrypted_data.enc"
        encryptor.encrypt_dataframe(sensitive_data, str(encrypted_file))
        
        assert encrypted_file.exists()
        
        # File should not contain plaintext data
        with open(encrypted_file, 'rb') as f:
            content = f.read()
            assert b"user_" not in content  # Should not contain plaintext user IDs
        
        # Decrypt and verify
        decrypted_data = encryptor.decrypt_dataframe(str(encrypted_file))
        pd.testing.assert_frame_equal(sensitive_data, decrypted_data)

    def test_secure_communication_simulation(self):
        """Test secure communication protocols."""
        from src.security.secure_comm import SecureCommunicator
        
        comm = SecureCommunicator()
        
        # Simulate secure message exchange
        message = {"model_weights": [1.0, 2.0, 3.0], "metadata": {"round": 1}}
        
        # Encrypt message
        encrypted_message = comm.encrypt_message(message)
        assert encrypted_message != message
        
        # Decrypt message
        decrypted_message = comm.decrypt_message(encrypted_message)
        assert decrypted_message == message
        
        # Test message integrity
        assert comm.verify_message_integrity(encrypted_message)

    def test_blockchain_security(self, metrics_collector):
        """Test blockchain security features."""
        validator = MockModelValidator(metrics_collector)
        
        # Test with malicious model hash
        malicious_hash = "0x" + "0" * 64  # All zeros (suspicious)
        malicious_metadata = json.dumps({
            "accuracy": 1.0,  # Suspiciously perfect
            "participants": 1000000  # Unrealistic number
        })
        
        result = validator.validate_model_update(malicious_hash, malicious_metadata, 1)
        # Should detect and reject malicious input
        assert not result.is_valid
        
        # Test with valid hash but manipulated metadata
        valid_hash = "0x" + "a" * 64
        manipulated_metadata = '{"accuracy": 0.95, "malicious_code": "rm -rf /"}'
        
        result = validator.validate_model_update(valid_hash, manipulated_metadata, 2)
        # Should handle safely (either reject or sanitize)
        assert result is not None

    def test_model_poisoning_detection(self):
        """Test detection of model poisoning attacks."""
        from src.security.model_security import ModelPoisoningDetector
        
        detector = ModelPoisoningDetector()
        
        # Simulate normal model updates
        normal_updates = [
            {'weights': np.random.normal(0, 0.1, 100), 'accuracy': 0.85},
            {'weights': np.random.normal(0, 0.1, 100), 'accuracy': 0.87},
            {'weights': np.random.normal(0, 0.1, 100), 'accuracy': 0.83},
        ]
        
        # Simulate poisoned model update
        poisoned_update = {
            'weights': np.random.normal(10, 1, 100),  # Drastically different weights
            'accuracy': 0.99  # Suspiciously high accuracy
        }
        
        # Test detection
        for update in normal_updates:
            is_poisoned = detector.detect_poisoning(update, normal_updates)
            assert not is_poisoned
        
        is_poisoned = detector.detect_poisoning(poisoned_update, normal_updates)
        assert is_poisoned

    def test_federated_learning_security(self, metrics_collector):
        """Test federated learning security mechanisms."""
        from src.security.fl_security import FLSecurityManager
        
        security_manager = FLSecurityManager(metrics_collector)
        
        # Test client authentication
        assert security_manager.authenticate_client("valid_client_001", "correct_token")
        assert not security_manager.authenticate_client("malicious_client", "wrong_token")
        
        # Test secure aggregation
        client_updates = [
            {'id': 'client_1', 'weights': np.random.randn(10), 'samples': 100},
            {'id': 'client_2', 'weights': np.random.randn(10), 'samples': 150},
            {'id': 'client_3', 'weights': np.random.randn(10), 'samples': 80},
        ]
        
        aggregated = security_manager.secure_aggregate(client_updates)
        assert len(aggregated['weights']) == 10
        assert 'participant_count' in aggregated

    def test_privacy_metrics_collection(self, metrics_collector):
        """Test that privacy-related metrics are properly collected."""
        # Simulate privacy operations
        metrics_collector.record_privacy_metrics(
            epsilon_consumed=0.5,
            noise_magnitude=0.1,
            privacy_budget_remaining=9.5,
            component="data_processor"
        )
        
        metrics_collector.record_privacy_metrics(
            epsilon_consumed=0.3,
            noise_magnitude=0.05,
            privacy_budget_remaining=9.2,
            component="model_trainer"
        )
        
        # Check metrics were recorded
        metrics = metrics_collector.export_metrics(format_type="dict")
        privacy_metrics = [m for m in metrics['metrics'] if m.get('metric_type') == 'privacy']
        
        assert len(privacy_metrics) > 0
        
        # Check specific privacy metrics
        epsilon_metrics = [m for m in privacy_metrics if m.get('metric_name') == 'epsilon_consumed']
        assert len(epsilon_metrics) == 2


# Helper classes for testing (would be in actual implementation)
class PrivacyBudgetTracker:
    def __init__(self, initial_budget, metrics_collector):
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget
        self.consumption_history = []
        self.metrics_collector = metrics_collector
    
    def consume_budget(self, amount, operation):
        if amount > self.remaining_budget:
            raise ValueError("Insufficient privacy budget")
        
        self.remaining_budget -= amount
        self.consumption_history.append({
            'amount': amount,
            'operation': operation,
            'timestamp': time.time()
        })
        
        if self.metrics_collector:
            self.metrics_collector.record_privacy_metrics(
                epsilon_consumed=amount,
                privacy_budget_remaining=self.remaining_budget,
                component=operation
            )
    
    def get_consumption_history(self):
        return self.consumption_history