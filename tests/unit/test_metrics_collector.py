"""Unit tests for the metrics collection system."""

import pytest
import time
import json
from unittest.mock import Mock, patch

from src.metrics.collector import MetricsCollector, MetricType


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    def test_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector("test_exp", enable_prometheus=False)
        
        assert collector.experiment_id == "test_exp"
        assert collector._metrics == []
        assert hasattr(collector, '_system_monitor')
        assert not collector.enable_prometheus

    def test_record_ml_metrics(self, metrics_collector):
        """Test ML metrics recording."""
        metrics_collector.record_ml_metrics(
            accuracy=0.85,
            loss=0.25,
            component="test_model",
            additional_metrics={"f1_score": 0.82}
        )
        
        assert len(metrics_collector._metrics) > 0
        
        # Find the accuracy metric
        accuracy_metric = next(
            (m for m in metrics_collector.metrics if m.get("metric_type") == "accuracy"),
            None
        )
        assert accuracy_metric is not None
        assert accuracy_metric["value"] == 0.85
        assert accuracy_metric["component"] == "test_model"

    def test_record_federated_metrics(self, metrics_collector):
        """Test federated learning metrics recording."""
        metrics_collector.record_federated_metrics(
            round_id=1,
            num_clients=5,
            round_duration=120.5,
            convergence_metric=0.95,
            component="fl_server"
        )
        
        # Check that federated learning metrics were recorded
        round_duration_metrics = [m for m in metrics_collector.metrics if m.get("metric_type") == "round_duration"]
        assert len(round_duration_metrics) > 0

    def test_record_system_metrics(self, metrics_collector):
        """Test system performance metrics recording."""
        metrics_collector.record_system_metrics(
            cpu_usage=75.5,
            memory_usage=60.2,
            network_latency=15.3,
            component="edge_node"
        )
        
        # Check that system metrics were recorded
        cpu_metrics = [m for m in metrics_collector.metrics if m.get("metric_type") == "cpu_usage"]
        assert len(cpu_metrics) > 0

    def test_record_privacy_metrics(self, metrics_collector):
        """Test privacy metrics recording."""
        metrics_collector.record_privacy_metrics(
            epsilon_consumed=0.1,
            noise_magnitude=0.05,
            privacy_budget_remaining=0.9,
            component="dp_mechanism"
        )
        
        # Check that privacy metrics were recorded
        epsilon_metrics = [m for m in metrics_collector.metrics if m.get("metric_type") == "epsilon_consumed"]
        assert len(epsilon_metrics) > 0

    def test_record_blockchain_metrics(self, metrics_collector):
        """Test blockchain metrics recording."""
        metrics_collector.record_blockchain_metrics(
            transaction_time=2.5,
            gas_used=150000,
            validation_success=True,
            component="model_validator"
        )
        
        blockchain_metrics = [m for m in metrics_collector.metrics if m.get("metric_type") == "blockchain"]
        assert len(blockchain_metrics) > 0

    def test_get_metrics_summary(self, metrics_collector):
        """Test metrics summary generation."""
        # Add various metrics
        for i in range(10):
            metrics_collector.record_ml_metrics(
                accuracy=0.8 + i * 0.01,
                loss=0.3 - i * 0.01,
                component="test_model"
            )
        
        summary = metrics_collector.get_metrics_summary(
            metric_type=MetricType.ACCURACY,
            time_window_minutes=60
        )
        
        assert "count" in summary
        assert "mean" in summary
        assert summary["count"] > 0

    def test_export_metrics_dict(self, metrics_collector):
        """Test metrics export to dictionary."""
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="test")
        
        exported = metrics_collector.export_metrics(format_type="dict")
        
        assert isinstance(exported, dict)
        assert "experiment_id" in exported
        assert "metrics" in exported
        assert "total_metrics" in exported
        assert exported["total_metrics"] > 0

    def test_export_metrics_json(self, metrics_collector):
        """Test metrics export to JSON."""
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="test")
        
        json_str = metrics_collector.export_metrics(format_type="json")
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "experiment_id" in parsed
        assert "metrics" in parsed

    def test_export_metrics_csv(self, metrics_collector, temp_dir):
        """Test metrics export to CSV."""
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="test")
        
        csv_path = temp_dir / "metrics.csv"
        metrics_collector.export_metrics(format_type="csv", file_path=str(csv_path))
        
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_start_system_monitoring(self, mock_memory, mock_cpu, metrics_collector):
        """Test system monitoring start."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 70.0
        
        metrics_collector.start_system_monitoring(interval_seconds=0.1)
        
        # Wait briefly for monitoring to collect data
        time.sleep(0.2)
        
        assert metrics_collector.system_monitor is not None
        assert metrics_collector.system_monitor.is_alive()
        
        metrics_collector.stop_system_monitoring()

    def test_stop_system_monitoring(self, metrics_collector):
        """Test system monitoring stop."""
        metrics_collector.start_system_monitoring(interval_seconds=0.1)
        time.sleep(0.1)
        
        metrics_collector.stop_system_monitoring()
        
        assert metrics_collector.system_monitor is None

    def test_custom_metric_recording(self, metrics_collector):
        """Test custom metric recording."""
        metrics_collector.record_custom_metric(
            metric_name="custom_accuracy",
            value=0.95,
            metric_type="custom",
            component="custom_component",
            metadata={"version": "1.0"}
        )
        
        custom_metrics = [m for m in metrics_collector.metrics if m.get("metadata", {}).get("metric_name") == "custom_accuracy"]
        assert len(custom_metrics) == 1
        assert custom_metrics[0]["metadata"]["version"] == "1.0"

    def test_metrics_filtering_by_component(self, metrics_collector):
        """Test filtering metrics by component."""
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="model_A")
        metrics_collector.record_ml_metrics(accuracy=0.8, loss=0.2, component="model_B")
        
        model_a_metrics = metrics_collector.get_metrics_by_component("model_A")
        model_b_metrics = metrics_collector.get_metrics_by_component("model_B")
        
        assert len(model_a_metrics) > 0
        assert len(model_b_metrics) > 0
        assert all(m["component"] == "model_A" for m in model_a_metrics)
        assert all(m["component"] == "model_B" for m in model_b_metrics)

    def test_metrics_filtering_by_time(self, metrics_collector):
        """Test filtering metrics by time range."""
        # Record metrics with slight delay
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="test")
        time.sleep(0.1)
        start_time = time.time()
        time.sleep(0.1)
        metrics_collector.record_ml_metrics(accuracy=0.8, loss=0.2, component="test")
        
        recent_metrics = metrics_collector.get_metrics_since(start_time)
        
        assert len(recent_metrics) >= 1  # At least the second metric

    def test_metric_validation(self, metrics_collector):
        """Test metric validation."""
        # Valid metrics should be accepted
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="test")
        
        # Invalid metrics should be handled gracefully
        with pytest.raises((ValueError, TypeError)):
            metrics_collector.record_ml_metrics(accuracy="invalid", loss=0.1, component="test")

    def test_metric_aggregation(self, metrics_collector):
        """Test metric aggregation functionality."""
        # Record multiple metrics of same type
        for i in range(5):
            metrics_collector.record_ml_metrics(
                accuracy=0.8 + i * 0.02,
                loss=0.3 - i * 0.02,
                component="test_model"
            )
        
        aggregated = metrics_collector.aggregate_metrics(
            metric_name="accuracy",
            aggregation_type="mean"
        )
        
        assert aggregated is not None
        assert isinstance(aggregated, (int, float))

    def test_clear_metrics(self, metrics_collector):
        """Test clearing all metrics."""
        metrics_collector.record_ml_metrics(accuracy=0.9, loss=0.1, component="test")
        assert len(metrics_collector.metrics) > 0
        
        metrics_collector.clear_metrics()
        assert len(metrics_collector.metrics) == 0