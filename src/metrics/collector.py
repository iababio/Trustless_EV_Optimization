"""Advanced metrics collection system with comprehensive research analytics."""

import time
import threading
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    import logging
    structlog = logging

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

if HAS_STRUCTLOG:
    logger = structlog.get_logger(__name__)
else:
    logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected for research analysis."""
    
    # ML Model Metrics
    ACCURACY = "accuracy"
    LOSS = "loss"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    INFERENCE_TIME = "inference_time"
    MODEL_SIZE = "model_size"
    
    # Federated Learning Metrics
    ROUND_DURATION = "round_duration"
    CLIENT_PARTICIPATION = "client_participation"
    CONVERGENCE_RATE = "convergence_rate"
    COMMUNICATION_COST = "communication_cost"
    
    # System Performance
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    NETWORK_LATENCY = "network_latency"
    DISK_IO = "disk_io"
    
    # Blockchain Metrics
    TRANSACTION_TIME = "transaction_time"
    GAS_COST = "gas_cost"
    VALIDATION_SUCCESS_RATE = "validation_success_rate"
    
    # Privacy Metrics
    EPSILON_CONSUMED = "epsilon_consumed"
    PRIVACY_BUDGET = "privacy_budget"
    NOISE_MAGNITUDE = "noise_magnitude"
    
    # Business Metrics
    ENERGY_EFFICIENCY = "energy_efficiency"
    CHARGING_OPTIMIZATION = "charging_optimization"
    COST_REDUCTION = "cost_reduction"


@dataclass
class MetricPoint:
    """Individual metric data point with metadata."""
    
    timestamp: datetime
    metric_type: MetricType
    value: Union[float, int, str]
    component: str
    station_id: Optional[str] = None
    experiment_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["metric_type"] = self.metric_type.value
        return data


class MetricsCollector:
    """Centralized metrics collection with real-time analytics."""
    
    def __init__(
        self,
        experiment_id: str,
        storage_backend: Optional[str] = None,
        enable_prometheus: bool = True,
        collection_interval: float = 1.0,
    ):
        self.experiment_id = experiment_id
        self.collection_interval = collection_interval
        self.enable_prometheus = enable_prometheus
        
        # Thread-safe storage
        self._metrics: List[MetricPoint] = []
        self._lock = threading.Lock()
        
        # System monitoring
        self._system_monitor = SystemMonitor()
        self._is_collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # Prometheus setup
        if enable_prometheus and HAS_PROMETHEUS:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        elif enable_prometheus and not HAS_PROMETHEUS:
            logger.warning("Prometheus requested but not available")
            self.enable_prometheus = False
        
        # Research-specific aggregations
        self.aggregations = {
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "median": np.median,
            "p95": lambda x: np.percentile(x, 95),
            "p99": lambda x: np.percentile(x, 99),
        }
        
        logger.info("MetricsCollector initialized", experiment_id=experiment_id)
    
    def _setup_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        self.prometheus_counters = {
            "fl_rounds_total": Counter(
                "fl_rounds_total",
                "Total number of FL rounds completed",
                registry=self.registry,
            ),
            "model_inferences_total": Counter(
                "model_inferences_total",
                "Total number of model inferences",
                ["station_id", "model_type"],
                registry=self.registry,
            ),
        }
        
        self.prometheus_histograms = {
            "inference_duration_seconds": Histogram(
                "inference_duration_seconds",
                "Model inference duration",
                ["station_id", "model_type"],
                registry=self.registry,
            ),
            "fl_round_duration_seconds": Histogram(
                "fl_round_duration_seconds",
                "Federated learning round duration",
                registry=self.registry,
            ),
        }
        
        self.prometheus_gauges = {
            "model_accuracy": Gauge(
                "model_accuracy",
                "Current model accuracy",
                ["station_id", "model_type"],
                registry=self.registry,
            ),
            "system_cpu_usage": Gauge(
                "system_cpu_usage",
                "System CPU usage percentage",
                registry=self.registry,
            ),
            "system_memory_usage": Gauge(
                "system_memory_usage",
                "System memory usage percentage",
                registry=self.registry,
            ),
        }
    
    def record_metric(
        self,
        metric_type: MetricType,
        value: Union[float, int, str],
        component: str,
        station_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single metric point."""
        metric_point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            metric_type=metric_type,
            value=value,
            component=component,
            station_id=station_id,
            experiment_id=self.experiment_id,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._metrics.append(metric_point)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self._update_prometheus_metric(metric_point)
        
        logger.debug(
            "Metric recorded",
            metric_type=metric_type.value,
            value=value,
            component=component,
        )
    
    def _update_prometheus_metric(self, metric_point: MetricPoint) -> None:
        """Update corresponding Prometheus metric."""
        metric_type = metric_point.metric_type
        value = metric_point.value
        
        try:
            if metric_type == MetricType.INFERENCE_TIME:
                labels = [metric_point.station_id or "unknown", metric_point.component]
                self.prometheus_histograms["inference_duration_seconds"].labels(*labels).observe(value)
            
            elif metric_type == MetricType.ACCURACY:
                labels = [metric_point.station_id or "unknown", metric_point.component]
                self.prometheus_gauges["model_accuracy"].labels(*labels).set(value)
            
            elif metric_type == MetricType.CPU_USAGE:
                self.prometheus_gauges["system_cpu_usage"].set(value)
            
            elif metric_type == MetricType.MEMORY_USAGE:
                self.prometheus_gauges["system_memory_usage"].set(value)
                
        except Exception as e:
            logger.warning("Failed to update Prometheus metric", error=str(e))
    
    def record_ml_metrics(
        self,
        accuracy: float,
        loss: float,
        component: str,
        station_id: Optional[str] = None,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record ML model performance metrics."""
        self.record_metric(MetricType.ACCURACY, accuracy, component, station_id)
        self.record_metric(MetricType.LOSS, loss, component, station_id)
        
        if additional_metrics:
            for metric_name, value in additional_metrics.items():
                if hasattr(MetricType, metric_name.upper()):
                    metric_type = MetricType(metric_name.lower())
                    self.record_metric(metric_type, value, component, station_id)
    
    def record_fl_round_metrics(
        self,
        round_id: int,
        duration: float,
        participants: int,
        global_accuracy: float,
        convergence_delta: float,
    ) -> None:
        """Record federated learning round metrics."""
        metadata = {"round_id": round_id, "participants": participants}
        
        self.record_metric(
            MetricType.ROUND_DURATION, duration, "fl_server", metadata=metadata
        )
        self.record_metric(
            MetricType.CLIENT_PARTICIPATION, participants, "fl_server", metadata=metadata
        )
        self.record_metric(
            MetricType.ACCURACY, global_accuracy, "fl_server", metadata=metadata
        )
        self.record_metric(
            MetricType.CONVERGENCE_RATE, convergence_delta, "fl_server", metadata=metadata
        )
        
        if self.enable_prometheus:
            self.prometheus_counters["fl_rounds_total"].inc()
            self.prometheus_histograms["fl_round_duration_seconds"].observe(duration)
    
    def record_privacy_metrics(
        self,
        epsilon_consumed: float,
        privacy_budget_remaining: float,
        noise_magnitude: float,
        component: str,
    ) -> None:
        """Record differential privacy metrics."""
        self.record_metric(MetricType.EPSILON_CONSUMED, epsilon_consumed, component)
        self.record_metric(MetricType.PRIVACY_BUDGET, privacy_budget_remaining, component)
        self.record_metric(MetricType.NOISE_MAGNITUDE, noise_magnitude, component)
    
    def start_system_monitoring(self) -> None:
        """Start continuous system metrics collection."""
        if self._is_collecting:
            logger.warning("System monitoring already started")
            return
        
        self._is_collecting = True
        self._collection_thread = threading.Thread(
            target=self._system_monitoring_loop, daemon=True
        )
        self._collection_thread.start()
        logger.info("System monitoring started")
    
    def stop_system_monitoring(self) -> None:
        """Stop system metrics collection."""
        self._is_collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _system_monitoring_loop(self) -> None:
        """Continuous system monitoring loop."""
        while self._is_collecting:
            try:
                metrics = self._system_monitor.get_metrics()
                
                for metric_name, value in metrics.items():
                    if hasattr(MetricType, metric_name.upper()):
                        metric_type = MetricType(metric_name)
                        self.record_metric(metric_type, value, "system")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error("System monitoring error", error=str(e))
                time.sleep(self.collection_interval)
    
    def get_metrics_summary(
        self,
        metric_type: Optional[MetricType] = None,
        component: Optional[str] = None,
        time_window_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get aggregated metrics summary for research analysis."""
        with self._lock:
            metrics = self._metrics.copy()
        
        # Filter metrics
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        if component:
            metrics = [m for m in metrics if m.component == component]
        if time_window_minutes:
            cutoff_time = datetime.now(timezone.utc).timestamp() - (time_window_minutes * 60)
            metrics = [m for m in metrics if m.timestamp.timestamp() > cutoff_time]
        
        if not metrics:
            return {"count": 0}
        
        # Extract numeric values only
        numeric_values = []
        for m in metrics:
            if isinstance(m.value, (int, float)):
                numeric_values.append(m.value)
        
        if not numeric_values:
            return {"count": len(metrics), "non_numeric": True}
        
        # Calculate aggregations
        summary = {"count": len(metrics), "numeric_count": len(numeric_values)}
        
        try:
            for agg_name, agg_func in self.aggregations.items():
                summary[agg_name] = float(agg_func(numeric_values))
        except Exception as e:
            logger.warning("Aggregation calculation failed", error=str(e))
        
        return summary
    
    def export_metrics(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export all collected metrics."""
        with self._lock:
            metrics_data = [m.to_dict() for m in self._metrics]
        
        export_data = {
            "experiment_id": self.experiment_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_metrics": len(metrics_data),
            "metrics": metrics_data,
        }
        
        if format_type == "json":
            return json.dumps(export_data, indent=2)
        else:
            return export_data
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
        logger.info("All metrics cleared")


class SystemMonitor:
    """System resource monitoring."""
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        if not HAS_PSUTIL:
            # Return mock metrics if psutil not available
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_io": 0.0,
                "network_latency": 0.0,
            }
        
        try:
            return {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_io": self._get_disk_io_rate(),
                "network_latency": self._measure_network_latency(),
            }
        except Exception as e:
            logger.error("System metrics collection failed", error=str(e))
            return {}
    
    def _get_disk_io_rate(self) -> float:
        """Calculate current disk I/O rate."""
        if not HAS_PSUTIL:
            return 0.0
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io:
                return disk_io.read_bytes + disk_io.write_bytes
            return 0.0
        except:
            return 0.0
    
    def _measure_network_latency(self) -> float:
        """Measure network latency (simplified)."""
        try:
            import subprocess
            result = subprocess.run(
                ["ping", "-c", "1", "8.8.8.8"],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if result.returncode == 0:
                # Extract latency from ping output
                for line in result.stdout.split('\n'):
                    if 'time=' in line:
                        latency_str = line.split('time=')[1].split()[0]
                        return float(latency_str)
            return -1.0
        except:
            return -1.0