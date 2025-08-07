"""Base classes for lightweight ML models optimized for edge deployment."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
import structlog

from ..metrics.collector import MetricsCollector, MetricType

logger = structlog.get_logger(__name__)


class EdgeMLModel(ABC, nn.Module):
    """Abstract base class for all edge-deployed ML models with comprehensive metrics."""
    
    def __init__(
        self,
        model_name: str,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.metrics_collector = metrics_collector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model metadata
        self.training_history: Dict[str, Any] = {}
        self.optimization_applied = False
        self.pruning_ratio = 0.0
        self.quantized = False
        
        logger.info("EdgeMLModel initialized", model_name=model_name, device=str(self.device))
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        pass
    
    def get_model_size(self) -> int:
        """Return model size in bytes."""
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                MetricType.MODEL_SIZE,
                total_size,
                self.model_name
            )
        
        return total_size
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }
    
    def measure_inference_time(self, x: torch.Tensor, num_runs: int = 100) -> Dict[str, float]:
        """Measure inference time with metrics tracking."""
        self.eval()
        times = []
        
        with torch.no_grad():
            # Warm-up runs
            for _ in range(10):
                _ = self(x)
            
            # Actual measurement
            for _ in range(num_runs):
                start_time = datetime.now()
                _ = self(x)
                end_time = datetime.now()
                times.append((end_time - start_time).total_seconds())
        
        inference_stats = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99),
        }
        
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                MetricType.INFERENCE_TIME,
                inference_stats["mean_time"],
                self.model_name,
                metadata={"batch_size": x.shape[0]}
            )
        
        logger.info(
            "Inference time measured",
            model_name=self.model_name,
            mean_time=inference_stats["mean_time"],
            batch_size=x.shape[0]
        )
        
        return inference_stats
    
    def optimize_for_edge(
        self,
        pruning_ratio: float = 0.8,
        quantize: bool = True,
        structured_pruning: bool = False,
    ) -> Dict[str, Any]:
        """Apply edge optimization techniques with comprehensive metrics."""
        optimization_start = datetime.now()
        original_size = self.get_model_size()
        
        optimization_report = {
            "original_size_bytes": original_size,
            "original_parameters": self.count_parameters(),
            "optimizations_applied": [],
        }
        
        try:
            # Apply pruning
            if pruning_ratio > 0:
                self._apply_pruning(pruning_ratio, structured_pruning)
                optimization_report["optimizations_applied"].append("pruning")
                self.pruning_ratio = pruning_ratio
            
            # Apply quantization
            if quantize:
                self._apply_quantization()
                optimization_report["optimizations_applied"].append("quantization")
                self.quantized = True
            
            # Measure results
            optimized_size = self.get_model_size()
            optimization_report.update({
                "optimized_size_bytes": optimized_size,
                "optimized_parameters": self.count_parameters(),
                "size_reduction_ratio": (original_size - optimized_size) / original_size,
                "compression_ratio": original_size / optimized_size,
                "optimization_time_seconds": (datetime.now() - optimization_start).total_seconds(),
            })
            
            self.optimization_applied = True
            
            # Record optimization metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.MODEL_SIZE,
                    optimized_size,
                    self.model_name,
                    metadata={"optimization": "post_optimization"}
                )
                
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    optimization_report["optimization_time_seconds"],
                    self.model_name,
                    metadata={"operation": "optimization"}
                )
            
            logger.info(
                "Edge optimization completed",
                model_name=self.model_name,
                size_reduction=f"{optimization_report['size_reduction_ratio']:.2%}",
                compression_ratio=f"{optimization_report['compression_ratio']:.2f}x"
            )
            
            return optimization_report
            
        except Exception as e:
            logger.error("Edge optimization failed", model_name=self.model_name, error=str(e))
            optimization_report["error"] = str(e)
            return optimization_report
    
    def _apply_pruning(self, pruning_ratio: float, structured: bool = False) -> None:
        """Apply pruning to reduce model size."""
        if structured:
            # Structured pruning - remove entire channels/filters
            for module in self.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    prune.ln_structured(
                        module, name="weight", amount=pruning_ratio, n=2, dim=0
                    )
        else:
            # Unstructured pruning - remove individual weights
            parameters_to_prune = []
            for module in self.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    parameters_to_prune.append((module, "weight"))
            
            if parameters_to_prune:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=pruning_ratio,
                )
        
        # Make pruning permanent
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
                prune.remove(module, "weight")
    
    def _apply_quantization(self) -> None:
        """Apply dynamic quantization for edge deployment."""
        try:
            # Dynamic quantization for Linear layers
            self.qconfig = torch.quantization.default_dynamic_qconfig
            torch.quantization.prepare(self, inplace=True)
            
            # For more aggressive quantization, we could use static quantization
            # but it requires calibration data
            
        except Exception as e:
            logger.warning("Quantization failed, skipping", error=str(e))
    
    def validate_model_accuracy(
        self,
        validation_data: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """Validate model accuracy after optimization."""
        if device is None:
            device = self.device
        
        self.eval()
        self.to(device)
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(validation_data):
                data, targets = data.to(device), targets.to(device)
                
                outputs = self(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # For classification tasks
                if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                    predicted = outputs.argmax(dim=1)
                    correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        validation_metrics = {
            "validation_loss": avg_loss,
            "validation_accuracy": accuracy,
            "total_samples": total_samples,
        }
        
        # Record validation metrics
        if self.metrics_collector:
            self.metrics_collector.record_ml_metrics(
                accuracy=accuracy,
                loss=avg_loss,
                component=self.model_name,
                additional_metrics={"validation_samples": total_samples}
            )
        
        logger.info(
            "Model validation completed",
            model_name=self.model_name,
            accuracy=accuracy,
            loss=avg_loss
        )
        
        return validation_metrics
    
    def save_model(self, filepath: str, include_optimizer: bool = False) -> None:
        """Save model with metadata."""
        save_dict = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_model_config(),
            "model_metadata": {
                "model_name": self.model_name,
                "optimization_applied": self.optimization_applied,
                "pruning_ratio": self.pruning_ratio,
                "quantized": self.quantized,
                "training_history": self.training_history,
                "saved_at": datetime.now().isoformat(),
            },
        }
        
        torch.save(save_dict, filepath)
        logger.info("Model saved", filepath=filepath, model_name=self.model_name)
    
    @classmethod
    def load_model(cls, filepath: str, metrics_collector: Optional[MetricsCollector] = None):
        """Load model with metadata."""
        checkpoint = torch.load(filepath, map_location="cpu")
        
        # Extract model configuration
        model_config = checkpoint["model_config"]
        model_metadata = checkpoint["model_metadata"]
        
        # Create model instance
        model = cls(
            model_name=model_metadata["model_name"],
            metrics_collector=metrics_collector,
            **model_config
        )
        
        # Load state and metadata
        model.load_state_dict(checkpoint["model_state_dict"])
        model.optimization_applied = model_metadata.get("optimization_applied", False)
        model.pruning_ratio = model_metadata.get("pruning_ratio", 0.0)
        model.quantized = model_metadata.get("quantized", False)
        model.training_history = model_metadata.get("training_history", {})
        
        logger.info("Model loaded", filepath=filepath, model_name=model.model_name)
        return model
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary for research analysis."""
        param_stats = self.count_parameters()
        
        summary = {
            "model_name": self.model_name,
            "model_size_bytes": self.get_model_size(),
            "model_size_mb": self.get_model_size() / (1024 * 1024),
            "parameters": param_stats,
            "optimization_status": {
                "optimization_applied": self.optimization_applied,
                "pruning_ratio": self.pruning_ratio,
                "quantized": self.quantized,
            },
            "device": str(self.device),
            "training_history": self.training_history,
        }
        
        return summary