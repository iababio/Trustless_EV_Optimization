"""Federated Learning client implementation for EV charging stations."""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import OrderedDict
import structlog
from datetime import datetime
import hashlib
import json

from ..ml_models.base import EdgeMLModel
from ..ml_models.lstm import LightweightLSTM
from ..ml_models.xgboost_model import LightweightXGBoost
from ..data_pipeline.processor import EVChargingDataProcessor
from ..metrics.collector import MetricsCollector, MetricType

logger = structlog.get_logger(__name__)


class EVChargingClient(fl.client.NumPyClient):
    """Federated Learning client for EV charging optimization with comprehensive metrics."""
    
    def __init__(
        self,
        station_id: str,
        model: Union[EdgeMLModel, LightweightXGBoost],
        data_path: Optional[str] = None,
        local_data: Optional[pd.DataFrame] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        privacy_epsilon: float = 1.0,
        min_samples_for_training: int = 10,
    ):
        self.station_id = station_id
        self.model = model
        self.data_path = data_path
        self.local_data = local_data
        self.metrics_collector = metrics_collector
        self.privacy_epsilon = privacy_epsilon
        self.min_samples_for_training = min_samples_for_training
        
        # FL round tracking
        self.current_round = 0
        self.participation_history: List[Dict[str, Any]] = []
        
        # Local training state
        self.local_epochs = 0
        self.last_training_time = 0.0
        self.local_performance_history: List[Dict[str, Any]] = []
        
        # Data processor
        self.data_processor = EVChargingDataProcessor(metrics_collector)
        
        # Load and preprocess data if needed
        if self.local_data is None and self.data_path:
            self._load_and_process_data()
        
        logger.info(
            "EVChargingClient initialized",
            station_id=station_id,
            model_type=type(model).__name__,
            data_samples=len(self.local_data) if self.local_data is not None else 0
        )
    
    def _load_and_process_data(self) -> None:
        """Load and preprocess local charging data."""
        try:
            processed_data, report = self.data_processor.process_pipeline(self.data_path)
            self.local_data = processed_data
            
            logger.info(
                "Local data processed",
                station_id=self.station_id,
                processed_samples=len(processed_data),
                data_quality_score=report.get("validation_report", {}).get("data_quality_score", 0.0)
            )
            
        except Exception as e:
            logger.error("Local data processing failed", station_id=self.station_id, error=str(e))
            self.local_data = pd.DataFrame()
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Get model parameters for federated aggregation."""
        start_time = datetime.now()
        
        try:
            if isinstance(self.model, EdgeMLModel):
                # PyTorch model
                parameters = [param.cpu().numpy() for param in self.model.parameters()]
            elif isinstance(self.model, LightweightXGBoost):
                # XGBoost model - serialize booster
                booster_bytes = self.model.model.get_booster().save_raw("ubj")
                parameters = [np.frombuffer(booster_bytes, dtype=np.uint8)]
            else:
                raise ValueError(f"Unsupported model type: {type(self.model)}")
            
            # Record parameter extraction time
            if self.metrics_collector:
                extraction_time = (datetime.now() - start_time).total_seconds()
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    extraction_time,
                    f"fl_client_{self.station_id}",
                    metadata={"operation": "get_parameters"}
                )
            
            logger.debug(
                "Parameters extracted",
                station_id=self.station_id,
                parameter_arrays=len(parameters),
                total_params=sum(p.size for p in parameters)
            )
            
            return parameters
            
        except Exception as e:
            logger.error("Parameter extraction failed", station_id=self.station_id, error=str(e))
            return []
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from federated aggregation."""
        start_time = datetime.now()
        
        try:
            if isinstance(self.model, EdgeMLModel):
                # PyTorch model
                params_dict = zip(self.model.state_dict().keys(), parameters)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=False)
                
            elif isinstance(self.model, LightweightXGBoost):
                # XGBoost model - load from bytes
                booster_bytes = parameters[0].tobytes()
                self.model.model.get_booster().load_model(bytearray(booster_bytes))
            
            # Record parameter setting time
            if self.metrics_collector:
                setting_time = (datetime.now() - start_time).total_seconds()
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    setting_time,
                    f"fl_client_{self.station_id}",
                    metadata={"operation": "set_parameters"}
                )
            
            logger.debug(
                "Parameters updated",
                station_id=self.station_id,
                parameter_arrays=len(parameters)
            )
            
        except Exception as e:
            logger.error("Parameter setting failed", station_id=self.station_id, error=str(e))
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Local training with privacy preservation and comprehensive metrics."""
        round_start = datetime.now()
        self.current_round = config.get("server_round", self.current_round + 1)
        
        logger.info(
            "Starting local training",
            station_id=self.station_id,
            round=self.current_round
        )
        
        # Check if we have enough data for training
        if self.local_data is None or len(self.local_data) < self.min_samples_for_training:
            logger.warning(
                "Insufficient data for training",
                station_id=self.station_id,
                data_samples=len(self.local_data) if self.local_data is not None else 0,
                required=self.min_samples_for_training
            )
            
            # Return current parameters without training
            return self.get_parameters(config), 0, {"status": "insufficient_data"}
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Prepare local training data
        train_loader, val_loader = self._prepare_data_loaders(config)
        
        if train_loader is None:
            return self.get_parameters(config), 0, {"status": "data_preparation_failed"}
        
        try:
            # Perform local training
            training_metrics = self._local_training(train_loader, val_loader, config)
            
            # Apply differential privacy if configured
            if self.privacy_epsilon > 0:
                training_metrics.update(self._apply_differential_privacy())
            
            # Get updated parameters
            updated_parameters = self.get_parameters(config)
            
            # Calculate training results
            num_examples = len(self.local_data)
            round_time = (datetime.now() - round_start).total_seconds()
            
            # Store participation record
            participation_record = {
                "round": self.current_round,
                "training_time_seconds": round_time,
                "num_examples": num_examples,
                "training_metrics": training_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            self.participation_history.append(participation_record)
            
            # Record FL round metrics
            if self.metrics_collector:
                self.metrics_collector.record_fl_round_metrics(
                    round_id=self.current_round,
                    duration=round_time,
                    participants=1,  # This client
                    global_accuracy=training_metrics.get("train_accuracy", 0.0),
                    convergence_delta=training_metrics.get("loss_improvement", 0.0)
                )
                
                # Record privacy metrics
                if self.privacy_epsilon > 0:
                    self.metrics_collector.record_privacy_metrics(
                        epsilon_consumed=training_metrics.get("epsilon_consumed", 0.0),
                        privacy_budget_remaining=self.privacy_epsilon - training_metrics.get("epsilon_consumed", 0.0),
                        noise_magnitude=training_metrics.get("noise_magnitude", 0.0),
                        component=f"fl_client_{self.station_id}"
                    )
            
            logger.info(
                "Local training completed",
                station_id=self.station_id,
                round=self.current_round,
                training_time=round_time,
                num_examples=num_examples,
                train_loss=training_metrics.get("train_loss", "N/A")
            )
            
            return updated_parameters, num_examples, training_metrics
            
        except Exception as e:
            logger.error(
                "Local training failed",
                station_id=self.station_id,
                round=self.current_round,
                error=str(e)
            )
            return self.get_parameters(config), 0, {"status": "training_failed", "error": str(e)}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Local evaluation with comprehensive metrics."""
        eval_start = datetime.now()
        
        logger.info(
            "Starting local evaluation",
            station_id=self.station_id,
            round=self.current_round
        )
        
        # Check data availability
        if self.local_data is None or len(self.local_data) == 0:
            return float('inf'), 0, {"status": "no_evaluation_data"}
        
        # Set parameters for evaluation
        self.set_parameters(parameters)
        
        try:
            # Prepare evaluation data
            _, val_loader = self._prepare_data_loaders(config, train_split=0.8)
            
            if val_loader is None:
                return float('inf'), 0, {"status": "evaluation_data_preparation_failed"}
            
            # Perform evaluation
            eval_metrics = self._local_evaluation(val_loader)
            
            eval_time = (datetime.now() - eval_start).total_seconds()
            num_examples = len(self.local_data)
            
            # Record evaluation metrics
            if self.metrics_collector:
                self.metrics_collector.record_ml_metrics(
                    accuracy=eval_metrics.get("accuracy", 0.0),
                    loss=eval_metrics.get("loss", float('inf')),
                    component=f"fl_client_{self.station_id}",
                    additional_metrics={
                        "evaluation_time": eval_time,
                        "evaluation_samples": num_examples
                    }
                )
            
            logger.info(
                "Local evaluation completed",
                station_id=self.station_id,
                eval_time=eval_time,
                eval_loss=eval_metrics.get("loss", "N/A"),
                eval_accuracy=eval_metrics.get("accuracy", "N/A")
            )
            
            return eval_metrics.get("loss", float('inf')), num_examples, eval_metrics
            
        except Exception as e:
            logger.error(
                "Local evaluation failed",
                station_id=self.station_id,
                error=str(e)
            )
            return float('inf'), 0, {"status": "evaluation_failed", "error": str(e)}
    
    def _prepare_data_loaders(
        self,
        config: Dict[str, Any],
        train_split: float = 0.8,
        batch_size: int = 32,
    ) -> Tuple[Optional[torch.utils.data.DataLoader], Optional[torch.utils.data.DataLoader]]:
        """Prepare PyTorch data loaders for training."""
        if self.local_data is None or len(self.local_data) == 0:
            return None, None
        
        try:
            # Extract features and targets
            feature_columns = [col for col in self.local_data.columns if col != 'meter_total_wh']
            target_column = 'meter_total_wh'
            
            if target_column not in self.local_data.columns:
                logger.error("Target column not found in data", target_column=target_column)
                return None, None
            
            X = self.local_data[feature_columns].select_dtypes(include=[np.number]).fillna(0).values
            y = self.local_data[target_column].values
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # For time series models, reshape data
            if isinstance(self.model, EdgeMLModel) and hasattr(self.model, 'input_size'):
                # Create sequences for LSTM
                sequence_length = min(24, len(X))  # Use up to 24 time steps
                X_sequences, y_sequences = self._create_sequences(X_tensor, y_tensor, sequence_length)
                X_tensor, y_tensor = X_sequences, y_sequences
            
            # Create dataset
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            
            # Split into train and validation
            train_size = int(train_split * len(dataset))
            val_size = len(dataset) - train_size
            
            if val_size == 0:
                # Use all data for training if dataset is too small
                train_dataset = dataset
                val_dataset = None
            else:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=min(batch_size, len(train_dataset)),
                shuffle=True
            )
            
            val_loader = None
            if val_dataset:
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=min(batch_size, len(val_dataset)),
                    shuffle=False
                )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error("Data loader preparation failed", error=str(e))
            return None, None
    
    def _create_sequences(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sequence_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for time series models."""
        if len(X) < sequence_length:
            return X.unsqueeze(0), y[:1]
        
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - sequence_length + 1):
            seq_x = X[i:i + sequence_length]
            seq_y = y[i + sequence_length - 1]
            sequences_X.append(seq_x)
            sequences_y.append(seq_y)
        
        return torch.stack(sequences_X), torch.stack(sequences_y)
    
    def _local_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform local model training."""
        training_start = datetime.now()
        
        # Training configuration
        local_epochs = config.get("local_epochs", 5)
        learning_rate = config.get("learning_rate", 0.001)
        
        if isinstance(self.model, EdgeMLModel):
            # PyTorch model training
            return self._train_pytorch_model(train_loader, val_loader, local_epochs, learning_rate)
        elif isinstance(self.model, LightweightXGBoost):
            # XGBoost model training
            return self._train_xgboost_model(train_loader, val_loader, config)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
    
    def _train_pytorch_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        local_epochs: int,
        learning_rate: float,
    ) -> Dict[str, Any]:
        """Train PyTorch model locally."""
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(local_epochs):
            epoch_train_losses = []
            
            for batch_data, batch_targets in train_loader:
                batch_data, batch_targets = batch_data.to(self.model.device), batch_targets.to(self.model.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_train_losses.append(loss.item())
            
            avg_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader:
                val_loss = self._evaluate_pytorch_model(val_loader, criterion)
                val_losses.append(val_loss)
        
        training_metrics = {
            "train_loss": np.mean(train_losses),
            "train_accuracy": 1.0 / (1.0 + np.mean(train_losses)),  # Approximation
            "val_loss": np.mean(val_losses) if val_losses else None,
            "local_epochs": local_epochs,
            "loss_improvement": train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0,
        }
        
        return training_metrics
    
    def _train_xgboost_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Train XGBoost model locally."""
        # Convert data loaders to numpy arrays
        X_train, y_train = [], []
        for batch_data, batch_targets in train_loader:
            X_train.append(batch_data.numpy())
            y_train.append(batch_targets.numpy())
        
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train).ravel()
        
        X_val, y_val = None, None
        if val_loader:
            X_val, y_val = [], []
            for batch_data, batch_targets in val_loader:
                X_val.append(batch_data.numpy())
                y_val.append(batch_targets.numpy())
            
            X_val = np.vstack(X_val)
            y_val = np.vstack(y_val).ravel()
        
        # Train XGBoost model
        training_summary = self.model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on training data
        train_predictions = self.model.predict(X_train)
        train_loss = np.mean((y_train - train_predictions) ** 2)
        
        training_metrics = {
            "train_loss": train_loss,
            "train_accuracy": 1.0 / (1.0 + train_loss),
            "training_summary": training_summary,
        }
        
        if X_val is not None:
            val_predictions = self.model.predict(X_val)
            val_loss = np.mean((y_val - val_predictions) ** 2)
            training_metrics["val_loss"] = val_loss
        
        return training_metrics
    
    def _local_evaluation(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Perform local model evaluation."""
        if isinstance(self.model, EdgeMLModel):
            criterion = nn.MSELoss()
            loss = self._evaluate_pytorch_model(val_loader, criterion)
            return {
                "loss": loss,
                "accuracy": 1.0 / (1.0 + loss),
            }
        elif isinstance(self.model, LightweightXGBoost):
            # Convert val_loader to numpy
            X_val, y_val = [], []
            for batch_data, batch_targets in val_loader:
                X_val.append(batch_data.numpy())
                y_val.append(batch_targets.numpy())
            
            X_val = np.vstack(X_val)
            y_val = np.vstack(y_val).ravel()
            
            eval_metrics = self.model.evaluate(X_val, y_val)
            return eval_metrics
    
    def _evaluate_pytorch_model(
        self,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Evaluate PyTorch model."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_data, batch_targets in val_loader:
                batch_data, batch_targets = batch_data.to(self.model.device), batch_targets.to(self.model.device)
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_targets)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def _apply_differential_privacy(self) -> Dict[str, Any]:
        """Apply differential privacy to model parameters."""
        if not isinstance(self.model, EdgeMLModel):
            logger.warning("Differential privacy only supported for PyTorch models")
            return {}
        
        # Calculate noise scale based on privacy budget
        sensitivity = 1.0  # Simplified assumption
        noise_scale = sensitivity / self.privacy_epsilon
        
        # Add noise to model parameters
        total_noise_magnitude = 0.0
        param_count = 0
        
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    noise = torch.normal(0, noise_scale, param.shape)
                    param.data += noise
                    
                    total_noise_magnitude += torch.norm(noise).item()
                    param_count += 1
        
        avg_noise_magnitude = total_noise_magnitude / param_count if param_count > 0 else 0.0
        
        privacy_metrics = {
            "epsilon_consumed": 0.1,  # Simplified calculation
            "noise_magnitude": avg_noise_magnitude,
            "noise_scale": noise_scale,
        }
        
        logger.info(
            "Differential privacy applied",
            station_id=self.station_id,
            epsilon_consumed=privacy_metrics["epsilon_consumed"],
            noise_magnitude=avg_noise_magnitude
        )
        
        return privacy_metrics
    
    def get_client_info(self) -> Dict[str, Any]:
        """Get comprehensive client information for monitoring."""
        info = {
            "station_id": self.station_id,
            "model_type": type(self.model).__name__,
            "current_round": self.current_round,
            "total_rounds_participated": len(self.participation_history),
            "data_samples": len(self.local_data) if self.local_data is not None else 0,
            "privacy_epsilon": self.privacy_epsilon,
            "min_samples_for_training": self.min_samples_for_training,
        }
        
        if self.participation_history:
            latest_participation = self.participation_history[-1]
            info["last_training_time"] = latest_participation["training_time_seconds"]
            info["last_training_metrics"] = latest_participation["training_metrics"]
        
        return info