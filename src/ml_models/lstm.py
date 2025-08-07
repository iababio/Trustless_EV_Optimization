"""Lightweight LSTM models for EV charging demand forecasting with edge optimization."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import numpy as np
import structlog

from .base import EdgeMLModel
from ..metrics.collector import MetricsCollector, MetricType

logger = structlog.get_logger(__name__)


class LightweightLSTM(EdgeMLModel):
    """Optimized LSTM model for edge deployment in EV charging stations."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        super().__init__("LightweightLSTM", metrics_collector)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers with optimizations for edge deployment
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Compact output layers
        self.output_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, output_size),
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        logger.info(
            "LightweightLSTM initialized",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            parameters=self.count_parameters()["total_parameters"]
        )
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    # Input-hidden weights
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    # Hidden-hidden weights
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # Initialize biases with small non-zero values
                    nn.init.uniform_(param, -0.1, 0.1)
                    # Set forget gate bias to 1 for better gradient flow
                    if 'bias_ih' in name:
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)
            elif 'output_layers' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    # Initialize biases with small non-zero values
                    nn.init.uniform_(param, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through the LSTM."""
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use the last time step for prediction
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_output_size)
        
        # Generate prediction
        prediction = self.output_layers(last_output)
        
        return prediction
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states."""
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        c0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )
        
        return (h0, c0)
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        future_steps: int = 1,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate multi-step predictions."""
        self.eval()
        
        with torch.no_grad():
            batch_size = x.shape[0]
            predictions = []
            current_input = x.clone()
            
            hidden = self.init_hidden(batch_size, x.device)
            
            for step in range(future_steps):
                # Single step prediction
                pred = self.forward(current_input, hidden)
                predictions.append(pred)
                
                # Update input for next prediction
                # Use the prediction as the next input (teacher forcing disabled)
                new_input = torch.cat([
                    current_input[:, 1:, :],  # Remove first timestep
                    pred.unsqueeze(1).expand(-1, -1, current_input.shape[2])  # Add prediction
                ], dim=1)
                
                current_input = new_input
        
        predictions_tensor = torch.stack(predictions, dim=1)  # (batch_size, future_steps, output_size)
        
        result = {"predictions": predictions_tensor}
        
        if return_attention:
            # Simplified attention weights (for interpretability)
            result["attention_weights"] = torch.ones(batch_size, future_steps) / future_steps
        
        return result
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
        }
    
    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """Train the LSTM model with comprehensive metrics tracking."""
        training_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        training_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if training_start_time:
            training_start_time.record()
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        # ReduceLROnPlateau scheduler (without verbose for compatibility)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience//2, factor=0.5
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
            "epoch_times": [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info("Starting LSTM training", num_epochs=num_epochs, learning_rate=learning_rate)
        
        for epoch in range(num_epochs):
            epoch_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if epoch_start:
                epoch_start.record()
            
            # Training phase
            self.train()
            train_losses = []
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                self.eval()
                val_losses = []
                
                with torch.no_grad():
                    for data, targets in val_loader:
                        data, targets = data.to(self.device), targets.to(self.device)
                        outputs = self.forward(data)
                        val_loss = criterion(outputs, targets)
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history["val_loss"].append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            history["learning_rates"].append(optimizer.param_groups[0]['lr'])
            
            if epoch_end:
                epoch_end.record()
                torch.cuda.synchronize()
                epoch_time = epoch_start.elapsed_time(epoch_end) / 1000.0  # Convert to seconds
                history["epoch_times"].append(epoch_time)
            
            # Record training metrics
            if self.metrics_collector:
                self.metrics_collector.record_ml_metrics(
                    accuracy=1.0 / (1.0 + avg_train_loss),  # Approximation for regression
                    loss=avg_train_loss,
                    component=self.model_name,
                    additional_metrics={
                        "val_loss": avg_val_loss if val_loader else None,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1,
                    }
                )
            
            if (epoch + 1) % 10 == 0:
                val_info = f", Val Loss: {avg_val_loss:.6f}" if val_loader else ""
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}{val_info}")
        
        if training_end_time:
            training_end_time.record()
            torch.cuda.synchronize()
            total_training_time = training_start_time.elapsed_time(training_end_time) / 1000.0
        else:
            total_training_time = sum(history.get("epoch_times", [0]))
        
        # Store training history
        self.training_history = history
        
        training_summary = {
            "total_epochs": len(history["train_loss"]),
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "best_val_loss": best_val_loss if val_loader else None,
            "total_training_time_seconds": total_training_time,
            "average_epoch_time": np.mean(history.get("epoch_times", [0])),
        }
        
        logger.info(
            "LSTM training completed",
            final_train_loss=training_summary["final_train_loss"],
            total_time=training_summary["total_training_time_seconds"]
        )
        
        return training_summary


class AttentionLSTM(EdgeMLModel):
    """LSTM with attention mechanism for improved forecasting accuracy."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        attention_dim: int = 16,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        super().__init__("AttentionLSTM", metrics_collector)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.attention_dim = attention_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, output_size),
        )
        
        self._initialize_weights()
        
        logger.info(
            "AttentionLSTM initialized",
            input_size=input_size,
            hidden_size=hidden_size,
            attention_dim=attention_dim,
            parameters=self.count_parameters()["total_parameters"]
        )
    
    def _initialize_weights(self) -> None:
        """Initialize weights for LSTM and attention layers."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    if 'bias_ih' in name:
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.0)
            elif 'attention' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'output_projection' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention mechanism."""
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Attention computation
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, hidden_size)
        
        # Final prediction
        prediction = self.output_projection(context_vector)
        
        return prediction, attention_weights.squeeze(-1)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
            "attention_dim": self.attention_dim,
        }