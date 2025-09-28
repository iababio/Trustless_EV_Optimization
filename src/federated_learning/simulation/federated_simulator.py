"""
Federated Learning Simulation Environment for EV Charging Research

This module implements a comprehensive federated learning simulation
environment for researching EV charging optimization across distributed nodes.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ClientType(Enum):
    """Types of federated learning clients."""
    HONEST = "honest"
    BYZANTINE = "byzantine"
    SLOW = "slow"
    DROPOUT = "dropout"


@dataclass
class ClientConfig:
    """Configuration for a federated learning client."""
    client_id: int
    client_type: ClientType = ClientType.HONEST
    data_samples: int = 0
    manufacturers: List[str] = field(default_factory=list)
    vehicle_categories: List[str] = field(default_factory=list)
    local_epochs: int = 1
    learning_rate: float = 0.01
    batch_size: int = 32
    dropout_probability: float = 0.0
    byzantine_severity: float = 0.0  # 0.0 to 1.0
    network_delay: float = 0.0  # seconds
    compute_capacity: float = 1.0  # relative capacity


@dataclass
class FederatedRound:
    """Results from a federated learning round."""
    round_number: int
    participating_clients: List[int]
    global_loss: float
    global_accuracy: float
    client_losses: Dict[int, float]
    client_accuracies: Dict[int, float]
    convergence_metrics: Dict[str, float]
    communication_cost: float
    round_duration: float
    model_update_size: float


class EVChargingClient:
    """
    Federated learning client for EV charging demand prediction.
    
    Simulates a charging station or fleet manager participating
    in federated learning for demand forecasting.
    """
    
    def __init__(self, config: ClientConfig, model_architecture: nn.Module):
        """
        Initialize the federated client.
        
        Args:
            config: Client configuration
            model_architecture: Neural network architecture
        """
        self.config = config
        self.client_id = config.client_id
        self.client_type = config.client_type
        
        # Initialize local model
        self.local_model = copy.deepcopy(model_architecture)
        self.optimizer = optim.SGD(
            self.local_model.parameters(), 
            lr=config.learning_rate
        )
        self.criterion = nn.MSELoss()
        
        # Client data
        self.local_data = None
        self.data_loader = None
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'participation': [],
            'updates_sent': 0,
            'updates_received': 0
        }
        
        # Simulation parameters
        self.is_available = True
        self.network_delay = config.network_delay
        self.compute_capacity = config.compute_capacity
        
    def set_local_data(self, X: np.ndarray, y: np.ndarray):
        """Set local training data for the client."""
        # Ensure all data is numeric and handle any remaining non-numeric values
        X_clean = np.array(X, dtype=np.float32)
        y_clean = np.array(y, dtype=np.float32)
        
        # Replace any NaN or invalid values with 0
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        y_clean = np.nan_to_num(y_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_clean)
        y_tensor = torch.FloatTensor(y_clean)
        
        # Create dataset and data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        self.local_data = (X_clean, y_clean)
        logger.info(f"Client {self.client_id}: Set local data with {len(X_clean)} samples")
    
    def update_model(self, global_model_state: Dict):
        """Update local model with global model parameters."""
        self.local_model.load_state_dict(global_model_state)
        self.training_history['updates_received'] += 1
    
    def local_training(self) -> Dict[str, Any]:
        """
        Perform local training on client data.
        
        Returns:
            Training results including loss and model updates
        """
        if self.data_loader is None:
            raise ValueError(f"Client {self.client_id}: No local data available")
        
        start_time = time.time()
        
        # Simulate network and compute delays
        if self.client_type == ClientType.SLOW:
            time.sleep(self.network_delay * random.uniform(2, 5))
        
        self.local_model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Local training epochs
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in self.data_loader:
                # Simulate compute capacity
                if self.compute_capacity < 1.0:
                    time.sleep((1 - self.compute_capacity) * 0.01)
                
                self.optimizer.zero_grad()
                outputs = self.local_model(batch_X)
                
                # Apply Byzantine attacks if configured
                if self.client_type == ClientType.BYZANTINE:
                    outputs = self._apply_byzantine_attack(outputs, batch_y)
                
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (num_batches * self.config.local_epochs) if num_batches > 0 else 0
        
        # Calculate training accuracy (R² for regression)
        self.local_model.eval()
        with torch.no_grad():
            predictions = []
            targets = []
            
            for batch_X, batch_y in self.data_loader:
                outputs = self.local_model(batch_X)
                # Handle scalar and vector outputs properly
                pred_values = outputs.squeeze().detach().numpy()
                if pred_values.ndim == 0:
                    predictions.append(pred_values.item())
                else:
                    predictions.extend(pred_values.tolist())
                    
                target_values = batch_y.detach().numpy()
                if target_values.ndim == 0:
                    targets.append(target_values.item())
                else:
                    targets.extend(target_values.tolist())
            
            # Calculate R² score
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history['losses'].append(avg_loss)
        self.training_history['accuracies'].append(r2_score)
        self.training_history['participation'].append(True)
        
        return {
            'loss': avg_loss,
            'accuracy': r2_score,
            'num_samples': len(self.local_data[0]),
            'training_time': training_time,
            'model_update': self.local_model.state_dict()
        }
    
    def _apply_byzantine_attack(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Apply Byzantine attack to model outputs."""
        if self.config.byzantine_severity == 0:
            return outputs
        
        # Random noise attack
        noise = torch.randn_like(outputs) * self.config.byzantine_severity
        return outputs + noise
    
    def is_available_for_round(self) -> bool:
        """Check if client is available for the current round."""
        if self.client_type == ClientType.DROPOUT:
            return random.random() > self.config.dropout_probability
        return self.is_available
    
    def get_client_info(self) -> Dict:
        """Get client information for monitoring."""
        return {
            'client_id': self.client_id,
            'client_type': self.client_type.value,
            'data_samples': len(self.local_data[0]) if self.local_data else 0,
            'manufacturers': self.config.manufacturers,
            'vehicle_categories': self.config.vehicle_categories,
            'participation_rate': sum(self.training_history['participation']) / max(len(self.training_history['participation']), 1)
        }


class FederatedAggregator:
    """Federated learning aggregation strategies."""
    
    @staticmethod
    def federated_averaging(client_updates: List[Dict[str, Any]], 
                          weights: Optional[List[float]] = None) -> Dict:
        """Standard FedAvg aggregation."""
        if not client_updates:
            raise ValueError("No client updates provided")
        
        if weights is None:
            weights = [update['num_samples'] for update in client_updates]
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter keys from first client
        param_keys = list(client_updates[0]['model_update'].keys())
        
        for key in param_keys:
            # Weighted average of parameters
            weighted_params = [
                client_updates[i]['model_update'][key] * weights[i]
                for i in range(len(client_updates))
            ]
            aggregated_params[key] = torch.stack(weighted_params).sum(dim=0)
        
        return aggregated_params
    
    @staticmethod
    def federated_median(client_updates: List[Dict[str, Any]]) -> Dict:
        """Median-based aggregation for Byzantine robustness."""
        if not client_updates:
            raise ValueError("No client updates provided")
        
        aggregated_params = {}
        param_keys = list(client_updates[0]['model_update'].keys())
        
        for key in param_keys:
            param_stack = torch.stack([
                client_updates[i]['model_update'][key] 
                for i in range(len(client_updates))
            ])
            
            # Compute median along client dimension
            aggregated_params[key] = torch.median(param_stack, dim=0)[0]
        
        return aggregated_params
    
    @staticmethod
    def trimmed_mean_aggregation(client_updates: List[Dict[str, Any]], 
                               trim_ratio: float = 0.2) -> Dict:
        """Trimmed mean aggregation for robustness."""
        if not client_updates:
            raise ValueError("No client updates provided")
        
        n_clients = len(client_updates)
        n_trim = int(n_clients * trim_ratio)
        
        aggregated_params = {}
        param_keys = list(client_updates[0]['model_update'].keys())
        
        for key in param_keys:
            param_stack = torch.stack([
                client_updates[i]['model_update'][key] 
                for i in range(len(client_updates))
            ])
            
            # Sort and trim
            sorted_params, _ = torch.sort(param_stack, dim=0)
            trimmed_params = sorted_params[n_trim:n_clients-n_trim]
            
            # Compute mean of trimmed values
            aggregated_params[key] = torch.mean(trimmed_params, dim=0)
        
        return aggregated_params


class FederatedChargingSimulator:
    """Comprehensive federated learning simulator for EV charging research."""
    
    def __init__(self, model_architecture: nn.Module, 
                 aggregation_strategy: str = "fedavg",
                 random_seed: int = 42):
        """
        Initialize the federated learning simulator.
        
        Args:
            model_architecture: Neural network architecture
            aggregation_strategy: Aggregation method ("fedavg", "median", "trimmed")
            random_seed: Random seed for reproducibility
        """
        self.model_architecture = model_architecture
        self.global_model = copy.deepcopy(model_architecture)
        # Use more secure aggregation by default
        self.aggregation_strategy = "median" if aggregation_strategy == "fedavg" else aggregation_strategy
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        
        # Simulation state
        self.clients = {}
        self.round_history = []
        self.current_round = 0
        
        # Aggregation function
        self.aggregator = self._get_aggregation_function()
        
        # Metrics tracking
        self.metrics = {
            'global_loss_history': [],
            'global_accuracy_history': [],
            'client_participation': defaultdict(list),
            'communication_costs': [],
            'convergence_metrics': []
        }
    
    def _get_aggregation_function(self) -> Callable:
        """Get aggregation function based on strategy."""
        if self.aggregation_strategy == "fedavg":
            return FederatedAggregator.federated_averaging
        elif self.aggregation_strategy == "median":
            return FederatedAggregator.federated_median
        elif self.aggregation_strategy == "trimmed":
            return FederatedAggregator.trimmed_mean_aggregation
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def add_client(self, config: ClientConfig, data: Tuple[np.ndarray, np.ndarray]):
        """Add a client to the federated system."""
        client = EVChargingClient(config, self.model_architecture)
        client.set_local_data(data[0], data[1])
        
        self.clients[config.client_id] = client
        logger.info(f"Added client {config.client_id} with {len(data[0])} samples")
    
    def create_clients_from_data(self, federated_data: Dict[int, pd.DataFrame],
                               target_col: str = 'Meter Total(Wh)',
                               feature_cols: Optional[List[str]] = None) -> List[ClientConfig]:
        """Create clients from federated data splits."""
        client_configs = []
        
        for client_id, data in federated_data.items():
            # Prepare features and targets
            if feature_cols is None:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols 
                              if col != target_col and 'ID' not in col.upper()]
            
            # Ensure we only use truly numeric columns and convert them properly
            X_df = data[feature_cols].copy()
            for col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            X_df = X_df.fillna(0)
            X = X_df.values.astype(np.float32)
            
            y = pd.to_numeric(data[target_col], errors='coerce').fillna(0).values.astype(np.float32)
            
            # Determine client characteristics
            manufacturers = data['Manufacturer'].unique().tolist() if 'Manufacturer' in data.columns else []
            categories = data['Category'].unique().tolist() if 'Category' in data.columns else []
            
            # Create client configuration with better training parameters
            config = ClientConfig(
                client_id=client_id,
                client_type=ClientType.HONEST,  # Default to honest
                data_samples=len(data),
                manufacturers=manufacturers,
                vehicle_categories=categories,
                local_epochs=5,  # Increased from 1 to 5 for better local training
                learning_rate=0.001,  # Reduced for more stable training
                batch_size=min(64, max(16, len(data) // 8))  # Better batch size calculation
            )
            
            # Add client
            self.add_client(config, (X, y))
            client_configs.append(config)
        
        return client_configs
    
    def simulate_byzantine_clients(self, n_byzantine: int, severity: float = 0.5):
        """Configure some clients as Byzantine."""
        client_ids = list(self.clients.keys())
        byzantine_clients = np.random.choice(client_ids, n_byzantine, replace=False)
        
        for client_id in byzantine_clients:
            self.clients[client_id].config.client_type = ClientType.BYZANTINE
            self.clients[client_id].config.byzantine_severity = severity
            self.clients[client_id].client_type = ClientType.BYZANTINE
        
        logger.info(f"Configured {n_byzantine} Byzantine clients with severity {severity}")
    
    def simulate_network_conditions(self, dropout_rate: float = 0.1, 
                                  slow_client_ratio: float = 0.2):
        """Simulate realistic network conditions."""
        client_ids = list(self.clients.keys())
        
        # Configure dropout clients
        n_dropout = int(len(client_ids) * dropout_rate)
        dropout_clients = np.random.choice(client_ids, n_dropout, replace=False)
        
        for client_id in dropout_clients:
            self.clients[client_id].config.client_type = ClientType.DROPOUT
            self.clients[client_id].config.dropout_probability = 0.3
            self.clients[client_id].client_type = ClientType.DROPOUT
        
        # Configure slow clients
        remaining_clients = [cid for cid in client_ids if cid not in dropout_clients]
        n_slow = int(len(remaining_clients) * slow_client_ratio)
        slow_clients = np.random.choice(remaining_clients, n_slow, replace=False)
        
        for client_id in slow_clients:
            self.clients[client_id].config.client_type = ClientType.SLOW
            self.clients[client_id].config.network_delay = np.random.uniform(0.1, 0.5)
            self.clients[client_id].config.compute_capacity = np.random.uniform(0.5, 0.8)
            self.clients[client_id].client_type = ClientType.SLOW
        
        logger.info(f"Configured network conditions: {n_dropout} dropout, {n_slow} slow clients")
    
    def run_federated_round(self, client_fraction: float = 1.0) -> FederatedRound:
        """Run a single federated learning round."""
        round_start_time = time.time()
        
        # Select participating clients
        available_clients = [
            client_id for client_id, client in self.clients.items()
            if client.is_available_for_round()
        ]
        
        n_selected = max(1, int(len(available_clients) * client_fraction))
        participating_clients = np.random.choice(available_clients, n_selected, replace=False)
        
        logger.info(f"Round {self.current_round}: {n_selected} clients participating")
        
        # Distribute global model to clients
        global_state = self.global_model.state_dict()
        for client_id in participating_clients:
            self.clients[client_id].update_model(global_state)
        
        # Collect client updates
        client_updates = []
        client_losses = {}
        client_accuracies = {}
        
        for client_id in participating_clients:
            try:
                update_result = self.clients[client_id].local_training()
                client_updates.append(update_result)
                client_losses[client_id] = update_result['loss']
                client_accuracies[client_id] = update_result['accuracy']
            except Exception as e:
                logger.warning(f"Client {client_id} failed in round {self.current_round}: {e}")
        
        # Aggregate updates
        if client_updates:
            aggregated_params = self.aggregator(client_updates)
            self.global_model.load_state_dict(aggregated_params)
        
        # Evaluate global model
        global_loss, global_accuracy = self._evaluate_global_model()
        
        # Calculate communication cost (simplified)
        model_size = sum(p.numel() for p in self.global_model.parameters())
        communication_cost = model_size * len(participating_clients) * 2  # Up + down
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(client_updates)
        
        round_duration = time.time() - round_start_time
        
        # Create round result
        round_result = FederatedRound(
            round_number=self.current_round,
            participating_clients=participating_clients.tolist(),
            global_loss=global_loss,
            global_accuracy=global_accuracy,
            client_losses=client_losses,
            client_accuracies=client_accuracies,
            convergence_metrics=convergence_metrics,
            communication_cost=communication_cost,
            round_duration=round_duration,
            model_update_size=model_size
        )
        
        # Update metrics
        self.metrics['global_loss_history'].append(global_loss)
        self.metrics['global_accuracy_history'].append(global_accuracy)
        self.metrics['communication_costs'].append(communication_cost)
        self.metrics['convergence_metrics'].append(convergence_metrics)
        
        for client_id in participating_clients:
            self.metrics['client_participation'][client_id].append(self.current_round)
        
        self.round_history.append(round_result)
        self.current_round += 1
        
        return round_result
    
    def _evaluate_global_model(self) -> Tuple[float, float]:
        """Evaluate global model on all client data."""
        self.global_model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for client in self.clients.values():
                if client.data_loader is not None:
                    for batch_X, batch_y in client.data_loader:
                        outputs = self.global_model(batch_X)
                        loss = nn.MSELoss()(outputs.squeeze(), batch_y)
                        
                        total_loss += loss.item() * len(batch_y)
                        total_samples += len(batch_y)
                        
                        # Handle scalar and vector outputs properly
                        pred_values = outputs.squeeze().detach().numpy()
                        if pred_values.ndim == 0:
                            all_predictions.append(pred_values.item())
                        else:
                            all_predictions.extend(pred_values.tolist())
                            
                        target_values = batch_y.detach().numpy()
                        if target_values.ndim == 0:
                            all_targets.append(target_values.item())
                        else:
                            all_targets.extend(target_values.tolist())
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        # Calculate R² as accuracy metric
        if all_predictions and all_targets:
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            r2_score = 0
        
        return avg_loss, r2_score
    
    def _calculate_convergence_metrics(self, client_updates: List[Dict]) -> Dict[str, float]:
        """Calculate convergence metrics for the round."""
        if len(client_updates) < 2:
            return {'client_variance': 0.0, 'update_norm': 0.0}
        
        # Calculate variance in client losses
        client_losses = [update['loss'] for update in client_updates]
        loss_variance = np.var(client_losses)
        
        # Calculate norm of aggregated update
        if client_updates:
            param_norms = []
            for key in client_updates[0]['model_update'].keys():
                param_values = [update['model_update'][key] for update in client_updates]
                param_stack = torch.stack(param_values)
                param_norms.append(torch.norm(param_stack).item())
            
            update_norm = np.mean(param_norms)
        else:
            update_norm = 0.0
        
        return {
            'client_variance': loss_variance,
            'update_norm': update_norm
        }
    
    def run_simulation(self, n_rounds: int, client_fraction: float = 1.0) -> Dict[str, Any]:
        """Run complete federated learning simulation."""
        logger.info(f"Starting federated simulation with {len(self.clients)} clients for {n_rounds} rounds")
        
        simulation_start = time.time()
        
        for round_num in range(n_rounds):
            round_result = self.run_federated_round(client_fraction)
            
            if round_num % 5 == 0 or round_num == n_rounds - 1:
                progress = (round_num + 1) / n_rounds * 100
                print(f"   Round {round_num + 1:3d}/{n_rounds} ({progress:5.1f}%): "
                      f"Loss={round_result.global_loss:.4f}, "
                      f"Accuracy={round_result.global_accuracy:.4f}, "
                      f"Clients={len(round_result.participating_clients)}")
                logger.info(
                    f"Round {round_num}: Loss={round_result.global_loss:.4f}, "
                    f"Accuracy={round_result.global_accuracy:.4f}, "
                    f"Clients={len(round_result.participating_clients)}"
                )
        
        simulation_duration = time.time() - simulation_start
        
        # Generate simulation summary
        summary = {
            'simulation_config': {
                'n_rounds': n_rounds,
                'n_clients': len(self.clients),
                'aggregation_strategy': self.aggregation_strategy,
                'client_fraction': client_fraction
            },
            'final_metrics': {
                'final_loss': self.metrics['global_loss_history'][-1] if self.metrics['global_loss_history'] else 0,
                'final_accuracy': self.metrics['global_accuracy_history'][-1] if self.metrics['global_accuracy_history'] else 0,
                'total_communication_cost': sum(self.metrics['communication_costs']),
                'avg_round_duration': np.mean([r.round_duration for r in self.round_history])
            },
            'client_statistics': self._generate_client_statistics(),
            'convergence_analysis': self._analyze_convergence(),
            'simulation_duration': simulation_duration,
            'round_history': self.round_history,
            'metrics': self.metrics
        }
        
        logger.info(f"Simulation completed in {simulation_duration:.2f} seconds")
        return summary
    
    def _generate_client_statistics(self) -> Dict:
        """Generate statistics about client participation and performance."""
        stats = {
            'client_types': defaultdict(int),
            'participation_rates': {},
            'avg_local_losses': {},
            'data_distribution': {}
        }
        
        for client_id, client in self.clients.items():
            # Count client types
            stats['client_types'][client.client_type.value] += 1
            
            # Calculate participation rate
            total_rounds = len(self.round_history)
            participated_rounds = len(self.metrics['client_participation'][client_id])
            stats['participation_rates'][client_id] = participated_rounds / total_rounds if total_rounds > 0 else 0
            
            # Average local loss
            if client.training_history['losses']:
                stats['avg_local_losses'][client_id] = np.mean(client.training_history['losses'])
            
            # Data distribution
            stats['data_distribution'][client_id] = {
                'samples': len(client.local_data[0]) if client.local_data else 0,
                'manufacturers': len(client.config.manufacturers),
                'categories': len(client.config.vehicle_categories)
            }
        
        return stats
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence properties of the federated training."""
        if not self.metrics['global_loss_history']:
            return {}
        
        loss_history = np.array(self.metrics['global_loss_history'])
        
        # Detect convergence point
        convergence_threshold = 0.001
        convergence_window = 10
        
        convergence_round = None
        for i in range(convergence_window, len(loss_history)):
            window_std = np.std(loss_history[i-convergence_window:i])
            if window_std < convergence_threshold:
                convergence_round = i
                break
        
        # Calculate convergence rate
        if len(loss_history) > 1:
            convergence_rate = (loss_history[0] - loss_history[-1]) / len(loss_history)
        else:
            convergence_rate = 0
        
        return {
            'converged': convergence_round is not None,
            'convergence_round': convergence_round,
            'convergence_rate': convergence_rate,
            'final_stability': np.std(loss_history[-10:]) if len(loss_history) >= 10 else np.inf
        }
    
    def get_client_info_summary(self) -> pd.DataFrame:
        """Get summary of all clients."""
        client_data = []
        
        for client_id, client in self.clients.items():
            info = client.get_client_info()
            info.update({
                'avg_loss': np.mean(client.training_history['losses']) if client.training_history['losses'] else 0,
                'training_rounds': len(client.training_history['losses'])
            })
            client_data.append(info)
        
        return pd.DataFrame(client_data)
    
    def save_simulation_results(self, filepath: str):
        """Save simulation results to file."""
        results = {
            'metrics': self.metrics,
            'client_summary': self.get_client_info_summary().to_dict('records'),
            'simulation_config': {
                'n_clients': len(self.clients),
                'aggregation_strategy': self.aggregation_strategy,
                'total_rounds': len(self.round_history)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Simulation results saved to {filepath}")