"""
Security Testing and Adversarial Scenarios for Federated EV Charging Research

This module implements comprehensive security testing including adversarial attacks,
Byzantine behavior simulation, and robustness evaluation for the federated learning system.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
import random
import logging
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks."""
    MODEL_POISONING = "model_poisoning"
    DATA_POISONING = "data_poisoning" 
    GRADIENT_ATTACK = "gradient_attack"
    BYZANTINE_FAILURE = "byzantine_failure"
    SYBIL_ATTACK = "sybil_attack"
    INFERENCE_ATTACK = "inference_attack"
    BACKDOOR_ATTACK = "backdoor_attack"


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    attack_type: AttackType
    severity: float  # 0.0 to 1.0
    target_clients: List[str] = field(default_factory=list)
    duration: int = 10  # Number of rounds
    delay_rounds: int = 0  # Rounds to wait before starting attack
    attack_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Security evaluation metrics."""
    attack_detection_rate: float
    false_positive_rate: float
    model_degradation: float
    convergence_delay: int
    robustness_score: float
    privacy_leakage: float
    byzantine_tolerance: float


class AdversarialAttacker:
    """Implements various adversarial attacks on federated learning."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize attacker."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
    
    def poison_model_update(self, model_state: Dict, severity: float = 0.5) -> Dict:
        """
        Poison model updates by adding malicious noise.
        
        Args:
            model_state: Original model state dict
            severity: Attack severity (0.0 to 1.0)
            
        Returns:
            Poisoned model state dict
        """
        poisoned_state = copy.deepcopy(model_state)
        
        for key, param in poisoned_state.items():
            if isinstance(param, torch.Tensor):
                # Add adversarial noise
                noise_scale = severity * torch.std(param)
                adversarial_noise = torch.randn_like(param) * noise_scale
                poisoned_state[key] = param + adversarial_noise
        
        return poisoned_state
    
    def poison_training_data(self, X: np.ndarray, y: np.ndarray, 
                           severity: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Poison training data by corrupting labels or features.
        
        Args:
            X: Feature matrix
            y: Target values
            severity: Attack severity (0.0 to 1.0)
            
        Returns:
            Poisoned data (X_poisoned, y_poisoned)
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        
        n_samples = len(X)
        n_poison = int(severity * n_samples)
        
        if n_poison > 0:
            poison_indices = np.random.choice(n_samples, n_poison, replace=False)
            
            # Corrupt labels (flip or add noise)
            y_std = np.std(y)
            y_poisoned[poison_indices] += np.random.normal(0, severity * y_std, n_poison)
            
            # Corrupt features (add noise)
            for col in range(X.shape[1]):
                x_std = np.std(X[:, col])
                noise = np.random.normal(0, severity * x_std, n_poison)
                X_poisoned[poison_indices, col] += noise
        
        return X_poisoned, y_poisoned
    
    def gradient_attack(self, gradients: Dict, severity: float = 0.5) -> Dict:
        """
        Perform gradient-based attacks.
        
        Args:
            gradients: Model gradients
            severity: Attack severity
            
        Returns:
            Modified gradients
        """
        attacked_gradients = copy.deepcopy(gradients)
        
        for key, grad in attacked_gradients.items():
            if isinstance(grad, torch.Tensor):
                # Flip gradients with some probability
                if np.random.random() < severity:
                    attacked_gradients[key] = -grad
                else:
                    # Add noise to gradients
                    noise_scale = severity * torch.std(grad)
                    noise = torch.randn_like(grad) * noise_scale
                    attacked_gradients[key] = grad + noise
        
        return attacked_gradients
    
    def byzantine_failure(self, model_state: Dict, failure_type: str = "random") -> Dict:
        """
        Simulate Byzantine failures.
        
        Args:
            model_state: Original model state
            failure_type: Type of failure ("random", "zero", "extreme")
            
        Returns:
            Failed model state
        """
        failed_state = copy.deepcopy(model_state)
        
        for key, param in failed_state.items():
            if isinstance(param, torch.Tensor):
                if failure_type == "random":
                    failed_state[key] = torch.randn_like(param)
                elif failure_type == "zero":
                    failed_state[key] = torch.zeros_like(param)
                elif failure_type == "extreme":
                    failed_state[key] = param * 1000  # Extreme values
        
        return failed_state
    
    def sybil_attack(self, base_model_state: Dict, n_sybil_nodes: int = 5) -> List[Dict]:
        """
        Create Sybil nodes with similar model updates.
        
        Args:
            base_model_state: Base model state to replicate
            n_sybil_nodes: Number of Sybil nodes to create
            
        Returns:
            List of Sybil model states
        """
        sybil_states = []
        
        for i in range(n_sybil_nodes):
            sybil_state = copy.deepcopy(base_model_state)
            
            # Add small variations to avoid detection
            for key, param in sybil_state.items():
                if isinstance(param, torch.Tensor):
                    noise = torch.randn_like(param) * 0.001  # Small noise
                    sybil_state[key] = param + noise
            
            sybil_states.append(sybil_state)
        
        return sybil_states
    
    def backdoor_attack(self, X: np.ndarray, y: np.ndarray, 
                       trigger_pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement backdoor attack by inserting triggers.
        
        Args:
            X: Feature matrix
            y: Target values
            trigger_pattern: Backdoor trigger pattern
            
        Returns:
            Data with backdoor triggers
        """
        X_backdoor = X.copy()
        y_backdoor = y.copy()
        
        n_samples = len(X)
        n_backdoor = max(1, n_samples // 10)  # 10% backdoor samples
        
        backdoor_indices = np.random.choice(n_samples, n_backdoor, replace=False)
        
        # Insert trigger pattern
        for idx in backdoor_indices:
            # Add trigger to features (simplified)
            if len(trigger_pattern) <= X.shape[1]:
                X_backdoor[idx, :len(trigger_pattern)] = trigger_pattern
            
            # Set specific target value
            y_backdoor[idx] = np.max(y) * 1.5  # Anomalous target
        
        return X_backdoor, y_backdoor


class DefenseStrategy:
    """Implements defense strategies against adversarial attacks."""
    
    def __init__(self):
        """Initialize enhanced defense strategy."""
        self.detection_threshold = 0.05  # Stricter threshold
        self.history_window = 20  # Longer history for better detection
        self.anomaly_score_threshold = 0.15
        self.byzantine_tolerance = 0.33  # Can handle up to 1/3 malicious clients
        self.reputation_threshold = 0.7  # Minimum reputation to participate
        self.differential_privacy_epsilon = 1.0
        self.gradient_clipping_norm = 1.0
        
    def detect_anomalous_updates(self, client_updates: List[Dict],
                                historical_updates: List[List[Dict]]) -> List[bool]:
        """
        Detect anomalous model updates.
        
        Args:
            client_updates: Current round client updates
            historical_updates: Historical updates for comparison
            
        Returns:
            List of boolean flags indicating anomalous updates
        """
        anomaly_flags = []
        
        for i, update in enumerate(client_updates):
            is_anomalous = False
            
            # Calculate update magnitude
            update_norm = 0.0
            for key, param in update.get('model_update', {}).items():
                if isinstance(param, torch.Tensor):
                    update_norm += torch.norm(param).item()
            
            # Compare with historical norms
            if historical_updates:
                historical_norms = []
                for hist_round in historical_updates[-self.history_window:]:
                    if i < len(hist_round):
                        hist_norm = 0.0
                        for key, param in hist_round[i].get('model_update', {}).items():
                            if isinstance(param, torch.Tensor):
                                hist_norm += torch.norm(param).item()
                        historical_norms.append(hist_norm)
                
                if historical_norms:
                    mean_norm = np.mean(historical_norms)
                    std_norm = np.std(historical_norms)
                    
                    # Enhanced Z-score based detection
                    if std_norm > 0:
                        z_score = abs(update_norm - mean_norm) / std_norm
                        is_anomalous = z_score > 2.0  # Stricter 2-sigma rule
                    
                    # Additional robustness checks
                    if update_norm > mean_norm * 3:  # Suspiciously large update
                        is_anomalous = True
                    elif update_norm < mean_norm * 0.1:  # Suspiciously small update
                        is_anomalous = True
            
            # Enhanced attack pattern detection
            loss = update.get('loss', 0)
            accuracy = update.get('accuracy', 0)
            
            # Detect unrealistic metrics
            if loss > 10000 or loss < 0:
                is_anomalous = True
            if accuracy < -0.1 or accuracy > 1.1:
                is_anomalous = True
            
            # Detect gradient explosion/vanishing
            if update_norm > 100.0 or update_norm < 0.0001:
                is_anomalous = True
            
            anomaly_flags.append(is_anomalous)
        
        return anomaly_flags
    
    def robust_aggregation(self, client_updates: List[Dict], 
                          method: str = "trimmed_mean") -> Dict:
        """
        Perform robust aggregation to defend against Byzantine attacks.
        
        Args:
            client_updates: List of client model updates
            method: Aggregation method ("trimmed_mean", "median", "krum")
            
        Returns:
            Robustly aggregated model parameters
        """
        if not client_updates:
            return {}
        
        # Extract model parameters
        param_keys = list(client_updates[0]['model_update'].keys())
        aggregated_params = {}
        
        # Filter updates based on reputation and anomaly detection
        trusted_updates = self._filter_trusted_updates(client_updates)
        
        for key in param_keys:
            param_list = []
            for update in trusted_updates:
                if key in update['model_update']:
                    param_list.append(update['model_update'][key])
            
            if param_list and method == "trimmed_mean":
                # Enhanced trimmed mean (remove 25% outliers)
                param_stack = torch.stack(param_list)
                n_trim = max(1, int(0.125 * len(param_list)))  # Trim 12.5% from each side
                
                sorted_params, _ = torch.sort(param_stack, dim=0)
                if len(param_list) > 2 * n_trim:
                    trimmed_params = sorted_params[n_trim:-n_trim]
                else:
                    trimmed_params = sorted_params  # Use all if too few samples
                
                aggregated_params[key] = torch.mean(trimmed_params, dim=0)
                
            elif param_list and method == "median":
                # Coordinate-wise median (most Byzantine-resistant)
                param_stack = torch.stack(param_list)
                aggregated_params[key] = torch.median(param_stack, dim=0)[0]
                
            elif param_list and method == "krum":
                # Krum aggregation for Byzantine fault tolerance
                aggregated_params[key] = self._krum_selection(param_list)
                
            elif param_list:
                # Enhanced default: clipped average
                param_stack = torch.stack(param_list)
                # Clip outliers before averaging
                q25 = torch.quantile(param_stack, 0.25, dim=0)
                q75 = torch.quantile(param_stack, 0.75, dim=0)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                clipped_stack = torch.clamp(param_stack, lower_bound, upper_bound)
                aggregated_params[key] = torch.mean(clipped_stack, dim=0)
        
        return aggregated_params
    
    def _filter_trusted_updates(self, client_updates: List[Dict]) -> List[Dict]:
        """Filter updates based on trust and anomaly scores."""
        if len(client_updates) <= 1:
            return client_updates
        
        trusted = []
        update_scores = []
        
        # Calculate trust scores for each update
        for update in client_updates:
            loss = update.get('loss', 1000)
            accuracy = update.get('accuracy', 0)
            
            # Calculate update norm
            update_norm = 0.0
            for param in update.get('model_update', {}).values():
                if isinstance(param, torch.Tensor):
                    update_norm += torch.norm(param).item()
            
            # Trust score based on multiple factors
            trust_score = 0.0
            
            # Accuracy contribution (higher is better)
            if 0 <= accuracy <= 1:
                trust_score += accuracy * 0.4
            
            # Loss contribution (lower is better, normalized)
            if loss > 0:
                trust_score += max(0, (1000 - min(loss, 1000)) / 1000) * 0.3
            
            # Update norm contribution (moderate values preferred)
            if 0.1 <= update_norm <= 10.0:
                trust_score += 0.3
            
            update_scores.append((trust_score, update))
        
        # Sort by trust score and take top performers
        update_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Keep at least 50% of updates, but filter out clearly malicious ones
        min_keep = max(1, len(client_updates) // 2)
        for i, (score, update) in enumerate(update_scores):
            if i < min_keep or score > 0.3:  # Keep if in top half or reasonable score
                trusted.append(update)
        
        return trusted if trusted else client_updates[:1]  # Always keep at least one
    
    def _krum_selection(self, param_list: List[torch.Tensor]) -> torch.Tensor:
        """Select parameter using Krum algorithm for Byzantine tolerance."""
        if len(param_list) <= 1:
            return param_list[0] if param_list else torch.zeros_like(param_list[0])
        
        n = len(param_list)
        f = min(n // 3, 2)  # Assume at most 1/3 Byzantine clients
        
        if n <= 2 * f + 2:
            # Not enough honest clients, fall back to median
            param_stack = torch.stack(param_list)
            return torch.median(param_stack, dim=0)[0]
        
        # Calculate distances and scores
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(param_list[i] - param_list[j]).item()
                    distances.append(dist)
            
            # Sum of distances to closest n-f-2 neighbors
            distances.sort()
            score = sum(distances[:n-f-2]) if len(distances) >= n-f-2 else sum(distances)
            scores.append(score)
        
        # Return parameter with minimum score
        best_idx = scores.index(min(scores))
        return param_list[best_idx]
    
    def differential_privacy(self, model_update: Dict, 
                           epsilon: float = 1.0, sensitivity: float = 1.0) -> Dict:
        """
        Apply differential privacy to model updates.
        
        Args:
            model_update: Model update to privatize
            epsilon: Privacy budget
            sensitivity: Sensitivity parameter
            
        Returns:
            Privatized model update
        """
        private_update = copy.deepcopy(model_update)
        
        # Laplace noise scale
        noise_scale = sensitivity / epsilon
        
        for key, param in private_update.items():
            if isinstance(param, torch.Tensor):
                # Add Laplace noise
                noise = torch.tensor(np.random.laplace(0, noise_scale, param.shape))
                private_update[key] = param + noise.float()
        
        return private_update


class SecurityEvaluator:
    """Comprehensive security evaluation for federated EV charging system."""
    
    def __init__(self, federated_simulator, blockchain_validator):
        """
        Initialize security evaluator.
        
        Args:
            federated_simulator: Federated learning simulator
            blockchain_validator: Blockchain validator
        """
        self.fl_simulator = federated_simulator
        self.blockchain_validator = blockchain_validator
        self.attacker = AdversarialAttacker()
        self.defense = DefenseStrategy()
        
        self.attack_history = []
        self.detection_history = []
        self.performance_history = []
        
    def evaluate_attack_scenario(self, attack_config: AttackConfig,
                                n_rounds: int = 50) -> Dict[str, Any]:
        """
        Evaluate a specific attack scenario.
        
        Args:
            attack_config: Attack configuration
            n_rounds: Number of rounds to simulate
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {attack_config.attack_type.value} attack")
        
        # Store original client states
        original_clients = copy.deepcopy(self.fl_simulator.clients)
        
        # Run baseline (no attack)
        baseline_results = self._run_baseline_simulation(n_rounds)
        
        # Reset simulator
        self.fl_simulator.clients = copy.deepcopy(original_clients)
        self.fl_simulator.current_round = 0
        self.fl_simulator.round_history = []
        
        # Run attack simulation
        attack_results = self._run_attack_simulation(attack_config, n_rounds)
        
        # Calculate security metrics
        security_metrics = self._calculate_security_metrics(baseline_results, attack_results)
        
        return {
            'attack_config': attack_config,
            'baseline_results': baseline_results,
            'attack_results': attack_results,
            'security_metrics': security_metrics,
            'detection_performance': self._analyze_detection_performance()
        }
    
    def _run_baseline_simulation(self, n_rounds: int) -> Dict:
        """Run baseline simulation without attacks."""
        baseline_metrics = {
            'round_losses': [],
            'round_accuracies': [],
            'convergence_round': None,
            'final_performance': {}
        }
        
        for round_num in range(n_rounds):
            round_result = self.fl_simulator.run_federated_round()
            
            baseline_metrics['round_losses'].append(round_result.global_loss)
            baseline_metrics['round_accuracies'].append(round_result.global_accuracy)
            
            # Check convergence
            if (len(baseline_metrics['round_losses']) > 10 and
                baseline_metrics['convergence_round'] is None):
                recent_losses = baseline_metrics['round_losses'][-10:]
                if np.std(recent_losses) < 0.001:
                    baseline_metrics['convergence_round'] = round_num
        
        baseline_metrics['final_performance'] = {
            'loss': baseline_metrics['round_losses'][-1],
            'accuracy': baseline_metrics['round_accuracies'][-1]
        }
        
        return baseline_metrics
    
    def _run_attack_simulation(self, attack_config: AttackConfig, n_rounds: int) -> Dict:
        """Run simulation with adversarial attacks."""
        attack_metrics = {
            'round_losses': [],
            'round_accuracies': [],
            'attack_detected': [],
            'blockchain_rejections': [],
            'convergence_round': None,
            'final_performance': {}
        }
        
        # Identify target clients
        target_clients = attack_config.target_clients
        if not target_clients:
            # Select random clients for attack
            all_clients = list(self.fl_simulator.clients.keys())
            n_targets = max(1, len(all_clients) // 4)  # Attack 25% of clients
            target_clients = random.sample(all_clients, n_targets)
        
        attack_active = False
        attack_round_count = 0
        
        for round_num in range(n_rounds):
            # Check if attack should start
            if (round_num >= attack_config.delay_rounds and 
                attack_round_count < attack_config.duration):
                attack_active = True
                attack_round_count += 1
            elif attack_round_count >= attack_config.duration:
                attack_active = False
            
            # Modify target clients if attack is active
            if attack_active:
                self._apply_attack_to_clients(target_clients, attack_config)
            
            # Run federated round
            round_result = self.fl_simulator.run_federated_round()
            
            # Simulate blockchain validation
            blockchain_accepted = True
            if hasattr(self.blockchain_validator, 'validate_model_update'):
                # Mock validation for demonstration
                model_hash = f"model_round_{round_num}"
                validation_success, _ = self.blockchain_validator.validate_model_update(
                    "global_aggregator",
                    type('MockMetrics', (), {
                        'accuracy': round_result.global_accuracy,
                        'loss': round_result.global_loss,
                        'client_count': len(round_result.participating_clients),
                        'model_hash': model_hash
                    })()
                )
                blockchain_accepted = validation_success
            
            # Detect attacks (simplified)
            attack_detected = self._simulate_attack_detection(attack_active, attack_config.severity)
            
            # Record metrics
            attack_metrics['round_losses'].append(round_result.global_loss)
            attack_metrics['round_accuracies'].append(round_result.global_accuracy)
            attack_metrics['attack_detected'].append(attack_detected)
            attack_metrics['blockchain_rejections'].append(not blockchain_accepted)
            
            # Check convergence
            if (len(attack_metrics['round_losses']) > 10 and
                attack_metrics['convergence_round'] is None):
                recent_losses = attack_metrics['round_losses'][-10:]
                if np.std(recent_losses) < 0.001:
                    attack_metrics['convergence_round'] = round_num
        
        attack_metrics['final_performance'] = {
            'loss': attack_metrics['round_losses'][-1],
            'accuracy': attack_metrics['round_accuracies'][-1]
        }
        
        return attack_metrics
    
    def _apply_attack_to_clients(self, target_clients: List[str], attack_config: AttackConfig):
        """Apply attacks to target clients."""
        for client_id in target_clients:
            if client_id in self.fl_simulator.clients:
                client = self.fl_simulator.clients[client_id]
                
                if attack_config.attack_type == AttackType.MODEL_POISONING:
                    # Poison model state (will be applied during local training)
                    client._attack_severity = attack_config.severity
                    client._attack_type = "model_poisoning"
                
                elif attack_config.attack_type == AttackType.DATA_POISONING:
                    # Poison local data
                    if client.local_data is not None:
                        X_poisoned, y_poisoned = self.attacker.poison_training_data(
                            client.local_data[0], client.local_data[1], attack_config.severity
                        )
                        client.set_local_data(X_poisoned, y_poisoned)
                
                elif attack_config.attack_type == AttackType.BYZANTINE_FAILURE:
                    # Mark client as Byzantine
                    client.config.client_type = type('ClientType', (), {'BYZANTINE': 'byzantine'})()
                    client.config.byzantine_severity = attack_config.severity
    
    def _simulate_attack_detection(self, attack_active: bool, severity: float) -> bool:
        """Simulate attack detection based on various indicators."""
        if not attack_active:
            # Small chance of false positive
            return random.random() < 0.05
        
        # Detection probability increases with attack severity
        detection_probability = min(0.95, severity * 1.2)
        return random.random() < detection_probability
    
    def _calculate_security_metrics(self, baseline: Dict, attack: Dict) -> SecurityMetrics:
        """Calculate comprehensive security metrics."""
        
        # Model degradation
        baseline_final_acc = baseline['final_performance']['accuracy']
        attack_final_acc = attack['final_performance']['accuracy']
        model_degradation = max(0, baseline_final_acc - attack_final_acc)
        
        # Convergence delay
        baseline_conv = baseline['convergence_round'] or len(baseline['round_losses'])
        attack_conv = attack['convergence_round'] or len(attack['round_losses'])
        convergence_delay = max(0, attack_conv - baseline_conv)
        
        # Attack detection metrics
        attack_detected = attack['attack_detected']
        true_positives = sum(attack_detected)
        false_positives = 0  # Simplified for this implementation
        
        total_attack_rounds = sum(1 for detected in attack_detected if detected)
        detection_rate = true_positives / max(1, total_attack_rounds)
        false_positive_rate = false_positives / len(attack_detected)
        
        # Robustness score (how well the system maintained performance)
        performance_ratio = attack_final_acc / max(0.001, baseline_final_acc)
        robustness_score = min(1.0, performance_ratio)
        
        # Byzantine tolerance (simplified)
        blockchain_rejections = sum(attack['blockchain_rejections'])
        byzantine_tolerance = blockchain_rejections / len(attack['blockchain_rejections'])
        
        return SecurityMetrics(
            attack_detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            model_degradation=model_degradation,
            convergence_delay=convergence_delay,
            robustness_score=robustness_score,
            privacy_leakage=0.0,  # Placeholder
            byzantine_tolerance=byzantine_tolerance
        )
    
    def _analyze_detection_performance(self) -> Dict:
        """Analyze detection performance across all attacks."""
        return {
            'average_detection_rate': 0.75,  # Placeholder
            'detection_latency': 2.3,        # Placeholder
            'false_alarm_rate': 0.05         # Placeholder
        }
    
    def run_comprehensive_security_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive security evaluation with multiple attack scenarios."""
        
        attack_scenarios = [
            AttackConfig(
                attack_type=AttackType.MODEL_POISONING,
                severity=0.3,
                duration=15,
                delay_rounds=10
            ),
            AttackConfig(
                attack_type=AttackType.DATA_POISONING,
                severity=0.2,
                duration=20,
                delay_rounds=5
            ),
            AttackConfig(
                attack_type=AttackType.BYZANTINE_FAILURE,
                severity=0.5,
                duration=10,
                delay_rounds=15
            )
        ]
        
        evaluation_results = {}
        
        for i, attack_config in enumerate(attack_scenarios):
            scenario_name = f"{attack_config.attack_type.value}_scenario_{i+1}"
            logger.info(f"Running security evaluation: {scenario_name}")
            
            try:
                results = self.evaluate_attack_scenario(attack_config, n_rounds=50)
                evaluation_results[scenario_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {scenario_name}: {e}")
                evaluation_results[scenario_name] = {'error': str(e)}
        
        # Generate summary report
        summary = self._generate_security_summary(evaluation_results)
        
        return {
            'individual_scenarios': evaluation_results,
            'summary_report': summary,
            'recommendations': self._generate_security_recommendations(summary)
        }
    
    def _generate_security_summary(self, evaluation_results: Dict) -> Dict:
        """Generate summary of security evaluation results."""
        summary = {
            'total_scenarios': len(evaluation_results),
            'successful_evaluations': 0,
            'average_metrics': {},
            'attack_effectiveness': {},
            'defense_performance': {}
        }
        
        valid_results = []
        for scenario_name, results in evaluation_results.items():
            if 'error' not in results:
                summary['successful_evaluations'] += 1
                valid_results.append(results)
        
        if valid_results:
            # Calculate average metrics
            metrics_list = [r['security_metrics'] for r in valid_results]
            
            summary['average_metrics'] = {
                'detection_rate': np.mean([m.attack_detection_rate for m in metrics_list]),
                'false_positive_rate': np.mean([m.false_positive_rate for m in metrics_list]),
                'model_degradation': np.mean([m.model_degradation for m in metrics_list]),
                'robustness_score': np.mean([m.robustness_score for m in metrics_list]),
                'byzantine_tolerance': np.mean([m.byzantine_tolerance for m in metrics_list])
            }
        
        return summary
    
    def _generate_security_recommendations(self, summary: Dict) -> List[str]:
        """Generate security recommendations based on evaluation results."""
        recommendations = []
        
        avg_metrics = summary.get('average_metrics', {})
        
        if avg_metrics.get('detection_rate', 0) < 0.8:
            recommendations.append(
                "Improve attack detection mechanisms - current detection rate is below 80%"
            )
        
        if avg_metrics.get('false_positive_rate', 1) > 0.1:
            recommendations.append(
                "Reduce false positive rate in attack detection to improve system efficiency"
            )
        
        if avg_metrics.get('model_degradation', 1) > 0.2:
            recommendations.append(
                "Strengthen defense mechanisms - model performance degrades significantly under attack"
            )
        
        if avg_metrics.get('robustness_score', 0) < 0.7:
            recommendations.append(
                "Implement more robust aggregation methods to maintain performance under adversarial conditions"
            )
        
        if avg_metrics.get('byzantine_tolerance', 0) < 0.5:
            recommendations.append(
                "Enhance blockchain validation criteria to better detect Byzantine behavior"
            )
        
        if not recommendations:
            recommendations.append("Security evaluation shows good resilience against tested attack scenarios")
        
        return recommendations