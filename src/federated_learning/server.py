"""Federated Learning server with trustless validation and comprehensive metrics."""

import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import structlog
from datetime import datetime
import hashlib
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle

from ..metrics.collector import MetricsCollector, MetricType
from ..blockchain.validator import ModelValidator

logger = structlog.get_logger(__name__)


class TrustlessStrategy(FedAvg):
    """Custom federated learning strategy with blockchain validation and advanced metrics."""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[callable] = None,
        evaluate_metrics_aggregation_fn: Optional[callable] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        blockchain_validator: Optional[ModelValidator] = None,
        enable_blockchain_validation: bool = False,
        quality_threshold: float = 0.1,  # Minimum improvement threshold
        max_validation_time: float = 30.0,  # Max time for validation
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.metrics_collector = metrics_collector
        self.blockchain_validator = blockchain_validator
        self.enable_blockchain_validation = enable_blockchain_validation
        self.quality_threshold = quality_threshold
        self.max_validation_time = max_validation_time
        
        # Round tracking
        self.current_round = 0
        self.round_history: List[Dict[str, Any]] = []
        
        # Client tracking
        self.client_contributions: Dict[str, List[Dict[str, Any]]] = {}
        self.client_reputation: Dict[str, float] = {}
        
        # Performance tracking
        self.global_model_history: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(
            "TrustlessStrategy initialized",
            min_fit_clients=min_fit_clients,
            blockchain_validation=enable_blockchain_validation,
            quality_threshold=quality_threshold
        )
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure the next round of federated learning with enhanced metrics."""
        round_start = datetime.now()
        self.current_round = server_round
        
        logger.info(f"Configuring FL round {server_round}")
        
        # Get standard configuration
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add custom configuration
        config.update({
            "server_round": server_round,
            "local_epochs": 5,
            "learning_rate": 0.001 * (0.95 ** (server_round - 1)),  # Learning rate decay
            "batch_size": 32,
        })
        
        # Select clients with reputation-based weighting
        fit_ins = fl.common.FitIns(parameters, config)
        
        # Sample clients with enhanced selection
        clients = self._select_clients_with_reputation(client_manager, self.fraction_fit)
        
        client_instructions = [(client, fit_ins) for client in clients]
        
        # Record configuration metrics
        if self.metrics_collector:
            config_time = (datetime.now() - round_start).total_seconds()
            self.metrics_collector.record_metric(
                MetricType.ROUND_DURATION,
                config_time,
                "fl_server",
                metadata={"operation": "configure_fit", "round": server_round}
            )
        
        logger.info(
            f"Round {server_round} configured",
            selected_clients=len(client_instructions),
            config=config
        )
        
        return client_instructions
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Aggregate model updates with trustless validation and comprehensive metrics."""
        aggregation_start = datetime.now()
        
        logger.info(
            f"Aggregating FL round {server_round}",
            successful_results=len(results),
            failures=len(failures)
        )
        
        if not results:
            logger.warning("No results to aggregate")
            return None, {}
        
        # Pre-validation of client contributions
        validated_results = []
        validation_metrics = []
        
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            validation_result = self._validate_client_contribution(
                client_id, fit_res, server_round
            )
            
            if validation_result["is_valid"]:
                validated_results.append((client_proxy, fit_res))
                validation_metrics.append(validation_result)
                
                # Update client reputation
                self._update_client_reputation(client_id, validation_result)
            else:
                logger.warning(
                    f"Client contribution rejected",
                    client_id=client_id,
                    reason=validation_result.get("rejection_reason", "unknown")
                )
        
        if not validated_results:
            logger.error("No valid contributions to aggregate")
            return None, {"status": "no_valid_contributions"}
        
        # Perform weighted aggregation
        aggregated_parameters = self._weighted_aggregate(validated_results, validation_metrics)
        
        # Blockchain validation (if enabled)
        blockchain_validation_result = {}
        if self.enable_blockchain_validation and self.blockchain_validator:
            blockchain_validation_result = self._blockchain_validate_aggregation(
                aggregated_parameters, server_round, validation_metrics
            )
        
        # Calculate aggregation metrics
        aggregation_time = (datetime.now() - aggregation_start).total_seconds()
        
        aggregation_metrics = {
            "server_round": server_round,
            "participants": len(validated_results),
            "rejected_contributions": len(results) - len(validated_results),
            "aggregation_time_seconds": aggregation_time,
            "validation_metrics": validation_metrics,
            "blockchain_validation": blockchain_validation_result,
            "quality_improvements": self._calculate_quality_improvements(validation_metrics),
        }
        
        # Store round history
        round_record = {
            "round": server_round,
            "timestamp": datetime.now().isoformat(),
            "participants": len(validated_results),
            "aggregation_metrics": aggregation_metrics,
            "global_model_quality": self._estimate_global_quality(validation_metrics),
        }
        self.round_history.append(round_record)
        
        # Record comprehensive FL metrics
        if self.metrics_collector:
            self.metrics_collector.record_fl_round_metrics(
                round_id=server_round,
                duration=aggregation_time,
                participants=len(validated_results),
                global_accuracy=round_record["global_model_quality"],
                convergence_delta=aggregation_metrics["quality_improvements"].get("avg_improvement", 0.0)
            )
            
            # Record blockchain metrics if applicable
            if blockchain_validation_result:
                self.metrics_collector.record_metric(
                    MetricType.VALIDATION_SUCCESS_RATE,
                    1.0 if blockchain_validation_result.get("validation_passed", False) else 0.0,
                    "fl_server",
                    metadata={"round": server_round}
                )
        
        logger.info(
            f"Round {server_round} aggregation completed",
            participants=len(validated_results),
            aggregation_time=aggregation_time,
            global_quality=round_record["global_model_quality"]
        )
        
        return aggregated_parameters, aggregation_metrics
    
    def _select_clients_with_reputation(
        self,
        client_manager: fl.server.ClientManager,
        fraction_fit: float,
    ) -> List[ClientProxy]:
        """Select clients based on reputation and availability."""
        all_clients = list(client_manager.all().values())
        
        if not all_clients:
            return []
        
        # Calculate selection probabilities based on reputation
        client_scores = []
        for client in all_clients:
            reputation = self.client_reputation.get(client.cid, 1.0)  # Default reputation
            availability_score = 1.0  # Could be enhanced with historical availability
            total_score = reputation * availability_score
            client_scores.append((client, total_score))
        
        # Sort by score and select top clients
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        num_clients = max(
            int(len(all_clients) * fraction_fit),
            min(self.min_fit_clients, len(all_clients))
        )
        
        selected_clients = [client for client, _ in client_scores[:num_clients]]
        
        return selected_clients
    
    def _validate_client_contribution(
        self,
        client_id: str,
        fit_res: FitRes,
        server_round: int,
    ) -> Dict[str, Any]:
        """Validate individual client contribution."""
        validation_start = datetime.now()
        
        validation_result = {
            "client_id": client_id,
            "is_valid": True,
            "validation_time": 0.0,
            "quality_score": 0.0,
            "rejection_reason": None,
        }
        
        try:
            # Extract metrics from client result
            client_metrics = fit_res.metrics
            num_examples = fit_res.num_examples
            
            # Basic validation checks
            if num_examples < 1:
                validation_result.update({
                    "is_valid": False,
                    "rejection_reason": "insufficient_examples"
                })
                return validation_result
            
            # Quality validation
            train_loss = client_metrics.get("train_loss", float('inf'))
            if train_loss > 1000 or train_loss != train_loss:  # Check for NaN
                validation_result.update({
                    "is_valid": False,
                    "rejection_reason": "invalid_loss"
                })
                return validation_result
            
            # Calculate quality score
            quality_score = self._calculate_contribution_quality(client_metrics, num_examples)
            validation_result["quality_score"] = quality_score
            
            # Check quality threshold
            if quality_score < self.quality_threshold:
                validation_result.update({
                    "is_valid": False,
                    "rejection_reason": "quality_below_threshold"
                })
                return validation_result
            
            # Parameter validation
            parameters = parameters_to_ndarrays(fit_res.parameters)
            if not self._validate_parameters(parameters):
                validation_result.update({
                    "is_valid": False,
                    "rejection_reason": "invalid_parameters"
                })
                return validation_result
            
            validation_result["validation_time"] = (datetime.now() - validation_start).total_seconds()
            
        except Exception as e:
            logger.error(f"Validation error for client {client_id}", error=str(e))
            validation_result.update({
                "is_valid": False,
                "rejection_reason": f"validation_error: {str(e)}"
            })
        
        return validation_result
    
    def _calculate_contribution_quality(
        self,
        client_metrics: Dict[str, Any],
        num_examples: int,
    ) -> float:
        """Calculate quality score for client contribution."""
        # Base quality from training metrics
        train_loss = client_metrics.get("train_loss", float('inf'))
        train_accuracy = client_metrics.get("train_accuracy", 0.0)
        
        # Quality based on loss improvement
        loss_quality = max(0, 1.0 - min(train_loss, 10.0) / 10.0)
        
        # Quality based on accuracy
        accuracy_quality = min(train_accuracy, 1.0)
        
        # Quality based on data quantity (logarithmic scaling)
        data_quality = min(1.0, np.log(num_examples + 1) / np.log(1000))
        
        # Weighted combination
        overall_quality = (
            0.4 * loss_quality +
            0.4 * accuracy_quality +
            0.2 * data_quality
        )
        
        return overall_quality
    
    def _validate_parameters(self, parameters: List[np.ndarray]) -> bool:
        """Validate parameter arrays for anomalies."""
        try:
            for param_array in parameters:
                # Check for NaN or infinite values
                if np.any(~np.isfinite(param_array)):
                    return False
                
                # Check for extreme values
                if np.any(np.abs(param_array) > 1e6):
                    return False
                
                # Check parameter magnitude
                param_norm = np.linalg.norm(param_array)
                if param_norm > 1e4:  # Arbitrary large threshold
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _weighted_aggregate(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        validation_metrics: List[Dict[str, Any]],
    ) -> Parameters:
        """Perform weighted aggregation based on client contribution quality."""
        # Extract parameters and weights
        parameters_list = []
        weights = []
        
        for (client_proxy, fit_res), validation_metric in zip(results, validation_metrics):
            parameters = parameters_to_ndarrays(fit_res.parameters)
            parameters_list.append(parameters)
            
            # Weight based on quality score and number of examples
            quality_weight = validation_metric["quality_score"]
            data_weight = fit_res.num_examples
            combined_weight = quality_weight * np.sqrt(data_weight)  # Square root to balance
            
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]
        
        # Perform weighted averaging
        aggregated_params = []
        for param_idx in range(len(parameters_list[0])):
            # Stack parameters from all clients for this layer
            param_stack = np.array([params[param_idx] for params in parameters_list])
            
            # Weighted average
            weighted_param = np.average(param_stack, axis=0, weights=weights)
            aggregated_params.append(weighted_param)
        
        return ndarrays_to_parameters(aggregated_params)
    
    def _blockchain_validate_aggregation(
        self,
        parameters: Parameters,
        server_round: int,
        validation_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate aggregated model on blockchain (if enabled)."""
        if not self.blockchain_validator:
            return {"validation_passed": False, "reason": "no_validator"}
        
        try:
            validation_start = datetime.now()
            
            # Calculate model hash
            param_arrays = parameters_to_ndarrays(parameters)
            model_bytes = pickle.dumps(param_arrays)
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            
            # Prepare validation metadata
            metadata = {
                "server_round": server_round,
                "participants": len(validation_metrics),
                "avg_quality_score": np.mean([m["quality_score"] for m in validation_metrics]),
                "total_examples": sum(m.get("num_examples", 0) for m in validation_metrics),
                "validation_timestamp": datetime.now().isoformat(),
            }
            
            # Perform blockchain validation
            validation_result = self.blockchain_validator.validate_model_update(
                model_hash=model_hash,
                metadata=json.dumps(metadata),
                round_id=server_round,
            )
            
            validation_time = (datetime.now() - validation_start).total_seconds()
            
            blockchain_result = {
                "validation_passed": validation_result.get("is_valid", False),
                "model_hash": model_hash,
                "validation_time_seconds": validation_time,
                "blockchain_tx_hash": validation_result.get("transaction_hash"),
                "validation_metadata": metadata,
            }
            
            logger.info(
                f"Blockchain validation completed",
                round=server_round,
                passed=blockchain_result["validation_passed"],
                validation_time=validation_time
            )
            
            return blockchain_result
            
        except Exception as e:
            logger.error("Blockchain validation failed", error=str(e))
            return {
                "validation_passed": False,
                "error": str(e),
                "validation_time_seconds": 0.0,
            }
    
    def _update_client_reputation(
        self,
        client_id: str,
        validation_result: Dict[str, Any],
    ) -> None:
        """Update client reputation based on contribution quality."""
        current_reputation = self.client_reputation.get(client_id, 1.0)
        quality_score = validation_result.get("quality_score", 0.0)
        
        # Reputation update with momentum
        learning_rate = 0.1
        new_reputation = (1 - learning_rate) * current_reputation + learning_rate * quality_score
        
        # Bound reputation between 0.1 and 2.0
        new_reputation = max(0.1, min(2.0, new_reputation))
        
        self.client_reputation[client_id] = new_reputation
        
        # Store contribution history
        if client_id not in self.client_contributions:
            self.client_contributions[client_id] = []
        
        contribution_record = {
            "round": self.current_round,
            "quality_score": quality_score,
            "reputation": new_reputation,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.client_contributions[client_id].append(contribution_record)
        
        logger.debug(
            f"Client reputation updated",
            client_id=client_id,
            new_reputation=new_reputation,
            quality_score=quality_score
        )
    
    def _calculate_quality_improvements(
        self,
        validation_metrics: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate quality improvement metrics."""
        if not validation_metrics:
            return {"avg_improvement": 0.0, "max_improvement": 0.0, "min_improvement": 0.0}
        
        quality_scores = [m["quality_score"] for m in validation_metrics]
        
        return {
            "avg_improvement": np.mean(quality_scores),
            "max_improvement": np.max(quality_scores),
            "min_improvement": np.min(quality_scores),
            "std_improvement": np.std(quality_scores),
        }
    
    def _estimate_global_quality(self, validation_metrics: List[Dict[str, Any]]) -> float:
        """Estimate global model quality from client contributions."""
        if not validation_metrics:
            return 0.0
        
        # Weighted average of client quality scores
        quality_scores = [m["quality_score"] for m in validation_metrics]
        return np.mean(quality_scores)
    
    def get_server_metrics(self) -> Dict[str, Any]:
        """Get comprehensive server metrics for monitoring."""
        metrics = {
            "current_round": self.current_round,
            "total_rounds": len(self.round_history),
            "active_clients": len(self.client_reputation),
            "avg_client_reputation": np.mean(list(self.client_reputation.values())) if self.client_reputation else 0.0,
            "blockchain_validation_enabled": self.enable_blockchain_validation,
        }
        
        if self.round_history:
            recent_rounds = self.round_history[-5:]  # Last 5 rounds
            metrics.update({
                "avg_participants_recent": np.mean([r["participants"] for r in recent_rounds]),
                "avg_global_quality_recent": np.mean([r["global_model_quality"] for r in recent_rounds]),
                "total_aggregation_time": sum([r["aggregation_metrics"]["aggregation_time_seconds"] for r in recent_rounds]),
            })
        
        return metrics


class FederatedLearningServer:
    """High-level federated learning server with comprehensive management."""
    
    def __init__(
        self,
        strategy: TrustlessStrategy,
        server_address: str = "0.0.0.0:8080",
        metrics_collector: Optional[MetricsCollector] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.strategy = strategy
        self.server_address = server_address
        self.metrics_collector = metrics_collector
        self.config = config or {}
        
        # Server state
        self.is_running = False
        self.server_thread: Optional[threading.Thread] = None
        
        logger.info(
            "FederatedLearningServer initialized",
            server_address=server_address,
            strategy_type=type(strategy).__name__
        )
    
    def start_server(self, num_rounds: int = 10, config_fn: Optional[callable] = None) -> Dict[str, Any]:
        """Start the federated learning server."""
        logger.info(f"Starting FL server for {num_rounds} rounds")
        
        server_config = fl.server.ServerConfig(num_rounds=num_rounds)
        
        # Configure server
        if config_fn is None:
            def config_fn(server_round: int) -> Dict[str, Any]:
                return {
                    "server_round": server_round,
                    "local_epochs": 5,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                }
        
        self.strategy.on_fit_config_fn = config_fn
        
        try:
            # Start server
            start_time = datetime.now()
            
            history = fl.server.start_server(
                server_address=self.server_address,
                config=server_config,
                strategy=self.strategy,
            )
            
            end_time = datetime.now()
            server_runtime = (end_time - start_time).total_seconds()
            
            # Compile server results
            server_results = {
                "total_runtime_seconds": server_runtime,
                "completed_rounds": num_rounds,
                "server_history": history,
                "strategy_metrics": self.strategy.get_server_metrics(),
                "client_reputation": self.strategy.client_reputation,
                "round_history": self.strategy.round_history,
            }
            
            # Record server completion metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.ROUND_DURATION,
                    server_runtime,
                    "fl_server",
                    metadata={"operation": "complete_server_run", "rounds": num_rounds}
                )
            
            logger.info(
                "FL server completed",
                rounds=num_rounds,
                runtime=server_runtime,
                final_participants=server_results["strategy_metrics"].get("active_clients", 0)
            )
            
            return server_results
            
        except Exception as e:
            logger.error("FL server failed", error=str(e))
            raise
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status and metrics."""
        status = {
            "is_running": self.is_running,
            "server_address": self.server_address,
            "current_round": self.strategy.current_round,
            "strategy_metrics": self.strategy.get_server_metrics(),
        }
        
        return status