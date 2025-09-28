"""
Blockchain Validation Integration for Federated Learning Research

This module provides Python integration with the blockchain-based
model validation system for EV charging optimization research.
"""

import hashlib
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch
import logging
from web3 import Web3
from eth_account import Account
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Metrics for model validation on blockchain."""
    accuracy: float
    loss: float
    client_count: int
    model_hash: str
    timestamp: int = None
    
    def to_blockchain_format(self) -> Dict:
        """Convert to blockchain-compatible format."""
        return {
            'accuracy': int(self.accuracy * 10000),  # Scale to integer
            'loss': int(self.loss * 10000),
            'clientCount': self.client_count,
            'modelHash': bytes.fromhex(self.model_hash.replace('0x', '')),
            'timestamp': self.timestamp or int(time.time())
        }


class ModelHashGenerator:
    """Generate reproducible hashes for ML models."""
    
    @staticmethod
    def hash_pytorch_model(model: torch.nn.Module) -> str:
        """Generate hash for PyTorch model state."""
        # Get model state dict
        state_dict = model.state_dict()
        
        # Convert to deterministic bytes
        model_bytes = b''
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            # Convert tensor to bytes
            tensor_bytes = tensor.detach().cpu().numpy().tobytes()
            model_bytes += key.encode('utf-8') + tensor_bytes
        
        # Generate SHA256 hash
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        return f"0x{model_hash}"
    
    @staticmethod
    def hash_model_updates(updates: List[Dict]) -> str:
        """Generate hash for aggregated model updates."""
        # Sort updates by client ID for deterministic hashing
        sorted_updates = sorted(updates, key=lambda x: x.get('client_id', 0))
        
        # Serialize updates
        update_bytes = json.dumps(sorted_updates, sort_keys=True).encode('utf-8')
        
        # Generate hash
        update_hash = hashlib.sha256(update_bytes).hexdigest()
        return f"0x{update_hash}"
    
    @staticmethod
    def hash_validation_data(metrics: ValidationMetrics) -> str:
        """Generate hash for validation data."""
        data_str = f"{metrics.accuracy}_{metrics.loss}_{metrics.client_count}_{metrics.timestamp}"
        data_hash = hashlib.sha256(data_str.encode('utf-8')).hexdigest()
        return f"0x{data_hash}"


class MockBlockchainValidator:
    """
    Mock blockchain validator for research when actual blockchain is not available.
    
    Simulates blockchain behavior for testing and development purposes.
    """
    
    def __init__(self):
        """Initialize mock blockchain validator."""
        self.validated_models = {}
        self.client_reputations = {}
        self.validation_history = []
        self.total_validations = 0
        
        # Enhanced validation rules for stronger security
        self.validation_rules = {
            'min_accuracy': 0.3,  # More lenient to avoid rejecting legitimate models
            'max_loss': 1000.0,  # Adjusted for realistic loss ranges
            'min_clients': 2,  # Reduced to handle smaller federations
            'max_accuracy_change': 0.2,  # Detect suspicious accuracy jumps
            'max_loss_change': 10.0,  # Detect suspicious loss changes
            'enabled': True
        }
        
        self.consensus_threshold = 0.6
        
    def register_client(self, client_address: str) -> bool:
        """Register a new client."""
        if client_address not in self.client_reputations:
            self.client_reputations[client_address] = {
                'total_validations': 0,
                'successful_validations': 0,
                'reputation': 0.5,  # 50% initial reputation
                'last_activity': time.time(),
                'is_active': True
            }
            logger.info(f"Registered client: {client_address}")
            return True
        return False
    
    def validate_model_update(self, client_address: str, 
                            metrics: ValidationMetrics) -> Tuple[bool, str]:
        """Validate a model update."""
        if not self.validation_rules['enabled']:
            return False, "Validation disabled"
        
        if client_address not in self.client_reputations:
            self.register_client(client_address)
        
        # Update client activity
        client_rep = self.client_reputations[client_address]
        client_rep['last_activity'] = time.time()
        client_rep['total_validations'] += 1
        
        # Check validation criteria
        if not self._meets_validation_criteria(metrics):
            return False, "Failed validation criteria"
        
        # Check for Byzantine behavior
        if self._detect_byzantine_behavior(client_address, metrics):
            self._penalize_client(client_address)
            return False, "Byzantine behavior detected"
        
        # Store validation
        self.validated_models[metrics.model_hash] = {
            'metrics': metrics,
            'client': client_address,
            'timestamp': time.time()
        }
        
        self.validation_history.append({
            'client': client_address,
            'model_hash': metrics.model_hash,
            'accuracy': metrics.accuracy,
            'loss': metrics.loss,
            'timestamp': time.time(),
            'success': True
        })
        
        self.total_validations += 1
        self._update_client_reputation(client_address, True)
        
        logger.info(f"Validated model update from {client_address}: {metrics.model_hash}")
        return True, "Validation successful"
    
    def _meets_validation_criteria(self, metrics: ValidationMetrics) -> bool:
        """Check if metrics meet validation criteria."""
        return (
            metrics.accuracy >= self.validation_rules['min_accuracy'] and
            metrics.loss <= self.validation_rules['max_loss'] and
            metrics.client_count >= self.validation_rules['min_clients']
        )
    
    def _detect_byzantine_behavior(self, client_address: str, 
                                 metrics: ValidationMetrics) -> bool:
        """Enhanced Byzantine behavior detection."""
        client_rep = self.client_reputations[client_address]
        
        # Check success rate over time
        if client_rep['total_validations'] > 3:
            success_rate = client_rep['successful_validations'] / client_rep['total_validations']
            if success_rate < 0.3:  # More lenient threshold
                return True
        
        # Check for extremely unrealistic values
        if metrics.accuracy > 1.0 or metrics.accuracy < -0.5:
            return True
        
        # Check for suspiciously high performance jumps
        if len(self.validation_history) > 0:
            recent_accuracies = [h['accuracy'] for h in self.validation_history[-5:] 
                               if h['client'] == client_address]
            if recent_accuracies:
                avg_recent = sum(recent_accuracies) / len(recent_accuracies)
                if metrics.accuracy - avg_recent > self.validation_rules['max_accuracy_change']:
                    return True
        
        # Check for loss anomalies
        if metrics.loss < 0 or metrics.loss > self.validation_rules['max_loss'] * 2:
            return True
        
        # Check client count anomalies
        if metrics.client_count < 1 or metrics.client_count > 50:
            return True
        
        return False
    
    def _update_client_reputation(self, client_address: str, successful: bool):
        """Update client reputation."""
        client_rep = self.client_reputations[client_address]
        
        if successful:
            client_rep['successful_validations'] += 1
            client_rep['reputation'] = min(1.0, client_rep['reputation'] + 0.01)
        else:
            client_rep['reputation'] = max(0.0, client_rep['reputation'] - 0.02)
        
        # Deactivate clients with very low reputation
        if client_rep['reputation'] < 0.1:
            client_rep['is_active'] = False
    
    def _penalize_client(self, client_address: str):
        """Penalize client for Byzantine behavior."""
        client_rep = self.client_reputations[client_address]
        client_rep['reputation'] = max(0.0, client_rep['reputation'] - 0.1)
        if client_rep['reputation'] < 0.1:
            client_rep['is_active'] = False
    
    def get_client_reputation(self, client_address: str) -> Dict:
        """Get client reputation information."""
        return self.client_reputations.get(client_address, {})
    
    def get_validation_statistics(self) -> Dict:
        """Get overall validation statistics."""
        active_clients = sum(1 for rep in self.client_reputations.values() if rep['is_active'])
        avg_reputation = np.mean([rep['reputation'] for rep in self.client_reputations.values()]) if self.client_reputations else 0
        
        return {
            'total_clients': len(self.client_reputations),
            'active_clients': active_clients,
            'total_validations': self.total_validations,
            'avg_reputation': avg_reputation
        }
    
    def is_model_validated(self, model_hash: str) -> bool:
        """Check if model is validated."""
        return model_hash in self.validated_models
    
    def validate_consensus(self, values: List[float], threshold: float = 0.8) -> bool:
        """
        Validate consensus among multiple values (e.g., accuracies).
        
        Args:
            values: List of values to check for consensus
            threshold: Minimum threshold for valid consensus
            
        Returns:
            True if consensus is valid, False otherwise
        """
        if not values:
            return False
            
        # Check if all values meet the threshold
        valid_values = [v for v in values if v >= threshold]
        
        # Calculate standard deviation to check consistency
        std_dev = np.std(values)
        mean_value = np.mean(values)
        
        # Consensus criteria:
        # 1. All values meet minimum threshold
        # 2. Values are reasonably consistent (low standard deviation)
        # 3. Mean is above threshold
        consensus_valid = (
            len(valid_values) == len(values) and  # All meet threshold
            std_dev < 0.1 and  # Low variation
            mean_value >= threshold  # Mean above threshold
        )
        
        logger.info(f"Consensus validation: {len(valid_values)}/{len(values)} valid, "
                   f"std={std_dev:.3f}, mean={mean_value:.3f}, result={consensus_valid}")
        
        return consensus_valid
    
    def simulate_consensus_validation(self, model_hash: str, 
                                    validators: List[str], 
                                    votes: List[bool]) -> bool:
        """Simulate consensus validation."""
        if len(validators) != len(votes):
            return False
        
        positive_votes = sum(votes)
        consensus_percentage = positive_votes / len(votes)
        
        consensus_reached = consensus_percentage >= self.consensus_threshold
        
        if consensus_reached:
            self.validated_models[model_hash] = {
                'consensus': True,
                'validators': validators,
                'votes': votes,
                'timestamp': time.time()
            }
            self.total_validations += 1
        
        return consensus_reached


class Web3BlockchainValidator:
    """
    Real blockchain validator using Web3 for actual blockchain integration.
    
    Note: This requires a running blockchain network and deployed smart contract.
    """
    
    def __init__(self, web3_provider: str, contract_address: str, 
                 contract_abi: List, private_key: str):
        """
        Initialize Web3 blockchain validator.
        
        Args:
            web3_provider: Web3 provider URL
            contract_address: Deployed contract address
            contract_abi: Contract ABI
            private_key: Private key for transactions
        """
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.account = Account.from_key(private_key)
        
        # Contract setup
        self.contract = self.w3.eth.contract(
            address=contract_address,
            abi=contract_abi
        )
        
        logger.info(f"Connected to blockchain at {web3_provider}")
        logger.info(f"Using contract at {contract_address}")
    
    def register_client(self, client_address: str) -> bool:
        """Register client on blockchain."""
        try:
            # Build transaction
            transaction = self.contract.functions.registerClient(client_address).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Client registered: {client_address}, Tx: {tx_hash.hex()}")
            return receipt.status == 1
            
        except Exception as e:
            logger.error(f"Failed to register client {client_address}: {e}")
            return False
    
    def validate_model_update(self, metrics: ValidationMetrics) -> Tuple[bool, str]:
        """Submit model validation to blockchain."""
        try:
            # Convert metrics to blockchain format
            blockchain_metrics = metrics.to_blockchain_format()
            
            # Build transaction
            transaction = self.contract.functions.validateModelUpdate(
                blockchain_metrics
            ).build_transaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logger.info(f"Model validated on blockchain: {metrics.model_hash}")
                return True, f"Validated, Tx: {tx_hash.hex()}"
            else:
                logger.warning(f"Validation failed on blockchain: {metrics.model_hash}")
                return False, "Transaction failed"
                
        except Exception as e:
            logger.error(f"Blockchain validation error: {e}")
            return False, str(e)
    
    def get_client_reputation(self, client_address: str) -> Dict:
        """Get client reputation from blockchain."""
        try:
            result = self.contract.functions.getClientReputation(client_address).call()
            
            return {
                'total_validations': result[0],
                'successful_validations': result[1],
                'reputation': result[2] / 10000,  # Convert back from scaled integer
                'last_activity': result[3],
                'is_active': result[4]
            }
        except Exception as e:
            logger.error(f"Failed to get reputation for {client_address}: {e}")
            return {}
    
    def get_validation_statistics(self) -> Dict:
        """Get validation statistics from blockchain."""
        try:
            result = self.contract.functions.getValidationStatistics().call()
            
            return {
                'total_clients': result[0],
                'active_clients': result[1],
                'total_validations': result[2],
                'avg_reputation': result[3] / 10000  # Convert back from scaled integer
            }
        except Exception as e:
            logger.error(f"Failed to get validation statistics: {e}")
            return {}
    
    def is_model_validated(self, model_hash: str) -> bool:
        """Check if model is validated on blockchain."""
        try:
            return self.contract.functions.isModelValidated(model_hash).call()
        except Exception as e:
            logger.error(f"Failed to check model validation: {e}")
            return False


class FederatedBlockchainIntegration:
    """
    Integration layer between federated learning and blockchain validation.
    
    Handles the complete workflow of model validation in federated learning.
    """
    
    def __init__(self, validator: Optional[Any] = None, use_mock: bool = True):
        """
        Initialize the integration.
        
        Args:
            validator: Blockchain validator instance
            use_mock: Whether to use mock validator for testing
        """
        if validator is None:
            if use_mock:
                self.validator = MockBlockchainValidator()
                logger.info("Using mock blockchain validator for research")
            else:
                raise ValueError("No validator provided and mock disabled")
        else:
            self.validator = validator
        
        self.hash_generator = ModelHashGenerator()
        self.validation_cache = {}
        
    def validate_federated_round(self, global_model: torch.nn.Module,
                                client_updates: List[Dict],
                                round_metrics: Dict) -> Tuple[bool, str]:
        """
        Validate a complete federated learning round.
        
        Args:
            global_model: Updated global model
            client_updates: List of client update information
            round_metrics: Round performance metrics
            
        Returns:
            Tuple of (validation_success, validation_message)
        """
        # Generate model hash
        model_hash = self.hash_generator.hash_pytorch_model(global_model)
        
        # Create validation metrics
        metrics = ValidationMetrics(
            accuracy=round_metrics.get('accuracy', 0.0),
            loss=round_metrics.get('loss', 0.0),
            client_count=len(client_updates),
            model_hash=model_hash,
            timestamp=int(time.time())
        )
        
        # Validate on blockchain
        if hasattr(self.validator, 'validate_model_update'):
            # For mock validator
            success, message = self.validator.validate_model_update("global_aggregator", metrics)
        else:
            # For Web3 validator
            success, message = self.validator.validate_model_update(metrics)
        
        # Cache result
        self.validation_cache[model_hash] = {
            'success': success,
            'message': message,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        return success, message
    
    def validate_client_contribution(self, client_id: str, 
                                   client_model: torch.nn.Module,
                                   client_metrics: Dict) -> Tuple[bool, str]:
        """
        Validate individual client contribution.
        
        Args:
            client_id: Client identifier
            client_model: Client's local model
            client_metrics: Client's training metrics
            
        Returns:
            Tuple of (validation_success, validation_message)
        """
        # Generate client model hash
        model_hash = self.hash_generator.hash_pytorch_model(client_model)
        
        # Create validation metrics
        metrics = ValidationMetrics(
            accuracy=client_metrics.get('accuracy', 0.0),
            loss=client_metrics.get('loss', 0.0),
            client_count=1,
            model_hash=model_hash,
            timestamp=int(time.time())
        )
        
        # Validate individual contribution
        if hasattr(self.validator, 'validate_model_update'):
            success, message = self.validator.validate_model_update(client_id, metrics)
        else:
            success, message = self.validator.validate_model_update(metrics)
        
        return success, message
    
    def get_client_trust_score(self, client_id: str) -> float:
        """Get trust score for a client."""
        reputation = self.validator.get_client_reputation(client_id)
        if reputation:
            return reputation.get('reputation', 0.0)
        return 0.0
    
    def get_system_security_metrics(self) -> Dict:
        """Get overall system security metrics."""
        stats = self.validator.get_validation_statistics()
        
        # Calculate additional security metrics
        recent_validations = len([v for v in self.validation_cache.values() 
                                if time.time() - v['timestamp'] < 3600])  # Last hour
        
        return {
            'blockchain_stats': stats,
            'recent_validations': recent_validations,
            'cached_validations': len(self.validation_cache),
            'avg_trust_score': stats.get('avg_reputation', 0.0)
        }
    
    def simulate_adversarial_scenario(self, n_byzantine_clients: int = 3,
                                    attack_severity: float = 0.5) -> Dict:
        """
        Simulate adversarial scenario for research.
        
        Args:
            n_byzantine_clients: Number of Byzantine clients
            attack_severity: Severity of the attack (0.0 to 1.0)
            
        Returns:
            Results of the adversarial simulation
        """
        results = {
            'attack_config': {
                'byzantine_clients': n_byzantine_clients,
                'severity': attack_severity
            },
            'detection_results': [],
            'system_response': {}
        }
        
        # Register Byzantine clients
        byzantine_clients = [f"byzantine_client_{i}" for i in range(n_byzantine_clients)]
        for client_id in byzantine_clients:
            self.validator.register_client(client_id)
        
        # Simulate attacks
        for i, client_id in enumerate(byzantine_clients):
            # Create malicious metrics
            malicious_metrics = ValidationMetrics(
                accuracy=np.random.uniform(0.95, 0.99) if attack_severity > 0.5 else np.random.uniform(0.1, 0.3),
                loss=np.random.uniform(0.001, 0.01) if attack_severity > 0.5 else np.random.uniform(10, 100),
                client_count=1,
                model_hash=f"0x{'a' * 64}",  # Fake hash
                timestamp=int(time.time())
            )
            
            # Attempt validation
            if hasattr(self.validator, 'validate_model_update'):
                success, message = self.validator.validate_model_update(client_id, malicious_metrics)
            else:
                success, message = self.validator.validate_model_update(malicious_metrics)
            
            results['detection_results'].append({
                'client_id': client_id,
                'detected': not success,
                'message': message,
                'metrics': malicious_metrics
            })
        
        # Get final system state
        results['system_response'] = self.get_system_security_metrics()
        
        # Calculate detection rate
        detected_attacks = sum(1 for r in results['detection_results'] if r['detected'])
        results['detection_rate'] = detected_attacks / len(results['detection_results'])
        
        return results