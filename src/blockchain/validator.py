"""Blockchain-based model validator for trustless FL validation."""

import json
import hashlib
from typing import Dict, Any, Optional, List
import structlog
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from dataclasses import dataclass

# Optional blockchain dependencies
try:
    from web3 import Web3
    from eth_account import Account
    HAS_BLOCKCHAIN = True
except ImportError:
    HAS_BLOCKCHAIN = False
    Web3 = None
    Account = None

from ..metrics.collector import MetricsCollector, MetricType

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    is_valid: bool
    model_hash: str
    transaction_hash: Optional[str] = None
    gas_used: Optional[int] = None
    validation_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ModelValidator:
    """Blockchain-based model validator with comprehensive metrics tracking."""
    
    def __init__(
        self,
        web3_provider_url: str,
        contract_address: str,
        contract_abi: List[Dict[str, Any]],
        private_key: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        gas_price_strategy: str = "medium",
        max_gas_limit: int = 500000,
    ):
        if not HAS_BLOCKCHAIN:
            raise ImportError(
                "Blockchain dependencies not available. "
                "Install with: pip install web3 eth-account eth-utils"
            )
        
        self.web3_provider_url = web3_provider_url
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.contract_abi = contract_abi
        self.private_key = private_key
        self.metrics_collector = metrics_collector
        self.gas_price_strategy = gas_price_strategy
        self.max_gas_limit = max_gas_limit
        
        # Initialize Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(web3_provider_url))
        
        # Initialize account if private key provided
        self.account = None
        if private_key:
            self.account = Account.from_key(private_key)
        
        # Initialize contract
        try:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
            
            # Verify contract is accessible
            if not self._verify_contract_connection():
                raise ConnectionError("Cannot connect to smart contract")
                
        except Exception as e:
            logger.error("Contract initialization failed", error=str(e))
            raise
        
        # Validation cache to prevent duplicate validations
        self.validation_cache: Dict[str, ValidationResult] = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(
            "ModelValidator initialized",
            contract_address=self.contract_address,
            network=self._get_network_name(),
            has_account=self.account is not None
        )
    
    def _verify_contract_connection(self) -> bool:
        """Verify connection to smart contract."""
        try:
            # Try to call a read-only function
            if hasattr(self.contract.functions, 'owner'):
                self.contract.functions.owner().call()
            return True
        except Exception as e:
            logger.error("Contract connection verification failed", error=str(e))
            return False
    
    def _get_network_name(self) -> str:
        """Get network name from chain ID."""
        try:
            chain_id = self.w3.eth.chain_id
            network_names = {
                1: "mainnet",
                3: "ropsten", 
                4: "rinkeby",
                5: "goerli",
                42: "kovan",
                1337: "localhost",
                31337: "hardhat"
            }
            return network_names.get(chain_id, f"unknown_{chain_id}")
        except:
            return "unknown"
    
    def validate_model_update(
        self,
        model_hash: str,
        metadata: str,
        round_id: int,
        timeout: float = 30.0,
    ) -> ValidationResult:
        """Validate model update on blockchain with comprehensive metrics."""
        validation_start = datetime.now()
        
        # Check cache first
        cache_key = f"{model_hash}_{round_id}"
        if cache_key in self.validation_cache:
            cached_result = self.validation_cache[cache_key]
            logger.info("Using cached validation result", model_hash=model_hash[:16])
            return cached_result
        
        logger.info(
            "Starting blockchain validation",
            model_hash=model_hash[:16],
            round_id=round_id
        )
        
        try:
            # Prepare validation transaction
            validation_result = self._execute_validation_transaction(
                model_hash, metadata, round_id, timeout
            )
            
            validation_time = (datetime.now() - validation_start).total_seconds()
            validation_result.validation_time = validation_time
            
            # Cache successful validations
            if validation_result.is_valid:
                self.validation_cache[cache_key] = validation_result
            
            # Record validation metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.TRANSACTION_TIME,
                    validation_time,
                    "blockchain_validator",
                    metadata={
                        "round_id": round_id,
                        "validation_passed": validation_result.is_valid
                    }
                )
                
                if validation_result.gas_used:
                    self.metrics_collector.record_metric(
                        MetricType.GAS_COST,
                        validation_result.gas_used,
                        "blockchain_validator",
                        metadata={"round_id": round_id}
                    )
            
            logger.info(
                "Blockchain validation completed",
                model_hash=model_hash[:16],
                is_valid=validation_result.is_valid,
                validation_time=validation_time,
                gas_used=validation_result.gas_used
            )
            
            return validation_result
            
        except Exception as e:
            validation_time = (datetime.now() - validation_start).total_seconds()
            error_result = ValidationResult(
                is_valid=False,
                model_hash=model_hash,
                validation_time=validation_time,
                error_message=str(e)
            )
            
            logger.error(
                "Blockchain validation failed",
                model_hash=model_hash[:16],
                error=str(e),
                validation_time=validation_time
            )
            
            return error_result
    
    def _execute_validation_transaction(
        self,
        model_hash: str,
        metadata: str,
        round_id: int,
        timeout: float,
    ) -> ValidationResult:
        """Execute the validation transaction on blockchain."""
        if not self.account:
            # Read-only validation (no transaction)
            return self._read_only_validation(model_hash, metadata, round_id)
        
        # Prepare transaction data
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        gas_price = self._get_gas_price()
        
        # Prepare function call
        function_call = self.contract.functions.validateModelUpdate(
            model_hash,
            metadata,
            round_id
        )
        
        # Estimate gas
        try:
            gas_estimate = function_call.estimate_gas({'from': self.account.address})
            gas_limit = min(int(gas_estimate * 1.2), self.max_gas_limit)  # 20% buffer
        except Exception as e:
            logger.warning(f"Gas estimation failed, using default: {e}")
            gas_limit = self.max_gas_limit // 2
        
        # Build transaction
        transaction = function_call.build_transaction({
            'from': self.account.address,
            'nonce': nonce,
            'gas': gas_limit,
            'gasPrice': gas_price,
        })
        
        # Sign and send transaction
        signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for transaction receipt
        try:
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
            
            # Parse transaction result
            is_valid = self._parse_validation_result(tx_receipt)
            
            return ValidationResult(
                is_valid=is_valid,
                model_hash=model_hash,
                transaction_hash=tx_receipt['transactionHash'].hex(),
                gas_used=tx_receipt['gasUsed'],
                metadata={"round_id": round_id, "block_number": tx_receipt['blockNumber']}
            )
            
        except Exception as e:
            logger.error("Transaction execution failed", tx_hash=tx_hash.hex(), error=str(e))
            return ValidationResult(
                is_valid=False,
                model_hash=model_hash,
                transaction_hash=tx_hash.hex(),
                error_message=f"Transaction failed: {str(e)}"
            )
    
    def _read_only_validation(
        self,
        model_hash: str,
        metadata: str,
        round_id: int,
    ) -> ValidationResult:
        """Perform read-only validation without blockchain transaction."""
        try:
            # Call read-only validation function
            if hasattr(self.contract.functions, 'isValidModel'):
                is_valid = self.contract.functions.isValidModel(model_hash).call()
            else:
                # Simple validation logic
                is_valid = self._simple_validation(model_hash, metadata)
            
            return ValidationResult(
                is_valid=is_valid,
                model_hash=model_hash,
                metadata={"round_id": round_id, "validation_type": "read_only"}
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                model_hash=model_hash,
                error_message=f"Read-only validation failed: {str(e)}"
            )
    
    def _simple_validation(self, model_hash: str, metadata: str) -> bool:
        """Simple validation logic when contract calls fail."""
        try:
            # Basic checks
            if len(model_hash) != 64:  # SHA256 hash should be 64 hex characters
                return False
            
            # Check if metadata is valid JSON
            metadata_dict = json.loads(metadata)
            
            # Check required fields
            required_fields = ["server_round", "participants", "avg_quality_score"]
            for field in required_fields:
                if field not in metadata_dict:
                    return False
            
            # Basic quality checks
            if metadata_dict.get("avg_quality_score", 0) < 0.1:
                return False
            
            if metadata_dict.get("participants", 0) < 1:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _get_gas_price(self) -> int:
        """Get appropriate gas price based on strategy."""
        try:
            if self.gas_price_strategy == "fast":
                return int(self.w3.eth.gas_price * 1.5)
            elif self.gas_price_strategy == "slow":
                return int(self.w3.eth.gas_price * 0.8)
            else:  # medium
                return self.w3.eth.gas_price
        except:
            return 20000000000  # 20 Gwei fallback
    
    def _parse_validation_result(self, tx_receipt: Dict[str, Any]) -> bool:
        """Parse validation result from transaction receipt."""
        try:
            # Check if transaction was successful
            if tx_receipt['status'] != 1:
                return False
            
            # Parse events/logs for validation result
            if tx_receipt.get('logs'):
                # Try to decode logs
                for log in tx_receipt['logs']:
                    try:
                        decoded_log = self.contract.events.ModelValidated().process_log(log)
                        return decoded_log['args'].get('isValid', False)
                    except:
                        continue
            
            # If no events found, assume success if transaction succeeded
            return True
            
        except Exception as e:
            logger.error("Failed to parse validation result", error=str(e))
            return False
    
    def get_validation_history(
        self,
        from_block: int = 0,
        to_block: str = "latest",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get validation history from blockchain."""
        try:
            # Get ModelValidated events
            event_filter = self.contract.events.ModelValidated.create_filter(
                fromBlock=from_block,
                toBlock=to_block
            )
            
            events = event_filter.get_all_entries()
            
            # Process events
            validation_history = []
            for event in events[-limit:]:  # Get last 'limit' events
                validation_record = {
                    "block_number": event['blockNumber'],
                    "transaction_hash": event['transactionHash'].hex(),
                    "model_hash": event['args'].get('modelHash', ''),
                    "is_valid": event['args'].get('isValid', False),
                    "round_id": event['args'].get('roundId', 0),
                    "timestamp": self._get_block_timestamp(event['blockNumber']),
                }
                validation_history.append(validation_record)
            
            return validation_history
            
        except Exception as e:
            logger.error("Failed to get validation history", error=str(e))
            return []
    
    def _get_block_timestamp(self, block_number: int) -> Optional[datetime]:
        """Get timestamp for a block."""
        try:
            block = self.w3.eth.get_block(block_number)
            return datetime.fromtimestamp(block['timestamp'])
        except:
            return None
    
    def get_validator_stats(self) -> Dict[str, Any]:
        """Get comprehensive validator statistics."""
        stats = {
            "network": self._get_network_name(),
            "contract_address": self.contract_address,
            "has_account": self.account is not None,
            "cached_validations": len(self.validation_cache),
            "connection_status": self.w3.is_connected(),
        }
        
        if self.account:
            try:
                balance = self.w3.eth.get_balance(self.account.address)
                stats["account_balance_eth"] = self.w3.from_wei(balance, 'ether')
                stats["account_address"] = self.account.address
            except:
                pass
        
        # Get recent validation statistics
        try:
            validation_history = self.get_validation_history(limit=50)
            if validation_history:
                stats.update({
                    "recent_validations": len(validation_history),
                    "recent_success_rate": sum(1 for v in validation_history if v["is_valid"]) / len(validation_history),
                    "latest_validation": validation_history[-1] if validation_history else None,
                })
        except:
            pass
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear validation cache."""
        self.validation_cache.clear()
        logger.info("Validation cache cleared")


# Smart Contract ABI (example)
MODEL_VALIDATOR_ABI = [
    {
        "inputs": [
            {"name": "modelHash", "type": "string"},
            {"name": "metadata", "type": "string"},
            {"name": "roundId", "type": "uint256"}
        ],
        "name": "validateModelUpdate",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "modelHash", "type": "string"}],
        "name": "isValidModel", 
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "modelHash", "type": "string"},
            {"indexed": False, "name": "isValid", "type": "bool"},
            {"indexed": False, "name": "roundId", "type": "uint256"},
            {"indexed": False, "name": "timestamp", "type": "uint256"}
        ],
        "name": "ModelValidated",
        "type": "event"
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view", 
        "type": "function"
    }
]


def create_model_validator(
    web3_provider_url: Optional[str] = None,
    contract_address: Optional[str] = None,
    private_key: Optional[str] = None,
    metrics_collector: Optional[MetricsCollector] = None,
) -> ModelValidator:
    """Factory function to create ModelValidator with environment configuration."""
    
    # Use environment variables as fallback
    web3_provider_url = web3_provider_url or os.getenv("ETHEREUM_NODE_URL", "http://localhost:8545")
    contract_address = contract_address or os.getenv("MODEL_VALIDATOR_CONTRACT_ADDRESS")
    private_key = private_key or os.getenv("ETHEREUM_PRIVATE_KEY")
    
    if not contract_address:
        # Deploy a mock contract for development
        logger.warning("No contract address provided, using mock validator")
        return MockModelValidator(metrics_collector)
    
    return ModelValidator(
        web3_provider_url=web3_provider_url,
        contract_address=contract_address,
        contract_abi=MODEL_VALIDATOR_ABI,
        private_key=private_key,
        metrics_collector=metrics_collector,
    )


class MockModelValidator:
    """Mock validator for development and testing."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector
        self.validation_count = 0
        
        logger.info("MockModelValidator initialized")
    
    def validate_model_update(self, model_hash: str, metadata: str, round_id: int) -> ValidationResult:
        """Mock validation that always passes."""
        import time
        time.sleep(0.1)  # Simulate blockchain delay
        
        self.validation_count += 1
        
        # Record mock metrics
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                MetricType.TRANSACTION_TIME,
                0.1,
                "mock_validator",
                metadata={"round_id": round_id}
            )
        
        return ValidationResult(
            is_valid=True,
            model_hash=model_hash,
            transaction_hash=f"mock_tx_{self.validation_count:06d}",
            gas_used=21000,
            validation_time=0.1,
            metadata={"round_id": round_id, "mock": True}
        )
    
    def get_validator_stats(self) -> Dict[str, Any]:
        """Get mock validator stats."""
        return {
            "network": "mock",
            "contract_address": "0x0000000000000000000000000000000000000000",
            "validation_count": self.validation_count,
            "mock": True,
        }