// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title ModelValidator
 * @dev Smart contract for trustless validation of federated learning model updates
 * @notice This contract provides decentralized validation for ML models in EV charging optimization
 */
contract ModelValidator is AccessControl, ReentrancyGuard, Pausable {
    
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    
    // Model validation structure
    struct ValidationResult {
        bytes32 modelHash;
        uint256 roundId;
        bool isValid;
        uint256 timestamp;
        string metadata;
        address validator;
        uint256 gasUsed;
    }
    
    // Client reputation structure
    struct ClientReputation {
        uint256 totalContributions;
        uint256 validContributions;
        uint256 reputation; // Scaled by 1000 (e.g., 1000 = 1.0)
        uint256 lastUpdate;
    }
    
    // Events
    event ModelValidated(
        bytes32 indexed modelHash,
        uint256 indexed roundId,
        bool isValid,
        address indexed validator,
        uint256 timestamp,
        uint256 gasUsed
    );
    
    event ClientReputationUpdated(
        address indexed client,
        uint256 newReputation,
        uint256 totalContributions
    );
    
    event QualityThresholdUpdated(uint256 oldThreshold, uint256 newThreshold);
    
    event EmergencyStop(address indexed caller, string reason);
    
    // State variables
    mapping(bytes32 => ValidationResult) public validations;
    mapping(address => ClientReputation) public clientReputations;
    mapping(uint256 => bytes32[]) public roundModels; // roundId => model hashes
    
    uint256 public qualityThreshold = 100; // Scaled by 1000 (0.1 = 100)
    uint256 public maxValidationsPerRound = 100;
    uint256 public validationCost = 0.001 ether;
    
    // Rate limiting
    mapping(address => uint256) public lastValidationTime;
    uint256 public minValidationInterval = 60; // seconds
    
    // Statistics
    uint256 public totalValidations;
    uint256 public totalValidValidations;
    uint256 public currentRound;
    
    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(VALIDATOR_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
    }
    
    /**
     * @dev Validate a model update from federated learning
     * @param _modelHashStr String representation of model hash
     * @param _metadata JSON metadata about the model
     * @param _roundId Current federated learning round
     * @return isValid Whether the model passed validation
     */
    function validateModelUpdate(
        string memory _modelHashStr,
        string memory _metadata,
        uint256 _roundId
    ) external payable nonReentrant whenNotPaused returns (bool isValid) {
        require(msg.value >= validationCost, "Insufficient validation fee");
        require(bytes(_modelHashStr).length > 0, "Invalid model hash");
        require(bytes(_metadata).length > 0, "Invalid metadata");
        require(_roundId > 0, "Invalid round ID");
        
        // Rate limiting
        require(
            block.timestamp >= lastValidationTime[msg.sender] + minValidationInterval,
            "Validation too frequent"
        );
        lastValidationTime[msg.sender] = block.timestamp;
        
        // Check round limits
        require(
            roundModels[_roundId].length < maxValidationsPerRound,
            "Max validations per round exceeded"
        );
        
        bytes32 modelHash = keccak256(bytes(_modelHashStr));
        require(validations[modelHash].timestamp == 0, "Model already validated");
        
        uint256 gasStart = gasleft();
        
        // Perform validation logic
        isValid = _performValidation(_modelHashStr, _metadata, _roundId);
        
        uint256 gasUsed = gasStart - gasleft();
        
        // Store validation result
        validations[modelHash] = ValidationResult({
            modelHash: modelHash,
            roundId: _roundId,
            isValid: isValid,
            timestamp: block.timestamp,
            metadata: _metadata,
            validator: msg.sender,
            gasUsed: gasUsed
        });
        
        // Update round tracking
        roundModels[_roundId].push(modelHash);
        if (_roundId > currentRound) {
            currentRound = _roundId;
        }
        
        // Update statistics
        totalValidations++;
        if (isValid) {
            totalValidValidations++;
        }
        
        // Update client reputation
        _updateClientReputation(msg.sender, isValid);
        
        // Emit event
        emit ModelValidated(
            modelHash,
            _roundId,
            isValid,
            msg.sender,
            block.timestamp,
            gasUsed
        );
        
        return isValid;
    }
    
    /**
     * @dev Internal validation logic
     * @param _modelHashStr Model hash string
     * @param _metadata Model metadata JSON
     * @param _roundId Round ID
     * @return isValid Validation result
     */
    function _performValidation(
        string memory _modelHashStr,
        string memory _metadata,
        uint256 _roundId
    ) internal view returns (bool isValid) {
        // Basic hash validation
        if (bytes(_modelHashStr).length != 64) {
            return false; // SHA256 should be 64 hex characters
        }
        
        // Metadata validation (simplified JSON parsing)
        if (!_isValidJSON(_metadata)) {
            return false;
        }
        
        // Quality threshold check (simplified)
        // In a real implementation, this would parse JSON and check quality metrics
        if (bytes(_metadata).length < 100) {
            return false; // Metadata too short
        }
        
        // Round consistency check
        if (_roundId < currentRound && currentRound > 0) {
            return false; // Old round
        }
        
        // Additional validation logic can be added here
        // For example: checking against previous models, reputation scores, etc.
        
        return true;
    }
    
    /**
     * @dev Simple JSON validation (checks for basic structure)
     * @param _json JSON string to validate
     * @return isValid Whether the string appears to be valid JSON
     */
    function _isValidJSON(string memory _json) internal pure returns (bool isValid) {
        bytes memory jsonBytes = bytes(_json);
        if (jsonBytes.length < 2) return false;
        
        // Check for basic JSON structure
        return (
            jsonBytes[0] == '{' && 
            jsonBytes[jsonBytes.length - 1] == '}'
        );
    }
    
    /**
     * @dev Update client reputation based on validation result
     * @param _client Client address
     * @param _isValid Whether the validation was successful
     */
    function _updateClientReputation(address _client, bool _isValid) internal {
        ClientReputation storage rep = clientReputations[_client];
        
        rep.totalContributions++;
        if (_isValid) {
            rep.validContributions++;
        }
        
        // Calculate new reputation (weighted average with momentum)
        uint256 successRate = (rep.validContributions * 1000) / rep.totalContributions;
        if (rep.reputation == 0) {
            rep.reputation = successRate;
        } else {
            // Exponential moving average with alpha = 0.1
            rep.reputation = (rep.reputation * 900 + successRate * 100) / 1000;
        }
        
        rep.lastUpdate = block.timestamp;
        
        emit ClientReputationUpdated(_client, rep.reputation, rep.totalContributions);
    }
    
    /**
     * @dev Check if a model is valid (read-only)
     * @param _modelHashStr Model hash string
     * @return isValid Validation status
     */
    function isValidModel(string memory _modelHashStr) external view returns (bool isValid) {
        bytes32 modelHash = keccak256(bytes(_modelHashStr));
        return validations[modelHash].isValid;
    }
    
    /**
     * @dev Get validation result for a model
     * @param _modelHashStr Model hash string
     * @return result Complete validation result
     */
    function getValidationResult(string memory _modelHashStr) 
        external 
        view 
        returns (ValidationResult memory result) 
    {
        bytes32 modelHash = keccak256(bytes(_modelHashStr));
        return validations[modelHash];
    }
    
    /**
     * @dev Get client reputation
     * @param _client Client address
     * @return reputation Client reputation data
     */
    function getClientReputation(address _client) 
        external 
        view 
        returns (ClientReputation memory reputation) 
    {
        return clientReputations[_client];
    }
    
    /**
     * @dev Get models validated in a specific round
     * @param _roundId Round ID
     * @return modelHashes Array of model hashes
     */
    function getRoundModels(uint256 _roundId) 
        external 
        view 
        returns (bytes32[] memory modelHashes) 
    {
        return roundModels[_roundId];
    }
    
    /**
     * @dev Get contract statistics
     * @return stats Array containing [totalValidations, totalValid, currentRound, qualityThreshold]
     */
    function getContractStats() 
        external 
        view 
        returns (uint256[4] memory stats) 
    {
        stats[0] = totalValidations;
        stats[1] = totalValidValidations;
        stats[2] = currentRound;
        stats[3] = qualityThreshold;
        return stats;
    }
    
    /**
     * @dev Set quality threshold (admin only)
     * @param _newThreshold New quality threshold (scaled by 1000)
     */
    function setQualityThreshold(uint256 _newThreshold) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(_newThreshold <= 1000, "Threshold too high");
        uint256 oldThreshold = qualityThreshold;
        qualityThreshold = _newThreshold;
        emit QualityThresholdUpdated(oldThreshold, _newThreshold);
    }
    
    /**
     * @dev Set validation cost (admin only)
     * @param _newCost New validation cost in wei
     */
    function setValidationCost(uint256 _newCost) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(_newCost <= 0.01 ether, "Cost too high");
        validationCost = _newCost;
    }
    
    /**
     * @dev Set maximum validations per round (admin only)
     * @param _newMax New maximum validations per round
     */
    function setMaxValidationsPerRound(uint256 _newMax) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        require(_newMax > 0 && _newMax <= 1000, "Invalid max value");
        maxValidationsPerRound = _newMax;
    }
    
    /**
     * @dev Emergency stop function
     * @param _reason Reason for emergency stop
     */
    function emergencyStop(string memory _reason) 
        external 
        onlyRole(PAUSER_ROLE) 
    {
        _pause();
        emit EmergencyStop(msg.sender, _reason);
    }
    
    /**
     * @dev Resume operations after emergency stop
     */
    function resumeOperations() 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        _unpause();
    }
    
    /**
     * @dev Withdraw accumulated fees (admin only)
     * @param _to Address to withdraw to
     * @param _amount Amount to withdraw
     */
    function withdrawFees(address payable _to, uint256 _amount) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
        nonReentrant 
    {
        require(_to != address(0), "Invalid address");
        require(_amount <= address(this).balance, "Insufficient balance");
        
        _to.transfer(_amount);
    }
    
    /**
     * @dev Get contract balance
     * @return balance Contract balance in wei
     */
    function getContractBalance() external view returns (uint256 balance) {
        return address(this).balance;
    }
    
    /**
     * @dev Add validator role to an address (admin only)
     * @param _validator Address to grant validator role
     */
    function addValidator(address _validator) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        grantRole(VALIDATOR_ROLE, _validator);
    }
    
    /**
     * @dev Remove validator role from an address (admin only)
     * @param _validator Address to revoke validator role
     */
    function removeValidator(address _validator) 
        external 
        onlyRole(DEFAULT_ADMIN_ROLE) 
    {
        revokeRole(VALIDATOR_ROLE, _validator);
    }
    
    /**
     * @dev Check if address has validator role
     * @param _validator Address to check
     * @return hasRole Whether address has validator role
     */
    function isValidator(address _validator) external view returns (bool) {
        return hasRole(VALIDATOR_ROLE, _validator);
    }
    
    // Fallback function to accept Ether
    receive() external payable {}
}