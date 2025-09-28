// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ModelValidatorResearch
 * @dev Research-focused smart contract for validating federated learning model updates
 * in EV charging optimization research. Designed for proof-of-concept rather than production.
 */
contract ModelValidatorResearch {
    
    // Events for research monitoring
    event ModelUpdateValidated(
        address indexed client,
        bytes32 indexed modelHash,
        uint256 accuracy,
        uint256 loss,
        uint256 timestamp
    );
    
    event ValidationRejected(
        address indexed client,
        bytes32 indexed modelHash,
        string reason,
        uint256 timestamp
    );
    
    event ReputationUpdated(
        address indexed client,
        uint256 oldReputation,
        uint256 newReputation
    );
    
    // Structs for validation data
    struct ValidationMetrics {
        uint256 accuracy;      // Scaled by 10000 (95.5% = 9550)
        uint256 loss;          // Scaled loss value
        uint256 clientCount;   // Number of clients in round
        bytes32 modelHash;     // Hash of model weights
        uint256 timestamp;     // Block timestamp
    }
    
    struct ClientReputation {
        uint256 totalValidations;
        uint256 successfulValidations;
        uint256 reputation;    // Score out of 10000
        uint256 lastActivity;
        bool isActive;
    }
    
    struct ValidationRule {
        uint256 minAccuracy;    // Minimum accuracy threshold
        uint256 maxLoss;        // Maximum loss threshold
        uint256 minClients;     // Minimum participating clients
        bool enabled;
    }
    
    // State variables
    mapping(address => ClientReputation) public clientReputations;
    mapping(bytes32 => ValidationMetrics) public validatedModels;
    mapping(address => bool) public authorizedClients;
    
    ValidationRule public validationRules;
    address public owner;
    uint256 public totalValidations;
    uint256 public consensusThreshold; // Percentage for consensus (out of 100)
    
    // Arrays for research analytics
    address[] public clientAddresses;
    bytes32[] public validatedModelHashes;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    modifier onlyAuthorizedClient() {
        require(authorizedClients[msg.sender], "Client not authorized");
        _;
    }
    
    constructor() {
        owner = msg.sender;
        
        // Initialize validation rules for research
        validationRules = ValidationRule({
            minAccuracy: 5000,    // 50% minimum accuracy
            maxLoss: 1000000,     // Maximum loss threshold
            minClients: 3,        // Minimum 3 clients for validation
            enabled: true
        });
        
        consensusThreshold = 60; // 60% consensus required
    }
    
    /**
     * @dev Register a new client for federated learning participation
     * @param clientAddress Address of the client to register
     */
    function registerClient(address clientAddress) external onlyOwner {
        require(clientAddress != address(0), "Invalid client address");
        require(!authorizedClients[clientAddress], "Client already registered");
        
        authorizedClients[clientAddress] = true;
        clientReputations[clientAddress] = ClientReputation({
            totalValidations: 0,
            successfulValidations: 0,
            reputation: 5000, // Start with 50% reputation
            lastActivity: block.timestamp,
            isActive: true
        });
        
        clientAddresses.push(clientAddress);
    }
    
    /**
     * @dev Submit model update for validation
     * @param metrics Validation metrics for the model update
     * @return success Whether validation passed
     */
    function validateModelUpdate(ValidationMetrics calldata metrics)
        external
        onlyAuthorizedClient
        returns (bool success)
    {
        require(validationRules.enabled, "Validation currently disabled");
        require(metrics.modelHash != bytes32(0), "Invalid model hash");
        
        // Update client activity
        clientReputations[msg.sender].lastActivity = block.timestamp;
        clientReputations[msg.sender].totalValidations++;
        
        // Validate against rules
        if (!_meetsValidationCriteria(metrics)) {
            emit ValidationRejected(
                msg.sender,
                metrics.modelHash,
                "Failed validation criteria",
                block.timestamp
            );
            return false;
        }
        
        // Check for potential Byzantine behavior
        if (_detectByzantineBehavior(msg.sender, metrics)) {
            emit ValidationRejected(
                msg.sender,
                metrics.modelHash,
                "Byzantine behavior detected",
                block.timestamp
            );
            _penalizeClient(msg.sender);
            return false;
        }
        
        // Store validated model
        validatedModels[metrics.modelHash] = ValidationMetrics({
            accuracy: metrics.accuracy,
            loss: metrics.loss,
            clientCount: metrics.clientCount,
            modelHash: metrics.modelHash,
            timestamp: block.timestamp
        });
        
        validatedModelHashes.push(metrics.modelHash);
        totalValidations++;
        
        // Update reputation
        _updateClientReputation(msg.sender, true);
        
        emit ModelUpdateValidated(
            msg.sender,
            metrics.modelHash,
            metrics.accuracy,
            metrics.loss,
            block.timestamp
        );
        
        return true;
    }
    
    /**
     * @dev Check if validation criteria are met
     */
    function _meetsValidationCriteria(ValidationMetrics calldata metrics)
        internal
        view
        returns (bool)
    {
        return (
            metrics.accuracy >= validationRules.minAccuracy &&
            metrics.loss <= validationRules.maxLoss &&
            metrics.clientCount >= validationRules.minClients
        );
    }
    
    /**
     * @dev Simple Byzantine behavior detection for research
     */
    function _detectByzantineBehavior(address client, ValidationMetrics calldata metrics)
        internal
        view
        returns (bool)
    {
        ClientReputation memory reputation = clientReputations[client];
        
        // Check for suspicious patterns
        if (reputation.totalValidations > 0) {
            uint256 successRate = (reputation.successfulValidations * 100) / reputation.totalValidations;
            
            // Flag if success rate suddenly drops significantly
            if (successRate < 20 && reputation.totalValidations > 5) {
                return true;
            }
        }
        
        // Check for extreme values (simplified detection)
        if (metrics.accuracy > 9900 || metrics.accuracy < 1000) {
            return true; // Suspiciously high or low accuracy
        }
        
        return false;
    }
    
    /**
     * @dev Update client reputation based on validation result
     */
    function _updateClientReputation(address client, bool successful) internal {
        ClientReputation storage reputation = clientReputations[client];
        uint256 oldReputation = reputation.reputation;
        
        if (successful) {
            reputation.successfulValidations++;
            // Increase reputation (max 10000)
            if (reputation.reputation < 9500) {
                reputation.reputation += 100;
            }
        } else {
            // Decrease reputation (min 0)
            if (reputation.reputation > 200) {
                reputation.reputation -= 200;
            }
        }
        
        // Deactivate clients with very low reputation
        if (reputation.reputation < 1000) {
            reputation.isActive = false;
        }
        
        emit ReputationUpdated(client, oldReputation, reputation.reputation);
    }
    
    /**
     * @dev Penalize client for Byzantine behavior
     */
    function _penalizeClient(address client) internal {
        ClientReputation storage reputation = clientReputations[client];
        
        // Significant reputation penalty
        if (reputation.reputation > 1000) {
            reputation.reputation -= 1000;
        } else {
            reputation.reputation = 0;
            reputation.isActive = false;
        }
    }
    
    /**
     * @dev Get client reputation information
     */
    function getClientReputation(address client)
        external
        view
        returns (
            uint256 totalValidations,
            uint256 successfulValidations,
            uint256 reputation,
            uint256 lastActivity,
            bool isActive
        )
    {
        ClientReputation memory rep = clientReputations[client];
        return (
            rep.totalValidations,
            rep.successfulValidations,
            rep.reputation,
            rep.lastActivity,
            rep.isActive
        );
    }
    
    /**
     * @dev Get validation statistics for research analysis
     */
    function getValidationStatistics()
        external
        view
        returns (
            uint256 totalClients,
            uint256 activeClients,
            uint256 totalValidationsCount,
            uint256 avgReputation
        )
    {
        uint256 active = 0;
        uint256 totalRep = 0;
        
        for (uint256 i = 0; i < clientAddresses.length; i++) {
            ClientReputation memory rep = clientReputations[clientAddresses[i]];
            if (rep.isActive) {
                active++;
            }
            totalRep += rep.reputation;
        }
        
        uint256 avgRep = clientAddresses.length > 0 ? totalRep / clientAddresses.length : 0;
        
        return (clientAddresses.length, active, totalValidations, avgRep);
    }
    
    /**
     * @dev Update validation rules (owner only)
     */
    function updateValidationRules(
        uint256 minAccuracy,
        uint256 maxLoss,
        uint256 minClients,
        bool enabled
    ) external onlyOwner {
        validationRules = ValidationRule({
            minAccuracy: minAccuracy,
            maxLoss: maxLoss,
            minClients: minClients,
            enabled: enabled
        });
    }
    
    /**
     * @dev Update consensus threshold
     */
    function updateConsensusThreshold(uint256 newThreshold) external onlyOwner {
        require(newThreshold <= 100, "Threshold must be <= 100");
        consensusThreshold = newThreshold;
    }
    
    /**
     * @dev Get recent validation history for research
     */
    function getRecentValidations(uint256 count)
        external
        view
        returns (bytes32[] memory hashes, uint256[] memory accuracies, uint256[] memory timestamps)
    {
        uint256 start = validatedModelHashes.length > count 
            ? validatedModelHashes.length - count 
            : 0;
        uint256 length = validatedModelHashes.length - start;
        
        hashes = new bytes32[](length);
        accuracies = new uint256[](length);
        timestamps = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            bytes32 hash = validatedModelHashes[start + i];
            ValidationMetrics memory metrics = validatedModels[hash];
            
            hashes[i] = hash;
            accuracies[i] = metrics.accuracy;
            timestamps[i] = metrics.timestamp;
        }
    }
    
    /**
     * @dev Emergency function to reset client reputation (research only)
     */
    function resetClientReputation(address client) external onlyOwner {
        require(authorizedClients[client], "Client not registered");
        
        clientReputations[client].reputation = 5000;
        clientReputations[client].isActive = true;
    }
    
    /**
     * @dev Get all registered clients for research analysis
     */
    function getAllClients() external view returns (address[] memory) {
        return clientAddresses;
    }
    
    /**
     * @dev Check if model hash has been validated
     */
    function isModelValidated(bytes32 modelHash) external view returns (bool) {
        return validatedModels[modelHash].timestamp != 0;
    }
    
    /**
     * @dev Simulate consensus validation (for research testing)
     */
    function simulateConsensusValidation(
        bytes32 modelHash,
        address[] calldata validators,
        bool[] calldata votes
    ) external onlyOwner returns (bool consensus) {
        require(validators.length == votes.length, "Mismatched arrays");
        require(validators.length >= validationRules.minClients, "Not enough validators");
        
        uint256 positiveVotes = 0;
        for (uint256 i = 0; i < votes.length; i++) {
            if (votes[i]) {
                positiveVotes++;
            }
        }
        
        uint256 consensusPercentage = (positiveVotes * 100) / validators.length;
        consensus = consensusPercentage >= consensusThreshold;
        
        if (consensus) {
            // Mark as validated through consensus
            validatedModels[modelHash] = ValidationMetrics({
                accuracy: 0, // Placeholder for consensus validation
                loss: 0,
                clientCount: validators.length,
                modelHash: modelHash,
                timestamp: block.timestamp
            });
            
            validatedModelHashes.push(modelHash);
            totalValidations++;
        }
        
        return consensus;
    }
}