# 🔗 Blockchain Smart Contracts Setup Guide

## Quick Answer: Starting the Ethereum Smart Contracts Server

The ReduceLROnPlateau error has been **fixed**! ✅

For blockchain functionality, you have **3 options**:

### Option 1: Local Development (Recommended)
```bash
# Install Node.js and Hardhat
npm install
npx hardhat node  # Starts local Ethereum node

# Deploy contracts (in another terminal)
npx hardhat deploy --network localhost
```

### Option 2: Use Mock Validator (Already Working)
```bash
python demo.py  # Uses mock blockchain validator automatically
```

### Option 3: Connect to Testnet
```bash
# Configure .env with testnet details
npx hardhat deploy --network goerli
```

---

## 🚀 Complete Setup Instructions

### Prerequisites

1. **Node.js 16+**
   ```bash
   # Check version
   node --version  # Should be 16.0.0 or higher
   ```

2. **Python Environment** (already working)
   ```bash
   python demo.py  # Verify this works first
   ```

### Step 1: Install Blockchain Dependencies

```bash
# Navigate to contracts directory
cd contracts

# Initialize npm if needed
npm init -y

# Install required packages
npm install --save-dev hardhat
npm install --save-dev @nomicfoundation/hardhat-toolbox
npm install @openzeppelin/contracts
```

### Step 2: Initialize Hardhat Project

```bash
# Initialize Hardhat (in contracts/ directory)
npx hardhat init

# Choose "Create a JavaScript project"
# Accept all defaults
```

### Step 3: Configure Hardhat

Create `contracts/hardhat.config.js`:
```javascript
require("@nomicfoundation/hardhat-toolbox");

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    hardhat: {
      chainId: 1337,
    },
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 1337,
    },
  },
};
```

### Step 4: Create Deployment Script

Create `contracts/scripts/deploy.js`:
```javascript
const hre = require("hardhat");

async function main() {
  console.log("Deploying ModelValidator contract...");

  // Deploy the contract
  const ModelValidator = await hre.ethers.getContractFactory("ModelValidator");
  const modelValidator = await ModelValidator.deploy();

  await modelValidator.deployed();

  console.log("ModelValidator deployed to:", modelValidator.address);
  console.log("Transaction hash:", modelValidator.deployTransaction.hash);

  // Verify deployment
  const stats = await modelValidator.getContractStats();
  console.log("Contract initialized with stats:", stats);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

### Step 5: Start Local Blockchain

**Terminal 1** - Start Hardhat Node:
```bash
cd contracts
npx hardhat node
```

You should see:
```
Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/

Accounts
========
Account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000 ETH)
Account #1: 0x70997970C51812dc3A010C7d01b50e0d17dc79C8 (10000 ETH)
...
```

**Terminal 2** - Deploy Contract:
```bash
cd contracts
npx hardhat run scripts/deploy.js --network localhost
```

You should see:
```
Deploying ModelValidator contract...
ModelValidator deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3
Transaction hash: 0x...
Contract initialized with stats: [0, 0, 0, 100]
```

### Step 6: Update Python Configuration

Create or update `.env` file:
```bash
# Ethereum Configuration
ETHEREUM_NODE_URL=http://127.0.0.1:8545
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3  # Use address from deployment
PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80  # Account #0 from Hardhat

# Optional: Enable blockchain features
ENABLE_BLOCKCHAIN=true
```

### Step 7: Test Blockchain Integration

```bash
# Run the complete demo with blockchain enabled
python examples/complete_demo.py
```

Look for blockchain-related output:
```
✅ Blockchain validator initialized
🔗 Smart contract deployed at: 0x5FbDB2315678afecb367f032d93F642f64180aa3
📊 Blockchain validation results: {"enabled": true, "validation_results": [...]}
```

---

## 🧪 Advanced Configuration

### Using Ethereum Testnets

1. **Get Test ETH**:
   - Goerli: https://goerlifaucet.com/
   - Sepolia: https://sepoliafaucet.com/

2. **Update hardhat.config.js**:
   ```javascript
   networks: {
     goerli: {
       url: "https://goerli.infura.io/v3/YOUR_INFURA_KEY",
       accounts: ["YOUR_PRIVATE_KEY"]
     }
   }
   ```

3. **Deploy to testnet**:
   ```bash
   npx hardhat run scripts/deploy.js --network goerli
   ```

### Production Deployment

For mainnet deployment:
```javascript
networks: {
  mainnet: {
    url: "https://mainnet.infura.io/v3/YOUR_INFURA_KEY",
    accounts: ["YOUR_PRIVATE_KEY"],
    gasPrice: 20000000000, // 20 gwei
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `Error: could not detect network`
**Solution**: Make sure Hardhat node is running in another terminal

**Issue**: `Error: insufficient funds for gas`
**Solution**: Use one of the pre-funded accounts from Hardhat node

**Issue**: `Error: contract not deployed`
**Solution**: Run the deployment script first: `npx hardhat run scripts/deploy.js --network localhost`

### Verification Commands

```bash
# Check if node is running
curl -X POST http://127.0.0.1:8545 \
  -H "Content-Type: application/json" \
  -d '{"method":"eth_blockNumber","params":[],"id":1,"jsonrpc":"2.0"}'

# Check contract deployment
npx hardhat console --network localhost
> const ModelValidator = await ethers.getContractFactory("ModelValidator");
> const contract = await ModelValidator.attach("YOUR_CONTRACT_ADDRESS");
> await contract.getContractStats();
```

---

## 📊 What You Get

With blockchain enabled, the system provides:

- **Trustless Validation**: ML models validated on-chain
- **Reputation System**: Client reputation tracking
- **Transparency**: All validations recorded immutably
- **Gas Optimization**: Efficient smart contract operations
- **Security**: Access control, rate limiting, emergency stops

### Sample Blockchain Metrics

```json
{
  "blockchain_validation": {
    "enabled": true,
    "total_validations": 15,
    "successful_validations": 12,
    "average_gas_used": 145000,
    "contract_balance": "0.015 ETH",
    "validator_reputation": 0.8
  }
}
```

---

## 🎯 Quick Commands Summary

```bash
# Start blockchain (run once)
cd contracts && npx hardhat node

# Deploy contracts (run once)
cd contracts && npx hardhat run scripts/deploy.js --network localhost

# Run demo with blockchain
python examples/complete_demo.py

# Check blockchain status
curl http://127.0.0.1:8545 -d '{"method":"eth_blockNumber","params":[],"id":1,"jsonrpc":"2.0"}'
```

The blockchain server will keep running until you stop it with Ctrl+C. Your ML system will automatically connect to it when `ENABLE_BLOCKCHAIN=true` in your environment.