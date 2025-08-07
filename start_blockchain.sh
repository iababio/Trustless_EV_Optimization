#!/bin/bash

echo "🚀 Starting Ethereum Smart Contracts Server for EV Charging Optimization"
echo "============================================================================"

# Change to contracts directory
cd "$(dirname "$0")/contracts" || exit 1

# Check if Hardhat is available
if ! command -v npx &> /dev/null; then
    echo "❌ Error: npm/npx not found. Please install Node.js first."
    exit 1
fi

# Check if dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Compile contracts
echo "🔧 Compiling smart contracts..."
npx hardhat compile

if [ $? -ne 0 ]; then
    echo "❌ Contract compilation failed"
    exit 1
fi

echo ""
echo "✅ Smart contracts compiled successfully!"
echo ""
echo "🌐 Starting local Ethereum network..."
echo "   - Network: Hardhat Local"
echo "   - Chain ID: 1337"
echo "   - RPC URL: http://127.0.0.1:8545"
echo ""
echo "💡 To deploy contracts, run this in another terminal:"
echo "   cd contracts && npx hardhat run scripts/deploy.js --network localhost"
echo ""
echo "🛑 Press Ctrl+C to stop the blockchain server"
echo ""

# Start Hardhat node (this will run until stopped)
npx hardhat node