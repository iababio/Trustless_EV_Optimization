#!/bin/bash

echo "📜 Deploying Smart Contracts to Local Network"
echo "=============================================="

cd "$(dirname "$0")/contracts" || exit 1

# Check if local network is running
if ! curl -s http://127.0.0.1:8545 > /dev/null; then
    echo "❌ Error: Local Ethereum network not running!"
    echo "💡 Start it first with: ./start_blockchain.sh"
    exit 1
fi

echo "🚀 Deploying ModelValidator contract..."
npx hardhat run scripts/deploy.js --network localhost

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo "💡 Your blockchain server is now ready for the EV optimization system!"
    echo "🔧 Make sure to update your .env file with the contract address shown above"
else
    echo "❌ Deployment failed"
    exit 1
fi