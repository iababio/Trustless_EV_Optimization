const hre = require("hardhat");

async function main() {
  console.log("🚀 Deploying ModelValidator contract...");

  // Deploy the contract
  const ModelValidator = await hre.ethers.getContractFactory("ModelValidator");
  const modelValidator = await ModelValidator.deploy();

  await modelValidator.waitForDeployment();

  const contractAddress = await modelValidator.getAddress();
  
  console.log("✅ ModelValidator deployed to:", contractAddress);
  console.log("📜 Transaction hash:", modelValidator.deploymentTransaction().hash);

  // Verify deployment by calling a read-only function
  try {
    const stats = await modelValidator.getContractStats();
    console.log("📊 Contract initialized with stats:", {
      totalValidations: stats[0].toString(),
      totalValidValidations: stats[1].toString(), 
      currentRound: stats[2].toString(),
      qualityThreshold: stats[3].toString()
    });
  } catch (error) {
    console.log("⚠️  Could not verify contract stats:", error.message);
  }

  console.log("\n🔧 Configuration for .env file:");
  console.log(`ETHEREUM_NODE_URL=http://127.0.0.1:8545`);
  console.log(`CONTRACT_ADDRESS=${contractAddress}`);
  console.log(`ENABLE_BLOCKCHAIN=true`);

  console.log("\n✅ Deployment complete! The smart contract is ready for use.");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });