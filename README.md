# ACChain: Decentralized AI Compute Allocation Platform


## Overview
AICChain is a blockchain-based platform designed to efficiently match AI workloads with distributed compute nodes, leveraging reinforcement learning (RL), a dual-token system, and mechanism design for trust and fairness. The platform optimizes resource allocation, ensures truthful reporting, and incentivizes participation through a transparent and decentralized governance model.

## Features

### 1. **AI Workload Orchestration**
- **Reinforcement Learning (RL)**: Uses a PPO-based RL model to dynamically match AI workloads to compute nodes based on latency, bandwidth, energy efficiency, and GPU availability.
- **Workload Representation**: Defines AI workloads with attributes like required GPUs, data size, compute intensity, and priority.
- **Compute Node Management**: Tracks node performance metrics (latency, bandwidth, energy, GPU count) to ensure optimal task allocation.

### 2. **Dual-Token System**
- **AICC (Payment Token)**: Facilitates payments for computational services between clients and nodes. A portion of each transaction contributes to a Community Fund to support equitable access.
- **AICG (Governance Token)**: Enables stakeholders to vote on network policies, ensuring decentralized governance.
- **Community Fund**: Accumulates AICC to subsidize access for under-resourced users, promoting inclusivity.

### 3. **Truthful Reporting Mechanism**
- **Performance Reports**: Nodes submit performance metrics, which are hashed and scored for trustworthiness.
- **Auditing**: A scoring system penalizes discrepancies between reported and actual metrics, ensuring nodes report truthfully.
- **Node Reliability**: Inactive or untrustworthy nodes are flagged and excluded from task allocation.

### 4. **On-Chain RL Policy Management**
- **Off-Chain Training**: RL policies are trained off-chain to optimize workload-node matching.
- **On-Chain Commitment**: Trained policy parameters are hashed and stored on the blockchain for transparency and immutability.
- **Deterministic Matching**: Ensures consistent and reproducible workload assignments based on trained RL models.

### 5. **Decentralized Governance**
- **Policy Voting**: Stakeholders use AICG tokens to vote on network policies, such as incentivizing renewable energy use.
- **Transparent Execution**: All transactions, votes, and policy commitments are recorded on the blockchain.

## Architecture
AICChain integrates several components to deliver a robust and scalable solution:

- **ComputeNode & AIWorkload**: Data classes representing nodes and tasks with relevant attributes.
- **AICChainEnv**: A Gymnasium-based RL environment for training and evaluating workload-node matching policies.
- **DualTokenSystem**: Manages AICC and AICG tokens, including transfers, payments, and voting.
- **TruthfulReporting**: Ensures node reliability through report verification and auditing.
- **RLPolicyManager**: Handles RL model training, policy commitment, and workload matching.
- **AICChain**: The main orchestrator, integrating all components for end-to-end operation.

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies: `numpy`, `torch`, `stable-baselines3`, `gymnasium`
- Install dependencies:
  ```bash
  pip install numpy torch stable-baselines3 gymnasium
  ```

### Running the Demo
The included `main()` function demonstrates AICChain's functionality:
1. Registers compute nodes with varying performance metrics.
2. Submits an AI workload from a client.
3. Trains an RL model to match workloads to nodes.
4. Orchestrates the workload, processes payments, and contributes to the Community Fund.
5. Simulates governance by voting on a network policy.

To run the demo:
```bash
python aicchain.py
```

### Example Output
```
=== Starting AICChain Demo ===
AICChain is a system that connects AI tasks to the best computers using a blockchain for fairness and efficiency.

Step 1: Adding computers (nodes) to the network...
Registering compute node node1 with 4 GPUs, ready to process AI tasks.
...

Step 4: Assigning the AI task to the best computer...
AI model selected node node1 for workload workload1 based on speed, bandwidth, and energy efficiency.
Task workload1 successfully assigned to node node1!

Step 6: Voting on a new network rule (e.g., rewarding green energy use)...
The vote passed, and the network rule is updated!

=== AICChain Demo Complete ===
```

## Use Cases
- **AI Compute Marketplaces**: Enables clients to access distributed GPU resources for AI training and inference.
- **Decentralized Cloud Computing**: Provides a trustless alternative to centralized cloud providers.
- **Sustainable Computing**: Incentivizes energy-efficient nodes through governance policies.
- **Community-Driven AI**: Uses the Community Fund to democratize access to AI compute resources.

## Future Enhancements
- **Scalability**: Support for larger networks with thousands of nodes and workloads.
- **Advanced RL Models**: Integration of more sophisticated RL algorithms for improved matching.
- **Interoperability**: Compatibility with existing blockchain ecosystems (e.g., Ethereum, Polkadot).
- **Real-Time Auditing**: Continuous monitoring and auditing of node performance on-chain.

## Contributing
Contributions are welcome! Please submit pull requests or open issues on our [GitHub repository](#). Focus areas include:
- Optimizing RL training for faster convergence.
- Enhancing the truthful reporting mechanism with cryptographic proofs.
- Adding support for cross-chain token transfers.

