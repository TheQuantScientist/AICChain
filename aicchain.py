import numpy as np
import hashlib
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from stable_baselines3 import PPO
from gymnasium import Env, spaces
import torch

# --- Compute Node Representation ---
@dataclass
class ComputeNode:
    node_id: str
    latency: float  # ms
    bandwidth: float  # Mbps
    energy: float  # kWh
    gpu_count: int
    is_active: bool = True

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "latency": self.latency,
            "bandwidth": self.bandwidth,
            "energy": self.energy,
            "gpu_count": self.gpu_count,
            "is_active": self.is_active
        }

# --- AI Workload Representation ---
@dataclass
class AIWorkload:
    workload_id: str
    required_gpus: int
    data_size: float  # MB
    compute_intensity: float  # FLOPS
    priority: float  # 0 to 1

    def to_dict(self) -> Dict:
        return {
            "workload_id": self.workload_id,
            "required_gpus": self.required_gpus,
            "data_size": self.data_size,
            "compute_intensity": self.compute_intensity,
            "priority": self.priority
        }

# --- RL Environment for Workload-Node Matching ---
class AICChainEnv(Env):
    def __init__(self, nodes: List[ComputeNode], workloads: List[AIWorkload]):
        super().__init__()
        self.nodes = nodes
        self.workloads = workloads if workloads else []
        self.current_workload_idx = 0
        self.action_space = spaces.Discrete(len(nodes))
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_workload_idx = 0
        if not self.workloads:
            return np.zeros(5, dtype=np.float32), {}
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        if not self.workloads or self.current_workload_idx >= len(self.workloads):
            return np.array([0, 0, 0, 0, 0], dtype=np.float32)
        workload = self.workloads[self.current_workload_idx]
        node = self.nodes[0]
        return np.array([
            node.latency,
            node.bandwidth,
            node.energy,
            node.gpu_count,
            workload.priority
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if not self.workloads:
            return np.zeros(5, dtype=np.float32), 0.0, True, False, {}
        
        workload = self.workloads[min(self.current_workload_idx, len(self.workloads) - 1)]
        node = self.nodes[action]

        reward = (
            -node.latency / 100
            + node.bandwidth / 100
            - node.energy / 10
            + (node.gpu_count >= workload.required_gpus) * 10
            + workload.priority * 5
        )

        self.current_workload_idx += 1
        done = self.current_workload_idx >= len(self.workloads)
        return self._get_observation(), reward, done, False, {}

# --- Dual-Token System ---
@dataclass
class Token:
    name: str
    balance: Dict[str, float]

    def transfer(self, sender: str, receiver: str, amount: float) -> bool:
        if self.balance.get(sender, 0) >= amount:
            self.balance[sender] = self.balance.get(sender, 0) - amount
            self.balance[receiver] = self.balance.get(receiver, 0) + amount
            return True
        return False

class DualTokenSystem:
    def __init__(self):
        self.aicc = Token("AICC", {})
        self.aicg = Token("AICG", {})
        self.community_fund = 0.0

    def initialize_account(self, account: str, aicc_amount: float, aicg_amount: float):
        self.aicc.balance[account] = aicc_amount
        self.aicg.balance[account] = aicg_amount
        print(f"Created account for {account} with {aicc_amount} AICC (payment tokens) and {aicg_amount} AICG (voting tokens).")

    def pay_for_execution(self, client: str, node: str, amount: float) -> bool:
        print(f"Processing payment: Client {client} pays {amount} AICC to node {node} for computation.")
        fund_contribution = amount ** 2 / 1000
        print(f"Contributing {fund_contribution} AICC to the Community Fund to support fair access.")
        self.community_fund += fund_contribution
        if self.aicc.transfer(client, node, amount + fund_contribution):
            print(f"Payment successful! {amount} AICC transferred to {node}, and {fund_contribution} AICC added to Community Fund.")
            return True
        print("Payment failed: Insufficient AICC balance.")
        return False

    def vote_on_policy(self, voter: str, policy_id: str, stake: float) -> bool:
        print(f"{voter} is voting on policy '{policy_id}' by staking {stake} AICG tokens.")
        if self.aicg.transfer(voter, "governance", stake):
            print(f"Vote successful! {voter} staked {stake} AICG to influence the policy.")
            return True
        print("Vote failed: Insufficient AICG balance.")
        return False

# --- Mechanism Design for Truthful Reporting ---
class TruthfulReporting:
    def __init__(self):
        self.reports: Dict[str, Dict] = {}
        self.scores: Dict[str, float] = {}

    def submit_report(self, node: ComputeNode, report: Dict):
        print(f"Node {node.node_id} submits its performance report (speed, bandwidth, energy use).")
        report_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
        self.reports[node.node_id] = {"report": report, "hash": report_hash}
        score = self._calculate_score(report, node)
        self.scores[node.node_id] = score
        print(f"Report verified with a trustworthiness score of {score:.2f}.")

    def _calculate_score(self, report: Dict, node: ComputeNode) -> float:
        reported = report.get("metrics", {})
        actual = node.to_dict()
        error = sum((reported.get(k, 0) - actual[k]) ** 2 for k in ["latency", "bandwidth", "energy"])
        return -error

    def audit_node(self, node_id: str) -> bool:
        score = self.scores.get(node_id, 0)
        is_trustworthy = score > -10
        print(f"Auditing node {node_id}: {'Trustworthy' if is_trustworthy else 'Untrustworthy'} (score: {score:.2f}).")
        return is_trustworthy

# --- On-Chain RL Policy Management ---
class RLPolicyManager:
    def __init__(self, model_path: str = None):
        self.env = None
        self.model = None
        if model_path:
            self.model = PPO.load(model_path)
        self.policy_hash = None

    def train_off_chain(self, env: AICChainEnv, total_timesteps: int = 10000):
        self.env = env
        if not env.workloads:
            raise ValueError("Cannot train RL policy with empty workloads")
        print("Training AI model to decide which compute node is best for each AI task...")
        self.model = PPO("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        print("AI model training complete! Generating a unique ID for the trained model.")
        policy_params = {
            k: v.cpu().numpy().tolist() for k, v in self.model.policy.state_dict().items()
        }
        policy_params_json = json.dumps(policy_params, sort_keys=True)
        self.policy_hash = hashlib.sha256(policy_params_json.encode()).hexdigest()
        print(f"Model ID (hash): {self.policy_hash}")

    def commit_policy(self) -> str:
        print("Storing the AI model's ID on the blockchain for transparency.")
        return self.policy_hash

    def match_workload(self, workload: AIWorkload) -> ComputeNode:
        if not self.model or not self.env:
            raise ValueError("Model not trained or environment not set")
        print(f"Using AI model to find the best compute node for workload {workload.workload_id}...")
        obs = self.env._get_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        matched_node = self.env.nodes[action]
        print(f"AI model selected node {matched_node.node_id} for workload {workload.workload_id} based on speed, bandwidth, and energy efficiency.")
        return matched_node

# --- AICChain Main Class ---
class AICChain:
    def __init__(self):
        self.nodes: List[ComputeNode] = []
        self.workloads: List[AIWorkload] = []
        self.token_system = DualTokenSystem()
        self.reporting = TruthfulReporting()
        self.rl_policy = RLPolicyManager()
        self.env = None

    def register_node(self, node: ComputeNode):
        print(f"Registering compute node {node.node_id} with {node.gpu_count} GPUs, ready to process AI tasks.")
        self.nodes.append(node)
        self.token_system.initialize_account(node.node_id, 1000.0, 100.0)

    def submit_workload(self, workload: AIWorkload, client: str):
        print(f"Client {client} submits AI task {workload.workload_id}, requiring {workload.required_gpus} GPUs.")
        self.workloads.append(workload)
        self.token_system.initialize_account(client, 2000.0, 50.0)

    def setup_rl_environment(self):
        if not self.workloads or not self.nodes:
            raise ValueError("Cannot setup RL environment without workloads and nodes")
        print("Setting up AI decision system to match tasks to compute nodes efficiently.")
        self.env = AICChainEnv(self.nodes, self.workloads)
        self.rl_policy.train_off_chain(self.env)

    def orchestrate_workload(self, workload: AIWorkload, client: str) -> ComputeNode:
        print(f"Starting process to assign AI task {workload.workload_id} to a compute node...")
        matched_node = self.rl_policy.match_workload(workload)
        print(f"Checking if node {matched_node.node_id} is active and trustworthy...")
        if not matched_node.is_active:
            print(f"Error: Node {matched_node.node_id} is not active.")
            raise ValueError(f"Node {matched_node.node_id} is inactive")
        if not self.reporting.audit_node(matched_node.node_id):
            print(f"Error: Node {matched_node.node_id} failed trustworthiness check.")
            matched_node.is_active = False
            raise ValueError(f"Node {matched_node.node_id} failed audit")
        payment_amount = workload.compute_intensity * 0.1
        print(f"Calculating payment: {payment_amount} AICC for task {workload.workload_id}.")
        if not self.token_system.pay_for_execution(client, matched_node.node_id, payment_amount):
            raise ValueError("Payment failed")
        print(f"Task {workload.workload_id} successfully assigned to node {matched_node.node_id}!")
        return matched_node

    def vote_on_policy(self, voter: str, policy_id: str, stake: float) -> bool:
        return self.token_system.vote_on_policy(voter, policy_id, stake)

# --- Demo Execution ---
def main():
    print("=== Starting AICChain Demo ===")
    print("AICChain is a system that connects AI tasks to the best computers using a blockchain for fairness and efficiency.\n")

    chain = AICChain()

    print("Step 1: Adding computers (nodes) to the network...")
    nodes = [
        ComputeNode("node1", latency=10.0, bandwidth=100.0, energy=0.5, gpu_count=4),
        ComputeNode("node2", latency=15.0, bandwidth=80.0, energy=0.7, gpu_count=2),
    ]
    for node in nodes:
        chain.register_node(node)
        chain.reporting.submit_report(node, {"metrics": node.to_dict()})

    print("\nStep 2: A client submits an AI task to be processed...")
    workload = AIWorkload("workload1", required_gpus=2, data_size=100.0, compute_intensity=1000.0, priority=0.8)
    chain.submit_workload(workload, "client1")

    print("\nStep 3: Preparing the AI decision system to assign the task...")
    try:
        chain.setup_rl_environment()
        policy_hash = chain.rl_policy.commit_policy()

        print("\nStep 4: Assigning the AI task to the best computer...")
        matched_node = chain.orchestrate_workload(workload, "client1")

        print("\nStep 5: Checking the Community Fund for fair access...")
        print(f"Community Fund now has {chain.token_system.community_fund} AICC to support more users.")

        print("\nStep 6: Voting on a new network rule (e.g., rewarding green energy use)...")
        if chain.vote_on_policy("client1", "renewable_incentive", 10.0):
            print("The vote passed, and the network rule is updated!")
        else:
            print("The vote failed due to insufficient voting power.")

    except ValueError as e:
        print(f"Error in process: {e}")

    print("\n=== AICChain Demo Complete ===")

if __name__ == "__main__":
    main()
