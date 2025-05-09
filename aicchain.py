import numpy as np
from gym import Env, spaces
from stable_baselines3 import PPO
import random
import torch as th

# Simulated blockchain for AICC and AICG tokens
class Blockchain:
    def __init__(self):
        self.balances = {"user": 200, "provider": 0, "community_fund": 0}
        self.aicg_holders = [1, 2, 3]
        self.transactions = []

    def transfer_aicc(self, sender, receiver, amount):
        if self.balances[sender] >= amount:
            self.balances[sender] -= amount
            self.balances[receiver] += amount
            self.transactions.append((sender, receiver, amount))
            print(f"   Transaction: {sender} -> {receiver}: {amount} AICC")
            return True
        return False

    def vote_subsidy(self):
        votes = [random.choice([True, True, False]) for _ in self.aicg_holders]
        print(f"   AICG Votes: {['Approve' if v else 'Reject' for v in votes]}")
        return sum(votes) > len(votes) / 2

# Gym environment for compute allocation
class ComputeAllocationEnv(Env):
    def __init__(self, gpu_req, bandwidth_req, latency_max):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.nodes = [
            {"gpu": 8, "bandwidth": 100, "latency": 10, "renewable": 1, "carbon": 0.1},  # Solar
            {"gpu": 4, "bandwidth": 50, "latency": 20, "renewable": 0, "carbon": 0.4},   # Non-renewable
            {"gpu": 16, "bandwidth": 200, "latency": 5, "renewable": 1, "carbon": 0.1}   # Solar
        ]
        self.job = {"gpu_req": gpu_req, "bandwidth_req": bandwidth_req, "latency_max": latency_max}
        self.carbon_emitted = 0

    def reset(self):
        state = np.array([
            self.job["gpu_req"]/16, self.job["bandwidth_req"]/200, self.job["latency_max"]/30,
            self.nodes[0]["gpu"]/16, self.nodes[0]["bandwidth"]/200, self.nodes[0]["latency"]/30,
            self.nodes[0]["renewable"]
        ])
        return state

    def step(self, action):
        node = self.nodes[action]
        reward = 0
        compatible = (node["gpu"] >= self.job["gpu_req"] and
                      node["bandwidth"] >= self.job["bandwidth_req"] and
                      node["latency"] <= self.job["latency_max"])
        if compatible:
            reward += 10
            if node["renewable"]:
                reward += 5
        else:
            reward -= 5
        self.carbon_emitted = node["carbon"]
        done = True
        state = self.reset()
        return state, reward, done, {"node": node, "compatible": compatible}

# Interactive demo with detailed RL and blockchain logs
def run_interactive_demo():
    print("AIComputeChain Demo: Training AI for Credit Scoring")
    print("Enter job requirements for your credit scoring AI model:")

    # User input
    try:
        gpu_req = int(input("Number of GPUs (4-16): "))
        bandwidth_req = int(input("Bandwidth in Mbps (50-200): "))
        latency_max = int(input("Maximum latency in ms (5-20): "))
    except ValueError:
        print("Invalid input. Using defaults: 8 GPUs, 100Mbps, 15ms.")
        gpu_req, bandwidth_req, latency_max = 8, 100, 15

    blockchain = Blockchain()
    env = ComputeAllocationEnv(gpu_req, bandwidth_req, latency_max)

    # Train RL model
    print("\nTraining RL model to optimize node matching...")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    print("RL training complete: Policy optimized for compatibility and sustainability.")

    # Simulate workflow
    print("\nRunning your job...")
    print("1. Providers registered nodes:")
    for i, node in enumerate(env.nodes):
        print(f"   Node {i}: {node['gpu']} GPUs, {node['bandwidth']}Mbps, {node['latency']}ms, "
              f"{'Solar' if node['renewable'] else 'Non-renewable'}")
    print(f"2. You submitted: {gpu_req} GPUs, {bandwidth_req}Mbps, <{latency_max}ms latency.")

    # RL matching with detailed logs
    print("3. RL Matching Process:")
    obs = env.reset()
    print(f"   State: [job: {obs[0]:.2f} GPUs, {obs[1]:.2f} Mbps, {obs[2]:.2f} ms; "
          f"node: {obs[3]:.2f} GPUs, {obs[4]:.2f} Mbps, {obs[5]:.2f} ms, renewable: {obs[6]}]")
    obs_tensor = th.as_tensor(obs[np.newaxis], device='cpu')
    action_probs = model.policy.get_distribution(obs_tensor).distribution.probs
    print(f"   Action Probabilities: {action_probs.detach().numpy()[0].round(3)}")
    action, _ = model.predict(obs, deterministic=True)
    state, reward, done, info = env.step(action)
    node = info["node"]
    compatible = info["compatible"]
    print(f"   Action: Selected node {action} ({node['gpu']} GPUs, {node['bandwidth']}Mbps, "
          f"{node['latency']}ms, {'Solar' if node['renewable'] else 'Non-renewable'})")
    print(f"   Reward: {reward} (Compatibility: {10 if compatible else -5}, "
          f"Sustainability: {5 if node['renewable'] and compatible else 0})")

    cost = 100
    print(f"4. Payment Process:")
    print(f"   Base cost: {cost} AICC")
    blockchain.transfer_aicc("user", "provider", cost * 0.9)
    blockchain.transfer_aicc("user", "community_fund", cost * 0.1)

    print("5. Subsidy Voting:")
    subsidy = cost * 0.5 if blockchain.vote_subsidy() else 0
    if subsidy:
        print(f"   Approved {subsidy} AICC subsidy.")
        blockchain.transfer_aicc("community_fund", "user", subsidy)
    else:
        print("   Subsidy rejected.")

    print("6. Provider Rewards:")
    bonus = 10 if node["renewable"] else 0
    if bonus:
        blockchain.transfer_aicc("community_fund", "provider", bonus)
        print(f"   Provider earns {cost * 0.9 + bonus} AICC (incl. {bonus} AICC solar bonus).")
    else:
        print(f"   Provider earns {cost * 0.9} AICC.")

    effective_cost = cost - subsidy
    carbon_saving = (0.4 - node["carbon"]) / 0.4 * 100
    print(f"7. Job Outcome:")
    print(f"   Cost: {effective_cost} AICC ({int((cost - effective_cost)/cost*100)}% cheaper than IO.Net)")
    print(f"   Carbon: {int(carbon_saving)}% reduction (solar: {node['carbon']} vs. non-renewable: 0.4)")

    # Results table
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Metric':<20} | {'Value':<25}")
    print("-" * 50)
    print(f"{'Effective Cost':<20} | {effective_cost} AICC")
    print(f"{'Cost Saving':<20} | {int((cost - effective_cost)/cost*100)}% vs. IO.Net")
    print(f"{'Carbon Reduction':<20} | {int(carbon_saving)}%")
    print(f"{'User Balance':<20} | {blockchain.balances['user']} AICC")
    print(f"{'Provider Balance':<20} | {blockchain.balances['provider']} AICC")
    print(f"{'Community Fund':<20} | {blockchain.balances['community_fund']} AICC")
    print("-" * 50)

# Run demo
if __name__ == "__main__":
    run_interactive_demo()
