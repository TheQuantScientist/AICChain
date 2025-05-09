# ACChain: Decentralized AI Compute Allocation Platform

Welcome to **AICChain**, a decentralized platform for AI compute allocation that optimizes resource matching, incentivizes sustainability, and leverages blockchain for transparent transactions. This repository contains a Python-based demo showcasing the core functionality of AICChain, including reinforcement learning (RL) for compute node selection, a simulated blockchain for AICC and AICG tokens, and a community-driven subsidy voting mechanism.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
AICChain is designed to connect AI compute providers with users seeking resources for tasks like credit scoring, model training, or inference. The platform uses a reinforcement learning model (PPO from Stable Baselines3) to match compute jobs to nodes based on GPU, bandwidth, latency, and sustainability criteria. A simulated blockchain manages AICC (transactional token) and AICG (governance token) to facilitate payments, community fund contributions, and provider incentives for renewable energy usage.

The demo simulates a credit scoring AI job, demonstrating how AICChain optimizes node selection, processes payments, and rewards sustainable practices while reducing costs and carbon emissions compared to competitors like IO.Net.

## Features
- **RL-based Compute Allocation**: Uses PPO to select optimal compute nodes based on job requirements (GPU, bandwidth, latency) and sustainability preferences.
- **Blockchain Integration**: Simulates AICC token transactions for payments, community fund contributions, and provider rewards.
- **Sustainability Incentives**: Rewards providers using renewable energy (e.g., solar-powered nodes) with bonus AICC tokens.
- **Community Governance**: AICG token holders vote on subsidies to reduce user costs, funded by a 10% community fund allocation.
- **Cost and Carbon Efficiency**: Achieves up to 50% cost savings and significant carbon emission reductions compared to traditional platforms.
- **Interactive Demo**: User-friendly interface for inputting job requirements and viewing detailed logs of RL decisions, transactions, and outcomes.

## Demo
The demo (`aicchain.py`) simulates an end-to-end workflow for a credit scoring AI job:
1. **Input Job Requirements**: Specify GPU count (4-16), bandwidth (50-200 Mbps), and maximum latency (5-20 ms).
2. **Node Registration**: Displays three compute nodes (two solar-powered, one non-renewable) with varying specs.
3. **RL Matching**: Trains a PPO model to select the best node, logging state, action probabilities, and rewards.
4. **Payment Processing**: Transfers AICC tokens from user to provider (90%) and community fund (10%).
5. **Subsidy Voting**: AICG holders vote to approve a 50% cost subsidy, funded by the community fund.
6. **Provider Rewards**: Awards bonus AICC for renewable energy usage.
7. **Results Summary**: Displays effective cost, cost savings (vs. IO.Net), carbon reduction, and final balances.

Sample output includes a detailed results table:
```
Results Summary:
--------------------------------------------------
Metric               | Value
--------------------------------------------------
Effective Cost       | 50 AICC
Cost Saving          | 50% vs. IO.Net
Carbon Reduction     | 75%
User Balance         | 150 AICC
Provider Balance     | 100 AICC
Community Fund       | 0 AICC
--------------------------------------------------
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aicchain.git
   cd aicchain
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure dependencies are installed (see [Dependencies](#dependencies)).
2. Run the demo:
   ```bash
   python aicchain.py
   ```
3. Follow the prompts to input job requirements (or press Enter for defaults: 8 GPUs, 100 Mbps, 15 ms).
4. Review the detailed logs and results table to understand the workflow.

## Code Structure
- `aicchain.py`: Main script containing the demo logic.
  - `Blockchain` class: Simulates AICC and AICG token transactions and subsidy voting.
  - `ComputeAllocationEnv` class: Gym environment for RL-based node matching.
  - `run_interactive_demo()`: Orchestrates the demo workflow with user input and detailed logs.

## Dependencies
- Python 3.8+
- `numpy`: For numerical computations and state normalization.
- `gym`: For the RL environment.
- `stable-baselines3`: For the PPO reinforcement learning model.
- `torch`: For PyTorch-based RL computations.

Install dependencies using:
```bash
pip install numpy gym stable-baselines3 torch
```

Or use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Contributing
We welcome contributions to enhance AICChain! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request with a clear description of your changes.

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.
