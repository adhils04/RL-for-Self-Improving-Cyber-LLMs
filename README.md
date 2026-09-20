# RL for Self-Improving Cyber LLMs (MAPPO Co-evolution)

## Project Overview
This repository contains the Capstone Project: **Multi-Agent Proximal Policy Optimization (MAPPO) for Co-evolving Adversarial and Defensive Cyber LLMs**. 

The core objective of this project is to create a self-improving cybersecurity environment where an **Attacker** (a red-teaming agent generating prompt injections) co-evolves alongside a **Defender** (an LLM executing legitimate cyber-support tasks while resisting attacks). 

By utilizing a **Centralized Critic (MAPPO)**, the system allows the Attacker to learn from the global state of the environment—including the Defender's internal policies—leading to more stable, diverse, and robust adversarial generation compared to a decentralized Independent PPO approach.

## Key Architectures

### 1. The MAPPO Setup (Centralized Critic)
- **Attacker Policy:** An MLP-based policy that generates varied prompt injection vectors.
- **Centralized Critic:** A `distilbert-base-uncased` model that evaluates the global state (including task context and environment state) to provide a centralized value function.
- **Defender Policy:** An LLM agent (equipped with a deterministic tool-calling policy gate) that attempts to execute safe tool calls while blocking injected malicious intent.

### 2. Independent PPO (Ablation)
- An ablation baseline where the Centralized Critic is disabled, forcing the Attacker and Defender to learn independently without global state sharing. Used to benchmark the stability and efficacy of MAPPO.

## Key Findings & Results

After running 500-episode co-evolution training loops on Kaggle, the following results were observed:

1. **Policy Convergence & Stability:**
   - The Centralized Critic (MAPPO) maintained a much more stable equilibrium in the Attack Success Rate (ASR) compared to Independent PPO. The global perspective provided by the critic helped regularize the adversarial policy updates.
   - The MAPPO Attacker's reward converged closer to zero smoothly, avoiding the deep, volatile drops seen in the Independent PPO run.

2. **Attack Diversity (Zero-Day Potential):**
   - Using `all-MiniLM-L6-v2` semantic embeddings to measure the pairwise cosine distance across 970 generated adversarial traces, the MAPPO Attacker achieved an **Average Pairwise Cosine Distance of 0.5361**.
   - This **HIGH Diversity** proves the Attacker successfully explored varied linguistic structures and prompt patterns, ensuring it acts as a comprehensive zero-day generator rather than exploiting a single, repeating vulnerability loop.

## Repository Structure

- `src/attacker_policy/`: The red-teaming PPO Attacker generating adversarial payloads.
- `src/defender_policy/`: The LLM Defender, including the strict policy gate (`gate.py`) and safety sandbox.
- `src/coevolution_env.py`: The custom Gym-like environment managing the adversarial interaction, state tracking, and reward calculation.
- `scripts/`: Training, evaluation, and plotting scripts (e.g., `run_mappo_coevolution.py`, `generate_phase3_report.py`, `analyze_attack_diversity.py`).
- `reports/`: Contains the generated training metrics, adversarial traces, plot images, and final markdown reports.
- `configs/`: Configuration YAMLs for MAPPO and ablation runs.

## How to Run

**1. Install Dependencies**
```bash
git clone https://github.com/subhojeetchanda/RL-for-Self-Improving-Cyber-LLMs.git
cd RL-for-Self-Improving-Cyber-LLMs
pip install -r requirements.txt
```

**2. Run Co-evolution Training**
(Recommended to run on a GPU-enabled environment like Kaggle/Colab)
```bash
python scripts/run_mappo_coevolution.py --config configs/mappo_config.yaml
```

**3. Generate Final Analysis Reports & Charts**
```bash
# Generate ASR and Reward Learning Curves
python scripts/generate_phase3_report.py

# Run Attack Diversity Analysis
python scripts/analyze_attack_diversity.py
```

## Team & Capstone Details
This repository consolidates the final Phase 1-3 deliverables for the MAPPO Security Capstone. For detailed trace logs and final metric charts, please see the `reports/` directory and `reports/phase3_final_report.md`.
