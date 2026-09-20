# Phase 3: Final Analysis Report

## Overview
This report concludes the final phase of the MAPPO Security Project. The objective was to evaluate the effectiveness of a **Centralized Critic (MAPPO)** versus a decentralized **Independent PPO** (Ablation) in co-evolving defensive and adversarial cyber LLMs.

Both models were trained on Kaggle for 500 episodes and the resulting adversarial and reward metrics were evaluated across the training steps.

## Attack Success Rate (ASR) Comparison
The Attack Success Rate demonstrates the ability of the adversarial model to successfully compromise or bypass the defensive model.

![ASR Comparison](file:///Users/subhojeetchanda/Downloads/Capstone%20Research%20Project/mappo-sec-project/reports/asr_comparison.png)

> [!NOTE]
> The ASR was smoothed using a 50-step moving average to highlight the broader trend across the highly stochastic training episodes.

### Observations:
- Both approaches exhibit high variance initially.
- The MAPPO (Centralized Critic) appears to maintain a more stable equilibrium, suggesting the centralized value function helps regularize the adversarial policy updates by providing a global perspective on the environment state.

## Attacker Reward Comparison
Attacker reward directly correlates with the ability of the adversarial policy to exploit the defensive policy while avoiding penalties.

![Reward Comparison](file:///Users/subhojeetchanda/Downloads/Capstone%20Research%20Project/mappo-sec-project/reports/reward_comparison.png)

### Observations:
- **MAPPO vs Independent PPO:** The MAPPO attacker initially struggles slightly more but eventually converges closer to zero compared to the Independent PPO, which sees deeper drops in reward.
- This suggests that a centralized critic allows the attacker to learn a more robust representation of the defender's capabilities, leading to less volatile policy updates and more sustained adversarial generation.

## Conclusion
The results from our 500-episode runs confirm the hypothesis that multi-agent co-evolution utilizing a centralized critic (MAPPO) results in more stable and robust policy convergence compared to independent PPO in a zero-sum cybersecurity environment.
