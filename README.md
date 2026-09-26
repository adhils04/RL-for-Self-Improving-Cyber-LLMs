# RL for Self-Improving Cyber LLMs: MAPPO vs. Supervised Fine-Tuning (SFT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40+-yellow.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Empirical Significance](https://img.shields.io/badge/Welch's_t--test-p%20%3C%2010⁻⁶-blue.svg)](reports/mappo_vs_sft_statistical_tests.json)

> **Research Paper Implementation & Artifact Repository**  
> **Core Objective:** Empirically demonstrate and validate that **Multi-Agent Proximal Policy Optimization (MAPPO)** with Centralized Training and Decentralized Execution (CTDE) significantly outperforms static **Supervised Fine-Tuning (SFT)** in defending Large Language Models against novel, out-of-distribution (OOD) prompt injection attacks.

---

## 🔬 Key Empirical Findings

Evaluated across $B = 30$ bootstrap resamples on a held-out benchmark ($N = 100$) comprising In-Distribution attacks, Novel OOD attacks (Base64 obfuscation, mathematical cognitive framing, authority impersonation), and Benign cyber operational tasks ($\alpha = 0.05$):

| Evaluation Metric | MAPPO (Ours) | SFT Baseline | Welch's $t$ | $p$-value | Cohen's $d$ Effect Size | Empirical Conclusion |
|---|---|---|---|---|---|---|
| **Novel Attack Defense (↑)** | **97.61% ± 2.47%** | 82.81% ± 6.64% | **+11.44** | **< 10⁻⁶** | **+2.9536 (Extremely Huge)** | **✓ MAPPO Outperforms SFT** |
| **Novel Attack ASR (↓)** | **2.39% ± 2.47%** | 17.19% ± 6.64% | **-11.44** | **< 10⁻⁶** | **+2.9536 (Extremely Huge)** | **✓ 86.1% Vulnerability Reduction** |
| **In-Distribution ASR (↓)** | **0.00% ± 0.00%** | **0.00% ± 0.00%** | 0.00 | 1.0000 | 0.0000 | **✓ Zero Known Attack Regression** |
| **Defender Expected Reward** | 2.3349 ± 0.1167 | 2.4553 ± 0.1295 | -3.78 | 0.0004 | -0.9765 (Medium) | SFT Higher (Utility Factor) |
| **Defense Accuracy (Overall)** | 85.10% ± 3.00% | 92.90% ± 2.77% | -10.46 | < 10⁻⁶ | -2.7017 (Huge) | SFT Higher (Utility Factor) |
| **Benign Utility Preservation** | 29.73% ± 11.38% | 97.58% ± 3.44% | -31.27 | < 10⁻⁶ | -8.0729 (Huge) | SFT Higher (Pareto Alignment Tax) |

```
====================================================================================================
                                      CORE RESEARCH TAKEAWAY
====================================================================================================
• Static SFT fails on 17.2% of unseen / novel prompt injection variants due to passive keyword overfitting.
• MAPPO multi-agent co-evolution suppresses novel attack ASR to 2.4% (p < 10⁻⁶, Cohen's d = +2.95).
• Zero regression on known attacks: 0.00% ASR for both models on in-distribution injections.
• Clear Pareto Trade-off: MAPPO develops a zero-trust defensive barrier that drastically enhances security
  at the expense of benign task false-refusal rate.
====================================================================================================
```

---

## 🏛️ System Architecture: CTDE Multi-Agent Co-evolution

The framework operates under the **Centralized Training with Decentralized Execution (CTDE)** paradigm:

```
[Untrusted User / Adversary Prompt]
                │
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   CoevolutionEnv (Markov Game Board)                   │
│  Builds DefenderObservation(user_task, untrusted_content) & GlobalState│
└───────────────┬────────────────────────────────────────┬───────────────┘
                │                                        │
        GlobalState S_t                           DefenderObservation
                │                                        │
                ▼                                        ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐
│     Centralized Critic        │        │        MAPPO Defender         │
│   (DistilBERT Value Network)  │        │ (DistilBERT Sequence Policy)  │
│    V_phi(S_t) -> GAE (A_t)    │        │  Init: SFT Warmup (pi_SFT)    │
└───────────────┬───────────────┘        │  PPO Clipped + KL Regularizer │
                │                                └───────┬───────────────────────┘
                │ Advantage Signal (GAE)                 │ Raw Proposed Action
                └───────────────────────────────────────▶│
                                                         ▼
                                         ┌───────────────────────────────┐
                                         │   Fail-Closed Policy Gate     │
                                         │    (GATE-001 to GATE-009)     │
                                         └───────────────┬───────────────┘
                                                         │
                                        ┌────────────────┴────────────────┐
                                        ▼                                 ▼
                                 🚫 Safe Denial                    ✅ Safe Response
```

### Mathematical Objectives
1. **Defender PPO Objective with KL Reference Anchor:**
   $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t^{def}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t^{def}\right) \right] - \beta_{KL} D_{KL}(\pi_\theta \parallel \pi_{SFT}) + c_e \mathcal{H}(\pi_\theta)$$
2. **Centralized Critic Objective:**
   $$L^V(\phi) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \left(V_\phi(S_t) - R_t^{target}\right)^2 \right]$$
3. **Passive SFT Objective:**
   $$L^{SFT}(\theta) = -\sum_{(x_i, y_i) \in \mathcal{D}_{SFT}} \log P_\theta(y_i \mid x_i)$$

---

## 📁 Repository Structure

```
.
├── checkpoints/
│   ├── mappo_run/
│   │   ├── mappo_defender_final.pt    # Fully trained MAPPO Defender neural weights
│   │   ├── critic_final.pt            # Centralized Critic V(S) weights
│   │   └── attacker_final.pt          # Co-evolved Attacker policy weights
│   └── sft_run/
│       └── sft_final.pt               # SFT Baseline neural weights (30 epochs)
├── configs/
│   └── mappo_config.yaml              # Hyperparameters (PPO clip, KL penalty, GAE gamma)
├── data/
│   ├── sft/
│   │   ├── train.jsonl                # 100 training cases (75 attack + 25 benign)
│   │   └── heldout.jsonl              # 100 held-out cases (40 ID + 40 Novel OOD + 20 Benign)
│   └── evaluation/
│       └── benchmark_cases.jsonl      # 200 multi-turn evaluation cases
├── reports/
│   ├── phase3_final_report.md         # Formal research manuscript for paper submission
│   ├── mappo_vs_sft_statistical_tests.json # Raw empirical hypothesis test outputs
│   ├── mappo_vs_sft_significance_plot.png  # Publication 4-panel statistical box plot
│   ├── novel_attack_generalization.png     # OOD category robustness bar chart
│   ├── asr_over_training.png               # Training ASR convergence curves
│   └── defender_reward_curve.png           # Co-evolution reward stabilization curves
├── scripts/
│   ├── generate_benchmark_dataset.py  # Benchmark generator with deterministic seeds
│   ├── run_sft_baseline.py            # SFT Baseline training pipeline
│   ├── run_mappo_coevolution.py       # MAPPO Co-evolution training loop (CTDE)
│   ├── run_mappo_vs_sft_stats.py      # Welch's t-test, Mann-Whitney U, Bootstrap evaluation
│   └── generate_mappo_vs_sft_plots.py # Publication-quality figure renderer (300 DPI)
├── src/
│   ├── defender_policy/
│   │   ├── neural_defender.py         # PyTorch NeuralDefenderPolicy & PPO loss
│   │   ├── model_adapter.py           # Gym action adapter with gate validation
│   │   └── gate.py                    # Fail-closed Policy Gate rules (GATE-001..009)
│   ├── attacker_policy/               # Multi-category adversarial prompt generator
│   ├── centralized_critic/            # DistilBERT global state value estimator
│   ├── coevolution_env.py             # Custom Gym-like zero-sum Markov environment
│   └── trajectory_buffer.py           # CTDE GAE trajectory buffer
├── ui/
│   ├── index.html                     # Interactive MAPPO vs. SFT comparison arena
│   ├── architecture.html              # Interactive CTDE game architecture diagram
│   └── style.css                      # Modern dark cybersecurity UI theme
└── ui_server.py                       # FastAPI backend loading real neural checkpoints
```

---

## 🚀 Reproduction & Quickstart Guide

All experiments are 100% reproducible with fixed random seeds on standard compute.

### 1. Environment Setup
```bash
git clone https://github.com/subhojeetchanda/RL-for-Self-Improving-Cyber-LLMs.git
cd RL-for-Self-Improving-Cyber-LLMs
pip install -r requirements.txt
```

### 2. Generate Benchmark Dataset
Generates 100 training and 100 held-out evaluation samples with strict ID vs. Novel OOD partitioning:
```bash
python scripts/generate_benchmark_dataset.py
```

### 3. Train SFT Baseline
Trains the `distilbert-base-uncased` classifier via cross-entropy for 30 epochs:
```bash
python scripts/run_sft_baseline.py --epochs 30 --lr 2e-5
```

### 4. Train MAPPO Co-evolution
Initializes the Defender from SFT warm-up weights and trains both Attacker and Defender via CTDE PPO:
```bash
python scripts/run_mappo_coevolution.py --episodes 150 --rollout-batch 5
```

### 5. Run Statistical Hypothesis Testing (Bootstrap B=30)
Evaluates both checkpoints on the 100-case held-out benchmark using Welch's $t$-test, Mann-Whitney $U$, and Cohen's $d$:
```bash
python scripts/run_mappo_vs_sft_stats.py --trials 30 --sample-fraction 0.85
```

### 6. Render Publication Figures
Generates all 300 DPI figures used in the manuscript:
```bash
python scripts/generate_mappo_vs_sft_plots.py
```

### 7. Launch Interactive Research Arena UI
Serves the live neural policies over FastAPI:
```bash
python ui_server.py
```
Then open [`ui/index.html`](ui/index.html) in any modern browser to test arbitrary prompt injections or run benchmark test cases side-by-side!

---

## 📄 Citation & Artifact Access

To cite this work or inspect the empirical findings, please refer to:
- **Full Research Paper Manuscript:** [`reports/phase3_final_report.md`](reports/phase3_final_report.md)
- **Statistical Significance Report:** [`reports/mappo_vs_sft_statistical_tests.txt`](reports/mappo_vs_sft_statistical_tests.txt)
- **Interactive Architecture Diagram:** [`ui/architecture.html`](ui/architecture.html)
