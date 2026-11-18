# RL for Self-Improving Cyber LLMs: MAPPO vs. Supervised Fine-Tuning (SFT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40+-yellow.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Research Paper Implementation & Artifact Repository**  
> **Research question:** Does MAPPO improve prompt-injection classification compared with an SFT classifier under controlled in-distribution and out-of-distribution conditions?

---

## Evidence status

The working copy has a local Qwen2.5-0.5B LoRA SFT smoke-test adapter, but it is not tracked in Git. It was trained for one epoch on 20 formatted examples and is not a MAPPO checkpoint. A separate DistilBERT classifier SFT/MAPPO pipeline is used for the comparison, but its trained checkpoints are absent from this checkout.

The saved reports contain conflicting metric values, and the four-quadrant report does not include per-case outputs or checkpoint hashes. Treat those figures as unverified artifacts, not as evidence of a completed LLM comparison. The MAPPO/SFT evaluation protocol is being tightened before new training runs. See [the four-quadrant experiment protocol](docs/four_quadrant_experiment_protocol.md) for the planned split, endpoints, and reporting rules.

---

## Implemented model paths

The MAPPO comparison is a DistilBERT binary classifier that predicts allow or defend. The attacker chooses among configured attack categories from an eight-dimensional hash encoding of the task identifier, while the critic supports policy updates. This is not free-form adaptive red teaming or a generative language-model agent, and the classifier does not execute tools. Separately, the untracked local Qwen2.5-0.5B LoRA smoke test exercises response generation; that adapter is not a MAPPO checkpoint.

The policy gate is a separate deterministic enforcement layer. Keep its version fixed across both classifier arms and report gate decisions separately from classifier predictions.

## Repository structure

```text
configs/mappo_config.yaml               MAPPO and model configuration
data/sft/train.jsonl                    245 source training records
data/sft/validation.jsonl               8 validation records
data/sft/heldout.jsonl                  100 held-out records
data/evaluation/benchmark_cases.jsonl  200 environment cases
data/evaluation/four_quadrants_benchmark.jsonl 100 exploratory cases
scripts/run_sft_baseline.py             DistilBERT SFT classifier trainer
scripts/run_mappo_coevolution.py        DistilBERT MAPPO classifier trainer
scripts/evaluate_four_quadrants.py      Paired stochastic four-quadrant evaluator
scripts/run_mappo_vs_sft_stats.py       Paired case-conditional descriptive analysis
reports/mappo_vs_sft_paired_comparison.json  Generated descriptive comparison with provenance
src/defender_policy/                    Classifier, policy adapter, and fixed gate
src/attacker_policy/                    Configured attack-category policy
src/centralized_critic/                 Critic used during policy training
reports/phase3_final_report.md          Preliminary implementation and evidence audit
docs/four_quadrant_experiment_protocol.md  Confirmatory evaluation protocol
```

The DistilBERT SFT and MAPPO checkpoints needed for the comparison are absent from this checkout. Existing evaluation reports are exploratory or unverified; they do not establish comparative performance.

---

## 🛡️ Interactive Research Arena UI

The UI is a classifier demonstration, not a generative-agent environment. It displays allow/defend predictions when trained checkpoints are available. Its heuristic prompt flag is illustrative and is not a policy gate or ground truth.

* **Side-by-Side View:** Compare classifier predictions from the **MAPPO** and **SFT** checkpoints.
* **Example Prompts:** Load examples for:
  1. *Recursive Base64 Obfuscation* (`Novel OOD`)
  2. *Authority Impersonation* (`Novel OOD`)
  3. *Direct Instruction Override* (`In-Dist`)
  4. *Cyber Log Inspection* (`Benign Task`)
* **Session Telemetry:** Counts classifier defend predictions and heuristic-flagged unsafe allow predictions.
* **Architecture Visualizer:** [`ui/architecture.html`](ui/architecture.html) describes the implemented training pipeline and its evidence status.

---

## 🚀 Quickstart & Reproduction Guide

Training requires the base model files and a supported accelerator. Checkpoints are excluded from Git, so the source code and reports alone do not reproduce a trained-model result.

### 1. Environment Setup

```bash
git clone https://github.com/subhojeetchanda/RL-for-Self-Improving-Cyber-LLMs.git
cd RL-for-Self-Improving-Cyber-LLMs

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies (PyTorch, Transformers, FastAPI, Uvicorn, etc.)
pip install -r requirements.txt
```

### 2. Prepare data splits

The dataset-generation script overwrites the training and held-out files and does not create the validation split. Do not run it over the checked-in data. Use the frozen files for exploratory setup; prepare and hash a separate train/validation/confirmatory-test set for a confirmatory study.

### 3. Train SFT Baseline
Trains the `distilbert-base-uncased` classifier via cross-entropy (default: 50 epochs) and saves checkpoint to `checkpoints/sft_run/sft_final.pt`:
```bash
python scripts/run_sft_baseline.py --epochs 30 --lr 2e-5
```

### 4. Train MAPPO Co-evolution
Initializes the DistilBERT Defender from the SFT classifier checkpoint and trains the attacker and defender using the configured episode count, saving checkpoints to `checkpoints/mappo_run/`:
```bash
python scripts/run_mappo_coevolution.py
```

> **Checkpoint requirement:** Checkpoint weights are stored in ignored directories. MAPPO now stops if the SFT classifier checkpoint is missing. Record each checkpoint hash with its evaluation report.

### 5. Evaluate the four quadrants

The evaluator requires both trained classifier checkpoints and records repeated stochastic decisions, per-case results, and checkpoint hashes:

```bash
python scripts/evaluate_four_quadrants.py --samples-per-case 20 --seed 42
```

The checked-in four-quadrant set has already been inspected and its summary reported. It is exploratory; use a separately frozen test set for a confirmatory paper result. Do not interpret bootstrap draws from one fixed test set as independent training runs.

### 6. Summarize paired case outcomes and render a figure

The paired analysis reports case-conditional intervals and does not estimate variation across training runs:
```bash
python scripts/run_mappo_vs_sft_stats.py --benchmark data/evaluation/four_quadrants_benchmark.jsonl
python scripts/generate_mappo_vs_sft_plots.py
```

These commands are exploratory when run on the checked-in benchmark. Use the separately frozen test file for a confirmatory comparison.

### 7. Launch Interactive Research Arena UI
Start the FastAPI backend server:
```bash
python ui_server.py
```
Then open the frontend in your browser:
```bash
# macOS
open ui/index.html

# Linux / Windows
# Open ui/index.html in any modern web browser
```

---

## 🧪 Test Suite

The repository records a 430-test suite covering the CTDE environment, policy gate rules, GAE advantage buffer, leak detector, and neural policies. This suite was not rerun during the manuscript/evaluation revisions:

```bash
PYTHONPATH=. python3 -m pytest
```
*Previously recorded output:* `430 passed`; rerun before relying on this status.

---

## 📄 Citation & Artifact Access

For the current implementation and evaluation status, please refer to:
- **Full Research Paper Manuscript:** [`reports/phase3_final_report.md`](reports/phase3_final_report.md)
- **Evaluation protocol:** [`docs/four_quadrant_experiment_protocol.md`](docs/four_quadrant_experiment_protocol.md)
- **Interactive Architecture Diagram:** [`ui/architecture.html`](ui/architecture.html)
- **Interactive Arena UI:** [`ui/index.html`](ui/index.html)
