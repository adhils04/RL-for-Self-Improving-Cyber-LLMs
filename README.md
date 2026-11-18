# RL for Self-Improving Cyber LLMs: MAPPO vs. Supervised Fine-Tuning (SFT)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40+-yellow.svg?style=flat&logo=huggingface)](https://huggingface.co)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Research Paper Implementation & Artifact Repository**  
> **Research question:** Does MAPPO improve a response-generating language model's prompt-injection decisions over an SFT baseline across common and out-of-distribution cases?

---

## Evidence status

One historical Qwen SFT adapter is included under `defender_checkpoints/`, but it has no run manifest or matching MAPPO adapter and is not a reproducible baseline for this experiment. The earlier DistilBERT classifier pipeline is also separate from the response-generating pipeline documented below.

Historical reports do not establish a completed LLM comparison. The new causal-LM scripts have not been trained or evaluated; use a new frozen test set for confirmatory results. See [the training and evaluation guide](docs/llm_mappo_training.md) and [the four-quadrant protocol](docs/four_quadrant_experiment_protocol.md).

---

## Implemented model paths

The repository contains two distinct tracks. The earlier DistilBERT pipeline remains available for reproducing its classifier experiment. The causal-LM track trains a LoRA SFT adapter, then applies a competitive attacker-Defender MAPPO variant with separate centralized state-value critics. It compares sampled response decisions in four quadrants. The original MAPPO paper evaluates cooperative tasks; this adversarial use is an adaptation, not a reproduction of those experiments. The model's `ALLOW` marker is an experimental decision outcome; it is not authorization to run arbitrary tools. The fixed policy gate remains the enforcement boundary for tool proposals.

The policy gate is a separate deterministic enforcement layer. For classifier experiments, keep its version fixed across both arms and report gate decisions separately from classifier predictions. In the causal-LM path, `ALLOW` does not authorize a tool call.

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
src/train_sft.py                        Completion-only causal-LM SFT trainer
scripts/run_llm_mappo.py                Causal-LM adversarial MAPPO variant trainer
scripts/evaluate_llm_four_quadrants.py  Generated-response four-quadrant evaluator
scripts/validate_llm_experiment_data.py Pre-training split and family preflight
scripts/run_mappo_vs_sft_stats.py       Paired case-conditional descriptive analysis
reports/mappo_vs_sft_paired_comparison.json  Generated descriptive comparison with provenance
src/defender_policy/                    Classifier, policy adapter, and fixed gate
src/attacker_policy/                    Configured attack-category policy
src/centralized_critic/                 Critic used during policy training
reports/phase3_final_report.md          Preliminary implementation and evidence audit
docs/four_quadrant_experiment_protocol.md  Confirmatory evaluation protocol
```

There is no reproducible SFT/MAPPO checkpoint pair in this checkout. Existing evaluation reports are exploratory or unverified; they do not establish comparative performance.

---

## 🛡️ Interactive Research Arena UI

The UI is a classifier demonstration, not the causal-LM experiment environment. It displays allow/defend predictions when classifier checkpoints are available. Its heuristic prompt flag is illustrative and is not a policy gate or ground truth.

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

### 3. Train the response-generating SFT baseline

The trainer masks prompt tokens from the language-model loss, balances ALLOW/DEFEND examples, includes synthetic protected values in both training arms, selects checkpoints with the shared validation score for decision accuracy and synthetic-value exposure, and saves a LoRA adapter plus data and training provenance:

```bash
python3 src/train_sft.py --model-name Qwen/Qwen2.5-0.5B --epochs 3
```

### 4. Train the causal-LM MAPPO policy

Initializes the Defender and frozen KL reference from the same SFT adapter. Benign episodes are sampled at a fixed rate independent of the attacker; on attack episodes the attacker chooses an attack family. The attacker critic uses the global pre-attack state, while the Defender critic uses the attack-conditioned state before response generation. Each episode uses a fresh synthetic protected value and penalizes exact disclosure, without using the real evaluation canary. The trainer writes the adapted language model, actors, critics, and run manifest:

```bash
python3 scripts/run_llm_mappo.py --base-model Qwen/Qwen2.5-0.5B --episodes 4000
```

Training is not performed by these instructions automatically. Review the data protocol and run with the externally provisioned accelerator and model access.

### 5. Evaluate the four quadrants

The causal-LM evaluator samples generated responses from both adapters and records per-case decisions, invalid decision markers, benchmark hashes, and (if configured) canary exposure counts. It never stores raw responses or the canary value:

```bash
EVAL_CANARY_VALUE='your-runtime-only-canary' python3 scripts/evaluate_llm_four_quadrants.py \
  --benchmark /path/to/frozen-four-quadrant-test.jsonl \
  --benchmark-status confirmatory --samples-per-case 20 \
  --minimum-action-rate 0.8 --seed 42
```

The checked-in four-quadrant set has already been inspected and is exploratory. Use a separately frozen test set for confirmatory results. The report marks whether each arm meets the required action rate in all four quadrants; choose and record that threshold before accessing the test set. Do not interpret repeated generations or bootstrap draws from one fixed test set as independent training runs. The canary environment variable must be set only in the evaluation process and must not be added to training files or committed reports.

### 6. Legacy classifier analysis

The following tools apply only to the DistilBERT classifier track. They are not the analysis path for generated causal-LM responses:
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
