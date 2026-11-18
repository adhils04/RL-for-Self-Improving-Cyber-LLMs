# Preliminary implementation and evidence audit

## Scope

This report records what the current checkout implements and what remains necessary before making a comparative performance claim. Training is planned on an external compute environment. No new model training or evaluation was performed for this report.

## Implemented model paths

The repository contains two separate experiments:

1. A Qwen2.5-0.5B LoRA language-model smoke test, represented in the working copy by an untracked local adapter trained for one epoch on 20 formatted examples. This is not the MAPPO model used in the classifier comparison and is not part of the versioned experiment artifacts.
2. A DistilBERT binary sequence-classification pipeline. `scripts/run_sft_baseline.py` trains the SFT classifier. `scripts/run_mappo_coevolution.py` initializes its defender from that SFT checkpoint and updates it using the configured environment and reward signals. This pipeline predicts allow/refuse labels; it does not generate assistant responses or execute tools.

The MAPPO attacker selects among fixed attack categories. Its eight-dimensional observation is an MD5 hash encoding of the task identifier, not a semantic task representation. The environment and payload templates constrain its action space, so this should not be described as free-form adaptive red teaming. The deterministic gate is a separate enforcement layer and must be held constant across conditions.

## Evidence available in this checkout

The DistilBERT SFT and MAPPO checkpoints required to reproduce the comparison are absent. The checked-in reports contain conflicting summary values and do not provide a sufficient chain from cases to checkpoint hashes. The four-quadrant benchmark has already been inspected, so it is exploratory for this project and cannot serve as an untouched confirmatory test set. Previous bootstrap draws from the same fixed benchmark do not represent independent training runs and do not support the reported hypothesis tests.

Accordingly, the earlier numerical superiority, significance, and generalization claims are withdrawn pending a new evaluation. The current repository state does not establish that MAPPO outperforms SFT, nor that either classifier protects a generative language model from retrieval of a secret. The word used as a canary must remain outside the repository and training data; report only whether it was exposed during a controlled evaluation.

## Required confirmatory evaluation

Before training, freeze a new test set with four quadrants, separate from training and validation by case and attack family where applicable. Keep the gate version, base model, tokenizer, input formatting, action threshold, and evaluation procedure identical across conditions. Train the SFT model first, initialize MAPPO from that exact SFT checkpoint, and record configuration, data hashes, software versions, seeds, and checkpoint hashes. Use validation data for model selection and inspect the confirmatory test set only after all choices are fixed.

Report per-quadrant allow and defend rates, case-level outcomes, benign false-refusal rate, injection miss rate, and uncertainty. Use multiple independent training seeds and treat runs as the replication unit for algorithm-level claims. If only one trained pair is available, label the result a single-run, case-conditional comparison; do not infer training variability or claim statistical significance from repeated draws of the same cases.

The prespecified directional hypotheses are: both methods should allow common benign cases and defend common injection cases; on unique benign cases MAPPO should allow more often; on unique injection cases MAPPO should defend more often. State thresholds and minimum useful effect sizes before accessing the new test set. Report all four quadrants, including failures, and avoid selecting examples after seeing model outputs.

## Submission status

This is an implementation and evaluation plan, not a completed empirical paper. Replace this report with the measured results only after the external training and sealed evaluation are complete. If those runs cannot be completed before submission, present the work as a system/protocol contribution and state that the comparative result remains unverified.
