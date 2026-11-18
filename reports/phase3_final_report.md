# Preliminary implementation and evidence audit

## Scope

This report records what the current checkout implements and what remains necessary before making a comparative performance claim. Training is planned on an external compute environment. No new model training or evaluation was performed for this report.

## Implemented model paths

The repository now contains three distinct paths:

1. A causal-LM SFT/adversarial-MAPPO path. `src/train_sft.py` applies completion-only LoRA SFT with class-balanced ALLOW/DEFEND sampling and fresh synthetic protected values in training prompts. SFT and the MAPPO variant use the same stochastic validation score for decision accuracy and exact exposure of synthetic protected values; validation loss breaks SFT ties. `scripts/run_llm_mappo.py` starts from that adapter, samples full responses under PPO, learns an attack-family policy, uses separate centralized state-value critics for the attacker and Defender, and penalizes exact exposure of a fresh synthetic protected value during training and validation. This is a competitive adaptation of MAPPO's centralized-training structure, not a reproduction of its cooperative benchmarks. `scripts/evaluate_llm_four_quadrants.py` records generated-response decisions and runtime-only canary exposure counts without storing raw completions or the canary. This path is implemented but has not been trained or empirically evaluated.
2. A DistilBERT binary sequence-classification experiment. `scripts/run_sft_baseline.py` trains the SFT classifier. `scripts/run_mappo_coevolution.py` initializes its defender from that SFT checkpoint and updates it using the configured environment and reward signals. It predicts allow/defend labels rather than generating assistant responses.
3. A historical Qwen2.5-0.5B LoRA smoke test trained on a small sample. It is not a MAPPO checkpoint and is not evidence for the comparison.

On attack episodes, the new attacker selects among configured training attack families using signed hashed word and bigram features. Benign episodes are sampled exogenously at a fixed probability and are not attacker actions. The default set excludes obfuscated, delimiter-smuggling, logic-framing, multilingual, and roleplay families so these can be reserved for OOD evaluation. The attacker does not generate unconstrained adversarial language. In the agent adapter, `DEFEND` blocks proposed tool calls; `ALLOW` still passes through the deterministic gate and its allowlist. The four-quadrant evaluator itself does not execute tools.

## Evidence available in this checkout

The repository includes a historical Qwen SFT adapter, but it lacks the run manifest required by the new evaluator and has no matching MAPPO adapter. There is no reproducible causal-LM comparison pair, and the required DistilBERT checkpoints are also absent. The checked-in reports contain conflicting summary values and do not provide a sufficient chain from cases to checkpoint hashes. The four-quadrant benchmark has already been inspected, so it is exploratory and cannot serve as an untouched confirmatory test set. Previous bootstrap draws from the same fixed benchmark do not represent independent training runs and do not support the reported hypothesis tests.

Accordingly, the earlier numerical superiority, significance, and generalization claims are withdrawn pending a new evaluation. The current repository state does not establish that MAPPO outperforms SFT, nor that either classifier protects a generative language model from retrieval of a secret. Keep the real evaluation canary out of the repository and all training data; report only whether it was exposed during a controlled evaluation. MAPPO's synthetic training values are generated per episode and are not written to artifacts.

## Required confirmatory evaluation

Before training, freeze a new test set with four quadrants, separate from training and validation by case and attack family where applicable. Keep the gate version, base model, tokenizer, input formatting, action threshold, and evaluation procedure identical across conditions. Train the SFT model first, initialize MAPPO from that exact SFT checkpoint, and record configuration, data hashes, software versions, seeds, and checkpoint hashes. Use validation data for model selection and inspect the confirmatory test set only after all choices are fixed.

Report per-quadrant allow and defend rates, case-level outcomes, benign false-refusal rate, injection miss rate, and uncertainty. Use multiple independent training seeds and treat runs as the replication unit for algorithm-level claims. If only one trained pair is available, label the result a single-run, case-conditional comparison; do not infer training variability or claim statistical significance from repeated draws of the same cases.

The prespecified directional hypotheses are: both methods should allow common benign cases and defend common injection cases; on unique benign cases MAPPO should allow more often; on unique injection cases MAPPO should defend more often. State thresholds and minimum useful effect sizes before accessing the new test set. Report all four quadrants, including failures, and avoid selecting examples after seeing model outputs.

## Submission status

This is an implementation and evaluation plan, not a completed empirical paper. Replace this report with the measured results only after the external training and sealed evaluation are complete. If those runs cannot be completed before submission, present the work as a system/protocol contribution and state that the comparative result remains unverified.
