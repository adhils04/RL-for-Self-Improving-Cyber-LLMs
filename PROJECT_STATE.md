# Project State

## Current phase

External SFT/MAPPO training and confirmatory evaluation are pending. The repository has separate Qwen language-model smoke-test and DistilBERT classifier pipelines. The classifier checkpoints required for comparison are absent from this checkout. Historical claims in earlier reports were withdrawn because the reports conflict and the resampling analysis did not support the reported significance tests. See `reports/phase3_final_report.md` and `docs/four_quadrant_experiment_protocol.md` for current evidence scope and the evaluation protocol.

The historical Review 1–3 fixture milestones below document prior development and are not evidence that the current DistilBERT comparison has been trained or validated.

## Completed

- Three-review MAPPO-Sec architecture documented.
- Member 2 Defender master plan documented.
- **Review 1 complete (Parts 1–3, 2026-08-26):** Gate GATE-001–009, SFT corpus 20/8/10, scripts, 5 sanitized traces, 43 tests.
- **Part 4 Review 2 integration contracts (2026-08-26):** model_adapter, rollout_adapter, rewards. 104 tests.
- **Part 5 MAPPO update + capability retention (2026-08-27):** mappo_defender, capability_retention. 162 tests.
- **Part 6 Fixture episode harness (2026-08-27):** episode_harness, run_review2_episode. 208 tests.
- **Part 7 Review 2 docs (2026-08-27):** review2_integration_summary, defender-review2-evidence. 208 tests.
- **Part 8 Checkpoint evaluator (2026-08-27):** evaluate_checkpoint, heldout guard. 242 tests.
- **Part 9 Ablation configs + retention analysis (2026-08-28):** 3 ablation JSONs, retention script, JSON+CSV reports. 284 tests.
- **Part 10 Review 3 final artefacts (2026-08-28):**
  - `reports/review3_readiness_summary.json`: machine-readable readiness record — all 12 deliverables (READY or BLOCKED_EXTERNAL), fixture baseline metrics (in-dist + OOD + retention), full ablation plan with hypotheses, interface readiness (Member 1/3/4 PENDING), 7-step run instructions for when training is unblocked.
  - `docs/defender-review3-evidence.md`: full Review 3 presentation — 5-slice evaluation matrix, metric comparison tables (SFT/MAPPO BLOCKED_EXTERNAL), per-family ASR, capability retention table, 8-probe per-category rates, ablation plan with no-gate NOT_RUN rationale, end-to-end run instructions, interface readiness, 284-test coverage table, 9 generated reports indexed, 6 open blockers.
  - All 284 tests still pass.
- **Review 2 MAPPO Co-evolution Training Pipeline (2026-09-18):**
  - Resolved 10 runtime and interface bugs in `scripts/run_mappo_coevolution.py`.
  - Integrated `CentralizedCritic`, `AttackerPolicy`, `CoevolutionEnv`, and `CoevolutionWandbLogger`.
  - Added periodic and final checkpointing into `checkpoints/mappo_run`.
  - Integrated `ATTACK_CATEGORIES` and `ATTACK_TEMPLATES` from `src/attacker_policy/attack_generator.py`.
  - Added `log_metrics()` method to `CoevolutionWandbLogger`.
  - 430/430 tests passing. Full 1-episode smoke test verified successfully.

## In progress

Nothing in progress.

## Next exact action

After external training produces the required SFT and MAPPO classifier checkpoints, run the four-quadrant evaluator on a newly frozen benchmark and preserve checkpoint/data hashes. Do not use previously inspected cases as confirmatory evidence.

## Latest verification

- Historical repository record: `PYTHONPATH=. python3 -m pytest` reported 430 tests passed. This was not rerun during the current paper/evaluation revisions.
- Historical MAPPO smoke test and dry-run checks are recorded below; they do not establish validity of the current comparative experiment.
- Scripts: the following historical checks were recorded:
  - `prepare_sft.py --dry-run`
  - `run_review1_baseline.py --dry-run`
  - `run_review2_episode.py --dry-run`
  - `evaluate_checkpoint.py --dry-run`
  - `run_capability_retention_analysis.py --dry-run`
  - `evaluate_checkpoint.py --checkpoint fixture_baseline --allow-heldout` (full run)

## Decisions locked

- Base model: PENDING. SFT training BLOCKED_EXTERNAL.
- Dataset version: `review1-fixture-v2` / `sft-v2`.
- Heldout splits FROZEN — `allow_heldout=True` required.
- Gate: GATE-001–009; fixed. Cannot be disabled for ablations.
- Action schema version: `gate-v1`.
- Reward weights: lambda_violation=1.0, lambda_refusal=0.3, lambda_cost=0.0.
- MAPPO defaults: clip_epsilon=0.2, beta_kl=0.1, value_loss_coeff=0.5, entropy_coeff=0.01, gamma=0.99, gae_lambda=0.95.
- Ablation betas: no_kl=0.0, default=0.1, high_kl=1.0.

## Blockers / questions for team

- Base LLM/checkpoint, training stack, LoRA/QLoRA/full-SFT, GPU budget → SFT BLOCKED_EXTERNAL.
- Member 1 AttackPayload schema → MAPPO self-play blocked.
- Member 3 Trajectory/Critic schema → trajectory assembly blocked.
- Member 4 EvaluationEvent/OOD suite → Review 3 eval blocked.
