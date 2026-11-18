"""Aggregate paired four-quadrant evaluations across independent training seeds."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


QUADRANTS = (
    "Q1_common_benign",
    "Q2_common_injection",
    "Q3_unique_benign",
    "Q4_unique_injection",
)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_metrics.hypothesis_checks import evaluate_expected_pattern


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "benchmark_sha256", "seed", "samples_per_case", "max_new_tokens",
        "generation", "base_model",
        "base_model_revision", "benchmark_status", "sft_training_seed",
        "mappo_training_seed", "sft_adapter_weights_sha256",
        "mappo_adapter_weights_sha256", "train_file_sha256", "validation_file_sha256",
        "minimum_action_rate", "hypothesis_pattern", "per_case", "quadrants",
    }
    missing = required.difference(report)
    if missing:
        raise ValueError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return report


def bootstrap_interval(values: list[float], rng: random.Random,
                       bootstrap_samples: int) -> list[float | None]:
    if len(values) < 2:
        return [None, None]
    replicates = sorted(
        mean([values[rng.randrange(len(values))] for _ in values])
        for _ in range(bootstrap_samples)
    )
    return [replicates[round(0.025 * (bootstrap_samples - 1))],
            replicates[round(0.975 * (bootstrap_samples - 1))]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path,
                        help="One evaluation JSON per independent paired training seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require-confirmatory", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    if len(args.reports) < 2:
        parser.error("At least two independent paired training runs are required")
    if args.output.resolve() in {path.resolve() for path in args.reports}:
        parser.error("Aggregate output must not overwrite an input evaluation report")

    reports = [load_report(path) for path in args.reports]
    reference = reports[0]
    for report, path in zip(reports[1:], args.reports[1:]):
        for field in ("benchmark_sha256", "train_file_sha256", "validation_file_sha256",
                      "seed", "samples_per_case", "max_new_tokens", "generation",
                      "base_model", "base_model_revision", "benchmark_status",
                      "minimum_action_rate"):
            if report.get(field) != reference.get(field):
                raise ValueError(f"{path} does not match the first report's {field}")
    if args.require_confirmatory:
        if any(report.get("benchmark_status") != "confirmatory" for report in reports):
            raise ValueError("Every input report must be marked confirmatory")
        if any(not report.get("canary_evaluation", {}).get("configured") for report in reports):
            raise ValueError("Every confirmatory report must include runtime canary evaluation")
    for report, path in zip(reports, args.reports):
        expected_pattern = evaluate_expected_pattern(
            report["quadrants"], report["minimum_action_rate"]
        )
        if report.get("hypothesis_pattern") != expected_pattern:
            raise ValueError(f"{path} has an inconsistent four-quadrant hypothesis summary")

    pairs = [(report["sft_training_seed"], report["mappo_training_seed"])
             for report in reports]
    if any(sft_seed != mappo_seed for sft_seed, mappo_seed in pairs):
        raise ValueError("Each report must contain a same-seed SFT/MAPPO pair")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Training seed pairs must be unique across reports")
    checkpoint_pairs = [(report["sft_adapter_weights_sha256"],
                         report["mappo_adapter_weights_sha256"]) for report in reports]
    if len(set(checkpoint_pairs)) != len(checkpoint_pairs):
        raise ValueError("Each report must represent a distinct checkpoint pair")

    first = reports[0]["per_case"]
    if any(not isinstance(first.get(arm), list) for arm in ("sft", "mappo")):
        raise ValueError("Evaluation reports must contain per-case rows for both arms")
    reference_rows = first
    reference_ids = [row.get("id") for row in reference_rows["sft"]]
    if reference_ids != [row.get("id") for row in reference_rows["mappo"]]:
        raise ValueError("SFT and MAPPO case IDs differ in the first report")
    quadrants = [row.get("quadrant") for row in reference_rows["sft"]]
    if any(not quadrant for quadrant in quadrants):
        raise ValueError("Every evaluated case must have a quadrant")
    expected_actions = [row.get("expected_action") for row in reference_rows["sft"]]
    sample_counts = [row.get("samples") for row in reference_rows["sft"]]
    if expected_actions != [row.get("expected_action") for row in reference_rows["mappo"]]:
        raise ValueError("SFT and MAPPO expected actions differ in the first report")
    if sample_counts != [row.get("samples") for row in reference_rows["mappo"]]:
        raise ValueError("SFT and MAPPO sample counts differ in the first report")
    for report, path in zip(reports, args.reports):
        for arm in ("sft", "mappo"):
            rows = report["per_case"].get(arm)
            if not isinstance(rows, list) or [row.get("id") for row in rows] != reference_ids:
                raise ValueError(f"{path} has different {arm} case IDs or ordering")
            if [row.get("quadrant") for row in rows] != quadrants:
                raise ValueError(f"{path} has different {arm} quadrant assignments")
            if [row.get("expected_action") for row in rows] != expected_actions:
                raise ValueError(f"{path} has different {arm} expected actions")
            if [row.get("samples") for row in rows] != sample_counts:
                raise ValueError(f"{path} has different {arm} per-case sample counts")
        if set(report.get("quadrants", {})) != set(QUADRANTS):
            raise ValueError(f"{path} does not report all four quadrants")

    rng = random.Random(args.seed)
    result = {}
    for quadrant in QUADRANTS:
        indexes = [index for index, name in enumerate(quadrants) if name == quadrant]
        if not indexes:
            raise ValueError(f"No cases found for {quadrant}")
        per_seed = []
        for report in reports:
            arm_cases = report["per_case"]
            accuracies = {}
            for arm in ("sft", "mappo"):
                selected = [arm_cases[arm][index] for index in indexes]
                if any(row.get("samples", 0) < 1 for row in selected):
                    raise ValueError("Per-case sample counts must be positive")
                accuracies[arm] = mean([
                    row["correct_count"] / row["samples"] for row in selected
                ])
            per_seed.append({
                "training_seed": report["sft_training_seed"],
                "sft_accuracy": accuracies["sft"],
                "mappo_accuracy": accuracies["mappo"],
                "accuracy_delta_mappo_minus_sft": accuracies["mappo"] - accuracies["sft"],
                "sft_canary_exposure_rate": report["quadrants"][quadrant]["sft"]["canary_exposure_rate"],
                "mappo_canary_exposure_rate": report["quadrants"][quadrant]["mappo"]["canary_exposure_rate"],
                "expected_quadrant_pattern_met": report["hypothesis_pattern"]["quadrants"][quadrant]["met"],
            })
        deltas = [row["accuracy_delta_mappo_minus_sft"] for row in per_seed]
        sft_leaks = [row["sft_canary_exposure_rate"] for row in per_seed
                     if row["sft_canary_exposure_rate"] is not None]
        mappo_leaks = [row["mappo_canary_exposure_rate"] for row in per_seed
                       if row["mappo_canary_exposure_rate"] is not None]
        leak_deltas = [row["mappo_canary_exposure_rate"] - row["sft_canary_exposure_rate"]
                       for row in per_seed
                       if row["mappo_canary_exposure_rate"] is not None
                       and row["sft_canary_exposure_rate"] is not None]
        for row in per_seed:
            sft_leak = row["sft_canary_exposure_rate"]
            mappo_leak = row["mappo_canary_exposure_rate"]
            row["canary_exposure_delta_mappo_minus_sft"] = (
                mappo_leak - sft_leak if mappo_leak is not None and sft_leak is not None else None
            )
        result[quadrant] = {
            "cases": len(indexes),
            "independent_training_pairs": len(reports),
            "training_pairs_meeting_expected_quadrant_pattern": sum(
                row["expected_quadrant_pattern_met"] for row in per_seed
            ),
            "mean_sft_accuracy": mean([row["sft_accuracy"] for row in per_seed]),
            "mean_mappo_accuracy": mean([row["mappo_accuracy"] for row in per_seed]),
            "mean_accuracy_delta_mappo_minus_sft": mean(deltas),
            "training_pair_bootstrap_95pct_ci_for_accuracy_delta": bootstrap_interval(
                deltas, rng, args.bootstrap_samples
            ),
            "mean_sft_canary_exposure_rate": mean(sft_leaks) if sft_leaks else None,
            "mean_mappo_canary_exposure_rate": mean(mappo_leaks) if mappo_leaks else None,
            "mean_canary_exposure_delta_mappo_minus_sft": mean(leak_deltas) if leak_deltas else None,
            "training_pair_bootstrap_95pct_ci_for_canary_delta": bootstrap_interval(
                leak_deltas, rng, args.bootstrap_samples
            ) if len(leak_deltas) == len(reports) else [None, None],
            "per_training_pair": per_seed,
        }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_sha256": reference["benchmark_sha256"],
        "evaluation_seed": reference["seed"],
        "samples_per_case": reference["samples_per_case"],
        "minimum_action_rate": reference["minimum_action_rate"],
        "max_new_tokens": reference["max_new_tokens"],
        "generation": reference["generation"],
        "independent_training_pairs": len(reports),
        "training_seed_pairs": pairs,
        "full_expected_pattern": {
            "independent_training_pairs_meeting_all_four_quadrants": sum(
                report["hypothesis_pattern"]["all_quadrants_met"] for report in reports
            ),
            "independent_training_pairs": len(reports),
            "per_training_pair": [
                {
                    "training_seed": report["sft_training_seed"],
                    "all_quadrants_met": report["hypothesis_pattern"]["all_quadrants_met"],
                    "quadrants": report["hypothesis_pattern"]["quadrants"],
                }
                for report in reports
            ],
        },
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "confidence_interval_unit": "independent paired training run",
        "interpretation": (
            "Intervals summarize run-to-run variation for these runs; they do not establish "
            "generalization beyond the benchmark and training setup."
        ),
        "quadrants": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "independent_training_pairs": len(reports),
                      "full_expected_pattern": output["full_expected_pattern"],
                      "quadrants": result}, indent=2))


if __name__ == "__main__":
    main()
