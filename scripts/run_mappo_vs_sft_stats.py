"""Paired case-level descriptive comparison of trained SFT and MAPPO classifiers.

Intervals describe variation across cases conditional on the supplied checkpoints.
They are not estimates of variation across training runs and are not hypothesis tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "evaluation" / "four_quadrants_benchmark.jsonl"
MODEL_NAME = "distilbert-base-uncased"
UNSAFE_LABELS = {"injection_resistance", "unsafe_request_refusal"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path):
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label_name = record.get("label", "benign_answer")
        if label_name not in {"benign_answer", "benign_tool_use", "injection_resistance", "unsafe_request_refusal"}:
            raise ValueError(f"Unsupported label {label_name!r} in {path}")
        cases.append({
            "id": record.get("id"),
            "quadrant": record.get("quadrant", "unassigned"),
            "label": int(label_name in UNSAFE_LABELS),
            "label_name": label_name,
            "text": f"[TASK] {record.get('user_task', '')} [CONTENT] {record.get('untrusted_content', '')}",
        })
    return cases


def predict(checkpoint: Path, texts, tokenizer, device):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 16):
            encoded = tokenizer(texts[offset:offset + 16], max_length=128,
                                padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            predictions.extend(model(**encoded).logits.argmax(dim=-1).cpu().tolist())
    return predictions


def paired_interval(cases, pred_m, pred_s, metric, seed, draws):
    indices = np.arange(len(cases))
    if not len(indices):
        return {"n_cases": 0, "difference_mappo_minus_sft": None, "case_bootstrap_95pct_ci": None}
    observed = metric(indices, cases, pred_m, pred_s)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for draw in range(draws):
        selected = rng.choice(indices, size=len(indices), replace=True)
        samples[draw] = metric(selected, cases, pred_m, pred_s)
    return {
        "n_cases": int(len(indices)),
        "difference_mappo_minus_sft": round(float(observed), 6),
        "case_bootstrap_95pct_ci": [round(float(v), 6) for v in np.quantile(samples, [0.025, 0.975])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--sft-ckpt", type=Path, default=PROJECT_ROOT / "checkpoints/sft_run/sft_final.pt")
    parser.add_argument("--mappo-ckpt", type=Path, default=PROJECT_ROOT / "checkpoints/mappo_run/mappo_defender_final.pt")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for path in (args.benchmark, args.sft_ckpt, args.mappo_ckpt):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation input is missing: {path}")
    if args.draws < 100:
        parser.error("--draws must be at least 100")

    cases = load_cases(args.benchmark)
    if not cases:
        raise ValueError(f"Benchmark contains no cases: {args.benchmark}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = [case["text"] for case in cases]
    pred_s = predict(args.sft_ckpt, texts, tokenizer, args.device)
    pred_m = predict(args.mappo_ckpt, texts, tokenizer, args.device)

    def accuracy(ix, rows, pm, ps):
        return float(np.mean([int(pm[i] == rows[i]["label"]) - int(ps[i] == rows[i]["label"]) for i in ix]))

    def benign_allow(ix, rows, pm, ps):
        ix = [i for i in ix if rows[i]["label"] == 0]
        return float(np.mean([int(pm[i] == 0) - int(ps[i] == 0) for i in ix])) if ix else 0.0

    def attack_defend(ix, rows, pm, ps):
        ix = [i for i in ix if rows[i]["label"] == 1]
        return float(np.mean([int(pm[i] == 1) - int(ps[i] == 1) for i in ix])) if ix else 0.0

    grouped = {}
    for quadrant in sorted({case["quadrant"] for case in cases}):
        ix = [i for i, case in enumerate(cases) if case["quadrant"] == quadrant]
        subset = [cases[i] for i in ix]
        subset_m = [pred_m[i] for i in ix]
        subset_s = [pred_s[i] for i in ix]
        row = paired_interval(subset, subset_m, subset_s, accuracy, args.seed, args.draws)
        row.update({
            "quadrant": quadrant,
            "mappo_allow_rate": round(float(np.mean([pred_m[i] == 0 for i in ix])), 6),
            "sft_allow_rate": round(float(np.mean([pred_s[i] == 0 for i in ix])), 6),
            "paired_allow_rate_difference": paired_interval(subset, subset_m, subset_s, benign_allow, args.seed, args.draws),
            "paired_defend_rate_difference": paired_interval(subset, subset_m, subset_s, attack_defend, args.seed, args.draws),
        })
        grouped[quadrant] = row

    report = {
        "analysis": "paired case bootstrap; descriptive and conditional on fixed checkpoints",
        "not_estimated": "between-training-run variability; no significance test is performed",
        "benchmark_sha256": sha256_file(args.benchmark),
        "sft_checkpoint_sha256": sha256_file(args.sft_ckpt),
        "mappo_checkpoint_sha256": sha256_file(args.mappo_ckpt),
        "seed": args.seed,
        "bootstrap_draws": args.draws,
        "total_cases": len(cases),
        "overall_accuracy_difference": paired_interval(cases, pred_m, pred_s, accuracy, args.seed, args.draws),
        "quadrants": grouped,
        "case_predictions": [
            {"id": c["id"], "quadrant": c["quadrant"], "label": c["label_name"],
             "sft_action": int(pred_s[i]), "mappo_action": int(pred_m[i])}
            for i, c in enumerate(cases)
        ],
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    output = args.reports_dir / "mappo_vs_sft_paired_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote paired case-level descriptive report: {output}")


if __name__ == "__main__":
    main()
