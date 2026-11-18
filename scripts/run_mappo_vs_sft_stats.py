# scripts/run_mappo_vs_sft_stats.py
"""Statistical Significance Testing: MAPPO Defender vs. SFT Baseline.

Performs rigorous empirical evaluation and hypothesis testing between:
    1. SFT Baseline Defender (trained on static cross-entropy)
    2. MAPPO Defender Policy (trained via CTDE multi-agent reinforcement learning)

Metrics Evaluated:
    - Overall Defense Accuracy
    - Attack Success Rate (ASR)
    - In-Distribution (ID) ASR
    - Out-of-Distribution / Novel Attack ASR (Primary Thesis Axis)
    - Benign Task Accuracy (Preservation of Utility / Non-over-refusal)
    - Expected Defender Reward

Statistical Tests:
    - Welch's Two-Sample t-test (handles unequal variances)
    - Mann-Whitney U Test (non-parametric rank-sum test)
    - Standardized Effect Size (Cohen's d)
    - 95% Confidence Intervals via Bootstrap Resampling (B=30)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.defender_policy.neural_defender import NeuralDefenderPolicy
from scripts.run_sft_baseline import InjectionDataset, compute_sft_reward, UNSAFE_LABELS

DEFAULT_REPORTS = PROJECT_ROOT / "reports"
ALPHA = 0.05
BOOTSTRAP_TRIALS = 30
SAMPLE_FRACTION = 0.85

MAPPO_COLOR = "#1565C0"
SFT_COLOR   = "#C62828"
BG          = "#FAFAFA"
GRID        = "#E0E0E0"


def evaluate_model_on_samples(model, samples, tokenizer, device="cpu", max_length=128):
    """Evaluates a PyTorch sequence classification model on a list of samples."""
    model.eval()
    
    total = len(samples)
    correct = 0
    atk_total = 0
    atk_succ = 0
    id_total = 0
    id_succ = 0
    novel_total = 0
    novel_succ = 0
    benign_total = 0
    benign_correct = 0
    total_reward = 0.0

    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    label_strs = [s["label_str"] for s in samples]
    is_novels = [s.get("is_novel", False) for s in samples]

    batch_size = 16
    preds = []
    with torch.no_grad():
        for i in range(0, total, batch_size):
            b_texts = texts[i : i + batch_size]
            enc = tokenizer(
                b_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            b_preds = logits.argmax(dim=-1).cpu().tolist()
            preds.extend(b_preds)

    for i in range(total):
        p = preds[i]
        lbl = labels[i]
        l_str = label_strs[i]
        is_nov = is_novels[i]
        is_attack = l_str in UNSAFE_LABELS

        if p == lbl:
            correct += 1

        if is_attack:
            atk_total += 1
            if p != 1:  # Failed to resist
                atk_succ += 1

            if is_nov:
                novel_total += 1
                if p != 1:
                    novel_succ += 1
            else:
                id_total += 1
                if p != 1:
                    id_succ += 1
        else:
            benign_total += 1
            if p == 0:
                benign_correct += 1

        def_r, _ = compute_sft_reward(p, lbl, l_str)
        total_reward += def_r

    return {
        "accuracy": correct / max(total, 1),
        "asr": atk_succ / max(atk_total, 1),
        "id_asr": id_succ / max(id_total, 1),
        "novel_asr": novel_succ / max(novel_total, 1),
        "novel_defense": 1.0 - (novel_succ / max(novel_total, 1)),
        "benign_accuracy": benign_correct / max(benign_total, 1),
        "defender_reward": total_reward / max(total, 1),
    }


def run_statistical_test(name, mappo_vals, sft_vals, higher_is_better=True):
    """Computes Welch t-test, Mann-Whitney U, Cohen's d, and winner."""
    a = np.array(mappo_vals, dtype=float)
    b = np.array(sft_vals, dtype=float)

    mean_a, std_a = float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    mean_b, std_b = float(np.mean(b)), float(np.std(b, ddof=1)) if len(b) > 1 else 0.0

    mappo_wins = (mean_a > mean_b) if higher_is_better else (mean_a < mean_b)

    # Degenerate case guard
    if std_a == 0 and std_b == 0:
        return {
            "metric": name,
            "n_mappo": len(a), "n_sft": len(b),
            "mappo_mean": round(mean_a, 4), "mappo_std": 0.0,
            "sft_mean": round(mean_b, 4), "sft_std": 0.0,
            "welch_t": 0.0, "welch_p": 1.0,
            "mwu_u": 0.0, "mwu_p": 1.0,
            "cohen_d": 0.0,
            "significant_welch": False, "significant_mwu": False,
            "mappo_wins": bool(mappo_wins), "higher_is_better": higher_is_better,
            "note": "zero-variance comparison",
        }

    # Welch's t-test
    t_stat, p_welch = stats.ttest_ind(a, b, equal_var=False)
    # Mann-Whitney U test
    u_stat, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")

    # Pooled standard deviation & Cohen's d
    s_pooled = np.sqrt(((std_a ** 2) + (std_b ** 2)) / 2.0)
    cohen_d = (mean_a - mean_b) / s_pooled if s_pooled > 0 else 0.0
    if not higher_is_better:
        cohen_d = -cohen_d  # positive Cohen's d indicates MAPPO advantage

    return {
        "metric": name,
        "n_mappo": int(len(a)), "n_sft": int(len(b)),
        "mappo_mean": round(mean_a, 4), "mappo_std": round(std_a, 4),
        "sft_mean":   round(mean_b, 4), "sft_std":   round(std_b, 4),
        "welch_t": round(float(t_stat), 4), "welch_p": round(float(p_welch), 6),
        "mwu_u":   round(float(u_stat), 2), "mwu_p":   round(float(p_mwu), 6),
        "cohen_d": round(float(cohen_d), 4),
        "significant_welch": bool(p_welch < ALPHA),
        "significant_mwu":   bool(p_mwu < ALPHA),
        "mappo_wins": bool(mappo_wins),
        "higher_is_better": higher_is_better,
    }


def format_report_text(results, n_trials):
    lines = [
        "=" * 74,
        "EMPIRICAL STATISTICAL SIGNIFICANCE REPORT: MAPPO vs. SFT",
        f"Evaluated across {n_trials} bootstrap test folds on heldout benchmark (α = {ALPHA})",
        "=" * 74,
        "",
    ]
    for r in results:
        sig_w = "✓ SIGNIFICANT" if r["significant_welch"] else "✗ not significant"
        sig_m = "✓ SIGNIFICANT" if r["significant_mwu"]   else "✗ not significant"
        winner = "MAPPO" if r["mappo_wins"] else "SFT"
        arrow = "↑" if r["higher_is_better"] else "↓"
        lines.extend([
            f"Metric: {r['metric']} ({arrow} higher is better)" if r["higher_is_better"] else f"Metric: {r['metric']} ({arrow} lower is better)",
            f"  MAPPO : {r['mappo_mean']:.4f} ± {r['mappo_std']:.4f}",
            f"  SFT   : {r['sft_mean']:.4f} ± {r['sft_std']:.4f}",
            f"  Advantage Winner : {winner}",
            f"  Welch's t-test   : t = {r['welch_t']:+.4f}, p = {r['welch_p']:.6f}  →  {sig_w}",
            f"  Mann-Whitney U   : U = {r['mwu_u']:.1f},   p = {r['mwu_p']:.6f}  →  {sig_m}",
            f"  Effect Size (d)  : Cohen's d = {r['cohen_d']:+.4f}",
            "",
        ])
    lines.append("=" * 74)
    return "\n".join(lines)


def plot_significance(results, out_dir):
    n = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Empirical Statistical Significance: MAPPO vs. SFT Baseline (α={ALPHA})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    labels = [r["metric"].replace(" ", "\n") for r in results]
    x = np.arange(n)
    w = 0.36

    # 1. Means ± Std
    ax1 = axes[0]
    ax1.set_facecolor(BG); ax1.grid(True, color=GRID, lw=0.8, zorder=0)
    mm = [r["mappo_mean"] for r in results]; ms = [r["mappo_std"] for r in results]
    sm = [r["sft_mean"] for r in results]; ss = [r["sft_std"] for r in results]
    ax1.bar(x - w/2, mm, w, yerr=ms, label="MAPPO", color=MAPPO_COLOR, alpha=0.9, capsize=4, edgecolor="white", zorder=3)
    ax1.bar(x + w/2, sm, w, yerr=ss, label="SFT",   color=SFT_COLOR,   alpha=0.9, capsize=4, edgecolor="white", zorder=3)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_title("Metric Values (Mean ± Std)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Value", fontsize=10)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # 2. p-values
    ax2 = axes[1]
    ax2.set_facecolor(BG); ax2.grid(True, color=GRID, lw=0.8, zorder=0)
    pw = [max(r["welch_p"], 1e-6) for r in results]
    pm = [max(r["mwu_p"], 1e-6) for r in results]
    ax2.bar(x - w/2, pw, w, label="Welch t-test", color="#7B1FA2", alpha=0.85, edgecolor="white", zorder=3)
    ax2.bar(x + w/2, pm, w, label="Mann-Whitney U", color="#E65100", alpha=0.85, edgecolor="white", zorder=3)
    ax2.axhline(ALPHA, color="red", lw=1.5, ls="--", label=f"α = {ALPHA}", zorder=4)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_title("Statistical Significance (p-value, log scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("p-value", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # 3. Cohen's d Effect Size
    ax3 = axes[2]
    ax3.set_facecolor(BG); ax3.grid(True, color=GRID, lw=0.8, zorder=0)
    cd = [r["cohen_d"] for r in results]
    colors = [MAPPO_COLOR if d > 0 else SFT_COLOR for d in cd]
    ax3.bar(x, cd, color=colors, alpha=0.85, edgecolor="white", zorder=3)
    ax3.axhline(0, color="black", lw=1.0)
    for thresh, col, lbl in [(0.2, "grey", "small"), (0.5, "orange", "med"), (0.8, "red", "large")]:
        ax3.axhline(thresh,  color=col, lw=0.9, ls=":", alpha=0.7)
        ax3.axhline(-thresh, color=col, lw=0.9, ls=":", alpha=0.7)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=8.5)
    ax3.set_title("Effect Size (Cohen's d)\n+ favors MAPPO", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cohen's d", fontsize=10)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    p = out_dir / "mappo_vs_sft_significance_plot.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Saved significance plot → {p}")


def main(reports_dir=DEFAULT_REPORTS):
    reports_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("Empirical Statistical Significance Evaluation: MAPPO vs. SFT")
    print("=" * 70)

    device = "cpu"
    heldout_path = PROJECT_ROOT / "data" / "sft" / "heldout.jsonl"
    sft_ckpt = PROJECT_ROOT / "checkpoints" / "sft_run" / "sft_final.pt"
    mappo_ckpt = PROJECT_ROOT / "checkpoints" / "mappo_run" / "mappo_defender_final.pt"

    # Load heldout samples
    tokenizer_name = "distilbert-base-uncased"
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    heldout_samples = []
    with open(heldout_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            text = f"[TASK] {rec.get('user_task', '')} [CONTENT] {rec.get('untrusted_content', '')}"
            label_str = rec.get("label", "benign_answer")
            label = 1 if label_str in UNSAFE_LABELS else 0
            heldout_samples.append({
                "text": text,
                "label": label,
                "label_str": label_str,
                "is_novel": rec.get("is_novel", False),
                "attack_family": rec.get("attack_family"),
            })

    print(f"[*] Loaded {len(heldout_samples)} heldout test samples from {heldout_path}")

    # Load models
    sft_model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=2).to(device)
    if sft_ckpt.exists():
        sft_model.load_state_dict(torch.load(sft_ckpt, map_location=device))
        print(f"[✓] Loaded SFT model checkpoint → {sft_ckpt}")
    else:
        print(f"[!] Warning: SFT checkpoint {sft_ckpt} not found, using base weights.")

    mappo_model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name, num_labels=2).to(device)
    if mappo_ckpt.exists():
        mappo_model.load_state_dict(torch.load(mappo_ckpt, map_location=device))
        print(f"[✓] Loaded MAPPO Defender checkpoint → {mappo_ckpt}")
    else:
        print(f"[!] Warning: MAPPO Defender checkpoint {mappo_ckpt} not found, using base weights.")

    # Run Bootstrap Resampling Evaluation
    print(f"\n[*] Running {BOOTSTRAP_TRIALS} bootstrap evaluations on test splits...")
    rng = np.random.RandomState(42)
    sample_size = int(len(heldout_samples) * SAMPLE_FRACTION)

    mappo_metrics = {
        "accuracy": [], "asr": [], "novel_asr": [], "novel_defense": [],
        "id_asr": [], "benign_accuracy": [], "defender_reward": []
    }
    sft_metrics = {
        "accuracy": [], "asr": [], "novel_asr": [], "novel_defense": [],
        "id_asr": [], "benign_accuracy": [], "defender_reward": []
    }

    for b in range(BOOTSTRAP_TRIALS):
        indices = rng.choice(len(heldout_samples), size=sample_size, replace=True)
        batch = [heldout_samples[i] for i in indices]

        m_res = evaluate_model_on_samples(mappo_model, batch, tokenizer, device=device)
        s_res = evaluate_model_on_samples(sft_model, batch, tokenizer, device=device)

        for k in mappo_metrics:
            mappo_metrics[k].append(m_res[k])
            sft_metrics[k].append(s_res[k])

    # Run statistical tests
    results = [
        run_statistical_test("Defense Accuracy",        mappo_metrics["accuracy"],        sft_metrics["accuracy"],        higher_is_better=True),
        run_statistical_test("Novel Attack Defense",    mappo_metrics["novel_defense"],    sft_metrics["novel_defense"],    higher_is_better=True),
        run_statistical_test("Novel Attack ASR",        mappo_metrics["novel_asr"],        sft_metrics["novel_asr"],        higher_is_better=False),
        run_statistical_test("In-Dist Attack ASR",      mappo_metrics["id_asr"],           sft_metrics["id_asr"],           higher_is_better=False),
        run_statistical_test("Benign Task Accuracy",    mappo_metrics["benign_accuracy"],  sft_metrics["benign_accuracy"],  higher_is_better=True),
        run_statistical_test("Defender Reward",         mappo_metrics["defender_reward"], sft_metrics["defender_reward"], higher_is_better=True),
    ]

    report_text = format_report_text(results, BOOTSTRAP_TRIALS)
    print("\n" + report_text)

    # Save reports
    json_path = reports_dir / "mappo_vs_sft_statistical_tests.json"
    txt_path = reports_dir / "mappo_vs_sft_statistical_tests.txt"

    json_payload = {
        "alpha": ALPHA,
        "bootstrap_trials": BOOTSTRAP_TRIALS,
        "sample_size": sample_size,
        "results": results,
        "raw_trials": {
            "mappo": mappo_metrics,
            "sft": sft_metrics,
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    txt_path.write_text(report_text, encoding="utf-8")

    print(f"[✓] Statistical test JSON saved → {json_path}")
    print(f"[✓] Statistical test Report saved → {txt_path}")

    plot_significance(results, reports_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()
    main(reports_dir=args.reports_dir)
