"""Plot observed quadrant allow rates from a completed paired case report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "reports" / "four_quadrants_evaluation.json"
OUTPUT = ROOT / "reports" / "mappo_vs_sft_quadrant_comparison.png"


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Run scripts/run_mappo_vs_sft_stats.py first: {INPUT}")
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    metadata = report.get("run_metadata", {})
    quadrants = report.get("quadrant_summaries", {})
    if not quadrants or not metadata.get("sft_checkpoint_sha256") or not metadata.get("mappo_checkpoint_sha256"):
        raise ValueError("Input is not a completed four-quadrant evaluation with checkpoint provenance")

    names = list(quadrants)
    sft = [quadrants[name]["sft"]["allowed_pct"] / 100 for name in names]
    mappo = [quadrants[name]["mappo"]["allowed_pct"] / 100 for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, sft, width, label="SFT", color="#C62828")
    ax.bar(x + width / 2, mappo, width, label="MAPPO", color="#1565C0")
    ax.set_ylabel("Observed allow rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, [name.replace("_", "\n") for name in names])
    ax.set_title("Allow rates by quadrant (conditional on these checkpoints and cases)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
