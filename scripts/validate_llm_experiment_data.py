"""Check data split and family constraints before confirmatory model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_metrics.experiment_data import (  # noqa: E402
    ATTACK_CATEGORY_FAMILIES,
    DEFAULT_ATTACK_CATEGORIES,
    load_jsonl,
    validate_confirmatory_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--attack-categories", nargs="+", choices=sorted(ATTACK_CATEGORY_FAMILIES),
        default=list(DEFAULT_ATTACK_CATEGORIES),
        help="Must match --attack-categories passed to scripts/run_llm_mappo.py",
    )
    args = parser.parse_args()
    try:
        result = validate_confirmatory_data(
            load_jsonl(args.train_file),
            load_jsonl(args.validation_file),
            load_jsonl(args.benchmark),
            args.attack_categories,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Data preflight failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
