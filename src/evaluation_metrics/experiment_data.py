"""Preflight checks for disjoint four-quadrant experiment data."""

from __future__ import annotations

import json
from pathlib import Path

from src.attacker_policy.attack_families import (
    ATTACK_CATEGORY_FAMILIES,
    DEFAULT_ATTACK_CATEGORIES,
)
EXPECTED_LABELS = {
    "Q1_common_benign": "ALLOW",
    "Q2_common_injection": "DEFEND",
    "Q3_unique_benign": "ALLOW",
    "Q4_unique_injection": "DEFEND",
}
LABEL_ACTION = {
    "benign_answer": "ALLOW",
    "benign_tool_use": "ALLOW",
    "injection_resistance": "DEFEND",
    "unsafe_request_refusal": "DEFEND",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each record must be a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _identity_sets(rows: list[dict], split_name: str) -> dict[str, set]:
    if not rows:
        raise ValueError(f"{split_name} split is empty")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"Every {split_name} record requires a non-empty id")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate id within {split_name}")
    for row in rows:
        scenario = row.get("scenario_id")
        if scenario is not None and (not isinstance(scenario, str) or not scenario.strip()):
            raise ValueError(f"{split_name} scenario_id must be a non-empty string when supplied")
        for field in ("task_family", "attack_family"):
            value = row.get(field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{split_name} {field} must be a non-empty string when supplied")
        target = row.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"{split_name} record {row.get('id')} needs a non-empty target")
    scenarios = [row.get("scenario_id") for row in rows if row.get("scenario_id")]
    if len(set(scenarios)) != len(scenarios):
        raise ValueError(f"Duplicate scenario_id within {split_name}")
    inputs = []
    for row in rows:
        task = row.get("user_task", "")
        content = row.get("untrusted_content", "")
        if not isinstance(task, str) or not isinstance(content, str):
            raise ValueError(f"{split_name} user_task and untrusted_content must be strings")
        inputs.append((task.strip(), content.strip()))
    if len(set(inputs)) != len(inputs):
        raise ValueError(f"Duplicate task/content pair within {split_name}")
    return {
        "ids": set(ids),
        "scenarios": set(scenarios),
        "inputs": set(inputs),
    }


def _check_disjoint(splits: dict[str, list[dict]]) -> None:
    identities = {name: _identity_sets(rows, name) for name, rows in splits.items()}
    names = list(splits)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            for key in ("ids", "scenarios", "inputs"):
                overlap = identities[left_name][key].intersection(identities[right_name][key])
                if overlap:
                    raise ValueError(
                        f"{left_name}/{right_name} overlap on {key}: {next(iter(overlap))}"
                    )


def validate_confirmatory_data(train_rows: list[dict], validation_rows: list[dict],
                               benchmark_rows: list[dict],
                               attack_categories: list[str] | tuple[str, ...] = DEFAULT_ATTACK_CATEGORIES
                               ) -> dict:
    """Validate all data/family conditions needed before confirmatory training."""
    splits = {"train": train_rows, "validation": validation_rows, "benchmark": benchmark_rows}
    _check_disjoint(splits)
    for row in train_rows + validation_rows:
        if row.get("label") not in LABEL_ACTION:
            raise ValueError(f"Unsupported training label in record {row.get('id')}")
        if LABEL_ACTION[row["label"]] == "DEFEND" and not row.get("attack_family"):
            raise ValueError(f"Unsafe training record {row.get('id')} needs an attack_family annotation")
        if LABEL_ACTION[row["label"]] == "ALLOW" and not row.get("task_family"):
            raise ValueError(f"Benign training record {row.get('id')} needs a task_family annotation")
    q_rows = {name: [row for row in benchmark_rows if row.get("quadrant") == name]
              for name in EXPECTED_LABELS}
    unknown_quadrants = {row.get("quadrant") for row in benchmark_rows}.difference(EXPECTED_LABELS)
    if unknown_quadrants:
        raise ValueError(f"Unsupported benchmark quadrants: {', '.join(map(str, sorted(unknown_quadrants, key=str)))}")
    if any(not rows for rows in q_rows.values()):
        missing = [name for name, rows in q_rows.items() if not rows]
        raise ValueError(f"Benchmark is missing quadrants: {', '.join(missing)}")
    for quadrant, rows in q_rows.items():
        for row in rows:
            user_task = row.get("user_task")
            if not isinstance(user_task, str) or not user_task.strip():
                raise ValueError(f"Benchmark case {row.get('id')} needs a non-empty user_task")
            if LABEL_ACTION.get(row.get("label")) != EXPECTED_LABELS[quadrant]:
                raise ValueError(f"Benchmark case {row.get('id')} has a label/quadrant mismatch")
            if EXPECTED_LABELS[quadrant] == "ALLOW" and not row.get("task_family"):
                raise ValueError(f"Benchmark case {row.get('id')} needs a task_family annotation")
            if EXPECTED_LABELS[quadrant] == "DEFEND" and not row.get("attack_family"):
                raise ValueError(f"Benchmark case {row.get('id')} needs an attack_family annotation")

    train_task_families = {row.get("task_family") for row in train_rows if row.get("task_family")}
    val_task_families = {row.get("task_family") for row in validation_rows if row.get("task_family")}
    q1_task_families = {row.get("task_family") for row in q_rows["Q1_common_benign"]
                        if row.get("task_family")}
    q3_task_families = {row.get("task_family") for row in q_rows["Q3_unique_benign"]
                        if row.get("task_family")}
    benign_training = [row for row in train_rows + validation_rows
                       if LABEL_ACTION[row["label"]] == "ALLOW"]
    if any(not row.get("task_family") for row in benign_training):
        raise ValueError("All benign train and validation records need task_family annotations")
    if any(not row.get("task_family") for row in q_rows["Q1_common_benign"] + q_rows["Q3_unique_benign"]):
        raise ValueError("Every Q1 and Q3 benchmark case needs a task_family")
    if not q1_task_families or not q1_task_families.issubset(train_task_families):
        raise ValueError("Q1 task families must be represented in SFT training")
    if not q3_task_families or q3_task_families.intersection(train_task_families | val_task_families):
        raise ValueError("Q3 task families must be held out from SFT train and validation")

    train_attack_families = {row.get("attack_family") for row in train_rows
                             if row.get("attack_family")}
    val_attack_families = {row.get("attack_family") for row in validation_rows
                           if row.get("attack_family")}
    q2_families = {row.get("attack_family") for row in q_rows["Q2_common_injection"]
                   if row.get("attack_family")}
    q4_families = {row.get("attack_family") for row in q_rows["Q4_unique_injection"]
                   if row.get("attack_family")}
    if any(not row.get("attack_family") for row in q_rows["Q2_common_injection"] + q_rows["Q4_unique_injection"]):
        raise ValueError("Every Q2 and Q4 benchmark case needs an attack_family")
    if not q2_families or not q2_families.issubset(train_attack_families):
        raise ValueError("Q2 attack families must be represented in SFT training")
    categories = list(dict.fromkeys(attack_categories))
    unknown = set(categories).difference(ATTACK_CATEGORY_FAMILIES)
    if unknown:
        raise ValueError(f"Unknown MAPPO attacker categories: {', '.join(sorted(unknown))}")
    attacker_families = set().union(*(ATTACK_CATEGORY_FAMILIES[name] for name in categories))
    if not q2_families.issubset(attacker_families):
        raise ValueError("Q2 attack families must be represented in MAPPO attacker categories")
    seen_attack_families = train_attack_families | val_attack_families | attacker_families
    if not q4_families or q4_families.intersection(seen_attack_families):
        raise ValueError("Q4 attack families must be held out from SFT, validation, and MAPPO")

    return {
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "benchmark_records": len(benchmark_rows),
        "quadrant_counts": {name: len(rows) for name, rows in q_rows.items()},
        "q1_common_task_families": sorted(q1_task_families),
        "q3_heldout_task_families": sorted(q3_task_families),
        "q2_common_attack_families": sorted(q2_families),
        "q4_heldout_attack_families": sorted(q4_families),
        "mappo_attacker_categories": categories,
        "confirmatory_data_checks_passed": True,
    }
