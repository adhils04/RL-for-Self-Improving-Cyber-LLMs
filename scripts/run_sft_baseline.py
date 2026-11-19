# scripts/run_sft_baseline.py
"""Supervised Fine-Tuning (SFT) baseline for prompt-injection defense.

Architecture
------------
A DistilBERT model is fine-tuned as a binary injection detector on the labeled
data/sft/train.jsonl dataset using cross-entropy loss.  This is the canonical
SFT approach: learn a fixed mapping (prompt + untrusted_content) → (safe/unsafe)
from human-labeled examples.

Scope
-----
This script trains a DistilBERT binary sequence classifier. It is separate from
the Qwen2.5-0.5B language-model SFT smoke test and does not train a response-
generating language model or execute tools.

Metrics logged (written to reports/sft_training_metrics.json)
-------------------------------------------------------------
  - epoch                : training epoch index (1-indexed)
  - train_loss           : cross-entropy loss on training set
  - train_accuracy       : classification accuracy on training set
  - validation_accuracy  : accuracy on validation scenarios, reported each epoch
  - final_heldout        : held-out metrics evaluated once after training
  - validation_novel_attack_accuracy: accuracy on novel families in validation

Usage
-----
    python scripts/run_sft_baseline.py [--epochs 50] [--lr 2e-5]
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_PATH   = PROJECT_ROOT / "data" / "sft" / "train.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "sft" / "validation.jsonl"
HELDOUT_PATH = PROJECT_ROOT / "data" / "sft" / "heldout.jsonl"
REPORTS_DIR  = PROJECT_ROOT / "reports"

# Label → class index mapping
SAFE_LABELS   = {"benign_answer", "benign_tool_use"}          # defender succeeds
UNSAFE_LABELS = {"injection_resistance", "unsafe_request_refusal"}  # must resist

# Reward constants matching CoevolutionEnv
REWARD_DEFEND_SUCCESS = +3.0   # defender correctly resists attack
REWARD_BENIGN_CORRECT = +2.0   # defender correctly handles benign
REWARD_ATTACK_SUCCESS = -2.0   # attacker wins (injection obeyed / refusal wrong)
REWARD_ATTACK_WIN_ATK = +1.0   # attacker perspective when attack succeeds



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_integrity(paths):
    """Reject duplicate IDs, scenarios, or exact inputs across data splits."""
    split_rows = {}
    for split_name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required {split_name} split is missing: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"The {split_name} split is empty: {path}")
        split_rows[split_name] = rows

    keys = ("id", "scenario_id")
    for key in keys:
        owners = {}
        for split_name, rows in split_rows.items():
            for row in rows:
                value = row.get(key)
                if not value:
                    continue
                if value in owners:
                    raise ValueError(f"{key} overlap between {owners[value]} and {split_name}: {value}")
                owners[value] = split_name

    owners = {}
    for split_name, rows in split_rows.items():
        for row in rows:
            fingerprint = (row.get("user_task", "").strip(), row.get("untrusted_content", "").strip())
            if fingerprint in owners:
                raise ValueError(f"Exact task/content duplicate between {owners[fingerprint]} and {split_name}")
            owners[fingerprint] = split_name
    return split_rows


# ── Dataset ───────────────────────────────────────────────────────────────────

class InjectionDataset(Dataset):
    """Loads SFT JSONL, builds text = user_task + untrusted_content, label = {0,1}."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_length: int = 128,
        include_taskfree: bool = False,
    ):
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                user_task = rec.get("user_task", "")
                untrusted = rec.get("untrusted_content", "")
                text = f"[TASK] {user_task} [CONTENT] {untrusted}"
                label_str = rec.get("label", "benign_answer")
                if label_str not in SAFE_LABELS | UNSAFE_LABELS:
                    raise ValueError(f"Unsupported label {label_str!r} in {jsonl_path}")
                # 1 = must resist (attack/unsafe), 0 = safe to answer
                label = 1 if label_str in UNSAFE_LABELS else 0
                attack_family = rec.get("attack_family")
                is_novel = bool(rec.get("is_novel", False))
                case_id = rec.get("id", "?")
                self.samples.append(
                    {
                        "text":          text,
                        "label":         label,
                        "label_str":     label_str,
                        "attack_family": attack_family,
                        "is_novel":      is_novel,
                        "case_id":       case_id,
                    }
                )
                if include_taskfree and user_task:
                    self.samples.append(
                        {
                            "text":          f"[TASK]  [CONTENT] {untrusted}",
                            "label":         label,
                            "label_str":     label_str,
                            "attack_family": attack_family,
                            "is_novel":      is_novel,
                            "case_id":       f"{case_id}_notask",
                        }
                    )
        self.tokenizer   = tokenizer
        self.max_length  = max_length
        if self.samples:
            enc_all = self.tokenizer(
                [s["text"] for s in self.samples],
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            self._input_ids = enc_all["input_ids"]
            self._attention_mask = enc_all["attention_mask"]
        else:
            self._input_ids = torch.empty((0, max_length), dtype=torch.long)
            self._attention_mask = torch.empty((0, max_length), dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        return {
            "input_ids":      self._input_ids[idx],
            "attention_mask": self._attention_mask[idx],
            "label":          torch.tensor(s["label"], dtype=torch.long),
            "attack_family":  s["attack_family"] or "benign",
            "label_str":      s["label_str"],
            "is_novel":       s["is_novel"],
        }


# ── Reward computation (mirrors MAPPO env reward schema) ─────────────────────

def compute_sft_reward(
    predicted_label: int,
    true_label: int,
    label_str: str,
) -> tuple[float, float]:
    """Compute defender and attacker reward for one SFT prediction.

    Returns (defender_reward, attacker_reward).
    Uses same sign convention as CoevolutionEnv so results are directly comparable.
    """
    is_attack = label_str in UNSAFE_LABELS

    if is_attack:
        if predicted_label == 1:          # correctly resisted
            return REWARD_DEFEND_SUCCESS, -REWARD_DEFEND_SUCCESS
        else:                             # obeyed injection — attacker wins
            return REWARD_ATTACK_SUCCESS, REWARD_ATTACK_WIN_ATK
    else:
        if predicted_label == 0:          # correctly answered benign
            return REWARD_BENIGN_CORRECT, -REWARD_BENIGN_CORRECT
        else:                             # over-refused benign input
            return -1.0, 0.5              # mild penalty for unnecessary refusal


def evaluate_classifier(model, loader, dataset, train_families, device):
    """Evaluate one split and return its aggregate classification metrics."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            predictions.extend(model(input_ids=ids, attention_mask=mask).logits.argmax(dim=-1).cpu().tolist())

    correct = id_correct = novel_correct = benign_correct = 0
    id_total = novel_total = benign_total = attack_total = attack_errors = 0
    reward_sum = 0.0
    for pred, sample in zip(predictions, dataset.samples):
        label = sample["label"]
        label_str = sample["label_str"]
        family = sample["attack_family"]
        is_novel = bool(sample["is_novel"] or (family and family not in train_families))
        correct += int(pred == label)
        if label_str in UNSAFE_LABELS:
            attack_total += 1
            attack_errors += int(pred != 1)
            if is_novel:
                novel_total += 1
                novel_correct += int(pred == label)
            else:
                id_total += 1
                id_correct += int(pred == label)
        else:
            benign_total += 1
            benign_correct += int(pred == 0)
        reward_sum += compute_sft_reward(pred, label, label_str)[0]

    n = max(len(dataset), 1)
    return {
        "accuracy": correct / n,
        "attack_success_rate": attack_errors / max(attack_total, 1),
        "in_distribution_accuracy": id_correct / max(id_total, 1),
        "in_distribution_attack_success_rate": (id_total - id_correct) / max(id_total, 1),
        "novel_attack_accuracy": novel_correct / max(novel_total, 1),
        "novel_attack_success_rate": (novel_total - novel_correct) / max(novel_total, 1),
        "benign_accuracy": benign_correct / max(benign_total, 1),
        "mean_defender_reward": reward_sum / n,
        "n_cases": len(dataset),
        "n_attacks": attack_total,
        "n_novel_attacks": novel_total,
        "n_benign": benign_total,
    }


# ── Main SFT trainer ─────────────────────────────────────────────────────────

def run_sft_baseline(
    epochs: int = 50,
    lr: float = 2e-5,
    batch_size: int = 4,
    max_length: int = 128,
    model_name: str = "roberta-base",
    seed: int = 42,
) -> Path:
    """Train SFT classifier and return path to metrics JSON."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("=" * 60)
    print("SFT Baseline — Prompt Injection Defender")
    print("=" * 60)
    print(f"  Model   : {model_name}")
    print(f"  Epochs  : {epochs}")
    print(f"  LR      : {lr}")
    print(f"  Train   : {TRAIN_PATH}")
    print(f"  Validation: {VALIDATION_PATH}")
    print(f"  Final held-out: {HELDOUT_PATH}")
    print()

    validate_split_integrity({"train": TRAIN_PATH, "validation": VALIDATION_PATH, "heldout": HELDOUT_PATH})
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device  : {device}", flush=True)

    # ── Tokenizer & model ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds   = InjectionDataset(
        TRAIN_PATH, tokenizer, max_length, include_taskfree=True
    )
    validation_ds = InjectionDataset(VALIDATION_PATH, tokenizer, max_length)
    heldout_ds = InjectionDataset(HELDOUT_PATH, tokenizer, max_length)

    train_loader   = DataLoader(train_ds,   batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=16, shuffle=False)
    heldout_loader = DataLoader(heldout_ds, batch_size=16, shuffle=False)

    print(f"  Train examples   : {len(train_ds)}")
    print(f"  Validation examples: {len(validation_ds)}")

    # Identify attack families in heldout but NOT in train → novelty axis
    train_families  = {s["attack_family"] for s in train_ds.samples if s["attack_family"]}
    print()

    # ── Training loop ─────────────────────────────────────────────────────────
    episode_records = []   # one record per epoch (maps to MAPPO episode schema)

    for epoch in range(1, epochs + 1):
        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        train_loss   = 0.0
        train_correct = 0
        train_total  = 0

        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * len(lbls)
            preds          = logits.argmax(dim=-1)
            train_correct += (preds == lbls).sum().item()
            train_total   += len(lbls)

        avg_train_loss = train_loss / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        # The validation split is used for epoch monitoring. The held-out split
        # is not touched until training is complete.
        validation_metrics = evaluate_classifier(
            model, validation_loader, validation_ds, train_families, device
        )

        record = {
            "episode":               epoch,
            "global_step":           epoch,
            "train_loss":            round(avg_train_loss, 6),
            "train_accuracy":        round(train_accuracy, 4),
            "validation_accuracy": round(validation_metrics["accuracy"], 4),
            "validation_attack_success_rate": round(validation_metrics["attack_success_rate"], 4),
            "validation_novel_attack_accuracy": round(validation_metrics["novel_attack_accuracy"], 4),
            "validation_benign_accuracy": round(validation_metrics["benign_accuracy"], 4),
            "validation_defender_reward": round(validation_metrics["mean_defender_reward"], 4),
            "critic_loss":           0.0,
            "attacker_loss":         0.0,
            "sft_cross_entropy":     round(avg_train_loss, 6),
            "gate_block_rate":       0.0,
            "kl_divergence":         0.0,
            "pre_gate_unsafe_rate":  0.0,
            "case_kind":             "sft_validation",
            "attack_family":         None,
        }
        episode_records.append(record)

        if epochs <= 20 or epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(
                f"[Epoch {epoch:>3}/{epochs}]  "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.3f} | "
                f"Validation Acc: {validation_metrics['accuracy']:.3f} | "
                f"Novel Acc: {validation_metrics['novel_attack_accuracy']:.3f} | "
                f"Benign Acc: {validation_metrics['benign_accuracy']:.3f} | "
                f"ASR: {validation_metrics['attack_success_rate']:.3f}",
                flush=True,
            )

    # ── Save final model ──────────────────────────────────────────────────────
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "sft_run"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    checkpoint_path = ckpt_dir / "sft_final.pt"
    torch.save(cpu_state, checkpoint_path)
    print(f"\n[✓] SFT model checkpoint saved → {checkpoint_path}")

    # Final evaluation only: held-out data is accessed once after training.
    heldout_metrics = evaluate_classifier(model, heldout_loader, heldout_ds, train_families, device)

    # ── Save metrics JSON in same schema as MAPPO ─────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # asr_vs_step alias for plot generator compatibility
    asr_vs_step = [
        {
            "global_step":          r["episode"],
            "asr":                  r["validation_attack_success_rate"],
            "defender_reward":      r["validation_defender_reward"],
            "attacker_reward":      0.0,
            "critic_loss":          0.0,
            "attacker_loss":        0.0,
            "defender_loss":        r["train_loss"],
            "sft_cross_entropy":    r["train_loss"],
            "gate_block_rate":      0.0,
            "kl_divergence":        0.0,
            "pre_gate_unsafe_rate": 0.0,
            "train_loss":           r["train_loss"],
            "train_accuracy":       r["train_accuracy"],
            "validation_accuracy":  r["validation_accuracy"],
            "validation_novel_attack_accuracy": r["validation_novel_attack_accuracy"],
            "case_kind":            "sft_validation",
            "attack_family":        None,
        }
        for r in episode_records
    ]

    n       = len(episode_records)
    report  = {
        "run_metadata": {
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "scope":          "SFT baseline — DistilBERT binary injection classifier trained on static labeled dataset",
            "algorithm":      "Supervised Fine-Tuning (SFT)",
            "model":          model_name,
            "n_epochs":       n,
            "train_examples": len(train_ds),
            "validation_examples": len(validation_ds),
            "heldout_examples": len(heldout_ds),
            "heldout_evaluated_once_after_training": True,
            "checkpoint_sha256": sha256_file(PROJECT_ROOT / "checkpoints/sft_run/sft_final.pt"),
            "lr":             lr,
            "batch_size":     batch_size,
            "seed":           seed,
            "training_data_sha256": sha256_file(TRAIN_PATH),
            "validation_data_sha256": sha256_file(VALIDATION_PATH),
            "heldout_data_sha256": sha256_file(HELDOUT_PATH),
            "note": (
                "Held-out data is used once after training; epoch monitoring uses validation data. "
                "This classifier experiment is distinct from the Qwen language-model smoke test."
            ),
        },
        "aggregate": {
            "final_train_loss":           round(episode_records[-1]["train_loss"], 6),
            "final_train_accuracy":       round(episode_records[-1]["train_accuracy"], 4),
            "final_validation_accuracy": round(episode_records[-1]["validation_accuracy"], 4),
            "final_heldout": {k: round(v, 4) if isinstance(v, float) else v for k, v in heldout_metrics.items()},
        },
        "asr_vs_step":     asr_vs_step,
        "episode_records": episode_records,
    }

    out = REPORTS_DIR / "sft_training_metrics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[✓] SFT training metrics saved → {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run SFT baseline for prompt injection defense"
    )
    parser.add_argument("--epochs",     type=int,   default=50,    help="Training epochs")
    parser.add_argument("--lr",         type=float, default=2e-5,  help="Learning rate")
    parser.add_argument("--batch-size", type=int,   default=4,     help="Batch size")
    parser.add_argument("--model",      type=str,   default="roberta-base")
    parser.add_argument("--seed",       type=int,   default=42, help="Random seed for this training run")
    args = parser.parse_args()
    run_sft_baseline(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_name=args.model,
        seed=args.seed,
    )
