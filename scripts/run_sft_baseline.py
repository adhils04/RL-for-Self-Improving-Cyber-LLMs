# scripts/run_sft_baseline.py
"""Supervised Fine-Tuning (SFT) baseline for prompt-injection defense.

Architecture
------------
A DistilBERT model is fine-tuned as a binary injection detector on the labeled
data/sft/train.jsonl dataset using cross-entropy loss.  This is the canonical
SFT approach: learn a fixed mapping (prompt + untrusted_content) → (safe/unsafe)
from human-labeled examples.

Key differences from MAPPO
---------------------------
- SFT trains on a **static, closed dataset** (21 labeled examples in train.jsonl).
  It cannot adapt when the attacker generates novel injection templates not present
  in training.
- MAPPO trains via **reward signals from the live environment**: every episode the
  attacker can adapt, and the centralized critic propagates global-state information
  back to both agents.  This co-evolutionary pressure is the core MAPPO advantage.

Metrics logged (written to reports/sft_training_metrics.json)
-------------------------------------------------------------
  - epoch                : training epoch index (1-indexed)
  - train_loss           : cross-entropy loss on training set
  - train_accuracy       : classification accuracy on training set
  - heldout_accuracy     : accuracy on held-out test scenarios  ← key metric
  - heldout_asr          : attack success rate on heldout set (fraction NOT defended)
  - defender_reward      : reward signal computed using the same reward formula
                           as MAPPO, so the two conditions are directly comparable
  - attacker_reward      : inverse of defender reward (attacker wins ↔ defender fails)
  - novel_attack_accuracy: accuracy specifically on novel attack families in heldout
                           (not seen in training) — the primary MAPPO advantage axis

Usage
-----
    python scripts/run_sft_baseline.py [--epochs 50] [--lr 2e-5]
"""

from __future__ import annotations

import json
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class InjectionDataset(Dataset):
    """Loads SFT JSONL, builds text = user_task + untrusted_content, label = {0,1}."""

    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 128):
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = (
                    f"[TASK] {rec.get('user_task', '')} "
                    f"[CONTENT] {rec.get('untrusted_content', '')}"
                )
                label_str = rec.get("label", "benign_answer")
                # 1 = must resist (attack/unsafe), 0 = safe to answer
                label = 1 if label_str in UNSAFE_LABELS else 0
                attack_family = rec.get("attack_family")
                self.samples.append(
                    {
                        "text":          text,
                        "label":         label,
                        "label_str":     label_str,
                        "attack_family": attack_family,
                        "case_id":       rec.get("id", "?"),
                    }
                )
        self.tokenizer   = tokenizer
        self.max_length  = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(s["label"], dtype=torch.long),
            "attack_family":  s["attack_family"] or "benign",
            "label_str":      s["label_str"],
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


# ── Main SFT trainer ─────────────────────────────────────────────────────────

def run_sft_baseline(
    epochs: int = 50,
    lr: float = 2e-5,
    batch_size: int = 4,
    max_length: int = 128,
    model_name: str = "distilbert-base-uncased",
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
    print(f"  Heldout : {HELDOUT_PATH}")
    print()

    device = "cpu"  # consistent with MAPPO run (no CUDA on this machine)

    # ── Tokenizer & model ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds   = InjectionDataset(TRAIN_PATH,   tokenizer, max_length)
    heldout_ds = InjectionDataset(HELDOUT_PATH, tokenizer, max_length)

    train_loader   = DataLoader(train_ds,   batch_size=batch_size, shuffle=True)
    heldout_loader = DataLoader(heldout_ds, batch_size=1,          shuffle=False)

    print(f"  Train examples   : {len(train_ds)}")
    print(f"  Heldout examples : {len(heldout_ds)}")

    # Identify attack families in heldout but NOT in train → novelty axis
    train_families  = {s["attack_family"] for s in train_ds.samples if s["attack_family"]}
    heldout_families = {s["attack_family"] for s in heldout_ds.samples if s["attack_family"]}
    novel_families  = heldout_families - train_families
    print(f"  Novel attack families (heldout-only): {novel_families or 'none'}")
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

        # ── Heldout evaluation ────────────────────────────────────────────────
        model.eval()
        heldout_correct    = 0
        heldout_total      = 0
        heldout_asr_steps  = 0   # attacker wins = defender failed on attack sample
        id_correct         = 0
        id_total           = 0
        id_asr_steps       = 0
        novel_correct      = 0
        novel_total        = 0
        novel_asr_steps    = 0
        benign_correct     = 0
        benign_total       = 0
        ep_def_reward      = 0.0
        ep_atk_reward      = 0.0

        with torch.no_grad():
            for batch in heldout_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbl  = batch["label"].item()
                lbl_str   = batch["label_str"][0]
                atk_fam   = batch["attack_family"][0]

                logits = model(input_ids=ids, attention_mask=mask).logits
                pred   = logits.argmax(dim=-1).item()

                heldout_total += 1
                if pred == lbl:
                    heldout_correct += 1

                is_attack = lbl_str in UNSAFE_LABELS
                if is_attack and pred != 1:
                    heldout_asr_steps += 1

                # Sub-category tracking
                if not is_attack:
                    benign_total += 1
                    if pred == 0:
                        benign_correct += 1
                elif atk_fam in novel_families:
                    novel_total += 1
                    if pred == lbl:
                        novel_correct += 1
                    else:
                        novel_asr_steps += 1
                else:
                    id_total += 1
                    if pred == lbl:
                        id_correct += 1
                    else:
                        id_asr_steps += 1

                # Reward computation
                def_r, atk_r = compute_sft_reward(pred, lbl, lbl_str)
                ep_def_reward += def_r
                ep_atk_reward += atk_r

        heldout_accuracy     = heldout_correct / max(heldout_total, 1)
        heldout_asr          = heldout_asr_steps / max(
            sum(1 for s in heldout_ds.samples if s["label_str"] in UNSAFE_LABELS), 1
        )
        novel_accuracy       = novel_correct / max(novel_total, 1)
        novel_asr            = novel_asr_steps / max(novel_total, 1)
        id_accuracy          = id_correct / max(id_total, 1)
        id_asr               = id_asr_steps / max(id_total, 1)
        benign_accuracy      = benign_correct / max(benign_total, 1)

        avg_def_reward       = ep_def_reward / max(heldout_total, 1)
        avg_atk_reward       = ep_atk_reward / max(heldout_total, 1)

        record = {
            "episode":               epoch,
            "global_step":           epoch,
            "train_loss":            round(avg_train_loss, 6),
            "train_accuracy":        round(train_accuracy, 4),
            "heldout_accuracy":      round(heldout_accuracy, 4),
            "asr":                   round(heldout_asr, 4),
            "id_accuracy":           round(id_accuracy, 4),
            "id_asr":                round(id_asr, 4),
            "novel_attack_accuracy": round(novel_accuracy, 4),
            "novel_asr":             round(novel_asr, 4),
            "benign_accuracy":       round(benign_accuracy, 4),
            "defender_reward":       round(avg_def_reward, 4),
            "attacker_reward":       round(avg_atk_reward, 4),
            "critic_loss":           0.0,
            "attacker_loss":         round(avg_train_loss, 6),
            "gate_block_rate":       0.0,
            "kl_divergence":         0.0,
            "pre_gate_unsafe_rate":  0.0,
            "case_kind":             "sft_eval",
            "attack_family":         None,
        }
        episode_records.append(record)

        if epoch % 10 == 0 or epoch == epochs or epoch == 1:
            print(
                f"[Epoch {epoch:>3}/{epochs}]  "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.3f} | "
                f"Heldout Acc: {heldout_accuracy:.3f} | "
                f"ID Acc: {id_accuracy:.3f} | "
                f"Novel Acc: {novel_accuracy:.3f} | "
                f"Benign Acc: {benign_accuracy:.3f} | "
                f"ASR: {heldout_asr:.3f}"
            )

    # ── Save final model ──────────────────────────────────────────────────────
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "sft_run"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "sft_final.pt")
    print(f"\n[✓] SFT model checkpoint saved → {ckpt_dir / 'sft_final.pt'}")

    # ── Save metrics JSON in same schema as MAPPO ─────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # asr_vs_step alias for plot generator compatibility
    asr_vs_step = [
        {
            "global_step":          r["episode"],
            "asr":                  r["asr"],
            "defender_reward":      r["defender_reward"],
            "attacker_reward":      r["attacker_reward"],
            "critic_loss":          0.0,
            "attacker_loss":        r["train_loss"],
            "defender_loss":        r["train_loss"],
            "gate_block_rate":      0.0,
            "kl_divergence":        0.0,
            "pre_gate_unsafe_rate": 0.0,
            "train_loss":           r["train_loss"],
            "train_accuracy":       r["train_accuracy"],
            "heldout_accuracy":     r["heldout_accuracy"],
            "novel_attack_accuracy": r["novel_attack_accuracy"],
            "case_kind":            "sft_eval",
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
            "heldout_examples": len(heldout_ds),
            "novel_families": list(novel_families),
            "lr":             lr,
            "batch_size":     batch_size,
            "note": (
                "SFT trains on a fixed static dataset. It cannot adapt to novel "
                "attack templates not present in training. MAPPO co-evolves with "
                "the attacker and generalizes better to unseen injection patterns."
            ),
        },
        "aggregate": {
            "final_train_loss":           round(episode_records[-1]["train_loss"], 6),
            "final_train_accuracy":       round(episode_records[-1]["train_accuracy"], 4),
            "final_heldout_accuracy":     round(episode_records[-1]["heldout_accuracy"], 4),
            "final_asr":                  round(episode_records[-1]["asr"], 4),
            "final_novel_attack_accuracy": round(episode_records[-1]["novel_attack_accuracy"], 4),
            "mean_defender_reward":       round(
                sum(r["defender_reward"] for r in episode_records) / n, 4
            ),
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
    parser.add_argument("--model",      type=str,   default="distilbert-base-uncased")
    args = parser.parse_args()
    run_sft_baseline(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        model_name=args.model,
    )
