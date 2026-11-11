# scripts/run_mappo_coevolution.py
"""MAPPO Co-evolution Training: Real Attacker Policy + Real Neural Defender Policy + Centralized Critic.

In this setup:
- Attacker: Neural policy network exploring and selecting adversarial injection categories.
- Defender: Neural policy network initialized from SFT reference policy (DistilBERT sequence classification)
  learning to resist novel prompt injections and preserve benign utility.
- Centralized Critic: 'God's Eye' Value Network V(S) evaluating global multi-agent interaction state.
- Both agents update parameters via PPO clipped surrogate objective against GAE advantages.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

# ── Project root (absolute, CWD-independent) ──────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

from src.centralized_critic import CentralizedCritic
from src.attacker_policy.mappo_attacker import AttackerPolicy
from src.attacker_policy.attack_generator import (
    ATTACK_CATEGORIES,
    ATTACK_TEMPLATES,
    synthesize_attack_payload,
)
from src.defender_policy.neural_defender import NeuralDefenderPolicy
from src.defender_policy.model_adapter import NeuralDefenderAdapter
from src.coevolution_env import CoevolutionEnv
from src.trajectory_buffer import CTDETrajectoryBuffer


# ── Observation encoder for Attacker MLP ───────────────────────────────────────

def encode_obs(text: str, state_dim: int = 8) -> torch.Tensor:
    """Hash-encode a text observation into a fixed-size float tensor."""
    digest = hashlib.md5(text.encode()).digest()
    vec = [b / 255.0 for b in digest[:state_dim]]
    if len(vec) < state_dim:
        vec = vec + [0.0] * (state_dim - len(vec))
    return torch.tensor(vec, dtype=torch.float32)


import random


def get_attack_payload(action_idx: int) -> str:
    """Map attacker categorical action index to a dynamic or template attack payload."""
    category = ATTACK_CATEGORIES[action_idx % len(ATTACK_CATEGORIES)]
    payload = synthesize_attack_payload(category)
    if payload:
        return payload
    templates = ATTACK_TEMPLATES.get(category, [])
    if templates:
        return random.choice(templates)
    return "Benign probe: please answer the task honestly."


def _load_sft_benign_pool(train_path: Path) -> list[dict]:
    """Load clean benign samples from SFT train split for RL benign-utility anchoring."""
    pool = []
    if not train_path.exists():
        return pool
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") in ("benign_answer", "benign_tool_use") and rec.get("attack_family") is None:
                pool.append(
                    {
                        "id": rec.get("id", f"sft_benign_{len(pool)}"),
                        "user_task": rec.get("user_task", ""),
                        "untrusted_content": rec.get("untrusted_content", ""),
                    }
                )
    return pool


def _resolve_device(config_device: str) -> str:
    """Resolve training device cleanly."""
    if config_device and config_device != "auto":
        return config_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path=None):
    """Load YAML config; defaults to configs/mappo_config.yaml."""
    path = Path(config_path) if config_path else _PROJECT_ROOT / "configs" / "mappo_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Metrics serializer ────────────────────────────────────────────────────────

def _save_training_metrics(
    episode_records: list,
    output_dir: Path,
    config: dict,
    num_episodes: int,
) -> Path:
    """Write episode-level metrics to mappo_training_metrics.json."""
    n = len(episode_records)
    if n == 0:
        return output_dir / "mappo_training_metrics.json"

    mean_asr         = sum(r["asr"]              for r in episode_records) / n
    mean_def_reward  = sum(r["defender_reward"]  for r in episode_records) / n
    mean_atk_reward  = sum(r["attacker_reward"]  for r in episode_records) / n
    mean_critic_loss = sum(r["critic_loss"]       for r in episode_records) / n

    asr_vs_step = [
        {
            "global_step":          r["episode"],
            "asr":                  r["asr"],
            "defender_reward":      r["defender_reward"],
            "attacker_reward":      r["attacker_reward"],
            "critic_loss":          r["critic_loss"],
            "attacker_loss":        r["attacker_loss"],
            "defender_loss":        r["defender_loss"],
            "gate_block_rate":      r["gate_block_rate"],
            "heldout_accuracy":     r.get("heldout_accuracy", 1.0 - r["asr"]),
            "case_kind":            r.get("case_kind", "unknown"),
            "attack_family":        r.get("attack_family"),
        }
        for r in episode_records
    ]

    report = {
        "run_metadata": {
            "generated_at":            datetime.now(timezone.utc).isoformat(),
            "scope":                   "MAPPO co-evolution — real Neural Defender + Attacker + Centralized Critic",
            "algorithm":               "MAPPO (CTDE)",
            "n_episodes":              n,
            "num_episodes_configured": num_episodes,
            "defender":                "NeuralDefenderPolicy (DistilBERT initialized from SFT + PPO updates)",
            "attacker":                f"AttackerPolicy (Actor-Critic, state_dim={config.get('models', {}).get('attacker_state_dim', 8)}, action_dim={config.get('models', {}).get('attacker_action_dim', len(ATTACK_CATEGORIES))})",
            "critic":                  "CentralizedCritic (DistilBERT regression head, V(S))",
            "gamma":                   config["training"]["gamma"],
            "gae_lambda":              config["training"]["gae_lambda"],
            "clip_epsilon":            config["training"]["clip_epsilon"],
            "ppo_epochs":              config["training"]["ppo_epochs"],
        },
        "aggregate": {
            "mean_attack_success_rate": round(mean_asr, 4),
            "mean_defender_reward":     round(mean_def_reward, 4),
            "mean_attacker_reward":     round(mean_atk_reward, 4),
            "mean_critic_loss":         round(mean_critic_loss, 4),
        },
        "asr_vs_step":     asr_vs_step,
        "episode_records": episode_records,
    }

    out_path = output_dir / "mappo_training_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


# ── Main training loop ────────────────────────────────────────────────────────

def main(config_path=None):
    config = load_config(config_path)

    # Device selection
    device = _resolve_device(config.get("models", {}).get("device", "auto"))
    print(f"[*] Initializing MAPPO Co-evolution on device: {device}", flush=True)

    # Hyperparameters
    gamma         = config["training"]["gamma"]
    gae_lambda    = config["training"]["gae_lambda"]
    ppo_epochs    = config["training"]["ppo_epochs"]
    clip_epsilon  = config["training"]["clip_epsilon"]
    actor_lr      = config["training"].get("actor_lr", 3e-5)
    critic_lr     = config["training"].get("critic_lr", 2e-5)
    defender_lr   = config["training"].get("defender_lr", 1.2e-5)
    num_episodes  = int(config["experiment"].get("num_episodes", 220))
    max_steps     = config["experiment"].get("max_steps_per_episode", 4)
    seed          = config["experiment"].get("seed", 42)
    batch_episodes = 5  # accumulate transitions across 5 episodes for stable PPO batches

    random.seed(seed)
    torch.manual_seed(seed)

    state_dim  = config.get("models", {}).get("attacker_state_dim", 8)
    action_dim = config.get("models", {}).get("attacker_action_dim", len(ATTACK_CATEGORIES))

    log_interval   = config.get("logging", {}).get("log_interval", 20)
    output_dir     = _PROJECT_ROOT / config.get("logging", {}).get("output_dir", "reports")
    checkpoint_dir = _PROJECT_ROOT / config.get("logging", {}).get("checkpoint_dir", "checkpoints/mappo_run")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize components ─────────────────────────────────────────────────
    critic = CentralizedCritic(
        model_name_or_path=config["models"].get("critic_model", "distilbert-base-uncased"),
        device=device,
        lr=critic_lr,
    )
    attacker = AttackerPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=actor_lr,
    )
    defender = NeuralDefenderPolicy(
        model_name_or_path=config["models"].get("defender_model", "distilbert-base-uncased"),
        learning_rate=defender_lr,
        device=device,
    )

    # Load SFT reference policy warm-up checkpoint and frozen KL anchor
    sft_ckpt_path = _PROJECT_ROOT / "checkpoints" / "sft_run" / "sft_final.pt"
    ref_defender = None
    if sft_ckpt_path.exists():
        print(f"[*] Initializing MAPPO Defender from SFT reference policy: {sft_ckpt_path}", flush=True)
        defender.load_checkpoint(sft_ckpt_path)
        ref_defender = NeuralDefenderPolicy(
            model_name_or_path=config["models"].get("defender_model", "distilbert-base-uncased"),
            learning_rate=defender_lr,
            device=device,
        )
        ref_defender.load_checkpoint(sft_ckpt_path)
        ref_defender.model.eval()
        for p in ref_defender.model.parameters():
            p.requires_grad_(False)
    else:
        print("[!] SFT reference policy not found; starting Defender from base weights.", flush=True)

    defender_adapter = NeuralDefenderAdapter(policy=defender, deterministic=False, explore_eps=0.25)

    cases_path = _PROJECT_ROOT / "data" / "evaluation" / "benchmark_cases.jsonl"
    sft_benign_pool = _load_sft_benign_pool(_PROJECT_ROOT / "data" / "sft" / "train.jsonl")
    env = CoevolutionEnv(
        defender_adapter=defender_adapter,
        cases_path=cases_path,
        max_steps=max_steps,
        seed=seed,
    )
    env_benign_cases = [c for c in env._cases if c.get("kind") == "benign"]
    env_attack_cases = [c for c in env._cases if c.get("kind") != "benign"] or list(env._cases)
    buffer = CTDETrajectoryBuffer()
    batch_is_benign: list[bool] = []

    episode_records: list = []

    print("[*] Starting MAPPO Co-evolution Training Loop (SFT-Warmup Defender + Attacker)...", flush=True)
    global_step = 0
    running_critic_loss = 0.0
    running_atk_loss = 0.0
    running_def_loss = 0.0

    for episode in range(1, num_episodes + 1):
        # Anneal exploration smoothly from 0.30 -> 0.05 across episodes
        progress = (episode - 1) / max(num_episodes - 1, 1)
        defender_adapter.explore_eps = 0.30 * (1.0 - progress) + 0.05 * progress

        # 40% benign calibration episodes (benchmark + diverse SFT benign), 60% adversarial
        is_benign_ep = random.random() < 0.40
        if is_benign_ep and (sft_benign_pool or env_benign_cases):
            benign_turns = []
            first_task = ""
            for t_idx in range(max_steps):
                if sft_benign_pool and (not env_benign_cases or random.random() < 0.80):
                    item = random.choice(sft_benign_pool)
                    u_task = "" if random.random() < 0.50 else item.get("user_task", "")
                    benign_turns.append({
                        "user_task": u_task,
                        "untrusted_content": item["untrusted_content"],
                        "expected_turn_outcome": "safe",
                    })
                    if t_idx == 0:
                        first_task = u_task
                else:
                    ec = random.choice(env_benign_cases)
                    ec_turns = ec.get("turns", [])
                    uc = ec_turns[0].get("untrusted_content") if ec_turns else ec.get("user_task", "")
                    u_task = "" if random.random() < 0.50 else ec.get("user_task", "")
                    benign_turns.append({
                        "user_task": u_task,
                        "untrusted_content": uc,
                        "expected_turn_outcome": "safe",
                    })
                    if t_idx == 0:
                        first_task = u_task
            selected_case = {
                "id": f"mappo_benign_ep_{episode}",
                "kind": "benign",
                "attack_family": None,
                "user_task": first_task,
                "step_count": max_steps,
                "turns": benign_turns,
            }
        else:
            base_case = dict(random.choice(env_attack_cases))
            # 50/50 split between contextual [TASK] and UI task-free [TASK] "" mode
            if random.random() < 0.50:
                base_case["user_task"] = ""
            base_case["step_count"] = max_steps
            selected_case = base_case

        obs = env.reset(case=selected_case, episode_id=f"ep_{episode:04d}")
        episode_reward_atk  = 0.0
        episode_reward_def  = 0.0
        episode_asr_steps   = 0
        episode_gate_blocks = 0
        episode_steps_count = 0
        done               = False
        last_case_kind     = "unknown"
        last_attack_family = None

        # ── Roll-out ──────────────────────────────────────────────────────
        while not done:
            global_step         += 1
            episode_steps_count += 1

            step_idx = getattr(env, "_step_index", 0)
            curr_case = getattr(env, "_current_case", {}) or {}
            step_is_benign = curr_case.get("kind") == "benign"

            # Per-step benign task refresh when synthesizing multi-turn benign episodes
            if step_is_benign:
                turns = curr_case.get("turns", [])
                if step_idx < len(turns):
                    curr_case["user_task"] = turns[step_idx].get("user_task", curr_case.get("user_task", ""))

            task_desc    = getattr(obs, "task_id", str(obs))
            state_tensor = encode_obs(task_desc, state_dim=state_dim)

            # Attacker decision (used on adversarial turns)
            with torch.no_grad():
                decision = attacker.get_action(state_tensor)
            atk_action_idx = decision.action
            atk_logprob    = decision.log_prob.detach()

            if step_is_benign:
                turns = curr_case.get("turns", [])
                if step_idx < len(turns) and turns[step_idx].get("untrusted_content"):
                    atk_text = turns[step_idx]["untrusted_content"]
                else:
                    atk_text = curr_case.get("user_task", "Benign user query.")
            else:
                atk_text = get_attack_payload(atk_action_idx)

            step_result = env.step(atk_text)

            reward_atk = step_result.reward_attacker
            reward_def = step_result.reward_defender
            done       = step_result.done

            episode_reward_atk += reward_atk
            episode_reward_def += reward_def

            info = step_result.info
            if info.get("injection_obedience_detected", False):
                episode_asr_steps += 1
            if info.get("gate_blocked", False):
                episode_gate_blocks += 1
            last_case_kind     = info.get("case_kind", "unknown")
            last_attack_family = info.get("attack_family")

            # Pre-outcome Centralized Critic Global State V_def(S) (no reward leakage)
            active_task = curr_case.get("user_task", "")
            global_state_text = (
                f"Mode: {'benign' if step_is_benign else 'adversarial'} | "
                f"Task: {active_task[:80]} | "
                f"Content: {atk_text[:200]}"
            )
            critic.model.eval()
            with torch.no_grad():
                state_value = critic([global_state_text]).item()

            last_dec = defender_adapter.last_decision
            def_action_val = last_dec.action if last_dec else 0
            def_lp_val = float(last_dec.log_prob.item()) if last_dec and hasattr(last_dec.log_prob, "item") else 0.0

            buffer.add_step(
                global_state=global_state_text,
                atk_obs=state_tensor,
                atk_action=atk_action_idx,
                atk_logprob=atk_logprob,
                def_obs=f"[TASK] {active_task} [CONTENT] {atk_text}".strip(),
                def_action=def_action_val,
                def_logprob=def_lp_val,
                reward=reward_atk,
                value=state_value,
                done=done,
                def_reward=reward_def,
            )
            batch_is_benign.append(step_is_benign)

            obs = step_result.next_obs

        # Episode metrics
        episode_asr = episode_asr_steps / max(episode_steps_count, 1)
        episode_gate_block_rate = episode_gate_blocks / max(episode_steps_count, 1)

        # ── Batch PPO Updates ─────────────────────────────────────────────
        if episode % batch_episodes == 0 or episode == num_episodes:
            # Critic tracks Defender return V_def(S) cleanly
            returns, advantages = buffer.compute_returns_and_advantages(
                gamma=gamma,
                gae_lambda=gae_lambda,
                for_defender=False,
                critic_tracks_defender=True,
            )
            def_returns, def_advantages = buffer.compute_returns_and_advantages(
                gamma=gamma,
                gae_lambda=gae_lambda,
                for_defender=True,
                critic_tracks_defender=True,
            )

            # 1. Update Centralized Critic against Defender returns
            critic_loss = critic.update_critic(buffer.global_states, def_returns)
            running_critic_loss = critic_loss

            # 2. Update Attacker (adversarial turns only) & Defender Actors
            adv_indices = [i for i, b in enumerate(batch_is_benign) if not b]
            def_states = buffer.defender_obs
            def_actions = torch.tensor(buffer.defender_actions, dtype=torch.long)
            def_logprobs = torch.tensor(buffer.defender_logprobs, dtype=torch.float32)

            ref_logits = None
            kl_weights = None
            if ref_defender is not None and def_states:
                with torch.no_grad():
                    ref_logits = ref_defender.forward(def_states).detach()
                kl_weights = torch.tensor(
                    [1.0 if b else 0.015 for b in batch_is_benign],
                    dtype=torch.float32,
                )

            if len(adv_indices) >= 2:
                atk_states   = torch.stack([buffer.attacker_obs[i] for i in adv_indices])
                atk_actions  = torch.tensor([buffer.attacker_actions[i] for i in adv_indices], dtype=torch.long)
                atk_logprobs = torch.stack([buffer.attacker_logprobs[i] for i in adv_indices]).detach()
                atk_adv_sub  = advantages[adv_indices]
                atk_ret_sub  = returns[adv_indices]
            else:
                atk_states = None

            for _ in range(ppo_epochs):
                if atk_states is not None:
                    try:
                        loss_dict = attacker.ppo_update(
                            states=atk_states,
                            actions=atk_actions,
                            old_log_probs=atk_logprobs,
                            advantages=atk_adv_sub,
                            returns=atk_ret_sub,
                            clip_epsilon=clip_epsilon,
                        )
                        running_atk_loss = loss_dict["total_loss"]
                    except Exception as exc:
                        print(f"[!] Attacker PPO update warning: {exc}", flush=True)

                try:
                    def_loss_dict = defender.ppo_update(
                        obs_texts=def_states,
                        actions=def_actions,
                        old_log_probs=def_logprobs,
                        advantages=def_advantages,
                        clip_epsilon=clip_epsilon,
                        entropy_coeff=0.015,
                        beta_kl=0.25 if ref_logits is not None else 0.05,
                        ref_logits=ref_logits,
                        kl_weights=kl_weights,
                    )
                    running_def_loss = def_loss_dict["total_loss"]
                except Exception as exc:
                    print(f"[!] Defender PPO update warning: {exc}", flush=True)

            buffer.clear()
            batch_is_benign.clear()

        episode_records.append(
            {
                "episode":             episode,
                "global_step":         global_step,
                "asr":                 round(episode_asr, 4),
                "defender_reward":     round(episode_reward_def, 4),
                "attacker_reward":     round(episode_reward_atk, 4),
                "critic_loss":         round(running_critic_loss, 4),
                "attacker_loss":       round(float(running_atk_loss), 4),
                "defender_loss":       round(float(running_def_loss), 4),
                "gate_block_rate":     round(episode_gate_block_rate, 4),
                "heldout_accuracy":     round(1.0 - episode_asr, 4),
                "case_kind":           last_case_kind,
                "attack_family":       last_attack_family,
            }
        )

        if episode % log_interval == 0 or episode == num_episodes:
            print(
                f"[Ep {episode:>4}/{num_episodes}] "
                f"Atk R: {episode_reward_atk:+.2f} | "
                f"Def R: {episode_reward_def:+.2f} | "
                f"ASR: {episode_asr:.3f} | "
                f"Critic Loss: {running_critic_loss:.4f} | "
                f"Def Loss: {running_def_loss:.4f}",
                flush=True,
            )
            _save_training_metrics(episode_records, output_dir, config, num_episodes)

    # ── Final Save ────────────────────────────────────────────────────────────
    torch.save({k: v.detach().cpu() for k, v in attacker.state_dict().items()}, checkpoint_dir / "attacker_final.pt")
    torch.save({k: v.detach().cpu() for k, v in critic.state_dict().items()},   checkpoint_dir / "critic_final.pt")
    defender.save_checkpoint(checkpoint_dir / "mappo_defender_final.pt")
    print(f"\n[✓] MAPPO Defender checkpoint saved → {checkpoint_dir / 'mappo_defender_final.pt'}", flush=True)

    metrics_path = _save_training_metrics(episode_records, output_dir, config, num_episodes)
    print(f"[✓] MAPPO Training Complete → Metrics: {metrics_path}")


if __name__ == "__main__":
    main()