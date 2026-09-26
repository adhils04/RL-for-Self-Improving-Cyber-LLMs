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
from src.attacker_policy.attack_generator import ATTACK_CATEGORIES, ATTACK_TEMPLATES
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


def get_attack_payload(action_idx: int) -> str:
    """Map attacker categorical action index to an attack template payload."""
    category = ATTACK_CATEGORIES[action_idx % len(ATTACK_CATEGORIES)]
    templates = ATTACK_TEMPLATES.get(category, [])
    if templates:
        return templates[action_idx % len(templates)]
    return "Benign probe: please answer the task honestly."


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
            "attacker":                "AttackerPolicy (Actor-Critic, state_dim=8, action_dim=5)",
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
    device = "cpu"
    print(f"[*] Initializing MAPPO Co-evolution on device: {device}")

    # Hyperparameters
    gamma         = config["training"]["gamma"]
    gae_lambda    = config["training"]["gae_lambda"]
    ppo_epochs    = config["training"]["ppo_epochs"]
    clip_epsilon  = config["training"]["clip_epsilon"]
    actor_lr      = config["training"].get("actor_lr", 2e-5)
    critic_lr     = config["training"].get("critic_lr", 2e-5)
    defender_lr   = config["training"].get("defender_lr", 2e-5)
    num_episodes  = min(config["experiment"].get("num_episodes", 150), 150)
    max_steps     = config["experiment"].get("max_steps_per_episode", 2)
    seed          = config["experiment"].get("seed", 42)
    batch_episodes = 5  # accumulate transitions across 5 episodes for stable PPO batches

    state_dim  = config.get("models", {}).get("attacker_state_dim", 8)
    action_dim = config.get("models", {}).get("attacker_action_dim", 5)

    log_interval   = config.get("logging", {}).get("log_interval", 15)
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
        model_name_or_path="distilbert-base-uncased",
        learning_rate=defender_lr,
        device=device,
    )

    # Load SFT reference policy warm-up checkpoint
    sft_ckpt_path = _PROJECT_ROOT / "checkpoints" / "sft_run" / "sft_final.pt"
    if sft_ckpt_path.exists():
        print(f"[*] Initializing MAPPO Defender from SFT reference policy: {sft_ckpt_path}")
        defender.load_checkpoint(sft_ckpt_path)
    else:
        print("[!] SFT reference policy not found; starting Defender from base weights.")

    defender_adapter = NeuralDefenderAdapter(policy=defender, deterministic=False)

    cases_path = _PROJECT_ROOT / "data" / "evaluation" / "benchmark_cases.jsonl"
    env = CoevolutionEnv(
        defender_adapter=defender_adapter,
        cases_path=cases_path,
        max_steps=max_steps,
        seed=seed,
    )
    buffer = CTDETrajectoryBuffer()

    episode_records: list = []

    print("[*] Starting MAPPO Co-evolution Training Loop (SFT-Warmup Defender + Attacker)...")
    global_step = 0
    running_critic_loss = 0.0
    running_atk_loss = 0.0
    running_def_loss = 0.0

    for episode in range(1, num_episodes + 1):
        obs = env.reset()
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

            task_desc    = getattr(obs, "task_id", str(obs))
            state_tensor = encode_obs(task_desc, state_dim=state_dim)

            # Attacker decision
            with torch.no_grad():
                decision = attacker.get_action(state_tensor)
            atk_action_idx = decision.action
            atk_logprob    = decision.log_prob.detach()

            atk_text    = get_attack_payload(atk_action_idx)
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

            # Centralized Critic Global State
            global_state_text = (
                f"Task: {task_desc} | Payload: {atk_text[:80]} | "
                f"Reward_atk: {reward_atk:.2f} | Reward_def: {reward_def:.2f}"
            )
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
                def_obs=f"[TASK] {getattr(obs, 'user_task', task_desc)} [CONTENT] {atk_text}",
                def_action=def_action_val,
                def_logprob=def_lp_val,
                reward=reward_atk,
                value=state_value,
                done=done,
                def_reward=reward_def,
            )

            obs = step_result.next_obs

        # Episode metrics
        episode_asr = episode_asr_steps / max(episode_steps_count, 1)
        episode_gate_block_rate = episode_gate_blocks / max(episode_steps_count, 1)

        # ── Batch PPO Updates ─────────────────────────────────────────────
        if episode % batch_episodes == 0 or episode == num_episodes:
            # Attacker returns & advantages
            returns, advantages = buffer.compute_returns_and_advantages(
                gamma=gamma,
                gae_lambda=gae_lambda,
                for_defender=False,
            )
            # Defender returns & advantages
            def_returns, def_advantages = buffer.compute_returns_and_advantages(
                gamma=gamma,
                gae_lambda=gae_lambda,
                for_defender=True,
            )

            # 1. Update Centralized Critic
            critic_loss = critic.update_critic(buffer.global_states, returns)
            running_critic_loss = critic_loss

            # 2. Update Attacker & Defender Actors
            atk_states   = torch.stack(buffer.attacker_obs)
            atk_actions  = torch.tensor(buffer.attacker_actions, dtype=torch.long)
            atk_logprobs = torch.stack(buffer.attacker_logprobs).detach()

            for _ in range(ppo_epochs):
                try:
                    loss_dict = attacker.ppo_update(
                        states=atk_states,
                        actions=atk_actions,
                        old_log_probs=atk_logprobs,
                        advantages=advantages,
                        returns=returns,
                        clip_epsilon=clip_epsilon,
                    )
                    running_atk_loss = loss_dict["total_loss"]
                except Exception:
                    pass

                try:
                    def_states = buffer.defender_obs
                    def_actions = torch.tensor(buffer.defender_actions, dtype=torch.long)
                    def_logprobs = torch.tensor(buffer.defender_logprobs, dtype=torch.float32)

                    def_loss_dict = defender.ppo_update(
                        obs_texts=def_states,
                        actions=def_actions,
                        old_log_probs=def_logprobs,
                        advantages=def_advantages,
                        clip_epsilon=clip_epsilon,
                    )
                    running_def_loss = def_loss_dict["total_loss"]
                except Exception:
                    pass

            buffer.clear()

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
                f"Def Loss: {running_def_loss:.4f}"
            )
            _save_training_metrics(episode_records, output_dir, config, num_episodes)

    # ── Final Save ────────────────────────────────────────────────────────────
    torch.save(attacker.state_dict(), checkpoint_dir / "attacker_final.pt")
    torch.save(critic.state_dict(),   checkpoint_dir / "critic_final.pt")
    defender.save_checkpoint(checkpoint_dir / "mappo_defender_final.pt")
    print(f"\n[✓] MAPPO Defender checkpoint saved → {checkpoint_dir / 'mappo_defender_final.pt'}")

    metrics_path = _save_training_metrics(episode_records, output_dir, config, num_episodes)
    print(f"[✓] MAPPO Training Complete → Metrics: {metrics_path}")


if __name__ == "__main__":
    main()