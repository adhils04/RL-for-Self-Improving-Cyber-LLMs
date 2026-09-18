# scripts/run_mappo_coevolution.py

import hashlib
import sys
from pathlib import Path

import torch
import yaml

# ── Project root (absolute, CWD-independent) ──────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_PROJECT_ROOT))

# Clean imports prefixed with src. for VS Code resolution
from src.centralized_critic import CentralizedCritic
from src.attacker_policy.mappo_attacker import AttackerPolicy
from src.attacker_policy.attack_generator import ATTACK_CATEGORIES, ATTACK_TEMPLATES
from src.defender_policy.mappo_defender import defender_update_step
from src.coevolution_env import CoevolutionEnv
from src.evaluation_metrics.wandb_logger import CoevolutionWandbLogger
from src.trajectory_buffer import CTDETrajectoryBuffer


# ── Observation encoder ───────────────────────────────────────────────────────

def encode_obs(text: str, state_dim: int = 8) -> torch.Tensor:
    """Hash-encode a text observation into a fixed-size float tensor.

    The AttackerPolicy is a lightweight MLP that expects a numeric tensor of
    shape (state_dim,). Since the raw observation is a string (task_id), we
    project it to a deterministic float vector via MD5 so the network always
    receives the same representation for the same task.
    """
    digest = hashlib.md5(text.encode()).digest()           # 16 bytes
    vec = [b / 255.0 for b in digest[:state_dim]]
    if len(vec) < state_dim:
        vec = vec + [0.0] * (state_dim - len(vec))
    return torch.tensor(vec, dtype=torch.float32)


def get_attack_payload(action_idx: int) -> str:
    """Map attacker categorical action index to an attack template payload."""
    category = ATTACK_CATEGORIES[action_idx % len(ATTACK_CATEGORIES)]
    templates = ATTACK_TEMPLATES.get(category, [])
    if templates:
        return templates[0]
    return "Benign probe: please answer the task honestly."


# ── Config loader (CWD-independent) ──────────────────────────────────────────

def load_config(config_path=None):
    """Load YAML config; defaults to configs/mappo_config.yaml relative to project root."""
    path = Path(config_path) if config_path else _PROJECT_ROOT / "configs" / "mappo_config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Main training loop ────────────────────────────────────────────────────────

def main(config_path=None):
    config = load_config(config_path)

    # Device selection with graceful fallback
    cfg_device = config.get("models", {}).get("device", "auto")
    if cfg_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif cfg_device == "auto" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        if cfg_device == "cuda":
            print("[!] CUDA requested in config but not available. Falling back to CPU.")

    print(f"[*] Initializing MAPPO Co-evolution on device: {device}")

    # ── Hyperparameters from config ───────────────────────────────────────────
    gamma         = config["training"]["gamma"]
    gae_lambda    = config["training"]["gae_lambda"]
    ppo_epochs    = config["training"]["ppo_epochs"]
    clip_epsilon  = config["training"]["clip_epsilon"]
    max_grad_norm = config["training"]["max_grad_norm"]
    actor_lr      = config["training"]["actor_lr"]
    critic_lr     = config["training"]["critic_lr"]
    num_episodes  = config["experiment"]["num_episodes"]
    max_steps     = config["experiment"]["max_steps_per_episode"]
    seed          = config["experiment"]["seed"]

    state_dim     = config.get("models", {}).get("attacker_state_dim", 8)
    action_dim    = config.get("models", {}).get("attacker_action_dim", 5)

    log_interval   = config.get("logging", {}).get("log_interval", 20)
    output_dir     = _PROJECT_ROOT / config.get("logging", {}).get("output_dir", "reports")
    wandb_mode     = config.get("logging", {}).get("wandb_mode", "disabled")
    checkpoint_dir = _PROJECT_ROOT / config.get("logging", {}).get("checkpoint_dir", "checkpoints/mappo_run")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize components ─────────────────────────────────────────────────
    env = CoevolutionEnv(
        max_steps=max_steps,
        seed=seed,
    )

    buffer = CTDETrajectoryBuffer()

    critic = CentralizedCritic(
        model_name_or_path=config["models"]["critic_model"],
        device=device,
        lr=critic_lr,
    )

    attacker = AttackerPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=actor_lr,
    )

    # Initialize W&B logger
    logger = CoevolutionWandbLogger(
        output_dir=output_dir,
        config_path=_PROJECT_ROOT / "configs" / "wandb_config.yaml",
        mode=wandb_mode,
    )
    logger.init(tags=["review2", "coevolution"])

    # ── Training loop ─────────────────────────────────────────────────────────
    print("[*] Starting Co-evolution Training Loop...")
    global_step = 0

    for episode in range(1, num_episodes + 1):
        obs = env.reset()                  # returns DefenderObservation
        episode_reward_atk = 0.0
        episode_reward_def = 0.0
        done = False

        # ── Roll-out phase ────────────────────────────────────────────────────
        while not done:
            global_step += 1
            # Extract a text description from the DefenderObservation
            task_desc = getattr(obs, "task_id", str(obs))

            # A. Encode text obs → fixed-size numeric tensor
            state_tensor = encode_obs(task_desc, state_dim=state_dim)

            # B. Attacker action selection
            decision      = attacker.get_action(state_tensor)
            atk_action_idx = decision.action               # int (0–4)
            atk_logprob    = decision.log_prob             # torch.Tensor scalar

            # C. Map action index → injection string using unified attack categories & templates
            atk_text = get_attack_payload(atk_action_idx)

            # D. Environment step — Defender handles the injection internally
            step_result = env.step(atk_text)

            # E. Rewards and status
            reward_atk = step_result.reward_attacker
            reward_def = step_result.reward_defender
            done       = step_result.done

            episode_reward_atk += reward_atk
            episode_reward_def += reward_def

            # F. Centralized Critic evaluates the global state
            global_state_text = (
                f"Task: {task_desc} | Payload: {atk_text[:80]} | "
                f"Reward_atk: {reward_atk:.2f} | Reward_def: {reward_def:.2f}"
            )
            with torch.no_grad():
                state_value = critic([global_state_text]).item()

            # G. Store step in CTDE Trajectory Buffer
            buffer.add_step(
                global_state=global_state_text,
                atk_obs=state_tensor,
                atk_action=atk_action_idx,
                atk_logprob=atk_logprob,
                def_obs=task_desc,
                def_action=atk_text,
                def_logprob=0.0,
                reward=reward_atk,
                value=state_value,
                done=done,
            )

            # Log step to logger
            logger.log_step(
                step_result=step_result,
                global_step=global_step,
                critic_loss=0.0,
            )

            # Advance observation for next step
            obs = step_result.next_obs

        # ── End-of-episode: compute returns & advantages ──────────────────────
        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # ── Update Centralized Critic ─────────────────────────────────────────
        critic_loss = critic.update_critic(buffer.global_states, returns)

        # ── Convert buffer lists to tensors for attacker ─────────────────────
        atk_states   = torch.stack(buffer.attacker_obs)                          # (T, state_dim)
        atk_actions  = torch.tensor(buffer.attacker_actions, dtype=torch.long)   # (T,)
        atk_logprobs = torch.stack(buffer.attacker_logprobs)                     # (T,)

        atk_loss = 0.0
        def_loss = 0.0

        # ── PPO update epochs ─────────────────────────────────────────────────
        for _ in range(ppo_epochs):
            # Attacker policy update
            try:
                loss_dict = attacker.ppo_update(
                    states=atk_states,
                    actions=atk_actions,
                    old_log_probs=atk_logprobs,
                    advantages=advantages,
                    returns=returns,
                    clip_epsilon=clip_epsilon,
                )
                atk_loss = loss_dict["total_loss"]
            except Exception as e:
                print(f"[!] Attacker update failed: {e}")
                atk_loss = 0.0

            # Defender update step
            try:
                adv_list        = advantages.tolist()
                ret_list        = returns.tolist()
                def_logprob_list = [
                    float(lp) for lp in buffer.defender_logprobs
                ]
                def_loss_output = defender_update_step(
                    advantages=adv_list,
                    log_ratios=[0.0] * len(adv_list),
                    current_logprobs=def_logprob_list,
                    ref_logprobs=[0.0] * len(def_logprob_list),
                    value_estimates=list(buffer.values),
                    returns=ret_list,
                )
                def_loss = def_loss_output.total_loss
            except Exception as e:
                print(f"[!] Defender update failed: {e}")
                def_loss = 0.0

        # ── Log Episode Metrics ───────────────────────────────────────────────
        try:
            logger.log_metrics({
                "episode":                 episode,
                "episode_reward_attacker": episode_reward_atk,
                "episode_reward_defender": episode_reward_def,
                "critic_loss":             critic_loss,
                "attacker_loss":           atk_loss,
                "defender_loss":           def_loss,
            }, step=global_step)
        except Exception as e:
            print(f"[!] Logger metrics failed: {e}")

        # ── Progress and Checkpointing ────────────────────────────────────────
        if episode % log_interval == 0 or episode == num_episodes:
            print(
                f"[Ep {episode:>4}/{num_episodes}] "
                f"Atk Reward: {episode_reward_atk:+.2f} | "
                f"Def Reward: {episode_reward_def:+.2f} | "
                f"Critic Loss: {critic_loss:.4f} | "
                f"Atk Loss: {atk_loss:.4f} | "
                f"Def Loss: {def_loss:.4f}"
            )
            # Save periodic checkpoints
            try:
                torch.save(attacker.state_dict(), checkpoint_dir / f"attacker_ep{episode}.pt")
                torch.save(critic.state_dict(), checkpoint_dir / f"critic_ep{episode}.pt")
            except Exception as e:
                print(f"[!] Checkpoint save failed: {e}")

        buffer.clear()

    # ── Final checkpoints and finish ──────────────────────────────────────────
    try:
        torch.save(attacker.state_dict(), checkpoint_dir / "attacker_final.pt")
        torch.save(critic.state_dict(), checkpoint_dir / "critic_final.pt")
    except Exception as e:
        print(f"[!] Final checkpoint save failed: {e}")

    logger.finish()
    print(f"[✓] MAPPO Co-evolution Training Complete. Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()