"""neural_defender.py

Neural Defender Policy for Multi-Agent PPO (MAPPO).
Uses DistilBERT sequence classification backbone to produce action probabilities
and updates parameters via PPO clipped surrogate loss.

Action space:
    0: Benign / Allow (process the user task, answer benignly)
    1: Resist / Refuse (block injection attempt, enforce security boundary)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class DefenderDecision:
    action: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    prob_resist: float


class NeuralDefenderPolicy(nn.Module):
    """Neural policy network for the Defender agent in MAPPO."""

    def __init__(
        self,
        model_name_or_path: str = "distilbert-base-uncased",
        learning_rate: float = 2e-5,
        device: Optional[str] = None,
        max_length: int = 128,
    ):
        super().__init__()
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Compute action logits for a batch of text observations."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**enc)
        return outputs.logits

    def get_action(
        self,
        user_task: str,
        untrusted_content: str = "",
        deterministic: bool = False,
        explore_eps: float = 0.0,
    ) -> DefenderDecision:
        """Sample or select an action given the current observation."""
        self.model.eval()
        obs_text = f"[TASK] {user_task} [CONTENT] {untrusted_content}".strip()

        with torch.no_grad():
            logits = self.forward([obs_text])
            dist = Categorical(logits=logits)
            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).squeeze()
            elif explore_eps > 0.0:
                mix_probs = (1.0 - explore_eps) * probs + explore_eps * 0.5
                behavior_dist = Categorical(probs=mix_probs)
                action = int(behavior_dist.sample().item())
                log_prob = behavior_dist.log_prob(torch.tensor(action, device=self.device)).squeeze()
            else:
                action = int(dist.sample().item())
                log_prob = dist.log_prob(torch.tensor(action, device=self.device)).squeeze()

            entropy = dist.entropy().squeeze()
            prob_resist = float(probs[0, 1].item())

        return DefenderDecision(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            prob_resist=prob_resist,
        )

    def evaluate_actions(
        self,
        obs_texts: List[str],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities and entropy of actions under current policy."""
        logits = self.forward(obs_texts)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.to(self.device))
        entropy = dist.entropy()
        return log_probs, entropy

    def ppo_update(
        self,
        obs_texts: List[str],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        beta_kl: float = 0.05,
        ref_logits: Optional[torch.Tensor] = None,
        kl_weights: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Perform one PPO policy update on Defender parameters."""
        self.model.train()

        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages
        if len(advantages) > 1 and float(advantages.std().item()) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits = self.forward(obs_texts)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Ratio r_t(theta) = exp(new_log_prob - old_log_prob)
        ratios = torch.exp(torch.clamp(new_log_probs - old_log_probs, -20.0, 20.0))

        # Clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        if ref_logits is not None:
            ref_logits_dev = ref_logits.to(self.device)
            ref_probs = torch.softmax(ref_logits_dev, dim=-1)
            ref_log_probs = torch.log_softmax(ref_logits_dev, dim=-1)
            cur_log_probs = torch.log_softmax(logits, dim=-1)
            per_sample_kl = (ref_probs * (ref_log_probs - cur_log_probs)).sum(dim=-1)
            if kl_weights is not None:
                weights_dev = kl_weights.to(self.device)
                kl_div = (per_sample_kl * weights_dev).mean()
            else:
                kl_div = per_sample_kl.mean()
        else:
            # Non-negative trust-region quadratic penalty when no frozen ref logits are given
            kl_div = 0.5 * ((new_log_probs - old_log_probs) ** 2).mean()

        entropy_bonus = -entropy_coeff * entropy.mean()
        total_loss = policy_loss + beta_kl * kl_div + entropy_bonus

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.mean().item()),
            "kl_div": float(kl_div.item()),
            "total_loss": float(total_loss.item()),
        }

    def save_checkpoint(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(cpu_state, p)

    def load_checkpoint(self, path: Path | str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
