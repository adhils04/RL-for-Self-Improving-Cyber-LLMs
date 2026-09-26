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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            torch_dtype=torch.float32,
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
    ) -> dict[str, float]:
        """Perform one PPO policy update on Defender parameters."""
        self.model.train()

        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)

        # Normalize advantages
        if len(advantages) > 1 and float(advantages.std().item()) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        new_log_probs, entropy = self.evaluate_actions(obs_texts, actions)

        # Ratio r_t(theta) = exp(new_log_prob - old_log_prob)
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Approximate KL divergence from initial policy to retain benign task utility
        kl_div = torch.clamp((old_log_probs - new_log_probs).mean(), min=0.0)
        entropy_bonus = -entropy_coeff * entropy.mean()
        total_loss = policy_loss + beta_kl * kl_div + entropy_bonus

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.mean().item()),
            "total_loss": float(total_loss.item()),
        }

    def save_checkpoint(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p)

    def load_checkpoint(self, path: Path | str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
