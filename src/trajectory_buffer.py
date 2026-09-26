# src/trajectory_buffer.py

import torch
import numpy as np

class CTDETrajectoryBuffer:
    """
    Shared Memory Buffer for Centralized Training with Decentralized Execution (CTDE).
    Stores transitions for both actors (Attacker, Defender) and the global state for the Critic.
    """
    def __init__(self):
        self.global_states = []       # S: Complete view for the Centralized Critic
        
        # Attacker Actor Data (pi_atk)
        self.attacker_obs = []        # Local observation (Task + History)
        self.attacker_actions = []    # Token actions (Payload)
        self.attacker_logprobs = []   # Log probs of generated payload
        
        # Defender Actor Data (pi_def)
        self.defender_obs = []        # Local observation (Poisoned Prompt)
        self.defender_actions = []    # Token actions (Tool/Answer)
        self.defender_logprobs = []   # Log probs of generated answer
        
        # Environmental Data
        self.rewards = []             # Attacker rewards
        self.defender_rewards = []    # Defender rewards
        self.values = []              # V(S) predicted by Centralized Critic
        self.dones = []               # Episode termination flag

    def add_step(self, global_state, atk_obs, atk_action, atk_logprob, 
                 def_obs, def_action, def_logprob, reward, value, done, def_reward=None):
        """Records a single multi-agent interaction step."""
        self.global_states.append(global_state)
        self.attacker_obs.append(atk_obs)
        self.attacker_actions.append(atk_action)
        self.attacker_logprobs.append(atk_logprob)
        self.defender_obs.append(def_obs)
        self.defender_actions.append(def_action)
        self.defender_logprobs.append(def_logprob)
        self.rewards.append(reward)
        self.defender_rewards.append(def_reward if def_reward is not None else -reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95, for_defender=False):
        """
        Computes GAE using the Centralized Critic's values.
        Supports both attacker rewards and defender rewards.
        """
        target_rewards = self.defender_rewards if for_defender else self.rewards
        advantages = np.zeros(len(target_rewards), dtype=np.float32)
        last_gae_lam = 0
        
        # Append a bootstrap value of 0 for the end of the trajectory
        values = np.append(self.values, 0)
        if for_defender:
            # For defender, value function sign is oriented to defender returns
            values = -values 
        
        for t in reversed(range(len(target_rewards))):
            next_non_terminal = 1.0 - self.dones[t]
            delta = target_rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + values[:-1]
        
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        if len(adv_tensor) > 1 and float(adv_tensor.std().item()) > 1e-6:
            adv_normalized = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)
        else:
            adv_normalized = adv_tensor
        
        return torch.tensor(returns, dtype=torch.float32), adv_normalized

    def clear(self):
        """Clears buffer after a PPO update step."""
        self.__init__()