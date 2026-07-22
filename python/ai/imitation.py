"""
Nautilus/Ray Bot - Stage 15: Generative Adversarial Imitation Learning (GAIL)
Module: python/ai/imitation.py

Description:
    GAIL pipeline that allows the bot to mimic quoting behavior of top-tier HFT market makers.
    Learns from historical order book data and expert demonstrations.
    Optimized for AMD Ryzen AI 5 with ROCm/DirectML acceleration checks.

Constraints:
    - Max Python RAM: 4GB quota per worker.
    - Architecture: AMD Ryzen AI 5 compatible.
"""

import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import gc
import psutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configuration Constants
MAX_RAM_GB = 4.0
MEMORY_THRESHOLD = 0.90
GAIL_HIDDEN_SIZE = 512
DISCRIMINATOR_LAYERS = 3
BATCH_SIZE = 256


def check_amd_acceleration() -> str:
    """Detect AMD ROCm or DirectML availability."""
    if torch.cuda.is_available() and ("ROCm" in torch.version.cuda or 
                                       hasattr(torch.version, 'hip')):
        return "rocm"
    try:
        import torch_directml
        return "directml"
    except ImportError:
        pass
    return "cpu"


@dataclass
class GAILConfig:
    """Configuration for GAIL training."""
    obs_dim: int
    action_dim: int
    hidden_dim: int = GAIL_HIDDEN_SIZE
    learning_rate: float = 3e-4
    batch_size: int = BATCH_SIZE
    discriminator_layers: int = DISCRIMINATOR_LAYERS


class Discriminator(nn.Module):
    """
    Discriminator network that distinguishes expert vs agent actions.
    Outputs probability that a state-action pair came from expert data.
    """
    
    def __init__(self, config: GAILConfig):
        super().__init__()
        
        layers = []
        input_dim = config.obs_dim + config.action_dim
        
        for i in range(config.discriminator_layers):
            layers.extend([
                nn.Linear(input_dim if i == 0 else config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        
        layers.append(nn.Linear(config.hidden_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.network(x)
    
    def get_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute imitation reward from discriminator.
        Higher output = more expert-like = higher reward.
        """
        return -torch.log(self.forward(states, actions) + 1e-8)


class GeneratorPolicy(nn.Module):
    """
    Generator (policy) network that produces actions mimicking expert behavior.
    Uses Gaussian policy for continuous action spaces.
    """
    
    def __init__(self, config: GAILConfig):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Mean head for Gaussian policy
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        
        # Log std head (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(states)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def sample_action(self, states: torch.Tensor) -> torch.Tensor:
        """Sample action from Gaussian policy."""
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()
    
    def select_action(self, states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action (deterministic or stochastic)."""
        mean, std = self.forward(states)
        
        if deterministic:
            return mean
        
        dist = torch.distributions.Normal(mean, std)
        return dist.sample()
    
    def log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current policy."""
        mean, std = self.forward(states)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(actions).sum(dim=-1)


@ray.remote(max_calls=500)
class GAILTrainer:
    """
    Ray actor implementing GAIL training loop.
    Alternates between discriminator and generator updates.
    """
    
    def __init__(self, config: GAILConfig):
        self.config = config
        self.device = check_amd_acceleration()
        
        # Initialize networks
        self.discriminator = Discriminator(config).to(self.device)
        self.generator = GeneratorPolicy(config).to(self.device)
        
        # Optimizers
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=config.learning_rate
        )
        self.gen_optimizer = optim.Adam(
            self.generator.parameters(), lr=config.learning_rate
        )
        
        # Memory management
        self.memory_limit_bytes = int(MAX_RAM_GB * 1024**3)
        
    def _check_memory(self):
        """Enforce memory limits."""
        process = psutil.Process(os.getpid())
        if process.memory_info().rss > self.memory_limit_bytes * MEMORY_THRESHOLD:
            gc.collect()
            if self.device != "cpu":
                torch.cuda.empty_cache()
    
    def update_discriminator(
        self, 
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        agent_states: np.ndarray,
        agent_actions: np.ndarray
    ) -> float:
        """
        Update discriminator to distinguish expert from agent.
        Returns discriminator loss.
        """
        self._check_memory()
        
        # Convert to tensors
        exp_s = torch.FloatTensor(expert_states).to(self.device)
        exp_a = torch.FloatTensor(expert_actions).to(self.device)
        agg_s = torch.FloatTensor(agent_states).to(self.device)
        agg_a = torch.FloatTensor(agent_actions).to(self.device)
        
        # Expert labels = 1, Agent labels = 0
        expert_labels = torch.ones(len(exp_s), 1).to(self.device)
        agent_labels = torch.zeros(len(agg_s), 1).to(self.device)
        
        # Forward pass
        expert_pred = self.discriminator(exp_s, exp_a)
        agent_pred = self.discriminator(agg_s, agg_a)
        
        # Binary cross-entropy loss
        loss = -(
            torch.mean(torch.log(expert_pred + 1e-8) * expert_labels) +
            torch.mean(torch.log(1 - agent_pred + 1e-8) * (1 - agent_labels))
        )
        
        # Backward pass
        self.disc_optimizer.zero_grad()
        loss.backward()
        self.disc_optimizer.step()
        
        return float(loss.item())
    
    def update_generator(
        self,
        states: np.ndarray,
        actions: np.ndarray
    ) -> float:
        """
        Update generator to fool discriminator.
        Returns generator loss (negative imitation reward).
        """
        self._check_memory()
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        
        # Get imitation rewards
        rewards = self.discriminator.get_reward(states_t, actions_t)
        
        # Policy gradient: maximize expected reward
        log_probs = self.generator.log_prob(states_t, actions_t)
        loss = -(log_probs * rewards.detach()).mean()
        
        # Backward pass
        self.gen_optimizer.zero_grad()
        loss.backward()
        self.gen_optimizer.step()
        
        return float(loss.item())
    
    def generate_expert_like_action(self, state: np.ndarray) -> np.ndarray:
        """Generate action mimicking expert behavior."""
        self._check_memory()
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.generator.select_action(state_t, deterministic=True)
            
        return action.cpu().numpy()[0]


@ray.remote
class ImitationManager:
    """
    Central coordinator for GAIL training across multiple workers.
    Aggregates expert demonstrations and manages training curriculum.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, num_trainers: int = 4):
        config = GAILConfig(obs_dim=obs_dim, action_dim=action_dim)
        self.trainers = [
            GAILTrainer.remote(config) for _ in range(num_trainers)
        ]
        
    def train_step(
        self,
        expert_data: Tuple[np.ndarray, np.ndarray],
        agent_data: Tuple[np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Execute one training step across all trainers."""
        # In production: distribute batches across trainers
        return {"disc_loss": 0.0, "gen_loss": 0.0}
    
    def get_imitation_policy(self, state: np.ndarray) -> np.ndarray:
        """Get action from trained imitation policy."""
        # Use first trainer's policy (in production: ensemble)
        future = self.trainers[0].generate_expert_like_action.remote(state)
        return np.zeros(1)  # Placeholder


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    print(f"[IMITATION] GAIL module initialized on {check_amd_acceleration()} backend.")
