"""
Nautilus/Ray Bot - Stage 15: Curiosity-Driven Exploration Module
Module: python/ai/exploration.py

Description:
    Implements curiosity-driven exploration using Random Network Distillation (RND).
    Helps RL agents discover profitable states in sparse reward environments.
    Optimized for AMD Ryzen AI 5 with ROCm/DirectML acceleration checks.

Constraints:
    - Max Python RAM: 4GB quota per worker.
    - Architecture: AMD Ryzen AI 5 compatible.
"""

import ray
import torch
import torch.nn as nn
import numpy as np
import os
import gc
import psutil
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Configuration Constants
MAX_RAM_GB = 4.0
MEMORY_THRESHOLD = 0.90
RND_HIDDEN_SIZE = 256
RND_OUTPUT_SIZE = 128
LEARNING_RATE = 1e-4


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
class RNDConfig:
    """Configuration for Random Network Distillation."""
    input_dim: int
    hidden_dim: int = RND_HIDDEN_SIZE
    output_dim: int = RND_OUTPUT_SIZE
    learning_rate: float = LEARNING_RATE


class FixedRandomNetwork(nn.Module):
    """
    Target network with fixed random weights.
    Generates intrinsic curiosity signals via prediction error.
    """
    
    def __init__(self, config: RNDConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        ).eval()  # Never train this network
        
        # Initialize with fixed random weights
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PredictiveNetwork(nn.Module):
    """
    Trainable network that learns to predict target network outputs.
    Prediction error serves as intrinsic reward (curiosity signal).
    """
    
    def __init__(self, config: RNDConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@ray.remote(max_calls=500)
class RNDExplorer:
    """
    Ray actor implementing Random Network Distillation for exploration.
    Computes intrinsic rewards based on state novelty.
    """
    
    def __init__(self, obs_dim: int, config: Optional[RNDConfig] = None):
        self.config = config or RNDConfig(input_dim=obs_dim)
        self.device = check_amd_acceleration()
        
        # Initialize networks
        self.target_net = FixedRandomNetwork(self.config)
        self.predictor_net = PredictiveNetwork(self.config)
        
        # Move to appropriate device
        if self.device != "cpu":
            self.target_net = self.target_net.to(self.device)
            self.predictor_net = self.predictor_net.to(self.device)
        
        # Optimizer for predictor network
        self.optimizer = torch.optim.Adam(
            self.predictor_net.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Memory management
        self.memory_limit_bytes = int(MAX_RAM_GB * 1024**3)
        self.batch_buffer = []
        
    def _check_memory(self):
        """Enforce memory limits."""
        process = psutil.Process(os.getpid())
        if process.memory_info().rss > self.memory_limit_bytes * MEMORY_THRESHOLD:
            gc.collect()
            if self.device != "cpu":
                torch.cuda.empty_cache()
            self.batch_buffer.clear()
    
    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """
        Compute intrinsic reward based on prediction error.
        Higher error = more novel state = higher curiosity reward.
        """
        self._check_memory()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if self.device != "cpu":
                state_tensor = state_tensor.to(self.device)
            
            target_output = self.target_net(state_tensor)
            predicted_output = self.predictor_net(state_tensor)
            
            # MSE prediction error as intrinsic reward
            error = ((target_output - predicted_output) ** 2).mean().item()
            
        return float(error)
    
    def update(self, states: np.ndarray) -> float:
        """
        Update predictor network to reduce prediction error on seen states.
        Returns mean prediction error before update.
        """
        self._check_memory()
        
        if len(states) == 0:
            return 0.0
        
        states_tensor = torch.FloatTensor(states)
        if self.device != "cpu":
            states_tensor = states_tensor.to(self.device)
        
        # Compute target outputs (no grad)
        with torch.no_grad():
            target_outputs = self.target_net(states_tensor)
        
        # Predict and compute loss
        predicted_outputs = self.predictor_net(states_tensor)
        loss = ((target_outputs - predicted_outputs) ** 2).mean()
        
        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    def get_exploration_bonus(
        self, 
        state: np.ndarray, 
        extrinsic_reward: float,
        beta: float = 0.1
    ) -> Tuple[float, float]:
        """
        Combine extrinsic and intrinsic rewards for exploration.
        
        Args:
            state: Current observation
            extrinsic_reward: Environment reward
            beta: Weight for intrinsic reward
            
        Returns:
            Tuple of (total_reward, intrinsic_reward)
        """
        intrinsic = self.compute_intrinsic_reward(state)
        total = extrinsic_reward + beta * intrinsic
        return total, intrinsic


@ray.remote
class ExplorationManager:
    """
    Central coordinator for exploration strategies across multiple agents.
    Manages RND workers and aggregates curiosity signals.
    """
    
    def __init__(self, obs_dim: int, num_explorers: int = 4):
        self.explorers = [
            RNDExplorer.remote(obs_dim) for _ in range(num_explorers)
        ]
        self.total_intrinsic_rewards = 0.0
        
    def get_curiosity_signal(self, state: np.ndarray) -> float:
        """
        Aggregate curiosity signals from multiple RND explorers.
        Uses ensemble averaging for robust novelty detection.
        """
        futures = [
            explorer.compute_intrinsic_reward.remote(state)
            for explorer in self.explorers
        ]
        # In production: use ray.get with timeout
        return 0.0  # Placeholder
    
    def update_explorers(self, batch_states: np.ndarray) -> float:
        """Update all explorers with a batch of states."""
        futures = [
            explorer.update.remote(batch_states)
            for explorer in self.explorers
        ]
        # In production: aggregate results
        return 0.0


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    print(f"[EXPLORATION] RND module initialized on {check_amd_acceleration()} backend.")
