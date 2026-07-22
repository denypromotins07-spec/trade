"""
Nautilus/Ray Bot - Stage 15: Hindsight Experience Replay (HER)
Module: python/ai/hindsight.py

Description:
    Hindsight Experience Replay buffers that relabel failed limit order placements
    as successful queue-positioning maneuvers, improving sample efficiency.
    Optimized for AMD Ryzen AI 5 with ROCm/DirectML acceleration checks.

Constraints:
    - Max Python RAM: 4GB quota per worker.
    - Architecture: AMD Ryzen AI 5 compatible.
"""

import ray
import torch
import numpy as np
import os
import gc
import psutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random

# Configuration Constants
MAX_RAM_GB = 4.0
MEMORY_THRESHOLD = 0.90
DEFAULT_BUFFER_SIZE = 100000
HER_K_SAMPLES = 4  # Number of hindsight samples per transition


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
class Transition:
    """Represents a single experience transition."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    goal: np.ndarray
    done: bool
    timestamp_ns: int = field(default_factory=lambda: 0)


@dataclass
class HERTransition(Transition):
    """Transition with hindsight relabeling metadata."""
    original_goal: np.ndarray = field(default_factory=lambda: np.array([]))
    relabeled: bool = False
    strategy: str = "none"  # 'future', 'random', 'last', 'none'


@ray.remote(max_calls=500)
class HERBuffer:
    """
    Hindsight Experience Replay buffer with goal relabeling.
    Converts failures into learning opportunities by changing the goal.
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        goal_dim: int,
        capacity: int = DEFAULT_BUFFER_SIZE
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.capacity = capacity
        
        self.device = check_amd_acceleration()
        
        # Pre-allocate circular buffers
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.original_goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.relabels = np.zeros(capacity, dtype=bool)
        self.strategies = np.empty(capacity, dtype=object)
        
        self.ptr = 0
        self.size = 0
        
        # Memory management
        self.memory_limit_bytes = int(MAX_RAM_GB * 1024**3)
        
    def _check_memory(self):
        """Enforce memory limits."""
        process = psutil.Process(os.getpid())
        if process.memory_info().rss > self.memory_limit_bytes * MEMORY_THRESHOLD:
            gc.collect()
            if self.device != "cpu":
                torch.cuda.empty_cache()
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool,
        timestamp_ns: int = 0
    ):
        """Store a transition in the buffer."""
        self._check_memory()
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.goals[self.ptr] = goal
        self.original_goals[self.ptr] = goal.copy()
        self.dones[self.ptr] = done
        self.relabels[self.ptr] = False
        self.strategies[self.ptr] = "none"
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _compute_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Extract achieved goal from state.
        For limit orders: actual fill price vs target price.
        """
        # In trading context: achieved goal could be actual execution price
        # Simplified: use state features directly as achieved goal
        return state[:self.goal_dim] if len(state) >= self.goal_dim else state
    
    def _relabel_transition(
        self, 
        idx: int, 
        strategy: str = "future"
    ) -> HERTransition:
        """
        Relabel a transition with a new goal using HER strategy.
        
        Strategies:
            - 'future': Use goal from future state in same episode
            - 'random': Use random goal from buffer
            - 'last': Use final state as goal
        """
        if strategy == "future":
            # Sample a future index from same episode
            future_idx = random.randint(idx, min(idx + 50, self.size - 1))
            new_goal = self._compute_achieved_goal(self.next_states[future_idx])
            
        elif strategy == "random":
            rand_idx = random.randint(0, self.size - 1)
            new_goal = self._compute_achieved_goal(self.next_states[rand_idx])
            
        elif strategy == "last":
            # Use the last state's achieved goal
            last_idx = (self.ptr - 1) % self.capacity
            new_goal = self._compute_achieved_goal(self.next_states[last_idx])
            
        else:
            new_goal = self.goals[idx]
        
        # Compute new reward based on relabeled goal
        # Distance-based reward: closer to goal = higher reward
        achieved = self._compute_achieved_goal(self.next_states[idx])
        distance = np.linalg.norm(achieved - new_goal)
        new_reward = -distance  # Negative distance as reward
        
        # Create relabeled transition
        return HERTransition(
            state=self.states[idx],
            action=self.actions[idx],
            reward=float(new_reward),
            next_state=self.next_states[idx],
            goal=new_goal,
            done=self.dones[idx],
            original_goal=self.original_goals[idx].copy(),
            relabeled=True,
            strategy=strategy
        )
    
    def sample(self, batch_size: int, her_ratio: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Sample a batch with HER relabeling.
        
        Args:
            batch_size: Number of transitions to sample
            her_ratio: Fraction of transitions to relabel
            
        Returns:
            Dictionary of batch arrays
        """
        self._check_memory()
        
        if self.size == 0:
            return {}
        
        indices = np.random.randint(0, self.size, batch_size)
        
        # Determine which samples to relabel
        her_mask = np.random.random(batch_size) < her_ratio
        
        # Build batch
        batch = {
            'state': self.states[indices],
            'action': self.actions[indices],
            'next_state': self.next_states[indices],
            'original_goal': self.original_goals[indices],
            'done': self.dones[indices],
            'her_mask': her_mask
        }
        
        # Apply HER relabeling
        rewards = self.rewards[indices].copy()
        goals = self.goals[indices].copy()
        
        for i, idx in enumerate(indices):
            if her_mask[i]:
                strategy = random.choice(['future', 'random', 'last'])
                relabeled = self._relabel_transition(idx, strategy)
                rewards[i] = relabeled.reward
                goals[i] = relabeled.goal
        
        batch['reward'] = rewards
        batch['goal'] = goals
        
        return batch
    
    def get_priority_indices(self, priorities: np.ndarray, k: int) -> np.ndarray:
        """Get indices of top-k highest priority transitions."""
        if self.size == 0:
            return np.array([], dtype=np.int64)
        valid_priorities = priorities[:self.size]
        return np.argsort(valid_priorities)[-k:]
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.goals.fill(0)
        self.original_goals.fill(0)
        self.dones.fill(False)
        self.relabels.fill(False)


@ray.remote
class HERManager:
    """
    Central coordinator for HER across multiple RL agents.
    Manages multiple buffers for different strategies/goals.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: int,
        num_buffers: int = 4
    ):
        self.buffers = [
            HERBuffer.remote(obs_dim, action_dim, goal_dim)
            for _ in range(num_buffers)
        ]
        self.total_transitions = 0
        
    def store_transition(
        self,
        buffer_idx: int,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        goal: np.ndarray,
        done: bool
    ):
        """Store transition in specified buffer."""
        self.buffers[buffer_idx].store.remote(
            state, action, reward, next_state, goal, done
        )
        self.total_transitions += 1
    
    def sample_batch(
        self, 
        batch_size: int, 
        her_ratio: float = 0.8
    ) -> Dict[str, np.ndarray]:
        """Sample batch from random buffer."""
        buffer_idx = random.randint(0, len(self.buffers) - 1)
        future = self.buffers[buffer_idx].sample.remote(batch_size, her_ratio)
        return {}  # Placeholder
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_transitions": self.total_transitions,
            "num_buffers": len(self.buffers),
            "device": check_amd_acceleration()
        }


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    print(f"[HER] Hindsight Experience Replay initialized on {check_amd_acceleration()} backend.")
