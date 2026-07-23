"""
Constrained Markov Decision Process (CMDP) Solvers with Lagrangian Relaxation

This module develops Constrained MDP solvers using Lagrangian relaxation to enforce 
hard risk limits directly inside the RL policy optimization loop, respecting the 
4GB Python RAM quota on Ray.

Optimized for:
- Hard constraint enforcement during training
- 4GB Python RAM quota via streaming batches
- AMD ROCm/DirectML acceleration checks
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc


@dataclass
class CMDPConfig:
    """Configuration for CMDP solver"""
    state_dim: int
    action_dim: int
    constraint_dims: int = 1  # Number of constraints
    lagrangian_lr: float = 0.01
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    gamma: float = 0.99
    constraint_threshold: float = 0.1  # Maximum allowed constraint violation
    max_lagrangian: float = 100.0
    device: Optional[str] = None


class LagrangianMultiplier:
    """Manages Lagrangian multipliers for constraint enforcement"""
    
    def __init__(
        self,
        n_constraints: int,
        initial_value: float = 1.0,
        lr: float = 0.01,
        max_value: float = 100.0
    ):
        self.n_constraints = n_constraints
        self.lr = lr
        self.max_value = max_value
        
        # Initialize multipliers (one per constraint)
        self.multipliers = torch.ones(n_constraints) * initial_value
        
    def update(self, constraint_violations: torch.Tensor) -> torch.Tensor:
        """
        Update Lagrangian multipliers based on constraint violations.
        Uses dual gradient ascent.
        """
        # Gradient ascent on dual function
        gradients = constraint_violations.detach()
        
        self.multipliers = self.multipliers + self.lr * gradients
        
        # Project to non-negative and clip
        self.multipliers = torch.clamp(self.multipliers, 0.0, self.max_value)
        
        return self.multipliers
    
    def get_multipliers(self) -> torch.Tensor:
        """Get current multiplier values"""
        return self.multipliers.clone()
    
    def reset(self):
        """Reset multipliers to initial values"""
        self.multipliers.fill(1.0)


class ConstrainedPolicyNetwork(nn.Module):
    """Policy network with constraint-aware output"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Constraint estimation head (predicts constraint costs)
        self.constraint_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single constraint cost estimate
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(
        self,
        states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning policy, constraint estimate, and value.
        """
        features = self.backbone(states)
        
        policy = self.policy_head(features)
        constraint_cost = self.constraint_head(features)
        value = self.value_head(features)
        
        return policy, constraint_cost, value
    
    def get_action(self, state: torch.Tensor, temperature: float = 1.0) -> int:
        """Sample action from policy"""
        policy, _, _ = self.forward(state.unsqueeze(0))
        
        if temperature != 1.0:
            policy = torch.pow(policy, 1.0 / temperature)
            policy = policy / policy.sum(dim=-1, keepdim=True)
        
        return torch.multinomial(policy.squeeze(0), 1).item()


class CMDPSolver:
    """
    Constrained MDP solver using Lagrangian relaxation.
    Implements primal-dual optimization for constrained RL.
    """
    
    def __init__(self, config: CMDPConfig):
        self.config = config
        
        # Device selection with AMD checks
        self.device = self._select_device(config.device)
        
        # Initialize networks
        self.policy = ConstrainedPolicyNetwork(
            config.state_dim,
            config.action_dim
        ).to(self.device)
        
        # Target network for stability
        self.target_policy = ConstrainedPolicyNetwork(
            config.state_dim,
            config.action_dim
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.policy_lr
        )
        
        self.value_optimizer = torch.optim.Adam(
            list(self.policy.value_head.parameters()),
            lr=config.value_lr
        )
        
        # Lagrangian multipliers
        self.lagrangian = LagrangianMultiplier(
            n_constraints=config.constraint_dims,
            initial_value=1.0,
            lr=config.lagrangian_lr,
            max_value=config.max_lagrangian
        )
        
        # Experience buffer (bounded for memory)
        self.buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 50000
        
        # Memory tracking
        self.memory_used_mb = 0
        self.max_memory_mb = 4096
        
        # Training statistics
        self.constraint_violations_history: List[float] = []
    
    def _select_device(self, requested_device: Optional[str]) -> str:
        """Select best available device with AMD ROCm/DirectML checks."""
        if requested_device:
            return requested_device
        
        if torch.cuda.is_available():
            return 'cuda'
        
        try:
            import torch_directml
            return 'dml'
        except ImportError:
            pass
        
        if torch.version.hip is not None:
            return 'cuda'
        
        return 'cpu'
    
    def _check_memory_quota(self) -> bool:
        """Check if we're within 4GB Python RAM quota."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            self.memory_used_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass
        
        return self.memory_used_mb < self.max_memory_mb * 0.9
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        constraint_cost: float
    ):
        """Store transition with constraint cost"""
        if not self._check_memory_quota():
            # Clear oldest transitions
            self.buffer = self.buffer[-self.max_buffer_size // 2:]
        
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'constraint_cost': constraint_cost
        })
        
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]
    
    def compute_lagrangian_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Lagrangian loss combining reward and constraints.
        L(s,a) = -E[R] + lambda * E[C - threshold]
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        constraint_costs = batch['constraint_costs']
        
        # Get policy output
        policy, _, values = self.policy(states)
        
        # Action probabilities for taken actions
        action_probs = policy.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Policy loss (negative log likelihood weighted by advantages)
        # Simplified: use rewards as advantage proxy
        policy_loss = -(action_probs * rewards).mean()
        
        # Constraint violation
        mean_constraint = constraint_costs.mean()
        violation = mean_constraint - self.config.constraint_threshold
        
        # Lagrangian term
        lagrangian_multipliers = self.lagrangian.get_multipliers().to(self.device)
        lagrangian_term = lagrangian_multipliers[0] * F.relu(violation)
        
        # Total loss
        total_loss = policy_loss + lagrangian_term
        
        return total_loss, violation
    
    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """Perform one training step with constraint enforcement"""
        if len(self.buffer) < batch_size:
            return {'loss': 0.0, 'violation': 0.0}
        
        # Sample mini-batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Prepare batch
        states = torch.FloatTensor(np.array([self.buffer[i]['state'] for i in indices])).to(self.device)
        actions = torch.LongTensor([self.buffer[i]['action'] for i in indices]).to(self.device)
        rewards = torch.FloatTensor([self.buffer[i]['reward'] for i in indices]).to(self.device)
        constraint_costs = torch.FloatTensor([self.buffer[i]['constraint_cost'] for i in indices]).to(self.device)
        
        batch = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'constraint_costs': constraint_costs
        }
        
        # Compute loss
        self.policy_optimizer.zero_grad()
        total_loss, violation = self.compute_lagrangian_loss(batch)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        
        self.policy_optimizer.step()
        
        # Update Lagrangian multipliers
        self.lagrangian.update(violation.unsqueeze(0))
        
        # Track violations
        self.constraint_violations_history.append(violation.item())
        if len(self.constraint_violations_history) > 1000:
            self.constraint_violations_history = self.constraint_violations_history[-1000:]
        
        # Update target network periodically
        self._soft_update_target()
        
        return {
            'loss': total_loss.item(),
            'violation': violation.item(),
            'lagrangian': self.lagrangian.get_multipliers()[0].item()
        }
    
    def _soft_update_target(self, tau: float = 0.005):
        """Soft update of target network"""
        with torch.no_grad():
            for target_param, param in zip(
                self.target_policy.parameters(),
                self.policy.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )
    
    def select_action(
        self,
        state: np.ndarray,
        check_constraints: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action with optional constraint checking.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, constraint_estimate, _ = self.policy(state_tensor)
            
            # Check if action would violate constraints
            action = torch.multinomial(policy.squeeze(0), 1).item()
            
            metadata = {
                'action_probs': policy.squeeze(0).cpu().numpy(),
                'constraint_estimate': constraint_estimate.item(),
                'lagrangian_multiplier': self.lagrangian.get_multipliers()[0].item()
            }
            
            # Apply constraint-based masking if enabled
            if check_constraints and constraint_estimate.item() > self.config.constraint_threshold:
                # Reduce probability of high-constraint actions
                mask = (constraint_estimate > self.config.constraint_threshold).float()
                adjusted_policy = policy * (1.0 - mask * 0.5)
                adjusted_policy = adjusted_policy / adjusted_policy.sum(dim=-1, keepdim=True)
                action = torch.multinomial(adjusted_policy.squeeze(0), 1).item()
                metadata['constraint_masked'] = True
            
            return action, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics"""
        recent_violations = self.constraint_violations_history[-100:] if self.constraint_violations_history else [0.0]
        
        return {
            'buffer_size': len(self.buffer),
            'memory_mb': self.memory_used_mb,
            'mean_violation': np.mean(recent_violations),
            'max_violation': np.max(recent_violations),
            'lagrangian_multiplier': self.lagrangian.get_multipliers()[0].item(),
            'device': self.device
        }
    
    def cleanup(self):
        """Cleanup to maintain memory quota"""
        self.buffer = self.buffer[-self.max_buffer_size // 4:]
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


@ray.remote(max_calls=500)
class DistributedCMDPWorker:
    """Ray-distributed CMDP worker with memory monitoring"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        constraint_threshold: float = 0.1
    ):
        config = CMDPConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            constraint_threshold=constraint_threshold
        )
        self.solver = CMDPSolver(config)
        self.steps_trained = 0
    
    def collect_experience(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> int:
        """Collect experience from trajectories"""
        if not self.solver._check_memory_quota():
            raise MemoryError("Exceeded 4GB Python RAM quota")
        
        count = 0
        for traj in trajectories:
            for t in range(len(traj['states']) - 1):
                self.solver.store_transition(
                    state=traj['states'][t],
                    action=traj['actions'][t],
                    reward=traj['rewards'][t],
                    next_state=traj['states'][t + 1],
                    done=t == len(traj['states']) - 2,
                    constraint_cost=traj.get('constraint_costs', [0.0] * len(traj['states']))[t]
                )
                count += 1
        
        return count
    
    def train(self, batch_size: int = 256) -> Dict[str, float]:
        """Run training step"""
        stats = self.solver.train_step(batch_size)
        self.steps_trained += 1
        return stats
    
    def get_action(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Get action from policy"""
        return self.solver.select_action(state)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        stats = self.solver.get_statistics()
        stats['steps_trained'] = self.steps_trained
        return stats


if __name__ == '__main__':
    import time
    
    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        _system_config={"object_store_memory": 1024*1024*1024}
    )
    
    # Create workers
    workers = [
        DistributedCMDPWorker.remote(state_dim=32, action_dim=6, constraint_threshold=0.1)
        for _ in range(4)
    ]
    
    # Generate fake trajectories
    test_trajectories = []
    for _ in range(10):
        n_steps = 100
        test_trajectories.append({
            'states': np.random.randn(n_steps, 32).astype(np.float32),
            'actions': np.random.randint(0, 6, n_steps),
            'rewards': np.random.randn(n_steps) * 0.1,
            'constraint_costs': np.abs(np.random.randn(n_steps)) * 0.05
        })
    
    # Collect experience
    start = time.time()
    futures = [w.collect_experience.remote(test_trajectories) for w in workers]
    counts = ray.get(futures)
    elapsed = time.time() - start
    
    print(f"Collected {sum(counts)} transitions in {elapsed*1000:.2f}ms")
    
    # Train
    train_futures = [w.train.remote() for w in workers]
    stats = ray.get(train_futures)
    print(f"Training stats: {stats}")
    
    ray.shutdown()
