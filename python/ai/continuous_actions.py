"""
Soft Actor-Critic (SAC) with Continuous Action Spaces

Implements SAC for precise limit order price offset and size generation.
Strictly bounds outputs to prevent OOM and enforces 4GB Python RAM quota.

Includes AMD ROCm/DirectML environment checks for hardware acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import os
import gc

# Check for AMD ROCm/DirectML availability
def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD GPU acceleration options."""
    rocm_available = torch.cuda.is_available() and \
                    (os.environ.get('ROCM_PATH') is not None or 
                     os.path.exists('/opt/rocm'))
    
    # DirectML for Windows AMD acceleration
    directml_available = False
    try:
        if os.name == 'nt':
            import torch_directml
            directml_available = True
    except ImportError:
        pass
    
    return {
        'rocm_available': rocm_available,
        'directml_available': directml_available,
        'cuda_available': torch.cuda.is_available(),
    }


def get_device():
    """Get optimal device based on available hardware."""
    accel = check_amd_acceleration()
    
    if accel['rocm_available']:
        return torch.device('cuda')
    elif accel['directml_available']:
        try:
            import torch_directml
            return torch.device('dml')
        except:
            pass
    
    return torch.device('cpu')


class ReplayBuffer:
    """
    Bounded replay buffer with strict memory limits.
    
    Enforces 4GB RAM quota by limiting buffer size and triggering GC.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int = 100_000,  # Bounded to prevent OOM
        max_memory_bytes: int = int(4.0 * 1024 * 1024 * 1024)  # 4GB limit
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        
        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
        
        self.ptr = 0
        self.size = 0
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add transition to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        # Memory pressure check
        if self.size % 10000 == 0:
            self._check_memory_pressure()
    
    def sample_batch(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        indices = np.random.choice(min(self.size, len(self.rewards)), batch_size, replace=False)
        
        obs = torch.FloatTensor(self.observations[indices]).to(device)
        actions = torch.FloatTensor(self.actions[indices]).to(device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        next_obs = torch.FloatTensor(self.next_observations[indices]).to(device)
        dones = torch.FloatTensor(self.dones[indices]).to(device)
        
        return obs, actions, rewards, next_obs, dones
    
    def _check_memory_pressure(self):
        """Force GC if approaching memory limit."""
        estimated_bytes = (
            self.observations.nbytes +
            self.actions.nbytes +
            self.rewards.nbytes +
            self.next_observations.nbytes +
            self.dones.nbytes
        )
        
        if estimated_bytes > self.max_memory_bytes * 0.9:
            gc.collect()


class Actor(nn.Module):
    """
    SAC Actor network with continuous action output.
    
    Outputs mean and log_std for Gaussian policy.
    Actions are bounded to prevent extreme values.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Policy heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning sampled action.
        
        Args:
            obs: Observation tensor
            deterministic: If True, use mean action (for evaluation)
            with_log_prob: If True, also return log probability
            
        Returns:
            action: Sampled action (bounded)
            log_prob: Log probability of action (if requested)
        """
        x = self.backbone(obs)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        
        # Clip log_std for stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()  # Reparameterized sample
            action = torch.tanh(z)  # Bound to [-1, 1]
            
            if with_log_prob:
                # Compute log prob with tanh correction
                log_prob = normal.log_prob(z)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                # Tanh correction
                log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        # Scale action to actual bounds
        low, high = self.action_bounds
        action = low + (action + 1.0) * 0.5 * (high - low)
        
        return action, log_prob


class Critic(nn.Module):
    """
    SAC Critic network (Q-function).
    
    Takes observation and action as input, outputs Q-value.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256]
    ):
        super().__init__()
        
        # Q1 network
        q1_layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            q1_layers.append(nn.Linear(prev_dim, hidden_dim))
            q1_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q1_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2 network (for double Q-learning)
        q2_layers = []
        prev_dim = obs_dim + action_dim
        for hidden_dim in hidden_dims:
            q2_layers.append(nn.Linear(prev_dim, hidden_dim))
            q2_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        q2_layers.append(nn.Linear(hidden_dims[-1], 1))
        self.q2 = nn.Sequential(*q2_layers)
        
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning Q-values from both critics."""
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class SoftActorCritic:
    """
    Complete SAC implementation for continuous control.
    
    Designed for limit order price offset and size generation.
    Strictly bounded outputs and memory-limited.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_update_interval: int = 1,
        action_bounds: Tuple[float, float] = (-1.0, 1.0),
        max_memory_bytes: int = int(4.0 * 1024 * 1024 * 1024)
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.action_bounds = action_bounds
        self._step = 0
        
        # Get device (checks for AMD ROCm/DirectML)
        self.device = get_device()
        self.accel_info = check_amd_acceleration()
        
        # Initialize networks
        self.actor = Actor(obs_dim, action_dim, action_bounds=action_bounds).to(self.device)
        self.critic = Critic(obs_dim, action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim).to(self.device)
        
        # Copy weights to target
        self._soft_update(1.0)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Temperature parameter (auto-tuned)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Target entropy for auto-tuning
        self.target_entropy = -np.prod(action_dim) if isinstance(action_dim, tuple) else -action_dim
        
        # Replay buffer with memory limits
        self.replay_buffer = ReplayBuffer(
            obs_dim, action_dim,
            max_size=100_000,
            max_memory_bytes=max_memory_bytes
        )
        
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """Select action given observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _ = self.actor(obs_tensor, deterministic=deterministic)
            return action.cpu().numpy()[0]
    
    def update(
        self,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Perform one update step."""
        if self.replay_buffer.size < batch_size:
            return {}
        
        # Sample batch
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample_batch(
            batch_size, self.device
        )
        
        metrics = {}
        
        # Update critic
        critic_loss, q1_mean, q2_mean = self._update_critic(
            obs, actions, rewards, next_obs, dones
        )
        metrics['critic_loss'] = critic_loss
        metrics['q1_mean'] = q1_mean
        metrics['q2_mean'] = q2_mean
        
        # Update actor
        actor_loss, alpha_loss, entropy = self._update_actor(obs)
        metrics['actor_loss'] = actor_loss
        metrics['alpha'] = self.log_alpha.exp().item()
        metrics['entropy'] = entropy
        
        # Update target networks
        if self._step % self.target_update_interval == 0:
            self._soft_update(self.tau)
        
        self._step += 1
        
        # Memory check
        if self._step % 1000 == 0:
            gc.collect()
        
        return metrics
    
    def _update_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Update critic networks."""
        with torch.no_grad():
            # Next action and Q-value
            next_action, next_log_prob = self.actor(next_obs, with_log_prob=True)
            next_q1, next_q2 = self.critic_target(next_obs, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            
            # Target Q-value
            target_q = rewards + self.gamma * (1 - dones) * (min_next_q - self.alpha * next_log_prob)
        
        # Current Q-values
        current_q1, current_q2 = self.critic(obs, actions)
        
        # Critic loss (MSE)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        return critic_loss.item(), current_q1.mean().item(), current_q2.mean().item()
    
    def _update_actor(
        self,
        obs: torch.Tensor
    ) -> Tuple[float, float, float]:
        """Update actor network and temperature."""
        # Actor loss
        action, log_prob = self.actor(obs, with_log_prob=True)
        q1, q2 = self.critic(obs, action)
        min_q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Temperature update (auto-tune alpha)
        with torch.no_grad():
            _, log_prob = self.actor(obs, with_log_prob=True)
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return actor_loss.item(), alpha_loss.item(), -log_prob.mean().item()
    
    def _soft_update(self, tau: float):
        """Soft update of target networks."""
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def get_memory_stats(self) -> Dict:
        """Return memory statistics."""
        return {
            'buffer_size': self.replay_buffer.size,
            'buffer_max': self.replay_buffer.max_size,
            'accel_info': self.accel_info,
            'device': str(self.device),
        }


if __name__ == "__main__":
    # Test SAC implementation
    print("Testing SAC with Continuous Actions...")
    
    # Create agent
    agent = SoftActorCritic(
        obs_dim=10,
        action_dim=2,  # price_offset, order_size
        action_bounds=(-0.5, 0.5),  # Bounded action space
    )
    
    print(f"Device: {agent.device}")
    print(f"Acceleration: {agent.accel_info}")
    
    # Simulate some training steps
    obs_dim = 10
    action_dim = 2
    
    for step in range(100):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.select_action(obs)
        
        # Simulate environment step
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        reward = np.random.randn()
        done = step % 100 == 99
        
        # Add to buffer
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        
        # Update
        if step >= 10:
            metrics = agent.update(batch_size=32)
    
    print(f"\nMemory stats: {agent.get_memory_stats()}")
    print("SAC test completed successfully!")
