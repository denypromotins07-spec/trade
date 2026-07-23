"""
python/ai/continuous_actions.py

Soft Actor-Critic (SAC) with Continuous Action Spaces

Implements SAC for precise limit order price offset and size generation.
Actions are strictly bounded to prevent OOM and ensure valid order parameters.
Optimized for AMD Ryzen AI 5 with ROCm/DirectML acceleration checks.

Memory Constraint: Network sizes bounded, gradient clipping enforced.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import os


def check_amd_acceleration() -> Dict[str, bool]:
    """Detect AMD ROCm/DirectML availability for PyTorch acceleration."""
    result = {
        "cuda": torch.cuda.is_available(),
        "rocm": False,
        "directml": False,
        "cpu": True,
    }
    
    if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
        result["rocm"] = True
    
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path and os.path.exists(rocm_path):
        result["rocm"] = True
    
    if os.name == 'nt':
        try:
            import torch_directml
            result["directml"] = True
        except ImportError:
            pass
    
    return result


@dataclass
class SACConfig:
    """Configuration for SAC continuous action space."""
    state_dim: int
    action_dim: int  # 2: [price_offset, order_size]
    hidden_dim: int = 256
    max_action_price_offset: float = 0.02  # Max 2% from mid price
    max_action_size: float = 1.0  # Normalized max order size
    min_action_size: float = 0.01  # Min 1% of portfolio
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: Optional[float] = None
    max_memory_gb: float = 4.0  # Python RAM quota


class BoundedContinuousAction(nn.Module):
    """Neural network for bounded continuous action output."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class ContinuousCritic(nn.Module):
    """Q-network critic for continuous actions."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class SoftActorCritic:
    """Soft Actor-Critic agent with continuous action spaces."""
    
    def __init__(self, config: SACConfig):
        self.config = config
        self.acceleration = check_amd_acceleration()
        self.device = self._select_device()
        
        if config.target_entropy is None:
            config.target_entropy = -np.prod(config.action_dim)
        
        self.actor = BoundedContinuousAction(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        self.critic = ContinuousCritic(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        self.critic_target = ContinuousCritic(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        self._soft_update(1.0)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        
        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.learning_rate
        )
        
        self._check_memory_usage()
    
    def _select_device(self) -> str:
        if self.acceleration["rocm"]:
            return "cuda"
        elif self.acceleration["directml"]:
            return "privateuseone"
        elif self.acceleration["cuda"]:
            return "cuda"
        return "cpu"
    
    def _check_memory_usage(self) -> None:
        import psutil
        process = psutil.Process()
        current_gb = process.memory_info().rss / (1024 ** 3)
        
        if current_gb > self.config.max_memory_gb * 0.8:
            print(f"Warning: Memory at {current_gb:.2f}GB, approaching {self.config.max_memory_gb}GB limit")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action(state_tensor, deterministic)
            scaled_action = self._scale_action(action.squeeze(0))
            return scaled_action.cpu().numpy(), log_prob.item()
    
    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        price_offset = action[..., 0] * self.config.max_action_price_offset
        order_size = (
            (action[..., 1] + 1) / 2 *
            (self.config.max_action_size - self.config.min_action_size) +
            self.config.min_action_size
        )
        return torch.stack([price_offset, order_size], dim=-1)
    
    def update(
        self, 
        replay_buffer: List[Tuple],
        batch_size: int = 256
    ) -> Dict[str, float]:
        if len(replay_buffer) < batch_size:
            return {}
        
        indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
        batch = [replay_buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        actor_loss = self._update_actor(states)
        alpha_loss = self._update_temperature(states)
        
        self._soft_update(self.config.tau)
        self._check_memory_usage()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item() if alpha_loss is not None else 0.0,
            "alpha": self.log_alpha.exp().item()
        }
    
    def _update_critic(
        self, states: torch.Tensor, actions: torch.Tensor,
        rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_states)
            next_scaled_action = self._scale_action(next_action)
            target_q1, target_q2 = self.critic_target(next_states, next_scaled_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.config.gamma * (1 - dones) * (
                target_q - self.log_alpha.exp() * next_log_prob
            )
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss
    
    def _update_actor(self, states: torch.Tensor) -> torch.Tensor:
        action, log_prob = self.actor.get_action(states)
        scaled_action = self._scale_action(action)
        q1, q2 = self.critic(states, scaled_action)
        q = torch.min(q1, q2)
        actor_loss = (self.log_alpha.exp() * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss
    
    def _update_temperature(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        with torch.no_grad():
            _, log_prob = self.actor.get_action(states)
        alpha_loss = -(self.log_alpha * (log_prob + self.config.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return alpha_loss
    
    def _soft_update(self, tau: float) -> None:
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


if __name__ == "__main__":
    print("SAC Continuous Actions - AMD Acceleration:", check_amd_acceleration())
    config = SACConfig(state_dim=10, action_dim=2)
    agent = SoftActorCritic(config)
    state = np.random.randn(10)
    action, log_prob = agent.select_action(state)
    print(f"Sample action: {action}, log_prob: {log_prob}")
