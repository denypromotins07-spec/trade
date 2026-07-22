"""
Custom Proximal Policy Optimization (PPO) with Generalized Advantage Estimation

This module implements a custom PPO network architecture featuring GAE,
specifically tuned for high-frequency trading with multi-discrete action spaces.

Key features:
- AMD DirectML/ROCm environment detection and optimization
- Generalized Advantage Estimation for stable policy gradients
- Multi-discrete action space support (order size × order side × price level)
- Clipped surrogate objective for stable training
- Memory-efficient implementation for 4GB RAM quota
- Optimized for Ryzen AI 5 architecture

Usage:
    ppo = CustomPPONetwork(state_dim=64, action_dims=[3, 5, 10])
    action, log_prob, value = ppo.select_action(state)
    loss = ppo.update(states, actions, old_log_probs, returns, advantages)
"""

import os
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Independent, Normal

# Import hardware detection from attention module
try:
    from .attention import DIRECTML_AVAILABLE, ROCM_AVAILABLE, RECOMMENDED_DEVICE
except ImportError:
    # Fallback if running standalone
    DIRECTML_AVAILABLE = False
    ROCM_AVAILABLE = False
    RECOMMENDED_DEVICE = "cpu"


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Network architecture
    hidden_dim: int = 256
    num_layers: int = 2
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # GAE parameters
    gae_lambda: float = 0.95
    gamma: float = 0.99
    
    # Training parameters
    learning_rate: float = 3e-4
    epochs_per_update: int = 10
    batch_size: int = 64
    normalize_advantages: bool = True
    
    # Action space
    action_dims: List[int] = None  # Will be set at initialization


class ActorCriticBase(nn.Module):
    """
    Base actor-critic network architecture.
    
    Shared backbone with separate heads for policy and value.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        self.shared_backbone = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features."""
        return self.shared_backbone(x)


class MultiDiscreteActor(nn.Module):
    """
    Actor head for multi-discrete action spaces.
    
    Each discrete dimension has its own output head.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        action_dims: List[int],
    ):
        super().__init__()
        
        self.action_dims = action_dims
        self.num_actions = len(action_dims)
        
        # Separate head for each action dimension
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, dim),
            )
            for dim in action_dims
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get logits for each action dimension."""
        return [head(x) for head in self.action_heads]
    
    def get_distribution(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[Categorical], torch.Tensor]:
        """
        Get categorical distributions for each action dimension.
        
        Returns:
            Tuple of (list of distributions, concatenated action tensor)
        """
        logits_list = self.forward(x)
        dists = [Categorical(logits=logits) for logits in logits_list]
        return dists, logits_list


class CriticHead(nn.Module):
    """
    Critic head for value estimation.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate state value."""
        return self.value_head(x)


class CustomPPONetwork(nn.Module):
    """
    Complete PPO network with actor-critic architecture.
    
    Supports multi-discrete action spaces typical in HFT:
    - Order side (buy/sell/hold)
    - Order size (multiple levels)
    - Price offset (multiple levels from mid)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dims: List[int],
        config: Optional[PPOConfig] = None,
    ):
        super().__init__()
        
        self.config = config or PPOConfig()
        self.config.action_dims = action_dims
        
        self.state_dim = state_dim
        self.action_dims = action_dims
        self.total_actions = sum(action_dims)
        
        # Move to optimal device
        self.device = RECOMMENDED_DEVICE
        if isinstance(self.device, int) or (isinstance(self.device, str) and self.device != "cpu"):
            self.to(self.device)
        
        # Build network components
        self.backbone = ActorCriticBase(
            state_dim=state_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        )
        
        self.actor = MultiDiscreteActor(
            hidden_dim=self.config.hidden_dim,
            action_dims=action_dims,
        )
        
        self.critic = CriticHead(hidden_dim=self.config.hidden_dim)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6,
        )
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
        }
    
    def forward(
        self,
        states: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through network.
        
        Returns:
            Tuple of (action logits, state values)
        """
        features = self.backbone(states)
        action_logits = self.actor(features)
        values = self.critic(features)
        return action_logits, values.squeeze(-1)
    
    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """
        Select action given state.
        
        Args:
            state: State tensor of shape (state_dim,) or (batch, state_dim)
            deterministic: Whether to use argmax instead of sampling
        
        Returns:
            Tuple of (actions, log probabilities, value)
        """
        # Ensure correct shape
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if state.device.type != self.device:
            state = state.to(self.device)
        
        with torch.no_grad():
            features = self.backbone(state)
            action_logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            
            if deterministic:
                actions = [logits.argmax(dim=-1) for logits in action_logits]
                log_probs = torch.zeros(len(actions), device=self.device)
            else:
                dists = [Categorical(logits=logits) for logits in action_logits]
                actions = [dist.sample() for dist in dists]
                log_probs = torch.stack([dist.log_prob(a) for dist, a in zip(dists, actions)])
                log_probs = log_probs.sum()  # Sum across action dimensions
            
            actions_tensor = torch.stack(actions)
        
        return actions_tensor.tolist() if len(actions_tensor) > 1 else [actions_tensor[0].item()], log_probs, value
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            states: State tensor
            actions: Action tensor of shape (num_actions, batch)
        
        Returns:
            Tuple of (log_probs, entropy, values)
        """
        features = self.backbone(states)
        action_logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        
        # Calculate log probabilities for each action dimension
        log_probs_list = []
        entropy_list = []
        
        for i, (logits, action) in enumerate(zip(action_logits, actions)):
            dist = Categorical(logits=logits)
            log_probs_list.append(dist.log_prob(action))
            entropy_list.append(dist.entropy())
        
        log_probs = torch.stack(log_probs_list).sum(dim=0)
        entropy = torch.stack(entropy_list).sum(dim=0)
        
        return log_probs, entropy, values
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Reward tensor of shape (T,)
            values: Value estimates of shape (T,)
            next_values: Next state values of shape (T,) or scalar
            dones: Done flags of shape (T,)
        
        Returns:
            Tuple of (advantages, returns)
        """
        T = len(rewards)
        
        # Calculate TD deltas
        if isinstance(next_values, torch.Tensor) and next_values.dim() > 0:
            next_vals = next_values
        else:
            next_vals = torch.zeros_like(values)
        
        deltas = rewards + self.config.gamma * next_vals * (1 - dones) - values
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        advantage = torch.zeros(1, device=self.device)
        for t in reversed(range(T)):
            advantage = deltas[t] + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
            returns[t] = advantage + values[t]
        
        # Normalize advantages if enabled
        if self.config.normalize_advantages and advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform PPO update step.
        
        Args:
            states: State tensor of shape (batch, state_dim)
            actions: Action tensor of shape (num_actions, batch)
            old_log_probs: Log probabilities from behavior policy
            returns: Return targets
            advantages: Advantage estimates
        
        Returns:
            Dictionary of loss metrics
        """
        batch_size = states.shape[0]
        
        # Shuffle data for mini-batch updates
        indices = torch.randperm(batch_size, device=self.device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        num_updates = 0
        
        for epoch in range(self.config.epochs_per_update):
            for start in range(0, batch_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[:, batch_indices] if actions.dim() > 1 else actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate current policy
                log_probs, entropy, values = self.evaluate_actions(batch_states, batch_actions)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_bonus = entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                clip_fraction = (ratio - 1).abs().gt(self.config.clip_epsilon).float().mean().item()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                total_clip_fraction += clip_fraction
                num_updates += 1
        
        # Update learning rate
        self.scheduler.step()
        
        # Store metrics
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
            'learning_rate': self.scheduler.get_last_lr()[0],
        }
        
        for key, value in metrics.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)


class HFTPolicyNetwork(CustomPPONetwork):
    """
    Specialized PPO network for high-frequency trading.
    
    Action space designed for market making:
    - Dimension 0: Order side (0=hold, 1=buy, 2=sell)
    - Dimension 1: Order size (5 levels)
    - Dimension 2: Price offset (10 levels from mid)
    """
    
    def __init__(
        self,
        state_dim: int,
        config: Optional[PPOConfig] = None,
    ):
        # Default HFT action space
        default_config = config or PPOConfig()
        default_config.action_dims = [3, 5, 10]  # side, size, offset
        
        super().__init__(
            state_dim=state_dim,
            action_dims=default_config.action_dims,
            config=default_config,
        )
    
    def decode_action(
        self,
        action: List[int],
        mid_price: float,
        tick_size: float,
    ) -> Dict[str, Any]:
        """
        Decode action indices into trading parameters.
        
        Args:
            action: List of action indices [side, size_level, offset_level]
            mid_price: Current mid price
            tick_size: Minimum price increment
        
        Returns:
            Dictionary with decoded trading parameters
        """
        side_idx, size_idx, offset_idx = action
        
        # Map side
        side_map = {0: 'hold', 1: 'buy', 2: 'sell'}
        side = side_map.get(side_idx, 'hold')
        
        # Map size (example: 1, 5, 10, 50, 100 units)
        size_levels = [1, 5, 10, 50, 100]
        quantity = size_levels[min(size_idx, len(size_levels) - 1)]
        
        # Map price offset (in ticks from mid)
        # Negative offset for buys (below mid), positive for sells (above mid)
        offset_ticks = offset_idx - 5  # Range: -5 to +4
        price_offset = offset_ticks * tick_size
        
        if side == 'buy':
            limit_price = mid_price - abs(price_offset)
        elif side == 'sell':
            limit_price = mid_price + abs(price_offset)
        else:
            limit_price = mid_price
        
        return {
            'side': side,
            'quantity': quantity,
            'limit_price': round(limit_price, 2),
            'offset_ticks': offset_ticks,
        }


def create_ppo_network(
    state_dim: int,
    action_dims: List[int] = None,
    use_gpu: bool = True,
) -> CustomPPONetwork:
    """
    Factory function to create PPO network.
    
    Automatically configures for AMD DirectML/ROCm if available.
    """
    if action_dims is None:
        action_dims = [3, 5, 10]  # Default HFT action space
    
    network = CustomPPONetwork(
        state_dim=state_dim,
        action_dims=action_dims,
    )
    
    # Force device placement if requested
    if use_gpu and ROCM_AVAILABLE:
        network.to("cuda")
    elif use_gpu and DIRECTML_AVAILABLE:
        network.to(torch_directml.device())
    
    return network


if __name__ == "__main__":
    # Test the PPO network
    print("Testing Custom PPO Network...")
    print(f"DirectML Available: {DIRECTML_AVAILABLE}")
    print(f"ROCm Available: {ROCM_AVAILABLE}")
    print(f"Recommended Device: {RECOMMENDED_DEVICE}")
    
    # Create network
    network = create_ppo_network(state_dim=64, action_dims=[3, 5, 10])
    network.eval()
    
    # Create sample input
    sample_state = torch.randn(64)
    
    # Select action
    with torch.no_grad():
        action, log_prob, value = network.select_action(sample_state)
    
    print(f"\nState dimension: 64")
    print(f"Action dimensions: [3, 5, 10]")
    print(f"Selected action: {action}")
    print(f"Log probability: {log_prob:.4f}")
    print(f"Value estimate: {value:.4f}")
    
    # Test HFT policy network
    hft_network = HFTPolicyNetwork(state_dim=64)
    hft_network.eval()
    
    with torch.no_grad():
        action, _, _ = hft_network.select_action(sample_state)
        decoded = hft_network.decode_action(action, mid_price=50000.0, tick_size=0.01)
    
    print(f"\nHFT Policy Network:")
    print(f"Decoded action: {decoded}")
    print("\nCustom PPO Network test complete!")
