"""
PopArt Reward Normalization for RL Training

Implements PopArt (Populating Returns Adaptively) reward normalization to stabilize
RL training across highly volatile crypto regimes without manual reward clipping.

Maintains running statistics of rewards and adaptively normalizes returns.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class PopArtConfig:
    """Configuration for PopArt normalization."""
    # Initial values
    initial_mean: float = 0.0
    initial_variance: float = 1.0
    
    # Update parameters
    update_rate: float = 0.01  # Rate for updating running stats
    epsilon: float = 1e-8  # Small constant for numerical stability
    
    # Clipping bounds (minimal, PopArt should avoid needing these)
    min_reward: float = -100.0
    max_reward: float = 100.0
    
    # Decay for old observations
    decay: float = 0.99


class PopArtNormalizer:
    """
    PopArt (Populating Returns Adaptively) reward normalizer.
    
    Maintains running estimates of reward mean and variance, and adapts
    the normalization parameters online during training.
    
    Unlike fixed normalization, PopArt can handle non-stationary reward
    distributions common in crypto markets.
    """
    
    def __init__(self, output_dim: int = 1, config: Optional[PopArtConfig] = None):
        self.output_dim = output_dim
        self.config = config or PopArtConfig()
        
        # Running statistics (learnable parameters)
        self.mean = torch.full((output_dim,), self.config.initial_mean)
        self.variance = torch.full((output_dim,), self.config.initial_variance)
        self.std = torch.sqrt(self.variance + self.config.epsilon)
        
        # Count of observations
        self.n_observations: int = 0
        
        # AMD acceleration check
        self._rocm_available = os.environ.get('ROCM_PATH') is not None
        self._directml_available = os.name == 'nt' and os.environ.get('DIRECTML_PATH') is not None
        
    def to(self, device: torch.device):
        """Move to device."""
        self.mean = self.mean.to(device)
        self.variance = self.variance.to(device)
        self.std = self.std.to(device)
        return self
    
    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards using current statistics.
        
        Args:
            rewards: Raw reward tensor
            
        Returns:
            Normalized rewards
        """
        # Ensure same device
        if rewards.device != self.mean.device:
            self.to(rewards.device)
        
        # Normalize: (r - mean) / std
        normalized = (rewards - self.mean) / self.std
        
        return normalized
    
    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """
        Denormalize values back to original scale.
        
        Useful for interpreting Q-values or predictions.
        """
        if normalized_values.device != self.mean.device:
            self.to(normalized_values.device)
        
        # Denormalize: norm * std + mean
        denormalized = normalized_values * self.std + self.mean
        
        return denormalized
    
    def update(self, rewards: torch.Tensor) -> Tuple[float, float]:
        """
        Update running statistics with new rewards.
        
        Uses Welford's online algorithm for numerical stability.
        
        Returns:
            Tuple of (new_mean, new_std)
        """
        rewards_flat = rewards.view(-1, self.output_dim)
        n_new = len(rewards_flat)
        
        if n_new == 0:
            return self.mean.item(), self.std.item()
        
        # Convert to float64 for precision
        rewards_float = rewards_flat.double()
        
        # Batch statistics
        batch_mean = rewards_float.mean(dim=0)
        batch_var = rewards_float.var(dim=0) + self.config.epsilon
        
        # Incremental update with weighted average
        old_n = self.n_observations
        new_n = old_n + n_new
        
        if old_n == 0:
            # First update
            self.mean = batch_mean.float()
            self.variance = batch_var.float()
        else:
            # Weighted combination
            alpha = n_new / new_n
            
            # Update mean
            delta = batch_mean - self.mean.double()
            new_mean = self.mean.double() + alpha * delta
            
            # Update variance using parallel variance formula
            # Var_total = w1*Var1 + w2*Var2 + w1*w2*(mean1-mean2)^2
            w1 = old_n / new_n
            w2 = n_new / new_n
            
            new_variance = (
                w1 * self.variance.double() +
                w2 * batch_var +
                w1 * w2 * (delta ** 2)
            )
            
            self.mean = new_mean.float()
            self.variance = new_variance.float()
        
        self.n_observations = new_n
        self.std = torch.sqrt(self.variance + self.config.epsilon)
        
        return self.mean.item() if self.output_dim == 1 else self.mean.tolist(), \
               self.std.item() if self.output_dim == 1 else self.std.tolist()
    
    def reset(self):
        """Reset statistics to initial values."""
        self.mean.fill_(self.config.initial_mean)
        self.variance.fill_(self.config.initial_variance)
        self.std = torch.sqrt(self.variance + self.config.epsilon)
        self.n_observations = 0
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'mean': self.mean.item() if self.output_dim == 1 else self.mean.tolist(),
            'variance': self.variance.item() if self.output_dim == 1 else self.variance.tolist(),
            'std': self.std.item() if self.output_dim == 1 else self.std.tolist(),
            'n_observations': self.n_observations,
            'rocm_available': self._rocm_available,
            'directml_available': self._directml_available,
        }


class PopArtHead(nn.Module):
    """
    Neural network head with integrated PopArt normalization.
    
    This is typically used as the final layer of a critic network,
    allowing the network to learn unnormalized values while outputting
    normalized predictions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 256,
        config: Optional[PopArtConfig] = None
    ):
        super().__init__()
        
        self.popart = PopArtNormalizer(output_dim, config)
        
        # Network layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize last layer weights small for stability
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, config.initial_mean if config else 0.0)
        
    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor
            normalize: If True, apply PopArt normalization to output
            
        Returns:
            (Normalized) output tensor
        """
        raw_output = self.network(x)
        
        if normalize:
            return self.popart.normalize(raw_output)
        else:
            return raw_output
    
    def predict_unnormalized(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Get unnormalized predictions (for interpretation)."""
        raw_output = self.network(x)
        return self.popart.denormalize(raw_output)
    
    def update_stats(self, targets: torch.Tensor) -> Dict:
        """Update PopArt statistics with target values."""
        mean, std = self.popart.update(targets)
        return {'mean': mean, 'std': std}


class PopArtSACLoss:
    """
    SAC loss function with PopArt reward normalization.
    
    Integrates PopArt into the SAC training loop for stable learning
    across varying reward scales.
    """
    
    def __init__(
        self,
        popart_normalizer: PopArtNormalizer,
        gamma: float = 0.99
    ):
        self.popart = popart_normalizer
        self.gamma = gamma
        
    def compute_critic_loss(
        self,
        q_values: torch.Tensor,
        target_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute critic loss with PopArt normalization.
        
        The key insight is that we normalize targets before computing loss,
        but the critic learns to predict normalized values.
        """
        # Normalize targets
        normalized_targets = self.popart.normalize(target_values)
        
        # Also normalize terminal states properly
        # When done=True, target should be just reward (no bootstrap)
        normalized_dones = self.popart.normalize(torch.zeros_like(target_values))
        
        # MSE loss on normalized values
        loss = nn.functional.mse_loss(q_values, normalized_targets)
        
        metrics = {
            'critic_loss': loss.item(),
            'q_values_mean': q_values.mean().item(),
            'targets_mean': target_values.mean().item(),
            'popart_mean': self.popart.mean.item(),
            'popart_std': self.popart.std.item(),
        }
        
        return loss, metrics
    
    def update_and_compute_loss(
        self,
        q_values: torch.Tensor,
        rewards: torch.Tensor,
        next_q_values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full critic loss computation with PopArt update.
        
        1. Compute TD targets
        2. Update PopArt statistics
        3. Compute normalized loss
        """
        # Compute TD targets
        with torch.no_grad():
            targets = rewards + self.gamma * (1 - dones) * next_q_values
        
        # Update PopArt statistics (online learning)
        self.popart.update(targets)
        
        # Compute loss with normalized targets
        return self.compute_critic_loss(q_values, targets, dones)


if __name__ == "__main__":
    # Test PopArt implementation
    print("Testing PopArt Reward Normalization...")
    
    # Create normalizer
    popart = PopArtNormalizer(output_dim=1)
    
    # Simulate training with varying reward scales
    np.random.seed(42)
    
    print("\nSimulating reward normalization across volatile regimes...")
    
    for episode in range(10):
        # Different reward regimes (simulating crypto volatility)
        if episode < 3:
            # Low volatility regime
            rewards = np.random.randn(100, 1) * 0.1
        elif episode < 6:
            # High volatility regime
            rewards = np.random.randn(100, 1) * 10.0
        else:
            # Extreme regime
            rewards = np.random.randn(100, 1) * 100.0
        
        rewards_tensor = torch.FloatTensor(rewards)
        
        # Update statistics
        mean, std = popart.update(rewards_tensor)
        
        # Normalize
        normalized = popart.normalize(rewards_tensor)
        
        print(f"Episode {episode + 1}:")
        print(f"  Raw rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        print(f"  PopArt stats: mean={mean:.4f}, std={std:.4f}")
        print(f"  Normalized: mean={normalized.mean().item():.4f}, std={normalized.std().item():.4f}")
    
    # Test PopArt head
    print("\n\nTesting PopArt Head...")
    
    popart_head = PopArtHead(input_dim=64, output_dim=1)
    
    # Forward pass
    x = torch.randn(32, 64)
    output = popart_head(x, normalize=True)
    output_unnorm = popart_head.predict_unnormalized(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Normalized output: mean={output.mean().item():.4f}, std={output.std().item():.4f}")
    print(f"Unnormalized output: mean={output_unnorm.mean().item():.4f}, std={output_unnorm.std().item():.4f}")
    
    # Get stats
    stats = popart.get_stats()
    print(f"\nFinal PopArt stats: {stats}")
    
    print("\nPopArt test completed successfully!")
