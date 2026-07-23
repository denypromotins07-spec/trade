"""
python/ai/reward_normalization.py

PopArt (Populating Returns Adaptively) Reward Normalization

Stabilizes RL training across highly volatile crypto regimes without manual reward clipping.
Adaptively normalizes returns while preserving the ability to learn positive/negative signals.

Memory Constraint: Running statistics only, O(1) memory per reward dimension.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import os


def check_amd_acceleration() -> Dict[str, bool]:
    """Detect AMD ROCm/DirectML availability."""
    result = {"cuda": torch.cuda.is_available(), "rocm": False, "directml": False, "cpu": True}
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
class PopArtConfig:
    shape: Tuple[int, ...] = (1,)  # Reward dimension
    clip_range: float = 10.0  # Safety clip after normalization
    decay: float = 0.999  # EMA decay for running stats
    min_std: float = 1e-4  # Minimum std to prevent division by zero


class PopArtNormalizer(nn.Module):
    """
    PopArt reward normalization that adaptively scales rewards
    while preserving the sign and relative magnitude.
    
    Unlike simple standardization, PopArt maintains a learned
    scale parameter that can grow/shrink with the reward distribution.
    """
    
    def __init__(self, config: PopArtConfig, device: str = "cpu"):
        super().__init__()
        self.config = config
        self.device = device
        self.acceleration = check_amd_acceleration()
        
        # Running statistics (not trained via backprop)
        self.register_buffer('running_mean', torch.zeros(config.shape))
        self.register_buffer('running_var', torch.ones(config.shape))
        self.register_buffer('running_max', torch.full(config.shape, -float('inf')))
        self.register_buffer('running_min', torch.full(config.shape, float('inf')))
        
        # Learned scale parameter (trained)
        self.scale = nn.Parameter(torch.ones(config.shape))
        self.shift = nn.Parameter(torch.zeros(config.shape))
        
        # Count for Welford's algorithm
        self.count = 0
        
    def update_stats(self, rewards: torch.Tensor) -> None:
        """
        Update running statistics using Welford's online algorithm.
        Handles batched or scalar rewards.
        """
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        
        batch_size = rewards.shape[0]
        self.count += batch_size
        
        # Update running mean and variance (Welford's)
        delta = rewards - self.running_mean
        self.running_mean += delta.sum(dim=0) / self.count
        
        if self.count > 1:
            delta2 = rewards - self.running_mean
            self.running_var += (delta * delta2).sum(dim=0)
        
        # Update running min/max for sanity checks
        self.running_max = torch.max(self.running_max, rewards.max(dim=0)[0])
        self.running_min = torch.min(self.running_min, rewards.min(dim=0)[0])
        
    def normalize(self, rewards: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Normalize rewards using current statistics and learned parameters.
        
        Args:
            rewards: Raw reward values
            update_stats: Whether to update running statistics
            
        Returns:
            Normalized rewards
        """
        if update_stats:
            self.update_stats(rewards)
        
        # Compute normalized value
        std = torch.sqrt(self.running_var / max(1, self.count) + self.config.min_std)
        normalized = (rewards - self.running_mean) / std
        
        # Apply learned scale and shift
        # This allows the network to learn appropriate scaling
        return normalized * self.scale + self.shift
    
    def denormalize(self, normalized_rewards: torch.Tensor) -> torch.Tensor:
        """Convert normalized rewards back to original scale."""
        std = torch.sqrt(self.running_var / max(1, self.count) + self.config.min_std)
        return (normalized_rewards - self.shift) / self.scale * std + self.running_mean
    
    def update_scale_shift(self, target_mean: float = 0.0, target_var: float = 1.0) -> None:
        """
        Update scale and shift to maintain target statistics.
        This is the "Pop" part of PopArt - popping the statistics.
        """
        std = torch.sqrt(self.running_var / max(1, self.count) + self.config.min_std)
        
        # Update scale to maintain unit variance
        new_scale = std * self.scale / torch.sqrt(torch.tensor(target_var))
        
        # Update shift to maintain zero mean
        new_shift = (self.running_mean - target_mean) * self.scale + self.shift
        
        self.scale.data = new_scale
        self.shift.data = new_shift
        
        # Reset running stats to target values
        self.running_mean.fill_(target_mean)
        self.running_var.fill_(target_var)


class AdaptiveRewardNormalizer:
    """
    Complete adaptive reward normalization system for RL training.
    Combines PopArt with additional crypto-specific adaptations.
    """
    
    def __init__(
        self, 
        config: PopArtConfig,
        use_sign_preservation: bool = True,
        device: str = "cpu"
    ):
        self.config = config
        self.device = device
        self.popart = PopArtNormalizer(config, device)
        self.use_sign_preservation = use_sign_preservation
        
        # Crypto-specific: track regime changes
        self.volatility_estimate = 0.0
        self.volatility_decay = 0.99
        
    def normalize_reward(
        self, 
        raw_reward: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Normalize a single reward value with optional volatility adjustment.
        
        In high volatility regimes, we reduce the effective learning rate
        by compressing extreme rewards.
        """
        reward_tensor = torch.tensor([raw_reward], dtype=torch.float32, device=self.device)
        
        # Apply PopArt normalization
        normalized = self.popart.normalize(reward_tensor, update_stats=True)
        
        # Volatility adjustment (crypto-specific)
        if volatility is not None:
            # Update exponential volatility estimate
            self.volatility_estimate = (
                self.volatility_decay * self.volatility_estimate +
                (1 - self.volatility_decay) * volatility
            )
            
            # Compress rewards during high volatility
            if self.volatility_estimate > 0.1:  # High vol threshold
                compression = 1.0 / (1.0 + self.volatility_estimate)
                normalized = normalized * compression
        
        # Sign preservation: ensure normalized reward has same sign as raw
        if self.use_sign_preservation:
            sign_mask = torch.sign(raw_reward) == torch.sign(normalized)
            if not sign_mask.all():
                # Adjust to preserve sign
                normalized = torch.where(
                    sign_mask,
                    normalized,
                    torch.sign(raw_reward) * normalized.abs()
                )
        
        # Safety clipping
        normalized = torch.clamp(normalized, -self.config.clip_range, self.config.clip_range)
        
        return normalized.item()
    
    def normalize_batch(
        self,
        rewards: np.ndarray,
        volatilities: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Normalize a batch of rewards."""
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        
        normalized = self.popart.normalize(reward_tensor, update_stats=True)
        
        if volatilities is not None:
            vol_tensor = torch.FloatTensor(volatilities).to(self.device)
            compression = 1.0 / (1.0 + vol_tensor)
            normalized = normalized * compression.unsqueeze(-1)
        
        normalized = torch.clamp(normalized, -self.config.clip_range, self.config.clip_range)
        
        return normalized.cpu().numpy()
    
    def get_state_dict(self) -> Dict:
        """Get serializable state dict."""
        return {
            'running_mean': self.popart.running_mean.cpu().numpy(),
            'running_var': self.popart.running_var.cpu().numpy(),
            'scale': self.popart.scale.data.cpu().numpy(),
            'shift': self.popart.shift.data.cpu().numpy(),
            'count': self.popart.count,
            'volatility_estimate': self.volatility_estimate,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """Load from state dict."""
        self.popart.running_mean = torch.FloatTensor(state['running_mean']).to(self.device)
        self.popart.running_var = torch.FloatTensor(state['running_var']).to(self.device)
        self.popart.scale.data = torch.FloatTensor(state['scale']).to(self.device)
        self.popart.shift.data = torch.FloatTensor(state['shift']).to(self.device)
        self.popart.count = state['count']
        self.volatility_estimate = state.get('volatility_estimate', 0.0)


if __name__ == "__main__":
    print("Reward Normalization (PopArt) - AMD Acceleration:", check_amd_acceleration())
    
    config = PopArtConfig(shape=(1,))
    normalizer = AdaptiveRewardNormalizer(config)
    
    # Simulate crypto reward sequence with varying volatility
    rewards = [1.0, -0.5, 2.0, -1.0, 5.0, -3.0, 10.0, -8.0]
    volatilities = [0.02, 0.03, 0.05, 0.08, 0.15, 0.20, 0.30, 0.25]
    
    print("\nRaw rewards -> Normalized:")
    for r, v in zip(rewards, volatilities):
        norm_r = normalizer.normalize_reward(r, volatility=v)
        print(f"  {r:7.2f} (vol={v:.2f}) -> {norm_r:7.4f}")
    
    print(f"\nFinal state: {normalizer.get_state_dict()}")
