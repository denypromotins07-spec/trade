"""
python/ai/action_masking.py

Dynamic Action Masking for RL Agent

Prevents the RL agent from selecting invalid order types or breaching risk limits
during exploration and exploitation phases. Includes AMD ROCm/DirectML checks.

Memory Constraint: Masks computed on-the-fly, no persistent storage overhead.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import IntEnum
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


class OrderType(IntEnum):
    LIMIT_BUY = 0
    LIMIT_SELL = 1
    MARKET_BUY = 2
    MARKET_SELL = 3
    CANCEL = 4


@dataclass
class RiskLimits:
    max_position_size: float = 10.0  # BTC equivalent
    max_order_size: float = 1.0
    max_daily_turnover: float = 100.0
    max_drawdown_pct: float = 0.05
    concentration_limit: float = 0.25  # Max 25% in single asset


class ActionMasker:
    """
    Dynamic action masking for RL trading agent.
    Prevents invalid actions based on current state and risk limits.
    """
    
    def __init__(self, num_actions: int = 64, device: str = "cpu"):
        self.num_actions = num_actions
        self.device = device
        self.acceleration = check_amd_acceleration()
        
    def compute_mask(
        self,
        current_position: float,
        available_balance: float,
        current_price: float,
        risk_limits: RiskLimits,
        market_state: Dict,
        action_metadata: List[Dict]
    ) -> torch.Tensor:
        """
        Compute binary action mask where 1 = allowed, 0 = masked.
        
        Args:
            current_position: Current position size (positive = long)
            available_balance: Available buying power
            current_price: Current mid price
            risk_limits: Risk limit configuration
            market_state: Market conditions (spread, volatility, etc.)
            action_metadata: Metadata for each action (type, size, etc.)
            
        Returns:
            Binary mask tensor of shape [num_actions]
        """
        mask = torch.ones(self.num_actions, dtype=torch.float32, device=self.device)
        
        for i, meta in enumerate(action_metadata):
            if i >= self.num_actions:
                break
                
            action_type = meta.get("type", OrderType.LIMIT_BUY)
            order_size = meta.get("size", 0.0)
            
            # Check position limits
            if action_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY]:
                new_position = current_position + order_size
                if new_position > risk_limits.max_position_size:
                    mask[i] = 0.0
                    
            elif action_type in [OrderType.MARKET_SELL, OrderType.LIMIT_SELL]:
                new_position = current_position - order_size
                if new_position < -risk_limits.max_position_size:
                    mask[i] = 0.0
            
            # Check balance constraints
            if action_type in [OrderType.MARKET_BUY, OrderType.LIMIT_BUY]:
                required_balance = order_size * current_price
                if required_balance > available_balance:
                    mask[i] = 0.0
            
            # Check order size limits
            if order_size > risk_limits.max_order_size:
                mask[i] = 0.0
            
            # Check market conditions
            spread_pct = market_state.get("spread_pct", 0.0)
            if spread_pct > 0.01:  # Spread > 1%
                # Disable market orders during wide spreads
                if action_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL]:
                    mask[i] = 0.0
            
            # Check volatility
            volatility = market_state.get("volatility", 0.0)
            if volatility > 0.1:  # High volatility
                # Reduce aggressive actions
                if action_type in [OrderType.MARKET_BUY, OrderType.MARKET_SELL]:
                    mask[i] *= 0.5  # Soft mask
        
        # Ensure at least one action is valid (CANCEL always allowed)
        if mask.sum() == 0:
            cancel_indices = [i for i, m in enumerate(action_metadata) 
                            if m.get("type") == OrderType.CANCEL]
            if cancel_indices:
                mask[cancel_indices[0]] = 1.0
            else:
                mask[0] = 1.0  # Fallback
        
        return mask
    
    def apply_mask_to_logits(
        self, 
        logits: torch.Tensor, 
        mask: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Apply mask to action logits with large negative penalty.
        
        Args:
            logits: Raw action scores from policy network
            mask: Binary mask (1 = allowed, 0 = masked)
            temperature: Sampling temperature
            
        Returns:
            Masked and scaled logits
        """
        masked_logits = logits + (mask - 1) * 1e9
        return masked_logits / temperature
    
    def sample_action(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Sample action from masked distribution.
        
        Returns:
            action_idx: Selected action index
            log_prob: Log probability of selected action
        """
        masked_logits = self.apply_mask_to_logits(logits, mask)
        probs = torch.softmax(masked_logits, dim=-1)
        
        if deterministic:
            action_idx = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
        
        log_prob = torch.log(probs[action_idx] + 1e-8)
        return int(action_idx), log_prob.item()


class RiskMonitor:
    """
    Real-time risk monitoring for action masking decisions.
    Tracks exposure and triggers hard masks when limits breached.
    """
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.daily_turnover = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
    def update(self, equity: float, trade_volume: float = 0.0):
        """Update risk monitor with latest values."""
        self.current_equity = equity
        self.daily_turnover += trade_volume
        
        if equity > self.peak_equity:
            self.peak_equity = equity
    
    def check_drawdown(self) -> float:
        """Return current drawdown percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def get_hard_mask(self) -> bool:
        """
        Return True if hard kill switch should activate.
        All trading actions will be masked.
        """
        # Check daily turnover
        if self.daily_turnover > self.limits.max_daily_turnover:
            return True
        
        # Check drawdown
        if self.check_drawdown() > self.limits.max_drawdown_pct:
            return True
        
        return False
    
    def reset_daily(self):
        """Reset daily counters."""
        self.daily_turnover = 0.0


if __name__ == "__main__":
    print("Action Masking - AMD Acceleration:", check_amd_acceleration())
    
    masker = ActionMasker(num_actions=10)
    limits = RiskLimits()
    
    # Test mask computation
    mask = masker.compute_mask(
        current_position=5.0,
        available_balance=50000.0,
        current_price=50000.0,
        risk_limits=limits,
        market_state={"spread_pct": 0.001, "volatility": 0.02},
        action_metadata=[{"type": OrderType.LIMIT_BUY, "size": 0.5}] * 10
    )
    print(f"Action mask: {mask}")
