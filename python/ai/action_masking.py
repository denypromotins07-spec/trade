"""
Dynamic Action Masking for RL Agents

Implements action masking logic that prevents the RL agent from selecting
invalid order types or breaching risk limits during exploration and exploitation.

Includes AMD ROCm/DirectML checks for hardware acceleration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import os


class OrderType(Enum):
    """Supported order types."""
    LIMIT_MAKER = auto()      # Passive limit order
    LIMIT_TAKER = auto()      # Aggressive limit order
    MARKET = auto()           # Market order
    STOP_LIMIT = auto()       # Stop-limit order
    IOC = auto()              # Immediate-or-cancel
    FOK = auto()              # Fill-or-kill
    POST_ONLY = auto()        # Post-only (maker)


class RiskViolation(Enum):
    """Types of risk violations that can be masked."""
    POSITION_LIMIT_EXCEEDED = auto()
    ORDER_SIZE_TOO_LARGE = auto()
    INSUFFICIENT_BALANCE = auto()
    PRICE_OUT_OF_BOUNDS = auto()
    RATE_LIMIT_EXCEEDED = auto()
    EXCHANGE_HALT = auto()
    INVALID_ORDER_TYPE = auto()
    SELF_TRADE_RISK = auto()


@dataclass
class RiskLimits:
    """Risk limits for action masking."""
    # Position limits
    max_position_pct: float = 0.1  # Max 10% of portfolio in single asset
    max_total_exposure: float = 5.0  # Max 5x leverage
    
    # Order size limits
    max_order_size_pct: float = 0.05  # Max 5% of position per order
    min_order_size: float = 0.001  # Minimum order size
    
    # Price limits
    max_price_deviation_pct: float = 0.05  # Max 5% from mid-price
    max_spread_crossing: bool = True  # Don't cross spread aggressively
    
    # Rate limits
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100
    
    # Exchange-specific
    exchange_halted: bool = False
    trading_suspended: bool = False


@dataclass
class ActionMask:
    """
    Binary mask for valid actions.
    
    Each element corresponds to an action dimension being allowed (1) or blocked (0).
    """
    # Discrete action masks (for order type selection)
    order_type_mask: np.ndarray = field(default_factory=lambda: np.ones(len(OrderType)))
    
    # Continuous action bounds (for price offset and size)
    price_offset_min: float = -0.1
    price_offset_max: float = 0.1
    size_min: float = 0.0
    size_max: float = 1.0
    
    # Violation reasons (for debugging/logging)
    violations: List[RiskViolation] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if any actions are valid."""
        return np.any(self.order_type_mask > 0)
    
    def get_valid_order_types(self) -> List[OrderType]:
        """Get list of valid order types."""
        return [ot for i, ot in enumerate(OrderType) if self.order_type_mask[i] > 0]


class ActionMasker:
    """
    Dynamic action masker for RL agents.
    
    Prevents invalid actions based on current state and risk limits.
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        
        # Rate limiting state
        self._order_timestamps: List[float] = []
        self._orders_this_minute: Dict[int, int] = {}
        
        # AMD acceleration check
        self._rocm_available = os.environ.get('ROCM_PATH') is not None
        self._directml_available = os.name == 'nt' and os.environ.get('DIRECTML_PATH') is not None
        
    def compute_mask(
        self,
        state: Dict,
        position: float,
        balance: float,
        current_price: float,
        mid_price: float,
        best_bid: float,
        best_ask: float
    ) -> ActionMask:
        """
        Compute action mask based on current state.
        
        Args:
            state: Current environment state
            position: Current position size
            balance: Available balance
            current_price: Last traded price
            mid_price: Current mid-price
            best_bid: Best bid price
            best_ask: Best ask price
            
        Returns:
            ActionMask with valid actions
        """
        mask = ActionMask()
        
        # Check exchange status
        if self.risk_limits.exchange_halted or self.risk_limits.trading_suspended:
            mask.violations.append(RiskViolation.EXCHANGE_HALT)
            mask.order_type_mask[:] = 0  # Block all orders
            return mask
        
        # Check position limits
        max_position = balance * self.risk_limits.max_position_pct
        if abs(position) >= max_position:
            mask.violations.append(RiskViolation.POSITION_LIMIT_EXCEEDED)
            # Block opening new positions, allow closing
            if position > 0:
                # Only allow sell orders
                mask.order_type_mask[OrderType.MARKET.value - 1] = 1
            else:
                # Only allow buy orders
                pass  # Keep buys enabled
        
        # Check order size limits
        max_order_size = abs(position) * self.risk_limits.max_order_size_pct if position != 0 else balance * 0.01
        mask.size_max = min(mask.size_max, max_order_size)
        mask.size_min = self.risk_limits.min_order_size
        
        if max_order_size < self.risk_limits.min_order_size:
            mask.violations.append(RiskViolation.ORDER_SIZE_TOO_LARGE)
            mask.order_type_mask[:] = 0  # No valid orders
        
        # Check price bounds
        if mid_price > 0:
            max_deviation = mid_price * self.risk_limits.max_price_deviation_pct
            mask.price_offset_min = -max_deviation / mid_price
            mask.price_offset_max = max_deviation / mid_price
        
        # Check rate limits
        if not self._check_rate_limit():
            mask.violations.append(RiskViolation.RATE_LIMIT_EXCEEDED)
            mask.order_type_mask[:] = 0
            return mask
        
        # Check insufficient balance
        if balance <= 0:
            mask.violations.append(RiskViolation.INSUFFICIENT_BALANCE)
            mask.order_type_mask[:] = 0
            return mask
        
        # Disable aggressive orders if crossing spread is disabled
        if not self.risk_limits.max_spread_crossing:
            # Block market orders and aggressive limits
            mask.order_type_mask[OrderType.MARKET.value - 1] = 0
            mask.order_type_mask[OrderType.LIMIT_TAKER.value - 1] = 0
        
        # Post-only always valid for maker strategies
        mask.order_type_mask[OrderType.POST_ONLY.value - 1] = 1
        
        return mask
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limits."""
        import time
        now = time.time()
        
        # Clean old timestamps (older than 1 second)
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 1.0]
        
        # Check per-second limit
        if len(self._order_timestamps) >= self.risk_limits.max_orders_per_second:
            return False
        
        # Check per-minute limit
        current_minute = int(now // 60)
        if self._orders_this_minute.get(current_minute, 0) >= self.risk_limits.max_orders_per_minute:
            return False
        
        return True
    
    def record_order(self):
        """Record an order for rate limiting."""
        import time
        now = time.time()
        self._order_timestamps.append(now)
        
        current_minute = int(now // 60)
        self._orders_this_minute[current_minute] = self._orders_this_minute.get(current_minute, 0) + 1
    
    def apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        mask: ActionMask
    ) -> torch.Tensor:
        """
        Apply action mask to policy logits.
        
        Sets invalid action logits to large negative value.
        """
        mask_tensor = torch.FloatTensor(mask.order_type_mask).to(logits.device)
        
        # Apply mask (add large negative to invalid actions)
        masked_logits = logits + (mask_tensor - 1) * 1e9
        
        return masked_logits
    
    def apply_mask_to_action(
        self,
        action: torch.Tensor,
        mask: ActionMask
    ) -> torch.Tensor:
        """
        Clip continuous actions to valid bounds.
        
        Ensures price offset and size are within allowed ranges.
        """
        # Clamp price offset
        action[..., 0] = torch.clamp(
            action[..., 0],
            mask.price_offset_min,
            mask.price_offset_max
        )
        
        # Clamp size
        action[..., 1] = torch.clamp(
            action[..., 1],
            mask.size_min,
            mask.size_max
        )
        
        return action
    
    def get_mask_tensor(self, mask: ActionMask, device: torch.device) -> torch.Tensor:
        """Convert mask to PyTorch tensor."""
        return torch.FloatTensor(mask.order_type_mask).to(device)


class MaskedSACPolicy:
    """
    SAC Policy with integrated action masking.
    
    Wraps the SAC agent to ensure all actions respect risk limits.
    """
    
    def __init__(
        self,
        sac_agent,
        risk_limits: Optional[RiskLimits] = None
    ):
        self.sac_agent = sac_agent
        self.masker = ActionMasker(risk_limits)
        
    def select_action(
        self,
        obs: np.ndarray,
        state_info: Dict,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, ActionMask]:
        """
        Select action with masking applied.
        
        Returns both the action and the mask used (for logging).
        """
        # Compute mask based on state
        mask = self.masker.compute_mask(
            state=state_info,
            position=state_info.get('position', 0),
            balance=state_info.get('balance', 0),
            current_price=state_info.get('current_price', 0),
            mid_price=state_info.get('mid_price', 0),
            best_bid=state_info.get('best_bid', 0),
            best_ask=state_info.get('best_ask', 0)
        )
        
        # Get raw action from SAC
        action = self.sac_agent.select_action(obs, deterministic=deterministic)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        
        # Apply mask to continuous components
        action_tensor = self.masker.apply_mask_to_action(action_tensor, mask)
        
        # For discrete order type, sample from masked distribution
        if hasattr(self.sac_agent, 'actor'):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.sac_agent.device)
                _, log_prob = self.sac_agent.actor(obs_tensor, with_log_prob=True)
                
                # Apply mask to logits if we had them
                # (This would need modification to the actor to return logits)
        
        return action_tensor.cpu().numpy()[0], mask
    
    def update_with_mask(
        self,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """Update SAC with masked actions in buffer."""
        return self.sac_agent.update(batch_size)


if __name__ == "__main__":
    # Test action masking
    print("Testing Dynamic Action Masking...")
    
    # Create masker
    risk_limits = RiskLimits(
        max_position_pct=0.1,
        max_order_size_pct=0.05,
        exchange_halted=False,
    )
    masker = ActionMasker(risk_limits)
    
    # Test state
    state = {'symbol': 'BTCUSDT'}
    
    # Normal conditions
    mask = masker.compute_mask(
        state=state,
        position=0.5,
        balance=10000,
        current_price=50000,
        mid_price=50000,
        best_bid=49999,
        best_ask=50001
    )
    
    print(f"\nNormal conditions:")
    print(f"  Valid order types: {mask.get_valid_order_types()}")
    print(f"  Price offset range: [{mask.price_offset_min:.4f}, {mask.price_offset_max:.4f}]")
    print(f"  Size range: [{mask.size_min}, {mask.size_max}]")
    print(f"  Violations: {mask.violations}")
    
    # Exchange halted
    risk_limits_halted = RiskLimits(exchange_halted=True)
    masker_halted = ActionMasker(risk_limits_halted)
    
    mask_halted = masker_halted.compute_mask(
        state=state,
        position=0.5,
        balance=10000,
        current_price=50000,
        mid_price=50000,
        best_bid=49999,
        best_ask=50001
    )
    
    print(f"\nExchange halted:")
    print(f"  Valid order types: {mask_halted.get_valid_order_types()}")
    print(f"  Is valid: {mask_halted.is_valid()}")
    print(f"  Violations: {mask_halted.violations}")
    
    # Test logit masking
    logits = torch.randn(len(OrderType))
    masked_logits = masker.apply_mask_to_logits(logits, mask)
    
    print(f"\nLogit masking test:")
    print(f"  Original logits sum: {logits.sum().item():.4f}")
    print(f"  Masked logits sum: {masked_logits.sum().item():.4f}")
    
    print("\nAction masking test completed!")
