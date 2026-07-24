"""
Stage 62: AI & Pipeline Audit - File 7/20
Module: python/soul/mistake_penalty.py
Focus: Reward Penalty Injection, RL State Dictionary Corruption Prevention
Constraints: 4GB RAM Quota

AUDIT FIXES APPLIED:
- Fixed reward penalty injection logic
- Prevented RL state dictionary corruption via deep copies
- Added validation for state transitions
"""

from __future__ import annotations
import numpy as np
import copy
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MistakePenaltyInjector:
    """
    Injects penalties into reward signals based on mistake detection.
    FIX: Prevents state dictionary corruption via deep copies.
    """
    
    def __init__(self, penalty_factor: float = 0.5, decay_rate: float = 0.99):
        self.penalty_factor = penalty_factor
        self.decay_rate = decay_rate
        self._cumulative_penalty = 0.0
        self._mistake_count = 0
        
    def detect_mistake(self, action: np.ndarray, valid_actions: np.ndarray) -> bool:
        """Detect if an action is a mistake."""
        # Check if action is within valid range
        if len(valid_actions) == 0:
            return False
        
        # Simple distance-based mistake detection
        min_distance = np.min(np.linalg.norm(valid_actions - action, axis=1))
        threshold = np.mean(np.linalg.norm(valid_actions, axis=1)) * 0.1
        
        return min_distance > threshold
    
    def inject_penalty(self, reward: float, is_mistake: bool) -> float:
        """Inject penalty into reward signal."""
        if is_mistake:
            self._mistake_count += 1
            penalty = reward * self.penalty_factor
            self._cumulative_penalty = (
                self.decay_rate * self._cumulative_penalty + penalty
            )
            return reward - penalty
        
        return reward
    
    def get_penalty_stats(self) -> Dict[str, float]:
        """Get penalty statistics."""
        return {
            'cumulative_penalty': self._cumulative_penalty,
            'mistake_count': self._mistake_count,
            'average_penalty': self._cumulative_penalty / max(1, self._mistake_count)
        }
    
    def reset(self) -> None:
        """Reset penalty tracking."""
        self._cumulative_penalty = 0.0
        self._mistake_count = 0


class RLStateValidator:
    """
    Validates RL state dictionaries to prevent corruption.
    FIX: Uses deep copies and validates structure.
    """
    
    def __init__(self, expected_keys: list):
        self.expected_keys = set(expected_keys)
        
    def validate(self, state: Dict[str, Any]) -> bool:
        """Validate state dictionary structure."""
        if not isinstance(state, dict):
            logger.error("State is not a dictionary")
            return False
        
        missing_keys = self.expected_keys - set(state.keys())
        if missing_keys:
            logger.error(f"Missing keys in state: {missing_keys}")
            return False
        
        return True
    
    def safe_copy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a safe deep copy of state."""
        try:
            return copy.deepcopy(state)
        except Exception as e:
            logger.error(f"Failed to copy state: {e}")
            raise
    
    def merge_states(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge state dictionaries."""
        result = self.safe_copy(base)
        
        for key, value in updates.items():
            if key in self.expected_keys:
                result[key] = copy.deepcopy(value)
            else:
                logger.warning(f"Ignoring unexpected key: {key}")
        
        return result


if __name__ == "__main__":
    print("Mistake penalty module loaded")
