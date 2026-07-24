"""
mistake_penalty.py - SOUL.md Mathematical Reward Penalty Injector
Stage 54: Nautilus/Ray Crypto Trading Bot
Injects severe mathematical penalties into RL agent for SOUL.md flagged patterns
AMD DirectML/ROCm GPU-accelerated inference for fast penalty calculation
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading

# Try to import GPU acceleration libraries
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PenaltySeverity(Enum):
    """Severity levels for mistake penalties"""
    LOW = 1.0
    MEDIUM = 2.5
    HIGH = 5.0
    CRITICAL = 10.0
    CATASTROPHIC = 50.0


@dataclass
class PenaltyConfig:
    """Configuration for penalty injection"""
    base_penalty_multiplier: float = 1.0
    max_penalty_multiplier: float = 50.0
    decay_rate: float = 0.95  # Penalty decay per episode
    gpu_acceleration: bool = True
    penalty_half_life_episodes: int = 10


@dataclass
class StrategyPattern:
    """Represents a strategy pattern being evaluated"""
    pattern_id: str
    symbol: str
    strategy_name: str
    features: Dict[str, float]
    timestamp: str
    context: Dict[str, Any]


@dataclass
class PenaltyResult:
    """Result of penalty evaluation"""
    pattern_id: str
    rule_violated: Optional[str]
    base_reward: float
    penalty_amount: float
    final_reward: float
    penalty_multiplier: float
    severity: PenaltySeverity
    timestamp: str


class GPUPenaltyModel:
    """GPU-accelerated penalty prediction model using AMD DirectML/ROCm"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the neural network for penalty prediction"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU fallback")
            return
        
        try:
            # Detect AMD GPU via ROCm
            if self.device == 'cuda':
                if torch.cuda.is_available():
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("Using CPU for penalty model")
            
            # Simple MLP for penalty score prediction
            self.model = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ).to(self.device)
            
            self.model.eval()
            logger.info("Penalty model initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize penalty model: {e}")
            self.model = None
    
    def predict_penalty_score(self, features: torch.Tensor) -> float:
        """Predict penalty score from features using GPU acceleration"""
        if self.model is None or not TORCH_AVAILABLE:
            return 0.5  # Default neutral score
        
        try:
            with torch.no_grad():
                features = features.to(self.device)
                output = self.model(features)
                return output.item()
        except Exception as e:
            logger.error(f"Penalty prediction failed: {e}")
            return 0.5
    
    def batch_predict(self, features_batch: torch.Tensor) -> List[float]:
        """Batch prediction for multiple patterns"""
        if self.model is None or not TORCH_AVAILABLE:
            return [0.5] * len(features_batch)
        
        try:
            with torch.no_grad():
                features_batch = features_batch.to(self.device)
                outputs = self.model(features_batch)
                return outputs.squeeze().tolist()
        except Exception as e:
            logger.error(f"Batch penalty prediction failed: {e}")
            return [0.5] * len(features_batch)


class MistakePenaltyInjector:
    """
    Injects mathematical reward penalties into RL agent
    when it attempts strategies flagged by SOUL.md
    """
    
    def __init__(self, config: Optional[PenaltyConfig] = None):
        self.config = config or PenaltyConfig()
        self.avoidance_rules: Dict[str, Dict] = {}
        self.penalty_history: List[PenaltyResult] = []
        self.pattern_cache: Dict[str, StrategyPattern] = {}
        
        # Initialize GPU penalty model
        device = 'cuda' if self.config.gpu_acceleration and TORCH_AVAILABLE else 'cpu'
        self.penalty_model = GPUPenaltyModel(device=device)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"MistakePenaltyInjector initialized (device: {device})")
    
    def load_avoidance_rules(self, rules: List[Dict]):
        """Load avoidance rules from SOUL.md learner"""
        with self._lock:
            for rule in rules:
                rule_id = rule.get('rule_id')
                if rule_id:
                    self.avoidance_rules[rule_id] = rule
            logger.info(f"Loaded {len(self.avoidance_rules)} avoidance rules")
    
    def check_pattern_against_rules(
        self, 
        pattern: StrategyPattern
    ) -> Tuple[Optional[str], float]:
        """
        Check if a strategy pattern violates any avoidance rules
        Returns (rule_id, penalty_multiplier) or (None, 1.0) if no violation
        """
        with self._lock:
            for rule_id, rule in self.avoidance_rules.items():
                if not rule.get('is_active', True):
                    continue
                
                condition = rule.get('condition', '')
                
                # Evaluate condition against pattern
                if self._evaluate_condition(condition, pattern):
                    multiplier = rule.get('penalty_multiplier', 1.0)
                    
                    # Update rule violation count
                    rule['violation_count'] = rule.get('violation_count', 0) + 1
                    rule['last_violation'] = datetime.utcnow().isoformat()
                    
                    return rule_id, min(multiplier, self.config.max_penalty_multiplier)
            
            return None, 1.0
    
    def _evaluate_condition(self, condition: str, pattern: StrategyPattern) -> bool:
        """Evaluate an avoidance condition against a pattern"""
        try:
            # Parse simple AND conditions
            parts = condition.split(' AND ')
            
            for part in parts:
                part = part.strip()
                
                if 'symbol ==' in part:
                    target = part.split("'")[1]
                    if pattern.symbol != target:
                        return False
                
                elif 'strategy ==' in part:
                    target = part.split("'")[1]
                    if pattern.strategy_name != target:
                        return False
                
                elif 'category ==' in part:
                    # Category check would require additional pattern metadata
                    pass
            
            return True
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def compute_penalty(
        self,
        pattern: StrategyPattern,
        base_reward: float
    ) -> PenaltyResult:
        """
        Compute the penalized reward for a strategy pattern
        Applies both rule-based and model-based penalties
        """
        # Cache the pattern
        self.pattern_cache[pattern.pattern_id] = pattern
        
        # Check against avoidance rules
        violated_rule, rule_multiplier = self.check_pattern_against_rules(pattern)
        
        # Get model-based penalty score
        model_score = self._get_model_penalty_score(pattern)
        
        # Combine penalties
        combined_multiplier = rule_multiplier * (1.0 + model_score)
        
        # Apply severity classification
        severity = self._classify_severity(combined_multiplier)
        
        # Calculate final penalty
        if combined_multiplier > 1.0:
            penalty_amount = base_reward * (combined_multiplier - 1.0)
            # For negative rewards (losses), amplify the penalty
            if base_reward < 0:
                final_reward = base_reward * combined_multiplier
            else:
                final_reward = base_reward / combined_multiplier
        else:
            penalty_amount = 0.0
            final_reward = base_reward
        
        # Clamp to reasonable bounds
        final_reward = max(-100.0, min(100.0, final_reward))
        
        result = PenaltyResult(
            pattern_id=pattern.pattern_id,
            rule_violated=violated_rule,
            base_reward=base_reward,
            penalty_amount=penalty_amount,
            final_reward=final_reward,
            penalty_multiplier=combined_multiplier,
            severity=severity,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store in history
        with self._lock:
            self.penalty_history.append(result)
            # Keep only last 1000 results
            if len(self.penalty_history) > 1000:
                self.penalty_history = self.penalty_history[-1000:]
        
        if violated_rule:
            logger.warning(
                f"Penalty applied: Rule {violated_rule}, "
                f"multiplier={combined_multiplier:.2f}x, "
                f"reward: {base_reward:.2f} -> {final_reward:.2f}"
            )
        
        return result
    
    def _get_model_penalty_score(self, pattern: StrategyPattern) -> float:
        """Get penalty score from GPU-accelerated model"""
        if not TORCH_AVAILABLE or not NUMPY_AVAILABLE:
            return 0.0
        
        try:
            # Convert pattern features to tensor
            feature_vector = self._extract_feature_vector(pattern)
            features_tensor = torch.FloatTensor([feature_vector])
            
            score = self.penalty_model.predict_penalty_score(features_tensor)
            return score
            
        except Exception as e:
            logger.error(f"Model penalty score failed: {e}")
            return 0.0
    
    def _extract_feature_vector(self, pattern: StrategyPattern) -> List[float]:
        """Extract fixed-size feature vector from pattern"""
        features = []
        
        # Symbol encoding (one-hot style)
        symbol_features = [
            1.0 if pattern.symbol == 'BTCUSDT' else 0.0,
            1.0 if pattern.symbol == 'ETHUSDT' else 0.0,
            1.0 if pattern.symbol == 'SOLUSDT' else 0.0,
        ]
        features.extend(symbol_features)
        
        # Strategy features
        features.extend(list(pattern.features.values())[:29])  # Pad to 32 total
        
        # Pad to exactly 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return features[:32]
    
    def _classify_severity(self, multiplier: float) -> PenaltySeverity:
        """Classify penalty severity based on multiplier"""
        if multiplier >= 40.0:
            return PenaltySeverity.CATASTROPHIC
        elif multiplier >= 8.0:
            return PenaltySeverity.CRITICAL
        elif multiplier >= 4.0:
            return PenaltySeverity.HIGH
        elif multiplier >= 2.0:
            return PenaltySeverity.MEDIUM
        else:
            return PenaltySeverity.LOW
    
    def apply_decay(self):
        """Apply decay to all active penalties"""
        with self._lock:
            for rule_id, rule in self.avoidance_rules.items():
                current_mult = rule.get('penalty_multiplier', 1.0)
                new_mult = max(1.0, current_mult * self.config.decay_rate)
                rule['penalty_multiplier'] = new_mult
    
    def get_statistics(self) -> Dict:
        """Get penalty injector statistics"""
        with self._lock:
            if not self.penalty_history:
                return {
                    'total_penalties': 0,
                    'rules_loaded': len(self.avoidance_rules),
                }
            
            severities = {}
            for sev in PenaltySeverity:
                count = sum(1 for p in self.penalty_history if p.severity == sev)
                severities[sev.name] = count
            
            avg_multiplier = sum(p.penalty_multiplier for p in self.penalty_history) / len(self.penalty_history)
            
            return {
                'total_penalties': len(self.penalty_history),
                'rules_loaded': len(self.avoidance_rules),
                'average_multiplier': round(avg_multiplier, 2),
                'severity_distribution': severities,
                'gpu_acceleration': self.penalty_model.device == 'cuda',
            }


def create_penalty_injector(
    soul_md_path: Optional[str] = None,
    gpu_enabled: bool = True
) -> MistakePenaltyInjector:
    """Factory function to create and configure penalty injector"""
    config = PenaltyConfig(gpu_acceleration=gpu_enabled)
    injector = MistakePenaltyInjector(config)
    
    # Load existing rules from SOUL.md if available
    if soul_md_path:
        try:
            soul_path = Path(soul_md_path)
            if soul_path.exists():
                # Parse SOUL.md for rules (simplified)
                logger.info(f"Would load rules from {soul_md_path}")
        except Exception as e:
            logger.error(f"Failed to load SOUL.md: {e}")
    
    return injector


if __name__ == "__main__":
    # Test the penalty injector
    injector = create_penalty_injector(gpu_enabled=True)
    
    # Create test pattern
    test_pattern = StrategyPattern(
        pattern_id="test-001",
        symbol="BTCUSDT",
        strategy_name="momentum_breakout",
        features={"volatility": 0.5, "volume_ratio": 1.2},
        timestamp=datetime.utcnow().isoformat(),
        context={}
    )
    
    # Test penalty computation
    result = injector.compute_penalty(test_pattern, base_reward=0.5)
    print(f"Penalty Result: {result}")
    
    # Get stats
    stats = injector.get_statistics()
    print(f"Statistics: {stats}")
