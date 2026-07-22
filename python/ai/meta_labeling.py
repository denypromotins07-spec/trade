"""
Meta-Labeling Architecture - Marcos Lopez de Prado Implementation

This module implements Marcos Lopez de Prado's meta-labeling architecture
to filter false positives from the primary model and dynamically size bets
using secondary probability outputs. Optimized for Ray workers with 4GB RAM ceiling.

Features:
- Primary model signal generation
- Meta-labeler for false positive filtering
- Dynamic bet sizing based on confidence
- AMD ROCm/DirectML environment checks
- Memory-efficient Ray worker management
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

import ray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class MetaLabelResult:
    """Result of meta-labeling prediction"""
    primary_signal: SignalType
    meta_label: int  # 1 = take signal, 0 = filter out
    primary_probability: float
    meta_probability: float
    bet_size: float  # Normalized 0-1
    expected_return: float
    timestamp_ns: int


def detect_amd_rocm() -> bool:
    """Detect AMD ROCm availability"""
    try:
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        if not os.path.exists(rocm_path):
            return False
        
        hip_lib = os.path.join(rocm_path, 'lib', 'libhipblas.so')
        if not os.path.exists(hip_lib):
            return False
        
        try:
            import torch
            if torch.version.hip is not None:
                logger.info(f"ROCm detected: {torch.version.hip}")
                return True
        except ImportError:
            pass
        
        return False
    except Exception as e:
        logger.warning(f"Error detecting ROCm: {e}")
        return False


def detect_directml() -> bool:
    """Detect Microsoft DirectML availability"""
    try:
        import torch
        try:
            import torch_directml
            logger.info("DirectML detected")
            return True
        except ImportError:
            return False
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"Error detecting DirectML: {e}")
        return False


def get_device_config() -> Dict[str, Any]:
    """Get optimal device configuration for AMD Ryzen AI 5"""
    config = {"device": "cpu", "use_gpu": False}
    
    if detect_amd_rocm():
        config["device"] = "cuda"
        config["use_gpu"] = True
        logger.info("Using AMD ROCm for meta-labeling")
    elif detect_directml():
        config["device"] = "directml"
        config["use_gpu"] = True
        logger.info("Using DirectML for meta-labeling")
    else:
        logger.info("Using CPU for meta-labeling")
    
    return config


@ray.remote(max_calls=500)
class PrimaryModel:
    """
    Primary trading signal model.
    Generates initial long/short/flat signals.
    """
    
    def __init__(self, model_params: Dict[str, Any]):
        self.params = model_params
        self.device_config = get_device_config()
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize primary model (simplified logistic regression for demo)"""
        # In production, this would be your actual ML model
        # Could be XGBoost, LightGBM, neural network, etc.
        self.weights = np.random.randn(self.params.get('input_dim', 10))
        self.bias = 0.0
        logger.info(f"Primary model initialized on {self.device_config['device']}")
    
    def predict(self, features: np.ndarray) -> Tuple[SignalType, float]:
        """
        Generate primary trading signal.
        Returns signal type and probability.
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Simple linear model (replace with actual model)
        logits = np.dot(features, self.weights) + self.bias
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        prob = float(probs[0])
        
        # Threshold for signal generation
        threshold = self.params.get('signal_threshold', 0.5)
        
        if prob > threshold + 0.1:
            signal = SignalType.LONG
        elif prob < threshold - 0.1:
            signal = SignalType.SHORT
        else:
            signal = SignalType.FLAT
        
        return signal, prob
    
    def predict_batch(self, features: np.ndarray) -> Tuple[List[SignalType], List[float]]:
        """Batch prediction"""
        signals = []
        probs = []
        
        for i in range(len(features)):
            signal, prob = self.predict(features[i])
            signals.append(signal)
            probs.append(prob)
        
        return signals, probs


@ray.remote(max_calls=500)
class MetaLabeler:
    """
    Meta-labeling model (secondary model).
    Learns to predict when primary model signals are correct.
    Implements Lopez de Prado's meta-labeling architecture.
    """
    
    def __init__(self, model_params: Dict[str, Any]):
        self.params = model_params
        self.device_config = get_device_config()
        self.model = None
        self.training_data: List[Tuple[np.ndarray, int]] = []
        self._init_model()
    
    def _init_model(self):
        """Initialize meta-labeler model"""
        # Meta-labeler typically uses different features than primary
        # Focuses on market regime, volatility, signal confidence, etc.
        input_dim = self.params.get('meta_input_dim', 15)
        self.weights = np.random.randn(input_dim)
        self.bias = 0.0
        logger.info(f"Meta-labeler initialized on {self.device_config['device']}")
    
    def prepare_meta_features(
        self,
        primary_features: np.ndarray,
        primary_prob: float,
        market_context: Dict[str, float]
    ) -> np.ndarray:
        """
        Prepare features for meta-labeler.
        Combines primary model output with market context.
        """
        # Primary model confidence
        confidence = abs(primary_prob - 0.5) * 2  # Normalize to 0-1
        
        # Market context features
        volatility = market_context.get('volatility', 0.0)
        trend_strength = market_context.get('trend_strength', 0.0)
        volume_ratio = market_context.get('volume_ratio', 1.0)
        spread = market_context.get('spread_bps', 10.0)
        
        # Technical indicators (would be computed in production)
        rsi = market_context.get('rsi', 50.0) / 100.0
        macd_signal = market_context.get('macd_signal', 0.0)
        
        # Combine all features
        meta_features = np.array([
            confidence,
            primary_prob,
            volatility,
            trend_strength,
            volume_ratio,
            spread / 100.0,  # Normalize
            rsi,
            macd_signal,
            # Add interaction terms
            confidence * volatility,
            confidence * trend_strength,
            primary_prob * volatility,
            # Regime indicators
            1.0 if volatility > 0.02 else 0.0,
            1.0 if trend_strength > 0.01 else 0.0,
            1.0 if rsi > 0.7 else 0.0,
            1.0 if rsi < 0.3 else 0.0,
        ])
        
        return meta_features
    
    def predict(
        self,
        meta_features: np.ndarray,
        primary_signal: SignalType
    ) -> Tuple[int, float]:
        """
        Predict whether to take the primary signal.
        Returns meta-label (1=take, 0=filter) and probability.
        """
        if primary_signal == SignalType.FLAT:
            return 0, 0.0  # No signal to validate
        
        if len(meta_features.shape) == 1:
            meta_features = meta_features.reshape(1, -1)
        
        # Simple model (replace with trained model)
        logits = np.dot(meta_features, self.weights) + self.bias
        probs = 1 / (1 + np.exp(-logits))
        
        prob = float(probs[0])
        
        # Threshold for taking signal
        threshold = self.params.get('meta_threshold', 0.5)
        label = 1 if prob > threshold else 0
        
        return label, prob
    
    def train_step(
        self,
        meta_features: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Train meta-labeler on labeled data.
        Labels indicate whether primary signal was profitable.
        """
        if len(meta_features) == 0:
            return 0.0
        
        # Simple gradient descent update (replace with proper training)
        predictions = 1 / (1 + np.exp(-(np.dot(meta_features, self.weights) + self.bias)))
        errors = predictions - labels
        
        # Gradient update
        lr = self.params.get('learning_rate', 0.01)
        gradient = np.dot(meta_features.T, errors) / len(meta_features)
        self.weights -= lr * gradient
        self.bias -= lr * np.mean(errors)
        
        # Return loss
        loss = -np.mean(labels * np.log(predictions + 1e-8) + 
                       (1 - labels) * np.log(1 - predictions + 1e-8))
        
        return float(loss)
    
    def add_training_sample(self, features: np.ndarray, label: int):
        """Add sample to training buffer"""
        self.training_data.append((features, label))
        
        # Limit buffer size for memory efficiency
        max_samples = self.params.get('max_buffer_size', 10000)
        if len(self.training_data) > max_samples:
            self.training_data.pop(0)


class BetSizer:
    """
    Dynamic bet sizing based on meta-labeling probabilities.
    Implements Kelly criterion and fractional Kelly for position sizing.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_size: float = 0.1,
        min_confidence: float = 0.5
    ):
        self.kelly_fraction = kelly_fraction  # Fractional Kelly
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.win_rate_history: List[float] = []
    
    def calculate_bet_size(
        self,
        meta_probability: float,
        primary_signal: SignalType,
        current_volatility: float = 0.01
    ) -> float:
        """
        Calculate position size using modified Kelly criterion.
        
        Args:
            meta_probability: Probability from meta-labeler that signal is good
            primary_signal: Primary model signal direction
            current_volatility: Current market volatility
        
        Returns:
            Normalized position size (0 to max_position_size)
        """
        if primary_signal == SignalType.FLAT or meta_probability < self.min_confidence:
            return 0.0
        
        # Estimate win probability from meta-labeler
        p_win = meta_probability
        
        # Estimate payoff ratio (simplified - would use historical data)
        # Assume average win/loss ratio of 1.2 for crypto
        b = 1.2
        
        # Kelly fraction: f* = (p * b - q) / b where q = 1 - p
        kelly = (p_win * b - (1 - p_win)) / b
        
        # Apply fractional Kelly for risk management
        fractional_kelly = kelly * self.kelly_fraction
        
        # Adjust for volatility (reduce size in high volatility)
        vol_adjustment = 1.0 / (1.0 + current_volatility * 10)
        
        # Final bet size
        bet_size = abs(fractional_kelly * vol_adjustment)
        
        # Clip to bounds
        bet_size = np.clip(bet_size, 0.0, self.max_position_size)
        
        return float(bet_size)
    
    def update_win_rate(self, was_profitable: bool):
        """Update historical win rate tracking"""
        self.win_rate_history.append(1.0 if was_profitable else 0.0)
        
        # Keep bounded
        if len(self.win_rate_history) > 100:
            self.win_rate_history.pop(0)
    
    def get_current_win_rate(self) -> float:
        """Get recent win rate"""
        if not self.win_rate_history:
            return 0.5
        return float(np.mean(self.win_rate_history))


class MetaLabelingSystem:
    """
    Complete meta-labeling system integrating primary model,
    meta-labeler, and dynamic bet sizing.
    """
    
    def __init__(
        self,
        primary_params: Dict[str, Any],
        meta_params: Dict[str, Any],
        bet_sizing_params: Dict[str, Any],
        memory_ceiling_gb: float = 4.0,
    ):
        self.memory_ceiling_bytes = int(memory_ceiling_gb * 1024**3)
        
        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init(
                object_store_memory=int(self.memory_ceiling_bytes * 0.3),
                _system_config={"max_direct_call_object_size": 1024 * 1024},
            )
            logger.info(f"Ray initialized with {memory_ceiling_gb}GB ceiling for meta-labeling")
        
        # Initialize components
        self.primary_model = PrimaryModel.remote(primary_params)
        self.meta_labeler = MetaLabeler.remote(meta_params)
        self.bet_sizer = BetSizer(**bet_sizing_params)
        
        # Training buffers
        self.pending_labels: List[Tuple[np.ndarray, int, float]] = []  # (features, label, timestamp)
        
        # Statistics
        self.stats = {
            'signals_generated': 0,
            'signals_taken': 0,
            'signals_filtered': 0,
            'profitable_trades': 0,
            'total_trades': 0,
        }
        
        logger.info("Meta-labeling system initialized")
    
    def generate_signal(
        self,
        features: np.ndarray,
        market_context: Dict[str, float]
    ) -> MetaLabelResult:
        """
        Generate trading signal with meta-labeling filter.
        
        Args:
            features: Feature vector for primary model
            market_context: Market state for meta-labeler
        
        Returns:
            MetaLabelResult with signal, filter decision, and bet size
        """
        # Get primary model prediction
        primary_future = self.primary_model.predict.remote(features)
        primary_signal, primary_prob = ray.get(primary_future)
        
        self.stats['signals_generated'] += 1
        
        if primary_signal == SignalType.FLAT:
            return MetaLabelResult(
                primary_signal=primary_signal,
                meta_label=0,
                primary_probability=primary_prob,
                meta_probability=0.0,
                bet_size=0.0,
                expected_return=0.0,
                timestamp_ns=time.time_ns(),
            )
        
        # Prepare meta-features
        meta_future = self.meta_labeler.prepare_meta_features.remote(
            features, primary_prob, market_context
        )
        meta_features = ray.get(meta_future)
        
        # Get meta-label prediction
        meta_future = self.meta_labeler.predict.remote(meta_features, primary_signal)
        meta_label, meta_prob = ray.get(meta_future)
        
        # Calculate bet size
        volatility = market_context.get('volatility', 0.01)
        bet_size = self.bet_sizer.calculate_bet_size(meta_prob, primary_signal, volatility)
        
        # Update statistics
        if meta_label == 1:
            self.stats['signals_taken'] += 1
        else:
            self.stats['signals_filtered'] += 1
        
        # Estimate expected return (simplified)
        expected_return = (meta_prob - 0.5) * 2 * bet_size
        
        return MetaLabelResult(
            primary_signal=primary_signal,
            meta_label=meta_label,
            primary_probability=primary_prob,
            meta_probability=meta_prob,
            bet_size=bet_size,
            expected_return=expected_return,
            timestamp_ns=time.time_ns(),
        )
    
    def record_outcome(
        self,
        result: MetaLabelResult,
        was_profitable: bool,
        return_pct: float
    ):
        """
        Record trade outcome for meta-labeler training.
        
        Args:
            result: Original prediction result
            was_profitable: Whether the trade was profitable
            return_pct: Actual return percentage
        """
        self.stats['total_trades'] += 1
        if was_profitable:
            self.stats['profitable_trades'] += 1
        
        # Update bet sizer win rate
        self.bet_sizer.update_win_rate(was_profitable)
        
        # Create training sample for meta-labeler
        label = 1 if was_profitable else 0
        
        # Store for batch training
        # Note: In production, you'd reconstruct meta-features here
        self.pending_labels.append((result.meta_probability, label, result.timestamp_ns))
        
        # Trigger training when enough samples accumulated
        if len(self.pending_labels) >= 100:
            self._train_meta_labeler()
    
    def _train_meta_labeler(self):
        """Train meta-labeler on accumulated samples"""
        if len(self.pending_labels) < 50:
            return
        
        # Convert to arrays (simplified - would need actual features)
        # In production, you'd store and retrieve the actual meta-features
        labels = np.array([label for _, label, _ in self.pending_labels[-100:]])
        
        # For demo, generate random features matching expected shape
        meta_params = {'meta_input_dim': 15}
        features = np.random.randn(len(labels), meta_params['meta_input_dim'])
        
        # Train
        train_future = self.meta_labeler.train_step.remote(features, labels)
        loss = ray.get(train_future)
        
        logger.info(f"Meta-labeler trained, loss: {loss:.4f}")
        
        # Clear old samples (keep recent)
        self.pending_labels = self.pending_labels[-50:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total = self.stats['total_trades']
        win_rate = self.stats['profitable_trades'] / total if total > 0 else 0.0
        
        return {
            **self.stats,
            'win_rate': win_rate,
            'filter_rate': self.stats['signals_filtered'] / self.stats['signals_generated'] 
                          if self.stats['signals_generated'] > 0 else 0.0,
            'current_win_rate': self.bet_sizer.get_current_win_rate(),
        }
    
    def shutdown(self):
        """Shutdown Ray actors and release resources"""
        try:
            ray.kill(self.primary_model)
            ray.kill(self.meta_labeler)
        except Exception:
            pass
        
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("Meta-labeling system shutdown complete")


import time


# Example usage
if __name__ == "__main__":
    # Configuration
    primary_params = {
        'input_dim': 10,
        'signal_threshold': 0.5,
    }
    
    meta_params = {
        'meta_input_dim': 15,
        'meta_threshold': 0.55,  # Slightly higher to be selective
        'learning_rate': 0.01,
        'max_buffer_size': 5000,
    }
    
    bet_params = {
        'kelly_fraction': 0.25,
        'max_position_size': 0.05,
        'min_confidence': 0.5,
    }
    
    # Create system
    system = MetaLabelingSystem(
        primary_params=primary_params,
        meta_params=meta_params,
        bet_sizing_params=bet_params,
        memory_ceiling_gb=4.0,
    )
    
    # Test signal generation
    features = np.random.randn(10)
    market_context = {
        'volatility': 0.015,
        'trend_strength': 0.008,
        'volume_ratio': 1.2,
        'spread_bps': 5.0,
        'rsi': 55.0,
        'macd_signal': 0.002,
    }
    
    result = system.generate_signal(features, market_context)
    
    print(f"\n=== Meta-Labeling Result ===")
    print(f"Primary Signal: {result.primary_signal.name}")
    print(f"Primary Probability: {result.primary_probability:.3f}")
    print(f"Meta-Label: {result.meta_label} (1=Take, 0=Filter)")
    print(f"Meta Probability: {result.meta_probability:.3f}")
    print(f"Bet Size: {result.bet_size:.4f}")
    print(f"Expected Return: {result.expected_return:.4f}")
    
    # Simulate outcome recording
    if result.meta_label == 1:
        was_profitable = np.random.random() > 0.45  # Simulated
        system.record_outcome(result, was_profitable, 0.02 if was_profitable else -0.01)
    
    # Print stats
    stats = system.get_statistics()
    print(f"\n=== Statistics ===")
    print(f"Signals Generated: {stats['signals_generated']}")
    print(f"Signals Taken: {stats['signals_taken']}")
    print(f"Signals Filtered: {stats['signals_filtered']}")
    print(f"Filter Rate: {stats['filter_rate']:.2%}")
    if stats['total_trades'] > 0:
        print(f"Win Rate: {stats['win_rate']:.2%}")
    
    # Cleanup
    system.shutdown()
