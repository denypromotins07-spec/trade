"""
python/factors/timing_models.py

Dynamic Factor Timing Models Using Kalman Filters

Implements Kalman filter-based regime detection to scale exposure to momentum
or mean-reversion factors based on real-time macroeconomic regime states.
Includes AMD ROCm/DirectML acceleration checks.

Memory Constraint: O(1) state storage per Kalman filter, no history required.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import os
import torch


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
class KalmanConfig:
    """Configuration for Kalman filter timing model."""
    state_dim: int = 2  # [level, slope]
    obs_dim: int = 1    # Observed factor return
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    initial_state: Optional[np.ndarray] = None
    initial_covariance: Optional[np.ndarray] = None


class KalmanFilter:
    """
    Standard Kalman filter for state estimation.
    Used to estimate latent regime state from noisy observations.
    """
    
    def __init__(self, config: KalmanConfig):
        self.config = config
        self.state_dim = config.state_dim
        self.obs_dim = config.obs_dim
        
        # State transition matrix (random walk + momentum)
        self.F = np.eye(config.state_dim)
        self.F[0, 1] = 1.0  # Level depends on slope
        
        # Observation matrix
        self.H = np.zeros((config.obs_dim, config.state_dim))
        self.H[0, 0] = 1.0  # Observe level
        
        # Process noise covariance
        self.Q = np.eye(config.state_dim) * config.process_noise
        
        # Measurement noise covariance
        self.R = np.eye(config.obs_dim) * config.measurement_noise
        
        # State estimate
        if config.initial_state is not None:
            self.x = config.initial_state.copy()
        else:
            self.x = np.zeros(config.state_dim)
        
        # State covariance
        if config.initial_covariance is not None:
            self.P = config.initial_covariance.copy()
        else:
            self.P = np.eye(config.state_dim)
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with new observation."""
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x.copy(), self.P.copy()
    
    def filter(self, observations: np.ndarray) -> np.ndarray:
        """Run filter over sequence of observations."""
        states = []
        for z in observations:
            self.predict()
            x, _ = self.update(np.atleast_1d(z))
            states.append(x.copy())
        return np.array(states)


class RegimeDetector:
    """
    Multi-filter regime detector using parallel Kalman filters.
    Detects momentum vs mean-reversion regimes.
    """
    
    def __init__(self, num_regimes: int = 3):
        self.num_regimes = num_regimes
        self.acceleration = check_amd_acceleration()
        
        # Create filters for different regime hypotheses
        self.filters = {
            'momentum': KalmanFilter(KalmanConfig(
                process_noise=0.02,  # Higher noise = more adaptive
                measurement_noise=0.05,
            )),
            'mean_reversion': KalmanFilter(KalmanConfig(
                process_noise=0.005,  # Lower noise = more stable
                measurement_noise=0.1,
            )),
            'neutral': KalmanFilter(KalmanConfig(
                process_noise=0.01,
                measurement_noise=0.1,
            )),
        }
        
        # Regime probabilities (uniform prior)
        self.regime_probs = {k: 1.0/num_regimes for k in self.filters.keys()}
        
        # Log-likelihood for each regime
        self.log_likelihoods = {k: 0.0 for k in self.filters.keys()}
    
    def update(self, observation: float) -> str:
        """
        Update regime detector with new observation.
        
        Returns:
            Current detected regime name
        """
        # Update each filter and compute likelihood
        for name, filt in self.filters.items():
            filt.predict()
            
            z = np.array([observation])
            x_pred = filt.H @ filt.x
            
            # Innovation
            y = z - x_pred
            
            # Innovation covariance
            S = filt.H @ filt.P @ filt.H.T + filt.R
            
            # Log-likelihood (Gaussian)
            log_lik = -0.5 * (
                np.log(2 * np.pi) + 
                np.log(S[0, 0]) + 
                y[0]**2 / S[0, 0]
            )
            
            self.log_likelihoods[name] = log_lik
            
            # Update filter
            filt.update(z)
        
        # Update regime probabilities using Bayes rule
        log_probs = {}
        for name in self.filters.keys():
            log_probs[name] = (
                np.log(self.regime_probs[name] + 1e-10) + 
                self.log_likelihoods[name]
            )
        
        # Normalize (log-sum-exp trick)
        max_log_prob = max(log_probs.values())
        probs = {}
        total = 0.0
        for name, lp in log_probs.items():
            p = np.exp(lp - max_log_prob)
            probs[name] = p
            total += p
        
        for name in probs:
            self.regime_probs[name] = probs[name] / total
        
        # Return most likely regime
        return max(self.regime_probs, key=self.regime_probs.get)
    
    def get_regime_probability(self, regime: str) -> float:
        """Get probability of specific regime."""
        return self.regime_probs.get(regime, 0.0)
    
    def get_all_probabilities(self) -> Dict[str, float]:
        """Get all regime probabilities."""
        return self.regime_probs.copy()


class FactorTimingModel:
    """
    Complete factor timing system that scales exposure based on regime.
    """
    
    def __init__(self, scaling_params: Optional[Dict[str, float]] = None):
        self.regime_detector = RegimeDetector()
        self.acceleration = check_amd_acceleration()
        
        # Exposure scaling by regime
        self.scaling_params = scaling_params or {
            'momentum': 1.5,       # Overweight in momentum regime
            'mean_reversion': 1.5,  # Overweight in MR regime  
            'neutral': 0.5,        # Underweight in uncertain regime
        }
        
        # Current regime and exposure
        self.current_regime = 'neutral'
        self.current_exposure = 1.0
    
    def update_and_get_exposure(self, factor_return: float) -> Tuple[str, float]:
        """
        Update regime estimate and return recommended exposure.
        
        Args:
            factor_return: Latest factor return observation
            
        Returns:
            Tuple of (detected_regime, recommended_exposure)
        """
        # Update regime detector
        self.current_regime = self.regime_detector.update(factor_return)
        
        # Get regime probability-weighted exposure
        probs = self.regime_detector.get_all_probabilities()
        
        exposure = 0.0
        for regime, prob in probs.items():
            scale = self.scaling_params.get(regime, 1.0)
            exposure += prob * scale
        
        self.current_exposure = exposure
        
        return self.current_regime, exposure
    
    def get_regime_confidence(self) -> float:
        """Get confidence in current regime estimate (max probability)."""
        probs = self.regime_detector.get_all_probabilities()
        return max(probs.values())
    
    def reset(self) -> None:
        """Reset all internal state."""
        self.regime_detector = RegimeDetector()
        self.current_regime = 'neutral'
        self.current_exposure = 1.0


if __name__ == "__main__":
    print("Factor Timing Models - AMD Acceleration:", check_amd_acceleration())
    
    # Test Kalman filter
    kf_config = KalmanConfig()
    kf = KalmanFilter(kf_config)
    
    # Simulate observations
    np.random.seed(42)
    true_state = np.array([1.0, 0.1])
    observations = []
    for t in range(100):
        true_state = kf.F @ true_state + np.random.randn(2) * 0.1
        obs = kf.H @ true_state + np.random.randn(1) * 0.3
        observations.append(obs[0])
    
    # Filter
    states = kf.filter(np.array(observations))
    print(f"Kalman filter final state: {states[-1]}")
    
    # Test regime detector
    detector = RegimeDetector()
    for obs in observations[:50]:
        regime = detector.update(obs)
    
    print(f"Detected regime: {regime}")
    print(f"Regime probabilities: {detector.get_all_probabilities()}")
    
    # Test factor timing
    timing = FactorTimingModel()
    for obs in observations[:50]:
        regime, exposure = timing.update_and_get_exposure(obs)
    
    print(f"Current regime: {timing.current_regime}")
    print(f"Recommended exposure: {timing.current_exposure:.2f}")
    print(f"Regime confidence: {timing.get_regime_confidence():.2%}")
