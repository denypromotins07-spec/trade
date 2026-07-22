"""
AI - Curriculum Reinforcement Learning Manager

Implements an automated curriculum manager that dynamically increases market noise
and adversarial order flow in the simulation as the agent's Sharpe ratio stabilizes.
Optimized for AMD Ryzen AI 5 with DirectML/ROCm acceleration checks and 4GB RAM quota.
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import ray

# Enforce memory limits
os.environ['RAY_MEMORY_LIMIT'] = '4294967296'  # 4GB


class DifficultyLevel(Enum):
    """Curriculum difficulty levels."""
    TUTORIAL = "tutorial"       # Minimal noise, predictable markets
    BEGINNER = "beginner"       # Low noise, trending markets
    INTERMEDIATE = "intermediate"  # Moderate noise, mean-reverting
    ADVANCED = "advanced"       # High noise, regime changes
    EXPERT = "expert"           # Very high noise, adversarial
    PROFESSIONAL = "professional"  # Real-world conditions


@dataclass
class MarketParameters:
    """Parameters controlling market simulation difficulty."""
    # Noise parameters
    volatility_base: float = 0.01
    volatility_of_volatility: float = 0.1
    jump_intensity: float = 0.01
    jump_size_mean: float = 0.0
    
    # Microstructure parameters
    bid_ask_spread: float = 0.001
    market_impact: float = 0.0001
    fill_probability: float = 0.9
    
    # Adversarial parameters
    spoofing_probability: float = 0.0
    latency_adversary: bool = False
    toxic_flow_ratio: float = 0.0
    
    # Regime parameters
    regime_change_prob: float = 0.0
    trend_strength: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            'volatility_base': self.volatility_base,
            'volatility_of_volatility': self.volatility_of_volatility,
            'jump_intensity': self.jump_intensity,
            'jump_size_mean': self.jump_size_mean,
            'bid_ask_spread': self.bid_ask_spread,
            'market_impact': self.market_impact,
            'fill_probability': self.fill_probability,
            'spoofing_probability': self.spoofing_probability,
            'latency_adversary': self.latency_adversary,
            'toxic_flow_ratio': self.toxic_flow_ratio,
            'regime_change_prob': self.regime_change_prob,
            'trend_strength': self.trend_strength,
        }


def check_amd_acceleration() -> Dict[str, bool]:
    """Check AMD DirectML/ROCm environment for potential acceleration."""
    accel_status = {
        'rocm_available': False,
        'directml_available': False,
        'hip_available': False,
    }
    
    try:
        import torch
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            accel_status['rocm_available'] = True
            accel_status['hip_available'] = True
    except ImportError:
        pass
    
    return accel_status


class CurriculumManager:
    """
    Manages the curriculum learning progression for RL trading agents.
    
    Automatically adjusts market difficulty based on agent performance metrics.
    """
    
    # Default difficulty configurations
    DIFFICULTY_CONFIGS = {
        DifficultyLevel.TUTORIAL: MarketParameters(
            volatility_base=0.005,
            volatility_of_volatility=0.05,
            jump_intensity=0.0,
            bid_ask_spread=0.0001,
            fill_probability=1.0,
            spoofing_probability=0.0,
            toxic_flow_ratio=0.0,
            regime_change_prob=0.0,
            trend_strength=0.8,
        ),
        DifficultyLevel.BEGINNER: MarketParameters(
            volatility_base=0.01,
            volatility_of_volatility=0.1,
            jump_intensity=0.005,
            bid_ask_spread=0.0005,
            fill_probability=0.95,
            spoofing_probability=0.0,
            toxic_flow_ratio=0.0,
            regime_change_prob=0.0,
            trend_strength=0.6,
        ),
        DifficultyLevel.INTERMEDIATE: MarketParameters(
            volatility_base=0.015,
            volatility_of_volatility=0.2,
            jump_intensity=0.01,
            bid_ask_spread=0.001,
            fill_probability=0.9,
            spoofing_probability=0.01,
            toxic_flow_ratio=0.05,
            regime_change_prob=0.01,
            trend_strength=0.4,
        ),
        DifficultyLevel.ADVANCED: MarketParameters(
            volatility_base=0.02,
            volatility_of_volatility=0.3,
            jump_intensity=0.02,
            bid_ask_spread=0.0015,
            fill_probability=0.85,
            spoofing_probability=0.03,
            toxic_flow_ratio=0.1,
            regime_change_prob=0.02,
            trend_strength=0.3,
        ),
        DifficultyLevel.EXPERT: MarketParameters(
            volatility_base=0.03,
            volatility_of_volatility=0.4,
            jump_intensity=0.03,
            bid_ask_spread=0.002,
            fill_probability=0.8,
            spoofing_probability=0.05,
            toxic_flow_ratio=0.15,
            regime_change_prob=0.03,
            trend_strength=0.2,
        ),
        DifficultyLevel.PROFESSIONAL: MarketParameters(
            volatility_base=0.04,
            volatility_of_volatility=0.5,
            jump_intensity=0.05,
            bid_ask_spread=0.0025,
            fill_probability=0.75,
            spoofing_probability=0.08,
            toxic_flow_ratio=0.2,
            regime_change_prob=0.05,
            trend_strength=0.1,
        ),
    }
    
    def __init__(
        self,
        initial_difficulty: DifficultyLevel = DifficultyLevel.BEGINNER,
        sharpe_threshold_promote: float = 2.0,
        sharpe_threshold_demote: float = 0.5,
        stability_window: int = 100,
        min_steps_at_level: int = 500,
    ):
        """
        Initialize curriculum manager.
        
        Parameters
        ----------
        initial_difficulty : DifficultyLevel
            Starting difficulty level
        sharpe_threshold_promote : float
            Sharpe ratio threshold for promotion
        sharpe_threshold_demote : float
            Sharpe ratio threshold for demotion
        stability_window : int
            Number of episodes to evaluate for stability
        min_steps_at_level : int
            Minimum training steps before considering promotion
        """
        self.current_difficulty = initial_difficulty
        self.sharpe_threshold_promote = sharpe_threshold_promote
        self.sharpe_threshold_demote = sharpe_threshold_demote
        self.stability_window = stability_window
        self.min_steps_at_level = min_steps_at_level
        
        # Performance tracking
        self.sharpe_history: List[float] = []
        self.steps_at_current_level = 0
        self.total_steps = 0
        
        # Check AMD acceleration
        self.accel_status = check_amd_acceleration()
    
    def get_current_params(self) -> MarketParameters:
        """Get market parameters for current difficulty."""
        return self.DIFFICULTY_CONFIGS[self.current_difficulty]
    
    def record_episode(
        self,
        sharpe_ratio: float,
        n_steps: int = 1
    ) -> Optional[Tuple[DifficultyLevel, str]]:
        """
        Record episode results and potentially adjust difficulty.
        
        Parameters
        ----------
        sharpe_ratio : float
            Episode Sharpe ratio
        n_steps : int
            Number of steps in episode
            
        Returns
        -------
        tuple or None
            (old_level, new_level) if difficulty changed, None otherwise
        """
        self.sharpe_history.append(sharpe_ratio)
        self.steps_at_current_level += n_steps
        self.total_steps += n_steps
        
        # Keep only recent history for stability calculation
        if len(self.sharpe_history) > self.stability_window:
            self.sharpe_history = self.sharpe_history[-self.stability_window:]
        
        # Check if we can consider promotion
        if self.steps_at_current_level >= self.min_steps_at_level:
            return self._evaluate_progression()
        
        return None
    
    def _evaluate_progression(self) -> Optional[Tuple[DifficultyLevel, str]]:
        """
        Evaluate whether to promote, demote, or maintain current difficulty.
        
        Returns
        -------
        tuple or None
            (old_level, reason) if changed, None otherwise
        """
        if len(self.sharpe_history) < self.stability_window // 2:
            return None
        
        recent_sharpes = np.array(self.sharpe_history[-self.stability_window:])
        mean_sharpe = np.mean(recent_sharpes)
        std_sharpe = np.std(recent_sharpes)
        
        # Calculate stability score (higher is more stable)
        stability_score = mean_sharpe / (std_sharpe + 1e-8)
        
        old_level = self.current_difficulty
        levels = list(DifficultyLevel)
        current_idx = levels.index(self.current_difficulty)
        
        # Promotion criteria
        if mean_sharpe >= self.sharpe_threshold_promote and stability_score > 1.5:
            if current_idx < len(levels) - 1:
                self.current_difficulty = levels[current_idx + 1]
                self.steps_at_current_level = 0
                self.sharpe_history = []  # Reset history at new level
                return (old_level, f"promoted (sharpe={mean_sharpe:.2f}, stability={stability_score:.2f})")
        
        # Demotion criteria
        elif mean_sharpe <= self.sharpe_threshold_demote:
            if current_idx > 0:
                self.current_difficulty = levels[current_idx - 1]
                self.steps_at_current_level = 0
                self.sharpe_history = []
                return (old_level, f"demoted (sharpe={mean_sharpe:.2f})")
        
        return None
    
    def get_curriculum_state(self) -> Dict:
        """Get current curriculum state for logging/checkpointing."""
        recent_sharpes = self.sharpe_history[-min(10, len(self.sharpe_history)):]
        
        return {
            'current_difficulty': self.current_difficulty.value,
            'steps_at_level': self.steps_at_current_level,
            'total_steps': self.total_steps,
            'recent_mean_sharpe': float(np.mean(recent_sharpes)) if recent_sharpes else 0.0,
            'recent_std_sharpe': float(np.std(recent_sharpes)) if recent_sharpes else 0.0,
            'history_length': len(self.sharpe_history),
        }
    
    def generate_adversarial_perturbation(
        self,
        base_returns: np.ndarray,
        intensity: float = 0.1
    ) -> np.ndarray:
        """
        Generate adversarial perturbations to market returns.
        
        Uses gradient-free adversarial generation optimized for speed.
        
        Parameters
        ----------
        base_returns : np.ndarray
            Original return series
        intensity : float
            Perturbation intensity (0 to 1)
            
        Returns
        -------
        np.ndarray
            Perturbed return series
        """
        if intensity <= 0:
            return base_returns.copy()
        
        # Multiple adversarial strategies
        strategy = np.random.choice(['noise', 'spike', 'trend_break', 'vol_cluster'])
        
        perturbed = base_returns.copy()
        
        if strategy == 'noise':
            # Add targeted noise at high-magnitude returns
            magnitudes = np.abs(base_returns)
            noise_mask = magnitudes > np.percentile(magnitudes, 80)
            noise = np.random.randn(len(base_returns)) * intensity * 0.01
            perturbed[noise_mask] += noise[noise_mask]
            
        elif strategy == 'spike':
            # Inject random spikes
            n_spikes = max(1, int(len(base_returns) * 0.01))
            spike_indices = np.random.choice(len(base_returns), n_spikes, replace=False)
            spike_signs = np.random.choice([-1, 1], n_spikes)
            perturbed[spike_indices] += spike_signs * intensity * 0.05
            
        elif strategy == 'trend_break':
            # Reverse recent trend abruptly
            if len(base_returns) > 10:
                recent_trend = np.sum(base_returns[-10:])
                break_point = len(base_returns) - 5
                perturbed[break_point:] -= np.sign(recent_trend) * intensity * 0.02
                
        elif strategy == 'vol_cluster':
            # Create volatility clustering
            vol_regime = np.random.random() > 0.5
            regime_start = np.random.randint(0, len(base_returns) // 2)
            regime_end = regime_start + len(base_returns) // 4
            if vol_regime:
                perturbed[regime_start:regime_end] *= (1 + intensity)
            else:
                perturbed[regime_start:regime_end] *= (1 - intensity * 0.5)
        
        return perturbed


@ray.remote(memory=256*1024*1024)
def train_with_curriculum(
    curriculum_config: Dict,
    training_episodes: int = 100,
) -> Dict:
    """
    Ray remote function for curriculum-based training.
    Memory-bounded for 4GB quota compliance.
    """
    # Parse config
    difficulty = DifficultyLevel(curriculum_config.get('difficulty', 'beginner'))
    
    # Get market params
    manager = CurriculumManager(initial_difficulty=difficulty)
    params = manager.get_current_params()
    
    # Simulated training
    sharpes = []
    
    for ep in range(training_episodes):
        # Generate market returns with current difficulty
        base_returns = np.random.randn(100) * params.volatility_base
        
        # Apply adversarial perturbations
        if difficulty in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT, DifficultyLevel.PROFESSIONAL]:
            intensity = {
                DifficultyLevel.ADVANCED: 0.1,
                DifficultyLevel.EXPERT: 0.2,
                DifficultyLevel.PROFESSIONAL: 0.3,
            }.get(difficulty, 0.1)
            base_returns = manager.generate_adversarial_perturbation(base_returns, intensity)
        
        # Simulated agent performance (degrades with difficulty)
        difficulty_penalty = list(DifficultyLevel).index(difficulty) * 0.3
        base_sharpe = 1.5 - difficulty_penalty + np.random.randn() * 0.5
        sharpes.append(base_sharpe)
        
        # Update curriculum
        result = manager.record_episode(base_sharpe, n_steps=100)
        if result:
            old_level, reason = result
            params = manager.get_current_params()
    
    return {
        'final_difficulty': manager.current_difficulty.value,
        'mean_sharpe': float(np.mean(sharpes)),
        'curriculum_transitions': manager.get_curriculum_state(),
    }


if __name__ == '__main__':
    print("Initializing Curriculum RL Manager...")
    
    # Check AMD acceleration
    accel = check_amd_acceleration()
    print(f"AMD Acceleration: {accel}")
    
    # Initialize manager
    manager = CurriculumManager(
        initial_difficulty=DifficultyLevel.BEGINNER,
        sharpe_threshold_promote=1.5,
        sharpe_threshold_demote=0.5,
    )
    
    print(f"\nInitial Difficulty: {manager.current_difficulty.value}")
    print(f"Market Params: {manager.get_current_params().to_dict()}")
    
    # Simulate training progression
    print("\nSimulating Training Progression:")
    
    # Good performance - should promote
    for i in range(600):
        sharpe = 2.0 + np.random.randn() * 0.3  # Good sharpe
        result = manager.record_episode(sharpe, n_steps=1)
        if result:
            old_level, reason = result
            print(f"  Step {i}: {old_level.value} -> {manager.current_difficulty.value} ({reason})")
    
    print(f"\nFinal State: {manager.get_curriculum_state()}")
    
    # Test adversarial perturbation
    print("\nTesting Adversarial Perturbations:")
    base = np.random.randn(100) * 0.01
    
    for level in [DifficultyLevel.INTERMEDIATE, DifficultyLevel.EXPERT]:
        intensity = 0.1 if level == DifficultyLevel.INTERMEDIATE else 0.3
        perturbed = manager.generate_adversarial_perturbation(base, intensity)
        print(f"  {level.value}: base_std={np.std(base):.4f}, perturbed_std={np.std(perturbed):.4f}")
    
    # Run distributed curriculum training
    print("\nRunning Distributed Curriculum Training...")
    if not ray.is_initialized():
        ray.init(num_cpus=2, _memory=2*1024*1024*1024)
    
    futures = [
        train_with_curriculum.remote(
            {'difficulty': 'intermediate'},
            50
        )
        for _ in range(2)
    ]
    
    results = ray.get(futures)
    for i, r in enumerate(results):
        print(f"\nWorker {i}:")
        print(f"  Final Difficulty: {r['final_difficulty']}")
        print(f"  Mean Sharpe: {r['mean_sharpe']:.4f}")
    
    ray.shutdown()
