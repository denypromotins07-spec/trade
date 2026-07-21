"""
Automated Curriculum Learning Manager for RL Agents

This module implements curriculum learning that progressively increases market
noise and slippage in the simulation environment as agents demonstrate mastery.
Enables gradual skill acquisition from simple to complex market conditions.

Optimized for AMD Ryzen AI 5 with DirectML/ROCm checks.
Respects strict 4GB Python RAM quota during Ray distribution.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import ray
from ray import tune
from ray.tune import ExperimentAnalysis

# AMD DirectML/ROCm environment check
def check_amd_acceleration() -> Dict[str, Any]:
    """Check for AMD DirectML/ROCm availability."""
    config = {
        "directml_available": False,
        "rocm_available": False,
        "gpu_device": None,
        "recommended_backend": "cpu"
    }
    
    try:
        # Check for ROCm
        try:
            import torch
            if torch.version.hip is not None:
                config["rocm_available"] = True
                config["gpu_device"] = f"ROCm ({torch.cuda.get_device_name(0)})"
                config["recommended_backend"] = "cuda"
        except (ImportError, AttributeError):
            pass
        
        # Check for DirectML (Windows)
        if os.name == 'nt':
            try:
                import torch_directml
                config["directml_available"] = True
                config["gpu_device"] = "DirectML"
                config["recommended_backend"] = "dml"
            except ImportError:
                pass
                
    except Exception as e:
        print(f"[WARN] AMD acceleration check failed: {e}")
    
    return config


class CurriculumStage(Enum):
    """Curriculum difficulty stages."""
    NOVICE = 0       # Minimal noise, no slippage
    BEGINNER = 1     # Low noise, minimal slippage
    INTERMEDIATE = 2 # Moderate noise, realistic slippage
    ADVANCED = 3     # High noise, variable slippage
    EXPERT = 4       # Extreme noise, adversarial slippage
    MASTER = 5       # Real-world market conditions


@dataclass
class MarketConditions:
    """Market condition parameters for each curriculum stage."""
    stage: CurriculumStage
    
    # Price dynamics
    base_volatility: float = 0.001
    noise_factor: float = 1.0
    drift_magnitude: float = 0.0
    
    # Execution costs
    slippage_factor: float = 0.0
    fee_multiplier: float = 1.0
    
    # Market microstructure
    spread_bps: float = 1.0
    impact_coefficient: float = 0.0001
    
    # Regime changes
    regime_change_prob: float = 0.0
    crash_prob: float = 0.0
    
    @classmethod
    def get_stage_config(cls, stage: CurriculumStage) -> "MarketConditions":
        """Get predefined configuration for each stage."""
        configs = {
            CurriculumStage.NOVICE: cls(
                stage=stage,
                base_volatility=0.0005,
                noise_factor=0.1,
                slippage_factor=0.0,
                spread_bps=0.5,
            ),
            CurriculumStage.BEGINNER: cls(
                stage=stage,
                base_volatility=0.001,
                noise_factor=0.3,
                slippage_factor=0.0001,
                spread_bps=1.0,
            ),
            CurriculumStage.INTERMEDIATE: cls(
                stage=stage,
                base_volatility=0.002,
                noise_factor=0.5,
                slippage_factor=0.0002,
                spread_bps=2.0,
                impact_coefficient=0.0002,
            ),
            CurriculumStage.ADVANCED: cls(
                stage=stage,
                base_volatility=0.003,
                noise_factor=0.7,
                slippage_factor=0.0003,
                spread_bps=3.0,
                impact_coefficient=0.0003,
                regime_change_prob=0.01,
            ),
            CurriculumStage.EXPERT: cls(
                stage=stage,
                base_volatility=0.005,
                noise_factor=1.0,
                slippage_factor=0.0005,
                spread_bps=5.0,
                impact_coefficient=0.0005,
                regime_change_prob=0.02,
                crash_prob=0.001,
            ),
            CurriculumStage.MASTER: cls(
                stage=stage,
                base_volatility=0.008,
                noise_factor=1.5,
                slippage_factor=0.0008,
                spread_bps=8.0,
                impact_coefficient=0.001,
                regime_change_prob=0.05,
                crash_prob=0.005,
            ),
        }
        return configs[stage]


@dataclass
class PerformanceMetrics:
    """Agent performance metrics for curriculum progression."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    episodes_completed: int = 0
    avg_episode_length: float = 0.0
    
    def meets_thresholds(
        self,
        min_return: float = 0.0,
        min_sharpe: float = 0.5,
        max_drawdown: float = -0.2,
        min_win_rate: float = 0.4,
        min_episodes: int = 50,
    ) -> bool:
        """Check if performance meets promotion thresholds."""
        return (
            self.total_return >= min_return and
            self.sharpe_ratio >= min_sharpe and
            self.max_drawdown >= max_drawdown and
            self.win_rate >= min_win_rate and
            self.episodes_completed >= min_episodes
        )


class CurriculumManager:
    """
    Manages curriculum progression for RL agents.
    
    Automatically adjusts market difficulty based on agent performance,
    implementing a mastery-based learning approach.
    """
    
    def __init__(
        self,
        initial_stage: CurriculumStage = CurriculumStage.NOVICE,
        ram_limit_gb: float = 4.0,
    ):
        self.current_stage = initial_stage
        self.ram_limit_gb = ram_limit_gb
        
        # Performance history per stage
        self.performance_history: Dict[CurriculumStage, List[PerformanceMetrics]] = {
            stage: [] for stage in CurriculumStage
        }
        
        # Promotion thresholds (relaxed for early stages)
        self.promotion_thresholds = {
            CurriculumStage.NOVICE: {"min_return": -0.1, "min_sharpe": 0.0, "min_episodes": 20},
            CurriculumStage.BEGINNER: {"min_return": -0.05, "min_sharpe": 0.3, "min_episodes": 30},
            CurriculumStage.INTERMEDIATE: {"min_return": 0.0, "min_sharpe": 0.5, "min_episodes": 40},
            CurriculumStage.ADVANCED: {"min_return": 0.05, "min_sharpe": 0.8, "min_episodes": 50},
            CurriculumStage.EXPERT: {"min_return": 0.1, "min_sharpe": 1.0, "min_episodes": 60},
            CurriculumStage.MASTER: {"min_return": 0.15, "min_sharpe": 1.2, "min_episodes": 100},
        }
        
        # Demotion thresholds (if agent struggles)
        self.demotion_thresholds = {
            CurriculumStage.BEGINNER: {"max_drawdown": -0.3, "min_sharpe": -0.5},
            CurriculumStage.INTERMEDIATE: {"max_drawdown": -0.25, "min_sharpe": 0.0},
            CurriculumStage.ADVANCED: {"max_drawdown": -0.2, "min_sharpe": 0.3},
            CurriculumStage.EXPERT: {"max_drawdown": -0.15, "min_sharpe": 0.5},
            CurriculumStage.MASTER: {"max_drawdown": -0.1, "min_sharpe": 0.8},
        }
        
        # Current market conditions
        self.current_conditions = MarketConditions.get_stage_config(initial_stage)
        
        # Track memory usage
        self._check_memory()
    
    def _check_memory(self):
        """Check current memory usage against limit."""
        import psutil
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024**3
        
        if memory_gb > self.ram_limit_gb * 0.9:
            print(f"[WARN] Memory usage at {memory_gb:.2f}GB (limit: {self.ram_limit_gb}GB)")
    
    def get_current_conditions(self) -> MarketConditions:
        """Get current market conditions."""
        return self.current_conditions
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record agent performance for current stage."""
        self.performance_history[self.current_stage].append(metrics)
        self._check_memory()
    
    def evaluate_progression(self) -> Tuple[bool, str]:
        """
        Evaluate whether agent should progress to next stage.
        
        Returns:
            (changed, message): Whether stage changed and explanation
        """
        recent_performance = self.performance_history[self.current_stage][-5:]
        
        if len(recent_performance) < 3:
            return False, "Insufficient recent performance data"
        
        # Aggregate recent metrics
        avg_return = np.mean([m.total_return for m in recent_performance])
        avg_sharpe = np.mean([m.sharpe_ratio for m in recent_performance])
        avg_drawdown = np.min([m.max_drawdown for m in recent_performance])
        avg_win_rate = np.mean([m.win_rate for m in recent_performance])
        total_episodes = sum([m.episodes_completed for m in recent_performance])
        
        thresholds = self.promotion_thresholds.get(self.current_stage, {})
        
        # Check for promotion
        if self.current_stage != CurriculumStage.MASTER:
            if (
                avg_return >= thresholds.get("min_return", 0) and
                avg_sharpe >= thresholds.get("min_sharpe", 0.5) and
                avg_drawdown >= thresholds.get("max_drawdown", -0.2) and
                total_episodes >= thresholds.get("min_episodes", 50)
            ):
                next_stage = CurriculumStage(self.current_stage.value + 1)
                self._promote(next_stage)
                return True, f"Promoted to {next_stage.name}"
        
        # Check for demotion (if struggling)
        if self.current_stage != CurriculumStage.NOVICE:
            demotion_thresh = self.demotion_thresholds.get(self.current_stage, {})
            if (
                avg_drawdown <= demotion_thresh.get("max_drawdown", -0.3) or
                avg_sharpe <= demotion_thresh.get("min_sharpe", -0.5)
            ):
                prev_stage = CurriculumStage(self.current_stage.value - 1)
                self._demote(prev_stage)
                return True, f"Demoted to {prev_stage.name} due to poor performance"
        
        return False, "Performance adequate, maintaining current stage"
    
    def _promote(self, new_stage: CurriculumStage):
        """Promote agent to new stage."""
        old_stage = self.current_stage
        self.current_stage = new_stage
        self.current_conditions = MarketConditions.get_stage_config(new_stage)
        
        print(
            f"[CURRICULUM] Promoted from {old_stage.name} to {new_stage.name}\n"
            f"  New conditions: vol={self.current_conditions.base_volatility}, "
            f"slippage={self.current_conditions.slippage_factor}, "
            f"spread={self.current_conditions.spread_bps}bps"
        )
    
    def _demote(self, new_stage: CurriculumStage):
        """Demote agent to previous stage."""
        old_stage = self.current_stage
        self.current_stage = new_stage
        self.current_conditions = MarketConditions.get_stage_config(new_stage)
        
        print(
            f"[CURRICULUM] Demoted from {old_stage.name} to {new_stage.name}\n"
            f"  Reduced difficulty for better learning"
        )
    
    def get_curriculum_progress(self) -> float:
        """Get overall curriculum progress (0.0 to 1.0)."""
        return self.current_stage.value / (len(CurriculumStage) - 1)
    
    def generate_training_env_config(self) -> Dict[str, Any]:
        """Generate environment configuration for current stage."""
        return {
            "base_volatility": self.current_conditions.base_volatility,
            "noise_factor": self.current_conditions.noise_factor,
            "slippage_factor": self.current_conditions.slippage_factor,
            "fee_multiplier": self.current_conditions.fee_multiplier,
            "spread_bps": self.current_conditions.spread_bps,
            "impact_coefficient": self.current_conditions.impact_coefficient,
            "regime_change_prob": self.current_conditions.regime_change_prob,
            "crash_prob": self.current_conditions.crash_prob,
        }


class CurriculumTradingEnv:
    """
    Trading environment wrapper with curriculum-based difficulty.
    Wraps a base trading environment and applies curriculum conditions.
    """
    
    def __init__(self, base_env, curriculum_manager: CurriculumManager):
        self.base_env = base_env
        self.curriculum = curriculum_manager
    
    def reset(self):
        """Reset environment with current curriculum conditions."""
        conditions = self.curriculum.get_current_conditions()
        
        # Apply conditions to base environment
        if hasattr(self.base_env, "set_market_conditions"):
            self.base_env.set_market_conditions(conditions)
        
        return self.base_env.reset()
    
    def step(self, action):
        """Execute step with curriculum-adjusted dynamics."""
        conditions = self.curriculum.get_current_conditions()
        
        # Modify price dynamics based on noise factor
        if hasattr(self.base_env, "_simulate_price"):
            original_simulate = self.base_env._simulate_price
            
            def noisy_simulate():
                base_price = original_simulate()
                noise = np.random.randn() * conditions.base_volatility * conditions.noise_factor
                return base_price * (1 + noise)
            
            self.base_env._simulate_price = noisy_simulate
        
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Apply slippage to execution
        if conditions.slippage_factor > 0:
            slippage_cost = abs(info.get("position_change", 0)) * info.get("price", 0) * conditions.slippage_factor
            reward -= slippage_cost
            info["slippage_cost"] = slippage_cost
        
        return obs, reward, terminated, truncated, info


def train_with_curriculum(
    training_fn: Callable,
    initial_stage: CurriculumStage = CurriculumStage.NOVICE,
    max_stages: int = 6,
    episodes_per_evaluation: int = 50,
    ram_limit_gb: float = 4.0,
) -> Dict[str, Any]:
    """
    Train agent through full curriculum progression.
    
    Args:
        training_fn: Function that trains agent for given environment config
        initial_stage: Starting curriculum stage
        max_stages: Maximum number of stages to progress through
        episodes_per_evaluation: Episodes between progression checks
        ram_limit_gb: Memory limit
    
    Returns:
        Training results dictionary
    """
    # Check AMD acceleration
    amd_config = check_amd_acceleration()
    print(f"[INFO] Using backend: {amd_config['recommended_backend']}")
    
    # Initialize curriculum manager
    curriculum = CurriculumManager(
        initial_stage=initial_stage,
        ram_limit_gb=ram_limit_gb,
    )
    
    results = {
        "stages_completed": [],
        "final_stage": initial_stage.name,
        "total_episodes": 0,
        "performance_history": [],
    }
    
    current_stage = initial_stage
    stage_episodes = 0
    
    print(f"\n[CURRICULUM] Starting training at {initial_stage.name}")
    print("=" * 60)
    
    while curriculum.current_stage.value < max_stages - 1:
        # Get current environment config
        env_config = curriculum.generate_training_env_config()
        
        # Train for evaluation period
        stage_results = training_fn(env_config, episodes=episodes_per_evaluation)
        
        # Record performance
        metrics = PerformanceMetrics(
            total_return=stage_results.get("total_return", 0),
            sharpe_ratio=stage_results.get("sharpe_ratio", 0),
            max_drawdown=stage_results.get("max_drawdown", 0),
            win_rate=stage_results.get("win_rate", 0),
            episodes_completed=episodes_per_evaluation,
        )
        
        curriculum.record_performance(metrics)
        results["total_episodes"] += episodes_per_evaluation
        results["performance_history"].append({
            "stage": curriculum.current_stage.name,
            **vars(metrics),
        })
        
        # Evaluate progression
        changed, message = curriculum.evaluate_progression()
        
        if changed:
            if curriculum.current_stage.value > current_stage.value:
                results["stages_completed"].append(current_stage.name)
            current_stage = curriculum.current_stage
        
        stage_episodes += episodes_per_evaluation
        
        # Progress report
        if stage_episodes % 200 == 0:
            print(
                f"\n[PROGRESS] Stage: {curriculum.current_stage.name} | "
                f"Progress: {curriculum.get_curriculum_progress()*100:.1f}% | "
                f"Episodes: {results['total_episodes']}"
            )
    
    # Final stage
    results["final_stage"] = curriculum.current_stage.name
    results["stages_completed"].append(curriculum.current_stage.name)
    
    print("\n" + "=" * 60)
    print(f"[CURRICULUM] Training complete!")
    print(f"  Final stage: {results['final_stage']}")
    print(f"  Total episodes: {results['total_episodes']}")
    print(f"  Stages completed: {results['stages_completed']}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Checking AMD acceleration...")
    amd_info = check_amd_acceleration()
    print(f"AMD Config: {amd_info}")
    
    print("\nInitializing curriculum training...")
    
    # Mock training function
    def mock_training(env_config, episodes):
        return {
            "total_return": np.random.randn() * 0.1 + 0.05,
            "sharpe_ratio": np.random.randn() * 0.3 + 0.8,
            "max_drawdown": -abs(np.random.randn() * 0.1),
            "win_rate": 0.5 + np.random.randn() * 0.1,
        }
    
    results = train_with_curriculum(
        training_fn=mock_training,
        initial_stage=CurriculumStage.NOVICE,
        episodes_per_evaluation=20,
        ram_limit_gb=4.0,
    )
    
    print(f"\nFinal Results: {results}")
