"""
AI - Multi-Objective Reinforcement Learning (MORL)

Implements Multi-Objective RL using scalarization techniques to simultaneously optimize
for Sharpe ratio, maximum drawdown, and execution slippage on Ray clusters.
Optimized for AMD Ryzen AI 5 with DirectML/ROCm acceleration checks and strict 4GB RAM quota.
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import ray

# Enforce memory limits
os.environ['RAY_MEMORY_LIMIT'] = '4294967296'  # 4GB


class ScalarizationMethod(Enum):
    """Available scalarization methods for MORL."""
    LINEAR = "linear"
    TCHEBYCHEFF = "tchebycheff"
    ACHIEVEMENT_SCALARIZING = "achievement"
    HYPERVOLUME = "hypervolume"


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization."""
    sharpe_ratio: float = 1.0
    max_drawdown: float = 1.0
    execution_slippage: float = 0.5
    transaction_cost: float = 0.3
    inventory_risk: float = 0.2
    
    def normalize(self) -> 'ObjectiveWeights':
        """Normalize weights to sum to 1."""
        total = sum([
            abs(self.sharpe_ratio),
            abs(self.max_drawdown),
            abs(self.execution_slippage),
            abs(self.transaction_cost),
            abs(self.inventory_risk)
        ])
        if total > 0:
            return ObjectiveWeights(
                sharpe_ratio=self.sharpe_ratio / total,
                max_drawdown=self.max_drawdown / total,
                execution_slippage=self.execution_slippage / total,
                transaction_cost=self.transaction_cost / total,
                inventory_risk=self.inventory_risk / total,
            )
        return self
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.sharpe_ratio,
            self.max_drawdown,
            self.execution_slippage,
            self.transaction_cost,
            self.inventory_risk
        ])


@dataclass
class ObjectiveValues:
    """Current objective values."""
    sharpe_ratio: float
    max_drawdown: float  # Negative value (we want to minimize)
    execution_slippage: float  # Negative value
    transaction_cost: float  # Negative value
    inventory_risk: float  # Negative value
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.sharpe_ratio,
            self.max_drawdown,
            self.execution_slippage,
            self.transaction_cost,
            self.inventory_risk
        ])
    
    def to_dict(self) -> Dict:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'execution_slippage': self.execution_slippage,
            'transaction_cost': self.transaction_cost,
            'inventory_risk': self.inventory_risk,
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


class MultiObjectiveScalarizer:
    """
    Implements various scalarization methods for converting multi-objective
    rewards into single scalar rewards for standard RL algorithms.
    """
    
    def __init__(
        self,
        method: ScalarizationMethod = ScalarizationMethod.LINEAR,
        weights: Optional[ObjectiveWeights] = None,
        reference_point: Optional[np.ndarray] = None,
        epsilon: float = 1e-8
    ):
        """
        Initialize scalarizer.
        
        Parameters
        ----------
        method : ScalarizationMethod
            Scalarization method to use
        weights : ObjectiveWeights, optional
            Objective weights (default: equal weights)
        reference_point : np.ndarray, optional
            Reference point for Tchebycheff/Achievement methods
        epsilon : float
            Small constant for numerical stability
        """
        self.method = method
        self.weights = weights.normalize() if weights else ObjectiveWeights().normalize()
        self.epsilon = epsilon
        
        # Default reference point (ideal values)
        if reference_point is None:
            self.reference_point = np.array([3.0, 0.0, 0.0, 0.0, 0.0])  # Ideal values
        else:
            self.reference_point = reference_point
        
        # Check AMD acceleration
        self.accel_status = check_amd_acceleration()
    
    def _linear_scalarize(self, objectives: np.ndarray) -> float:
        """
        Linear weighted sum scalarization.
        
        R = sum(w_i * o_i)
        """
        weights = self.weights.to_array()
        return float(np.dot(weights, objectives))
    
    def _tchebycheff_scalarize(self, objectives: np.ndarray) -> float:
        """
        Tchebycheff scalarization (min-max approach).
        
        R = -max_i(w_i * |o_i - z_i|)
        where z is the reference point.
        """
        weights = self.weights.to_array()
        deviations = np.abs(objectives - self.reference_point)
        weighted_deviations = weights * deviations
        return float(-np.max(weighted_deviations))
    
    def _achievement_scalarize(
        self,
        objectives: np.ndarray,
        aspiration_point: Optional[np.ndarray] = None
    ) -> float:
        """
        Achievement scalarizing function.
        
        R = max_i(w_i * (z_i - o_i)) + rho * sum(w_i * (z_i - o_i))
        """
        if aspiration_point is None:
            aspiration_point = self.reference_point
        
        weights = self.weights.to_array()
        differences = aspiration_point - objectives
        
        max_term = np.max(weights * differences)
        sum_term = 0.1 * np.sum(weights * differences)  # rho = 0.1
        
        return float(max_term + sum_term)
    
    def _hypervolume_contribution(
        self,
        objectives: np.ndarray,
        pareto_front: Optional[np.ndarray] = None
    ) -> float:
        """
        Approximate hypervolume contribution scalarization.
        
        Uses a simplified approximation for efficiency.
        """
        if pareto_front is None:
            # Use reference point as naive pareto front
            pareto_front = self.reference_point.reshape(1, -1)
        
        # Calculate dominated hypervolume (simplified 2D projection)
        weights = self.weights.to_array()
        weighted_obj = objectives * weights
        
        # Hypervolume approximation
        hv = np.prod(np.maximum(weighted_obj, self.epsilon))
        return float(hv)
    
    def scalarize(
        self,
        objectives: Union[ObjectiveValues, np.ndarray],
        pareto_front: Optional[np.ndarray] = None
    ) -> float:
        """
        Convert multi-objective values to scalar reward.
        
        Parameters
        ----------
        objectives : ObjectiveValues or np.ndarray
            Current objective values
        pareto_front : np.ndarray, optional
            Current pareto front for hypervolume calculation
            
        Returns
        -------
        float
            Scalarized reward
        """
        if isinstance(objectives, ObjectiveValues):
            obj_array = objectives.to_array()
        else:
            obj_array = objectives
        
        if self.method == ScalarizationMethod.LINEAR:
            return self._linear_scalarize(obj_array)
        elif self.method == ScalarizationMethod.TCHEBYCHEFF:
            return self._tchebycheff_scalarize(obj_array)
        elif self.method == ScalarizationMethod.ACHIEVEMENT_SCALARIZING:
            return self._achievement_scalarize(obj_array)
        elif self.method == ScalarizationMethod.HYPERVOLUME:
            return self._hypervolume_contribution(obj_array, pareto_front)
        else:
            raise ValueError(f"Unknown scalarization method: {self.method}")


@ray.remote(memory=256*1024*1024)  # 256MB per task
def evaluate_policy_morl(
    policy_params: Dict,
    weights: Dict,
    n_episodes: int = 100,
    method: str = "linear"
) -> Dict:
    """
    Ray remote function for evaluating policy with multi-objective rewards.
    Memory-bounded for 4GB global quota compliance.
    """
    # Parse weights
    obj_weights = ObjectiveWeights(**weights)
    
    # Initialize scalarizer
    scalarizer = MultiObjectiveScalarizer(
        method=ScalarizationMethod(method),
        weights=obj_weights
    )
    
    # Simulate policy evaluation
    all_objectives = []
    
    for ep in range(n_episodes):
        # Simulated episode results (would come from env in practice)
        sharpe = np.random.randn() * 0.5 + 1.5  # Target ~1.5 Sharpe
        drawdown = -abs(np.random.exponential(0.05))  # Negative
        slippage = -abs(np.random.exponential(0.001))  # Negative
        tx_cost = -abs(np.random.exponential(0.0005))  # Negative
        inv_risk = -abs(np.random.exponential(0.01))  # Negative
        
        objectives = ObjectiveValues(
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            execution_slippage=slippage,
            transaction_cost=tx_cost,
            inventory_risk=inv_risk
        )
        
        all_objectives.append(objectives.to_array())
    
    all_objectives = np.array(all_objectives)
    
    # Compute mean objectives
    mean_objectives = np.mean(all_objectives, axis=0)
    
    # Scalarize
    scalarized_rewards = [
        scalarizer.scalarize(obj) for obj in all_objectives
    ]
    
    return {
        'mean_objectives': ObjectiveValues(
            sharpe_ratio=float(mean_objectives[0]),
            max_drawdown=float(mean_objectives[1]),
            execution_slippage=float(mean_objectives[2]),
            transaction_cost=float(mean_objectives[3]),
            inventory_risk=float(mean_objectives[4])
        ).to_dict(),
        'mean_scalarized_reward': float(np.mean(scalarized_rewards)),
        'std_scalarized_reward': float(np.std(scalarized_rewards)),
        'method': method,
    }


class MORLTrainer:
    """
    Multi-Objective RL Trainer that manages distributed training on Ray.
    Enforces strict 4GB RAM quota across all workers.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        scalarization_method: str = "linear",
        initial_weights: Optional[ObjectiveWeights] = None
    ):
        """
        Initialize MORL trainer.
        
        Parameters
        ----------
        max_workers : int
            Maximum number of parallel Ray workers
        scalarization_method : str
            Default scalarization method
        initial_weights : ObjectiveWeights, optional
            Initial objective weights
        """
        self.max_workers = max_workers
        self.scalarization_method = scalarization_method
        self.weights = initial_weights or ObjectiveWeights()
        
        # Check AMD acceleration
        self.accel_status = check_amd_acceleration()
        print(f"AMD Acceleration Status: {self.accel_status}")
        
        # Initialize Ray with memory limits
        if not ray.is_initialized():
            total_memory = max_workers * 256 * 1024 * 1024  # 256MB per worker
            ray.init(
                num_cpus=max_workers,
                _memory=total_memory,
                object_store_memory=total_memory // 2,
            )
        
        # Pareto front archive
        self.pareto_front: List[np.ndarray] = []
    
    def update_pareto_front(self, new_objectives: np.ndarray):
        """Update the pareto front archive with new solutions."""
        # Check if new solution dominates any existing
        dominated_indices = []
        
        for i, existing in enumerate(self.pareto_front):
            # Check domination (higher is better for all objectives after sign flip)
            if np.all(new_objectives >= existing) and np.any(new_objectives > existing):
                dominated_indices.append(i)
        
        # Remove dominated solutions
        for idx in sorted(dominated_indices, reverse=True):
            self.pareto_front.pop(idx)
        
        # Check if new solution is dominated by any existing
        is_dominated = False
        for existing in self.pareto_front:
            if np.all(existing >= new_objectives) and np.any(existing > new_objectives):
                is_dominated = True
                break
        
        # Add non-dominated solution
        if not is_dominated:
            self.pareto_front.append(new_objectives)
    
    def train_epoch(
        self,
        policy_params: Dict,
        n_evaluations: int = 10
    ) -> Dict:
        """
        Run one epoch of MORL training.
        
        Parameters
        ----------
        policy_params : Dict
            Current policy parameters
        n_evaluations : int
            Number of parallel evaluations
            
        Returns
        -------
        Dict
            Training metrics
        """
        # Launch parallel evaluations
        futures = []
        for i in range(min(n_evaluations, self.max_workers)):
            future = evaluate_policy_morl.remote(
                policy_params,
                self.weights.__dict__,
                n_episodes=50,
                method=self.scalarization_method
            )
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Aggregate metrics
        all_sharpes = [r['mean_objectives']['sharpe_ratio'] for r in results]
        all_drawdowns = [r['mean_objectives']['max_drawdown'] for r in results]
        all_rewards = [r['mean_scalarized_reward'] for r in results]
        
        # Update pareto front
        for r in results:
            obj_array = np.array([
                r['mean_objectives']['sharpe_ratio'],
                r['mean_objectives']['max_drawdown'],
                r['mean_objectives']['execution_slippage'],
                r['mean_objectives']['transaction_cost'],
                r['mean_objectives']['inventory_risk']
            ])
            self.update_pareto_front(obj_array)
        
        return {
            'mean_sharpe': float(np.mean(all_sharpes)),
            'std_sharpe': float(np.std(all_sharpes)),
            'mean_drawdown': float(np.mean(all_drawdowns)),
            'mean_scalarized_reward': float(np.mean(all_rewards)),
            'pareto_front_size': len(self.pareto_front),
        }
    
    def adapt_weights(
        self,
        current_metrics: Dict,
        target_sharpe: float = 2.0,
        max_acceptable_drawdown: float = -0.1
    ):
        """
        Adaptively adjust weights based on current performance.
        
        Parameters
        ----------
        current_metrics : Dict
            Current performance metrics
        target_sharpe : float
            Target Sharpe ratio
        max_acceptable_drawdown : float
            Maximum acceptable drawdown
        """
        current_sharpe = current_metrics.get('mean_sharpe', 0)
        current_drawdown = current_metrics.get('mean_drawdown', 0)
        
        # Increase drawdown weight if exceeding threshold
        if current_drawdown < max_acceptable_drawdown:
            self.weights.max_drawdown *= 1.1
        
        # Increase Sharpe weight if below target
        if current_sharpe < target_sharpe:
            self.weights.sharpe_ratio *= 1.05
        
        # Normalize after adjustments
        self.weights = self.weights.normalize()
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()


if __name__ == '__main__':
    print("Initializing Multi-Objective RL Module...")
    
    # Check AMD acceleration
    accel = check_amd_acceleration()
    print(f"AMD Acceleration: {accel}")
    
    # Test different scalarization methods
    methods = [
        ScalarizationMethod.LINEAR,
        ScalarizationMethod.TCHEBYCHEFF,
        ScalarizationMethod.ACHIEVEMENT_SCALARIZING,
    ]
    
    test_objectives = ObjectiveValues(
        sharpe_ratio=1.5,
        max_drawdown=-0.05,
        execution_slippage=-0.001,
        transaction_cost=-0.0005,
        inventory_risk=-0.01
    )
    
    print("\nScalarization Method Comparison:")
    for method in methods:
        scalarizer = MultiObjectiveScalarizer(method=method)
        reward = scalarizer.scalarize(test_objectives)
        print(f"  {method.value}: {reward:.6f}")
    
    # Run distributed evaluation
    print("\nRunning distributed MORL evaluation...")
    
    trainer = MORLTrainer(
        max_workers=4,
        scalarization_method="linear",
        initial_weights=ObjectiveWeights(
            sharpe_ratio=1.0,
            max_drawdown=1.5,  # Higher weight on drawdown control
            execution_slippage=0.5,
        )
    )
    
    # Simulated policy params
    policy_params = {'learning_rate': 0.001, 'gamma': 0.99}
    
    # Run training epoch
    metrics = trainer.train_epoch(policy_params, n_evaluations=4)
    
    print(f"\nTraining Epoch Results:")
    print(f"  Mean Sharpe: {metrics['mean_sharpe']:.4f}")
    print(f"  Mean Drawdown: {metrics['mean_drawdown']:.4f}")
    print(f"  Pareto Front Size: {metrics['pareto_front_size']}")
    
    trainer.shutdown()
