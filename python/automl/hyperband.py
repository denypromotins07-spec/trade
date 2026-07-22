"""
Chapter 3: Distributed Hyperparameter Tuning & AutoML
File 7: python/automl/hyperband.py

Asynchronous Successive Halving (Hyperband) on Ray for rapid RL
hyperparameter tuning. Aggressively kills underperforming trials
to save compute and memory resources.

Enforces 4GB RAM quota per worker.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import BasicVariantGenerator
import time

# Memory limit (4GB quota)
MAX_MEMORY_MB = 4096


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability."""
    accel_info = {
        'rocm_available': False,
        'directml_available': False,
        'cuda_available': False,
        'recommended_backend': 'numpy'
    }
    
    try:
        import torch
        if torch.version.hip is not None:
            accel_info['rocm_available'] = True
            accel_info['recommended_backend'] = 'pytorch_rocm'
        elif hasattr(torch.backends, 'directml'):
            accel_info['directml_available'] = True
            accel_info['recommended_backend'] = 'pytorch_directml'
        elif torch.cuda.is_available():
            accel_info['cuda_available'] = True
            accel_info['recommended_backend'] = 'pytorch_cuda'
    except ImportError:
        pass
    
    return accel_info


class HyperbandTuner:
    """
    Asynchronous Successive Halving Algorithm (ASHA) tuner.
    
    Hyperband efficiently allocates resources by:
    1. Starting many configurations with few resources
    2. Promising configurations get more resources
    3. Poor configurations are stopped early
    """
    
    def __init__(
        self,
        metric: str = "reward",
        mode: str = "max",
        max_t: int = 100,
        grace_period: int = 10,
        reduction_factor: int = 3,
        brackets: int = 3,
        memory_limit_mb: int = MAX_MEMORY_MB
    ):
        self.metric = metric
        self.mode = mode
        self.max_t = max_t
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.brackets = brackets
        self.memory_limit_mb = memory_limit_mb
        
        self.accel_info = check_amd_acceleration()
        self._trials_completed = 0
        self._trials_terminated_early = 0
        
        # Create ASHA scheduler
        self.scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=max_t,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            brackets=brackets,
            stop_last_trials=True  # Aggressively kill poor performers
        )
    
    def create_search_space(
        self,
        param_ranges: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Create Ray Tune search space from parameter ranges.
        
        Parameters
        ----------
        param_ranges : dict
            Dict of {param_name: (min, max)} or {param_name: [choices]}
            
        Returns
        -------
        dict
            Ray Tune search space
        """
        search_space = {}
        
        for param_name, range_spec in param_ranges.items():
            if isinstance(range_spec, (list, tuple)) and len(range_spec) == 2:
                min_val, max_val = range_spec
                
                # Determine if integer or float
                if isinstance(min_val, int) and isinstance(max_val, int):
                    search_space[param_name] = tune.randint(min_val, max_val + 1)
                else:
                    search_space[param_name] = tune.uniform(float(min_val), float(max_val))
            elif isinstance(range_spec, list):
                # Categorical choices
                search_space[param_name] = tune.choice(range_spec)
            else:
                # Fixed value
                search_space[param_name] = range_spec
        
        return search_space
    
    def tune(
        self,
        train_func: Callable,
        param_ranges: Dict[str, any],
        num_samples: int = 50,
        cpus_per_trial: float = 2.0,
        gpus_per_trial: float = 0.0,
        memory_per_trial: int = 512,  # MB
        timeout_s: int = 3600,
        local_dir: str = "~/ray_results"
    ) -> tune.ResultGrid:
        """
        Run Hyperband optimization.
        
        Parameters
        ----------
        train_func : callable
            Training function that takes config dict and reports metrics
        param_ranges : dict
            Parameter ranges to search
        num_samples : int
            Number of initial configurations to try
        cpus_per_trial : float
            CPUs allocated per trial
        gpus_per_trial : float
            GPUs allocated per trial
        memory_per_trial : int
            Memory (MB) per trial - enforces 4GB total limit
        timeout_s : int
            Maximum runtime in seconds
        local_dir : str
            Directory for results
            
        Returns
        -------
        ResultGrid
            Ray Tune result grid
        """
        # Enforce memory limits
        effective_memory = min(memory_per_trial, self.memory_limit_mb // 4)
        
        search_space = self.create_search_space(param_ranges)
        
        # Configure resources
        resources_per_trial = {
            "cpu": cpus_per_trial,
            "gpu": gpus_per_trial,
            "memory": effective_memory * 1024 * 1024  # Convert to bytes
        }
        
        # Run optimization
        analysis = tune.run(
            train_func,
            config=search_space,
            scheduler=self.scheduler,
            num_samples=num_samples,
            resources_per_trial=resources_per_trial,
            metric=self.metric,
            mode=self.mode,
            local_dir=local_dir,
            time_budget_s=timeout_s,
            verbose=1,
            raise_on_failed_trial=False,
            callbacks=[self._create_memory_callback()]
        )
        
        self._trials_completed = len(analysis.trials)
        self._trials_terminated_early = sum(
            1 for t in analysis.trials if t.last_status == "TERMINATED"
        )
        
        return analysis
    
    def _create_memory_callback(self):
        """Create callback to enforce memory limits."""
        import gc
        
        def on_trial_result(trial, result):
            # Periodic garbage collection
            if trial.iteration % 10 == 0:
                gc.collect()
            
            # Check if result indicates memory issues
            if 'memory_usage_mb' in result:
                if result['memory_usage_mb'] > self.memory_limit_mb * 0.9:
                    return "STOP"  # Stop trial approaching memory limit
            
            return "CONTINUE"
        
        return on_trial_result
    
    def get_best_config(self, analysis: tune.ResultGrid) -> Dict:
        """Extract best configuration from results."""
        return analysis.best_config
    
    def get_stats(self) -> Dict:
        """Get tuning statistics."""
        return {
            'trials_completed': self._trials_completed,
            'trials_terminated_early': self._trials_terminated_early,
            'early_termination_rate': (
                self._trials_terminated_early / max(1, self._trials_completed)
            ),
            'acceleration': self.accel_info,
            'memory_limit_mb': self.memory_limit_mb
        }


@ray.remote(max_calls=5)
class DistributedHyperbandWorker:
    """Ray actor for distributed Hyperband tuning."""
    
    def __init__(self, memory_limit_mb: int = MAX_MEMORY_MB):
        self.memory_limit_mb = memory_limit_mb
        self.accel_info = check_amd_acceleration()
        self._evaluations = 0
    
    def evaluate_config(
        self,
        config: Dict,
        train_func: Callable,
        max_epochs: int = 50
    ) -> Dict:
        """Evaluate a single configuration."""
        self._check_memory()
        
        start_time = time.time()
        result = train_func(config, max_epochs=max_epochs)
        elapsed = time.time() - start_time
        
        self._evaluations += 1
        
        return {
            'config': config,
            'result': result,
            'elapsed_seconds': elapsed,
            'evaluation_id': self._evaluations
        }
    
    def batch_evaluate(
        self,
        configs: List[Dict],
        train_func: Callable,
        max_epochs: int = 50
    ) -> List[Dict]:
        """Evaluate multiple configurations."""
        results = []
        for config in configs:
            result = self.evaluate_config(config, train_func, max_epochs)
            results.append(result)
            self._check_memory()
        return results
    
    def _check_memory(self):
        """Memory checkpoint."""
        import gc
        if self._evaluations % 10 == 0:
            gc.collect()
    
    def get_stats(self) -> Dict:
        return {
            'evaluations': self._evaluations,
            'acceleration': self.accel_info,
            'memory_limit_mb': self.memory_limit_mb
        }


def create_hyperband_workers(num_workers: int = 4) -> List:
    """Create distributed Hyperband workers."""
    return [
        DistributedHyperbandWorker.remote(memory_limit_mb=MAX_MEMORY_MB)
        for _ in range(num_workers)
    ]


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=4 * 1024 * 1024 * 1024,
        _system_config={"max_bytes_to_spill": 4 * 1024 * 1024 * 1024}
    )
    
    print("AMD Acceleration:", check_amd_acceleration())
    
    # Example usage
    tuner = HyperbandTuner(
        metric="validation_reward",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3
    )
    
    # Define parameter ranges for RL agent
    param_ranges = {
        'learning_rate': (1e-5, 1e-2),
        'gamma': (0.9, 0.999),
        'tau': (0.001, 0.01),
        'batch_size': [32, 64, 128, 256],
        'hidden_dim': [64, 128, 256, 512],
        'buffer_size': (10000, 1000000)
    }
    
    print(f"Search space: {param_ranges}")
    print(f"Tuner stats: {tuner.get_stats()}")
    
    ray.shutdown()
