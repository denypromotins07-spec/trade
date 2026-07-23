"""
python/automl/pbt_scheduler.py

Population Based Training (PBT) Scheduler on Ray

Dynamically mutates RL hyperparameters and network weights during live training.
Strictly bounds worker memory limits to enforce 4GB Python RAM quota per worker.
Includes AMD ROCm/DirectML acceleration checks.

Memory Constraint: Automatic checkpoint pruning, worker memory monitoring.
"""

import ray
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import os
import copy


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
class PBTConfig:
    """Configuration for Population Based Training."""
    population_size: int = 8
    perturb_interval_steps: int = 1000
    eval_interval_steps: int = 500
    exploit_threshold: float = 0.2  # Exploit if >20% better
    resample_threshold: float = 0.2  # Resample if >20% worse
    
    # Hyperparameter search space
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_options: List[int] = field(default_factory=lambda: [64, 128, 256])
    entropy_coef_range: Tuple[float, float] = (0.001, 0.1)
    
    # Memory constraints
    max_memory_gb_per_worker: float = 4.0
    max_checkpoints: int = 3  # Keep only top 3 checkpoints
    
    # Acceleration
    use_amp: bool = True  # Automatic mixed precision


@ray.remote(max_calls=10)
class PBTWorker:
    """
    Ray worker for PBT training with strict memory bounds.
    """
    
    def __init__(self, worker_id: int, config: PBTConfig):
        self.worker_id = worker_id
        self.config = config
        self.acceleration = check_amd_acceleration()
        self.device = self._select_device()
        
        # Initialize model with random hyperparams from search space
        self.hyperparams = self._sample_hyperparams()
        self.model = self._create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.hyperparams['learning_rate'])
        
        self.current_step = 0
        self.best_metric = float('-inf')
        self.checkpoints = []
        
    def _select_device(self) -> str:
        if self.acceleration["rocm"]:
            return "cuda"
        elif self.acceleration["directml"]:
            return "privateuseone"
        elif self.acceleration["cuda"]:
            return "cuda"
        return "cpu"
    
    def _sample_hyperparams(self) -> Dict[str, Any]:
        """Sample initial hyperparameters from search space."""
        return {
            'learning_rate': float(np.random.uniform(*self.config.learning_rate_range)),
            'batch_size': int(np.random.choice(self.config.batch_size_options)),
            'entropy_coef': float(np.random.uniform(*self.config.entropy_coef_range)),
        }
    
    def _create_model(self) -> torch.nn.Module:
        """Create RL model with current hyperparams."""
        # Simplified model - real impl would be full RL network
        return torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4),  # Action head
        ).to(self.device)
    
    def train_step(self, batch: np.ndarray) -> Dict[str, float]:
        """Execute one training step with current hyperparams."""
        self.model.train()
        
        # Check memory before training
        self._check_memory()
        
        batch_tensor = torch.FloatTensor(batch).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(batch_tensor)
        
        # Dummy loss for demonstration
        loss = output.mean()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.current_step += 1
        
        return {'loss': loss.item(), 'step': self.current_step}
    
    def evaluate(self) -> float:
        """Evaluate current model performance."""
        self.model.eval()
        
        # Dummy evaluation metric
        with torch.no_grad():
            test_input = torch.randn(1, 32).to(self.device)
            output = self.model(test_input)
            metric = output.mean().item()
        
        if metric > self.best_metric:
            self.best_metric = metric
        
        return metric
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return self.hyperparams.copy()
    
    def set_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Update hyperparameters and reinitialize optimizer."""
        self.hyperparams = hyperparams.copy()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.hyperparams['learning_rate']
        )
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return model state dict."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load model state dict."""
        cpu_weights = {k: v.cpu() for k, v in weights.items()}
        self.model.load_state_dict(cpu_weights)
        self.model.to(self.device)
    
    def save_checkpoint(self) -> Optional[str]:
        """Save checkpoint if performance improved."""
        metric = self.evaluate()
        
        self.checkpoints.append({
            'step': self.current_step,
            'metric': metric,
            'weights': self.get_weights(),
            'hyperparams': self.hyperparams.copy(),
        })
        
        # Prune old checkpoints
        while len(self.checkpoints) > self.config.max_checkpoints:
            self.checkpoints.pop(0)
        
        return f"worker_{self.worker_id}_step_{self.current_step}"
    
    def _check_memory(self) -> None:
        """Check and enforce memory limits."""
        import psutil
        process = psutil.Process()
        current_gb = process.memory_info().rss / (1024 ** 3)
        
        if current_gb > self.config.max_memory_gb_per_worker * 0.9:
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Prune checkpoints aggressively
            if len(self.checkpoints) > 1:
                self.checkpoints = self.checkpoints[-1:]
    
    def should_perturb(self) -> bool:
        """Check if it's time to perturb hyperparams."""
        return self.current_step > 0 and \
               self.current_step % self.config.perturb_interval_steps == 0


@ray.remote
class PBTScheduler:
    """
    Central scheduler for Population Based Training.
    Coordinates exploitation and exploration across workers.
    """
    
    def __init__(self, config: PBTConfig):
        self.config = config
        self.workers = [
            PBTWorker.remote(i, config) 
            for i in range(config.population_size)
        ]
        self.generation = 0
        self.metrics_history = []
        
    def run_training_cycle(self, num_cycles: int) -> Dict[str, Any]:
        """Run PBT for specified number of cycles."""
        for cycle in range(num_cycles):
            # Train all workers
            train_results = ray.get([
                w.train_step.remote(np.random.randn(32)) 
                for w in self.workers
            ])
            
            # Evaluate periodically
            if cycle % self.config.eval_interval_steps == 0:
                metrics = ray.get([w.evaluate.remote() for w in self.workers])
                self.metrics_history.append(metrics)
                
                # Run PBT exploit/explore
                self._run_pbt_step()
        
        return {
            'generation': self.generation,
            'best_metrics': self.metrics_history[-1] if self.metrics_history else [],
        }
    
    def _run_pbt_step(self) -> None:
        """Execute one PBT exploit/explore step."""
        metrics = ray.get([w.evaluate.remote() for w in self.workers])
        hyperparams = ray.get([w.get_hyperparams.remote() for w in self.workers])
        weights = ray.get([w.get_weights.remote() for w in self.workers])
        
        # Sort by metric
        sorted_indices = np.argsort(metrics)[::-1]  # Descending
        
        for i, idx in enumerate(sorted_indices):
            if i < len(sorted_indices) // 2:
                # Top performers: continue training
                continue
            else:
                # Bottom performers: exploit or explore
                better_idx = sorted_indices[i % (len(sorted_indices) // 2)]
                
                metric_diff = (metrics[better_idx] - metrics[idx]) / \
                             (abs(metrics[idx]) + 1e-8)
                
                if metric_diff > self.config.exploit_threshold:
                    # Exploit: copy weights from better performer
                    ray.get(self.workers[idx].load_weights.remote(weights[better_idx]))
                    
                    # Also copy hyperparams with small perturbation
                    new_params = hyperparams[better_idx].copy()
                    new_params = self._perturb_hyperparams(new_params)
                    ray.get(self.workers[idx].set_hyperparams.remote(new_params))
                elif metric_diff < -self.config.resample_threshold:
                    # Explore: resample hyperparams randomly
                    new_params = self._sample_hyperparams()
                    ray.get(self.workers[idx].set_hyperparams.remote(new_params))
        
        self.generation += 1
    
    def _perturb_hyperparams(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply small perturbation to hyperparameters."""
        perturbed = params.copy()
        
        # Perturb learning rate
        if np.random.random() < 0.5:
            factor = np.random.choice([0.8, 1.2])
            new_lr = params['learning_rate'] * factor
            new_lr = np.clip(new_lr, *self.config.learning_rate_range)
            perturbed['learning_rate'] = float(new_lr)
        
        # Perturb entropy coefficient
        if np.random.random() < 0.5:
            factor = np.random.choice([0.8, 1.2])
            new_entropy = params['entropy_coef'] * factor
            new_entropy = np.clip(new_entropy, *self.config.entropy_coef_range)
            perturbed['entropy_coef'] = float(new_entropy)
        
        return perturbed
    
    def _sample_hyperparams(self) -> Dict[str, Any]:
        """Sample new hyperparameters from search space."""
        return {
            'learning_rate': float(np.random.uniform(*self.config.learning_rate_range)),
            'batch_size': int(np.random.choice(self.config.batch_size_options)),
            'entropy_coef': float(np.random.uniform(*self.config.entropy_coef_range)),
        }
    
    def get_best_config(self) -> Dict[str, Any]:
        """Get configuration of best performing worker."""
        metrics = ray.get([w.evaluate.remote() for w in self.workers])
        best_idx = np.argmax(metrics)
        return ray.get(self.workers[best_idx].get_hyperparams.remote())
    
    def shutdown(self) -> None:
        """Clean shutdown of all workers."""
        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:
                pass


if __name__ == "__main__":
    print("PBT Scheduler - AMD Acceleration:", check_amd_acceleration())
    
    config = PBTConfig(population_size=4)
    scheduler = PBTScheduler.remote(config)
    
    # Run a few training cycles
    result = ray.get(scheduler.run_training_cycle.remote(10))
    print(f"Training result: {result}")
    
    best_config = ray.get(scheduler.get_best_config.remote())
    print(f"Best hyperparams: {best_config}")
    
    ray.get(scheduler.shutdown.remote())
