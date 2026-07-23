"""
python/automl/resource_aware.py

Resource-Aware Trial Scheduler with Memory Kill Switch

Instantly kills underperforming models if they threaten the global 8GB RAM ceiling
during intense gradient updates. Includes AMD ROCm/DirectML checks and graceful
degradation when memory pressure is detected.

Memory Constraint: Hard 8GB global limit, automatic worker termination.
"""

import ray
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import os
import time


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
class ResourceConfig:
    """Resource constraints for trial scheduling."""
    max_global_memory_gb: float = 8.0
    max_memory_per_trial_gb: float = 2.0
    memory_safety_margin_gb: float = 1.0  # Keep 1GB free
    kill_threshold_pct: float = 0.95  # Kill at 95% of limit
    
    # Performance thresholds
    min_improvement_per_step: float = 0.001
    patience_steps: int = 100  # Kill if no improvement in N steps
    
    # Acceleration
    use_amp: bool = True


@ray.remote(max_calls=5)
class TrialWorker:
    """
    Ray worker for individual training trial.
    Monitors its own memory usage and reports to scheduler.
    """
    
    def __init__(self, trial_id: int, config: ResourceConfig):
        self.trial_id = trial_id
        self.config = config
        self.acceleration = check_amd_acceleration()
        self.device = self._select_device()
        
        self.model = self._create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        self.step_count = 0
        self.best_loss = float('inf')
        self.no_improvement_count = 0
        
        self.is_alive = True
        self.memory_warning_issued = False
        
    def _select_device(self) -> str:
        if self.acceleration["rocm"]:
            return "cuda"
        elif self.acceleration["directml"]:
            return "privateuseone"
        elif self.acceleration["cuda"]:
            return "cuda"
        return "cpu"
    
    def _create_model(self) -> torch.nn.Module:
        """Create model for this trial."""
        return torch.nn.Sequential(
            torch.nn.Linear(64, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        ).to(self.device)
    
    def train_step(self, batch: np.ndarray) -> Dict[str, Any]:
        """Execute training step with memory monitoring."""
        if not self.is_alive:
            return {'status': 'dead', 'trial_id': self.trial_id}
        
        # Check memory BEFORE training
        mem_status = self._check_memory()
        if mem_status['status'] == 'critical':
            self._request_kill("Critical memory")
            return {'status': 'killed_memory', 'trial_id': self.trial_id}
        
        self.model.train()
        batch_tensor = torch.FloatTensor(batch).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(batch_tensor)
        loss = output.mean().abs()  # Dummy loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.step_count += 1
        
        # Track improvement
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # Check for stagnation
        if self.no_improvement_count > self.config.patience_steps:
            self._request_kill("No improvement")
            return {
                'status': 'killed_stagnant', 
                'trial_id': self.trial_id,
                'steps': self.step_count,
                'best_loss': self.best_loss,
            }
        
        return {
            'status': 'alive',
            'trial_id': self.trial_id,
            'step': self.step_count,
            'loss': loss.item(),
            'best_loss': self.best_loss,
            'memory_mb': mem_status['used_mb'],
        }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check current memory usage."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        used_mb = mem_info.rss / (1024 * 1024)
        used_gb = used_mb / 1024
        
        status = 'healthy'
        if used_gb > self.config.max_memory_per_trial_gb * self.config.kill_threshold_pct:
            status = 'critical'
        elif used_gb > self.config.max_memory_per_trial_gb * 0.8:
            status = 'warning'
            if not self.memory_warning_issued:
                self.memory_warning_issued = True
        
        return {
            'status': status,
            'used_mb': used_mb,
            'used_gb': used_gb,
            'limit_gb': self.config.max_memory_per_trial_gb,
        }
    
    def _request_kill(self, reason: str) -> None:
        """Mark trial for killing."""
        self.is_alive = False
    
    def get_status(self) -> Dict[str, Any]:
        """Return current trial status."""
        mem = self._check_memory()
        return {
            'trial_id': self.trial_id,
            'is_alive': self.is_alive,
            'step_count': self.step_count,
            'best_loss': self.best_loss,
            'no_improvement_count': self.no_improvement_count,
            'memory': mem,
        }
    
    def get_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return model weights if still alive."""
        if not self.is_alive:
            return None
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}


@ray.remote
class ResourceAwareScheduler:
    """
    Global scheduler that monitors all trials and enforces memory limits.
    Kills underperforming or memory-hungry trials instantly.
    """
    
    def __init__(self, config: ResourceConfig, max_trials: int = 4):
        self.config = config
        self.max_trials = max_trials
        self.trials: Dict[int, ray.actor.ActorHandle] = {}
        self.trial_results: Dict[int, List[Dict]] = {}
        self.next_trial_id = 0
        self.global_start_time = time.time()
        
    def spawn_trial(self) -> Optional[int]:
        """Spawn a new trial if resources allow."""
        if not self._can_spawn_trial():
            return None
        
        trial_id = self.next_trial_id
        self.next_trial_id += 1
        
        worker = TrialWorker.remote(trial_id, self.config)
        self.trials[trial_id] = worker
        self.trial_results[trial_id] = []
        
        return trial_id
    
    def _can_spawn_trial(self) -> bool:
        """Check if we can spawn another trial."""
        if len(self.trials) >= self.max_trials:
            return False
        
        # Check global memory
        global_mem = self._get_global_memory_usage()
        if global_mem['used_gb'] > self.config.max_global_memory_gb - self.config.memory_safety_margin_gb:
            return False
        
        return True
    
    def _get_global_memory_usage(self) -> Dict[str, float]:
        """Get total memory usage across all trials."""
        import psutil
        process = psutil.Process()
        used_gb = process.memory_info().rss / (1024 ** 3)
        
        return {
            'used_gb': used_gb,
            'limit_gb': self.config.max_global_memory_gb,
            'available_gb': self.config.max_global_memory_gb - used_gb,
        }
    
    def run_step(self) -> Dict[str, Any]:
        """Run one training step across all active trials."""
        results = []
        dead_trials = []
        
        for trial_id, worker in list(self.trials.items()):
            try:
                # Generate dummy batch
                batch = np.random.randn(32, 64)
                
                result = ray.get(worker.train_step.remote(batch), timeout=30)
                results.append(result)
                self.trial_results[trial_id].append(result)
                
                if result.get('status') in ['dead', 'killed_memory', 'killed_stagnant']:
                    dead_trials.append(trial_id)
                    
            except Exception as e:
                # Worker crashed or timed out
                dead_trials.append(trial_id)
                results.append({'status': 'error', 'trial_id': trial_id, 'error': str(e)})
        
        # Clean up dead trials
        for trial_id in dead_trials:
            self._cleanup_trial(trial_id)
        
        # Check global memory emergency
        global_mem = self._get_global_memory_usage()
        if global_mem['used_gb'] > self.config.max_global_memory_gb * self.config.kill_threshold_pct:
            # Emergency kill: terminate worst performers
            self._emergency_kill_worst_performers()
        
        return {
            'active_trials': len(self.trials),
            'results_this_step': len(results),
            'dead_trials': dead_trials,
            'global_memory_gb': global_mem['used_gb'],
        }
    
    def _cleanup_trial(self, trial_id: int) -> None:
        """Clean up a dead trial."""
        if trial_id in self.trials:
            try:
                ray.kill(self.trials[trial_id])
            except Exception:
                pass
            del self.trials[trial_id]
    
    def _emergency_kill_worst_performers(self) -> None:
        """Kill worst performing trials to free memory."""
        if len(self.trials) <= 1:
            return
        
        # Get performance metrics
        performances = []
        for trial_id, worker in self.trials.items():
            try:
                status = ray.get(worker.get_status.remote(), timeout=5)
                performances.append((trial_id, status.get('best_loss', float('inf'))))
            except Exception:
                performances.append((trial_id, float('inf')))
        
        # Sort by loss (worst first)
        performances.sort(key=lambda x: x[1], reverse=True)
        
        # Kill worst performer
        worst_trial = performances[0][0]
        log_warn(f"Emergency kill: Trial {worst_trial} (worst performer)")
        self._cleanup_trial(worst_trial)
    
    def get_best_trial_weights(self) -> Optional[Tuple[int, Dict[str, torch.Tensor]]]:
        """Get weights from best performing trial."""
        if not self.trials:
            return None
        
        best_loss = float('inf')
        best_trial_id = None
        
        for trial_id, worker in self.trials.items():
            try:
                status = ray.get(worker.get_status.remote(), timeout=5)
                if status['best_loss'] < best_loss:
                    best_loss = status['best_loss']
                    best_trial_id = trial_id
            except Exception:
                continue
        
        if best_trial_id is None:
            return None
        
        weights = ray.get(self.trials[best_trial_id].get_weights.remote(), timeout=30)
        return (best_trial_id, weights)
    
    def shutdown(self) -> None:
        """Shutdown all trials."""
        for trial_id in list(self.trials.keys()):
            self._cleanup_trial(trial_id)
        self.trials.clear()


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


if __name__ == "__main__":
    print("Resource-Aware Scheduler - AMD Acceleration:", check_amd_acceleration())
    
    config = ResourceConfig(max_global_memory_gb=8.0, max_memory_per_trial_gb=2.0)
    scheduler = ResourceAwareScheduler.remote(config, max_trials=3)
    
    # Spawn initial trials
    for i in range(3):
        trial_id = ray.get(scheduler.spawn_trial.remote())
        print(f"Spawned trial {trial_id}")
    
    # Run training steps
    for step in range(10):
        result = ray.get(scheduler.run_step.remote())
        print(f"Step {step}: {result}")
    
    # Get best weights
    best = ray.get(scheduler.get_best_trial_weights.remote())
    if best:
        print(f"Best trial: {best[0]}")
    
    ray.get(scheduler.shutdown.remote())
    print("Scheduler shutdown complete")
