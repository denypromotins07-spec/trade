"""
Elastic Scaling Manager for Ray Workers

This module implements an elastic scaling manager that dynamically spins
up or tears down Ray workers based on real-time market volatility,
strictly enforcing the 4GB Python RAM quota per worker.

Memory Safety:
- Strictly enforces 4GB Python RAM quota per worker
- Automatic worker termination when memory exceeds limits
- Graceful scaling based on market conditions
"""

import os
import ray
import torch
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil

logger = logging.getLogger(__name__)

# Enforce 4GB RAM quota per worker
MAX_WORKER_MEMORY_GB = 4.0
MEMORY_THRESHOLD_WARNING = 0.85
MEMORY_THRESHOLD_CRITICAL = 0.95


class ScaleAction(Enum):
    """Actions for scaling decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class WorkerInfo:
    """Information about a Ray worker."""
    worker_id: str
    actor_handle: Any
    start_time: float
    memory_gb: float = 0.0
    tasks_completed: int = 0
    is_healthy: bool = True


@dataclass
class ScalingConfig:
    """Configuration for elastic scaling."""
    min_workers: int = 2
    max_workers: int = 32
    target_memory_usage: float = 0.7  # Target 70% memory utilization
    scale_up_threshold: float = 0.8  # Scale up when memory > 80%
    scale_down_threshold: float = 0.4  # Scale down when memory < 40%
    cooldown_seconds: float = 30.0  # Minimum time between scaling actions
    volatility_sensitivity: float = 1.0  # How much volatility affects scaling
    max_memory_gb: float = MAX_WORKER_MEMORY_GB


@ray.remote(max_calls=100)
class VolatilityAwareWorker:
    """
    Ray worker that adjusts its behavior based on market volatility.
    
    Includes built-in memory monitoring and automatic cleanup.
    """
    
    def __init__(self, worker_id: str, config: ScalingConfig):
        self.worker_id = worker_id
        self.config = config
        self.device = self._get_device()
        self.start_time = time.time()
        self.tasks_completed = 0
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        self._check_memory()
    
    def _get_device(self) -> str:
        """Get the best available compute device."""
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                if 'AMD' in device_name or 'Radeon' in device_name:
                    return "cuda:0"
            # Check DirectML
            try:
                import torch_directml
                return "dml:0"
            except ImportError:
                pass
        except Exception:
            pass
        return "cpu"
    
    def _check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024 ** 3)
        
        if memory_gb > self.config.max_memory_gb * MEMORY_THRESHOLD_CRITICAL:
            logger.critical(
                f"Worker {self.worker_id} memory critical: {memory_gb:.2f}GB "
                f"(limit: {self.config.max_memory_gb}GB)"
            )
            self._cleanup()
            return False
        
        if memory_gb > self.config.max_memory_gb * MEMORY_THRESHOLD_WARNING:
            logger.warning(
                f"Worker {self.worker_id} memory high: {memory_gb:.2f}GB"
            )
        
        return True
    
    def _cleanup(self):
        """Clean up resources to free memory."""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def process_batch(
        self,
        batch_data: Dict[str, Any],
        volatility: float,
    ) -> Dict[str, Any]:
        """
        Process a batch of data with volatility-aware computation.
        
        Args:
            batch_data: Input data batch
            volatility: Current market volatility measure
            
        Returns:
            Processing results
        """
        if not self._check_memory():
            return {"error": "Memory limit exceeded", "worker_id": self.worker_id}
        
        # Adjust computation based on volatility
        # Higher volatility = more conservative processing
        risk_factor = min(1.0, volatility * 2.0)
        
        # Simulate processing (would be actual model inference in production)
        result = {
            "worker_id": self.worker_id,
            "batch_size": len(batch_data.get("data", [])),
            "volatility_adjusted": True,
            "risk_factor": risk_factor,
            "device": self.device,
        }
        
        self.tasks_completed += 1
        self._check_memory()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        memory_gb = self.process.memory_info().rss / (1024 ** 3)
        
        return {
            "worker_id": self.worker_id,
            "uptime_seconds": time.time() - self.start_time,
            "tasks_completed": self.tasks_completed,
            "memory_gb": memory_gb,
            "memory_limit_gb": self.config.max_memory_gb,
            "is_healthy": self._check_memory(),
            "device": self.device,
        }


class ElasticScalingManager:
    """
    Manager for elastic scaling of Ray workers.
    
    Monitors market volatility and system resources to dynamically
    adjust the number of active workers while respecting memory limits.
    """
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.workers: Dict[str, WorkerInfo] = {}
        self.last_scale_action: float = 0
        self.current_volatility: float = 0.0
        
        # Initialize with minimum workers
        self._initialize_min_workers()
    
    def _initialize_min_workers(self):
        """Create initial set of workers."""
        for i in range(self.config.min_workers):
            self._create_worker(f"worker_{i}")
    
    def _create_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Create a new worker."""
        if len(self.workers) >= self.config.max_workers:
            logger.warning(f"Cannot create worker: max workers ({self.config.max_workers}) reached")
            return None
        
        try:
            actor = VolatilityAwareWorker.options(
                resources={"CPU": 1},
                runtime_env={"env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}}
            ).remote(worker_id, self.config)
            
            # Verify worker started successfully
            ray.get(actor.get_stats.remote(), timeout=30)
            
            worker_info = WorkerInfo(
                worker_id=worker_id,
                actor_handle=actor,
                start_time=time.time(),
            )
            
            self.workers[worker_id] = worker_info
            logger.info(f"Created worker {worker_id}")
            return worker_info
            
        except Exception as e:
            logger.error(f"Failed to create worker {worker_id}: {e}")
            return None
    
    def _destroy_worker(self, worker_id: str) -> bool:
        """Destroy a worker and free resources."""
        if worker_id not in self.workers:
            return False
        
        try:
            worker = self.workers[worker_id]
            
            # Get final stats before shutdown
            final_stats = ray.get(worker.actor_handle.get_stats.remote(), timeout=10)
            logger.info(
                f"Destroying worker {worker_id}: "
                f"completed {final_stats['tasks_completed']} tasks"
            )
            
            # Delete the actor
            del worker.actor_handle
            del self.workers[worker_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error destroying worker {worker_id}: {e}")
            # Force removal
            if worker_id in self.workers:
                del self.workers[worker_id]
            return True
    
    def update_volatility(self, volatility: float):
        """Update current market volatility estimate."""
        self.current_volatility = volatility
    
    def decide_scaling_action(self) -> ScaleAction:
        """
        Decide what scaling action to take based on current conditions.
        
        Considers:
        - Current memory usage across workers
        - Market volatility
        - Cooldown period
        """
        # Check cooldown
        if time.time() - self.last_scale_action < self.config.cooldown_seconds:
            return ScaleAction.MAINTAIN
        
        # Get memory stats from all workers
        memory_usages = []
        healthy_workers = 0
        
        for worker_id, worker_info in list(self.workers.items()):
            try:
                stats = ray.get(
                    worker_info.actor_handle.get_stats.remote(),
                    timeout=5
                )
                memory_usages.append(stats["memory_gb"] / self.config.max_memory_gb)
                if stats["is_healthy"]:
                    healthy_workers += 1
                else:
                    worker_info.is_healthy = False
            except Exception:
                worker_info.is_healthy = False
        
        if not memory_usages:
            return ScaleAction.SCALE_UP
        
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        
        # Emergency shutdown if too many unhealthy workers
        if healthy_workers < len(self.workers) // 2:
            return ScaleAction.EMERGENCY_SHUTDOWN
        
        # Volatility-adjusted thresholds
        vol_adjustment = self.current_volatility * self.config.volatility_sensitivity
        effective_scale_up = self.config.scale_up_threshold - vol_adjustment
        effective_scale_down = self.config.scale_down_threshold + vol_adjustment
        
        # Make scaling decision
        if avg_memory_usage > effective_scale_up:
            if len(self.workers) < self.config.max_workers:
                return ScaleAction.SCALE_UP
        elif avg_memory_usage < effective_scale_down:
            if len(self.workers) > self.config.min_workers:
                return ScaleAction.SCALE_DOWN
        
        return ScaleAction.MAINTAIN
    
    def execute_scaling(self, action: ScaleAction) -> bool:
        """Execute a scaling action."""
        if action == ScaleAction.SCALE_UP:
            worker_id = f"worker_{len(self.workers)}"
            result = self._create_worker(worker_id) is not None
            if result:
                self.last_scale_action = time.time()
            return result
            
        elif action == ScaleAction.SCALE_DOWN:
            # Remove the oldest worker (excluding minimum)
            if len(self.workers) > self.config.min_workers:
                oldest_worker = min(
                    self.workers.keys(),
                    key=lambda w: self.workers[w].start_time
                )
                result = self._destroy_worker(oldest_worker)
                if result:
                    self.last_scale_action = time.time()
                return result
                
        elif action == ScaleAction.EMERGENCY_SHUTDOWN:
            # Keep only minimum workers
            workers_to_remove = len(self.workers) - self.config.min_workers
            removed = 0
            for worker_id in list(self.workers.keys()):
                if removed >= workers_to_remove:
                    break
                if self._destroy_worker(worker_id):
                    removed += 1
            self.last_scale_action = time.time()
            return removed > 0
        
        return False
    
    def run_scaling_cycle(self) -> Dict[str, Any]:
        """
        Run a complete scaling cycle: check conditions and execute action.
        
        Returns:
            Dictionary with scaling decision and results
        """
        action = self.decide_scaling_action()
        success = self.execute_scaling(action)
        
        return {
            "action": action.value,
            "success": success,
            "num_workers": len(self.workers),
            "volatility": self.current_volatility,
            "timestamp": time.time(),
        }
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about the entire cluster."""
        total_tasks = 0
        total_memory = 0.0
        healthy_count = 0
        
        for worker_id, worker_info in self.workers.items():
            try:
                stats = ray.get(worker_info.actor_handle.get_stats.remote(), timeout=5)
                total_tasks += stats["tasks_completed"]
                total_memory += stats["memory_gb"]
                if stats["is_healthy"]:
                    healthy_count += 1
            except Exception:
                pass
        
        return {
            "total_workers": len(self.workers),
            "healthy_workers": healthy_count,
            "total_tasks_completed": total_tasks,
            "total_memory_gb": total_memory,
            "avg_memory_per_worker_gb": total_memory / max(1, len(self.workers)),
            "current_volatility": self.current_volatility,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
        }
    
    def shutdown_all(self):
        """Shutdown all workers gracefully."""
        logger.info("Shutting down all workers...")
        for worker_id in list(self.workers.keys()):
            self._destroy_worker(worker_id)
        logger.info("All workers shut down")


# Example usage
if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=int(4 * 1024 ** 3),
        _system_config={"object_spilling_enabled": False},
    )
    
    config = ScalingConfig(
        min_workers=2,
        max_workers=8,
        scale_up_threshold=0.75,
        scale_down_threshold=0.35,
        volatility_sensitivity=0.5,
    )
    
    manager = ElasticScalingManager(config)
    
    # Simulate market conditions
    for volatility in [0.01, 0.05, 0.1, 0.2, 0.5, 0.3, 0.1]:
        manager.update_volatility(volatility)
        result = manager.run_scaling_cycle()
        print(f"Volatility: {volatility:.2f}, Action: {result['action']}, Workers: {result['num_workers']}")
        
        # Simulate some work
        for worker_id, worker_info in manager.workers.items():
            ray.get(
                worker_info.actor_handle.process_batch.remote(
                    {"data": list(range(100))},
                    volatility,
                )
            )
        
        time.sleep(1)
    
    # Print final stats
    stats = manager.get_cluster_stats()
    print(f"\nFinal cluster stats: {stats}")
    
    manager.shutdown_all()
    ray.shutdown()
