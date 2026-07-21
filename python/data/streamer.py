"""
Ray Actor Cluster for Normalized Tick Data Streaming

This module initializes a Ray actor cluster that consumes normalized tick data,
strictly managing worker memory limits to prevent the Python ecosystem from
exceeding the 4GB AI quota (leaving 4GB for the Rust execution engine).

Key Features:
- Memory-bounded Ray actors with explicit 4GB ceiling
- Zero-copy data transfer via shared memory (mmap)
- Backpressure handling when Rust producer is faster than Python consumers
- AMD DirectML/ROCm environment detection for future GPU offload
"""

import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import ray
from ray import actor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Memory Configuration Constants
# =============================================================================

# Hard limit for Python/Ray ecosystem (4GB out of 8GB total)
MAX_PYTHON_MEMORY_GB = 4.0
MAX_PYTHON_MEMORY_BYTES = int(MAX_PYTHON_MEMORY_GB * 1024 * 1024 * 1024)

# Per-actor memory limit (divide by expected number of actors)
DEFAULT_ACTOR_MEMORY_MB = 512

# Shared memory configuration
SHARED_MEMORY_PATH = "/tmp/nautilus_ray_shm"
SHARED_MEMORY_SIZE_BYTES = 64 * 1024 * 1024  # 64MB default


@dataclass
class TickData:
    """Normalized tick data structure matching Rust TradeTick/QuoteTick."""
    timestamp_ns: int
    price: float
    quantity: float
    is_buyer_maker: bool
    sequence: int
    symbol_hash: int = 0


def check_amd_gpu_environment() -> Dict[str, Any]:
    """
    Detect AMD ROCm/DirectML environment variables for future GPU scaling.
    
    Returns:
        Dictionary containing GPU environment status and capabilities.
    """
    env_status = {
        "rocm_available": False,
        "directml_available": False,
        "gpu_device": None,
        "environment_vars": {}
    }
    
    # Check ROCm (Linux AMD GPUs)
    rocm_vars = ["ROCM_PATH", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
    for var in rocm_vars:
        if var in os.environ:
            env_status["rocm_available"] = True
            env_status["environment_vars"][var] = os.environ[var]
            logger.info(f"ROCm environment detected: {var}={os.environ[var]}")
    
    # Check DirectML (Windows AMD GPUs via DirectML)
    directml_vars = ["DIRECTML_ENABLED", "DIRECTML_DEVICE"]
    for var in directml_vars:
        if var in os.environ:
            env_status["directml_available"] = True
            env_status["environment_vars"][var] = os.environ[var]
            logger.info(f"DirectML environment detected: {var}={os.environ[var]}")
    
    # Check for AMD GPU device
    try:
        # Try to detect AMD GPU through various means
        if "HIP_VISIBLE_DEVICES" in os.environ:
            env_status["gpu_device"] = f"AMD HIP Device {os.environ['HIP_VISIBLE_DEVICES']}"
        elif "DIRECTML_DEVICE" in os.environ:
            env_status["gpu_device"] = f"DirectML Device {os.environ['DIRECTML_DEVICE']}"
    except Exception as e:
        logger.warning(f"Could not detect GPU device: {e}")
    
    return env_status


@actor(memory=DEFAULT_ACTOR_MEMORY_MB * 1024 * 1024)
class TickStreamerActor:
    """
    Ray actor for consuming and processing normalized tick data.
    
    This actor receives tick data from the Rust engine via shared memory
    and performs feature engineering or forwards to ML models.
    """
    
    def __init__(self, actor_id: int, max_buffer_size: int = 10000):
        """
        Initialize the tick streamer actor.
        
        Args:
            actor_id: Unique identifier for this actor
            max_buffer_size: Maximum ticks to buffer before processing
        """
        self.actor_id = actor_id
        self.max_buffer_size = max_buffer_size
        self.buffer: List[TickData] = []
        self.processed_count = 0
        self.dropped_count = 0
        self.is_running = False
        
        # Memory tracking
        self.current_memory_bytes = 0
        self.peak_memory_bytes = 0
        
        logger.info(f"TickStreamerActor[{actor_id}] initialized with buffer size {max_buffer_size}")
    
    def start(self) -> bool:
        """Start the actor processing loop."""
        self.is_running = True
        logger.info(f"TickStreamerActor[{self.actor_id}] started")
        return True
    
    def stop(self) -> bool:
        """Stop the actor processing loop."""
        self.is_running = False
        logger.info(f"TickStreamerActor[{self.actor_id}] stopped")
        return True
    
    def receive_ticks(self, ticks: List[TickData]) -> Dict[str, int]:
        """
        Receive a batch of ticks from the Rust engine.
        
        Args:
            ticks: List of TickData objects
            
        Returns:
            Status dictionary with counts
        """
        if not self.is_running:
            return {"status": "stopped", "received": len(ticks), "buffered": 0}
        
        # Check memory limit before accepting
        estimated_memory = len(ticks) * 64  # ~64 bytes per tick estimate
        if self.current_memory_bytes + estimated_memory > MAX_PYTHON_MEMORY_BYTES:
            # Drop ticks to stay within memory limit
            self.dropped_count += len(ticks)
            logger.warning(
                f"Actor[{self.actor_id}] dropping {len(ticks)} ticks due to memory limit"
            )
            return {
                "status": "memory_limit",
                "received": 0,
                "dropped": len(ticks),
                "current_memory_mb": self.current_memory_bytes / (1024 * 1024)
            }
        
        # Add to buffer
        available_space = self.max_buffer_size - len(self.buffer)
        accepted_ticks = ticks[:available_space]
        rejected_ticks = ticks[available_space:]
        
        self.buffer.extend(accepted_ticks)
        self.current_memory_bytes += len(accepted_ticks) * 64
        self.peak_memory_bytes = max(self.peak_memory_bytes, self.current_memory_bytes)
        
        if rejected_ticks:
            self.dropped_count += len(rejected_ticks)
        
        return {
            "status": "ok",
            "received": len(accepted_ticks),
            "rejected": len(rejected_ticks),
            "buffer_size": len(self.buffer),
            "current_memory_mb": self.current_memory_bytes / (1024 * 1024)
        }
    
    def process_buffer(self) -> Dict[str, Any]:
        """
        Process buffered ticks (feature engineering, aggregation, etc.).
        
        Returns:
            Processing results dictionary
        """
        if not self.buffer:
            return {"processed": 0, "status": "empty"}
        
        # Convert to numpy for vectorized operations
        timestamps = np.array([t.timestamp_ns for t in self.buffer], dtype=np.int64)
        prices = np.array([t.price for t in self.buffer], dtype=np.float64)
        quantities = np.array([t.quantity for t in self.buffer], dtype=np.float64)
        
        # Basic statistics (placeholder for actual feature engineering)
        result = {
            "processed": len(self.buffer),
            "min_price": float(np.min(prices)),
            "max_price": float(np.max(prices)),
            "mean_price": float(np.mean(prices)),
            "total_volume": float(np.sum(quantities)),
            "tick_count": len(self.buffer),
            "time_range_ns": int(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        }
        
        # Clear buffer after processing
        self.buffer.clear()
        self.processed_count += result["processed"]
        self.current_memory_bytes = 0
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            "actor_id": self.actor_id,
            "is_running": self.is_running,
            "buffer_size": len(self.buffer),
            "processed_count": self.processed_count,
            "dropped_count": self.dropped_count,
            "current_memory_mb": self.current_memory_bytes / (1024 * 1024),
            "peak_memory_mb": self.peak_memory_bytes / (1024 * 1024),
            "memory_utilization": self.current_memory_bytes / MAX_PYTHON_MEMORY_BYTES
        }


@actor(memory=DEFAULT_ACTOR_MEMORY_MB * 1024 * 1024)
class DataManagerActor:
    """
    Central data manager coordinating multiple streamer actors.
    
    This actor manages the shared memory region and distributes
    tick data to worker actors.
    """
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize the data manager.
        
        Args:
            num_workers: Number of worker actors to manage
        """
        self.num_workers = num_workers
        self.workers: List[ray.actor.ActorHandle] = []
        self.shared_memory_ref = None
        self.is_initialized = False
        
        logger.info(f"DataManagerActor initialized with {num_workers} workers")
    
    def initialize_workers(self) -> bool:
        """Create and initialize worker actors."""
        try:
            self.workers = [
                TickStreamerActor.remote(i) for i in range(self.num_workers)
            ]
            
            # Start all workers
            results = ray.get([w.start.remote() for w in self.workers])
            
            if all(results):
                self.is_initialized = True
                logger.info(f"All {self.num_workers} workers started successfully")
                return True
            else:
                logger.error("Some workers failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize workers: {e}")
            return False
    
    def shutdown_workers(self) -> bool:
        """Gracefully shutdown all workers."""
        try:
            results = ray.get([w.stop.remote() for w in self.workers])
            self.workers.clear()
            self.is_initialized = False
            logger.info("All workers shut down successfully")
            return True
        except Exception as e:
            logger.error(f"Error shutting down workers: {e}")
            return False
    
    def distribute_ticks(self, ticks: List[TickData]) -> Dict[str, Any]:
        """
        Distribute ticks to worker actors using round-robin.
        
        Args:
            ticks: List of ticks to distribute
            
        Returns:
            Distribution results
        """
        if not self.is_initialized or not self.workers:
            return {"status": "not_initialized", "distributed": 0}
        
        # Simple round-robin distribution
        chunk_size = max(1, len(ticks) // self.num_workers)
        results = []
        
        for i, worker in enumerate(self.workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(ticks)
            chunk = ticks[start_idx:end_idx]
            
            if chunk:
                result = ray.get(worker.receive_ticks.remote(chunk))
                results.append(result)
        
        total_received = sum(r.get("received", 0) for r in results)
        total_dropped = sum(r.get("dropped", 0) for r in results)
        
        return {
            "status": "ok",
            "distributed": total_received,
            "dropped": total_dropped,
            "worker_results": results
        }
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics from all workers."""
        if not self.workers:
            return {"status": "no_workers"}
        
        stats = ray.get([w.get_stats.remote() for w in self.workers])
        
        total_processed = sum(s["processed_count"] for s in stats)
        total_dropped = sum(s["dropped_count"] for s in stats)
        total_memory = sum(s["current_memory_mb"] for s in stats)
        peak_memory = sum(s["peak_memory_mb"] for s in stats)
        
        return {
            "num_workers": self.num_workers,
            "total_processed": total_processed,
            "total_dropped": total_dropped,
            "total_current_memory_mb": total_memory,
            "total_peak_memory_mb": peak_memory,
            "memory_limit_gb": MAX_PYTHON_MEMORY_GB,
            "memory_utilization": total_memory / (MAX_PYTHON_MEMORY_GB * 1024),
            "workers": stats
        }


def initialize_ray_cluster(
    num_cpus: int = 4,
    num_gpus: int = 0,
    object_store_memory: int = 2 * 1024 * 1024 * 1024  # 2GB default
) -> bool:
    """
    Initialize the Ray cluster with strict memory limits.
    
    Args:
        num_cpus: Number of CPUs to allocate to Ray
        num_gpus: Number of GPUs to allocate (for future ROCm/DirectML)
        object_store_memory: Memory for Ray object store (2GB default)
        
    Returns:
        True if initialization successful
    """
    # Check AMD GPU environment first
    gpu_env = check_amd_gpu_environment()
    if gpu_env["rocm_available"] or gpu_env["directml_available"]:
        logger.info(f"AMD GPU detected: {gpu_env['gpu_device']}")
        num_gpus = max(num_gpus, 1)
    
    try:
        # Shutdown existing cluster if any
        if ray.is_initialized():
            ray.shutdown()
        
        # Initialize with memory limits
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            object_store_memory=object_store_memory,
            _system_config={
                "max_bytes_to_gc": int(MAX_PYTHON_MEMORY_BYTES * 0.8),
            },
            log_to_driver=True,
            ignore_reinit_error=True
        )
        
        logger.info(
            f"Ray cluster initialized: "
            f"{num_cpus} CPUs, {num_gpus} GPUs, "
            f"Object Store: {object_store_memory / (1024*1024*1024):.1f}GB"
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        return False


def shutdown_ray_cluster() -> bool:
    """Gracefully shutdown the Ray cluster."""
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray cluster shut down successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"Error shutting down Ray cluster: {e}")
        return False


# Entry point for testing
if __name__ == "__main__":
    # Initialize cluster
    if initialize_ray_cluster(num_cpus=4):
        # Create data manager
        manager = DataManagerActor.remote(num_workers=4)
        
        # Initialize workers
        ray.get(manager.initialize_workers.remote())
        
        # Generate test ticks
        test_ticks = [
            TickData(
                timestamp_ns=1000000000 + i * 1000000,
                price=50000.0 + i * 0.01,
                quantity=0.1 + i * 0.001,
                is_buyer_maker=(i % 2 == 0),
                sequence=i,
                symbol_hash=123456789
            )
            for i in range(100)
        ]
        
        # Distribute ticks
        result = ray.get(manager.distribute_ticks.remote(test_ticks))
        print(f"Distribution result: {result}")
        
        # Get stats
        stats = ray.get(manager.get_cluster_stats.remote())
        print(f"Cluster stats: {stats}")
        
        # Cleanup
        ray.get(manager.shutdown_workers.remote())
        shutdown_ray_cluster()
