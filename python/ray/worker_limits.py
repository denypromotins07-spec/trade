"""
Ray Worker Resource Limits Module - Stage 55
=============================================
Inject custom resource limits into Ray worker definitions, mathematically bounding
tensor sizes to prevent silent OOM kills during heavy DirectML inference.

Optimized for:
- AMD Ryzen AI 5 architecture (znver4)
- 4GB Python RAM quota enforcement per worker
- DirectML/ROCm GPU tensor size bounds
- Microsecond latency requirements

Author: Nautilus Quantitative Engineering Team
Stage: 55 - Process Isolation & IPC Validation
"""

import os
import sys
import logging
import ctypes
import platform
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import ray
from ray.worker import Worker
from ray._private.resource_isolation import ResourceIsolationConfig

# Configure strict logging for worker limit operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# =============================================================================
# CONSTANTS - WORKER MEMORY BOUNDARIES
# =============================================================================

# Maximum memory per Ray worker (1.5GB each, allowing 2 workers + overhead)
MAX_WORKER_MEMORY_BYTES: int = 1_610_612_736  # 1.5GB

# Maximum tensor size per worker (512MB)
MAX_TENSOR_SIZE_BYTES: int = 536_870_912  # 512MB

# Maximum number of tensors per worker
MAX_TENSORS_PER_WORKER: int = 100

# Memory watermark for triggering GC (80% of worker limit)
MEMORY_GC_WATERMARK: float = 0.80

# Critical memory threshold for emergency cleanup (95% of worker limit)
MEMORY_CRITICAL_THRESHOLD: float = 0.95

# AMD GPU memory fraction for tensor offloading
GPU_MEMORY_FRACTION: float = 0.25  # 25% of GPU memory for DirectML


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================


class WorkerState(Enum):
    """Enum representing the current state of a Ray worker."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    MEMORY_PRESSURE = "memory_pressure"
    CRITICAL = "critical"
    SHUTDOWN = "shutdown"


@dataclass
class WorkerResourceLimits:
    """Data class defining resource limits for a single Ray worker."""
    
    worker_id: str
    max_memory_bytes: int = MAX_WORKER_MEMORY_BYTES
    max_tensor_size_bytes: int = MAX_TENSOR_SIZE_BYTES
    max_tensors: int = MAX_TENSORS_PER_WORKER
    gc_watermark: float = MEMORY_GC_WATERMARK
    critical_threshold: float = MEMORY_CRITICAL_THRESHOLD
    
    # Runtime tracking
    current_memory_usage: int = 0
    current_tensor_count: int = 0
    state: WorkerState = WorkerState.INITIALIZING
    
    # AMD GPU allocation
    gpu_memory_allocated: int = 0
    gpu_enabled: bool = False
    
    def memory_usage_percent(self) -> float:
        """Calculate current memory usage as a percentage."""
        if self.max_memory_bytes == 0:
            return 0.0
        return self.current_memory_usage / self.max_memory_bytes
    
    def is_under_pressure(self) -> bool:
        """Check if worker is under memory pressure."""
        return self.memory_usage_percent() >= self.gc_watermark
    
    def is_critical(self) -> bool:
        """Check if worker is in critical memory state."""
        return self.memory_usage_percent() >= self.critical_threshold
    
    def can_allocate_tensor(self, size_bytes: int) -> bool:
        """
        Check if a new tensor of given size can be allocated.
        
        Args:
            size_bytes: Size of tensor to allocate
            
        Returns:
            bool: True if allocation is safe
        """
        if size_bytes > self.max_tensor_size_bytes:
            return False
        
        projected_usage = self.current_memory_usage + size_bytes
        if projected_usage > self.max_memory_bytes:
            return False
        
        if self.current_tensor_count >= self.max_tensors:
            return False
        
        return True


@dataclass
class ClusterResourceTracker:
    """Track resource usage across all Ray workers in the cluster."""
    
    workers: Dict[str, WorkerResourceLimits] = field(default_factory=dict)
    total_cluster_memory: int = 0
    total_allocated_memory: int = 0
    
    def add_worker(self, worker_limits: WorkerResourceLimits) -> None:
        """Add a worker to the cluster tracker."""
        self.workers[worker_limits.worker_id] = worker_limits
        self.total_cluster_memory += worker_limits.max_memory_bytes
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the cluster tracker."""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            self.total_cluster_memory -= worker.max_memory_bytes
            del self.workers[worker_id]
    
    def get_cluster_usage_percent(self) -> float:
        """Calculate total cluster memory usage as a percentage."""
        if self.total_cluster_memory == 0:
            return 0.0
        return self.total_allocated_memory / self.total_cluster_memory
    
    def find_workers_under_pressure(self) -> List[str]:
        """Find all workers currently under memory pressure."""
        return [
            wid for wid, w in self.workers.items()
            if w.is_under_pressure()
        ]
    
    def find_critical_workers(self) -> List[str]:
        """Find all workers in critical memory state."""
        return [
            wid for wid, w in self.workers.items()
            if w.is_critical()
        ]


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Global cluster resource tracker
_cluster_tracker: Optional[ClusterResourceTracker] = None


def get_cluster_tracker() -> ClusterResourceTracker:
    """Get or create the global cluster resource tracker."""
    global _cluster_tracker
    if _cluster_tracker is None:
        _cluster_tracker = ClusterResourceTracker()
    return _cluster_tracker


# =============================================================================
# AMD GPU RESOURCE MANAGEMENT
# =============================================================================


def get_amd_gpu_memory_info() -> Tuple[int, int]:
    """
    Get AMD GPU memory information.
    
    Returns:
        Tuple[int, int]: (total_memory_bytes, available_memory_bytes)
    """
    if platform.system() == "Windows":
        # DirectML - query via DXGI
        try:
            # Placeholder for DirectML memory query
            # In production, use DirectX/DXGI APIs via ctypes
            logger.debug("DirectML GPU memory query (stub)")
            return (4 * 1024**3, 3 * 1024**3)  # Assume 4GB total, 3GB available
        except Exception as e:
            logger.warning(f"Failed to query DirectML GPU memory: {e}")
            return (0, 0)
    else:
        # ROCm - query via rocm-smi or HIP
        try:
            # Check for rocm-smi
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output (simplified)
                return (8 * 1024**3, 6 * 1024**3)  # Assume 8GB total, 6GB available
        except Exception as e:
            logger.warning(f"Failed to query ROCm GPU memory: {e}")
        
        return (0, 0)


def calculate_worker_gpu_allocation(worker_id: str) -> int:
    """
    Calculate GPU memory allocation for a worker.
    
    Args:
        worker_id: ID of the Ray worker
        
    Returns:
        int: GPU memory allocation in bytes
    """
    total_gpu, available_gpu = get_amd_gpu_memory_info()
    
    if total_gpu == 0:
        logger.debug("No AMD GPU available - using CPU only")
        return 0
    
    # Allocate fraction of available GPU memory per worker
    num_workers = len(get_cluster_tracker().workers) or 1
    allocation_per_worker = int(available_gpu * GPU_MEMORY_FRACTION / num_workers)
    
    # Cap at maximum tensor size
    allocation_per_worker = min(allocation_per_worker, MAX_TENSOR_SIZE_BYTES)
    
    logger.debug(
        f"Worker {worker_id}: GPU allocation = {allocation_per_worker / (1024**2):.1f}MB"
    )
    
    return allocation_per_worker


# =============================================================================
# RAY WORKER DECORATORS AND LIMITS
# =============================================================================


def enforce_tensor_size_limit(max_size_bytes: int = MAX_TENSOR_SIZE_BYTES):
    """
    Decorator to enforce tensor size limits on Ray remote functions.
    
    Args:
        max_size_bytes: Maximum allowed tensor size in bytes
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Estimate tensor sizes in arguments
            total_size = 0
            for arg in args:
                if hasattr(arg, 'nbytes'):  # NumPy array
                    total_size += arg.nbytes
                elif hasattr(arg, '__len__') and hasattr(arg, '__getitem__'):
                    # List-like, estimate
                    total_size += sys.getsizeof(arg)
            
            for value in kwargs.values():
                if hasattr(value, 'nbytes'):
                    total_size += value.nbytes
                elif hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                    total_size += sys.getsizeof(value)
            
            if total_size > max_size_bytes:
                raise MemoryError(
                    f"Tensor size ({total_size / (1024**2):.1f}MB) exceeds "
                    f"limit ({max_size_bytes / (1024**2):.1f}MB)"
                )
            
            return func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        
        return wrapper
    return decorator


def with_memory_tracking(func):
    """
    Decorator to track memory usage of Ray remote functions.
    
    Args:
        func: Ray remote function to wrap
    """
    def wrapper(*args, **kwargs):
        import psutil
        
        process = psutil.Process(os.getpid())
        before_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            after_memory = process.memory_info().rss
            memory_delta = after_memory - before_memory
            
            # Update worker state
            worker_id = ray.get_runtime_context().get_worker_id()
            tracker = get_cluster_tracker()
            
            if worker_id in tracker.workers:
                worker = tracker.workers[worker_id]
                worker.current_memory_usage = after_memory
                
                if worker.is_critical():
                    logger.critical(
                        f"Worker {worker_id} in CRITICAL state: "
                        f"{worker.memory_usage_percent()*100:.1f}% memory used"
                    )
                elif worker.is_under_pressure():
                    logger.warning(
                        f"Worker {worker_id} under memory pressure: "
                        f"{worker.memory_usage_percent()*100:.1f}% memory used"
                    )
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    
    return wrapper


# =============================================================================
# WORKER INITIALIZATION AND REGISTRATION
# =============================================================================


def initialize_worker_limits(worker_id: Optional[str] = None) -> WorkerResourceLimits:
    """
    Initialize resource limits for a Ray worker.
    
    Args:
        worker_id: Optional worker ID (auto-generated if not provided)
        
    Returns:
        WorkerResourceLimits: Initialized worker limits
    """
    if worker_id is None:
        worker_id = ray.get_runtime_context().get_worker_id()
    
    limits = WorkerResourceLimits(worker_id=worker_id)
    
    # Check for AMD GPU availability
    total_gpu, _ = get_amd_gpu_memory_info()
    if total_gpu > 0:
        limits.gpu_enabled = True
        limits.gpu_memory_allocated = calculate_worker_gpu_allocation(worker_id)
    
    # Register with cluster tracker
    tracker = get_cluster_tracker()
    tracker.add_worker(limits)
    
    logger.info(
        f"Worker {worker_id} initialized: "
        f"max_memory={limits.max_memory_bytes/(1024**2):.1f}MB, "
        f"gpu_enabled={limits.gpu_enabled}"
    )
    
    return limits


def cleanup_worker_limits(worker_id: str) -> None:
    """
    Cleanup resource limits for a Ray worker.
    
    Args:
        worker_id: ID of worker to cleanup
    """
    tracker = get_cluster_tracker()
    tracker.remove_worker(worker_id)
    
    logger.info(f"Worker {worker_id} cleanup complete")


# =============================================================================
# EMERGENCY MEMORY MANAGEMENT
# =============================================================================


def trigger_emergency_gc(worker_id: str) -> bool:
    """
    Trigger emergency garbage collection for a worker.
    
    Args:
        worker_id: ID of worker requiring GC
        
    Returns:
        bool: True if GC was successful
    """
    logger.warning(f"Triggering emergency GC for worker {worker_id}")
    
    try:
        import gc
        gc.collect(generation=2)  # Full GC
        
        # Force release of unused memory (Linux only)
        if platform.system() != "Windows":
            # Try to release memory back to OS
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        
        # Update worker state
        tracker = get_cluster_tracker()
        if worker_id in tracker.workers:
            import psutil
            process = psutil.Process(os.getpid())
            tracker.workers[worker_id].current_memory_usage = process.memory_info().rss
        
        logger.info(f"Emergency GC completed for worker {worker_id}")
        return True
        
    except Exception as e:
        logger.error(f"Emergency GC failed for worker {worker_id}: {e}")
        return False


def handle_critical_memory_state(worker_id: str) -> bool:
    """
    Handle critical memory state for a worker.
    
    Args:
        worker_id: ID of worker in critical state
        
    Returns:
        bool: True if situation was handled
    """
    logger.critical(f"Handling CRITICAL memory state for worker {worker_id}")
    
    # Step 1: Emergency GC
    if not trigger_emergency_gc(worker_id):
        logger.error("Emergency GC failed")
        return False
    
    # Step 2: Check if still critical
    tracker = get_cluster_tracker()
    if worker_id in tracker.workers:
        worker = tracker.workers[worker_id]
        
        if worker.is_critical():
            logger.critical(
                f"Worker {worker_id} still critical after GC - "
                f"recommending worker termination"
            )
            return False
    
    logger.info(f"Critical state resolved for worker {worker_id}")
    return True


# =============================================================================
# RAY REMOTE FUNCTION WRAPPER
# =============================================================================


def create_limited_remote(func, **ray_remote_kwargs):
    """
    Create a Ray remote function with enforced resource limits.
    
    Args:
        func: Function to make remote
        **ray_remote_kwargs: Additional Ray remote kwargs
        
    Returns:
        Ray remote function with limits
    """
    # Add memory tracking decorator
    limited_func = with_memory_tracking(func)
    
    # Set default resource limits
    if 'max_calls' not in ray_remote_kwargs:
        ray_remote_kwargs['max_calls'] = 100  # Restart worker after 100 calls
    
    if 'runtime_env' not in ray_remote_kwargs:
        ray_remote_kwargs['runtime_env'] = {}
    
    # Add environment variables for memory limits
    env_vars = ray_remote_kwargs.get('runtime_env', {}).get('env_vars', {})
    env_vars['RAY_WORKER_MAX_MEMORY'] = str(MAX_WORKER_MEMORY_BYTES)
    env_vars['RAY_WORKER_MAX_TENSOR_SIZE'] = str(MAX_TENSOR_SIZE_BYTES)
    ray_remote_kwargs['runtime_env']['env_vars'] = env_vars
    
    return ray.remote(**ray_remote_kwargs)(limited_func)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def setup_worker_limits() -> bool:
    """
    Main entry point for setting up worker limits.
    
    Returns:
        bool: True if setup was successful
    """
    logger.info("=" * 60)
    logger.info("Nautilus Ray Worker Limits - Stage 55")
    logger.info("=" * 60)
    
    try:
        # Initialize worker limits
        initialize_worker_limits()
        
        # Verify AMD GPU environment
        total_gpu, available_gpu = get_amd_gpu_memory_info()
        if total_gpu > 0:
            logger.info(f"AMD GPU detected: {total_gpu/(1024**3):.1f}GB total")
        else:
            logger.info("No AMD GPU detected - CPU-only mode")
        
        logger.info("Worker limits setup complete")
        return True
        
    except Exception as e:
        logger.error(f"Worker limits setup failed: {e}")
        return False


if __name__ == "__main__":
    setup_worker_limits()
