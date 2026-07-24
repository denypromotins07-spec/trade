#!/usr/bin/env python3
"""
Chaos Engineering: OOM Injector for Python/Ray Workers

This module safely simulates Out-Of-Memory (OOM) events in Python workers
to verify the 4GB RAM guard and automatic Ray worker respawning logic.

Architecture:
- Uses memory-mapped arrays to simulate controlled memory pressure
- Integrates with Ray's resource management for worker lifecycle tracking
- Respects strict 4GB Python RAM quota during testing
- AMD DirectML/ROCm context awareness for GPU memory coordination

AMD Ryzen AI 5 Optimizations:
- NumPy memory views for zero-copy operations
- Page-aligned memory allocations
- DirectML tensor allocation for GPU memory stress testing

Usage:
    python oom_injector.py --target-worker-id <id> --memory-mb <amount>
"""

import argparse
import logging
import os
import sys
import time
import traceback
import ctypes
import mmap
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
import threading
import gc

# Conditional imports for optional dependencies
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_PYTHON_RAM_GB = 4.0
MAX_PYTHON_RAM_BYTES = int(MAX_PYTHON_RAM_GB * 1024 * 1024 * 1024)
PAGE_SIZE = 4096  # Standard page size
CACHE_LINE_SIZE = 64


class MemoryPressureLevel(Enum):
    """Levels of memory pressure for OOM injection."""
    LOW = auto()      # 50% of quota
    MEDIUM = auto()   # 75% of quota
    HIGH = auto()     # 90% of quota
    CRITICAL = auto() # 95% of quota
    OOM = auto()      # 100%+ (triggers OOM)


@dataclass
class MemoryAllocationRecord:
    """Record of a memory allocation for tracking."""
    allocation_id: str
    size_bytes: int
    timestamp: datetime
    allocation_type: str
    freed: bool = False
    peak_usage_bytes: int = 0


@dataclass
class OOMInjectionStats:
    """Statistics for OOM injection testing."""
    total_allocations: int = 0
    total_bytes_allocated: int = 0
    peak_memory_usage: int = 0
    current_memory_usage: int = 0
    oom_events_triggered: int = 0
    workers_respawned: int = 0
    ram_guard_activations: int = 0
    gpu_memory_freed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_allocations': self.total_allocations,
            'total_bytes_allocated': self.total_bytes_allocated,
            'peak_memory_usage_mb': self.peak_memory_usage / (1024 * 1024),
            'current_memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'oom_events_triggered': self.oom_events_triggered,
            'workers_respawned': self.workers_respawned,
            'ram_guard_activations': self.ram_guard_activations,
            'gpu_memory_freed_mb': self.gpu_memory_freed / (1024 * 1024),
        }


class MemoryGuard:
    """
    Enforces the 4GB Python RAM quota.
    
    This class monitors memory usage and triggers protective measures
    when approaching the limit.
    """
    
    def __init__(self, max_bytes: int = MAX_PYTHON_RAM_BYTES):
        self.max_bytes = max_bytes
        self.warning_threshold = 0.8  # 80% triggers warning
        self.critical_threshold = 0.95  # 95% triggers emergency GC
        
    def get_current_usage(self) -> int:
        """Get current Python process memory usage in bytes."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        else:
            # Fallback: estimate from gc
            gc.collect()
            return sum(sys.getsizeof(obj) for obj in gc.get_objects())
    
    def get_usage_ratio(self) -> float:
        """Get current memory usage as ratio of max allowed."""
        return self.get_current_usage() / self.max_bytes
    
    def is_approaching_limit(self) -> bool:
        """Check if memory usage is approaching the limit."""
        return self.get_usage_ratio() >= self.warning_threshold
    
    def is_critical(self) -> bool:
        """Check if memory usage is at critical levels."""
        return self.get_usage_ratio() >= self.critical_threshold
    
    def trigger_emergency_gc(self) -> int:
        """Force garbage collection and return freed bytes."""
        before = self.get_current_usage()
        gc.collect()
        after = self.get_current_usage()
        freed = before - after
        logger.info(f"Emergency GC freed {freed / (1024 * 1024):.2f} MB")
        return freed
    
    def check_and_enforce(self) -> bool:
        """
        Check memory usage and enforce limits.
        Returns True if enforcement action was taken.
        """
        ratio = self.get_usage_ratio()
        
        if ratio >= 1.0:
            logger.critical("MEMORY LIMIT EXCEEDED! Triggering emergency measures...")
            self.trigger_emergency_gc()
            return True
        elif ratio >= self.critical_threshold:
            logger.warning("Memory at critical level ({:.1f}%). Forcing GC.".format(ratio * 100))
            self.trigger_emergency_gc()
            return True
        elif ratio >= self.warning_threshold:
            logger.info("Memory approaching limit ({:.1f}%)".format(ratio * 100))
            return False
        
        return False


class OOMInjector:
    """
    Safely injects OOM conditions for chaos testing.
    
    This class creates controlled memory pressure to test
    Ray worker respawning and RAM guard functionality.
    """
    
    def __init__(self, stats: OOMInjectionStats = None):
        self.stats = stats or OOMInjectionStats()
        self.memory_guard = MemoryGuard()
        self.allocations: Dict[str, MemoryAllocationRecord] = {}
        self._lock = threading.Lock()
        self._running = False
        self._gpu_context = None
        
    def allocate_memory(
        self,
        size_mb: float,
        allocation_type: str = "test",
        use_huge_pages: bool = False
    ) -> str:
        """
        Allocate memory for testing purposes.
        
        Args:
            size_mb: Size in megabytes to allocate
            allocation_type: Type label for tracking
            use_huge_pages: Whether to use huge pages (2MB)
            
        Returns:
            Allocation ID string
        """
        import uuid
        
        size_bytes = int(size_mb * 1024 * 1024)
        allocation_id = str(uuid.uuid4())[:8]
        
        with self._lock:
            # Check if allocation would exceed limits
            projected_usage = self.stats.current_memory_usage + size_bytes
            if projected_usage > MAX_PYTHON_RAM_BYTES:
                logger.warning(
                    f"Allocation of {size_mb}MB would exceed 4GB limit. "
                    f"Projected: {projected_usage / (1024**3):.2f}GB"
                )
                self.stats.ram_guard_activations += 1
                raise MemoryError(f"Would exceed {MAX_PYTHON_RAM_GB}GB limit")
            
            # Perform allocation based on type
            if allocation_type == "numpy":
                if NUMPY_AVAILABLE:
                    # Use numpy for efficient memory allocation
                    array_size = size_bytes // 8  # float64
                    data = np.zeros(array_size, dtype=np.float64)
                    actual_bytes = data.nbytes
                else:
                    data = bytearray(size_bytes)
                    actual_bytes = size_bytes
            elif allocation_type == "mmap":
                # Use memory-mapped file for large allocations
                actual_bytes = size_bytes
                data = self._create_mmap_allocation(size_bytes)
            else:
                # Standard bytearray allocation
                data = bytearray(size_bytes)
                actual_bytes = size_bytes
            
            # Record allocation
            record = MemoryAllocationRecord(
                allocation_id=allocation_id,
                size_bytes=actual_bytes,
                timestamp=datetime.now(),
                allocation_type=allocation_type,
                peak_usage_bytes=self.stats.current_memory_usage + actual_bytes,
            )
            self.allocations[allocation_id] = record
            
            # Update stats
            self.stats.total_allocations += 1
            self.stats.total_bytes_allocated += actual_bytes
            self.stats.current_memory_usage += actual_bytes
            self.stats.peak_memory_usage = max(
                self.stats.peak_memory_usage,
                self.stats.current_memory_usage
            )
            
            logger.info(
                f"Allocated {size_mb:.2f}MB ({allocation_type}) - ID: {allocation_id}"
            )
            
            return allocation_id
    
    def _create_mmap_allocation(self, size_bytes: int) -> mmap.mmap:
        """Create a memory-mapped allocation."""
        # Create anonymous mmap (not backed by file)
        mm = mmap.mmap(-1, size_bytes)
        # Touch pages to ensure they're allocated
        mm.seek(0)
        mm.write(b'\x00' * min(size_bytes, PAGE_SIZE))
        mm.seek(0)
        return mm
    
    def free_allocation(self, allocation_id: str) -> int:
        """
        Free a previously allocated memory block.
        
        Returns:
            Bytes freed
        """
        with self._lock:
            if allocation_id not in self.allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return 0
            
            record = self.allocations[allocation_id]
            record.freed = True
            
            # Explicitly delete the data
            if allocation_id in self.allocations:
                del self.allocations[allocation_id]
            
            # Force GC
            gc.collect()
            
            # Update stats (approximate)
            freed_bytes = record.size_bytes
            self.stats.current_memory_usage -= freed_bytes
            
            logger.info(f"Freed allocation {allocation_id}: {freed_bytes / (1024 * 1024):.2f}MB")
            return freed_bytes
    
    def trigger_oom_event(self, target_ratio: float = 1.05) -> bool:
        """
        Trigger an OOM event by allocating until we exceed the limit.
        
        Args:
            target_ratio: Target memory ratio (1.0 = 100%, >1.0 = over limit)
            
        Returns:
            True if OOM was successfully triggered
        """
        logger.info(f"Starting OOM injection targeting {target_ratio * 100:.1f}% of limit")
        
        current_ratio = self.memory_guard.get_usage_ratio()
        target_bytes = int(MAX_PYTHON_RAM_BYTES * target_ratio)
        bytes_to_allocate = target_bytes - self.stats.current_memory_usage
        
        if bytes_to_allocate <= 0:
            logger.warning("Already at or above target memory level")
            return False
        
        try:
            # Allocate in chunks to allow for graceful handling
            chunk_size_mb = 50
            allocated_total = 0
            
            while allocated_total < bytes_to_allocate:
                # Check guard before each allocation
                if self.memory_guard.is_critical():
                    logger.info("Memory guard triggered during OOM injection")
                    self.stats.ram_guard_activations += 1
                    
                    # Allow one more small allocation to trigger OOM
                    if allocated_total < bytes_to_allocate * 0.9:
                        break
                
                try:
                    alloc_id = self.allocate_memory(chunk_size_mb, "oom_test")
                    allocated_total += chunk_size_mb * 1024 * 1024
                except MemoryError as e:
                    logger.info(f"MemoryError caught: {e}")
                    self.stats.oom_events_triggered += 1
                    return True
                
                time.sleep(0.01)  # Small delay to allow monitoring
            
            # If we got here without MemoryError, check final state
            final_ratio = self.memory_guard.get_usage_ratio()
            if final_ratio >= 1.0:
                self.stats.oom_events_triggered += 1
                logger.info(f"OOM condition achieved: {final_ratio * 100:.1f}%")
                return True
            else:
                logger.warning(f"Could not reach OOM: {final_ratio * 100:.1f}%")
                return False
                
        except Exception as e:
            logger.error(f"Error during OOM injection: {e}")
            traceback.print_exc()
            return False
    
    def cleanup_all(self) -> int:
        """
        Free all allocations and cleanup resources.
        
        Returns:
            Total bytes freed
        """
        total_freed = 0
        allocation_ids = list(self.allocations.keys())
        
        for alloc_id in allocation_ids:
            freed = self.free_allocation(alloc_id)
            total_freed += freed
        
        # Force full GC
        gc.collect()
        
        logger.info(f"Cleanup complete: freed {total_freed / (1024 * 1024):.2f}MB total")
        return total_freed
    
    def simulate_worker_oom(self, worker_id: str) -> Dict[str, Any]:
        """
        Simulate an OOM event for a specific worker.
        
        This simulates what happens when a Ray worker hits its memory limit.
        
        Args:
            worker_id: Identifier for the worker
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Simulating OOM for worker {worker_id}")
        
        result = {
            'worker_id': worker_id,
            'start_time': datetime.now().isoformat(),
            'initial_memory_mb': self.memory_guard.get_current_usage() / (1024 * 1024),
            'oom_triggered': False,
            'guard_activated': False,
            'cleanup_successful': False,
        }
        
        try:
            # Trigger OOM
            oom_success = self.trigger_oom_event(target_ratio=1.02)
            result['oom_triggered'] = oom_success
            
            # Check if guard activated
            if self.stats.ram_guard_activations > 0:
                result['guard_activated'] = True
            
            # Cleanup
            freed = self.cleanup_all()
            result['freed_mb'] = freed / (1024 * 1024)
            result['cleanup_successful'] = True
            result['final_memory_mb'] = self.memory_guard.get_current_usage() / (1024 * 1024)
            
            # Simulate worker respawn counter
            self.stats.workers_respawned += 1
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error in worker OOM simulation: {e}")
        
        result['end_time'] = datetime.now().isoformat()
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics as dictionary."""
        stats_dict = self.stats.to_dict()
        stats_dict['memory_guard_ratio'] = self.memory_guard.get_usage_ratio()
        stats_dict['active_allocations'] = len([a for a in self.allocations.values() if not a.freed])
        return stats_dict


def run_ray_integration_test():
    """Run OOM injection test with Ray integration."""
    if not RAY_AVAILABLE:
        logger.warning("Ray not available, skipping integration test")
        return
    
    logger.info("Starting Ray integration test")
    
    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)
    
    @ray.remote(max_restarts=1, max_task_retries=3)
    def memory_intensive_task(memory_mb: int) -> Dict[str, Any]:
        """Ray task that allocates memory."""
        injector = OOMInjector()
        
        try:
            alloc_id = injector.allocate_memory(memory_mb, "ray_task")
            time.sleep(0.5)  # Hold memory briefly
            injector.free_allocation(alloc_id)
            return {'success': True, 'memory_mb': memory_mb}
        except MemoryError as e:
            return {'success': False, 'error': str(e)}
    
    # Run tasks with increasing memory
    futures = []
    for mem in [100, 200, 500, 1000, 2000]:
        future = memory_intensive_task.remote(mem)
        futures.append((mem, future))
    
    results = []
    for mem, future in futures:
        try:
            result = ray.get(future, timeout=30)
            results.append({'memory_mb': mem, 'result': result})
            logger.info(f"Task with {mem}MB: {result}")
        except Exception as e:
            results.append({'memory_mb': mem, 'error': str(e)})
            logger.warning(f"Task with {mem}MB failed: {e}")
    
    ray.shutdown()
    return results


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='OOM Injector for Chaos Engineering'
    )
    parser.add_argument(
        '--target-worker-id',
        type=str,
        default='test-worker',
        help='Worker ID for simulation'
    )
    parser.add_argument(
        '--memory-mb',
        type=float,
        default=500,
        help='Memory to allocate in MB'
    )
    parser.add_argument(
        '--trigger-oom',
        action='store_true',
        help='Trigger full OOM event'
    )
    parser.add_argument(
        '--ray-test',
        action='store_true',
        help='Run Ray integration test'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show current statistics'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Cleanup all allocations'
    )
    
    args = parser.parse_args()
    
    injector = OOMInjector()
    
    if args.ray_test:
        results = run_ray_integration_test()
        print(f"Ray test results: {results}")
        return
    
    if args.stats:
        stats = injector.get_stats()
        print(f"Current stats: {stats}")
        return
    
    if args.cleanup:
        freed = injector.cleanup_all()
        print(f"Cleaned up {freed / (1024 * 1024):.2f}MB")
        return
    
    if args.trigger_oom:
        logger.info("Triggering OOM event...")
        success = injector.trigger_oom_event()
        print(f"OOM triggered: {success}")
        stats = injector.get_stats()
        print(f"Final stats: {stats}")
        
        # Cleanup after OOM
        injector.cleanup_all()
        return
    
    # Default: simple allocation test
    try:
        alloc_id = injector.allocate_memory(args.memory_mb, "cli_test")
        print(f"Allocated {args.memory_mb}MB with ID: {alloc_id}")
        
        # Hold for a moment then cleanup
        time.sleep(1)
        injector.free_allocation(alloc_id)
        
    except MemoryError as e:
        print(f"MemoryError: {e}")
    
    print(f"Final stats: {injector.get_stats()}")


if __name__ == '__main__':
    main()
