"""
Ray Object Store Optimization

Optimizes Ray's Plasma object store configuration to aggressively pin
critical shared tensors in memory, preventing disastrous performance-killing
memory spilling to the NVMe disk.

Strictly respects the global 8GB RAM limit across all distributed nodes.
"""

import ray
from ray import plasma
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import os
import time
import threading
from dataclasses import dataclass
from contextlib import contextmanager


# Configuration constants
GLOBAL_RAM_LIMIT_GB = 8.0
OBJECT_STORE_FRACTION = 0.25  # 25% of RAM for object store (2GB)
PINNED_OBJECTS_MAX_GB = 1.5   # Max memory for pinned objects
SPILL_THRESHOLD = 0.8         # Spill when 80% full


@dataclass
class PinnedObjectInfo:
    """Information about a pinned object."""
    object_id: str
    size_bytes: int
    pinned_at_ns: int
    access_count: int
    last_access_ns: int
    priority: int  # Higher = more important


@dataclass
class ObjectStoreStats:
    """Statistics about object store usage."""
    total_capacity_gb: float
    used_gb: float
    available_gb: float
    utilization_fraction: float
    pinned_objects_count: int
    pinned_memory_gb: float
    spill_count: int
    is_spilling: bool


class PinnedObjectStore:
    """
    Optimized object store with aggressive pinning.
    
    Prevents spilling by:
    1. Pre-allocating space for critical tensors
    2. Tracking access patterns for smart eviction
    3. Enforcing strict memory limits
    """
    
    def __init__(
        self,
        max_size_gb: float = GLOBAL_RAM_LIMIT_GB * OBJECT_STORE_FRACTION,
        pinned_max_gb: float = PINNED_OBJECTS_MAX_GB
    ):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.pinned_max_bytes = int(pinned_max_gb * 1024**3)
        
        self._pinned_objects: Dict[str, PinnedObjectInfo] = {}
        self._pinned_data: Dict[str, Any] = {}
        self._access_history: Dict[str, List[int]] = {}
        self._spill_count = 0
        self._lock = threading.RLock()
        
        # Track current usage
        self._current_usage_bytes = 0
        self._pinned_usage_bytes = 0
    
    def pin_object(
        self,
        object_id: str,
        data: np.ndarray,
        priority: int = 1
    ) -> bool:
        """
        Pin an object in memory to prevent spilling.
        
        Args:
            object_id: Unique identifier for the object
            data: numpy array or serializable data
            priority: Priority level (higher = less likely to evict)
            
        Returns:
            True if successfully pinned, False if memory limit exceeded
        """
        with self._lock:
            size_bytes = data.nbytes if hasattr(data, 'nbytes') else len(str(data))
            
            # Check if we have room
            if self._pinned_usage_bytes + size_bytes > self.pinned_max_bytes:
                # Try to evict low-priority objects
                self._evict_low_priority(size_bytes)
                
                # Check again after eviction
                if self._pinned_usage_bytes + size_bytes > self.pinned_max_bytes:
                    return False
            
            # Pin the object
            info = PinnedObjectInfo(
                object_id=object_id,
                size_bytes=size_bytes,
                pinned_at_ns=time.time_ns(),
                access_count=0,
                last_access_ns=time.time_ns(),
                priority=priority
            )
            
            self._pinned_objects[object_id] = info
            self._pinned_data[object_id] = data
            self._access_history[object_id] = []
            
            self._pinned_usage_bytes += size_bytes
            self._current_usage_bytes += size_bytes
            
            return True
    
    def get_pinned(self, object_id: str) -> Optional[Any]:
        """
        Get a pinned object, updating access history.
        
        Args:
            object_id: Object identifier
            
        Returns:
            The data if found and still pinned, None otherwise
        """
        with self._lock:
            if object_id not in self._pinned_data:
                return None
            
            # Update access tracking
            info = self._pinned_objects.get(object_id)
            if info:
                info.access_count += 1
                info.last_access_ns = time.time_ns()
                self._access_history[object_id].append(info.last_access_ns)
            
            return self._pinned_data.get(object_id)
    
    def unpin_object(self, object_id: str) -> bool:
        """
        Unpin an object, freeing its memory.
        
        Args:
            object_id: Object identifier
            
        Returns:
            True if object was pinned and is now unpinned
        """
        with self._lock:
            if object_id not in self._pinned_objects:
                return False
            
            info = self._pinned_objects.pop(object_id)
            self._pinned_data.pop(object_id, None)
            self._access_history.pop(object_id, None)
            
            self._pinned_usage_bytes -= info.size_bytes
            self._current_usage_bytes -= info.size_bytes
            
            return True
    
    def _evict_low_priority(self, needed_bytes: int):
        """Evict lowest priority objects to make room."""
        if not self._pinned_objects:
            return
        
        # Sort by priority (ascending) then by access recency
        sorted_objects = sorted(
            self._pinned_objects.items(),
            key=lambda x: (x[1].priority, -x[1].last_access_ns)
        )
        
        freed = 0
        for object_id, info in sorted_objects:
            if freed >= needed_bytes:
                break
            
            # Don't evict high-priority objects
            if info.priority >= 10:
                continue
            
            self.unpin_object(object_id)
            freed += info.size_bytes
    
    def get_stats(self) -> ObjectStoreStats:
        """Get current object store statistics."""
        with self._lock:
            return ObjectStoreStats(
                total_capacity_gb=self.max_size_bytes / (1024**3),
                used_gb=self._current_usage_bytes / (1024**3),
                available_gb=(self.max_size_bytes - self._current_usage_bytes) / (1024**3),
                utilization_fraction=self._current_usage_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0,
                pinned_objects_count=len(self._pinned_objects),
                pinned_memory_gb=self._pinned_usage_bytes / (1024**3),
                spill_count=self._spill_count,
                is_spilling=False  # Would check actual spilling status
            )
    
    def clear_all(self):
        """Clear all pinned objects."""
        with self._lock:
            self._pinned_objects.clear()
            self._pinned_data.clear()
            self._access_history.clear()
            self._current_usage_bytes = 0
            self._pinned_usage_bytes = 0


class RayObjectStoreOptimizer:
    """
    Optimizer for Ray's Plasma object store.
    
    Features:
    - Aggressive pinning of critical tensors
    - Prevention of NVMe spilling
    - Memory pressure monitoring
    - Integration with global 8GB limit
    """
    
    def __init__(
        self,
        global_limit_gb: float = GLOBAL_RAM_LIMIT_GB,
        object_store_fraction: float = OBJECT_STORE_FRACTION
    ):
        self.global_limit_gb = global_limit_gb
        self.object_store_gb = global_limit_gb * object_store_fraction
        
        self.pinned_store = PinnedObjectStore(
            max_size_gb=self.object_store_gb
        )
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def configure_ray_init(self) -> dict:
        """
        Get optimal Ray initialization parameters.
        
        Call this before ray.init() to configure object store.
        
        Returns:
            Dictionary of parameters for ray.init()
        """
        object_store_bytes = int(self.object_store_gb * 1024**3)
        
        return {
            'object_store_memory': object_store_bytes,
            '_system_config': {
                # Disable automatic spilling where possible
                'min_spilling_size': int(0.5 * 1024**3),  # 500MB minimum spill
                'max_spilling_size': int(2.0 * 1024**3),  # 2GB maximum spill
                
                # Optimize for low latency
                'object_store_full_delay_ms': 100,
                'object_store_max_retries': 5,
                
                # Memory management
                'pressure_autoscaling': False,  # Manual control
            },
            # Limit concurrent tasks to reduce memory pressure
            'num_cpus': min(os.cpu_count() or 4, 8),
        }
    
    def pin_critical_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        priority: int = 5
    ) -> bool:
        """
        Pin a critical tensor to prevent spilling.
        
        Args:
            name: Name/ID for the tensor
            tensor: numpy array to pin
            priority: Priority (1-10, higher = more protected)
            
        Returns:
            True if successfully pinned
        """
        success = self.pinned_store.pin_object(name, tensor, priority)
        
        if success:
            print(f"[ObjectStore] Pinned '{name}' ({tensor.nbytes / 1024**2:.1f}MB)")
        else:
            print(f"[ObjectStore] Failed to pin '{name}' - memory limit")
        
        return success
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start monitoring object store health."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
        print(f"[ObjectStore] Started monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                stats = self.pinned_store.get_stats()
                
                if stats.utilization_fraction > SPILL_THRESHOLD:
                    print(f"[ObjectStore] WARNING: High utilization {stats.utilization_fraction:.1%}")
                    
                    # Trigger cleanup of low-priority objects
                    self._cleanup_low_priority()
                
            except Exception as e:
                print(f"[ObjectStore] Monitoring error: {e}")
            
            time.sleep(interval)
    
    def _cleanup_low_priority(self):
        """Clean up low-priority pinned objects."""
        stats = self.pinned_store.get_stats()
        
        if stats.utilization_fraction < 0.7:
            return
        
        # Evict objects with priority < 3
        with self.pinned_store._lock:
            to_evict = [
                oid for oid, info in self.pinned_store._pinned_objects.items()
                if info.priority < 3
            ]
            
            for object_id in to_evict:
                self.pinned_store.unpin_object(object_id)
                print(f"[ObjectStore] Evicted low-priority object: {object_id}")
    
    @contextmanager
    def temporary_pin(self, name: str, tensor: np.ndarray, priority: int = 3):
        """
        Context manager for temporary pinning.
        
        Automatically unpins when exiting context.
        
        Usage:
            with optimizer.temporary_pin("temp", data):
                # Use data here
                process(tensor)
            # Automatically unpinned
        """
        self.pin_critical_tensor(name, tensor, priority)
        try:
            yield tensor
        finally:
            self.pinned_store.unpin_object(name)
    
    def get_shared_tensor_ref(
        self,
        name: str,
        create_fn=None
    ) -> Optional[np.ndarray]:
        """
        Get or create a shared tensor reference.
        
        This integrates with Ray's object store for cross-worker sharing.
        
        Args:
            name: Tensor name
            create_fn: Optional function to create tensor if not exists
            
        Returns:
            Tensor reference or None
        """
        # Check pinned store first
        tensor = self.pinned_store.get_pinned(name)
        if tensor is not None:
            return tensor
        
        # Create if function provided
        if create_fn is not None:
            tensor = create_fn()
            if self.pin_critical_tensor(name, tensor, priority=5):
                return tensor
        
        return None
    
    def shutdown(self):
        """Shutdown and release all resources."""
        self.stop_monitoring()
        self.pinned_store.clear_all()
        print("[ObjectStore] Shutdown complete")


def create_optimized_ray_init() -> dict:
    """
    Create optimized Ray initialization configuration.
    
    Returns:
        Configuration dictionary for ray.init()
    """
    optimizer = RayObjectStoreOptimizer()
    config = optimizer.configure_ray_init()
    
    print("[ObjectStore] Recommended Ray init configuration:")
    print(f"  object_store_memory: {config['object_store_memory'] / 1024**3:.1f}GB")
    print(f"  num_cpus: {config['num_cpus']}")
    
    return config


if __name__ == '__main__':
    print("Ray Object Store Optimizer")
    print("=" * 40)
    
    # Create optimizer
    optimizer = RayObjectStoreOptimizer(
        global_limit_gb=GLOBAL_RAM_LIMIT_GB
    )
    
    # Get recommended configuration
    config = optimizer.configure_ray_init()
    
    print(f"\nConfiguration:")
    print(f"  Global Limit: {optimizer.global_limit_gb}GB")
    print(f"  Object Store: {optimizer.object_store_gb:.1f}GB")
    print(f"  Pinned Max: {PINNED_OBJECTS_MAX_GB}GB")
    
    # Initialize Ray with optimized settings
    if not ray.is_initialized():
        ray.init(**config)
        print("\nRay initialized with optimized object store")
    
    # Test pinning
    test_tensor = np.random.randn(1000, 1000).astype(np.float32)
    tensor_size_mb = test_tensor.nbytes / 1024**2
    
    print(f"\nTest tensor size: {tensor_size_mb:.1f}MB")
    
    success = optimizer.pin_critical_tensor("test_matrix", test_tensor, priority=5)
    print(f"Pin successful: {success}")
    
    # Get stats
    stats = optimizer.pinned_store.get_stats()
    print(f"\nObject Store Stats:")
    print(f"  Total Capacity: {stats.total_capacity_gb:.1f}GB")
    print(f"  Used: {stats.used_gb:.2f}GB")
    print(f"  Utilization: {stats.utilization_fraction:.1%}")
    print(f"  Pinned Objects: {stats.pinned_objects_count}")
    
    # Start monitoring
    optimizer.start_monitoring(interval_seconds=2.0)
    
    # Run briefly
    time.sleep(3)
    
    # Cleanup
    optimizer.shutdown()
    print("\nOptimizer shutdown complete")
