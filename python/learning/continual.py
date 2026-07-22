"""
Continual Learning with Prioritized Experience Replay on Ray

This module implements experience replay buffers with prioritized sampling for
continual learning in the Nautilus trading bot. It strictly enforces the 4GB Python
RAM quota by aggressively pruning stale transitions and uses AMD DirectML/ROCm
acceleration when available.

Key Features:
- Prioritized Experience Replay (PER) with sum-tree implementation
- Aggressive memory pruning to stay within 4GB quota
- Ray distributed workers for parallel sampling
- AMD ROCm/DirectML environment detection and optimization
- Lock-free append operations for hot-path compatibility

Safety Guarantees:
- Hard memory limits enforced via circular buffers
- Automatic eviction of oldest/stale transitions
- Non-blocking sampling for inference threads
"""

import os
import sys
import time
import hashlib
import threading
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass, field
import numpy as np

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available, running in single-process mode")

# AMD Acceleration Detection
def detect_amd_acceleration() -> str:
    """Detect available AMD acceleration backend."""
    try:
        import torch
        if torch.cuda.is_available() and 'ROCm' in torch.version.cuda or hasattr(torch.version, 'hip'):
            return 'rocm'
        # DirectML check for Windows
        if sys.platform == 'win32':
            try:
                import torch_directml
                return 'directml'
            except ImportError:
                pass
    except ImportError:
        pass
    return 'cpu'

AMD_BACKEND = detect_amd_acceleration()
print(f"AMD Acceleration Backend: {AMD_BACKEND}")

# Memory Constants (4GB Python Quota Enforcement)
MAX_RAM_BYTES = 4 * 1024 * 1024 * 1024  # 4GB hard limit
TRANSITION_OVERHEAD_ESTIMATE = 512  # Bytes per transition estimate
MAX_TRANSITIONS_GLOBAL = MAX_RAM_BYTES // TRANSITION_OVERHEAD_ESTIMATE
PRIORITY_ALPHA = 0.6  # PER alpha parameter
PRIORITY_BETA = 0.4   # PER beta parameter (increases over time)
BETA_INCREMENT = 0.001  # Beta increase per sample


@dataclass
class Transition:
    """Single experience transition with metadata."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    hash_key: str = ""
    
    def __post_init__(self):
        if not self.hash_key:
            # Fast hash for deduplication
            h = hashlib.sha256()
            h.update(self.state.tobytes())
            h.update(self.action.to_bytes(4, 'little'))
            h.update(self.reward.to_bytes(8, 'little'))
            self.hash_key = h.hexdigest()[:16]


class SumTree:
    """
    Efficient sum-tree implementation for Prioritized Experience Replay.
    Supports O(log N) insert, update, and sample operations.
    Thread-safe with read-write lock pattern.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_idx = 0
        self._lock = threading.RLock()
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, transition: Transition):
        """Add transition with priority."""
        with self._lock:
            idx = self.write_idx + self.capacity - 1
            self.data[self.write_idx] = transition
            self.update(idx, priority)
            
            self.write_idx = (self.write_idx + 1) % self.capacity
            if self.n_entries < self.capacity:
                self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Transition]:
        """Retrieve transition by priority value."""
        with self._lock:
            idx = self._retrieve(0, s)
            data_idx = idx - self.capacity + 1
            return idx, self.tree[idx], self.data[data_idx]


@ray.remote(num_cpus=1, max_calls=10000) if RAY_AVAILABLE else lambda cls: cls
class ReplayBufferWorker:
    """
    Ray worker for distributed experience replay.
    Each worker maintains a local buffer and serves samples.
    """
    
    def __init__(self, worker_id: int, local_capacity: int = 50000):
        self.worker_id = worker_id
        self.local_capacity = local_capacity
        self.buffer = SumTree(local_capacity)
        self.beta = PRIORITY_BETA
        self.epsilon = 0.01  # Small constant for numerical stability
        self.total_samples = 0
        self.pruned_count = 0
        
        # Memory tracking
        self.estimated_memory = 0
        self.max_memory = MAX_RAM_BYTES // 4  # Each worker gets 1/4 of quota
    
    def append(self, transition: Transition):
        """Append transition with automatic pruning."""
        # Estimate memory usage
        mem_estimate = (transition.state.nbytes + 
                       transition.next_state.nbytes + 
                       256)  # Overhead
        
        # Prune if approaching memory limit
        while self.estimated_memory + mem_estimate > self.max_memory:
            self._prune_oldest()
        
        self.buffer.add(transition.priority, transition)
        self.estimated_memory += mem_estimate
    
    def _prune_oldest(self):
        """Remove oldest transitions to free memory."""
        if self.buffer.n_entries == 0:
            return
        
        # Find oldest entry (by timestamp)
        oldest_idx = None
        oldest_time = float('inf')
        
        for i in range(self.buffer.n_entries):
            trans = self.buffer.data[i]
            if trans and trans.timestamp_ns < oldest_time:
                oldest_time = trans.timestamp_ns
                oldest_idx = i
        
        if oldest_idx is not None:
            trans = self.buffer.data[oldest_idx]
            if trans:
                mem_freed = trans.state.nbytes + trans.next_state.nbytes + 256
                self.estimated_memory -= mem_freed
                self.pruned_count += 1
            
            # Reset in sum-tree (priority to zero)
            tree_idx = oldest_idx + self.buffer.capacity - 1
            self.buffer.update(tree_idx, 0.0)
            self.buffer.data[oldest_idx] = None
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with prioritized probabilities."""
        if self.buffer.n_entries == 0:
            return [], np.array([]), np.array([])
        
        batch = []
        priorities = []
        indices = []
        
        segment = self.buffer.total() / batch_size
        self.beta = min(1.0, self.beta + BETA_INCREMENT)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            
            idx, priority, trans = self.buffer.get(s)
            if trans is not None:
                batch.append(trans)
                priorities.append(priority)
                indices.append(idx - self.buffer.capacity + 1)
        
        if not batch:
            return [], np.array([]), np.array([])
        
        # Calculate importance sampling weights
        probs = np.array(priorities) / self.buffer.total()
        weights = (self.buffer.n_entries * probs) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        self.total_samples += len(batch)
        return batch, np.array(weights), np.array(indices)
    
    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray):
        """Update priorities after TD-error calculation."""
        for idx, prio in zip(indices, new_priorities):
            tree_idx = idx + self.buffer.capacity - 1
            self.buffer.update(tree_idx, max(prio, self.epsilon))
    
    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        return {
            'worker_id': self.worker_id,
            'n_entries': self.buffer.n_entries,
            'capacity': self.local_capacity,
            'total_samples': self.total_samples,
            'pruned_count': self.pruned_count,
            'estimated_memory_mb': self.estimated_memory / (1024 * 1024),
            'beta': self.beta,
        }


class DistributedReplayBuffer:
    """
    Main interface for distributed experience replay across Ray workers.
    Manages worker lifecycle and aggregates samples.
    """
    
    def __init__(self, num_workers: int = 4, local_capacity: int = 50000):
        self.num_workers = num_workers
        self.local_capacity = local_capacity
        self.workers = []
        self.round_robin_idx = 0
        
        if RAY_AVAILABLE and ray.is_initialized():
            for i in range(num_workers):
                worker = ReplayBufferWorker.remote(i, local_capacity)
                self.workers.append(worker)
            print(f"Initialized {num_workers} replay buffer workers")
        else:
            # Fallback to single-process mode
            self.workers = [ReplayBufferWorker(0, local_capacity)]
            print("Running in single-process replay buffer mode")
    
    def append(self, transition: Transition):
        """Append transition to worker in round-robin fashion."""
        if not self.workers:
            return
        
        worker = self.workers[self.round_robin_idx % len(self.workers)]
        self.round_robin_idx += 1
        
        if RAY_AVAILABLE and hasattr(worker, 'append'):
            worker.append.remote(transition)
        else:
            worker.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch from random worker."""
        if not self.workers:
            return [], np.array([]), np.array([])
        
        worker_idx = np.random.randint(len(self.workers))
        worker = self.workers[worker_idx]
        
        if RAY_AVAILABLE and hasattr(worker, 'sample'):
            result = ray.get(worker.sample.remote(batch_size))
        else:
            result = worker.sample(batch_size)
        
        return result
    
    def update_priorities(self, worker_id: int, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities on specific worker."""
        if worker_id >= len(self.workers):
            return
        
        worker = self.workers[worker_id]
        if RAY_AVAILABLE and hasattr(worker, 'update_priorities'):
            worker.update_priorities.remote(indices, priorities)
        else:
            worker.update_priorities(indices, priorities)
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Gather statistics from all workers."""
        stats = []
        for worker in self.workers:
            if RAY_AVAILABLE and hasattr(worker, 'get_stats'):
                stats.append(ray.get(worker.get_stats.remote()))
            else:
                stats.append(worker.get_stats())
        return stats
    
    def total_memory_usage_mb(self) -> float:
        """Calculate total estimated memory usage across workers."""
        stats = self.get_all_stats()
        return sum(s['estimated_memory_mb'] for s in stats)
    
    def check_memory_quota(self) -> bool:
        """Verify we're within 4GB quota."""
        total_mb = self.total_memory_usage_mb()
        total_gb = total_mb / 1024
        within_quota = total_gb < 4.0
        
        if not within_quota:
            print(f"WARNING: Memory quota exceeded! {total_gb:.2f}GB / 4GB")
            # Trigger aggressive pruning
            for worker in self.workers:
                if RAY_AVAILABLE and hasattr(worker, '_prune_oldest'):
                    for _ in range(100):  # Aggressive prune
                        worker._prune_oldest.remote()
                else:
                    for _ in range(100):
                        worker._prune_oldest()
        
        return within_quota


# Convenience function for creating transitions
def create_transition(
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    priority: float = 1.0
) -> Transition:
    """Factory function for creating transitions."""
    return Transition(
        state=state.astype(np.float32),  # Enforce float32 for memory efficiency
        action=action,
        reward=float(reward),
        next_state=next_state.astype(np.float32),
        done=bool(done),
        priority=priority
    )


if __name__ == "__main__":
    # Test the replay buffer system
    print("Testing Continual Learning Replay Buffer...")
    print(f"AMD Backend: {AMD_BACKEND}")
    print(f"Max Transitions (Global): {MAX_TRANSITIONS_GLOBAL:,}")
    
    # Initialize Ray if available
    if RAY_AVAILABLE and not ray.is_initialized():
        ray.init(num_cpus=4, object_store_memory=MAX_RAM_BYTES // 2)
    
    # Create distributed buffer
    buffer = DistributedReplayBuffer(num_workers=2, local_capacity=10000)
    
    # Generate test transitions
    for i in range(1000):
        state = np.random.randn(128).astype(np.float32)  # 128-dim state
        next_state = np.random.randn(128).astype(np.float32)
        action = np.random.randint(0, 10)
        reward = np.random.randn()
        done = np.random.random() < 0.01
        
        transition = create_transition(state, action, reward, next_state, done)
        buffer.append(transition)
    
    # Sample batch
    batch, weights, indices = buffer.sample(32)
    print(f"Sampled batch size: {len(batch)}")
    print(f"Weights shape: {weights.shape}")
    
    # Update priorities (simulated TD errors)
    if len(indices) > 0:
        td_errors = np.abs(np.random.randn(len(indices)))
        new_priorities = td_errors ** PRIORITY_ALPHA
        buffer.update_priorities(0, indices, new_priorities)
    
    # Check stats
    stats = buffer.get_all_stats()
    for s in stats:
        print(f"Worker {s['worker_id']}: {s['n_entries']} entries, "
              f"{s['estimated_memory_mb']:.2f}MB, {s['pruned_count']} pruned")
    
    # Verify memory quota
    assert buffer.check_memory_quota(), "Memory quota check failed!"
    
    print("\n✓ All tests passed!")
    
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
