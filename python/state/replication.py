"""
In-Memory State Replication across Python Workers.

Implements in-memory state replication across Python workers to prevent
single-point-of-failure, strictly managing object sizes to respect the
4GB Python RAM quota. Includes AMD DirectML/ROCm environment checks
for tensor offload capabilities.

Optimized for Ray distributed computing on AMD Ryzen AI 5 architecture.
"""

import ray
import os
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TOTAL_MEMORY_BYTES = int(os.getenv("PYTHON_RAM_QUOTA_GB", "4")) * 1024 * 1024 * 1024
MAX_OBJECT_SIZE_BYTES = int(os.getenv("MAX_OBJECT_SIZE_MB", "100")) * 1024 * 1024
REPLICATION_FACTOR = int(os.getenv("REPLICATION_FACTOR", "3"))


def check_rocm_directml_support() -> Dict[str, bool]:
    """
    Check for AMD ROCm and DirectML support for tensor offload.
    
    Returns dictionary with capability flags.
    """
    capabilities = {
        'rocm_available': False,
        'directml_available': False,
        'gpu_offload_capable': False,
        'preferred_backend': 'cpu'
    }
    
    # Check ROCm (AMD GPUs on Linux)
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            # Check if it's actually ROCm (hip)
            if os.environ.get('ROCM_PATH') or os.path.exists('/opt/rocm'):
                capabilities['rocm_available'] = True
                capabilities['gpu_offload_capable'] = True
                capabilities['preferred_backend'] = 'rocm'
                logger.info("AMD ROCm detected for GPU tensor offload")
    except ImportError:
        pass
    
    # Check DirectML (Windows AMD GPU support)
    try:
        if os.name == 'nt':  # Windows
            import subprocess
            result = subprocess.run(
                ['dxdiag', '/t'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if 'AMD' in result.stdout.upper() or 'RADEON' in result.stdout.upper():
                capabilities['directml_available'] = True
                capabilities['gpu_offload_capable'] = True
                if not capabilities['rocm_available']:
                    capabilities['preferred_backend'] = 'directml'
                logger.info("AMD DirectML detected for GPU tensor offload")
    except Exception:
        pass
    
    return capabilities


@dataclass
class ReplicatedObject:
    """Wrapper for replicated objects with metadata."""
    key: str
    data: bytes
    checksum: str
    size_bytes: int
    version: int
    created_at: float
    replicas: Set[int] = field(default_factory=set)
    
    def verify_integrity(self) -> bool:
        """Verify object integrity via checksum."""
        actual_checksum = hashlib.sha256(self.data).hexdigest()
        return actual_checksum == self.checksum


@ray.remote
class ReplicationWorker:
    """Ray worker for state replication."""
    
    def __init__(self, worker_id: int, total_workers: int):
        self.worker_id = worker_id
        self.total_workers = total_workers
        self.local_store: Dict[str, ReplicatedObject] = {}
        self.memory_used = 0
        self.gpu_capabilities = check_rocm_directml_support()
        
    def put(self, key: str, data: Any, version: int = 1) -> bool:
        """Store object locally with replication metadata."""
        # Serialize data
        import pickle
        serialized = pickle.dumps(data)
        
        # Check size limit
        if len(serialized) > MAX_OBJECT_SIZE_BYTES:
            logger.warning(f"Object {key} exceeds max size limit")
            return False
        
        # Check memory quota
        if self.memory_used + len(serialized) > MAX_TOTAL_MEMORY_BYTES // self.total_workers:
            # Evict oldest entries if needed
            self._evict_if_needed(len(serialized))
        
        # Create replicated object
        obj = ReplicatedObject(
            key=key,
            data=serialized,
            checksum=hashlib.sha256(serialized).hexdigest(),
            size_bytes=len(serialized),
            version=version,
            created_at=datetime.now().timestamp(),
            replicas={self.worker_id}
        )
        
        self.local_store[key] = obj
        self.memory_used += len(serialized)
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve object from local store."""
        obj = self.local_store.get(key)
        if obj is None:
            return None
        
        # Verify integrity
        if not obj.verify_integrity():
            logger.error(f"Integrity check failed for {key}")
            return None
        
        # Deserialize
        import pickle
        return pickle.loads(obj.data)
    
    def replicate_to(self, key: str, target_worker: 'ReplicationWorker') -> bool:
        """Replicate object to another worker."""
        obj = self.local_store.get(key)
        if obj is None:
            return False
        
        # Add target to replica set
        obj.replicas.add(target_worker.worker_id)
        
        # Send data to target
        success = target_worker.receive_replication.remote(obj)
        return True
    
    @ray.remote
    def receive_replication(self, obj: ReplicatedObject) -> bool:
        """Receive replicated object from another worker."""
        if obj.key in self.local_store:
            existing = self.local_store[obj.key]
            if existing.version >= obj.version:
                return False  # Already have newer version
        
        self.local_store[obj.key] = obj
        self.memory_used += obj.size_bytes
        
        return True
    
    def _evict_if_needed(self, needed_bytes: int):
        """Evict oldest entries if memory pressure detected."""
        while (self.memory_used + needed_bytes > 
               MAX_TOTAL_MEMORY_BYTES // self.total_workers):
            if not self.local_store:
                break
            
            # Find oldest entry
            oldest_key = min(
                self.local_store.keys(),
                key=lambda k: self.local_store[k].created_at
            )
            
            evicted = self.local_store.pop(oldest_key)
            self.memory_used -= evicted.size_bytes
            logger.info(f"Evicted {oldest_key} to free memory")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'objects_count': len(self.local_store),
            'memory_used_bytes': self.memory_used,
            'memory_quota_bytes': MAX_TOTAL_MEMORY_BYTES // self.total_workers,
            'gpu_capabilities': self.gpu_capabilities
        }
    
    def list_keys(self) -> List[str]:
        """List all keys in local store."""
        return list(self.local_store.keys())


@ray.remote
class ReplicationCoordinator:
    """Coordinates state replication across workers."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = [
            ReplicationWorker.remote(i, num_workers) 
            for i in range(num_workers)
        ]
        self.key_to_workers: Dict[str, Set[int]] = {}
        
    def put(self, key: str, data: Any, version: int = 1) -> bool:
        """Store data with replication."""
        # Primary storage (hash-based assignment)
        primary_idx = hash(key) % self.num_workers
        
        # Store on primary
        success = ray.get(
            self.workers[primary_idx].put.remote(key, data, version)
        )
        
        if not success:
            return False
        
        # Track placement
        self.key_to_workers[key] = {primary_idx}
        
        # Replicate to REPLICATION_FACTOR - 1 additional workers
        replicas_created = 1
        for i in range(self.num_workers):
            if i == primary_idx:
                continue
            if replicas_created >= REPLICATION_FACTOR:
                break
            
            # Async replication
            self.workers[primary_idx].replicate_to.remote(
                key, self.workers[i]
            )
            self.key_to_workers[key].add(i)
            replicas_created += 1
        
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from any available replica."""
        if key not in self.key_to_workers:
            return None
        
        # Try each replica until successful
        for worker_idx in self.key_to_workers[key]:
            result = ray.get(self.workers[worker_idx].get.remote(key))
            if result is not None:
                return result
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete key from all replicas."""
        if key not in self.key_to_workers:
            return False
        
        for worker_idx in self.key_to_workers[key]:
            ray.get(self.workers[worker_idx].delete.remote(key))
        
        del self.key_to_workers[key]
        return True
    
    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics from all workers."""
        return ray.get([w.get_stats.remote() for w in self.workers])


class DistributedStateReplicator:
    """High-level interface for distributed state replication."""
    
    def __init__(self, num_workers: int = 4):
        if not ray.is_initialized():
            ray.init(
                include_dashboard=False,
                _system_config={
                    "object_store_memory": MAX_TOTAL_MEMORY_BYTES
                }
            )
        
        self.coordinator = ReplicationCoordinator.remote(num_workers)
        self.gpu_capabilities = check_rocm_directml_support()
        logger.info(f"GPU capabilities: {self.gpu_capabilities}")
        
    def put(self, key: str, data: Any, version: int = 1) -> bool:
        """Store data with replication."""
        return ray.get(self.coordinator.put.remote(key, data, version))
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data."""
        return ray.get(self.coordinator.get.remote(key))
    
    def delete(self, key: str) -> bool:
        """Delete data."""
        return ray.get(self.coordinator.delete.remote(key))
    
    def get_stats(self) -> List[Dict[str, Any]]:
        """Get system statistics."""
        return ray.get(self.coordinator.get_all_stats.remote())
    
    def shutdown(self):
        """Shutdown replicator."""
        ray.shutdown()


# Convenience functions for /START and /KILL orchestration
def initialize_replicator(num_workers: int = 4) -> DistributedStateReplicator:
    """Initialize state replicator (call during /START)."""
    return DistributedStateReplicator(num_workers=num_workers)


def cleanup_replicator():
    """Cleanup replicator resources (call during /KILL)."""
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    # Example usage
    print("Testing Distributed State Replication...")
    
    # Initialize
    replicator = initialize_replicator(num_workers=3)
    
    # Test data
    test_data = {
        'weights': np.random.randn(100, 50).astype(np.float32),
        'metadata': {'iteration': 1000, 'reward': 123.45}
    }
    
    # Store with replication
    success = replicator.put("test_key", test_data, version=1)
    print(f"Put success: {success}")
    
    # Retrieve
    retrieved = replicator.get("test_key")
    if retrieved:
        print(f"Retrieved data with {len(retrieved)} keys")
        assert np.allclose(test_data['weights'], retrieved['weights'])
        print("Data integrity verified!")
    
    # Get stats
    stats = replicator.get_stats()
    print(f"System stats: {stats}")
    
    # Cleanup
    replicator.shutdown()
    print("Test complete!")
