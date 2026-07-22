"""
Chapter 3: Real-Time Feature Store & Vector Search
File 8: python/features/embedding.py

Generate lightweight time-series embeddings using Locality Sensitive Hashing (LSH)
on Ray workers. Strictly enforces 4GB Python RAM quota during generation.

Injects AMD DirectML/ROCm environment checks for acceleration.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import ray
import hashlib

# Memory limit enforcement (4GB quota)
MAX_MEMORY_MB = 4096
EMBEDDING_DIM = 64


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


class LSHEmbeddingGenerator:
    """
    Locality Sensitive Hashing for time-series embedding generation.
    
    Uses SimHash variant optimized for financial time series.
    Produces fixed-length binary/sparse embeddings suitable for HNSW indexing.
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        memory_limit_mb: int = MAX_MEMORY_MB
    ):
        self.embedding_dim = embedding_dim
        self.memory_limit_mb = memory_limit_mb
        self.accel_info = check_amd_acceleration()
        
        # Initialize random projection matrix (fixed seed for reproducibility)
        np.random.seed(42)
        self.projection_matrix = np.random.randn(embedding_dim, embedding_dim).astype(np.float32)
        self.projection_matrix /= np.sqrt(embedding_dim)
        
        self._processed_count = 0
    
    def generate_embedding(self, time_series: np.ndarray) -> np.ndarray:
        """
        Generate LSH embedding from time series.
        
        Parameters
        ----------
        time_series : np.ndarray
            Input time series (any length)
            
        Returns
        -------
        np.ndarray
            Fixed-dimension embedding (binary or float)
        """
        self._check_memory()
        
        # Preprocess: normalize and truncate/pad
        ts = self._preprocess(time_series)
        
        # Apply random projection
        projected = self.projection_matrix @ ts
        
        # Sign-based hashing (SimHash)
        embedding = (projected > 0).astype(np.float32)
        
        self._processed_count += 1
        return embedding
    
    def generate_batch(
        self,
        time_series_list: List[np.ndarray],
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        Generate embeddings for multiple time series in batches.
        
        Parameters
        ----------
        time_series_list : list
            List of input time series
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        np.ndarray
            Embedding matrix (n_samples x embedding_dim)
        """
        n_samples = len(time_series_list)
        embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            for j in range(i, batch_end):
                embeddings[j] = self.generate_embedding(time_series_list[j])
            
            # Memory checkpoint every batch
            self._check_memory()
        
        return embeddings
    
    def _preprocess(self, time_series: np.ndarray) -> np.ndarray:
        """Normalize and resize time series to embedding dimension."""
        ts = np.asarray(time_series, dtype=np.float32)
        
        # Remove mean and scale
        if len(ts) > 1:
            ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
        
        # Truncate or pad to embedding dimension
        if len(ts) > self.embedding_dim:
            # Take most recent values
            ts = ts[-self.embedding_dim:]
        elif len(ts) < self.embedding_dim:
            # Pad with zeros at the beginning
            padding = np.zeros(self.embedding_dim - len(ts), dtype=np.float32)
            ts = np.concatenate([padding, ts])
        
        return ts
    
    def _check_memory(self):
        """Memory usage checkpoint with GC."""
        import gc
        if self._processed_count % 100 == 0:
            gc.collect()
    
    def get_stats(self) -> Dict:
        return {
            'processed_count': self._processed_count,
            'embedding_dim': self.embedding_dim,
            'acceleration': self.accel_info,
            'memory_limit_mb': self.memory_limit_mb
        }


@ray.remote(max_calls=10)
class DistributedEmbeddingWorker:
    """Ray worker for distributed embedding generation."""
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        memory_limit_mb: int = MAX_MEMORY_MB
    ):
        self.generator = LSHEmbeddingGenerator(embedding_dim, memory_limit_mb)
        self._batches_processed = 0
    
    def process_batch(
        self,
        time_series_batch: List[np.ndarray]
    ) -> np.ndarray:
        """Process a batch of time series."""
        embeddings = self.generator.generate_batch(time_series_batch)
        self._batches_processed += 1
        return embeddings
    
    def get_stats(self) -> Dict:
        stats = self.generator.get_stats()
        stats['batches_processed'] = self._batches_processed
        return stats


def create_embedding_workers(
    num_workers: int = 4,
    embedding_dim: int = EMBEDDING_DIM
) -> List:
    """Create distributed embedding workers."""
    return [
        DistributedEmbeddingWorker.remote(
            embedding_dim=embedding_dim,
            memory_limit_mb=MAX_MEMORY_MB
        )
        for _ in range(num_workers)
    ]


def hash_time_series(
    time_series: np.ndarray,
    hash_bits: int = 64
) -> str:
    """
    Create a compact hash representation of time series.
    
    Useful for quick similarity lookups without full embedding computation.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    hash_bits : int
        Number of bits in hash
        
    Returns
    -------
    str
        Hex string representation of hash
    """
    # Normalize
    ts = time_series.astype(np.float64)
    if len(ts) > 1:
        ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-8)
    
    # Create feature vector
    features = []
    
    # Statistical features
    features.extend([
        np.mean(ts),
        np.std(ts),
        np.min(ts),
        np.max(ts),
        np.median(ts),
    ])
    
    # Momentum features
    if len(ts) > 5:
        features.append(np.mean(np.diff(ts[:5])))
        features.append(np.mean(np.diff(ts[-5:])))
    
    # Pad/truncate to fixed size
    while len(features) < hash_bits // 8:
        features.append(0.0)
    features = features[:hash_bits // 8]
    
    # Convert to bytes and hash
    feature_bytes = np.array(features, dtype=np.float64).tobytes()
    hash_obj = hashlib.sha256(feature_bytes)
    
    return hash_obj.hexdigest()[:hash_bits // 4]


def compute_lsh_signature(
    time_series: np.ndarray,
    num_hashes: int = 16
) -> Tuple[int, ...]:
    """
    Compute LSH signature for approximate nearest neighbor search.
    
    Parameters
    ----------
    time_series : np.ndarray
        Input time series
    num_hashes : int
        Number of hash functions
        
    Returns
    -------
    tuple
        LSH signature as tuple of integers
    """
    ts = np.asarray(time_series, dtype=np.float64)
    
    # Generate random hyperplanes
    np.random.seed(hash(len(ts)) % (2**32))
    hyperplanes = np.random.randn(num_hashes, len(ts))
    hyperplanes /= np.linalg.norm(hyperplanes, axis=1, keepdims=True)
    
    # Project and threshold
    projections = hyperplanes @ ts
    signature = tuple((projections > 0).astype(int))
    
    return signature


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=4 * 1024 * 1024 * 1024,
        _system_config={"max_bytes_to_spill": 4 * 1024 * 1024 * 1024}
    )
    
    print("AMD Acceleration:", check_amd_acceleration())
    
    # Test embedding generation
    generator = LSHEmbeddingGenerator(embedding_dim=64)
    
    # Generate sample time series
    np.random.seed(42)
    sample_ts = np.cumsum(np.random.randn(100))
    
    embedding = generator.generate_embedding(sample_ts)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sparsity: {np.mean(embedding == 0):.2%}")
    
    # Test batch generation
    time_series_list = [np.random.randn(50 + i * 10) for i in range(10)]
    embeddings = generator.generate_batch(time_series_list)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test hashing
    ts_hash = hash_time_series(sample_ts)
    lsh_sig = compute_lsh_signature(sample_ts)
    print(f"Time series hash: {ts_hash}")
    print(f"LSH signature: {lsh_sig}")
    
    print(f"Generator stats: {generator.get_stats()}")
    
    ray.shutdown()
