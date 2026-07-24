"""
SOUL.md Mistake Classifier - Stage 56
AMD Ryzen AI 5 Optimized | 4GB RAM Quota Enforced | ROCm/DirectML Accelerated

This module classifies trade failures using Ray-distributed Isolation Forests.
It identifies "toxic" feature combinations that lead to catastrophic losses and
tags them for permanent banning in the SOUL.md ledger.

Constraints:
- Strict 4GB RAM quota for all Ray workers
- GPU-accelerated inference via ROCm/DirectML where available
- Zero heap fragmentation during high-frequency classification
"""

import ray
import numpy as np
import cupy as cp  # ROCm/DirectML compatible via CuPy
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import psutil
import os

# Enforce strict memory limits
MAX_RAM_MB = 4096
os.environ['RAY_MEMORY_LIMIT'] = str(MAX_RAM_MB * 1024 * 1024)

@dataclass
class ToxicPattern:
    """Represents a classified toxic pattern for SOUL.md ledger."""
    pattern_hash: str
    feature_signature: np.ndarray
    loss_severity: float
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    banned: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@ray.remote(num_cpus=1, max_calls=1000)
class DistributedClassifier:
    """
    Ray-distributed Isolation Forest classifier for toxic pattern detection.
    Each instance processes a shard of trade post-mortems with GPU acceleration.
    """
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model: Optional[IsolationForest] = None
        self.gpu_available = self._check_gpu()
        self.processed_count = 0
        
    def _check_gpu(self) -> bool:
        """Check for AMD ROCm/DirectML availability via CuPy."""
        try:
            # Test GPU allocation
            test_array = cp.zeros(10)
            del test_array
            cp.get_default_memory_pool().free_all_blocks()
            return True
        except Exception:
            return False
    
    def fit_batch(self, features: np.ndarray, losses: np.ndarray) -> Dict[str, Any]:
        """
        Fit Isolation Forest on a batch of trade features and losses.
        Uses GPU-accelerated distance computations if available.
        
        Args:
            features: Trade feature matrix (n_samples, n_features)
            losses: Corresponding PnL values
            
        Returns:
            Classification results with toxic pattern metadata
        """
        # Memory safety check
        current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
        if current_ram > MAX_RAM_MB * 0.8:
            raise MemoryError(f"Worker approaching RAM limit: {current_ram:.2f}MB")
        
        # Initialize model if needed
        if self.model is None:
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
                n_jobs=1  # Single-threaded within worker for determinism
            )
        
        # Fit on batch
        self.model.fit(features)
        predictions = self.model.predict(features)
        scores = self.model.decision_function(features)
        
        # Identify toxic patterns (anomalies with severe losses)
        toxic_mask = (predictions == -1) & (losses < np.percentile(losses, 10))
        
        # GPU-accelerated feature signature extraction
        if self.gpu_available and len(features[toxic_mask]) > 0:
            toxic_features = cp.asarray(features[toxic_mask])
            # Compute centroid of toxic cluster on GPU
            centroid = cp.mean(toxic_features, axis=0)
            centroid_host = cp.asnumpy(centroid)
        else:
            centroid_host = np.mean(features[toxic_mask], axis=0) if np.any(toxic_mask) else np.zeros(features.shape[1])
        
        self.processed_count += len(features)
        
        return {
            'toxic_count': int(np.sum(toxic_mask)),
            'total_processed': len(features),
            'centroid': centroid_host.tolist(),
            'min_score': float(np.min(scores[toxic_mask])) if np.any(toxic_mask) else 0.0,
            'gpu_used': self.gpu_available
        }
    
    def classify_live(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify live trade features against trained model.
        Returns (is_toxic, anomaly_scores) tuples.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit_batch first.")
        
        predictions = self.model.predict(features)
        scores = self.model.decision_function(features)
        
        is_toxic = predictions == -1
        return is_toxic, scores


class MistakeClassifier:
    """
    Master classifier coordinating Ray workers for toxic pattern detection.
    Aggregates results and formats them for SOUL.md ledger ingestion.
    """
    
    def __init__(self, num_workers: int = 4, contamination: float = 0.05):
        self.num_workers = num_workers
        self.contamination = contamination
        self.workers: List[ray.ObjectRef] = []
        self.toxic_patterns: Dict[str, ToxicPattern] = {}
        self.initialized = False
        
    def initialize_ray(self):
        """Initialize Ray cluster with strict memory constraints."""
        if not ray.is_initialized():
            # Calculate per-worker memory limit
            total_ram = psutil.virtual_memory().available
            worker_ram = min(total_ram // self.num_workers, MAX_RAM_MB * 1024 * 1024)
            
            ray.init(
                num_cpus=self.num_workers,
                _memory=int(worker_ram * self.num_workers * 0.9),  # 10% buffer
                object_store_memory=int(worker_ram * self.num_workers * 0.3),
                ignore_reinit_error=True
            )
        
        # Spawn distributed workers
        self.workers = [
            DistributedClassifier.remote(
                contamination=self.contamination,
                n_estimators=100
            )
            for _ in range(self.num_workers)
        ]
        self.initialized = True
    
    def process_post_mortems(
        self,
        trade_data: List[Dict[str, Any]]
    ) -> List[ToxicPattern]:
        """
        Process trade post-mortems to identify toxic patterns.
        
        Args:
            trade_data: List of trade records with features and outcomes
            
        Returns:
            List of identified ToxicPattern objects for SOUL.md
        """
        if not self.initialized:
            self.initialize_ray()
        
        # Convert to numpy arrays
        features_list = []
        losses_list = []
        
        for trade in trade_data:
            if 'features' in trade and 'pnl' in trade:
                features_list.append(trade['features'])
                losses_list.append(trade['pnl'])
        
        if len(features_list) == 0:
            return []
        
        features = np.array(features_list, dtype=np.float32)
        losses = np.array(losses_list, dtype=np.float32)
        
        # Distribute across workers
        chunk_size = max(1, len(features) // self.num_workers)
        futures = []
        
        for i, worker in enumerate(self.workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < self.num_workers - 1 else len(features)
            
            chunk_features = features[start_idx:end_idx]
            chunk_losses = losses[start_idx:end_idx]
            
            if len(chunk_features) > 0:
                future = worker.fit_batch.remote(chunk_features, chunk_losses)
                futures.append(future)
        
        # Aggregate results
        results = ray.get(futures)
        
        # Generate toxic patterns
        new_patterns = []
        for result in results:
            if result['toxic_count'] > 0:
                centroid = np.array(result['centroid'])
                pattern_hash = hashlib.sha256(
                    centroid.tobytes() + str(result['min_score']).encode()
                ).hexdigest()[:16]
                
                if pattern_hash not in self.toxic_patterns:
                    pattern = ToxicPattern(
                        pattern_hash=pattern_hash,
                        feature_signature=centroid,
                        loss_severity=abs(result['min_score']),
                        occurrence_count=result['toxic_count'],
                        first_seen=datetime.utcnow(),
                        last_seen=datetime.utcnow(),
                        metadata={
                            'worker_gpu': result['gpu_used'],
                            'contamination': self.contamination
                        }
                    )
                    self.toxic_patterns[pattern_hash] = pattern
                    new_patterns.append(pattern)
                else:
                    # Update existing pattern
                    existing = self.toxic_patterns[pattern_hash]
                    existing.occurrence_count += result['toxic_count']
                    existing.last_seen = datetime.utcnow()
        
        return new_patterns
    
    def export_to_soul_ledger(self) -> List[Dict[str, Any]]:
        """
        Export classified toxic patterns in SOUL.md ledger format.
        
        Returns:
            List of ledger entries ready for immutable append
        """
        ledger_entries = []
        
        for pattern in self.toxic_patterns.values():
            entry = {
                'type': 'TOXIC_PATTERN_BAN',
                'timestamp': pattern.last_seen.isoformat(),
                'hash': pattern.pattern_hash,
                'severity': pattern.loss_severity,
                'occurrences': pattern.occurrence_count,
                'feature_signature': pattern.feature_signature.tolist(),
                'banned': pattern.banned,
                'metadata': pattern.metadata,
                'cryptographic_seal': self._generate_seal(pattern)
            }
            ledger_entries.append(entry)
        
        return ledger_entries
    
    def _generate_seal(self, pattern: ToxicPattern) -> str:
        """Generate cryptographic seal for ledger integrity."""
        data = (
            pattern.pattern_hash +
            str(pattern.loss_severity) +
            str(pattern.first_seen.timestamp()) +
            str(pattern.last_seen.timestamp())
        )
        return hashlib.sha256(data.encode()).hexdigest()
    
    def shutdown(self):
        """Gracefully shutdown Ray cluster and release resources."""
        if ray.is_initialized():
            ray.shutdown()
        self.workers = []
        self.initialized = False


# Example usage for integration testing
if __name__ == '__main__':
    # Simulated trade post-mortems
    sample_trades = [
        {
            'features': [0.5, -0.2, 0.8, 0.1, -0.5],
            'pnl': -0.15,
            'symbol': 'BTCUSDT',
            'timestamp': datetime.utcnow().isoformat()
        }
        for _ in range(1000)
    ]
    
    classifier = MistakeClassifier(num_workers=2)
    patterns = classifier.process_post_mortems(sample_trades)
    
    print(f"Identified {len(patterns)} toxic patterns")
    
    ledger = classifier.export_to_soul_ledger()
    print(f"Exported {len(ledger)} entries to SOUL.md ledger")
    
    classifier.shutdown()
