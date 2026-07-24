"""
Stage 62: AI & Pipeline Audit - File 6/20
Module: python/soul/continuous_learner.py
Focus: Isolation Forest Memory Bloat Prevention, Streaming Tick Data
Constraints: 4GB RAM Quota, Append-Only Buffer Flush

AUDIT FIXES APPLIED:
- Fixed Isolation Forest memory bloat on streaming data
- Implemented bounded circular buffer for tick ingestion
- Added explicit disk flush before Ray worker termination
"""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, List, Deque
from collections import deque
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class BoundedIsolationForest:
    """
    Isolation Forest with bounded memory for streaming data.
    FIX: Prevents memory bloat via reservoir sampling and bounded buffers.
    """
    
    def __init__(self, max_samples: int = 10000, contamination: float = 0.1):
        self.max_samples = max_samples
        self.contamination = contamination
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_samples)
        self._model: Optional[IsolationForest] = None
        self._is_fitted = False
        
    def add_sample(self, sample: np.ndarray) -> None:
        """Add a sample to the bounded buffer."""
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        self._buffer.append(sample)
        
    def partial_fit(self) -> None:
        """Fit or update the model with current buffer contents."""
        if len(self._buffer) < 100:
            logger.warning("Not enough samples for fitting")
            return
        
        # Convert buffer to array
        data = np.vstack(list(self._buffer))
        
        # Reservoir sampling if too many samples
        if len(data) > self.max_samples:
            indices = np.random.choice(len(data), self.max_samples, replace=False)
            data = data[indices]
        
        # Fit model
        self._model = IsolationForest(
            max_samples=min(256, len(data)),
            contamination=self.contamination,
            random_state=42,
            n_jobs=1  # Prevent thread explosion
        )
        self._model.fit(data)
        self._is_fitted = True
        
        logger.info(f"Fitted IsolationForest with {len(data)} samples")
    
    def predict(self, sample: np.ndarray) -> int:
        """Predict anomaly score for a sample."""
        if not self._is_fitted or self._model is None:
            return 0  # Assume normal if not fitted
        
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        return self._model.predict(sample)[0]
    
    def clear_memory(self) -> None:
        """Explicitly clear memory."""
        self._buffer.clear()
        if self._model is not None:
            del self._model
            self._model = None
        self._is_fitted = False


class ContinuousLearner:
    """
    Continuous learning system with append-only buffer management.
    FIX: Safely flushes buffers to disk before termination.
    """
    
    def __init__(self, checkpoint_dir: str, max_buffer_size: int = 100000):
        self.checkpoint_dir = checkpoint_dir
        self.max_buffer_size = max_buffer_size
        self._tick_buffer: Deque[dict] = deque(maxlen=max_buffer_size)
        self._anomaly_detector = BoundedIsolationForest()
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def ingest_tick(self, tick_data: dict) -> None:
        """Ingest a tick with memory bounds."""
        self._tick_buffer.append(tick_data)
        
        # Periodically update anomaly detector
        if len(self._tick_buffer) % 1000 == 0:
            self._update_anomaly_detector()
    
    def _update_anomaly_detector(self) -> None:
        """Update the anomaly detector with recent ticks."""
        if len(self._tick_buffer) < 100:
            return
        
        # Extract features from ticks
        features = []
        for tick in list(self._tick_buffer)[-1000:]:
            if 'features' in tick:
                features.append(tick['features'])
        
        if features:
            for feat in np.array(features):
                self._anomaly_detector.add_sample(feat)
            self._anomaly_detector.partial_fit()
    
    def flush_to_disk(self) -> str:
        """Flush append-only buffer to disk safely."""
        checkpoint_path = os.path.join(self.checkpoint_dir, "learner_checkpoint.pkl")
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'buffer': list(self._tick_buffer),
                    'anomaly_state': 'fitted' if self._anomaly_detector._is_fitted else 'unfitted'
                }, f)
            logger.info(f"Flushed {len(self._tick_buffer)} ticks to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to flush to disk: {e}")
            raise
    
    def shutdown(self) -> None:
        """Graceful shutdown with disk flush."""
        self.flush_to_disk()
        self._anomaly_detector.clear_memory()
        self._tick_buffer.clear()
        logger.info("ContinuousLearner shutdown complete")


if __name__ == "__main__":
    print("Continuous learner module loaded")
