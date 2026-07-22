"""
Cross-Validation for Time-Series - Purged K-Fold and Combinatorial CV

This module implements Purged K-Fold and Combinatorial Purged Cross-Validation
algorithms to strictly prevent data leakage in time-series training while
respecting the 4GB Python RAM ceiling. Based on Marcos Lopez de Prado's work.

Features:
- Purged K-Fold cross-validation
- Combinatorial Purged CV (CPCV)
- Embargo periods for leakage prevention
- Memory-efficient data handling
- Ray integration for parallel fold training
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from itertools import combinations

import ray

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVMethod(Enum):
    """Cross-validation method types"""
    PURGED_KFOLD = "purged_kfold"
    COMBINATORIAL = "combinatorial"
    EXPANDING_WINDOW = "expanding_window"


@dataclass
class CVSplit:
    """Represents a single train/test split"""
    train_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: int
    embargo_start: Optional[int] = None
    embargo_end: Optional[int] = None


@dataclass
class CVResult:
    """Results from cross-validation"""
    fold_id: int
    train_size: int
    test_size: int
    metric_score: float
    metric_name: str
    fit_time_ms: float
    predict_time_ms: float


def estimate_memory_usage(n_samples: int, n_features: int, dtype_bytes: int = 8) -> int:
    """Estimate memory usage for dataset"""
    return n_samples * n_features * dtype_bytes


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time-series data.
    
    Implements gap between training and testing sets to prevent
    information leakage from overlapping data points.
    
    Args:
        n_splits: Number of folds
        embargo_pct: Percentage of data to exclude after test set (0.0-1.0)
        shuffle: Whether to shuffle data before splitting (default False for TS)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.embargo_pct = max(0.0, min(1.0, embargo_pct))
        self.shuffle = shuffle
        self.random_state = random_state
        
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        
        logger.info(f"PurgedKFold initialized: splits={n_splits}, embargo={embargo_pct:.1%}")
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[CVSplit]:
        """
        Generate purged train/test splits.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (optional)
            groups: Group labels for grouped CV (optional)
        
        Yields:
            CVSplit objects with train/test indices
        """
        n_samples = len(X)
        
        # Create indices
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        
        # Calculate fold sizes
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        for fold_id in range(self.n_splits):
            # Test set boundaries
            test_start = fold_id * fold_size
            test_end = test_start + fold_size if fold_id < self.n_splits - 1 else n_samples
            
            # Test indices
            test_indices = indices[test_start:test_end]
            
            # Embargo: exclude samples immediately after test set
            embargo_end = min(test_end + embargo_size, n_samples)
            
            # Train indices: all samples except test and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:embargo_end] = False
            train_indices = indices[train_mask]
            
            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                fold_id=fold_id,
                embargo_start=test_end,
                embargo_end=embargo_end,
            )
    
    def get_n_splits(self) -> int:
        """Return number of splits"""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Generates all possible combinations of N folds taken K at a time,
    providing more robust validation than standard K-fold.
    
    This is computationally expensive but provides better estimates
    of out-of-sample performance for financial time series.
    
    Args:
        n_folds: Total number of folds to create
        n_test_folds: Number of test folds in each combination
        embargo_pct: Gap percentage between train and test
    """
    
    def __init__(
        self,
        n_folds: int = 6,
        n_test_folds: int = 2,
        embargo_pct: float = 0.01,
        random_state: Optional[int] = None,
        max_combinations: int = 100,  # Limit for memory management
    ):
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.embargo_pct = max(0.0, min(1.0, embargo_pct))
        self.random_state = random_state
        self.max_combinations = max_combinations
        
        if n_folds < 2 or n_test_folds < 1 or n_test_folds >= n_folds:
            raise ValueError("Invalid fold configuration")
        
        # Calculate total combinations
        self.total_combinations = len(list(combinations(range(n_folds), n_test_folds)))
        self.actual_combinations = min(self.total_combinations, max_combinations)
        
        logger.info(
            f"CPCV initialized: {n_folds} folds, {n_test_folds} test folds, "
            f"{self.actual_combinations}/{self.total_combinations} combinations"
        )
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Iterator[CVSplit]:
        """
        Generate combinatorial train/test splits.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Yields:
            CVSplit objects for each combination
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_folds
        
        # Pre-compute fold boundaries
        fold_boundaries = []
        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n_samples
            fold_boundaries.append((start, end))
        
        # Generate combinations
        combo_count = 0
        for test_folds in combinations(range(self.n_folds), self.n_test_folds):
            if combo_count >= self.actual_combinations:
                break
            
            # Combine test folds
            test_indices_list = []
            embargo_regions = []
            
            for fold_idx in test_folds:
                start, end = fold_boundaries[fold_idx]
                test_indices_list.append(np.arange(start, end))
                
                # Add embargo after each test fold
                embargo_end = min(end + int(n_samples * self.embargo_pct), n_samples)
                embargo_regions.append((end, embargo_end))
            
            # Concatenate test indices
            test_indices = np.concatenate(test_indices_list)
            
            # Sort embargo regions and merge overlapping
            embargo_regions.sort()
            
            # Create train mask excluding test and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Exclude test regions
            for start, end in fold_boundaries:
                if fold_boundaries.index((start, end)) in test_folds:
                    train_mask[start:end] = False
            
            # Exclude embargo regions
            for emb_start, emb_end in embargo_regions:
                train_mask[emb_start:emb_end] = False
            
            train_indices = np.arange(n_samples)[train_mask]
            
            yield CVSplit(
                train_indices=train_indices,
                test_indices=test_indices,
                fold_id=combo_count,
            )
            
            combo_count += 1
    
    def get_n_splits(self) -> int:
        """Return number of splits (combinations)"""
        return self.actual_combinations


@ray.remote(max_calls=100)
class CVWorker:
    """
    Ray worker for parallel fold training.
    Memory-limited to prevent exceeding 4GB ceiling.
    """
    
    def __init__(
        self,
        model_class: Any,
        model_params: Dict[str, Any],
        memory_limit_mb: int = 512,
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.memory_limit_mb = memory_limit_mb
        self.model = None
        
        # Set memory limit
        try:
            import resource
            memory_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            logger.info(f"CVWorker memory limited to {memory_limit_mb}MB")
        except Exception as e:
            logger.warning(f"Could not set memory limit: {e}")
    
    def train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fold_id: int,
        metric_name: str = 'accuracy',
    ) -> CVResult:
        """Train model on single fold and evaluate"""
        import time
        start_time = time.time()
        
        try:
            # Initialize and train model
            self.model = self.model_class(**self.model_params)
            self.model.fit(X_train, y_train)
            
            fit_time = (time.time() - start_time) * 1000
            
            # Predict and evaluate
            pred_start = time.time()
            predictions = self.model.predict(X_test)
            predict_time = (time.time() - pred_start) * 1000
            
            # Calculate metric
            if metric_name == 'accuracy':
                score = float(np.mean(predictions == y_test))
            elif metric_name == 'mse':
                score = float(np.mean((predictions - y_test) ** 2))
            elif metric_name == 'sharpe':
                # Simplified Sharpe ratio for trading signals
                returns = predictions * y_test  # Assume y_test contains returns
                if np.std(returns) > 0:
                    score = float(np.mean(returns) / np.std(returns))
                else:
                    score = 0.0
            else:
                score = 0.0
            
            return CVResult(
                fold_id=fold_id,
                train_size=len(X_train),
                test_size=len(X_test),
                metric_score=score,
                metric_name=metric_name,
                fit_time_ms=fit_time,
                predict_time_ms=predict_time,
            )
            
        except Exception as e:
            logger.error(f"Fold {fold_id} training failed: {e}")
            return CVResult(
                fold_id=fold_id,
                train_size=len(X_train),
                test_size=len(X_test),
                metric_score=0.0,
                metric_name=metric_name,
                fit_time_ms=0.0,
                predict_time_ms=0.0,
            )


class TimeSeriesCrossValidator:
    """
    Main cross-validation orchestrator for time-series data.
    Manages Ray workers and aggregates results.
    
    Args:
        cv_method: Type of cross-validation to use
        n_splits: Number of splits/folds
        embargo_pct: Gap percentage for leakage prevention
        memory_ceiling_gb: Maximum memory to use (default 4GB)
        n_workers: Number of parallel Ray workers
    """
    
    def __init__(
        self,
        cv_method: CVMethod = CVMethod.PURGED_KFOLD,
        n_splits: int = 5,
        n_test_folds: int = 2,
        embargo_pct: float = 0.01,
        memory_ceiling_gb: float = 4.0,
        n_workers: int = 4,
        random_state: Optional[int] = None,
    ):
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.n_test_folds = n_test_folds
        self.embargo_pct = embargo_pct
        self.memory_ceiling_bytes = int(memory_ceiling_gb * 1024**3)
        self.n_workers = n_workers
        self.random_state = random_state
        
        # Initialize CV splitter
        if cv_method == CVMethod.PURGED_KFOLD:
            self.cv_splitter = PurgedKFold(
                n_splits=n_splits,
                embargo_pct=embargo_pct,
                random_state=random_state,
            )
        elif cv_method == CVMethod.COMBINATORIAL:
            self.cv_splitter = CombinatorialPurgedCV(
                n_folds=n_splits,
                n_test_folds=n_test_folds,
                embargo_pct=embargo_pct,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown CV method: {cv_method}")
        
        # Initialize Ray if needed
        self.ray_initialized = False
        if not ray.is_initialized():
            ray.init(
                object_store_memory=int(self.memory_ceiling_bytes * 0.3),
                _system_config={"max_direct_call_object_size": 1024 * 1024},
            )
            self.ray_initialized = True
            logger.info(f"Ray initialized with {memory_ceiling_gb}GB ceiling for CV")
        
        # Workers pool
        self.workers: List[ray.actor.ActorHandle] = []
        
        logger.info(f"TimeSeriesCrossValidator initialized: method={cv_method.value}")
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_class: Any,
        model_params: Dict[str, Any],
        metric_name: str = 'accuracy',
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on time-series data.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_class: Model class to instantiate
            model_params: Parameters for model
            metric_name: Evaluation metric name
            sample_weight: Optional sample weights
        
        Returns:
            Dictionary with CV results and statistics
        """
        # Check memory constraints
        estimated_mem = estimate_memory_usage(len(X), X.shape[1])
        if estimated_mem > self.memory_ceiling_bytes * 0.5:
            logger.warning(
                f"Dataset may exceed memory limits: {estimated_mem / 1024**3:.2f}GB estimated"
            )
        
        # Initialize workers
        memory_per_worker = int((self.memory_ceiling_bytes * 0.5) / self.n_workers / 1024 / 1024)
        self.workers = [
            CVWorker.remote(model_class, model_params, memory_per_worker)
            for _ in range(self.n_workers)
        ]
        
        # Collect all splits
        splits = list(self.cv_splitter.split(X, y))
        logger.info(f"Running {len(splits)} CV splits")
        
        # Dispatch folds to workers (round-robin)
        futures = []
        for i, split in enumerate(splits):
            worker_idx = i % self.n_workers
            worker = self.workers[worker_idx]
            
            # Extract data for this fold
            X_train, X_test = X[split.train_indices], X[split.test_indices]
            y_train, y_test = y[split.train_indices], y[split.test_indices]
            
            future = worker.train_fold.remote(
                X_train, y_train, X_test, y_test,
                split.fold_id, metric_name,
            )
            futures.append(future)
        
        # Gather results
        results = ray.get(futures)
        
        # Aggregate statistics
        scores = [r.metric_score for r in results]
        fit_times = [r.fit_time_ms for r in results]
        predict_times = [r.predict_time_ms for r in results]
        
        summary = {
            'cv_method': self.cv_method.value,
            'n_splits': len(splits),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'metric_name': metric_name,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'mean_fit_time_ms': float(np.mean(fit_times)),
            'mean_predict_time_ms': float(np.mean(predict_times)),
            'fold_results': [
                {
                    'fold_id': r.fold_id,
                    'score': r.metric_score,
                    'train_size': r.train_size,
                    'test_size': r.test_size,
                }
                for r in results
            ],
        }
        
        logger.info(
            f"CV complete: {metric_name}={summary['mean_score']:.4f} ± {summary['std_score']:.4f}"
        )
        
        return summary
    
    def shutdown(self):
        """Shutdown Ray workers and release resources"""
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        
        if self.ray_initialized and ray.is_initialized():
            ray.shutdown()
        
        logger.info("Cross-validator shutdown complete")


# Example usage
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    
    # Generate synthetic time-series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.randn(n_samples) > 0).astype(int)
    
    # Test Purged K-Fold
    print("\n=== Testing Purged K-Fold ===")
    cv_purged = TimeSeriesCrossValidator(
        cv_method=CVMethod.PURGED_KFOLD,
        n_splits=5,
        embargo_pct=0.01,
        memory_ceiling_gb=4.0,
        n_workers=2,
    )
    
    results_purged = cv_purged.cross_validate(
        X, y,
        LogisticRegression,
        {'max_iter': 100},
        metric_name='accuracy',
    )
    
    print(f"Mean Accuracy: {results_purged['mean_score']:.4f}")
    print(f"Std Accuracy: {results_purged['std_score']:.4f}")
    
    cv_purged.shutdown()
    
    # Test Combinatorial CV (smaller due to computational cost)
    print("\n=== Testing Combinatorial Purged CV ===")
    cv_combo = TimeSeriesCrossValidator(
        cv_method=CVMethod.COMBINATORIAL,
        n_splits=4,
        n_test_folds=2,
        embargo_pct=0.01,
        memory_ceiling_gb=4.0,
        n_workers=2,
    )
    
    results_combo = cv_combo.cross_validate(
        X[:500], y[:500],  # Smaller dataset for demo
        LogisticRegression,
        {'max_iter': 100},
        metric_name='accuracy',
    )
    
    print(f"Mean Accuracy: {results_combo['mean_score']:.4f}")
    print(f"Std Accuracy: {results_combo['std_score']:.4f}")
    
    cv_combo.shutdown()
    
    print("\n=== Cross-Validation Tests Complete ===")
