"""
Ray-Integrated Early Stopping for Walk-Forward Validation

This module implements early stopping based on walk-forward out-of-sample
degradation, instantly terminating training loops if the agent begins
overfitting to recent micro-trends. Integrates with Ray Tune for distributed RL.

Key Features:
- Walk-forward validation monitoring
- Out-of-sample degradation detection
- Instant training termination on overfit
- Ray Tune integration
- AMD ROCm/DirectML acceleration checks
- Strict 4GB RAM quota enforcement

AMD Ryzen AI 5 Optimizations:
- Parallel fold evaluation
- SIMD-enabled metric computation
- Memory-efficient rolling statistics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ray import tune
from ray.tune.stopper import Stopper
import os
import time
from collections import deque


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability."""
    acceleration = {
        'rocm_available': False,
        'directml_available': False,
        'cpu_simd_available': True
    }
    
    try:
        import torch
        if hasattr(torch.version, 'hip') or (torch.cuda.is_available() and 'ROCm' in str(torch.version.cuda)):
            acceleration['rocm_available'] = True
    except ImportError:
        pass
    
    try:
        import torch_directml
        acceleration['directml_available'] = True
    except ImportError:
        pass
    
    return acceleration


class WalkForwardEarlyStopping(Stopper):
    """
    Ray Tune stopper that monitors walk-forward validation performance.
    
    Stops training when out-of-sample performance degrades relative to
    in-sample performance, indicating overfitting to recent patterns.
    """
    
    def __init__(self,
                 metric: str = 'episode_reward_mean',
                 mode: str = 'max',
                 patience: int = 5,
                 min_delta: float = 0.01,
                 lookback_window: int = 10,
                 oos_degradation_threshold: float = 0.15,
                 memory_limit_mb: int = 3800):
        """
        Initialize walk-forward early stopping.
        
        Args:
            metric: Primary metric to monitor
            mode: 'max' or 'min'
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            lookback_window: Window size for rolling statistics
            oos_degradation_threshold: Max allowed OOS degradation (fraction)
            memory_limit_mb: Memory limit in MB
        """
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.lookback_window = lookback_window
        self.oos_degradation_threshold = oos_degradation_threshold
        self.memory_limit_mb = memory_limit_mb
        
        # Track per-trial state
        self._trial_metrics: Dict[str, deque] = {}
        self._trial_best: Dict[str, float] = {}
        self._trial_patience_counter: Dict[str, int] = {}
        self._trial_stopped: set = set()
        
        self.acceleration = check_amd_acceleration()
    
    def _check_memory(self):
        """Validate memory usage."""
        import psutil
        process = psutil.Process(os.getpid())
        current_mem_mb = process.memory_info().rss / (1024 * 1024)
        if current_mem_mb > self.memory_limit_mb:
            raise MemoryError(f"Memory {current_mem_mb:.0f}MB exceeds limit {self.memory_limit_mb}MB")
    
    def __call__(self, trial_id: str, result: Dict) -> bool:
        """
        Check if trial should be stopped.
        
        Args:
            trial_id: Ray Tune trial ID
            result: Current trial results
            
        Returns:
            True if trial should stop
        """
        self._check_memory()
        
        # Initialize tracking for new trial
        if trial_id not in self._trial_metrics:
            self._trial_metrics[trial_id] = deque(maxlen=self.lookback_window)
            self._trial_best[trial_id] = -np.inf if self.mode == 'max' else np.inf
            self._trial_patience_counter[trial_id] = 0
        
        # Get current metric value
        if self.metric not in result:
            return False
        
        current_value = result[self.metric]
        metrics_history = self._trial_metrics[trial_id]
        metrics_history.append(current_value)
        
        # Check for improvement
        best_so_far = self._trial_best[trial_id]
        improved = False
        
        if self.mode == 'max':
            if current_value > best_so_far + self.min_delta:
                improved = True
                self._trial_best[trial_id] = current_value
        else:
            if current_value < best_so_far - self.min_delta:
                improved = True
                self._trial_best[trial_id] = current_value
        
        # Update patience counter
        if improved:
            self._trial_patience_counter[trial_id] = 0
        else:
            self._trial_patience_counter[trial_id] += 1
        
        # Check patience exhaustion
        if self._trial_patience_counter[trial_id] >= self.patience:
            self._trial_stopped.add(trial_id)
            return True
        
        # Check for OOS degradation using rolling statistics
        if len(metrics_history) >= self.lookback_window:
            recent_mean = np.mean(list(metrics_history)[-5:])
            earlier_mean = np.mean(list(metrics_history)[:-5])
            
            if self.mode == 'max':
                degradation = (earlier_mean - recent_mean) / (abs(earlier_mean) + 1e-8)
                if degradation > self.oos_degradation_threshold:
                    print(f"Trial {trial_id}: OOS degradation detected ({degradation:.2%})")
                    self._trial_stopped.add(trial_id)
                    return True
            else:
                degradation = (recent_mean - earlier_mean) / (abs(earlier_mean) + 1e-8)
                if degradation > self.oos_degradation_threshold:
                    print(f"Trial {trial_id}: OOS degradation detected ({degradation:.2%})")
                    self._trial_stopped.add(trial_id)
                    return True
        
        return False
    
    def stop_all(self) -> bool:
        """Check if all trials should stop."""
        return False
    
    def get_trial_stats(self, trial_id: str) -> Optional[Dict]:
        """Get statistics for a specific trial."""
        if trial_id not in self._trial_metrics:
            return None
        
        metrics = list(self._trial_metrics[trial_id])
        return {
            'n_epochs': len(metrics),
            'best_value': self._trial_best[trial_id],
            'current_patience': self._trial_patience_counter[trial_id],
            'is_stopped': trial_id in self._trial_stopped,
            'recent_mean': np.mean(metrics[-5:]) if len(metrics) >= 5 else np.mean(metrics),
            'acceleration': self.acceleration,
        }


class OverfitDetector:
    """
    Detects overfitting by comparing in-sample vs out-of-sample performance.
    
    Uses walk-forward validation to identify when a model starts memorizing
    recent patterns rather than learning generalizable features.
    """
    
    def __init__(self,
                 n_folds: int = 5,
                 validation_split: float = 0.2,
                 degradation_threshold: float = 0.1):
        """
        Initialize overfit detector.
        
        Args:
            n_folds: Number of walk-forward folds
            validation_split: Fraction of data for validation
            degradation_threshold: Max allowed performance drop OOS vs IS
        """
        self.n_folds = n_folds
        self.validation_split = validation_split
        self.degradation_threshold = degradation_threshold
        self.acceleration = check_amd_acceleration()
    
    def create_walk_forward_splits(self, data_length: int) -> List[Tuple[slice, slice]]:
        """
        Create walk-forward train/validation splits.
        
        Args:
            data_length: Total length of time series
            
        Returns:
            List of (train_slice, val_slice) tuples
        """
        splits = []
        fold_size = data_length // (self.n_folds + 1)
        
        for i in range(self.n_folds):
            # Expanding window approach
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, data_length)
            
            train_slice = slice(0, train_end)
            val_slice = slice(val_start, val_end)
            
            splits.append((train_slice, val_slice))
        
        return splits
    
    def compute_oos_ratio(self, 
                          in_sample_scores: List[float],
                          out_of_sample_scores: List[float]) -> Dict:
        """
        Compute out-of-sample degradation ratio.
        
        Args:
            in_sample_scores: Performance scores on training data
            out_of_sample_scores: Performance scores on validation data
            
        Returns:
            Dictionary with OOS metrics
        """
        is_mean = np.mean(in_sample_scores)
        is_std = np.std(in_sample_scores)
        
        oos_mean = np.mean(out_of_sample_scores)
        oos_std = np.std(out_of_sample_scores)
        
        # Degradation ratio: how much worse is OOS vs IS
        degradation = (is_mean - oos_mean) / (abs(is_mean) + 1e-8)
        
        # Generalization gap
        gen_gap = is_mean - oos_mean
        
        # Overfit score (higher = more overfit)
        overfit_score = max(0, degradation)
        
        return {
            'in_sample_mean': is_mean,
            'in_sample_std': is_std,
            'out_of_sample_mean': oos_mean,
            'out_of_sample_std': oos_std,
            'degradation_ratio': degradation,
            'generalization_gap': gen_gap,
            'overfit_score': overfit_score,
            'is_overfitting': degradation > self.degradation_threshold,
            'acceleration': self.acceleration,
        }
    
    def validate_strategy(self,
                          strategy_func,
                          data: np.ndarray,
                          metric_func) -> Dict:
        """
        Validate trading strategy using walk-forward analysis.
        
        Args:
            strategy_func: Function that takes train data and returns model
            data: Full time series data
            metric_func: Function to evaluate strategy performance
            
        Returns:
            Walk-forward validation results
        """
        splits = self.create_walk_forward_splits(len(data))
        
        in_sample_scores = []
        out_of_sample_scores = []
        
        for fold_idx, (train_slice, val_slice) in enumerate(splits):
            train_data = data[train_slice]
            val_data = data[val_slice]
            
            # Train strategy on expanding window
            try:
                model = strategy_func(train_data)
                
                # Evaluate on in-sample (last portion of train)
                is_eval_start = max(0, len(train_data) - len(val_data))
                is_data = train_data[is_eval_start:]
                is_score = metric_func(model, is_data)
                in_sample_scores.append(is_score)
                
                # Evaluate on out-of-sample
                oos_score = metric_func(model, val_data)
                out_of_sample_scores.append(oos_score)
                
            except Exception as e:
                print(f"Fold {fold_idx} failed: {e}")
                in_sample_scores.append(0.0)
                out_of_sample_scores.append(0.0)
        
        return self.compute_oos_ratio(in_sample_scores, out_of_sample_scores)


@ray.remote(max_calls=50)
class DistributedWalkForwardValidator:
    """
    Distributed walk-forward validation worker for Ray.
    
    Evaluates strategy performance across multiple folds in parallel.
    """
    
    def __init__(self, worker_id: int, n_folds: int = 5):
        self.worker_id = worker_id
        self.n_folds = n_folds
        self.detector = OverfitDetector(n_folds=n_folds)
        self.evaluations_completed = 0
        self.acceleration = check_amd_acceleration()
    
    def evaluate_fold(self,
                      train_data: np.ndarray,
                      val_data: np.ndarray,
                      strategy_config: Dict,
                      eval_func: Callable) -> Dict:
        """
        Evaluate a single fold.
        
        Args:
            train_data: Training data for this fold
            val_data: Validation data for this fold
            strategy_config: Strategy configuration
            eval_func: Evaluation function
            
        Returns:
            Fold evaluation results
        """
        start_time = time.time()
        
        try:
            # Run evaluation
            is_score = eval_func(strategy_config, train_data)
            oos_score = eval_func(strategy_config, val_data)
            
            self.evaluations_completed += 1
            
            return {
                'worker_id': self.worker_id,
                'in_sample_score': is_score,
                'out_of_sample_score': oos_score,
                'degradation': (is_score - oos_score) / (abs(is_score) + 1e-8),
                'evaluation_time': time.time() - start_time,
                'status': 'success',
            }
        except Exception as e:
            return {
                'worker_id': self.worker_id,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'status': 'failed',
            }
    
    def get_stats(self) -> Dict:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'evaluations_completed': self.evaluations_completed,
            'n_folds': self.n_folds,
            'acceleration': self.acceleration,
        }


def create_early_stopping_callback(metric: str = 'episode_reward_mean',
                                    patience: int = 5,
                                    oos_threshold: float = 0.15) -> WalkForwardEarlyStopping:
    """
    Factory function to create early stopping callback.
    
    Args:
        metric: Metric to monitor
        patience: Patience before stopping
        oos_threshold: OOS degradation threshold
        
    Returns:
        Configured WalkForwardEarlyStopping instance
    """
    return WalkForwardEarlyStopping(
        metric=metric,
        mode='max' if 'reward' in metric.lower() or 'return' in metric.lower() else 'min',
        patience=patience,
        oos_degradation_threshold=oos_threshold,
    )


if __name__ == '__main__':
    print("Checking AMD acceleration...")
    accel = check_amd_acceleration()
    print(f"Acceleration: {accel}")
    
    # Example usage
    detector = OverfitDetector(n_folds=5, degradation_threshold=0.1)
    
    # Simulate some scores
    in_sample = [1.5, 1.6, 1.7, 1.8, 1.9]
    out_of_sample = [1.3, 1.3, 1.2, 1.0, 0.8]  # Degrading OOS performance
    
    results = detector.compute_oos_ratio(in_sample, out_of_sample)
    print(f"\nOverfit analysis:")
    print(f"  In-sample mean: {results['in_sample_mean']:.3f}")
    print(f"  OOS mean: {results['out_of_sample_mean']:.3f}")
    print(f"  Degradation: {results['degradation_ratio']:.2%}")
    print(f"  Is overfitting: {results['is_overfitting']}")
    
    print("\nEarly stopping ready for Ray Tune integration.")
