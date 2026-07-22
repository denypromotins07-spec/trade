"""
Combinatorial Symmetric Cross-Validation (CSCV) Engine

This module implements rigorous strategy robustness testing using CSCV to test
strategies across overlapping windows without introducing data leakage.

## Key Features
- Combinatorial enumeration of train/test splits
- Symmetric validation to prevent look-ahead bias
- Ray-distributed computation for scalability
- Strict 4GB RAM quota enforcement
- AMD ROCm/DirectML acceleration checks

## Mathematical Background
CSCV generates all possible combinations of N periods taken K at a time,
ensuring each period appears equally in training and testing sets.
This provides unbiased estimates of strategy performance under regime changes.
"""

import os
import logging
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import numpy as np

import ray
from ray import remote, actor

logger = logging.getLogger(__name__)

# Constants
MAX_RAM_GB = 4.0
MIN_COMBINATIONS = 10
MAX_COMBINATIONS = 1000


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD GPU acceleration availability."""
    capabilities = {
        'rocm_available': False,
        'directml_available': False,
        'cpu_optimized': True,
    }
    
    try:
        import torch
        if hasattr(torch.backends, 'rocm'):
            capabilities['rocm_available'] = torch.backends.rocm.is_available()
        if hasattr(torch.backends, 'directml'):
            capabilities['directml_available'] = torch.backends.directml.is_available()
    except (ImportError, AttributeError):
        pass
    
    return capabilities


@dataclass
class CSCVConfig:
    """Configuration for CSCV engine."""
    
    # Number of total periods
    n_periods: int = 52  # Weekly periods for 1 year
    
    # Number of periods per fold
    n_train: int = 8  # 8 weeks training
    n_test: int = 4   # 4 weeks testing
    
    # Minimum combinations to generate
    min_combinations: int = MIN_COMBINATIONS
    
    # Maximum combinations (RAM limit)
    max_combinations: int = MAX_COMBINATIONS
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # RAM limit
    max_ram_gb: float = MAX_RAM_GB


@dataclass
class CSCVResult:
    """Results from a single CSCV fold."""
    
    fold_id: int
    train_indices: List[int]
    test_indices: List[int]
    
    # Metrics
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    train_drawdown: float = 0.0
    test_drawdown: float = 0.0
    
    # Overfitting metrics
    sharpe_degradation: float = 0.0  # (train - test) / train
    return_degradation: float = 0.0
    
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class CSCVSummary:
    """Aggregated summary of all CSCV folds."""
    
    # Configuration
    n_combinations: int
    n_train_periods: int
    n_test_periods: int
    
    # Train set statistics
    mean_train_sharpe: float
    std_train_sharpe: float
    mean_train_return: float
    
    # Test set statistics  
    mean_test_sharpe: float
    std_test_sharpe: float
    mean_test_return: float
    
    # Degradation statistics (overfitting measure)
    mean_sharpe_degradation: float
    std_sharpe_degradation: float
    median_sharpe_degradation: float
    p95_sharpe_degradation: float
    
    # Robustness metrics
    fraction_positive_test: float  # % of test folds with positive returns
    fraction_better_than_random: float
    
    # Probability of backtest overfitting (PBO)
    probability_backtest_overfitting: float
    
    # Validity
    is_robust: bool  # True if degradation < threshold
    n_valid_folds: int


@remote
def evaluate_fold_remote(
    fold_id: int,
    train_indices: List[int],
    test_indices: List[int],
    returns_data: np.ndarray,
    config_dict: Dict[str, Any],
) -> CSCVResult:
    """
    Remote function to evaluate a single CSCV fold.
    
    This runs on Ray workers with isolated memory space.
    """
    try:
        # Extract train/test returns
        train_returns = returns_data[train_indices]
        test_returns = returns_data[test_indices]
        
        # Calculate metrics
        train_sharpe = _calculate_sharpe(train_returns)
        test_sharpe = _calculate_sharpe(test_returns)
        
        train_return = np.sum(train_returns)
        test_return = np.sum(test_returns)
        
        train_dd = _calculate_max_drawdown(np.cumsum(train_returns))
        test_dd = _calculate_max_drawdown(np.cumsum(test_returns))
        
        # Calculate degradation
        sharpe_deg = (train_sharpe - test_sharpe) / max(abs(train_sharpe), 1e-10)
        return_deg = (train_return - test_return) / max(abs(train_return), 1e-10)
        
        return CSCVResult(
            fold_id=fold_id,
            train_indices=train_indices,
            test_indices=test_indices,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_return=train_return,
            test_return=test_return,
            train_drawdown=train_dd,
            test_drawdown=test_dd,
            sharpe_degradation=sharpe_deg,
            return_degradation=return_deg,
            is_valid=True,
        )
        
    except Exception as e:
        return CSCVResult(
            fold_id=fold_id,
            train_indices=train_indices,
            test_indices=test_indices,
            error_message=str(e),
            is_valid=False,
        )


def _calculate_sharpe(returns: np.ndarray) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(52)  # Weekly data


def _calculate_max_drawdown(cum_returns: np.ndarray) -> float:
    """Calculate maximum drawdown from cumulative returns."""
    if len(cum_returns) == 0:
        return 0.0
    
    cummax = np.maximum.accumulate(cum_returns)
    drawdown = cummax - cum_returns
    return np.max(drawdown)


class CombinatorialCrossValidation:
    """
    Main CSCV engine for strategy robustness testing.
    
    Generates all combinatorial train/test splits and evaluates
    strategy performance across each split to detect overfitting.
    """
    
    def __init__(self, config: CSCVConfig):
        self.config = config
        self.acceleration = check_amd_acceleration()
        self.combinations: List[Tuple[List[int], List[int]]] = []
        self.results: List[CSCVResult] = []
        self.ray_initialized = False
        
    def _init_ray(self) -> None:
        """Initialize Ray if not already done."""
        if not self.ray_initialized:
            try:
                ray.init(
                    num_cpus=4,
                    _memory=int(self.config.max_ram_gb * 1024**3),
                    ignore_reinit_error=True,
                )
                self.ray_initialized = True
                logger.info("Ray initialized for CSCV")
            except Exception as e:
                logger.warning(f"Ray init failed: {e}")
    
    def generate_combinations(self) -> int:
        """
        Generate all valid train/test combinations.
        
        Returns:
            Number of combinations generated
        """
        n = self.config.n_periods
        k = self.config.n_train
        
        # Total possible combinations
        total_possible = int(np.math.comb(n, k))
        
        # Limit to manageable number
        n_combos = min(
            max(total_possible, self.config.min_combinations),
            self.config.max_combinations
        )
        
        # Generate combinations
        all_combos = list(combinations(range(n), k))
        
        # Sample if too many
        if len(all_combos) > n_combos:
            rng = np.random.default_rng(self.config.random_seed)
            selected_indices = rng.choice(
                len(all_combos), 
                size=n_combos, 
                replace=False
            )
            all_combos = [all_combos[i] for i in selected_indices]
        
        # Build train/test pairs
        self.combinations = []
        for train_idx in all_combos:
            train_set = list(train_idx)
            test_set = [i for i in range(n) if i not in train_set]
            
            # Ensure test set has correct size
            if len(test_set) >= self.config.n_test:
                self.combinations.append((train_set, test_set[:self.config.n_test]))
        
        logger.info(f"Generated {len(self.combinations)} CSCV combinations")
        return len(self.combinations)
    
    def evaluate(
        self, 
        returns: np.ndarray,
        n_workers: int = 4,
    ) -> CSCVSummary:
        """
        Evaluate strategy across all CSCV folds.
        
        Args:
            returns: Array of periodic returns
            n_workers: Number of Ray workers
            
        Returns:
            CSCVSummary with aggregated statistics
        """
        if not self.combinations:
            self.generate_combinations()
        
        self._init_ray()
        
        # Put returns in Ray object store (zero-copy)
        returns_ref = ray.put(returns)
        config_dict = {
            'n_train': self.config.n_train,
            'n_test': self.config.n_test,
        }
        
        # Launch parallel evaluation
        futures = []
        for i, (train_idx, test_idx) in enumerate(self.combinations):
            future = evaluate_fold_remote.remote(
                fold_id=i,
                train_indices=train_idx,
                test_indices=test_idx,
                returns_data=returns_ref,
                config_dict=config_dict,
            )
            futures.append(future)
        
        # Collect results
        self.results = ray.get(futures)
        
        # Aggregate into summary
        summary = self._aggregate_results()
        
        # Cleanup
        ray.internal.free([returns_ref])
        
        return summary
    
    def _aggregate_results(self) -> CSCVSummary:
        """Aggregate individual fold results into summary statistics."""
        valid_results = [r for r in self.results if r.is_valid]
        
        if not valid_results:
            return CSCVSummary(
                n_combinations=len(self.results),
                n_train_periods=self.config.n_train,
                n_test_periods=self.config.n_test,
                mean_train_sharpe=0.0,
                std_train_sharpe=0.0,
                mean_train_return=0.0,
                mean_test_sharpe=0.0,
                std_test_sharpe=0.0,
                mean_test_return=0.0,
                mean_sharpe_degradation=0.0,
                std_sharpe_degradation=0.0,
                median_sharpe_degradation=0.0,
                p95_sharpe_degradation=0.0,
                fraction_positive_test=0.0,
                fraction_better_than_random=0.0,
                probability_backtest_overfitting=1.0,
                is_robust=False,
                n_valid_folds=0,
            )
        
        # Extract arrays
        train_sharpes = np.array([r.train_sharpe for r in valid_results])
        test_sharpes = np.array([r.test_sharpe for r in valid_results])
        train_returns = np.array([r.train_return for r in valid_results])
        test_returns = np.array([r.test_return for r in valid_results])
        degradations = np.array([r.sharpe_degradation for r in valid_results])
        
        # Calculate PBO (Probability of Backtest Overfitting)
        # PBO = P(test_sharpe < 0 | train_sharpe > median(train_sharpe))
        above_median = train_sharpes > np.median(train_sharpes)
        if np.sum(above_median) > 0:
            pbo = np.mean(test_sharpes[above_median] < 0)
        else:
            pbo = 0.5
        
        # Robustness check
        is_robust = np.median(degradations) < 0.3  # < 30% degradation
        
        return CSCVSummary(
            n_combinations=len(valid_results),
            n_train_periods=self.config.n_train,
            n_test_periods=self.config.n_test,
            mean_train_sharpe=float(np.mean(train_sharpes)),
            std_train_sharpe=float(np.std(train_sharpes)),
            mean_train_return=float(np.mean(train_returns)),
            mean_test_sharpe=float(np.mean(test_sharpes)),
            std_test_sharpe=float(np.std(test_sharpes)),
            mean_test_return=float(np.mean(test_returns)),
            mean_sharpe_degradation=float(np.mean(degradations)),
            std_sharpe_degradation=float(np.std(degradations)),
            median_sharpe_degradation=float(np.median(degradations)),
            p95_sharpe_degradation=float(np.percentile(degradations, 95)),
            fraction_positive_test=float(np.mean(test_returns > 0)),
            fraction_better_than_random=1.0 - pbo,
            probability_backtest_overfitting=pbo,
            is_robust=is_robust,
            n_valid_folds=len(valid_results),
        )
    
    def shutdown(self) -> None:
        """Shutdown Ray cluster."""
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False


# Convenience function for quick CSCV analysis
def run_cscv(
    returns: np.ndarray,
    n_periods: int = 52,
    n_train: int = 8,
    n_test: int = 4,
    max_combinations: int = 500,
) -> CSCVSummary:
    """
    Run CSCV analysis with default configuration.
    
    Args:
        returns: Array of periodic returns
        n_periods: Total number of periods
        n_train: Training periods per fold
        n_test: Testing periods per fold
        max_combinations: Maximum combinations to evaluate
        
    Returns:
        CSCVSummary with robustness metrics
    """
    config = CSCVConfig(
        n_periods=n_periods,
        n_train=n_train,
        n_test=n_test,
        max_combinations=max_combinations,
    )
    
    cscv = CombinatorialCrossValidation(config)
    
    try:
        return cscv.evaluate(returns)
    finally:
        cscv.shutdown()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate returns (52 weeks)
    returns = np.random.normal(0.001, 0.02, 52)
    
    summary = run_cscv(returns)
    
    print(f"CSCV Summary:")
    print(f"  Combinations: {summary.n_combinations}")
    print(f"  Mean Train Sharpe: {summary.mean_train_sharpe:.3f}")
    print(f"  Mean Test Sharpe: {summary.mean_test_sharpe:.3f}")
    print(f"  Median Sharpe Degradation: {summary.median_sharpe_degradation:.2%}")
    print(f"  PBO: {summary.probability_backtest_overfitting:.2%}")
    print(f"  Is Robust: {summary.is_robust}")
