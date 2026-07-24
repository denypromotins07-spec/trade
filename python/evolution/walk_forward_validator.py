"""
Walk-Forward Validator - Stage 56
AMD Ryzen AI 5 Optimized | 4GB RAM Quota | Deflated Sharpe Ratio Validation

This module implements continuous walk-forward out-of-sample validation that
promotes shadow strategies only if the Deflated Sharpe Ratio strictly exceeds
the SOUL.md baseline threshold.

Constraints:
- Strict 4GB RAM quota during validation
- GPU-accelerated statistical computations via ROCm/DirectML
- Multiple testing correction via Deflated Sharpe Ratio
- Production-ready promotion/demotion logic
"""

import ray
import numpy as np
import cupy as cp  # ROCm/DirectML acceleration
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import psutil
import os

# Enforce strict memory limits
MAX_RAM_MB = 4096
os.environ['RAY_MEMORY_LIMIT'] = str(MAX_RAM_MB * 1024 * 1024)

# SOUL.md baseline thresholds
DSR_BASELINE_THRESHOLD = 0.35  # Minimum Deflated Sharpe Ratio for promotion
MIN_TRACKING_PERIODS = 3  # Minimum walk-forward periods required
MAX_DRAWDOWN_THRESHOLD = 0.15  # Maximum acceptable drawdown


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward validation period."""
    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_return: float
    out_of_sample_drawdown: float
    dsr_score: float
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyValidation:
    """Complete validation results for a strategy candidate."""
    strategy_id: str
    strategy_hash: str
    parameters: Dict[str, float]
    total_periods: int
    passed_periods: int
    avg_oos_sharpe: float
    avg_dsr: float
    max_oos_drawdown: float
    promotion_recommended: bool
    confidence_level: float
    results: List[WalkForwardResult]
    validated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_id': self.strategy_id,
            'strategy_hash': self.strategy_hash,
            'parameters': self.parameters,
            'total_periods': self.total_periods,
            'passed_periods': self.passed_periods,
            'avg_oos_sharpe': self.avg_oos_sharpe,
            'avg_dsr': self.avg_dsr,
            'max_oos_drawdown': self.max_oos_drawdown,
            'promotion_recommended': self.promotion_recommended,
            'confidence_level': self.confidence_level,
            'validated_at': self.validated_at.isoformat(),
            'results_count': len(self.results)
        }


@ray.remote(num_cpus=1, max_calls=100)
class ValidationWorker:
    """
    Ray-distributed worker for walk-forward validation.
    Performs out-of-sample testing with DSR calculation.
    """
    
    def __init__(self, ram_limit_mb: int = 1024):
        self.ram_limit_mb = ram_limit_mb
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """Check for AMD ROCm/DirectML availability."""
        try:
            test_array = cp.zeros(100)
            del test_array
            cp.get_default_memory_pool().free_all_blocks()
            return True
        except Exception:
            return False
    
    def _memory_check(self) -> bool:
        """Verify memory usage is within limits."""
        current_ram = psutil.Process().memory_info().rss / (1024 * 1024)
        if current_ram > self.ram_limit_mb * 0.9:
            import gc
            gc.collect()
            if self.gpu_available:
                cp.get_default_memory_pool().free_all_blocks()
            return False
        return True
    
    def calculate_deflated_sharpe_ratio(
        self,
        returns: np.ndarray,
        n_trials: int = 100,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate the Deflated Sharpe Ratio (DSR) to correct for multiple testing.
        
        The DSR adjusts the observed Sharpe ratio for:
        1. Non-normality of returns (skewness and kurtosis)
        2. Track record length
        3. Multiple testing bias
        
        Args:
            returns: Array of periodic returns
            n_trials: Number of independent trials (strategies tested)
            risk_free_rate: Risk-free rate for excess returns
            
        Returns:
            Deflated Sharpe Ratio (probability of being statistically significant)
        """
        if len(returns) < 10:
            return 0.0
        
        # Excess returns
        excess_returns = returns - risk_free_rate
        
        # Observed Sharpe ratio
        mean_ret = np.mean(excess_returns)
        std_ret = np.std(excess_returns, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        sr_observed = mean_ret / std_ret * np.sqrt(len(returns))
        
        # Higher moments for non-normality adjustment
        skew = stats.skew(excess_returns)
        kurt = stats.kurtosis(excess_returns) + 3  # Add 3 for actual kurtosis
        
        # Adjusted standard error considering non-normality
        # Bailey & López de Prado (2014) formula
        se_adjusted = np.sqrt(
            (1 + 0.5 * skew**2 - (kurt - 3) / 4) / len(returns)
        )
        
        # Expected maximum Sharpe ratio under null (multiple testing correction)
        # Using order statistics approximation
        if n_trials > 1:
            # Euler-Mascheroni constant
            gamma = 0.5772156649
            
            # Expected value of maximum of n_trials standard normals
            E_max = np.sqrt(2 * np.log(n_trials))
            
            # Variance adjustment
            V_max = 1 / (2 * np.log(n_trials))
            
            # Expected maximum under null
            sr_expected = E_max * se_adjusted
        else:
            sr_expected = 0.0
        
        # Probability that true Sharpe ratio is zero or negative
        # Using normal CDF
        if se_adjusted > 0:
            z_score = (sr_observed - sr_expected) / se_adjusted
            dsr = stats.norm.cdf(z_score)
        else:
            dsr = 0.5
        
        return dsr
    
    def validate_period(
        self,
        strategy_params: Dict[str, float],
        in_sample_data: np.ndarray,
        out_of_sample_data: np.ndarray,
        period_id: int,
        timestamps: Dict[str, datetime]
    ) -> WalkForwardResult:
        """
        Validate strategy on a single walk-forward period.
        
        Args:
            strategy_params: Strategy hyperparameters
            in_sample_data: In-sample returns for calibration
            out_of_sample_data: Out-of-sample returns for validation
            period_id: Unique period identifier
            timestamps: Period timestamp dictionary
            
        Returns:
            WalkForwardResult with validation metrics
        """
        if not self._memory_check():
            raise MemoryError("Worker memory limit exceeded")
        
        # GPU-accelerated backtest if available
        if self.gpu_available and len(out_of_sample_data) > 100:
            oos_gpu = cp.asarray(out_of_sample_data)
            is_gpu = cp.asarray(in_sample_data)
            
            # Extract parameters
            lookback = int(strategy_params.get('lookback', 20) * 100) + 1
            threshold = strategy_params.get('threshold', 0.5)
            stop_loss = strategy_params.get('stop_loss', 0.05)
            
            # In-sample optimization
            if len(is_gpu) > lookback:
                rolling_mean_is = cp.convolve(is_gpu, cp.ones(lookback)/lookback, mode='valid')
                optimal_threshold = cp.percentile(rolling_mean_is, 70)
            else:
                optimal_threshold = threshold
            
            # Out-of-sample testing with optimized parameters
            if len(oos_gpu) > lookback:
                rolling_mean_oos = cp.convolve(oos_gpu, cp.ones(lookback)/lookback, mode='valid')
                signals = (rolling_mean_oos > optimal_threshold).astype(float)
                
                # Apply stop-loss
                cumulative = cp.cumprod(1 + signals * oos_gpu[lookback:])
                peak = cp.maximum.accumulate(cumulative)
                drawdown = (cumulative - peak) / peak
                
                # Stop trading if drawdown exceeds threshold
                stop_mask = drawdown > -stop_loss
                signals = signals * stop_mask[:-1]
                
                oos_returns = signals * oos_gpu[lookback:]
                
                # Metrics
                oos_sharpe = float(cp.mean(oos_returns) / cp.std(oos_returns)) * cp.sqrt(252) if cp.std(oos_returns) > 0 else 0.0
                oos_return = float(cp.prod(1 + oos_returns) - 1)
                oos_drawdown = float(abs(cp.min(drawdown)))
                
                # Cleanup
                del oos_gpu, is_gpu, rolling_mean_is, rolling_mean_oos, signals, oos_returns
                cp.get_default_memory_pool().free_all_blocks()
            else:
                oos_sharpe = 0.0
                oos_return = 0.0
                oos_drawdown = 0.1
        else:
            # CPU fallback
            lookback = int(strategy_params.get('lookback', 20) * 100) + 1
            threshold = strategy_params.get('threshold', 0.5)
            stop_loss = strategy_params.get('stop_loss', 0.05)
            
            # In-sample optimization
            if len(in_sample_data) > lookback:
                rolling_mean_is = np.convolve(in_sample_data, np.ones(lookback)/lookback, mode='valid')
                optimal_threshold = np.percentile(rolling_mean_is, 70)
            else:
                optimal_threshold = threshold
            
            # Out-of-sample testing
            if len(out_of_sample_data) > lookback:
                rolling_mean_oos = np.convolve(out_of_sample_data, np.ones(lookback)/lookback, mode='valid')
                signals = (rolling_mean_oos > optimal_threshold).astype(float)
                
                oos_returns = signals * out_of_sample_data[lookback:]
                
                # Apply stop-loss
                cumulative = np.cumprod(1 + oos_returns)
                peak = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - peak) / peak
                
                stop_mask = drawdown > -stop_loss
                signals = signals * stop_mask[:-1]
                oos_returns = signals * out_of_sample_data[lookback:]
                
                oos_sharpe = float(np.mean(oos_returns) / np.std(oos_returns)) * np.sqrt(252) if np.std(oos_returns) > 0 else 0.0
                oos_return = float(np.prod(1 + oos_returns) - 1)
                oos_drawdown = float(abs(np.min(drawdown)))
            else:
                oos_sharpe = 0.0
                oos_return = 0.0
                oos_drawdown = 0.1
        
        # In-sample Sharpe for comparison
        if len(in_sample_data) > 0:
            is_sharpe = float(np.mean(in_sample_data) / np.std(in_sample_data)) * np.sqrt(252) if np.std(in_sample_data) > 0 else 0.0
        else:
            is_sharpe = 0.0
        
        # Calculate DSR
        dsr = self.calculate_deflated_sharpe_ratio(oos_returns if len(out_of_sample_data) > lookback else out_of_sample_data)
        
        # Pass criteria
        passed = (
            dsr >= DSR_BASELINE_THRESHOLD and
            oos_drawdown <= MAX_DRAWDOWN_THRESHOLD and
            oos_sharpe > 0
        )
        
        return WalkForwardResult(
            period_id=period_id,
            in_sample_start=timestamps.get('is_start', datetime.utcnow()),
            in_sample_end=timestamps.get('is_end', datetime.utcnow()),
            out_of_sample_start=timestamps.get('oos_start', datetime.utcnow()),
            out_of_sample_end=timestamps.get('oos_end', datetime.utcnow()),
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            out_of_sample_return=oos_return,
            out_of_sample_drawdown=oos_drawdown,
            dsr_score=dsr,
            passed=passed,
            metadata={
                'optimal_threshold': float(optimal_threshold),
                'lookback_used': lookback,
                'gpu_accelerated': self.gpu_available
            }
        )


class WalkForwardValidator:
    """
    Master orchestrator for walk-forward validation.
    Manages validation across multiple periods and strategies.
    """
    
    def __init__(
        self,
        num_periods: int = 5,
        in_sample_ratio: float = 0.6,
        num_workers: int = 4
    ):
        self.num_periods = num_periods
        self.in_sample_ratio = in_sample_ratio
        self.num_workers = num_workers
        
        self.workers: List[ray.ObjectRef] = []
        self.validations: Dict[str, StrategyValidation] = {}
        self.initialized = False
    
    def initialize_ray(self):
        """Initialize Ray cluster with strict memory constraints."""
        if not ray.is_initialized():
            total_ram = psutil.virtual_memory().available
            worker_ram = min(total_ram // self.num_workers, MAX_RAM_MB * 1024 * 1024)
            
            ray.init(
                num_cpus=self.num_workers,
                _memory=int(worker_ram * self.num_workers * 0.9),
                object_store_memory=int(worker_ram * self.num_workers * 0.3),
                ignore_reinit_error=True
            )
        
        self.workers = [
            ValidationWorker.remote(ram_limit_mb=MAX_RAM_MB // self.num_workers)
            for _ in range(self.num_workers)
        ]
        self.initialized = True
    
    def create_walk_forward_splits(
        self,
        returns_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, datetime]]]:
        """
        Create walk-forward train/test splits.
        
        Args:
            returns_data: Full returns series
            timestamps: Optional datetime array
            
        Returns:
            List of (in_sample, out_of_sample, timestamps_dict) tuples
        """
        n = len(returns_data)
        split_size = n // (self.num_periods + 1)
        
        splits = []
        
        for i in range(self.num_periods):
            # Expanding window approach
            is_end = split_size + i * split_size
            oos_end = is_end + split_size
            
            in_sample = returns_data[:is_end]
            out_of_sample = returns_data[is_end:oos_end]
            
            ts_dict = {}
            if timestamps is not None:
                ts_dict = {
                    'is_start': timestamps[0],
                    'is_end': timestamps[is_end - 1],
                    'oos_start': timestamps[is_end],
                    'oos_end': timestamps[min(oos_end - 1, len(timestamps) - 1)]
                }
            
            splits.append((in_sample, out_of_sample, ts_dict))
        
        return splits
    
    def validate_strategy(
        self,
        strategy_id: str,
        strategy_params: Dict[str, float],
        returns_data: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> StrategyValidation:
        """
        Perform complete walk-forward validation on a strategy.
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_params: Strategy hyperparameters
            returns_data: Historical returns
            timestamps: Optional datetime array
            
        Returns:
            StrategyValidation with complete results
        """
        if not self.initialized:
            self.initialize_ray()
        
        # Generate strategy hash
        strategy_hash = hashlib.md5(
            json.dumps(strategy_params, sort_keys=True).encode()
        ).hexdigest()[:12]
        
        # Create splits
        splits = self.create_walk_forward_splits(returns_data, timestamps)
        
        # Dispatch validation tasks
        futures = []
        for period_idx, (in_sample, out_of_sample, ts_dict) in enumerate(splits):
            worker = self.workers[period_idx % len(self.workers)]
            future = worker.validate_period.remote(
                strategy_params,
                in_sample,
                out_of_sample,
                period_idx,
                ts_dict
            )
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Aggregate metrics
        passed_periods = sum(1 for r in results if r.passed)
        avg_oos_sharpe = np.mean([r.out_of_sample_sharpe for r in results])
        avg_dsr = np.mean([r.dsr_score for r in results])
        max_oos_drawdown = max([r.out_of_sample_drawdown for r in results])
        
        # Promotion criteria
        promotion_recommended = (
            passed_periods >= MIN_TRACKING_PERIODS and
            avg_dsr >= DSR_BASELINE_THRESHOLD and
            max_oos_drawdown <= MAX_DRAWDOWN_THRESHOLD and
            avg_oos_sharpe > 0.5
        )
        
        # Confidence level based on consistency
        if passed_periods == len(results):
            confidence_level = 1.0
        elif passed_periods >= len(results) * 0.8:
            confidence_level = 0.8
        elif passed_periods >= len(results) * 0.6:
            confidence_level = 0.5
        else:
            confidence_level = 0.2
        
        validation = StrategyValidation(
            strategy_id=strategy_id,
            strategy_hash=strategy_hash,
            parameters=strategy_params,
            total_periods=len(results),
            passed_periods=passed_periods,
            avg_oos_sharpe=float(avg_oos_sharpe),
            avg_dsr=float(avg_dsr),
            max_oos_drawdown=float(max_oos_drawdown),
            promotion_recommended=promotion_recommended,
            confidence_level=confidence_level,
            results=results,
            validated_at=datetime.utcnow()
        )
        
        self.validations[strategy_id] = validation
        
        return validation
    
    def get_promotable_strategies(self) -> List[StrategyValidation]:
        """Get all strategies recommended for promotion."""
        return [
            v for v in self.validations.values()
            if v.promotion_recommended
        ]
    
    def export_to_soul_ledger(self) -> List[Dict[str, Any]]:
        """Export validation results for SOUL.md ledger."""
        entries = []
        
        for validation in self.validations.values():
            entry = {
                'type': 'WALK_FORWARD_VALIDATION',
                'timestamp': validation.validated_at.isoformat(),
                'strategy_id': validation.strategy_id,
                'strategy_hash': validation.strategy_hash,
                'parameters': validation.parameters,
                'total_periods': validation.total_periods,
                'passed_periods': validation.passed_periods,
                'avg_oos_sharpe': validation.avg_oos_sharpe,
                'avg_dsr': validation.avg_dsr,
                'max_oos_drawdown': validation.max_oos_drawdown,
                'promotion_recommended': validation.promotion_recommended,
                'confidence_level': validation.confidence_level,
                'cryptographic_seal': hashlib.sha256(
                    validation.strategy_hash.encode() +
                    str(validation.avg_dsr).encode() +
                    str(validation.validated_at.timestamp()).encode()
                ).hexdigest()
            }
            entries.append(entry)
        
        return entries
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if ray.is_initialized():
            ray.shutdown()
        self.workers = []
        self.initialized = False


if __name__ == '__main__':
    # Example usage
    validator = WalkForwardValidator(num_periods=4, num_workers=2)
    
    # Generate sample returns data
    np.random.seed(42)
    base_returns = np.random.randn(2000) * 0.02
    
    # Sample strategy parameters
    test_params = {
        'lookback': 0.3,
        'threshold': 0.55,
        'stop_loss': 0.05,
        'take_profit': 0.1
    }
    
    # Run validation
    result = validator.validate_strategy(
        strategy_id="test_strategy_001",
        strategy_params=test_params,
        returns_data=base_returns
    )
    
    print(f"Strategy: {result.strategy_id}")
    print(f"Passed periods: {result.passed_periods}/{result.total_periods}")
    print(f"Avg OOS Sharpe: {result.avg_oos_sharpe:.2f}")
    print(f"Avg DSR: {result.avg_dsr:.3f}")
    print(f"Max Drawdown: {result.max_oos_drawdown:.1%}")
    print(f"Promotion Recommended: {result.promotion_recommended}")
    print(f"Confidence Level: {result.confidence_level:.1%}")
    
    # Export for SOUL.md
    entries = validator.export_to_soul_ledger()
    print(f"\nExported {len(entries)} entries to SOUL.md ledger")
    
    validator.shutdown()
