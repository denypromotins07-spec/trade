"""
High-Frequency Cointegration Testing on Ray Workers

This module implements streaming Engle-Granger cointegration tests optimized for
high-frequency cryptocurrency pairs trading. Runs on Ray workers with strict
4GB RAM quota enforcement via mini-batch processing.

Optimized for:
- Streaming tick data processing
- 4GB Python RAM quota per worker
- AMD ROCm/DirectML acceleration checks
- Memory-bounded mini-batch operations
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from collections import deque
import ray

# AMD ROCm/DirectML environment detection
def detect_amd_acceleration() -> Dict[str, bool]:
    """Detect available AMD acceleration hardware."""
    result = {
        "rocm_available": False,
        "directml_available": False,
        "hip_available": False,
    }
    
    try:
        import torch
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            result["rocm_available"] = True
            result["hip_available"] = True
    except ImportError:
        pass
    
    try:
        import torch_directml
        result["directml_available"] = True
    except ImportError:
        pass
    
    return result


@dataclass
class CointegrationResult:
    """Result of a cointegration test."""
    is_cointegrated: bool
    hedge_ratio: float
    adf_statistic: float
    critical_value_1pct: float
    critical_value_5pct: float
    critical_value_10pct: float
    p_value: Optional[float]
    half_life: Optional[float]  # Mean reversion half-life in ticks
    spread_mean: float
    spread_std: float
    sample_count: int


class StreamingCointegrationTester:
    """
    Streaming Engle-Granger cointegration tester for high-frequency data.
    
    Uses bounded buffers and mini-batch processing to maintain strict
    memory limits suitable for Ray worker deployment.
    """
    
    # Critical values for ADF test (approximate, no constant)
    CRITICAL_VALUES = {
        0.01: -3.43,
        0.05: -2.86,
        0.10: -2.57,
    }
    
    def __init__(
        self,
        max_samples: int = 10000,
        mini_batch_size: int = 500,
        ram_quota_mb: int = 4096,
    ):
        """
        Initialize the streaming cointegration tester.
        
        Args:
            max_samples: Maximum samples to retain (bounds memory)
            mini_batch_size: Size of mini-batches for processing
            ram_quota_mb: RAM quota in MB for this worker
        """
        self.max_samples = max_samples
        self.mini_batch_size = mini_batch_size
        self.ram_quota_bytes = ram_quota_mb * 1024 * 1024
        
        # Bounded buffers for price series
        self.series1_buffer: deque = deque(maxlen=max_samples)
        self.series2_buffer: deque = deque(maxlen=max_samples)
        
        # Running statistics for O(1) updates
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._count = 0
        
        # Spread statistics
        self._spread_buffer: deque = deque(maxlen=max_samples)
        self._spread_mean = 0.0
        self._spread_m2 = 0.0  # For Welford's algorithm
        
        # AMD acceleration status
        self.amd_status = detect_amd_acceleration()
        
        # Estimate memory usage per sample (~100 bytes per sample pair)
        self._bytes_per_sample = 100
        self._max_samples_by_ram = (self.ram_quota_bytes // 2) // self._bytes_per_sample
        self.max_samples = min(self.max_samples, self._max_samples_by_ram)
        
    def add_tick(self, price1: float, price2: float, timestamp_ns: int) -> None:
        """
        Add a tick to the streaming buffers.
        
        Args:
            price1: Price of first asset
            price2: Price of second asset
            timestamp_ns: Timestamp in nanoseconds
        """
        # Check memory bounds
        if len(self.series1_buffer) >= self.max_samples:
            # Remove oldest and adjust running stats
            old_p1 = self.series1_buffer[0] if self.series1_buffer else 0
            old_p2 = self.series2_buffer[0] if self.series2_buffer else 0
            self._adjust_stats_on_remove(old_p1, old_p2)
        
        self.series1_buffer.append(price1)
        self.series2_buffer.append(price2)
        
        # Update running statistics
        self._sum_x += price1
        self._sum_y += price2
        self._sum_xy += price1 * price2
        self._sum_x2 += price1 * price1
        self._sum_y2 += price2 * price2
        self._count += 1
        
        # Calculate and store spread
        hedge_ratio = self._calculate_hedge_ratio_fast()
        if hedge_ratio != 0:
            spread = price1 - hedge_ratio * price2
            self._update_spread_stats(spread)
            self._spread_buffer.append(spread)
    
    def _adjust_stats_on_remove(self, p1: float, p2: float) -> None:
        """Adjust running statistics when removing oldest sample."""
        self._sum_x -= p1
        self._sum_y -= p2
        self._sum_xy -= p1 * p2
        self._sum_x2 -= p1 * p1
        self._sum_y2 -= p2 * p2
        self._count = max(0, self._count - 1)
    
    def _update_spread_stats(self, spread: float) -> None:
        """Update spread statistics using Welford's online algorithm."""
        delta = spread - self._spread_mean
        self._spread_mean += delta / len(self._spread_buffer)
        delta2 = spread - self._spread_mean
        self._spread_m2 += delta * delta2
    
    def _calculate_hedge_ratio_fast(self) -> float:
        """Calculate hedge ratio using running statistics."""
        if self._count < 2:
            return 0.0
        
        n = float(self._count)
        numerator = n * self._sum_xy - self._sum_x * self._sum_y
        denominator = n * self._sum_x2 - self._sum_x * self._sum_x
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    @property
    def spread_std(self) -> float:
        """Calculate spread standard deviation."""
        if len(self._spread_buffer) < 2:
            return 0.0
        variance = self._spread_m2 / (len(self._spread_buffer) - 1)
        return np.sqrt(variance)
    
    def _compute_adf_statistic(self) -> Tuple[float, Optional[float]]:
        """
        Compute augmented Dickey-Fuller test statistic on spread.
        
        Returns:
            Tuple of (ADF statistic, approximate p-value)
        """
        if len(self._spread_buffer) < 10:
            return 0.0, None
        
        spreads = np.array(list(self._spread_buffer), dtype=np.float64)
        
        # Simple ADF(1) regression: Δy_t = α + β*y_{t-1} + ε_t
        y_lag = spreads[:-1]
        delta_y = spreads[1:] - spreads[:-1]
        
        if len(y_lag) < 5 or np.std(y_lag) < 1e-10:
            return 0.0, None
        
        # OLS regression without intercept (for cointegration)
        # β = Σ(y_lag * delta_y) / Σ(y_lag^2)
        numerator = np.dot(y_lag, delta_y)
        denominator = np.dot(y_lag, y_lag)
        
        if abs(denominator) < 1e-10:
            return 0.0, None
        
        beta = numerator / denominator
        
        # Calculate t-statistic
        residuals = delta_y - beta * y_lag
        ssr = np.sum(residuals ** 2)
        mse = ssr / (len(residuals) - 1) if len(residuals) > 1 else 1.0
        
        se_beta = np.sqrt(mse / denominator)
        t_stat = beta / se_beta if se_beta > 0 else 0.0
        
        # Approximate p-value using normal distribution (rough approximation)
        p_value = None
        if abs(t_stat) > 0:
            # Very rough approximation
            p_value = min(1.0, 3.0 / abs(t_stat))
        
        return t_stat, p_value
    
    def _estimate_half_life(self) -> Optional[float]:
        """
        Estimate mean reversion half-life from AR(1) coefficient.
        
        Returns:
            Half-life in number of ticks, or None if not estimable
        """
        if len(self._spread_buffer) < 20:
            return None
        
        spreads = np.array(list(self._spread_buffer), dtype=np.float64)
        
        # Fit AR(1): y_t = α + φ*y_{t-1} + ε_t
        y_lag = spreads[:-1]
        y_curr = spreads[1:]
        
        if np.std(y_lag) < 1e-10:
            return None
        
        # OLS for φ
        y_lag_centered = y_lag - np.mean(y_lag)
        y_curr_centered = y_curr - np.mean(y_curr)
        
        phi = np.dot(y_lag_centered, y_curr_centered) / np.dot(y_lag_centered, y_lag_centered)
        
        if phi >= 1.0 or phi <= 0:
            return None  # Not mean-reverting
        
        # Half-life = -ln(2) / ln(φ)
        try:
            half_life = -np.log(2) / np.log(phi)
            return max(1.0, half_life)  # At least 1 tick
        except (ValueError, ZeroDivisionError):
            return None
    
    def test_cointegration(
        self,
        significance_level: float = 0.05,
    ) -> Optional[CointegrationResult]:
        """
        Perform Engle-Granger cointegration test.
        
        Args:
            significance_level: Significance level for the test (0.01, 0.05, or 0.10)
        
        Returns:
            CointegrationResult or None if insufficient data
        """
        if len(self.series1_buffer) < self.mini_batch_size:
            return None
        
        # Calculate hedge ratio
        hedge_ratio = self._calculate_hedge_ratio_fast()
        
        if hedge_ratio == 0:
            return None
        
        # Compute ADF statistic on spread
        adf_stat, p_value = self._compute_adf_statistic()
        
        # Get critical value
        critical_value = self.CRITICAL_VALUES.get(significance_level, -2.86)
        
        # Determine if cointegrated
        is_cointegrated = adf_stat < critical_value
        
        # Estimate half-life
        half_life = self._estimate_half_life()
        
        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            hedge_ratio=hedge_ratio,
            adf_statistic=adf_stat,
            critical_value_1pct=self.CRITICAL_VALUES[0.01],
            critical_value_5pct=self.CRITICAL_VALUES[0.05],
            critical_value_10pct=self.CRITICAL_VALUES[0.10],
            p_value=p_value,
            half_life=half_life,
            spread_mean=self._spread_mean,
            spread_std=self.spread_std,
            sample_count=len(self.series1_buffer),
        )
    
    def get_memory_usage_bytes(self) -> int:
        """Estimate current memory usage."""
        buffer_size = len(self.series1_buffer) * self._bytes_per_sample * 2
        spread_size = len(self._spread_buffer) * 8  # float64 per spread
        overhead = 1024 * 100  # ~100KB overhead
        return buffer_size + spread_size + overhead
    
    def check_ram_quota(self) -> bool:
        """Check if we're within RAM quota."""
        return self.get_memory_usage_bytes() < self.ram_quota_bytes
    
    def reset(self) -> None:
        """Reset all buffers and statistics."""
        self.series1_buffer.clear()
        self.series2_buffer.clear()
        self._spread_buffer.clear()
        self._sum_x = 0.0
        self._sum_y = 0.0
        self._sum_xy = 0.0
        self._sum_x2 = 0.0
        self._sum_y2 = 0.0
        self._count = 0
        self._spread_mean = 0.0
        self._spread_m2 = 0.0


@ray.remote(num_cpus=1, memory=4 * 1024 * 1024 * 1024)
class CointegrationWorker:
    """
    Ray worker for distributed cointegration testing.
    
    Enforces 4GB RAM quota and processes mini-batches efficiently.
    """
    
    def __init__(self, worker_id: int, symbol_pair: str):
        self.worker_id = worker_id
        self.symbol_pair = symbol_pair
        self.tester = StreamingCointegrationTester(
            max_samples=10000,
            mini_batch_size=500,
            ram_quota_mb=3500,  # Leave headroom
        )
        self.results_history: List[CointegrationResult] = []
    
    def process_mini_batch(
        self,
        prices1: List[float],
        prices2: List[float],
        timestamps: List[int],
    ) -> Dict[str, Any]:
        """Process a mini-batch of tick data."""
        if len(prices1) != len(prices2) or len(prices1) != len(timestamps):
            return {"error": "Mismatched batch lengths"}
        
        for p1, p2, ts in zip(prices1, prices2, timestamps):
            self.tester.add_tick(p1, p2, ts)
        
        # Check RAM quota
        if not self.tester.check_ram_quota():
            # Force garbage collection and trim buffers
            import gc
            gc.collect()
            # Trim to 80% of max
            new_max = int(self.tester.max_samples * 0.8)
            while len(self.tester.series1_buffer) > new_max:
                self.tester.series1_buffer.popleft()
                self.tester.series2_buffer.popleft()
        
        return {
            "worker_id": self.worker_id,
            "samples_processed": len(prices1),
            "total_samples": len(self.tester.series1_buffer),
            "memory_bytes": self.tester.get_memory_usage_bytes(),
        }
    
    def run_test(self, significance: float = 0.05) -> Optional[Dict[str, Any]]:
        """Run cointegration test and return results."""
        result = self.tester.test_cointegration(significance)
        
        if result is None:
            return None
        
        result_dict = {
            "symbol_pair": self.symbol_pair,
            "is_cointegrated": result.is_cointegrated,
            "hedge_ratio": result.hedge_ratio,
            "adf_statistic": result.adf_statistic,
            "critical_value": getattr(result, f"critical_value_{int(significance*100)}pct"),
            "half_life_ticks": result.half_life,
            "spread_mean": result.spread_mean,
            "spread_std": result.spread_std,
            "sample_count": result.sample_count,
            "amd_rocm": self.tester.amd_status["rocm_available"],
            "amd_directml": self.tester.amd_status["directml_available"],
        }
        
        self.results_history.append(result)
        return result_dict
    
    def get_status(self) -> Dict[str, Any]:
        """Get worker status including memory usage."""
        return {
            "worker_id": self.worker_id,
            "symbol_pair": self.symbol_pair,
            "samples": len(self.tester.series1_buffer),
            "memory_bytes": self.tester.get_memory_usage_bytes(),
            "within_quota": self.tester.check_ram_quota(),
            "amd_status": self.tester.amd_status,
        }
    
    def reset(self) -> Dict[str, bool]:
        """Reset worker state."""
        self.tester.reset()
        self.results_history.clear()
        return {"reset": True}


def create_cointegration_pool(
    num_workers: int,
    symbol_pairs: List[str],
) -> List[ray.actor.ActorHandle]:
    """
    Create a pool of cointegration workers on Ray.
    
    Args:
        num_workers: Number of workers to create
        symbol_pairs: List of symbol pairs to test
    
    Returns:
        List of worker actor handles
    """
    workers = []
    for i in range(min(num_workers, len(symbol_pairs))):
        worker = CointegrationWorker.remote(i, symbol_pairs[i])
        workers.append(worker)
    
    return workers


async def distribute_mini_batches(
    workers: List[ray.actor.ActorHandle],
    batches: List[Tuple[List[float], List[float], List[int]]],
) -> List[Dict[str, Any]]:
    """
    Distribute mini-batches across workers round-robin style.
    
    Args:
        workers: List of worker actors
        batches: List of (prices1, prices2, timestamps) tuples
    
    Returns:
        List of processing results
    """
    results = []
    for i, batch in enumerate(batches):
        worker = workers[i % len(workers)]
        prices1, prices2, timestamps = batch
        result = await worker.process_mini_batch.remote(prices1, prices2, timestamps)
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store
        _system_config={"max_worker_size": 4 * 1024 * 1024 * 1024},  # 4GB max worker
    )
    
    # Example usage
    pairs = ["ETH-BTC", "SOL-BTC", "AVAX-BTC", "MATIC-BTC"]
    workers = create_cointegration_pool(4, pairs)
    
    print(f"Created {len(workers)} cointegration workers")
    print(f"AMD Status: {ray.get(workers[0].get_status.remote())['amd_status']}")
    
    # Cleanup
    ray.shutdown()
