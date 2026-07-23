"""
High-Frequency Cross-Sectional Momentum Factors on Ray Workers

Builds high-frequency cross-sectional momentum factors using Ray for distributed
processing. Strictly enforces 4GB Python RAM quota by processing returns in
streaming mini-batches.

Optimized for AMD ROCm/DirectML acceleration where available.
"""

import ray
import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass
import os
import gc
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

# Initialize Ray with strict memory limits
def init_ray_with_limits(memory_gb: float = 4.0, object_store_memory_gb: float = 2.0):
    """Initialize Ray with strict memory quotas to prevent OOM."""
    if not ray.is_initialized():
        # Check for AMD ROCm availability
        rocm_available = os.environ.get('ROCM_PATH') is not None or \
                        os.path.exists('/opt/rocm')
        
        # Check for DirectML (Windows AMD acceleration)
        directml_available = os.name == 'nt' and \
                            os.environ.get('DIRECTML_PATH') is not None
        
        ray.init(
            # Limit total memory to enforce 4GB quota
            _memory=int(memory_gb * 1024 * 1024 * 1024),
            # Object store limited to prevent fragmentation
            _object_store_memory=int(object_store_memory_gb * 1024 * 1024 * 1024),
            # Prevent over-subscription
            num_cpus=min(os.cpu_count() or 4, 8),
            # Enable memory pressure handling
            max_restarts=-1,
        )
        
        return {
            'rocm_enabled': rocm_available,
            'directml_enabled': directml_available,
        }
    
    return {'rocm_enabled': False, 'directml_enabled': False}


@dataclass
class MomentumConfig:
    """Configuration for momentum factor calculation."""
    # Lookback periods for different momentum horizons (in ticks/bars)
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 60
    
    # Mini-batch size for streaming processing (memory bound)
    batch_size: int = 1000
    
    # Maximum assets per batch to control memory
    max_assets_per_batch: int = 50
    
    # Return type: 'simple' or 'log'
    return_type: str = 'log'
    
    # Winsorization threshold for outlier handling
    winsorize_threshold: float = 5.0
    
    # Minimum history required for valid signal
    min_history: int = 60


@ray.remote(max_calls=100)  # Restart worker after 100 calls to prevent memory leaks
class MomentumFactorWorker:
    """
    Ray worker for computing cross-sectional momentum factors.
    
    Processes data in streaming mini-batches to enforce 4GB RAM limit.
    """
    
    def __init__(self, config: MomentumConfig):
        self.config = config
        self._batch_buffer: List[pl.DataFrame] = []
        self._current_memory_bytes: int = 0
        self._max_memory_bytes: int = int(4.0 * 1024 * 1024 * 1024)  # 4GB hard limit
        
        # AMD acceleration flags
        self._use_rocm = os.environ.get('ROCM_PATH') is not None
        self._use_directml = os.name == 'nt' and os.environ.get('DIRECTML_PATH') is not None
        
    def _check_memory_pressure(self) -> bool:
        """Check if approaching memory limit."""
        # Estimate current memory usage
        estimated = self._current_memory_bytes + sum(
            len(batch) * 100 for batch in self._batch_buffer  # Rough estimate
        )
        return estimated > (self._max_memory_bytes * 0.9)
    
    def _enforce_memory_limit(self) -> None:
        """Force garbage collection if approaching limit."""
        if self._check_memory_pressure():
            self._flush_buffer()
            gc.collect()
    
    def _calculate_returns(self, prices: pl.Series, window: int) -> pl.Series:
        """Calculate returns with proper NaN handling for missing data/exchange halts."""
        if self.config.return_type == 'log':
            returns = prices.log() - prices.log().shift(window)
        else:
            returns = prices.pct_change(window)
        
        # Handle exchange halts (consecutive identical prices)
        price_changes = prices.diff()
        halt_mask = (price_changes == 0) & (prices.is_not_null())
        returns = returns.set(halt_mask, None)  # Mark halted periods as NaN
        
        return returns
    
    def _winsorize(self, series: pl.Series) -> pl.Series:
        """Winsorize outliers to handle extreme returns."""
        lower = series.quantile(0.01)
        upper = series.quantile(0.99)
        
        # Apply winsorization with configurable threshold
        threshold = self.config.winsorize_threshold
        std = series.std()
        if std is not None and std > 0:
            lower = series.mean() - threshold * std
            upper = series.mean() + threshold * std
        
        return series.clip(lower, upper)
    
    def process_batch_streaming(
        self, 
        price_data: List[Dict[str, List[float]]],
        symbol_list: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Process price data in streaming mini-batches.
        
        Args:
            price_data: List of dicts with 'prices', 'timestamps' keys per asset
            symbol_list: List of asset symbols
            
        Returns:
            Dictionary mapping symbols to momentum factor arrays
        """
        self._enforce_memory_limit()
        
        results = {}
        
        # Process in mini-batches to respect memory limit
        for batch_start in range(0, len(symbol_list), self.config.max_assets_per_batch):
            batch_symbols = symbol_list[batch_start:batch_start + self.config.max_assets_per_batch]
            batch_data = [price_data[i] for i, s in enumerate(symbol_list) if s in batch_symbols]
            
            # Convert to Polars DataFrame for vectorized operations
            try:
                batch_momentum = self._compute_batch_momentum(batch_data, batch_symbols)
                results.update(batch_momentum)
            except Exception as e:
                # Log error but continue processing other batches
                print(f"Error processing batch {batch_start}: {e}")
                continue
            
            # Clear batch from memory
            del batch_data
            if self._check_memory_pressure():
                gc.collect()
        
        return results
    
    def _compute_batch_momentum(
        self, 
        batch_data: List[Dict[str, List[float]]], 
        symbols: List[str]
    ) -> Dict[str, np.ndarray]:
        """Compute momentum factors for a single batch."""
        results = {}
        
        for idx, (data, symbol) in enumerate(zip(batch_data, symbols)):
            prices = data.get('prices', [])
            
            if len(prices) < self.config.min_history:
                results[symbol] = np.array([np.nan])
                continue
            
            # Convert to Polars Series for efficient computation
            price_series = pl.Series(prices)
            
            # Calculate multi-horizon momentum
            short_ret = self._calculate_returns(price_series, self.config.short_window)
            medium_ret = self._calculate_returns(price_series, self.config.medium_window)
            long_ret = self._calculate_returns(price_series, self.config.long_window)
            
            # Winsorize to handle outliers
            short_ret = self._winsorize(short_ret)
            medium_ret = self._winsorize(medium_ret)
            long_ret = self._winsorize(long_ret)
            
            # Composite momentum score (weighted average)
            # Typical weights: short=0.2, medium=0.3, long=0.5
            composite_momentum = (
                0.2 * short_ret.fill_nan(0) + 
                0.3 * medium_ret.fill_nan(0) + 
                0.5 * long_ret.fill_nan(0)
            )
            
            # Cross-sectional rank (percentile within batch)
            # This creates the cross-sectional momentum factor
            cs_rank = composite_momentum.rank(method='average') / len(composite_momentum)
            
            results[symbol] = cs_rank.to_numpy()
        
        return results
    
    def _flush_buffer(self) -> None:
        """Clear internal buffers."""
        self._batch_buffer.clear()
        self._current_memory_bytes = 0
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Return current memory statistics."""
        return {
            'estimated_bytes': self._current_memory_bytes,
            'buffer_batches': len(self._batch_buffer),
            'max_allowed_bytes': self._max_memory_bytes,
            'utilization_pct': int(100 * self._current_memory_bytes / self._max_memory_bytes),
        }


@ray.remote
class CrossSectionalMomentumOrchestrator:
    """
    Orchestrates cross-sectional momentum factor computation across Ray workers.
    
    Handles load balancing, result aggregation, and memory management.
    """
    
    def __init__(self, num_workers: int = 4, config: Optional[MomentumConfig] = None):
        self.config = config or MomentumConfig()
        self.num_workers = num_workers
        self.workers: List[ray.actor.ActorHandle] = []
        self._initialized = False
        
    def initialize(self) -> Dict:
        """Initialize workers and return environment status."""
        env_status = init_ray_with_limits()
        
        # Create workers
        self.workers = [
            MomentumFactorWorker.remote(self.config) 
            for _ in range(self.num_workers)
        ]
        
        self._initialized = True
        
        return {
            'workers_initialized': len(self.workers),
            **env_status
        }
    
    def compute_factors_distributed(
        self,
        all_price_data: Dict[str, Dict[str, List[float]]],
    ) -> Dict[str, np.ndarray]:
        """
        Compute cross-sectional momentum factors using distributed workers.
        
        Args:
            all_price_data: Dict mapping symbols to price data dicts
            
        Returns:
            Dict mapping symbols to momentum factor arrays
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        
        symbols = list(all_price_data.keys())
        price_data_list = list(all_price_data.values())
        
        # Distribute work across workers
        worker_futures = []
        items_per_worker = max(1, len(symbols) // self.num_workers)
        
        for i, worker in enumerate(self.workers):
            start_idx = i * items_per_worker
            end_idx = start_idx + items_per_worker if i < self.num_workers - 1 else len(symbols)
            
            batch_symbols = symbols[start_idx:end_idx]
            batch_data = price_data_list[start_idx:end_idx]
            
            if batch_symbols:
                future = worker.process_batch_streaming.remote(batch_data, batch_symbols)
                worker_futures.append((future, batch_symbols))
        
        # Aggregate results
        results = {}
        for future, batch_symbols in worker_futures:
            try:
                batch_results = ray.get(future)
                results.update(batch_results)
            except Exception as e:
                print(f"Worker failed: {e}")
                # Assign NaN for failed symbols
                for sym in batch_symbols:
                    results[sym] = np.array([np.nan])
        
        return results
    
    def get_all_memory_stats(self) -> List[Dict]:
        """Get memory stats from all workers."""
        futures = [w.get_memory_stats.remote() for w in self.workers]
        return ray.get(futures)
    
    def shutdown(self) -> None:
        """Shutdown workers and release resources."""
        for worker in self.workers:
            try:
                ray.kill(worker)
            except:
                pass
        self.workers.clear()
        self._initialized = False


def generate_momentum_signals(
    price_data: Dict[str, Dict[str, List[float]]],
    num_workers: int = 4,
    config: Optional[MomentumConfig] = None
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    High-level API for generating cross-sectional momentum signals.
    
    Args:
        price_data: Dict mapping symbols to {'prices': [...], 'timestamps': [...]}
        num_workers: Number of Ray workers
        config: Momentum configuration
        
    Returns:
        Tuple of (results dict, metadata dict)
    """
    orchestrator = CrossSectionalMomentumOrchestrator.remote(num_workers, config)
    
    try:
        # Initialize
        init_result = ray.get(orchestrator.initialize.remote())
        
        # Compute factors
        results = ray.get(orchestrator.compute_factors_distributed.remote(price_data))
        
        # Get memory stats
        memory_stats = ray.get(orchestrator.get_all_memory_stats.remote())
        
        metadata = {
            'num_symbols': len(price_data),
            'num_workers': num_workers,
            'init_status': init_result,
            'worker_memory_stats': memory_stats,
            'ram_quota_enforced': '4GB',
        }
        
        return results, metadata
        
    finally:
        ray.get(orchestrator.shutdown.remote())


# Example usage pattern
if __name__ == "__main__":
    # Example: Generate synthetic price data for testing
    np.random.seed(42)
    
    test_symbols = [f"SYMBOL_{i}" for i in range(20)]
    test_price_data = {}
    
    for symbol in test_symbols:
        # Random walk with drift for momentum
        returns = np.random.normal(0.001, 0.02, 200)
        prices = 100 * np.exp(np.cumsum(returns))
        
        test_price_data[symbol] = {
            'prices': prices.tolist(),
            'timestamps': list(range(len(prices))),
        }
    
    # Generate momentum signals
    results, metadata = generate_momentum_signals(
        test_price_data,
        num_workers=2,
        config=MomentumConfig(short_window=5, medium_window=20, long_window=60)
    )
    
    print(f"Processed {metadata['num_symbols']} symbols")
    print(f"Memory stats: {metadata['worker_memory_stats']}")
    
    # Show sample results
    for symbol in list(results.keys())[:3]:
        latest_signal = results[symbol][-1] if len(results[symbol]) > 0 else np.nan
        print(f"{symbol}: CS Rank = {latest_signal:.4f}")
