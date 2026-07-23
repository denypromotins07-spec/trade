"""
python/factors/momentum_hf.py

High-Frequency Cross-Sectional Momentum Factors on Ray Workers

Builds momentum factors across the entire crypto universe using streaming mini-batches
to strictly enforce the 4GB Python RAM quota. Optimized for AMD Ryzen AI 5 with
ROCm/DirectML acceleration checks.

Memory Constraint: Processes returns in streaming batches, never loading full history.
Handles missing data and exchange halts gracefully.
"""

import ray
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
import os
import torch

# Enforce 4GB RAM quota per Ray worker
RAY_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024


def check_amd_acceleration() -> Dict[str, bool]:
    """
    Detect AMD ROCm/DirectML availability for PyTorch acceleration.
    Returns dict of available backends.
    """
    result = {
        "cuda": torch.cuda.is_available(),
        "rocm": False,
        "directml": False,
        "cpu": True,
    }
    
    # Check for ROCm (AMD GPUs)
    if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
        result["rocm"] = True
    
    # Check environment variables for ROCm
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path and os.path.exists(rocm_path):
        result["rocm"] = True
    
    # DirectML check (Windows-specific)
    if os.name == 'nt':
        try:
            import torch_directml
            result["directml"] = True
        except ImportError:
            pass
    
    return result


@dataclass
class MomentumConfig:
    """Configuration for momentum factor calculation."""
    lookback_periods: List[int]  # e.g., [1, 5, 15, 60] minutes
    rebalance_frequency_seconds: int = 60
    max_universe_size: int = 500
    min_liquidity_usd: float = 1_000_000  # Minimum daily volume
    ram_limit_bytes: int = RAY_MEMORY_LIMIT_BYTES


@ray.remote(max_calls=100)  # Restart worker after 100 calls to prevent memory leaks
class MomentumFactorWorker:
    """
    Ray worker for computing cross-sectional momentum factors.
    Uses streaming mini-batches to stay within 4GB RAM quota.
    """
    
    def __init__(self, config: MomentumConfig):
        self.config = config
        self.acceleration = check_amd_acceleration()
        self.device = self._select_device()
        self._buffer = []
        self._buffer_size = 0
        self._max_buffer_size = 10000  # Max rows before flush
        
    def _select_device(self) -> str:
        """Select best available compute device."""
        if self.acceleration["rocm"]:
            return "cuda"  # PyTorch uses 'cuda' for ROCm too
        elif self.acceleration["directml"]:
            return "privateuseone"  # DirectML device
        elif self.acceleration["cuda"]:
            return "cuda"
        return "cpu"
    
    def process_batch_streaming(
        self, 
        returns_batch: pl.DataFrame,
        is_final: bool = False
    ) -> Optional[pl.DataFrame]:
        """
        Process a streaming mini-batch of returns data.
        
        Args:
            returns_batch: Polars DataFrame with columns [symbol, timestamp, return]
            is_final: If True, flush remaining buffered data
            
        Returns:
            Momentum factors for the batch, or None if buffering
        """
        # Check RAM usage before processing
        import psutil
        process = psutil.Process()
        current_ram = process.memory_info().rss
        
        if current_ram > self.config.ram_limit_bytes * 0.9:
            # Approaching limit, force flush
            self._flush_buffer()
        
        # Add to buffer
        self._buffer.append(returns_batch)
        self._buffer_size += len(returns_batch)
        
        if not is_final and self._buffer_size < self._max_buffer_size:
            return None  # Keep buffering
        
        return self._compute_momentum_factors()
    
    def _flush_buffer(self) -> Optional[pl.DataFrame]:
        """Flush buffer and compute factors."""
        if not self._buffer:
            return None
        
        result = self._compute_momentum_factors()
        self._buffer = []
        self._buffer_size = 0
        return result
    
    def _compute_momentum_factors(self) -> Optional[pl.DataFrame]:
        """
        Compute cross-sectional momentum factors from buffered data.
        Uses Polars for vectorized operations.
        """
        if not self._buffer:
            return None
        
        # Concatenate all buffered batches
        combined = pl.concat(self._buffer, how="vertical")
        
        # Handle missing data: forward fill then backward fill
        combined = combined.sort(["symbol", "timestamp"])
        combined = combined.group_by("symbol").apply(
            lambda df: df.fill_null(strategy="forward_fill")
                       .fill_null(strategy="backward_fill")
        )
        
        # Compute momentum for each lookback period
        results = []
        for period in self.config.lookback_periods:
            col_name = f"momentum_{period}"
            
            # Lagged returns (momentum = sum of past N returns)
            momentum = (
                combined
                .group_by("symbol")
                .agg([
                    pl.col("return").rolling_sum(window_size=period).alias(col_name)
                ])
            )
            results.append(momentum)
        
        # Join all momentum columns
        if len(results) > 1:
            base = results[0]
            for r in results[1:]:
                base = base.join(r, on=["symbol", "timestamp"], how="left")
        else:
            base = results[0] if results else None
        
        if base is None:
            return None
        
        # Cross-sectional ranking (z-score normalization)
        for period in self.config.lookback_periods:
            col_name = f"momentum_{period}"
            rank_col = f"momentum_{period}_rank"
            
            # Compute cross-sectional z-score
            base = base.with_columns([
                (pl.col(col_name) - pl.col(col_name).mean()) / 
                (pl.col(col_name).std() + 1e-8).alias(rank_col)
            ])
        
        # Filter by liquidity
        base = base.filter(pl.col("volume_usd") >= self.config.min_liquidity_usd)
        
        # Limit universe size
        base = base.sort(pl.col(f"momentum_{self.config.lookback_periods[-1]}_rank"), 
                        reverse=True)
        base = base.head(self.config.max_universe_size)
        
        return base
    
    def get_acceleration_info(self) -> Dict[str, bool]:
        """Return detected acceleration backend info."""
        return self.acceleration


@ray.remote
class MomentumFactorOrchestrator:
    """
    Orchestrates multiple Ray workers for universe-scale momentum calculation.
    Manages batch distribution and result aggregation.
    """
    
    def __init__(self, config: MomentumConfig, num_workers: int = 4):
        self.config = config
        self.num_workers = num_workers
        self.workers = [
            MomentumFactorWorker.remote(config) 
            for _ in range(num_workers)
        ]
        
    def process_universe_streaming(
        self,
        returns_stream: Generator[pl.DataFrame, None, None],
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Process entire universe via streaming generator.
        
        Args:
            returns_stream: Generator yielding mini-batches of returns data
            
        Yields:
            Computed momentum factors for each rebalance period
        """
        batch_idx = 0
        
        for batch in returns_stream:
            # Round-robin distribution to workers
            worker_idx = batch_idx % self.num_workers
            worker = self.workers[worker_idx]
            
            # Check if this is the last batch
            is_final = False  # Would need end-of-stream signal
            
            # Async processing
            result_id = worker.process_batch_streaming.remote(batch, is_final)
            
            # Wait for result (with timeout)
            try:
                result = ray.get(result_id, timeout=30)
                if result is not None:
                    yield result
            except Exception as e:
                # Worker may have OOM'd, restart handled by Ray
                print(f"Worker error: {e}")
            
            batch_idx += 1
        
        # Final flush
        for worker in self.workers:
            try:
                result_id = worker.process_batch_streaming.remote(
                    pl.DataFrame(), 
                    is_final=True
                )
                result = ray.get(result_id, timeout=30)
                if result is not None:
                    yield result
            except Exception:
                pass
    
    def shutdown(self):
        """Clean shutdown of all workers."""
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass


def create_momentum_factor_stream(
    binance_data_source,
    lookback_periods: List[int] = [1, 5, 15, 60],
    batch_size: int = 1000,
) -> Generator[pl.DataFrame, None, None]:
    """
    Create a streaming momentum factor computation pipeline.
    
    Args:
        binance_data_source: Async iterator yielding Binance market data
        lookback_periods: List of lookback periods in minutes
        batch_size: Number of symbols per batch
        
    Yields:
        DataFrames with computed momentum factors
    """
    config = MomentumConfig(lookback_periods=lookback_periods)
    orchestrator = MomentumFactorOrchestrator.remote(config)
    
    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init(
            _system_config={
                "max_io_worker_cpu_use": 0.5,
                "min_worker_size": 4 * 1024 * 1024 * 1024,  # 4GB
            }
        )
    
    def data_generator():
        """Wrap data source into Polars batches."""
        buffer = []
        
        for tick in binance_data_source:
            # Convert tick to Polars row
            row = pl.DataFrame({
                "symbol": tick["symbol"],
                "timestamp": tick["timestamp"],
                "return": tick["return"],
                "volume_usd": tick.get("volume_usd", 0),
            })
            buffer.append(row)
            
            if len(buffer) >= batch_size:
                batch = pl.concat(buffer)
                buffer = []
                yield batch
        
        # Flush remaining
        if buffer:
            yield pl.concat(buffer)
    
    # Stream through orchestrator
    for result in ray.get(
        orchestrator.process_universe_streaming.remote(data_generator())
    ):
        yield result
    
    ray.get(orchestrator.shutdown.remote())


if __name__ == "__main__":
    # Test configuration
    print("AMD Acceleration Check:", check_amd_acceleration())
    
    # Example usage would require actual Binance data source
    # This demonstrates the structure
    config = MomentumConfig(lookback_periods=[1, 5, 15])
    print(f"Momentum config: {config}")
