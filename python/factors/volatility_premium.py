"""
python/factors/volatility_premium.py

Realized-Implied Volatility Premium Calculator

Calculates the realized-implied volatility premium across the entire Binance universe,
utilizing Polars for lightning-fast vectorized cross-asset rankings. Optimized for
AMD Ryzen AI 5 with streaming mini-batches to enforce 4GB Python RAM quota.

Memory Constraint: Processes data in streaming batches, never loading full history.
Handles missing data and exchange halts gracefully.
"""

import ray
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Generator, Tuple
from dataclasses import dataclass
import os
import torch


def check_amd_acceleration() -> Dict[str, bool]:
    """Detect AMD ROCm/DirectML availability for PyTorch acceleration."""
    result = {
        "cuda": torch.cuda.is_available(),
        "rocm": False,
        "directml": False,
        "cpu": True,
    }
    
    if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
        result["rocm"] = True
    
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path and os.path.exists(rocm_path):
        result["rocm"] = True
    
    if os.name == 'nt':
        try:
            import torch_directml
            result["directml"] = True
        except ImportError:
            pass
    
    return result


@dataclass
class VolatilityPremiumConfig:
    """Configuration for volatility premium calculation."""
    realized_window_minutes: int = 60  # Window for realized vol
    implied_source: str = "options"  # or "perpetual_funding"
    max_universe_size: int = 500
    min_liquidity_usd: float = 1_000_000
    ram_limit_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB


@ray.remote(max_calls=100)
class VolatilityPremiumWorker:
    """
    Ray worker for computing realized-implied volatility premium.
    Uses Polars for vectorized operations with streaming batches.
    """
    
    def __init__(self, config: VolatilityPremiumConfig):
        self.config = config
        self.acceleration = check_amd_acceleration()
        self._price_buffer = []
        self._implied_buffer = []
        self._buffer_size = 0
        self._max_buffer_size = 5000
        
    def process_price_batch(
        self, 
        price_batch: pl.DataFrame,
        is_final: bool = False
    ) -> Optional[pl.DataFrame]:
        """
        Process streaming mini-batch of price data for realized vol calculation.
        
        Args:
            price_batch: Polars DataFrame with [symbol, timestamp, price, volume_usd]
            is_final: If True, flush remaining buffered data
            
        Returns:
            Volatility premium factors, or None if buffering
        """
        import psutil
        process = psutil.Process()
        current_ram = process.memory_info().rss
        
        if current_ram > self.config.ram_limit_bytes * 0.9:
            self._flush_buffers()
        
        self._price_buffer.append(price_batch)
        self._buffer_size += len(price_batch)
        
        if not is_final and self._buffer_size < self._max_buffer_size:
            return None
        
        return self._compute_volatility_premium()
    
    def set_implied_vol_batch(self, implied_batch: pl.DataFrame) -> bool:
        """Set implied volatility data from options or funding rates."""
        self._implied_buffer = [implied_batch]
        return True
    
    def _flush_buffers(self) -> None:
        """Clear all buffers."""
        self._price_buffer = []
        self._implied_buffer = []
        self._buffer_size = 0
    
    def _compute_volatility_premium(self) -> Optional[pl.DataFrame]:
        """
        Compute realized-implied volatility premium using Polars.
        Premium = Implied Vol - Realized Vol
        Positive premium = options expensive relative to realized movement
        """
        if not self._price_buffer:
            return None
        
        # Concatenate price batches
        prices = pl.concat(self._price_buffer, how="vertical")
        
        # Handle missing data gracefully
        prices = prices.sort(["symbol", "timestamp"])
        prices = prices.group_by("symbol").apply(
            lambda df: df.fill_null(strategy="forward_fill")
                       .fill_null(strategy="backward_fill")
        )
        
        # Calculate log returns
        prices = prices.with_columns([
            (pl.col("price").log() - pl.col("price").log().shift(1))
            .over("symbol").alias("log_return")
        ])
        
        # Compute realized volatility (rolling std of returns, annualized)
        window_size = self.config.realized_window_minutes
        prices = prices.with_columns([
            (pl.col("log_return").rolling_std(window_size=window_size).over("symbol"))
            .alias("realized_vol_raw")
        ])
        
        # Annualize (assuming minute data, 365*24*60 minutes per year)
        annualization_factor = np.sqrt(365 * 24 * 60)
        prices = prices.with_columns([
            (pl.col("realized_vol_raw") * annualization_factor).alias("realized_vol")
        ])
        
        # Get implied vol from buffer
        if self._implied_buffer:
            implied = pl.concat(self._implied_buffer, how="vertical")
            
            # Join realized and implied
            combined = prices.join(
                implied.select(["symbol", "timestamp", "implied_vol"]),
                on=["symbol", "timestamp"],
                how="left"
            )
            
            # Forward fill implied vol if sparse
            combined = combined.with_columns([
                pl.col("implied_vol").fill_null(strategy="forward_fill")
            ])
        else:
            # Use perpetual funding as proxy for implied vol
            # Simplified: implied_vol ≈ |funding_rate| * sqrt(annualization)
            combined = prices.with_columns([
                (pl.col("funding_rate").abs() * annualization_factor)
                .alias("implied_vol")
            ] if "funding_rate" in prices.columns else [
                pl.lit(0.0).alias("implied_vol")
            ])
        
        # Calculate volatility premium
        combined = combined.with_columns([
            (pl.col("implied_vol") - pl.col("realized_vol")).alias("vol_premium")
        ])
        
        # Cross-sectional ranking (z-score)
        combined = combined.with_columns([
            ((pl.col("vol_premium") - pl.col("vol_premium").mean()) /
             (pl.col("vol_premium").std() + 1e-8)).alias("vol_premium_rank")
        ])
        
        # Filter by liquidity
        combined = combined.filter(
            pl.col("volume_usd") >= self.config.min_liquidity_usd
        )
        
        # Limit universe size
        combined = combined.sort("vol_premium_rank", reverse=True)
        combined = combined.head(self.config.max_universe_size)
        
        return combined
    
    def get_acceleration_info(self) -> Dict[str, bool]:
        """Return detected acceleration backend info."""
        return self.acceleration


@ray.remote
class VolatilityPremiumOrchestrator:
    """
    Orchestrates volatility premium calculation across multiple Ray workers.
    Manages batch distribution and cross-asset ranking aggregation.
    """
    
    def __init__(self, config: VolatilityPremiumConfig, num_workers: int = 4):
        self.config = config
        self.num_workers = num_workers
        self.workers = [
            VolatilityPremiumWorker.remote(config)
            for _ in range(num_workers)
        ]
        
    def process_streaming(
        self,
        price_stream: Generator[pl.DataFrame, None, None],
        implied_stream: Optional[Generator[pl.DataFrame, None, None]] = None,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Process streaming price and implied vol data.
        
        Yields:
            DataFrames with computed volatility premium factors
        """
        batch_idx = 0
        
        for price_batch in price_stream:
            worker_idx = batch_idx % self.num_workers
            worker = self.workers[worker_idx]
            
            # Process implied vol if available
            if implied_stream:
                try:
                    implied_batch = next(implied_stream)
                    ray.get(worker.set_implied_vol_batch.remote(implied_batch))
                except StopIteration:
                    pass
            
            result_id = worker.process_price_batch.remote(price_batch, is_final=False)
            
            try:
                result = ray.get(result_id, timeout=30)
                if result is not None:
                    yield result
            except Exception as e:
                print(f"Worker error: {e}")
            
            batch_idx += 1
        
        # Final flush
        for worker in self.workers:
            try:
                result_id = worker.process_price_batch.remote(
                    pl.DataFrame(), is_final=True
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


def calculate_volatility_premium_universe(
    price_data_source,
    implied_data_source=None,
    realized_window: int = 60,
) -> Generator[pl.DataFrame, None, None]:
    """
    Create streaming volatility premium calculation pipeline.
    
    Args:
        price_data_source: Generator yielding price DataFrames
        implied_data_source: Optional generator for implied vol data
        realized_window: Window in minutes for realized vol calculation
        
    Yields:
        DataFrames with volatility premium rankings
    """
    config = VolatilityPremiumConfig(realized_window_minutes=realized_window)
    orchestrator = VolatilityPremiumOrchestrator.remote(config)
    
    if not ray.is_initialized():
        ray.init(
            _system_config={
                "max_io_worker_cpu_use": 0.5,
                "min_worker_size": 4 * 1024 * 1024 * 1024,
            }
        )
    
    for result in ray.get(
        orchestrator.process_streaming.remote(price_data_source, implied_data_source)
    ):
        yield result
    
    ray.get(orchestrator.shutdown.remote())


if __name__ == "__main__":
    print("Volatility Premium Module - AMD Acceleration:", check_amd_acceleration())
    config = VolatilityPremiumConfig()
    print(f"Config: realized_window={config.realized_window_minutes}min")
