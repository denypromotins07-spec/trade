"""
python/factors/orderflow_toxicity.py

Cross-Sectional Order Flow Toxicity Aggregator

Aggregates VPIN (Volume-Synchronized Probability of Informed Trading) and toxicity scores
cross-sectionally to identify which specific altcoins are experiencing the highest informed
trading pressure. Uses streaming mini-batches to enforce 4GB Python RAM quota.

Memory Constraint: Processes order flow in streaming batches with Polars vectorization.
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
class ToxicityConfig:
    """Configuration for order flow toxicity calculation."""
    bucket_size: int = 1000  # Number of trades per VPIN bucket
    num_buckets: int = 50    # Number of buckets for VPIN calculation
    lookback_windows: List[int] = None  # Windows for multi-scale toxicity
    max_universe_size: int = 500
    min_liquidity_usd: float = 1_000_000
    ram_limit_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [5, 15, 60]  # minutes


@ray.remote(max_calls=100)
class OrderFlowToxicityWorker:
    """
    Ray worker for computing VPIN and order flow toxicity scores.
    Uses streaming mini-batches to stay within 4GB RAM quota.
    """
    
    def __init__(self, config: ToxicityConfig):
        self.config = config
        self.acceleration = check_amd_acceleration()
        self._trade_buffer = {}  # Per-symbol buffers
        self._total_trades = 0
        self._max_total_trades = 100000
        
    def process_trade_batch(
        self,
        trade_batch: pl.DataFrame,
        is_final: bool = False
    ) -> Optional[pl.DataFrame]:
        """
        Process streaming mini-batch of trade data for VPIN calculation.
        
        Args:
            trade_batch: Polars DataFrame with [symbol, timestamp, price, volume, side]
                         side: +1 for buy, -1 for sell
            is_final: If True, flush remaining buffered data
            
        Returns:
            Toxicity scores including VPIN, or None if buffering
        """
        import psutil
        process = psutil.Process()
        current_ram = process.memory_info().rss
        
        if current_ram > self.config.ram_limit_bytes * 0.9:
            self._flush_buffers()
        
        # Add to per-symbol buffers
        for symbol in trade_batch["symbol"].unique():
            symbol_data = trade_batch.filter(pl.col("symbol") == symbol)
            if symbol not in self._trade_buffer:
                self._trade_buffer[symbol] = []
            self._trade_buffer[symbol].append(symbol_data)
        
        self._total_trades += len(trade_batch)
        
        if not is_final and self._total_trades < self._max_total_trades:
            return None
        
        return self._compute_toxicity_scores()
    
    def _flush_buffers(self) -> None:
        """Clear all buffers."""
        self._trade_buffer = {}
        self._total_trades = 0
    
    def _compute_toxicity_scores(self) -> Optional[pl.DataFrame]:
        """
        Compute VPIN and cross-sectional toxicity scores.
        
        VPIN Formula:
        VPIN = (1/n) * sum(|V_buy - V_sell|) / (V_buy + V_sell)
        
        Higher VPIN indicates more informed trading (toxicity).
        """
        if not self._trade_buffer:
            return None
        
        results = []
        
        for symbol, batches in self._trade_buffer.items():
            if not batches:
                continue
            
            # Concatenate all batches for this symbol
            trades = pl.concat(batches, how="vertical")
            
            # Handle missing data
            trades = trades.sort("timestamp").fill_null(strategy="forward_fill")
            
            # Calculate VPIN using volume buckets
            vpin_result = self._calculate_vpin(trades, symbol)
            if vpin_result is not None:
                results.append(vpin_result)
        
        if not results:
            return None
        
        # Combine into single DataFrame
        combined = pl.concat(results, how="vertical")
        
        # Cross-sectional ranking
        for window in self.config.lookback_windows:
            col = f"vpin_{window}m"
            if col in combined.columns:
                rank_col = f"toxicity_rank_{window}m"
                combined = combined.with_columns([
                    ((pl.col(col) - pl.col(col).mean()) /
                     (pl.col(col).std() + 1e-8)).alias(rank_col)
                ])
        
        # Filter by liquidity
        if "volume_usd" in combined.columns:
            combined = combined.filter(
                pl.col("volume_usd") >= self.config.min_liquidity_usd
            )
        
        # Limit universe size and sort by toxicity
        if "toxicity_rank_60m" in combined.columns:
            combined = combined.sort("toxicity_rank_60m", reverse=True)
            combined = combined.head(self.config.max_universe_size)
        
        return combined
    
    def _calculate_vpin(
        self, 
        trades: pl.DataFrame, 
        symbol: str
    ) -> Optional[pl.DataFrame]:
        """
        Calculate VPIN for a single symbol using volume bucketing.
        """
        if len(trades) < self.config.bucket_size:
            return None
        
        # Classify trades as buy/sell using tick rule
        # Buy if price > previous price, Sell if price < previous price
        trades = trades.with_columns([
            (pl.col("price") - pl.col("price").shift(1)).alias("price_change")
        ])
        
        trades = trades.with_columns([
            pl.when(pl.col("price_change") > 0)
            .then(1)
            .when(pl.col("price_change") < 0)
            .then(-1)
            .otherwise(0)
            .alias("side_inferred")
        ])
        
        # Use provided side if available, otherwise inferred
        if "side" in trades.columns:
            trades = trades.with_columns([
                pl.col("side").fill_null(pl.col("side_inferred"))
            ])
        else:
            trades = trades.with_columns([
                pl.col("side_inferred").alias("side")
            ])
        
        # Calculate buy and sell volumes
        trades = trades.with_columns([
            pl.when(pl.col("side") > 0)
            .then(pl.col("volume"))
            .otherwise(0)
            .alias("buy_volume"),
            pl.when(pl.col("side") < 0)
            .then(pl.col("volume"))
            .otherwise(0)
            .alias("sell_volume"),
        ])
        
        # Aggregate into buckets
        bucket_size = self.config.bucket_size
        num_buckets = min(self.config.num_buckets, len(trades) // bucket_size)
        
        if num_buckets < 2:
            return None
        
        vpin_values = []
        
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size
            
            bucket = trades.slice(start_idx, bucket_size)
            
            total_buy = bucket["buy_volume"].sum()
            total_sell = bucket["sell_volume"].sum()
            total_volume = total_buy + total_sell
            
            if total_volume > 0:
                vpin = abs(total_buy - total_sell) / total_volume
                vpin_values.append(vpin)
        
        if not vpin_values:
            return None
        
        # Average VPIN across buckets
        avg_vpin = np.mean(vpin_values)
        std_vpin = np.std(vpin_values) if len(vpin_values) > 1 else 0
        
        # Get latest timestamp and volume
        latest = trades.tail(1)
        latest_ts = latest["timestamp"][0] if len(latest) > 0 else 0
        latest_vol = latest["volume"].sum() if "volume" in latest.columns else 0
        
        return pl.DataFrame({
            "symbol": [symbol],
            "timestamp": [latest_ts],
            "vpin": [avg_vpin],
            "vpin_std": [std_vpin],
            "num_buckets": [num_buckets],
            "volume_usd": [latest_vol],
        })
    
    def get_acceleration_info(self) -> Dict[str, bool]:
        """Return detected acceleration backend info."""
        return self.acceleration


@ray.remote
class OrderFlowToxicityOrchestrator:
    """
    Orchestrates toxicity calculation across multiple Ray workers.
    Manages batch distribution and cross-sectional aggregation.
    """
    
    def __init__(self, config: ToxicityConfig, num_workers: int = 4):
        self.config = config
        self.num_workers = num_workers
        self.workers = [
            OrderFlowToxicityWorker.remote(config)
            for _ in range(num_workers)
        ]
        
    def process_streaming(
        self,
        trade_stream: Generator[pl.DataFrame, None, None],
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Process streaming trade data for toxicity calculation.
        
        Yields:
            DataFrames with VPIN and toxicity rankings
        """
        batch_idx = 0
        
        for trade_batch in trade_stream:
            worker_idx = batch_idx % self.num_workers
            worker = self.workers[worker_idx]
            
            result_id = worker.process_trade_batch.remote(trade_batch, is_final=False)
            
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
                result_id = worker.process_trade_batch.remote(
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


def calculate_orderflow_toxicity(
    trade_data_source: Generator[pl.DataFrame, None, None],
    bucket_size: int = 1000,
    num_buckets: int = 50,
) -> Generator[pl.DataFrame, None, None]:
    """
    Create streaming order flow toxicity calculation pipeline.
    
    Args:
        trade_data_source: Generator yielding trade DataFrames
        bucket_size: Number of trades per VPIN bucket
        num_buckets: Number of buckets for VPIN calculation
        
    Yields:
        DataFrames with VPIN and toxicity rankings
    """
    config = ToxicityConfig(bucket_size=bucket_size, num_buckets=num_buckets)
    orchestrator = OrderFlowToxicityOrchestrator.remote(config)
    
    if not ray.is_initialized():
        ray.init(
            _system_config={
                "max_io_worker_cpu_use": 0.5,
                "min_worker_size": 4 * 1024 * 1024 * 1024,
            }
        )
    
    for result in ray.get(orchestrator.process_streaming.remote(trade_data_source)):
        yield result
    
    ray.get(orchestrator.shutdown.remote())


if __name__ == "__main__":
    print("Order Flow Toxicity Module - AMD Acceleration:", check_amd_acceleration())
    config = ToxicityConfig()
    print(f"Config: bucket_size={config.bucket_size}, num_buckets={config.num_buckets}")
