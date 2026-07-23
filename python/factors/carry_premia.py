"""
python/factors/carry_premia.py

High-Frequency Crypto Carry and Roll-Down Premia Calculator

Calculates carry and roll-down premia on Ray workers using streaming mini-batches
to strictly enforce the 4GB Python RAM quota. Processes funding rates in chunks.

Memory Constraint: Streaming batches, never loads full history into memory.
AMD ROCm/DirectML acceleration checks included.
"""

import ray
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
import os
import torch


def check_amd_acceleration() -> Dict[str, bool]:
    """Detect AMD ROCm/DirectML availability."""
    result = {"cuda": torch.cuda.is_available(), "rocm": False, "directml": False, "cpu": True}
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
class CarryConfig:
    """Configuration for carry premia calculation."""
    funding_rate_window: int = 60  # Minutes for funding rate avg
    roll_lookback_days: int = 7
    max_universe_size: int = 500
    min_liquidity_usd: float = 1_000_000
    ram_limit_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB quota


@ray.remote(max_calls=100)
class CarryPremiaWorker:
    """
    Ray worker for computing carry and roll-down premia.
    Uses streaming mini-batches to stay within 4GB RAM quota.
    """
    
    def __init__(self, config: CarryConfig):
        self.config = config
        self.acceleration = check_amd_acceleration()
        self._funding_buffer = []
        self._basis_buffer = []
        self._buffer_size = 0
        self._max_buffer_size = 5000
        
    def process_funding_batch(
        self, 
        funding_batch: pl.DataFrame,
        is_final: bool = False
    ) -> Optional[pl.DataFrame]:
        """
        Process streaming mini-batch of funding rate data.
        
        Args:
            funding_batch: Polars DataFrame with [symbol, timestamp, funding_rate]
            is_final: If True, flush remaining buffered data
            
        Returns:
            Carry premia factors, or None if buffering
        """
        import psutil
        process = psutil.Process()
        current_ram = process.memory_info().rss
        
        if current_ram > self.config.ram_limit_bytes * 0.9:
            self._flush_buffers()
        
        self._funding_buffer.append(funding_batch)
        self._buffer_size += len(funding_batch)
        
        if not is_final and self._buffer_size < self._max_buffer_size:
            return None
        
        return self._compute_carry_premia()
    
    def set_basis_data(self, basis_batch: pl.DataFrame) -> bool:
        """Set basis/futures spread data for roll-down calculation."""
        self._basis_buffer = [basis_batch]
        return True
    
    def _flush_buffers(self) -> None:
        """Clear all buffers."""
        self._funding_buffer = []
        self._basis_buffer = []
        self._buffer_size = 0
    
    def _compute_carry_premia(self) -> Optional[pl.DataFrame]:
        """
        Compute carry and roll-down premia.
        
        Carry = Annualized funding rate
        Roll-down = Return from futures converging to spot
        """
        if not self._funding_buffer:
            return None
        
        # Concatenate funding batches
        funding = pl.concat(self._funding_buffer, how="vertical")
        
        # Handle missing data
        funding = funding.sort(["symbol", "timestamp"])
        funding = funding.group_by("symbol").apply(
            lambda df: df.fill_null(strategy="forward_fill")
                       .fill_null(strategy="backward_fill")
        )
        
        # Calculate average funding rate (annualized)
        # Funding typically every 8 hours, so multiply by 3*365
        annualization_factor = 3 * 365
        
        carry = (
            funding
            .group_by("symbol")
            .agg([
                pl.col("funding_rate")
                .rolling_mean(window_size=self.config.funding_rate_window)
                .alias("avg_funding"),
                pl.col("funding_rate")
                .rolling_std(window_size=self.config.funding_rate_window)
                .alias("funding_vol"),
            ])
        )
        
        # Annualize carry
        carry = carry.with_columns([
            (pl.col("avg_funding") * annualization_factor).alias("carry_annualized")
        ])
        
        # Add roll-down if basis data available
        if self._basis_buffer:
            basis = pl.concat(self._basis_buffer, how="vertical")
            
            # Roll-down = basis / days_to_expiry * 365
            roll_down = basis.with_columns([
                ((pl.col("basis_pct") / pl.col("days_to_expiry")) * 365)
                .alias("roll_down_annualized")
            ])
            
            # Join carry and roll-down
            combined = carry.join(
                roll_down.select(["symbol", "timestamp", "roll_down_annualized"]),
                on=["symbol", "timestamp"],
                how="left"
            )
        else:
            combined = carry
            combined = combined.with_columns([
                pl.lit(0.0).alias("roll_down_annualized")
            ])
        
        # Total premia
        combined = combined.with_columns([
            (pl.col("carry_annualized") + pl.col("roll_down_annualized"))
            .alias("total_carry_premia")
        ])
        
        # Cross-sectional ranking
        combined = combined.with_columns([
            ((pl.col("total_carry_premia") - pl.col("total_carry_premia").mean()) /
             (pl.col("total_carry_premia").std() + 1e-8))
            .alias("carry_rank")
        ])
        
        # Filter by liquidity
        if "volume_usd" in combined.columns:
            combined = combined.filter(
                pl.col("volume_usd") >= self.config.min_liquidity_usd
            )
        
        # Limit universe
        combined = combined.sort("carry_rank", reverse=True)
        combined = combined.head(self.config.max_universe_size)
        
        return combined
    
    def get_acceleration_info(self) -> Dict[str, bool]:
        """Return detected acceleration backend info."""
        return self.acceleration


@ray.remote
class CarryPremiaOrchestrator:
    """
    Orchestrates carry premia calculation across multiple Ray workers.
    Manages batch distribution and result aggregation.
    """
    
    def __init__(self, config: CarryConfig, num_workers: int = 4):
        self.config = config
        self.num_workers = num_workers
        self.workers = [
            CarryPremiaWorker.remote(config)
            for _ in range(num_workers)
        ]
        
    def process_streaming(
        self,
        funding_stream: Generator[pl.DataFrame, None, None],
        basis_stream: Optional[Generator[pl.DataFrame, None, None]] = None,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Process streaming funding and basis data.
        
        Yields:
            DataFrames with computed carry premia
        """
        batch_idx = 0
        
        for funding_batch in funding_stream:
            worker_idx = batch_idx % self.num_workers
            worker = self.workers[worker_idx]
            
            # Process basis if available
            if basis_stream:
                try:
                    basis_batch = next(basis_stream)
                    ray.get(worker.set_basis_data.remote(basis_batch))
                except StopIteration:
                    pass
            
            result_id = worker.process_funding_batch.remote(funding_batch, is_final=False)
            
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
                result_id = worker.process_funding_batch.remote(
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


def calculate_carry_premia_universe(
    funding_data_source,
    basis_data_source=None,
    funding_window: int = 60,
) -> Generator[pl.DataFrame, None, None]:
    """
    Create streaming carry premia calculation pipeline.
    
    Args:
        funding_data_source: Generator yielding funding rate DataFrames
        basis_data_source: Optional generator for basis data
        funding_window: Window in minutes for funding rate averaging
        
    Yields:
        DataFrames with carry premia rankings
    """
    config = CarryConfig(funding_rate_window=funding_window)
    orchestrator = CarryPremiaOrchestrator.remote(config)
    
    if not ray.is_initialized():
        ray.init(
            _system_config={
                "max_io_worker_cpu_use": 0.5,
                "min_worker_size": 4 * 1024 * 1024 * 1024,
            }
        )
    
    for result in ray.get(
        orchestrator.process_streaming.remote(funding_data_source, basis_data_source)
    ):
        yield result
    
    ray.get(orchestrator.shutdown.remote())


if __name__ == "__main__":
    print("Carry Premia Module - AMD Acceleration:", check_amd_acceleration())
    config = CarryConfig()
    print(f"Config: funding_window={config.funding_rate_window}min, RAM limit=4GB")
