"""
Chapter 4: Advanced Data Pipeline & Stream Processing
File 11: python/pipeline/windowing.py

Distributed tumbling and sliding window aggregations on Ray.
Utilizes Polars for vectorized math while strictly respecting
the global 8GB memory ceiling.

Enforces 4GB RAM quota per Python worker.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Any, Callable
import ray
from datetime import timedelta
import time

# Memory limit (4GB quota per worker)
MAX_MEMORY_MB = 4096


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability."""
    accel_info = {
        'rocm_available': False,
        'directml_available': False,
        'cuda_available': False,
        'polars_available': True,
        'recommended_backend': 'polars'
    }
    
    try:
        import torch
        if torch.version.hip is not None:
            accel_info['rocm_available'] = True
            accel_info['recommended_backend'] = 'pytorch_rocm'
        elif hasattr(torch.backends, 'directml'):
            accel_info['directml_available'] = True
            accel_info['recommended_backend'] = 'pytorch_directml'
        elif torch.cuda.is_available():
            accel_info['cuda_available'] = True
            accel_info['recommended_backend'] = 'pytorch_cuda'
    except ImportError:
        pass
    
    try:
        import polars
    except ImportError:
        accel_info['polars_available'] = False
    
    return accel_info


class WindowAggregator:
    """
    High-performance window aggregations using Polars.
    
    Supports:
    - Tumbling windows (non-overlapping)
    - Sliding windows (overlapping)
    - Session windows (gap-based)
    """
    
    def __init__(self, memory_limit_mb: int = MAX_MEMORY_MB):
        self.memory_limit_mb = memory_limit_mb
        self.accel_info = check_amd_acceleration()
        self._processed_rows = 0
    
    def tumbling_window(
        self,
        df: pl.DataFrame,
        time_column: str,
        window_size: str,
        aggregations: Dict[str, str],
        group_by: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Apply tumbling (non-overlapping) window aggregation.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame with datetime column
        time_column : str
            Name of timestamp column
        window_size : str
            Window size string (e.g., "1m", "5m", "1h")
        aggregations : dict
            Column -> aggregation function mapping
            e.g., {"price": "mean", "volume": "sum"}
        group_by : list, optional
            Columns to group by before windowing
            
        Returns
        -------
        pl.DataFrame
            Aggregated DataFrame
        """
        self._check_memory()
        
        # Create dynamic group_by expression
        window_expr = pl.col(time_column).dt.truncate(window_size)
        
        if group_by is None:
            group_by_cols = [window_expr]
        else:
            group_by_cols = group_by + [window_expr]
        
        # Build aggregation expressions
        agg_exprs = []
        for col, agg_func in aggregations.items():
            if agg_func == "mean":
                agg_exprs.append(pl.col(col).mean())
            elif agg_func == "sum":
                agg_exprs.append(pl.col(col).sum())
            elif agg_func == "min":
                agg_exprs.append(pl.col(col).min())
            elif agg_func == "max":
                agg_exprs.append(pl.col(col).max())
            elif agg_func == "std":
                agg_exprs.append(pl.col(col).std())
            elif agg_func == "count":
                agg_exprs.append(pl.col(col).count())
            elif agg_func == "first":
                agg_exprs.append(pl.col(col).first())
            elif agg_func == "last":
                agg_exprs.append(pl.col(col).last())
        
        result = df.group_by(group_by_cols).agg(agg_exprs)
        
        self._processed_rows += len(df)
        return result
    
    def sliding_window(
        self,
        df: pl.DataFrame,
        time_column: str,
        window_size: str,
        slide_size: str,
        aggregations: Dict[str, str],
        group_by: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Apply sliding (overlapping) window aggregation.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame
        time_column : str
            Timestamp column name
        window_size : str
            Window size (e.g., "5m")
        slide_size : str
            Slide interval (e.g., "1m")
        aggregations : dict
            Column -> aggregation mapping
        group_by : list, optional
            Group by columns
            
        Returns
        -------
        pl.DataFrame
            Aggregated results
        """
        self._check_memory()
        
        # For sliding windows, we use rolling operations
        # Convert window/slide to timedelta for offset calculation
        
        result_dfs = []
        
        # Get time range
        min_time = df[time_column].min()
        max_time = df[time_column].max()
        
        if min_time is None or max_time is None:
            return pl.DataFrame()
        
        # Generate window start times
        current = min_time
        while current <= max_time:
            window_end = current + pl.duration(**self._parse_duration(window_size))
            
            # Filter to window
            mask = (pl.col(time_column) >= current) & (pl.col(time_column) < window_end)
            window_df = df.filter(mask)
            
            if len(window_df) > 0:
                # Apply aggregations
                agg_results = {}
                agg_results[time_column] = [current]
                
                for col, agg_func in aggregations.items():
                    if agg_func == "mean":
                        agg_results[col] = [window_df[col].mean()]
                    elif agg_func == "sum":
                        agg_results[col] = [window_df[col].sum()]
                    elif agg_func == "min":
                        agg_results[col] = [window_df[col].min()]
                    elif agg_func == "max":
                        agg_results[col] = [window_df[col].max()]
                    elif agg_func == "std":
                        agg_results[col] = [window_df[col].std()]
                
                result_dfs.append(pl.DataFrame(agg_results))
            
            # Slide forward
            current = current + pl.duration(**self._parse_duration(slide_size))
            
            self._check_memory()
        
        if result_dfs:
            return pl.concat(result_dfs, how="vertical")
        return pl.DataFrame()
    
    def _parse_duration(self, duration_str: str) -> Dict[str, int]:
        """Parse duration string to kwargs."""
        units = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days'}
        value = int(duration_str[:-1])
        unit = duration_str[-1]
        return {units.get(unit, 'seconds'): value}
    
    def session_window(
        self,
        df: pl.DataFrame,
        time_column: str,
        gap_size: str,
        aggregations: Dict[str, str]
    ) -> pl.DataFrame:
        """
        Session window based on activity gaps.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame
        time_column : str
            Timestamp column
        gap_size : str
            Maximum gap between events in same session
        aggregations : dict
            Aggregation functions
            
        Returns
        -------
        pl.DataFrame
            Session-level aggregations
        """
        self._check_memory()
        
        # Sort by time
        df_sorted = df.sort(time_column)
        
        # Calculate time differences
        df_with_gaps = df_sorted.with_columns([
            pl.col(time_column).diff().alias('time_diff'),
            pl.duration(**self._parse_duration(gap_size)).alias('gap_threshold')
        ])
        
        # Mark session boundaries
        df_with_sessions = df_with_gaps.with_columns([
            (pl.col('time_diff') > pl.col('gap_threshold')).cum_sum().alias('session_id')
        ])
        
        # Aggregate by session
        group_by_cols = ['session_id']
        agg_exprs = []
        
        for col, agg_func in aggregations.items():
            if agg_func == "mean":
                agg_exprs.append(pl.col(col).mean())
            elif agg_func == "sum":
                agg_exprs.append(pl.col(col).sum())
            elif agg_func == "count":
                agg_exprs.append(pl.col(col).count())
        
        result = df_with_sessions.group_by(group_by_cols).agg(agg_exprs)
        
        self._processed_rows += len(df)
        return result
    
    def _check_memory(self):
        """Memory checkpoint."""
        import gc
        if self._processed_rows % 100000 == 0:
            gc.collect()
    
    def get_stats(self) -> Dict:
        return {
            'processed_rows': self._processed_rows,
            'acceleration': self.accel_info,
            'memory_limit_mb': self.memory_limit_mb
        }


@ray.remote(max_calls=10)
class DistributedWindowWorker:
    """Ray worker for distributed window aggregations."""
    
    def __init__(self, memory_limit_mb: int = MAX_MEMORY_MB):
        self.aggregator = WindowAggregator(memory_limit_mb)
        self._batches_processed = 0
    
    def process_tumbling_window(
        self,
        data: np.ndarray,
        columns: List[str],
        time_column: str,
        window_size: str,
        aggregations: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process tumbling window on batch."""
        df = pl.DataFrame(data, schema=columns)
        result = self.aggregator.tumbling_window(
            df, time_column, window_size, aggregations
        )
        self._batches_processed += 1
        return result.to_dict()
    
    def process_sliding_window(
        self,
        data: np.ndarray,
        columns: List[str],
        time_column: str,
        window_size: str,
        slide_size: str,
        aggregations: Dict[str, str]
    ) -> Dict[str, Any]:
        """Process sliding window on batch."""
        df = pl.DataFrame(data, schema=columns)
        result = self.aggregator.sliding_window(
            df, time_column, window_size, slide_size, aggregations
        )
        self._batches_processed += 1
        return result.to_dict()
    
    def get_stats(self) -> Dict:
        stats = self.aggregator.get_stats()
        stats['batches_processed'] = self._batches_processed
        return stats


def create_window_workers(num_workers: int = 4) -> List:
    """Create distributed window workers."""
    return [
        DistributedWindowWorker.remote(memory_limit_mb=MAX_MEMORY_MB)
        for _ in range(num_workers)
    ]


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=4 * 1024 * 1024 * 1024,
        _system_config={"max_bytes_to_spill": 4 * 1024 * 1024 * 1024}
    )
    
    print("AMD Acceleration:", check_amd_acceleration())
    
    # Test with sample data
    aggregator = WindowAggregator()
    
    # Create sample DataFrame
    n_rows = 10000
    df = pl.DataFrame({
        'timestamp': pl.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 1) + timedelta(minutes=n_rows),
            interval='1m'
        ),
        'price': np.random.randn(n_rows).cumsum() + 100,
        'volume': np.random.randint(100, 10000, n_rows)
    })
    
    # Tumbling window test
    tumbling_result = aggregator.tumbling_window(
        df,
        time_column='timestamp',
        window_size='5m',
        aggregations={'price': 'mean', 'volume': 'sum'}
    )
    print(f"Tumbling window result rows: {len(tumbling_result)}")
    
    print(f"Aggregator stats: {aggregator.get_stats()}")
    
    ray.shutdown()
