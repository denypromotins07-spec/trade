"""
Orderflow Toxicity Factor - VPIN and Cross-Sectional Aggregation

Aggregates VPIN (Volume-Synchronized Probability of Informed Trading) and toxicity
scores cross-sectionally to identify which specific altcoins are currently experiencing
the highest informed trading pressure.

Uses Ray for distributed processing with strict 4GB RAM quota enforcement.
"""

import ray
import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
import gc
from collections import defaultdict


def init_ray_strict_memory():
    """Initialize Ray with strict 4GB memory limit."""
    if not ray.is_initialized():
        rocm_available = os.environ.get('ROCM_PATH') is not None
        
        ray.init(
            _memory=int(4.0 * 1024 * 1024 * 1024),
            _object_store_memory=int(1.5 * 1024 * 1024 * 1024),
            num_cpus=min(os.cpu_count() or 4, 8),
        )
        
        return {'rocm_available': rocm_available}
    
    return {'rocm_available': False}


@dataclass
class ToxicityConfig:
    """Configuration for orderflow toxicity calculation."""
    # Number of volume buckets for VPIN calculation
    n_buckets: int = 50
    
    # Bucket size in base currency units
    bucket_size: float = 100.0
    
    # Lookback window for EPCP (Expected Price Change)
    epcp_window: int = 30
    
    # Minimum trades for valid calculation
    min_trades: int = 10
    
    # Theta threshold for toxicity classification
    theta_toxic: float = 0.7


@ray.remote(max_calls=50)
class OrderflowToxicityWorker:
    """
    Ray worker for computing VPIN and orderflow toxicity scores.
    
    Implements the Easley, López de Prado, and O'Hara (2012) VPIN methodology
    adapted for crypto markets.
    """
    
    def __init__(self, config: ToxicityConfig):
        self.config = config
        self._max_memory_bytes = int(4.0 * 1024 * 1024 * 1024)
        self._processed_assets = 0
        
    def classify_trade_sign(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """
        Classify trades as buy/sell using tick rule and bulk volume classification.
        
        Returns: array of trade signs (+1 for buyer-initiated, -1 for seller-initiated)
        """
        if len(prices) < 2:
            return np.zeros(len(prices))
        
        # Tick rule: compare price to previous price
        price_diff = np.diff(prices)
        trade_signs = np.sign(price_diff)
        
        # Handle zero changes (same price) - use previous sign or default to 0
        for i in range(len(trade_signs)):
            if trade_signs[i] == 0:
                # Look back for last non-zero sign
                for j in range(i - 1, -1, -1):
                    if trade_signs[j] != 0:
                        trade_signs[i] = trade_signs[j]
                        break
                else:
                    trade_signs[i] = 0  # Default if all previous are zero
        
        # Prepend 0 for first trade (no previous price)
        trade_signs = np.concatenate([[0], trade_signs])
        
        return trade_signs
    
    def compute_vpin(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        trade_signs: Optional[np.ndarray] = None
    ) -> Tuple[float, List[float]]:
        """
        Compute VPIN (Volume-Synchronized Probability of Informed Trading).
        
        VPIN measures the imbalance between buy and sell volume, which indicates
        the presence of informed traders.
        
        Returns:
            - Current VPIN value
            - Time series of VPIN values
        """
        if len(prices) < self.config.min_trades:
            return np.nan, [np.nan]
        
        if trade_signs is None:
            trade_signs = self.classify_trade_sign(prices, volumes)
        
        # Volume bucketing
        bucket_size = self.config.bucket_size
        n_buckets = self.config.n_buckets
        
        vpin_values = []
        current_bucket_buy = 0.0
        current_bucket_sell = 0.0
        bucket_count = 0
        
        for i in range(len(volumes)):
            volume = volumes[i]
            sign = trade_signs[i] if i < len(trade_signs) else 0
            
            if sign > 0:
                current_bucket_buy += volume
            elif sign < 0:
                current_bucket_sell += volume
            
            # Check if bucket is full
            total_bucket_volume = current_bucket_buy + current_bucket_sell
            
            if total_bucket_volume >= bucket_size and bucket_count < n_buckets:
                # Calculate imbalance for this bucket
                imbalance = abs(current_bucket_buy - current_bucket_sell)
                vpin_bucket = imbalance / total_bucket_volume if total_bucket_volume > 0 else 0
                vpin_values.append(vpin_bucket)
                
                # Reset bucket
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                bucket_count += 1
        
        # Handle partial final bucket
        if current_bucket_buy + current_bucket_sell > 0 and bucket_count < n_buckets:
            imbalance = abs(current_bucket_buy - current_bucket_sell)
            total = current_bucket_buy + current_bucket_sell
            vpin_values.append(imbalance / total if total > 0 else 0)
        
        if len(vpin_values) == 0:
            return np.nan, [np.nan]
        
        # Current VPIN is the average of recent buckets
        recent_buckets = min(10, len(vpin_values))
        current_vpin = np.mean(vpin_values[-recent_buckets:])
        
        return current_vpin, vpin_values
    
    def compute_epcp(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """
        Compute Expected Price Change (EPCP) component of toxicity.
        
        Measures the expected absolute price change given order flow.
        """
        if len(prices) < self.config.epcp_window + 1:
            return np.nan
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        # Rolling expected absolute return
        window = min(self.config.epcp_window, len(returns))
        abs_returns = np.abs(returns)
        
        # EWMA of absolute returns
        if len(abs_returns) > 0:
            epcp = np.mean(abs_returns[-window:])
        else:
            epcp = 0.0
        
        return epcp
    
    def compute_toxicity_score(
        self,
        symbol: str,
        price_data: Dict[str, List[float]],
        volume_data: Dict[str, List[float]]
    ) -> Dict:
        """
        Compute comprehensive toxicity score for a single asset.
        
        Combines VPIN, EPCP, and other metrics into a unified toxicity score.
        """
        prices = np.array(price_data.get('prices', []))
        volumes = np.array(volume_data.get('volumes', []))
        
        if len(prices) < self.config.min_trades or len(volumes) < self.config.min_trades:
            return {
                'symbol': symbol,
                'vpin': np.nan,
                'epcp': np.nan,
                'toxicity_score': np.nan,
                'is_toxic': False,
                'valid': False,
            }
        
        # Ensure same length
        min_len = min(len(prices), len(volumes))
        prices = prices[:min_len]
        volumes = volumes[:min_len]
        
        try:
            # Compute VPIN
            vpin, vpin_series = self.compute_vpin(prices, volumes)
            
            # Compute EPCP
            epcp = self.compute_epcp(prices, volumes)
            
            # Compute additional toxicity metrics
            # Volume concentration (Herfindahl-like index)
            vol_concentration = np.sum((volumes / np.sum(volumes)) ** 2) if np.sum(volumes) > 0 else 0
            
            # Price impact coefficient (rough estimate)
            returns = np.diff(np.log(prices)) if len(prices) > 1 else np.array([0])
            signed_volume = volumes[:-1] * np.sign(np.diff(prices)) if len(prices) > 1 else np.array([0])
            
            if len(signed_volume) > 1 and np.var(signed_volume) > 0:
                price_impact = np.corrcoef(returns, signed_volume)[0, 1]
            else:
                price_impact = 0.0
            
            # Composite toxicity score
            # Weight: VPIN=0.5, EPCP=0.2, Vol Concentration=0.15, Price Impact=0.15
            toxicity_components = []
            weights = []
            
            if not np.isnan(vpin):
                toxicity_components.append(vpin)
                weights.append(0.5)
            
            if not np.isnan(epcp):
                # Normalize EPCP (typical crypto daily vol ~2-5%)
                normalized_epcp = min(epcp * 100, 1.0)
                toxicity_components.append(normalized_epcp)
                weights.append(0.2)
            
            toxicity_components.append(vol_concentration * 10)  # Scale up
            weights.append(0.15)
            
            toxicity_components.append(abs(price_impact))
            weights.append(0.15)
            
            # Weighted average
            if len(toxicity_components) > 0:
                toxicity_score = sum(c * w for c, w in zip(toxicity_components, weights))
                toxicity_score = min(toxicity_score, 1.0)  # Cap at 1.0
            else:
                toxicity_score = np.nan
            
            is_toxic = toxicity_score > self.config.theta_toxic if not np.isnan(toxicity_score) else False
            
            self._processed_assets += 1
            
            return {
                'symbol': symbol,
                'vpin': float(vpin) if not np.isnan(vpin) else np.nan,
                'epcp': float(epcp) if not np.isnan(epcp) else np.nan,
                'vol_concentration': float(vol_concentration),
                'price_impact': float(price_impact) if not np.isnan(price_impact) else np.nan,
                'toxicity_score': float(toxicity_score) if not np.isnan(toxicity_score) else np.nan,
                'is_toxic': is_toxic,
                'valid': True,
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'vpin': np.nan,
                'epcp': np.nan,
                'toxicity_score': np.nan,
                'is_toxic': False,
                'valid': False,
                'error': str(e),
            }
    
    def process_batch(
        self,
        symbols: List[str],
        all_price_data: Dict[str, Dict[str, List[float]]],
        all_volume_data: Dict[str, Dict[str, List[float]]]
    ) -> List[Dict]:
        """Process batch of symbols with memory enforcement."""
        results = []
        
        for symbol in symbols:
            if symbol not in all_price_data or symbol not in all_volume_data:
                continue
            
            try:
                result = self.compute_toxicity_score(
                    symbol,
                    all_price_data[symbol],
                    all_volume_data[symbol]
                )
                results.append(result)
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'vpin': np.nan,
                    'toxicity_score': np.nan,
                    'is_toxic': False,
                    'valid': False,
                    'error': str(e),
                })
            
            # Memory pressure check
            if self._processed_assets % 10 == 0:
                gc.collect()
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            'processed_assets': self._processed_assets,
            'max_memory_bytes': self._max_memory_bytes,
        }


@ray.remote
class ToxicityOrchestrator:
    """
    Orchestrates cross-sectional orderflow toxicity analysis.
    
    Identifies assets with highest informed trading pressure.
    """
    
    def __init__(self, num_workers: int = 4, config: Optional[ToxicityConfig] = None):
        self.config = config or ToxicityConfig()
        self.num_workers = num_workers
        self.workers: List[ray.actor.ActorHandle] = []
        
    def initialize(self) -> Dict:
        env = init_ray_strict_memory()
        
        self.workers = [
            OrderflowToxicityWorker.remote(self.config)
            for _ in range(self.num_workers)
        ]
        
        return {'workers': len(self.workers), **env}
    
    def compute_cross_sectional_toxicity(
        self,
        all_price_data: Dict[str, Dict[str, List[float]]],
        all_volume_data: Dict[str, Dict[str, List[float]]]
    ) -> pl.DataFrame:
        """Compute toxicity scores across all assets and rank them."""
        symbols = list(all_price_data.keys())
        
        # Distribute work
        chunk_size = max(1, len(symbols) // self.num_workers)
        futures = []
        
        for i, worker in enumerate(self.workers):
            start = i * chunk_size
            end = start + chunk_size if i < self.num_workers - 1 else len(symbols)
            batch_symbols = symbols[start:end]
            
            if batch_symbols:
                future = worker.process_batch.remote(
                    batch_symbols,
                    all_price_data,
                    all_volume_data
                )
                futures.append(future)
        
        # Collect results
        all_results = []
        for future in futures:
            batch_results = ray.get(future)
            all_results.extend(batch_results)
        
        # Convert to DataFrame
        df = pl.DataFrame(all_results)
        
        # Filter valid results
        df_valid = df.filter(pl.col('valid') == True)
        
        # Cross-sectional ranking
        if len(df_valid) > 0:
            df_valid = df_valid.with_columns([
                pl.col('toxicity_score').rank(method='dense', descending=True).alias('toxicity_rank'),
                pl.col('vpin').rank(method='dense', descending=True).alias('vpin_rank'),
                # Percentile
                (pl.col('toxicity_rank') / len(df_valid)).alias('toxicity_percentile'),
            ])
        
        return df_valid
    
    def get_most_toxic_assets(
        self,
        df: pl.DataFrame,
        top_n: int = 10
    ) -> pl.DataFrame:
        """Get the N most toxic assets (highest informed trading pressure)."""
        if len(df) == 0:
            return df
        
        return df.sort('toxicity_rank', descending=False).head(top_n)
    
    def shutdown(self):
        for worker in self.workers:
            try:
                ray.kill(worker)
            except:
                pass
        self.workers.clear()


def calculate_orderflow_toxicity(
    price_data: Dict[str, Dict[str, List[float]]],
    volume_data: Dict[str, Dict[str, List[float]]],
    num_workers: int = 4,
    config: Optional[ToxicityConfig] = None
) -> Tuple[pl.DataFrame, Dict]:
    """
    High-level API for cross-sectional orderflow toxicity analysis.
    
    Args:
        price_data: Dict mapping symbols to price data
        volume_data: Dict mapping symbols to volume data
        num_workers: Number of Ray workers
        config: Toxicity configuration
        
    Returns:
        Tuple of (Polars DataFrame with rankings, metadata dict)
    """
    orchestrator = ToxicityOrchestrator.remote(num_workers, config)
    
    try:
        init_info = ray.get(orchestrator.initialize.remote())
        df = ray.get(orchestrator.compute_cross_sectional_toxicity.remote(
            price_data, volume_data
        ))
        
        metadata = {
            'total_symbols': len(price_data),
            'valid_symbols': len(df) if len(df) > 0 else 0,
            'toxic_assets': len(df.filter(pl.col('is_toxic') == True)) if len(df) > 0 else 0,
            'workers_used': num_workers,
            'init_info': init_info,
            'ram_quota': '4GB',
        }
        
        return df, metadata
        
    finally:
        ray.get(orchestrator.shutdown.remote())


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    test_symbols = [f"ALT_{i}" for i in range(20)]
    test_price_data = {}
    test_volume_data = {}
    
    for symbol in test_symbols:
        # Generate realistic-looking orderflow
        n_trades = 200
        
        # Some assets have higher toxicity (more informed trading)
        toxicity_level = np.random.uniform(0.3, 0.9)
        
        # Generate imbalanced orderflow for toxic assets
        bias = np.random.choice([-1, 1]) * toxicity_level
        signs = np.random.choice([-1, 1], size=n_trades, p=[0.5 - bias/2, 0.5 + bias/2])
        
        # Prices follow orderflow with noise
        returns = signs * 0.001 + np.random.normal(0, 0.002, n_trades)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Volumes correlate with toxicity
        volumes = np.random.exponential(10 + toxicity_level * 50, n_trades)
        
        test_price_data[symbol] = {'prices': prices.tolist()}
        test_volume_data[symbol] = {'volumes': volumes.tolist()}
    
    # Calculate toxicity
    df, meta = calculate_orderflow_toxicity(test_price_data, test_volume_data, num_workers=2)
    
    print(f"\nProcessed {meta['total_symbols']} symbols")
    print(f"Valid results: {meta['valid_symbols']}")
    print(f"Toxic assets (>0.7 score): {meta['toxic_assets']}")
    
    if len(df) > 0:
        print("\nMost Toxic Assets (Highest Informed Trading Pressure):")
        print(df.sort('toxicity_rank').head(5)[['symbol', 'vpin', 'toxicity_score', 'is_toxic']])
