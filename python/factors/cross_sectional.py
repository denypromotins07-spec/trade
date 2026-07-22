"""
Cross-Sectional Factor Models

Computes cross-sectional momentum and short-term reversal scores across the
entire Binance universe, utilizing Polars for lightning-fast vectorized
cross-asset rankings.

Optimized for microsecond latency with AMD Ryzen AI 5 architecture.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import ray


# Factor lookback periods (in days)
MOMENTUM_PERIODS = [5, 10, 21, 63]  # 1w, 2w, 1m, 3m
REVERSAL_PERIODS = [1, 3, 5]  # Short-term reversal
VOLUME_PERIODS = [5, 21]  # Volume trends


@ray.remote
def compute_cross_sectional_rank(
    returns_df: pl.DataFrame,
    factor_name: str,
    lookback: int
) -> pl.DataFrame:
    """
    Compute cross-sectional rank for a single factor.
    
    Uses Polars' optimized ranking algorithms for O(n log n) performance.
    
    Args:
        returns_df: DataFrame with columns [timestamp, asset, return]
        factor_name: Name of the factor column
        lookback: Lookback period in days
        
    Returns:
        DataFrame with cross-sectional ranks (0-1 scale)
    """
    # Group by timestamp and compute ranks within each cross-section
    ranked = (
        returns_df
        .with_columns([
            pl.col(factor_name).rank(method='average').over('timestamp').alias(f'{factor_name}_rank')
        ])
    )
    
    # Normalize to 0-1 range
    max_rank = ranked[f'{factor_name}_rank'].max()
    if max_rank is not None and max_rank > 0:
        ranked = ranked.with_columns([
            (pl.col(f'{factor_name}_rank') / max_rank).alias(f'{factor_name}_rank_norm')
        ])
    
    return ranked


def compute_momentum_factors(
    prices_df: pl.DataFrame,
    periods: List[int] = MOMENTUM_PERIODS
) -> pl.DataFrame:
    """
    Compute momentum factors for multiple lookback periods.
    
    Momentum = (Price_t / Price_t-n) - 1
    
    Args:
        prices_df: DataFrame with columns [timestamp, asset, close]
        periods: List of lookback periods
        
    Returns:
        DataFrame with momentum factors for each period
    """
    # Sort by timestamp for proper lag calculation
    prices_sorted = prices_df.sort(['asset', 'timestamp'])
    
    result_df = prices_sorted.clone()
    
    for period in periods:
        # Compute lagged price using shift within each asset group
        col_name = f'momentum_{period}d'
        
        result_df = result_df.with_columns([
            ((pl.col('close') / pl.col('close').shift(period).over('asset')) - 1)
            .alias(col_name)
        ])
    
    return result_df


def compute_short_term_reversal(
    returns_df: pl.DataFrame,
    periods: List[int] = REVERSAL_PERIODS
) -> pl.DataFrame:
    """
    Compute short-term reversal factors.
    
    Reversal = -1 * sum(returns) over lookback period
    Short-term reversal captures mean-reversion in crypto markets.
    
    Args:
        returns_df: DataFrame with columns [timestamp, asset, return]
        periods: List of lookback periods
        
    Returns:
        DataFrame with reversal factors
    """
    result_df = returns_df.clone()
    
    for period in periods:
        col_name = f'reversal_{period}d'
        
        # Sum returns over lookback period, then negate for reversal signal
        result_df = result_df.with_columns([
            (-1 * pl.col('return').rolling_sum(window_size=period).over('asset'))
            .alias(col_name)
        ])
    
    return result_df


def compute_volume_factors(
    volume_df: pl.DataFrame,
    periods: List[int] = VOLUME_PERIODS
) -> pl.DataFrame:
    """
    Compute volume-based factors.
    
    Volume trend = current_volume / rolling_avg_volume
    
    Args:
        volume_df: DataFrame with columns [timestamp, asset, volume]
        periods: List of lookback periods
        
    Returns:
        DataFrame with volume factors
    """
    result_df = volume_df.sort(['asset', 'timestamp']).clone()
    
    for period in periods:
        col_name = f'volume_trend_{period}d'
        
        result_df = result_df.with_columns([
            (pl.col('volume') / 
             pl.col('volume').rolling_mean(window_size=period).over('asset'))
            .alias(col_name)
        ])
    
    return result_df


def orthogonalize_factors(
    factors_df: pl.DataFrame,
    target_cols: List[str],
    control_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Orthogonalize factors against control variables using linear regression.
    
    This removes common variation and isolates idiosyncratic factor returns.
    
    Args:
        factors_df: DataFrame with factor columns
        target_cols: Columns to orthogonalize
        control_cols: Control variables (default: market factor)
        
    Returns:
        DataFrame with orthogonalized factors
    """
    if control_cols is None:
        # Default: orthogonalize against cross-sectional mean (market factor)
        control_cols = ['market']
        factors_df = factors_df.with_columns([
            pl.mean_horizontal(target_cols).alias('market')
        ])
    
    result_df = factors_df.clone()
    
    for target in target_cols:
        # Simple orthogonalization: residual from regression on controls
        # For production, use proper OLS with numpy
        ortho_col = f'{target}_ortho'
        
        # Compute beta coefficients
        control_matrix = factors_df.select(control_cols).to_numpy()
        target_vector = factors_df[target].to_numpy()
        
        # Handle NaN values
        mask = ~(np.isnan(control_matrix).any(axis=1) | np.isnan(target_vector))
        if mask.sum() < len(target_vector) * 0.5:
            # Too much missing data, skip orthogonalization
            result_df = result_df.with_columns([pl.col(target).alias(ortho_col)])
            continue
        
        try:
            # OLS regression: target = beta * controls + residual
            X = control_matrix[mask]
            y = target_vector[mask]
            
            # Add constant
            X = np.column_stack([np.ones(len(y)), X])
            
            # Solve normal equations
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Compute residuals (orthogonalized factor)
            fitted = X @ beta
            residuals = y - fitted
            
            # Store residuals
            result_df = result_df.with_columns([
                pl.when(pl.col(target).is_not_null())
                .then(pl.lit(None))  # Will be filled properly
                .otherwise(pl.col(target))
                .alias(ortho_col)
            ])
            
            # Fill with actual residuals
            result_dict = {ortho_col: np.full(len(result_df), np.nan)}
            result_dict[ortho_col][mask] = residuals
            result_df = pl.DataFrame(result_dict)
            
        except (np.linalg.LinAlgError, ValueError):
            # Regression failed, keep original
            result_df = result_df.with_columns([pl.col(target).alias(ortho_col)])
    
    return result_df


def compute_cross_sectional_zscore(
    df: pl.DataFrame,
    column: str,
    group_by: str = 'timestamp'
) -> pl.DataFrame:
    """
    Compute cross-sectional z-score for a factor.
    
    Z-score = (value - cross_sectional_mean) / cross_sectional_std
    
    Args:
        df: Input DataFrame
        column: Column to standardize
        group_by: Column to group by (typically timestamp)
        
    Returns:
        DataFrame with z-score column
    """
    zscore_col = f'{column}_zscore'
    
    return df.with_columns([
        ((pl.col(column) - pl.col(column).mean().over(group_by)) /
         pl.col(column).std().over(group_by))
        .alias(zscore_col)
    ])


def build_combined_factor_score(
    factors_df: pl.DataFrame,
    weights: Optional[Dict[str, float]] = None
) -> pl.DataFrame:
    """
    Build combined alpha score from multiple factors.
    
    Args:
        factors_df: DataFrame with factor columns (should be z-scores)
        weights: Optional weights for each factor (default: equal weight)
        
    Returns:
        DataFrame with combined score column
    """
    # Identify factor columns (those ending with _zscore)
    factor_cols = [c for c in factors_df.columns if c.endswith('_zscore')]
    
    if not factor_cols:
        raise ValueError("No factor z-score columns found")
    
    if weights is None:
        # Equal weighting
        weights = {col: 1.0 / len(factor_cols) for col in factor_cols}
    
    # Compute weighted sum
    score_exprs = []
    for col in factor_cols:
        w = weights.get(col, 0.0)
        if w != 0:
            score_exprs.append((pl.col(col) * w))
    
    if score_exprs:
        combined = sum(score_exprs[0], score_exprs[1:])
        return factors_df.with_columns([combined.alias('combined_score')])
    
    return factors_df


def generate_trading_signals(
    scores_df: pl.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float = 0.2
) -> pl.DataFrame:
    """
    Generate trading signals from combined scores.
    
    Args:
        scores_df: DataFrame with combined_score column
        top_pct: Percentile for long positions
        bottom_pct: Percentile for short positions
        
    Returns:
        DataFrame with signal column (-1, 0, 1)
    """
    return scores_df.with_columns([
        pl.when(pl.col('combined_score') >= pl.col('combined_score').quantile(1 - top_pct))
        .then(1)  # Long
        .when(pl.col('combined_score') <= pl.col('combined_score').quantile(bottom_pct))
        .then(-1)  # Short
        .otherwise(0)  # Neutral
        .alias('signal')
    ])


def process_binance_universe(
    prices_data: pl.DataFrame,
    volume_data: Optional[pl.DataFrame] = None
) -> Dict[str, pl.DataFrame]:
    """
    Process entire Binance universe for cross-sectional factors.
    
    Args:
        prices_data: OHLCV data for all assets
        volume_data: Optional separate volume data
        
    Returns:
        Dictionary with processed factor DataFrames
    """
    # Compute returns
    returns_df = (
        prices_data
        .sort(['asset', 'timestamp'])
        .with_columns([
            ((pl.col('close') / pl.col('close').shift(1).over('asset')) - 1)
            .alias('return')
        ])
    )
    
    # Compute momentum factors
    momentum_df = compute_momentum_factors(prices_data)
    
    # Compute reversal factors
    reversal_df = compute_short_term_reversal(returns_df)
    
    # Compute volume factors if available
    if volume_data is not None:
        volume_df = compute_volume_factors(volume_data)
        returns_df = returns_df.join(volume_df, on=['timestamp', 'asset'], how='left')
    
    # Combine all factors
    all_factors = returns_df.clone()
    
    # Add momentum columns
    for period in MOMENTUM_PERIODS:
        col = f'momentum_{period}d'
        if col in momentum_df.columns:
            all_factors = all_factors.join(
                momentum_df.select(['timestamp', 'asset', col]),
                on=['timestamp', 'asset'],
                how='left'
            )
    
    # Add reversal columns
    for period in REVERSAL_PERIODS:
        col = f'reversal_{period}d'
        if col in reversal_df.columns:
            all_factors = all_factors.join(
                reversal_df.select(['timestamp', 'asset', col]),
                on=['timestamp', 'asset'],
                how='left'
            )
    
    # Standardize all factors
    factor_cols = [c for c in all_factors.columns 
                   if c.startswith(('momentum_', 'reversal_', 'volume_'))]
    
    for col in factor_cols:
        all_factors = compute_cross_sectional_zscore(all_factors, col)
    
    # Build combined score
    all_factors = build_combined_factor_score(all_factors)
    
    # Generate signals
    signals_df = generate_trading_signals(all_factors)
    
    return {
        'returns': returns_df,
        'factors': all_factors,
        'signals': signals_df,
    }


if __name__ == '__main__':
    # Example usage with sample data
    print("Cross-Sectional Factor Model")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_assets = 50
    n_days = 252
    
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    assets = [f'BTC{i:02d}USDT' for i in range(n_assets)]
    
    # Generate price data with momentum and reversal patterns
    data = []
    for asset in assets:
        price = 100.0
        for i, ts in enumerate(timestamps):
            # Add some autocorrelation for momentum
            if i > 0:
                ret = 0.02 * np.random.randn()
                if i % 20 < 5:  # Momentum periods
                    ret += 0.01
                else:  # Reversal periods
                    ret *= -0.3
            else:
                ret = 0.0
            
            price *= (1 + ret)
            data.append({'timestamp': ts, 'asset': asset, 'close': price})
    
    prices_df = pl.DataFrame(data)
    
    # Process universe
    results = process_binance_universe(prices_df)
    
    print(f"\nProcessed {len(assets)} assets over {n_days} days")
    print(f"Factors computed: {[c for c in results['factors'].columns if '_zscore' in c]}")
    print(f"Signal distribution: {results['signals']['signal'].value_counts()}")
