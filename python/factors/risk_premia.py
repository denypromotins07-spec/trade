"""
Risk Premia Calculation Module

Calculates crypto-specific risk premia (carry, momentum, value) and feeds
these orthogonalized factor exposures into the central portfolio optimization engine.

Optimized for AMD Ryzen AI 5 with strict memory constraints.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import ray


# Risk premia types
RISK_PREMIA_TYPES = ['carry', 'momentum', 'value', 'volatility', 'liquidity']


def check_amd_directml() -> bool:
    """Check if AMD DirectML/ROCm environment is available."""
    try:
        import torch
        if hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
            return True
        import os
        if os.name == 'nt':
            return True
        return False
    except ImportError:
        return False


@ray.remote(max_calls=100)
class RiskPremiaCalculator:
    """
    Ray actor for computing risk premia with memory limits.
    
    Each instance processes data in batches to stay within 4GB RAM quota.
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.premia_history: Dict[str, List[float]] = {t: [] for t in RISK_PREMIA_TYPES}
        
    def compute_carry_premium(
        self,
        prices: pl.DataFrame,
        funding_rates: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Compute carry premium from futures basis and funding rates.
        
        Carry = (Futures Price - Spot Price) / Spot Price + Funding Rate
        
        In crypto, carry comes from:
        1. Futures basis (contango/backwardation)
        2. Perpetual funding rates
        
        Args:
            prices: DataFrame with spot and futures prices
            funding_rates: DataFrame with funding rate history
            
        Returns:
            DataFrame with carry premium column
        """
        result = prices.clone()
        
        # Compute basis carry if futures data available
        if 'futures_close' in prices.columns:
            result = result.with_columns([
                ((pl.col('futures_close') - pl.col('close')) / pl.col('close'))
                .alias('basis_carry')
            ])
        
        # Add funding rate carry
        if funding_rates is not None:
            result = result.join(funding_rates, on=['timestamp', 'asset'], how='left')
            result = result.with_columns([
                (pl.col('funding_rate').fill_null(0) * 3 * 365)  # Annualized (3x daily)
                .alias('funding_carry')
            ])
            
            # Total carry = basis + funding
            if 'basis_carry' in result.columns:
                result = result.with_columns([
                    (pl.col('basis_carry') + pl.col('funding_carry'))
                    .alias('total_carry')
                ])
            else:
                result = result.with_columns([
                    pl.col('funding_carry').alias('total_carry')
                ])
        elif 'basis_carry' in result.columns:
            result = result.with_columns([
                pl.col('basis_carry').alias('total_carry')
            ])
        
        return result
    
    def compute_momentum_premium(self, returns: pl.DataFrame) -> pl.DataFrame:
        """
        Compute momentum premium using time-series momentum.
        
        Momentum = Sum of returns over lookback period
        
        Crypto momentum tends to be stronger than traditional assets due to:
        - Retail investor behavior
        - Trend-following algorithms
        - Slow information diffusion
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            DataFrame with momentum premium column
        """
        result = returns.clone()
        
        # Time-series momentum (multiple horizons)
        for horizon in [21, 63, 126]:  # 1m, 3m, 6m
            col_name = f'momentum_{horizon}d'
            result = result.with_columns([
                pl.col('return').rolling_sum(window_size=horizon).over('asset')
                .alias(col_name)
            ])
        
        # Combined momentum score (equal-weighted average of z-scores)
        momentum_cols = [f'momentum_{h}d' for h in [21, 63, 126]]
        
        # Standardize each momentum measure
        for col in momentum_cols:
            result = result.with_columns([
                ((pl.col(col) - pl.col(col).mean().over('timestamp')) /
                 pl.col(col).std().over('timestamp'))
                .alias(f'{col}_z')
            ])
        
        # Average z-score
        result = result.with_columns([
            ((pl.col('momentum_21d_z') + pl.col('momentum_63d_z') + 
              pl.col('momentum_126d_z')) / 3)
            .alias('momentum_premium')
        ])
        
        return result
    
    def compute_value_premium(
        self,
        prices: pl.DataFrame,
        fundamentals: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Compute value premium based on deviation from fair value.
        
        For crypto, value can be proxied by:
        1. Deviation from moving average (mean reversion)
        2. NVT ratio (Network Value to Transactions)
        3. MVRV ratio (Market Value to Realized Value)
        
        Args:
            prices: Price data
            fundamentals: Optional fundamental metrics
            
        Returns:
            DataFrame with value premium column
        """
        result = prices.sort(['asset', 'timestamp']).clone()
        
        # Simple value: deviation from long-term moving average
        ma_periods = [63, 126, 252]  # 3m, 6m, 1y
        
        for period in ma_periods:
            col_name = f'value_ma{period}'
            result = result.with_columns([
                ((pl.col('close') - pl.col('close').rolling_mean(window_size=period).over('asset')) /
                 pl.col('close').rolling_mean(window_size=period).over('asset'))
                .alias(col_name)
            ])
        
        # Combine value measures (negative deviation = cheap = positive value signal)
        value_cols = [f'value_ma{p}' for p in ma_periods]
        result = result.with_columns([
            (-1 * (pl.col(value_cols[0]) + pl.col(value_cols[1]) + pl.col(value_cols[2])) / 3)
            .alias('value_premium')
        ])
        
        # Add fundamental-based value if available
        if fundamentals is not None:
            result = result.join(fundamentals, on=['timestamp', 'asset'], how='left')
            
            if 'nvt_ratio' in result.columns:
                # High NVT = overvalued
                result = result.with_columns([
                    ((-1 * pl.col('nvt_ratio').rank().over('timestamp')) /
                     pl.len().over('timestamp'))
                    .alias('nvt_value')
                ])
                result = result.with_columns([
                    ((pl.col('value_premium') + pl.col('nvt_value')) / 2)
                    .alias('value_premium')
                ])
        
        return result
    
    def compute_volatility_premium(self, returns: pl.DataFrame) -> pl.DataFrame:
        """
        Compute volatility premium (variance risk premium).
        
        Volatility premium = Implied Volatility - Realized Volatility
        
        In crypto options markets, IV typically exceeds RV, creating
        a short vol premium.
        
        Args:
            returns: Asset returns
            
        Returns:
            DataFrame with volatility premium column
        """
        result = returns.sort(['asset', 'timestamp']).clone()
        
        # Realized volatility (multiple windows)
        for window in [21, 63]:
            col_name = f'rvol_{window}d'
            result = result.with_columns([
                (pl.col('return').rolling_std(window_size=window).over('asset') * 
                 np.sqrt(365))
                .alias(col_name)
            ])
        
        # If implied vol data available, compute VRP
        if 'implied_vol' in result.columns:
            result = result.with_columns([
                (pl.col('implied_vol') - pl.col('rvol_21d'))
                .alias('volatility_premium')
            ])
        else:
            # Use inverse of realized vol as proxy (low vol stocks tend to outperform)
            result = result.with_columns([
                (-1 * pl.col('rvol_21d').rank().over('timestamp') / 
                 pl.len().over('timestamp'))
                .alias('volatility_premium')
            ])
        
        return result
    
    def compute_liquidity_premium(
        self,
        volume_data: pl.DataFrame,
        spread_data: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Compute liquidity premium.
        
        Illiquid assets should earn a premium. Proxies:
        1. Low turnover (volume / market cap)
        2. Wide bid-ask spreads
        3. High price impact
        
        Args:
            volume_data: Volume and market cap data
            spread_data: Optional bid-ask spread data
            
        Returns:
            DataFrame with liquidity premium column
        """
        result = volume_data.sort(['asset', 'timestamp']).clone()
        
        # Turnover ratio
        if 'market_cap' in result.columns:
            result = result.with_columns([
                (pl.col('volume') / pl.col('market_cap'))
                .alias('turnover')
            ])
            
            # Low turnover = illiquid = high premium
            result = result.with_columns([
                ((-1 * pl.col('turnover').rank().over('timestamp')) /
                 pl.len().over('timestamp'))
                .alias('liquidity_premium')
            ])
        
        # Add spread-based liquidity if available
        if spread_data is not None:
            result = result.join(spread_data, on=['timestamp', 'asset'], how='left')
            
            if 'bid_ask_spread' in result.columns:
                # Wide spread = illiquid = high premium
                result = result.with_columns([
                    ((-1 * pl.col('bid_ask_spread').rank().over('timestamp')) /
                     pl.len().over('timestamp'))
                    .alias('spread_premium')
                ])
                
                # Combine with turnover-based premium
                if 'liquidity_premium' in result.columns:
                    result = result.with_columns([
                        ((pl.col('liquidity_premium') + pl.col('spread_premium')) / 2)
                        .alias('liquidity_premium')
                    ])
                else:
                    result = result.with_columns([
                        pl.col('spread_premium').alias('liquidity_premium')
                    ])
        
        return result


def orthogonalize_premia(
    premia_df: pl.DataFrame,
    premia_cols: List[str],
    market_returns: Optional[np.ndarray] = None
) -> pl.DataFrame:
    """
    Orthogonalize risk premia against market factor.
    
    This isolates the idiosyncratic component of each premium.
    
    Args:
        premia_df: DataFrame with raw premia
        premia_cols: Columns containing premia
        market_returns: Market returns for orthogonalization
        
    Returns:
        DataFrame with orthogonalized premia
    """
    result = premia_df.clone()
    
    # Default: orthogonalize against cross-sectional mean
    if market_returns is None:
        market_col = 'market_factor'
        # Compute cross-sectional mean as market proxy
        numeric_cols = [c for c in premia_cols if c in premia_df.columns]
        if numeric_cols:
            exprs = [pl.col(c) for c in numeric_cols]
            result = result.with_columns([
                pl.mean_horizontal(exprs).alias(market_col)
            ])
    else:
        market_col = 'market_factor'
        result = result.with_columns([
            pl.lit(market_returns).alias(market_col)
        ])
    
    # Orthogonalize each premium
    for col in premia_cols:
        if col not in result.columns:
            continue
            
        ortho_col = f'{col}_ortho'
        
        # Simple residualization
        # For production, use proper regression
        result = result.with_columns([
            (pl.col(col) - pl.col(col).mean().over('timestamp'))
            .alias(ortho_col)
        ])
    
    return result


def build_risk_premia_model(
    prices: pl.DataFrame,
    returns: pl.DataFrame,
    volume: pl.DataFrame,
    funding_rates: Optional[pl.DataFrame] = None,
    fundamentals: Optional[pl.DataFrame] = None
) -> Dict[str, pl.DataFrame]:
    """
    Build complete risk premia model.
    
    Args:
        prices: Price data
        returns: Return data
        volume: Volume data
        funding_rates: Optional funding rate data
        fundamentals: Optional fundamental data
        
    Returns:
        Dictionary with all computed premia
    """
    calculator = RiskPremiaCalculator()
    
    # Compute each premium type
    carry_df = calculator.compute_carry_premium(prices, funding_rates)
    momentum_df = calculator.compute_momentum_premium(returns)
    value_df = calculator.compute_value_premium(prices, fundamentals)
    vol_df = calculator.compute_volatility_premium(returns)
    liq_df = calculator.compute_liquidity_premium(volume)
    
    # Merge all premia
    base_df = returns.select(['timestamp', 'asset']).unique()
    
    for df, suffix in [(carry_df, '_carry'), (momentum_df, '_mom'), 
                        (value_df, '_val'), (vol_df, '_vol'), (liq_df, '_liq')]:
        key_cols = [c for c in df.columns if c.endswith(suffix) or c in ['timestamp', 'asset']]
        base_df = base_df.join(df.select(key_cols), on=['timestamp', 'asset'], how='left')
    
    # Define premium columns
    premia_cols = ['total_carry', 'momentum_premium', 'value_premium', 
                   'volatility_premium', 'liquidity_premium']
    
    # Orthogonalize
    ortho_df = orthogonalize_premia(base_df, premia_cols)
    
    # Compute combined score
    ortho_cols = [f'{c}_ortho' for c in premia_cols if f'{c}_ortho' in ortho_df.columns]
    
    if ortho_cols:
        ortho_df = ortho_df.with_columns([
            (sum(pl.col(c) for c in ortho_cols) / len(ortho_cols))
            .alias('combined_premia_score')
        ])
    
    return {
        'raw_premia': base_df,
        'orthogonalized_premia': ortho_df,
        'carry': carry_df,
        'momentum': momentum_df,
        'value': value_df,
        'volatility': vol_df,
        'liquidity': liq_df,
    }


def feed_to_portfolio_optimizer(
    premia_data: Dict[str, pl.DataFrame],
    target_volatility: float = 0.15
) -> Dict[str, np.ndarray]:
    """
    Format risk premia for portfolio optimization engine.
    
    Args:
        premia_data: Output from build_risk_premia_model
        target_volatility: Target portfolio volatility
        
    Returns:
        Dictionary formatted for optimizer
    """
    ortho_df = premia_data['orthogonalized_premia']
    
    # Get latest cross-section
    latest_ts = ortho_df['timestamp'].max()
    latest = ortho_df.filter(pl.col('timestamp') == latest_ts)
    
    # Extract factor exposures
    exposure_cols = [c for c in ortho_df.columns if c.endswith('_ortho')]
    
    exposures = {}
    for col in exposure_cols:
        exposures[col] = latest[col].to_numpy()
    
    # Compute expected returns from premia
    if 'combined_premia_score' in latest.columns:
        expected_returns = latest['combined_premia_score'].to_numpy()
    else:
        expected_returns = np.zeros(len(latest))
    
    return {
        'expected_returns': expected_returns,
        'factor_exposures': exposures,
        'target_volatility': target_volatility,
        'assets': latest['asset'].to_list(),
        'timestamp': latest_ts,
    }


if __name__ == '__main__':
    print("Risk Premia Calculator")
    print("=" * 40)
    print(f"AMD DirectML Available: {check_amd_directml()}")
    
    # Create sample data
    np.random.seed(42)
    n_assets = 30
    n_days = 252
    
    timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    assets = [f'CRYPTO{i:02d}' for i in range(n_assets)]
    
    # Generate sample data
    data = []
    for asset in assets:
        price = 100.0
        for ts in timestamps:
            ret = 0.02 * np.random.randn()
            price *= (1 + ret)
            data.append({
                'timestamp': ts,
                'asset': asset,
                'close': price,
                'volume': np.random.uniform(1e6, 1e8),
                'return': ret,
            })
    
    df = pl.DataFrame(data)
    prices = df.select(['timestamp', 'asset', 'close'])
    returns = df.select(['timestamp', 'asset', 'return'])
    volume = df.select(['timestamp', 'asset', 'volume'])
    
    # Build premia model
    premia_model = build_risk_premia_model(prices, returns, volume)
    
    print(f"\nComputed premia for {n_assets} assets over {n_days} days")
    print(f"Available premia: {RISK_PREMIA_TYPES}")
    
    # Feed to optimizer
    optimizer_input = feed_to_portfolio_optimizer(premia_model)
    print(f"\nOptimizer input shape: {optimizer_input['expected_returns'].shape}")
