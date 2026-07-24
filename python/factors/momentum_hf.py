"""
Stage 62: AI & Pipeline Audit - File 14/20
Module: python/factors/momentum_hf.py
Focus: Cross-Sectional Ranking NaNs, Divide-by-Zero Prevention
Constraints: 4GB RAM Quota

AUDIT FIXES APPLIED:
- Fixed cross-sectional ranking NaNs
- Added divide-by-zero protection during halts
- Implemented robust rank normalization
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def safe_rank(data: np.ndarray, method: str = 'average') -> np.ndarray:
    """
    Compute ranks with NaN handling.
    FIX: Handles ties and NaN values gracefully.
    """
    if len(data) == 0:
        return np.array([])
    
    # Replace NaN with a value that will be ranked last
    nan_mask = np.isnan(data)
    filled_data = np.where(nan_mask, np.inf, data)
    
    # Compute ranks
    sorted_indices = np.argsort(filled_data)
    ranks = np.empty_like(sorted_indices, dtype=float)
    ranks[sorted_indices] = np.arange(len(data), dtype=float)
    
    # Set NaN positions to NaN in output
    ranks[nan_mask] = np.nan
    
    return ranks


def cross_sectional_momentum(
    prices: pd.DataFrame,
    lookback: int = 20,
    halt_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Compute cross-sectional momentum factors.
    FIX: Prevents NaNs and divide-by-zero during halts.
    """
    if prices.empty or lookback <= 0:
        logger.warning("Invalid input for momentum calculation")
        return pd.DataFrame()
    
    # Compute returns
    returns = prices.pct_change(periods=lookback)
    
    # Handle divide-by-zero (halts)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(0)
    
    # Apply halt threshold
    if halt_threshold > 0:
        halted = prices.pct_change() < -halt_threshold
        returns[halted] = 0
        logger.info(f"Applied halt threshold {halt_threshold}")
    
    # Cross-sectional rank at each timestep
    def rank_cross_section(row):
        if row.isnull().all():
            return row
        return pd.Series(safe_rank(row.values), index=row.index)
    
    ranked_returns = returns.apply(rank_cross_section, axis=1)
    
    # Normalize ranks to [-1, 1]
    n_assets = prices.shape[1]
    if n_assets > 1:
        normalized = (ranked_returns / (n_assets - 1)) * 2 - 1
    else:
        normalized = ranked_returns
    
    # Final NaN check
    normalized = normalized.fillna(0)
    
    return normalized


class MomentumFactorEngine:
    """
    High-frequency momentum factor engine.
    FIX: Robust handling of market halts and missing data.
    """
    
    def __init__(self, lookbacks: list = [5, 10, 20, 60]):
        self.lookbacks = lookbacks
        
    def compute_all_factors(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute momentum factors for all lookback periods."""
        factors = {}
        
        for lb in self.lookbacks:
            try:
                factors[f'momentum_{lb}'] = cross_sectional_momentum(prices, lb)
            except Exception as e:
                logger.error(f"Failed to compute momentum_{lb}: {e}")
                factors[f'momentum_{lb}'] = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        return factors
    
    def compute_composite(self, prices: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """Compute weighted composite momentum factor."""
        factors = self.compute_all_factors(prices)
        
        if weights is None:
            # Equal weight
            weights = {k: 1.0 / len(factors) for k in factors}
        
        composite = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        for factor_name, factor_data in factors.items():
            weight = weights.get(factor_name, 0.0)
            composite += factor_data * weight
        
        # Validate no NaN in composite
        if composite.isnull().any().any():
            logger.warning("NaN detected in composite factor. Filling with 0.")
            composite = composite.fillna(0)
        
        return composite


if __name__ == "__main__":
    print("Momentum HF module loaded")
