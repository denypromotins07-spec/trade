"""
Regime Router: Ray-distributed worker routing capital to strategies based on live market regimes.
Optimized for AMD Ryzen AI 5 with ROCm/DirectML GPU acceleration for covariance matrix computation.
Strictly enforces 4GB Python RAM quota while evaluating multi-asset covariance matrices.
"""

import ray
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import psutil
import os
import time

# Enforce strict 4GB RAM quota for this Ray worker
MAX_RAM_BYTES = 4 * 1024 * 1024 * 1024  # 4GB

@dataclass
class MarketRegime:
    """Represents current market regime classification."""
    regime_id: int
    volatility_level: float  # 0.0 - 1.0
    trend_strength: float    # 0.0 - 1.0
    correlation_cluster: int
    timestamp_ms: int

@dataclass
class StrategyAllocation:
    """Capital allocation recommendation for a strategy."""
    strategy_id: int
    symbol: str
    allocation_fraction: float  # 0.0 - 1.0
    confidence_score: float
    regime_match_score: float

@ray.remote(max_calls=1000)
class RegimeRouter:
    """
    Ray-distributed regime router that allocates capital across strategies.
    Uses GPU-accelerated covariance computation via ROCm/DirectML on AMD hardware.
    """
    
    def __init__(self, symbols: List[str], max_strategies: int = 64):
        self.symbols = symbols
        self.max_strategies = max_strategies
        self.n_symbols = len(symbols)
        
        # Pre-allocate arrays to avoid heap fragmentation within 4GB limit
        self.returns_buffer = np.zeros((1000, self.n_symbols), dtype=np.float32)
        self.covariance_matrix = np.zeros((self.n_symbols, self.n_symbols), dtype=np.float32)
        self.correlation_matrix = np.zeros((self.n_symbols, self.n_symbols), dtype=np.float32)
        
        # Regime history (circular buffer)
        self.regime_history_max = 100
        self.regime_history: List[MarketRegime] = []
        
        # Strategy performance tracking
        self.strategy_scores: Dict[int, float] = {}
        
        # RAM monitoring
        self.process = psutil.Process(os.getpid())
        self._check_ram_usage()
    
    def _check_ram_usage(self) -> None:
        """Enforce 4GB RAM quota - raise exception if exceeded."""
        ram_used = self.process.memory_info().rss
        if ram_used > MAX_RAM_BYTES:
            raise MemoryError(
                f"RegimeRouter exceeded 4GB RAM quota: {ram_used / 1e9:.2f}GB used"
            )
    
    def _gpu_accelerated_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix using GPU acceleration if available.
        Falls back to optimized NumPy CPU path for AMD Ryzen AI 5.
        """
        try:
            # Attempt ROCm/DirectML acceleration via cupy if available
            import cupy as cp
            gpu_returns = cp.asarray(returns)
            gpu_cov = cp.cov(gpu_returns, rowvar=False)
            return cp.asnumpy(gpu_cov).astype(np.float32)
        except ImportError:
            # Optimized CPU path using NumPy with SIMD (AMD Ryzen benefits from this)
            # Use ddof=1 for sample covariance
            return np.cov(returns, rowvar=False).astype(np.float32)
    
    def _classify_regime(
        self, 
        returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> MarketRegime:
        """
        Classify current market regime based on volatility and correlation structure.
        Returns regime ID: 0=trending_low_vol, 1=trending_high_vol, 
                         2=mean_reverting_low_vol, 3=mean_reverting_high_vol,
                         4=chaotic_transition
        """
        # Compute average volatility (diagonal of covariance)
        volatilities = np.sqrt(np.diag(cov_matrix))
        avg_vol = np.mean(volatilities)
        
        # Compute average correlation (off-diagonal elements)
        np.fill_diagonal(cov_matrix, 0)
        avg_corr = np.mean(cov_matrix[np.nonzero(cov_matrix)])
        np.fill_diagonal(cov_matrix, volatilities**2)  # Restore diagonal
        
        # Normalize volatility to 0-1 scale (calibrated for crypto)
        vol_normalized = min(1.0, avg_vol / 0.05)  # 5% daily vol = 1.0
        
        # Determine trend vs mean-reverting based on autocorrelation
        if len(returns) > 10:
            autocorr = np.corrcoef(returns[:-1].flatten(), returns[1:].flatten())[0, 1]
        else:
            autocorr = 0.0
        
        trend_strength = max(0.0, autocorr)  # Positive autocorr = trending
        mean_rev_strength = max(0.0, -autocorr)  # Negative autocorr = mean-reverting
        
        # Classify regime
        if vol_normalized < 0.3 and trend_strength > 0.3:
            regime_id = 0  # Trending low vol
        elif vol_normalized >= 0.3 and trend_strength > 0.3:
            regime_id = 1  # Trending high vol
        elif vol_normalized < 0.3 and mean_rev_strength > 0.3:
            regime_id = 2  # Mean-reverting low vol
        elif vol_normalized >= 0.3 and mean_rev_strength > 0.3:
            regime_id = 3  # Mean-reverting high vol
        else:
            regime_id = 4  # Chaotic/transition
        
        return MarketRegime(
            regime_id=regime_id,
            volatility_level=vol_normalized,
            trend_strength=trend_strength,
            correlation_cluster=int(avg_corr > 0.5),
            timestamp_ms=int(time.time() * 1000)
        )
    
    def update_returns(self, new_returns: np.ndarray) -> None:
        """
        Update returns buffer with new data. Maintains circular buffer.
        new_returns shape: (n_samples, n_symbols)
        """
        n_new = new_returns.shape[0]
        
        if n_new >= self.returns_buffer.shape[0]:
            # Replace entire buffer
            self.returns_buffer = new_returns[-self.returns_buffer.shape[0]:].copy()
        else:
            # Shift and append (circular buffer simulation)
            self.returns_buffer = np.roll(self.returns_buffer, -n_new, axis=0)
            self.returns_buffer[-n_new:] = new_returns
        
        self._check_ram_usage()
    
    def compute_allocations(
        self,
        strategy_metadatas: List[Dict]
    ) -> List[StrategyAllocation]:
        """
        Compute optimal strategy allocations based on current regime.
        Returns list of StrategyAllocation objects for each symbol-strategy pair.
        """
        # Compute covariance matrix (GPU-accelerated)
        self.covariance_matrix = self._gpu_accelerated_covariance(self.returns_buffer)
        
        # Classify current regime
        current_regime = self._classify_regime(
            self.returns_buffer,
            self.covariance_matrix.copy()
        )
        
        # Add to regime history
        self.regime_history.append(current_regime)
        if len(self.regime_history) > self.regime_history_max:
            self.regime_history.pop(0)
        
        allocations: List[StrategyAllocation] = []
        
        for strat_meta in strategy_metadatas:
            strategy_id = strat_meta['id']
            symbol = strat_meta['symbol']
            strategy_regime_affinity = strat_meta.get('regime_affinity', {})
            
            # Compute regime match score
            regime_match = strategy_regime_affinity.get(
                str(current_regime.regime_id), 
                0.5
            )
            
            # Adjust by recent performance
            recent_performance = self.strategy_scores.get(strategy_id, 0.5)
            
            # Combined confidence score
            confidence = 0.6 * regime_match + 0.4 * recent_performance
            
            # Allocation fraction based on confidence and regime volatility
            base_allocation = 0.1  # 10% base
            vol_adjustment = 1.0 - current_regime.volatility_level * 0.5
            allocation_fraction = min(0.25, base_allocation * confidence * vol_adjustment)
            
            allocations.append(StrategyAllocation(
                strategy_id=strategy_id,
                symbol=symbol,
                allocation_fraction=allocation_fraction,
                confidence_score=confidence,
                regime_match_score=regime_match
            ))
        
        self._check_ram_usage()
        return allocations
    
    def update_strategy_score(self, strategy_id: int, pnl: float, risk: float) -> None:
        """Update strategy performance score based on PnL and risk."""
        if risk == 0:
            sharpe = 0
        else:
            sharpe = pnl / risk
        
        # Exponential moving average update
        alpha = 0.1
        old_score = self.strategy_scores.get(strategy_id, 0.5)
        new_score = alpha * (sharpe + 1) / 2 + (1 - alpha) * old_score  # Normalize to 0-1
        self.strategy_scores[strategy_id] = max(0.0, min(1.0, new_score))
        
        self._check_ram_usage()
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """Return the most recently classified regime."""
        if self.regime_history:
            return self.regime_history[-1]
        return None
    
    def force_garbage_collect(self) -> None:
        """Force garbage collection to stay within 4GB limit."""
        import gc
        gc.collect()
        self._check_ram_usage()


# Ray worker initialization helper
def create_regime_router(symbols: List[str]) -> ray.ObjectRef:
    """Create a RegimeRouter Ray actor."""
    return RegimeRouter.remote(symbols)


if __name__ == "__main__":
    # Initialize Ray with memory limits
    ray.init(
        object_store_memory=MAX_RAM_BYTES // 2,  # 2GB for object store
        _system_config={"max_bytes_to_spill": MAX_RAM_BYTES}
    )
    
    # Example usage
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
    router_ref = create_regime_router(symbols)
    router = ray.get(router_ref)
    
    # Simulate returns
    np.random.seed(42)
    mock_returns = np.random.randn(100, len(symbols)) * 0.02
    
    ray.get(router.update_returns.remote(mock_returns))
    allocations = ray.get(router.compute_allocations.remote([]))
    
    print(f"Computed {len(allocations)} allocations")
    print(f"Current regime: {ray.get(router.get_current_regime.remote())}")
    
    ray.shutdown()
