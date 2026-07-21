"""
Cross-Chain & DEX Aggregation - Chapter 3
File 9: arbitrage.py

Implements Bellman-Ford and Floyd-Warshall algorithms for spatial
and triangular arbitrage pathfinding across CEX and DEX venues,
utilizing Numba for C-level speeds. Includes AMD ROCm/DirectML checks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import time

# Check for Numba availability (C-level acceleration)
try:
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    njit = jit
    prange = range

# AMD ROCm/DirectML check for graph computations
def check_graph_acceleration() -> Dict[str, bool]:
    """Check for hardware acceleration available for graph algorithms."""
    status = {
        'numba_available': NUMBA_AVAILABLE,
        'cuda_available': False,
        'rocm_available': False,
        'directml_available': False,
    }
    
    if NUMBA_AVAILABLE:
        try:
            from numba import cuda
            status['cuda_available'] = cuda.is_available()
        except Exception:
            pass
        
        try:
            import os
            rocm_paths = ['/opt/rocm', '/usr/lib/rocm', os.environ.get('ROCM_PATH', '')]
            status['rocm_available'] = any(os.path.exists(p) for p in rocm_paths if p)
        except Exception:
            pass
    
    return status


@dataclass
class ArbitragePath:
    """Represents a detected arbitrage opportunity."""
    path: List[str]  # Sequence of tokens/exchanges
    profit_percentage: float
    input_amount: float
    expected_output: float
    gas_cost_usd: float
    net_profit_usd: float
    confidence: float
    timestamp_ns: int


@dataclass
class ExchangeRate:
    """Exchange rate between two tokens on a specific venue."""
    base_token: str
    quote_token: str
    rate: float
    inverse_rate: float
    venue: str
    liquidity_usd: float
    fee_bps: int  # Basis points (1 bp = 0.01%)


class GraphBuilder:
    """Builds weighted graph representation for arbitrage detection."""
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[Tuple[str, str], List[ExchangeRate]] = defaultdict(list)
    
    def add_exchange_rate(self, rate: ExchangeRate) -> None:
        """Add an exchange rate to the graph."""
        self.nodes.add(rate.base_token)
        self.nodes.add(rate.quote_token)
        
        edge_key = (rate.base_token, rate.quote_token)
        self.edges[edge_key].append(rate)
        
        # Also add reverse edge with inverse rate
        reverse_key = (rate.quote_token, rate.base_token)
        reverse_rate = ExchangeRate(
            base_token=rate.quote_token,
            quote_token=rate.base_token,
            rate=rate.inverse_rate,
            inverse_rate=rate.rate,
            venue=rate.venue,
            liquidity_usd=rate.liquidity_usd,
            fee_bps=rate.fee_bps,
        )
        self.edges[reverse_key].append(reverse_rate)
    
    def get_best_rate(self, from_token: str, to_token: str) -> Optional[ExchangeRate]:
        """Get the best exchange rate between two tokens."""
        edge_key = (from_token, to_token)
        rates = self.edges.get(edge_key, [])
        
        if not rates:
            return None
        
        # Return rate with best effective value after fees
        best = max(rates, key=lambda r: r.rate * (1 - r.fee_bps / 10000))
        return best


if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def bellman_ford_numba(
        n_nodes: int,
        start_node: int,
        edges_src: np.ndarray,
        edges_dst: np.ndarray,
        weights: np.ndarray,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numba-accelerated Bellman-Ford algorithm for negative cycle detection.
        Returns distances and predecessor arrays.
        """
        INF = 1e18
        dist = np.full(n_nodes, INF, dtype=np.float64)
        pred = np.full(n_nodes, -1, dtype=np.int32)
        
        dist[start_node] = 0.0
        
        for _ in range(max_iterations):
            updated = False
            for i in prange(len(edges_src)):
                u = edges_src[i]
                v = edges_dst[i]
                w = weights[i]
                
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    updated = True
            
            if not updated:
                break
        
        return dist, pred

    @njit(parallel=True, cache=True)
    def floyd_warshall_numba(
        n_nodes: int,
        adj_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Numba-accelerated Floyd-Warshall algorithm for all-pairs shortest paths.
        Returns distance matrix.
        """
        dist = adj_matrix.copy()
        
        for k in range(n_nodes):
            for i in prange(n_nodes):
                for j in range(n_nodes):
                    if dist[i, k] + dist[j, k] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist
else:
    def bellman_ford_numba(*args, **kwargs):
        raise RuntimeError("Numba not available")
    
    def floyd_warshall_numba(*args, **kwargs):
        raise RuntimeError("Numba not available")


class ArbitrageDetector:
    """
    Detects arbitrage opportunities using graph algorithms.
    Supports both Bellman-Ford (single-source) and Floyd-Warshall (all-pairs).
    """
    
    def __init__(self, min_profit_threshold_pct: float = 0.1):
        self.graph = GraphBuilder()
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.min_profit_threshold = min_profit_threshold_pct
        self._last_update_ns = 0
        self.acceleration_status = check_graph_acceleration()
    
    def update_exchange_rate(self, rate: ExchangeRate) -> None:
        """Update exchange rate in the graph."""
        self.graph.add_exchange_rate(rate)
        
        # Update token index mapping
        if rate.base_token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[rate.base_token] = idx
            self.idx_to_token[idx] = rate.base_token
        
        if rate.quote_token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[rate.quote_token] = idx
            self.idx_to_token[idx] = rate.quote_token
        
        self._last_update_ns = time.time_ns()
    
    def build_weight_matrix(self) -> np.ndarray:
        """Build adjacency weight matrix for graph algorithms."""
        n = len(self.token_to_idx)
        INF = 1e18
        
        # Initialize with infinity
        weights = np.full((n, n), INF, dtype=np.float64)
        np.fill_diagonal(weights, 0.0)
        
        # Fill in edge weights (negative log of rates for arbitrage)
        for (src, dst), rates in self.graph.edges.items():
            if src in self.token_to_idx and dst in self.token_to_idx:
                src_idx = self.token_to_idx[src]
                dst_idx = self.token_to_idx[dst]
                
                # Get best rate considering fees
                best_rate = max(
                    (r.rate * (1 - r.fee_bps / 10000) for r in rates),
                    default=0
                )
                
                if best_rate > 0:
                    # Negative log transform: profitable arb = negative cycle
                    weights[src_idx, dst_idx] = -np.log(best_rate)
        
        return weights
    
    def detect_triangular_arbitrage_bellman_ford(
        self,
        start_token: str
    ) -> List[ArbitragePath]:
        """
        Detect triangular arbitrage starting from a specific token
        using Bellman-Ford algorithm.
        """
        if start_token not in self.token_to_idx:
            return []
        
        n = len(self.token_to_idx)
        start_idx = self.token_to_idx[start_token]
        
        # Build edge lists for Numba
        edges_src = []
        edges_dst = []
        weights_list = []
        
        for (src, dst), rates in self.graph.edges.items():
            if src in self.token_to_idx and dst in self.token_to_idx:
                best_rate = max(
                    (r.rate * (1 - r.fee_bps / 10000) for r in rates),
                    default=0
                )
                if best_rate > 0:
                    edges_src.append(self.token_to_idx[src])
                    edges_dst.append(self.token_to_idx[dst])
                    weights_list.append(-np.log(best_rate))
        
        if not edges_src:
            return []
        
        edges_src = np.array(edges_src, dtype=np.int32)
        edges_dst = np.array(edges_dst, dtype=np.int32)
        weights_arr = np.array(weights_list, dtype=np.float64)
        
        try:
            if NUMBA_AVAILABLE:
                dist, pred = bellman_ford_numba(
                    n, start_idx, edges_src, edges_dst, weights_arr
                )
            else:
                # Pure Python fallback
                dist, pred = self._bellman_ford_python(
                    n, start_idx, edges_src, edges_dst, weights_arr
                )
        except Exception:
            return []
        
        # Check for negative cycles (arbitrage opportunities)
        opportunities = []
        
        for i in range(len(edges_src)):
            u = edges_src[i]
            v = edges_dst[i]
            w = weights_arr[i]
            
            if dist[u] != 1e18 and dist[u] + w < dist[v]:
                # Negative cycle detected - reconstruct path
                path = self._reconstruct_path(pred, v, start_idx)
                if path:
                    profit_pct = self._calculate_profit(path)
                    if profit_pct > self.min_profit_threshold:
                        opportunities.append(ArbitragePath(
                            path=[self.idx_to_token[idx] for idx in path],
                            profit_percentage=profit_pct,
                            input_amount=1000.0,
                            expected_output=1000.0 * np.exp(-dist[v]),
                            gas_cost_usd=50.0,
                            net_profit_usd=1000.0 * (np.exp(profit_pct / 100) - 1) - 50,
                            confidence=min(1.0, profit_pct / 2.0),
                            timestamp_ns=time.time_ns(),
                        ))
        
        return opportunities
    
    def detect_all_arbitrage_floyd_warshall(
        self
    ) -> List[ArbitragePath]:
        """
        Detect all arbitrage opportunities using Floyd-Warshall algorithm.
        More comprehensive but O(n^3) complexity.
        """
        n = len(self.token_to_idx)
        if n < 3 or n > 500:  # Limit for performance
            return []
        
        weights = self.build_weight_matrix()
        
        try:
            if NUMBA_AVAILABLE:
                dist = floyd_warshall_numba(n, weights)
            else:
                dist = self._floyd_warshall_python(weights)
        except Exception:
            return []
        
        opportunities = []
        
        # Check diagonal for negative values (negative cycles)
        for i in range(n):
            if dist[i, i] < 0:
                # Arbitrage opportunity exists
                profit_pct = (np.exp(-dist[i, i]) - 1) * 100
                
                if profit_pct > self.min_profit_threshold:
                    token = self.idx_to_token[i]
                    opportunities.append(ArbitragePath(
                        path=[token],  # Simplified - would reconstruct full path
                        profit_percentage=profit_pct,
                        input_amount=1000.0,
                        expected_output=1000.0 * np.exp(-dist[i, i]),
                        gas_cost_usd=50.0,
                        net_profit_usd=1000.0 * (np.exp(profit_pct / 100) - 1) - 50,
                        confidence=min(1.0, profit_pct / 2.0),
                        timestamp_ns=time.time_ns(),
                    ))
        
        return opportunities
    
    def _bellman_ford_python(
        self,
        n: int,
        start: int,
        edges_src: np.ndarray,
        edges_dst: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure Python fallback for Bellman-Ford."""
        INF = 1e18
        dist = np.full(n, INF, dtype=np.float64)
        pred = np.full(n, -1, dtype=np.int32)
        
        dist[start] = 0.0
        
        for _ in range(n - 1):
            updated = False
            for i in range(len(edges_src)):
                u, v, w = edges_src[i], edges_dst[i], weights[i]
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    updated = True
            if not updated:
                break
        
        return dist, pred
    
    def _floyd_warshall_python(self, weights: np.ndarray) -> np.ndarray:
        """Pure Python fallback for Floyd-Warshall."""
        n = weights.shape[0]
        dist = weights.copy()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        return dist
    
    def _reconstruct_path(
        self,
        pred: np.ndarray,
        end: int,
        start: int
    ) -> List[int]:
        """Reconstruct path from predecessor array."""
        path = [end]
        current = end
        
        for _ in range(len(pred)):
            current = pred[current]
            if current == -1:
                return []
            path.append(current)
            if current == start:
                break
        
        path.reverse()
        return path if path[0] == start else []
    
    def _calculate_profit(self, path: List[int]) -> float:
        """Calculate profit percentage for a path."""
        if len(path) < 2:
            return 0.0
        
        total_log_return = 0.0
        
        for i in range(len(path) - 1):
            src = self.idx_to_token[path[i]]
            dst = self.idx_to_token[path[i + 1]]
            
            best_rate = self.graph.get_best_rate(src, dst)
            if best_rate:
                effective_rate = best_rate.rate * (1 - best_rate.fee_bps / 10000)
                total_log_return += np.log(effective_rate)
        
        return (np.exp(total_log_return) - 1) * 100
    
    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        return {
            'tokens_tracked': len(self.token_to_idx),
            'exchange_pairs': len(self.graph.edges),
            'acceleration': self.acceleration_status,
            'last_update_ns': self._last_update_ns,
            'numba_available': NUMBA_AVAILABLE,
        }


class CrossVenueArbitrage:
    """
    Coordinates arbitrage detection across CEX and DEX venues.
    Combines orderbook data from CEX with pool data from DEX.
    """
    
    def __init__(self):
        self.detector = ArbitrageDetector(min_profit_threshold_pct=0.05)
        self.venue_rates: Dict[str, List[ExchangeRate]] = defaultdict(list)
    
    def add_cex_rate(
        self,
        exchange: str,
        base: str,
        quote: str,
        bid: float,
        ask: float,
        liquidity_usd: float,
        fee_bps: int = 10
    ) -> None:
        """Add CEX orderbook-derived rate."""
        mid_rate = (bid + ask) / 2
        spread_adjusted = mid_rate * (1 - (ask - bid) / mid_rate / 2)
        
        rate = ExchangeRate(
            base_token=base,
            quote_token=quote,
            rate=spread_adjusted,
            inverse_rate=1 / spread_adjusted,
            venue=f"CEX:{exchange}",
            liquidity_usd=liquidity_usd,
            fee_bps=fee_bps,
        )
        
        self.detector.update_exchange_rate(rate)
        self.venue_rates[f"CEX:{exchange}"].append(rate)
    
    def add_dex_rate(
        self,
        dex_name: str,
        token0: str,
        token1: str,
        reserve0: int,
        reserve1: int,
        fee_bps: int = 30
    ) -> None:
        """Add DEX pool-derived rate."""
        if reserve0 == 0 or reserve1 == 0:
            return
        
        rate = reserve1 / reserve0
        
        exchange_rate = ExchangeRate(
            base_token=token0,
            quote_token=token1,
            rate=rate,
            inverse_rate=1 / rate,
            venue=f"DEX:{dex_name}",
            liquidity_usd=(reserve0 + reserve1) / 1e18 * 2000,  # Approximate
            fee_bps=fee_bps,
        )
        
        self.detector.update_exchange_rate(exchange_rate)
        self.venue_rates[f"DEX:{dex_name}"].append(exchange_rate)
    
    def find_best_arbitrage(
        self,
        method: str = 'bellman_ford'
    ) -> Optional[ArbitragePath]:
        """Find the best arbitrage opportunity."""
        if method == 'bellman_ford':
            # Try from major tokens
            all_opportunities = []
            for token in ['USDT', 'USDC', 'ETH', 'BTC']:
                opps = self.detector.detect_triangular_arbitrage_bellman_ford(token)
                all_opportunities.extend(opps)
        else:
            all_opportunities = self.detector.detect_all_arbitrage_floyd_warshall()
        
        if not all_opportunities:
            return None
        
        # Return highest net profit opportunity
        return max(all_opportunities, key=lambda o: o.net_profit_usd)


if __name__ == '__main__':
    print("Graph Acceleration Status:", check_graph_acceleration())
    print("Numba Available:", NUMBA_AVAILABLE)
    
    # Example usage
    arb = CrossVenueArbitrage()
    
    # Add some sample rates
    arb.add_cex_rate('Binance', 'ETH', 'USDT', 3000.0, 3001.0, 10000000)
    arb.add_cex_rate('Binance', 'BTC', 'USDT', 60000.0, 60050.0, 50000000)
    arb.add_dex_rate('UniswapV3', 'ETH', 'USDT', 1000000000000000000000, 3000000000000)
    
    stats = arb.detector.get_statistics()
    print("\nDetector Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
