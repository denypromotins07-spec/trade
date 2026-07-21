"""
Cross-Chain & DEX Aggregation - Chapter 3
File 7: aggregator.py

Builds a Ray-distributed DEX liquidity scanner that batches RPC calls
across multiple Layer 2 networks, strictly managing worker memory to
respect the 4GB Python quota. Includes AMD ROCm/DirectML environment checks.
"""

import os
import time
import asyncio
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

# AMD ROCm/DirectML environment check
def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability for accelerated graph computations."""
    acceleration_status = {
        'rocm_available': False,
        'directml_available': False,
        'hip_available': False,
        'recommended_backend': 'cpu'
    }
    
    # Check ROCm environment
    rocm_paths = ['/opt/rocm', '/usr/lib/rocm', os.environ.get('ROCM_PATH', '')]
    acceleration_status['rocm_available'] = any(
        os.path.exists(path) for path in rocm_paths if path
    )
    
    # Check DirectML (Windows)
    if os.name == 'nt':
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            acceleration_status['directml_available'] = 'DirectMLExecutionProvider' in providers
        except ImportError:
            pass
    
    # Check HIP (AMD's CUDA equivalent)
    try:
        import ctypes
        hip_lib = ctypes.util.find_library('amdhip64') or ctypes.util.find_library('hip')
        acceleration_status['hip_available'] = hip_lib is not None
    except Exception:
        pass
    
    # Determine recommended backend
    if acceleration_status['rocm_available'] or acceleration_status['hip_available']:
        acceleration_status['recommended_backend'] = 'rocm'
    elif acceleration_status['directml_available']:
        acceleration_status['recommended_backend'] = 'directml'
    
    return acceleration_status


@dataclass
class PoolLiquidity:
    """Represents liquidity data for a DEX pool."""
    pool_address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    fee_tier: int
    chain_id: int
    dex_name: str
    timestamp_ns: int
    price_usd: float = 0.0
    
    @property
    def total_liquidity_usd(self) -> float:
        """Estimate total liquidity in USD."""
        return (self.reserve0 * self.price_usd + self.reserve1) / 1e18


@dataclass
class ArbitragePath:
    """Represents a potential arbitrage path."""
    path: List[str]
    exchanges: List[str]
    expected_profit_pct: float
    gas_cost_usd: float
    net_profit_pct: float
    confidence: float


class RPCCacheManager:
    """Manages shared memory caching for RPC responses."""
    
    def __init__(self, max_entries: int = 10000, ttl_seconds: int = 5):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, method: str, params: tuple) -> str:
        """Generate cache key from RPC call parameters."""
        key_data = f"{method}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
    
    def get(self, method: str, params: tuple) -> Optional[Any]:
        """Get cached RPC response if valid."""
        key = self._generate_key(method, params)
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl_seconds:
                self._hits += 1
                return value
            else:
                del self._cache[key]
        self._misses += 1
        return None
    
    def set(self, method: str, params: tuple, value: Any) -> None:
        """Cache RPC response."""
        key = self._generate_key(method, params)
        
        # Evict oldest entries if at capacity
        if len(self._cache) >= self._max_entries:
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[key] = (value, time.time())
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


class RateLimiter:
    """Rate limiter for RPC calls with exponential backoff."""
    
    def __init__(self, calls_per_second: int = 100, burst_size: int = 200):
        self._calls_per_second = calls_per_second
        self._burst_size = burst_size
        self._tokens = burst_size
        self._last_refill = time.time()
        self._total_calls = 0
        self._rate_limited_count = 0
    
    async def acquire(self) -> None:
        """Acquire permission to make an RPC call."""
        while True:
            now = time.time()
            elapsed = now - self._last_refill
            
            # Refill tokens based on elapsed time
            refill_amount = elapsed * self._calls_per_second
            self._tokens = min(self._burst_size, self._tokens + refill_amount)
            self._last_refill = now
            
            if self._tokens >= 1:
                self._tokens -= 1
                self._total_calls += 1
                return
            
            # Rate limited - wait and retry
            self._rate_limited_count += 1
            await asyncio.sleep(0.01)  # 10ms backoff
    
    @property
    def rate_limit_ratio(self) -> float:
        """Ratio of rate-limited calls."""
        if self._total_calls == 0:
            return 0.0
        return self._rate_limited_count / self._total_calls


class DEXLiquidityScanner:
    """Ray-distributed DEX liquidity scanner with memory management."""
    
    SUPPORTED_CHAINS = {
        1: {'name': 'Ethereum', 'rpc': 'https://eth.llamarpc.com'},
        56: {'name': 'BSC', 'rpc': 'https://bsc-dataseed.binance.org'},
        137: {'name': 'Polygon', 'rpc': 'https://polygon-rpc.com'},
        42161: {'name': 'Arbitrum', 'rpc': 'https://arb1.arbitrum.io/rpc'},
        10: {'name': 'Optimism', 'rpc': 'https://mainnet.optimism.io'},
        8453: {'name': 'Base', 'rpc': 'https://mainnet.base.org'},
    }
    
    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.cache = RPCCacheManager()
        self.rate_limiter = RateLimiter()
        self.pools_by_chain: Dict[int, List[PoolLiquidity]] = defaultdict(list)
        self._scan_start_time = 0
        self._pools_scanned = 0
        self._memory_usage_mb = 0
        
        # Check AMD acceleration
        self.acceleration = check_amd_acceleration()
    
    def _check_memory_limit(self) -> bool:
        """Check if memory usage is within limits."""
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB
        self._memory_usage_mb = usage
        return usage < self.max_memory_mb
    
    async def fetch_pool_reserves(
        self, 
        chain_id: int, 
        pool_address: str
    ) -> Optional[PoolLiquidity]:
        """Fetch pool reserves with caching and rate limiting."""
        # Check cache first
        cached = self.cache.get('eth_getReserves', (pool_address,))
        if cached:
            return cached
        
        # Rate limit
        await self.rate_limiter.acquire()
        
        # Fetch from RPC (simplified - would use web3.py in production)
        rpc_url = self.SUPPORTED_CHAINS.get(chain_id, {}).get('rpc')
        if not rpc_url:
            return None
        
        try:
            # Simulated RPC call structure
            payload = {
                'jsonrpc': '2.0',
                'method': 'eth_call',
                'params': [{
                    'to': pool_address,
                    'data': '0x0902f1ac'  # getReserves() selector
                }, 'latest'],
                'id': 1
            }
            
            # In production: async HTTP request to RPC endpoint
            # For now, return simulated data
            pool_data = PoolLiquidity(
                pool_address=pool_address,
                token0='0xToken0',
                token1='0xToken1',
                reserve0=1000000000000000000000,
                reserve1=500000000000000000000,
                fee_tier=3000,
                chain_id=chain_id,
                dex_name='UniswapV3',
                timestamp_ns=int(time.time() * 1e9)
            )
            
            # Cache the result
            self.cache.set('eth_getReserves', (pool_address,), pool_data)
            
            return pool_data
            
        except Exception as e:
            print(f"Error fetching pool {pool_address}: {e}")
            return None
    
    async def scan_chain_pools(
        self,
        chain_id: int,
        pool_addresses: List[str],
        batch_size: int = 50
    ) -> List[PoolLiquidity]:
        """Scan pools on a specific chain with batching."""
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(pool_addresses), batch_size):
            batch = pool_addresses[i:i + batch_size]
            
            # Check memory before processing batch
            if not self._check_memory_limit():
                print(f"Memory limit approaching ({self._memory_usage_mb}MB), pausing...")
                await asyncio.sleep(1)
                # Force garbage collection
                import gc
                gc.collect()
            
            tasks = [
                self.fetch_pool_reserves(chain_id, addr) 
                for addr in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, PoolLiquidity):
                    results.append(result)
                    self._pools_scanned += 1
                elif isinstance(result, Exception):
                    print(f"Pool scan error: {result}")
            
            # Small delay between batches
            await asyncio.sleep(0.01)
        
        return results
    
    async def scan_all_chains(
        self,
        pools_by_chain: Dict[int, List[str]]
    ) -> Dict[int, List[PoolLiquidity]]:
        """Scan pools across all supported chains."""
        self._scan_start_time = time.time()
        results = {}
        
        # Create tasks for each chain
        tasks = {}
        for chain_id, addresses in pools_by_chain.items():
            if chain_id in self.SUPPORTED_CHAINS:
                tasks[chain_id] = self.scan_chain_pools(chain_id, addresses)
        
        # Execute scans concurrently
        chain_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for idx, chain_id in enumerate(tasks.keys()):
            result = chain_results[idx]
            if isinstance(result, list):
                results[chain_id] = result
                self.pools_by_chain[chain_id] = result
            else:
                print(f"Chain {chain_id} scan failed: {result}")
        
        return results
    
    def find_best_price(
        self,
        token_in: str,
        token_out: str,
        amount_in: int
    ) -> Optional[Tuple[PoolLiquidity, float]]:
        """Find the best price across all scanned pools."""
        best_pool = None
        best_price = 0.0
        
        for chain_pools in self.pools_by_chain.values():
            for pool in chain_pools:
                # Check if pool matches token pair
                if (pool.token0.lower() == token_in.lower() and 
                    pool.token1.lower() == token_out.lower()):
                    # Calculate effective price
                    price = pool.reserve1 / pool.reserve0
                    if price > best_price:
                        best_price = price
                        best_pool = pool
                
                elif (pool.token1.lower() == token_in.lower() and 
                      pool.token0.lower() == token_out.lower()):
                    # Reverse pair
                    price = pool.reserve0 / pool.reserve1
                    if price > best_price:
                        best_price = price
                        best_pool = pool
        
        if best_pool:
            return (best_pool, best_price)
        return None
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get statistics about the scan operation."""
        elapsed = time.time() - self._scan_start_time if self._scan_start_time else 0
        
        return {
            'pools_scanned': self._pools_scanned,
            'elapsed_seconds': elapsed,
            'pools_per_second': self._pools_scanned / elapsed if elapsed > 0 else 0,
            'cache_hit_rate': self.cache.hit_rate,
            'rate_limit_ratio': self.rate_limiter.rate_limit_ratio,
            'memory_usage_mb': self._memory_usage_mb,
            'acceleration_backend': self.acceleration['recommended_backend'],
            'chains_scanned': len([c for c in self.pools_by_chain if self.pools_by_chain[c]]),
        }


# Ray worker function for distributed scanning
def scan_pools_worker(
    chain_id: int,
    pool_batch: List[str],
    cache_manager: RPCCacheManager
) -> List[Dict]:
    """
    Ray worker function for parallel pool scanning.
    Memory-efficient implementation for 4GB quota.
    """
    import gc
    
    results = []
    
    for pool_addr in pool_batch:
        # Check cache
        cached = cache_manager.get('eth_getReserves', (pool_addr,))
        if cached:
            results.append({
                'pool_address': cached.pool_address,
                'reserve0': cached.reserve0,
                'reserve1': cached.reserve1,
                'chain_id': cached.chain_id,
            })
            continue
        
        # Simulate pool data fetch
        pool_data = {
            'pool_address': pool_addr,
            'reserve0': 1000000000000000000000,
            'reserve1': 500000000000000000000,
            'chain_id': chain_id,
        }
        results.append(pool_data)
        
        # Periodic cleanup for memory management
        if len(results) % 100 == 0:
            gc.collect()
    
    return results


if __name__ == '__main__':
    # Example usage
    async def main():
        scanner = DEXLiquidityScanner(max_memory_mb=4096)
        
        print("AMD Acceleration Status:", scanner.acceleration)
        
        # Sample pools to scan
        test_pools = {
            1: ['0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640'],  # ETH USDC
            56: ['0x58F876857a02D6762E0101bb5C46A8c1ED44Dc16'],  # BNB BUSD
        }
        
        results = await scanner.scan_all_chains(test_pools)
        stats = scanner.get_scan_statistics()
        
        print("\nScan Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    asyncio.run(main())
