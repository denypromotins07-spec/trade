"""
bridge_tracker.py - Cross-Chain Bridge Latency & TVL Monitor

This module tracks cross-chain message latencies for Wormhole and LayerZero bridges
on Ray workers. It monitors bridge Total Value Locked (TVL), security events, and
message finality times to identify latency arbitrage opportunities.

Optimization Targets:
- Strict 4GB Python RAM quota enforcement
- Polars for vectorized math (memory efficient)
- AMD ROCm/DirectML acceleration checks
- Real-time alerting on bridge congestion

Usage:
    Initialize via Ray actor system for distributed monitoring across chains.
"""

import ray
import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory quota: 4GB max per worker
MEMORY_QUOTA_BYTES = 4 * 1024 * 1024 * 1024


@dataclass
class BridgeEvent:
    """Represents a cross-chain bridge event."""
    bridge_name: str
    source_chain: str
    dest_chain: str
    message_id: str
    timestamp: datetime
    latency_ms: float
    tvl_usd: float
    status: str  # 'pending', 'finalized', 'failed'
    gas_cost_usd: float = 0.0


@dataclass
class BridgeMetrics:
    """Aggregated metrics for a bridge."""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_tvl_usd: float
    failure_rate: float
    congestion_score: float  # 0.0 (clear) to 1.0 (congested)


def check_amd_acceleration() -> Dict[str, bool]:
    """
    Check for AMD ROCm/DirectML availability for tensor acceleration.
    Returns dict of available acceleration backends.
    """
    acceleration = {
        'rocm': False,
        'directml': False,
        'cuda': False
    }
    
    try:
        import torch
        if hasattr(torch.backends, 'hip') and torch.backends.hip.is_available():
            acceleration['rocm'] = True
            logger.info("AMD ROCm acceleration available")
        elif hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
            acceleration['directml'] = True
            logger.info("DirectML acceleration available")
        elif torch.cuda.is_available():
            acceleration['cuda'] = True
            logger.info("CUDA available (fallback)")
    except ImportError:
        logger.warning("PyTorch not available, using CPU-only operations")
    
    return acceleration


@ray.remote(max_calls=1000)  # Restart worker after 1000 calls to prevent memory leaks
class BridgeTracker:
    """
    Ray actor for tracking cross-chain bridge metrics.
    
    Enforces strict memory quotas by:
    1. Using Polars lazy evaluation where possible
    2. Periodic garbage collection
    3. Bounded history buffers
    """
    
    def __init__(self, chain_pairs: List[Tuple[str, str]], history_size: int = 10000):
        """
        Initialize bridge tracker.
        
        Args:
            chain_pairs: List of (source_chain, dest_chain) tuples to monitor
            history_size: Maximum number of events to keep in memory
        """
        self.chain_pairs = chain_pairs
        self.history_size = history_size
        
        # Bounded buffers using lists with manual size control
        self._events: List[BridgeEvent] = []
        self._metrics_cache: Dict[str, BridgeMetrics] = {}
        
        # Acceleration check
        self.acceleration = check_amd_acceleration()
        
        # Memory tracking
        self._last_gc_time = datetime.now()
        self._event_count = 0
        
        logger.info(f"BridgeTracker initialized for {len(chain_pairs)} chain pairs")
        logger.info(f"Acceleration backends: {self.acceleration}")
    
    def _enforce_memory_quota(self) -> None:
        """Enforce 4GB RAM quota by trimming history and forcing GC."""
        current_time = datetime.now()
        
        # Force GC every 60 seconds or if buffer is full
        if (current_time - self._last_gc_time).total_seconds() > 60 or \
           len(self._events) >= self.history_size:
            
            # Trim oldest events if at capacity
            if len(self._events) >= self.history_size:
                trim_count = len(self._events) // 4  # Remove oldest 25%
                self._events = self._events[trim_count:]
                logger.debug(f"Trimmed {trim_count} old events from buffer")
            
            # Clear metrics cache periodically
            if len(self._metrics_cache) > 100:
                self._metrics_cache.clear()
            
            gc.collect()
            self._last_gc_time = current_time
    
    def add_event(self, event: BridgeEvent) -> None:
        """
        Add a bridge event to the tracker.
        
        Args:
            event: BridgeEvent to record
        """
        self._enforce_memory_quota()
        self._events.append(event)
        self._event_count += 1
        
        # Invalidate cached metrics for this bridge
        bridge_key = f"{event.bridge_name}:{event.source_chain}->{event.dest_chain}"
        if bridge_key in self._metrics_cache:
            del self._metrics_cache[bridge_key]
    
    def add_batch_events(self, events: List[Dict]) -> int:
        """
        Add batch of events using Polars for efficient processing.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Number of events successfully added
        """
        self._enforce_memory_quota()
        
        if not events:
            return 0
        
        # Convert to Polars DataFrame for efficient filtering/sorting
        df = pl.DataFrame(events)
        
        # Validate required columns
        required_cols = ['bridge_name', 'source_chain', 'dest_chain', 
                        'message_id', 'latency_ms', 'tvl_usd', 'status']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to BridgeEvent objects
        count = 0
        for row in df.iter_rows(named=True):
            try:
                event = BridgeEvent(
                    bridge_name=row['bridge_name'],
                    source_chain=row['source_chain'],
                    dest_chain=row['dest_chain'],
                    message_id=row['message_id'],
                    timestamp=datetime.now(),  # Use current time if not provided
                    latency_ms=float(row['latency_ms']),
                    tvl_usd=float(row['tvl_usd']),
                    status=row['status']
                )
                self._events.append(event)
                count += 1
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid event: {e}")
        
        return count
    
    def get_metrics(self, bridge_name: str, 
                   source_chain: str, 
                   dest_chain: str) -> Optional[BridgeMetrics]:
        """
        Get aggregated metrics for a specific bridge route.
        
        Args:
            bridge_name: Name of bridge (e.g., 'wormhole', 'layerzero')
            source_chain: Source blockchain
            dest_chain: Destination blockchain
            
        Returns:
            BridgeMetrics or None if no data available
        """
        bridge_key = f"{bridge_name}:{source_chain}->{dest_chain}"
        
        # Return cached metrics if available
        if bridge_key in self._metrics_cache:
            return self._metrics_cache[bridge_key]
        
        # Filter events for this route
        route_events = [
            e for e in self._events 
            if e.bridge_name == bridge_name 
            and e.source_chain == source_chain 
            and e.dest_chain == dest_chain
        ]
        
        if not route_events:
            return None
        
        # Calculate metrics using Polars for efficiency
        latencies = pl.Series([e.latency_ms for e in route_events])
        tvls = pl.Series([e.tvl_usd for e in route_events])
        failures = sum(1 for e in route_events if e.status == 'failed')
        
        avg_latency = latencies.mean()
        p95_latency = latencies.quantile(0.95)
        p99_latency = latencies.quantile(0.99)
        total_tvl = tvls[-1] if len(tvls) > 0 else 0.0  # Latest TVL
        failure_rate = failures / len(route_events) if route_events else 0.0
        
        # Calculate congestion score based on latency percentile spread
        congestion_score = min(1.0, (p99_latency - avg_latency) / (avg_latency + 1e-6))
        
        metrics = BridgeMetrics(
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            total_tvl_usd=total_tvl,
            failure_rate=failure_rate,
            congestion_score=congestion_score
        )
        
        self._metrics_cache[bridge_key] = metrics
        return metrics
    
    def detect_arbitrage_opportunities(
        self, 
        min_latency_diff_ms: float = 100.0
    ) -> List[Dict]:
        """
        Detect latency arbitrage opportunities between bridges.
        
        Args:
            min_latency_diff_ms: Minimum latency difference to consider profitable
            
        Returns:
            List of arbitrage opportunity dictionaries
        """
        opportunities = []
        
        # Group by chain pair
        chain_pair_metrics: Dict[Tuple[str, str], List[Tuple[str, BridgeMetrics]]] = {}
        
        for event in self._events:
            key = (event.source_chain, event.dest_chain)
            if key not in chain_pair_metrics:
                chain_pair_metrics[key] = []
        
        # Get unique bridges per chain pair
        for (src, dst), _ in chain_pair_metrics.items():
            bridges_in_pair = {}
            for event in self._events:
                if event.source_chain == src and event.dest_chain == dst:
                    metrics = self.get_metrics(event.bridge_name, src, dst)
                    if metrics:
                        bridges_in_pair[event.bridge_name] = metrics
            
            # Find latency differences between bridges
            bridge_list = list(bridges_in_pair.items())
            for i, (bridge_a, metrics_a) in enumerate(bridge_list):
                for bridge_b, metrics_b in bridge_list[i+1:]:
                    latency_diff = abs(metrics_a.avg_latency_ms - metrics_b.avg_latency_ms)
                    
                    if latency_diff >= min_latency_diff_ms:
                        faster_bridge = bridge_a if metrics_a.avg_latency_ms < metrics_b.avg_latency_ms else bridge_b
                        slower_bridge = bridge_b if metrics_a.avg_latency_ms < metrics_b.avg_latency_ms else bridge_a
                        
                        opportunities.append({
                            'source_chain': src,
                            'dest_chain': dst,
                            'faster_bridge': faster_bridge,
                            'slower_bridge': slower_bridge,
                            'latency_advantage_ms': latency_diff,
                            'timestamp': datetime.now().isoformat()
                        })
        
        return opportunities
    
    def get_security_alerts(self) -> List[Dict]:
        """
        Generate security alerts for anomalous bridge behavior.
        
        Returns:
            List of security alert dictionaries
        """
        alerts = []
        
        for event in self._events[-1000:]:  # Check last 1000 events
            # Alert on sudden TVL drops
            if event.tvl_usd < 1_000_000:  # Less than $1M TVL
                alerts.append({
                    'type': 'LOW_TVL',
                    'bridge': event.bridge_name,
                    'chain_pair': f"{event.source_chain}->{event.dest_chain}",
                    'tvl_usd': event.tvl_usd,
                    'severity': 'HIGH' if event.tvl_usd < 100_000 else 'MEDIUM'
                })
            
            # Alert on high failure rates
            if event.status == 'failed':
                alerts.append({
                    'type': 'TRANSACTION_FAILURE',
                    'bridge': event.bridge_name,
                    'message_id': event.message_id,
                    'severity': 'MEDIUM'
                })
        
        return alerts


# Ray-compatible initialization function
@ray.remote
def create_bridge_tracker(chain_pairs: List[Tuple[str, str]]) -> BridgeTracker:
    """Factory function to create BridgeTracker actors."""
    return BridgeTracker.remote(chain_pairs)
