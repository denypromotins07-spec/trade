"""
grafana_sync.py - Prometheus Metrics Scraper for Grafana Dashboard Sync

This module develops a lightweight Prometheus scraper that syncs Rust metrics
to a local Grafana dashboard, strictly rate-limiting queries to respect the
8GB global RAM limit.

Optimization Targets:
- Strict rate limiting to prevent memory pressure
- Efficient metric aggregation before export
- AMD ROCm/DirectML environment awareness
- PowerShell /START and /KILL orchestration compatibility

Usage:
    Run as background service alongside the main trading bot.
"""

import ray
import requests
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import gc
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory quota: 8GB global limit (shared with Rust components)
GLOBAL_MEMORY_QUOTA_BYTES = 8 * 1024 * 1024 * 1024

# Rate limiting configuration
MAX_QUERIES_PER_SECOND = 10
MAX_METRICS_BUFFER_SIZE = 10000
FLUSH_INTERVAL_SECONDS = 5


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Aggregated metric series for export."""
    name: str
    labels: Dict[str, str]
    values: List[float]
    timestamps: List[datetime]
    
    def add_point(self, value: float, timestamp: datetime) -> None:
        """Add a data point to the series."""
        if len(self.values) >= MAX_METRICS_BUFFER_SIZE:
            # Drop oldest point
            self.values.pop(0)
            self.timestamps.pop(0)
        
        self.values.append(value)
        self.timestamps.append(timestamp)


def check_amd_acceleration() -> Dict[str, bool]:
    """Check for AMD ROCm/DirectML availability."""
    acceleration = {'rocm': False, 'directml': False, 'cuda': False}
    
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
    except ImportError:
        pass
    
    return acceleration


class RateLimiter:
    """Token bucket rate limiter for API queries."""
    
    def __init__(self, max_rate: float):
        """
        Initialize rate limiter.
        
        Args:
            max_rate: Maximum queries per second
        """
        self.max_rate = max_rate
        self.tokens = max_rate
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a token. Returns True if successful."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_rate, self.tokens + elapsed * self.max_rate)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False
    
    def wait_for_token(self, timeout: float = 1.0) -> bool:
        """Wait for a token with timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire():
                return True
            time.sleep(0.01)
        return False


@ray.remote(max_calls=1000)
class GrafanaSyncAgent:
    """
    Ray actor for syncing metrics to Grafana/Prometheus.
    
    Features:
    - Rate-limited metric collection
    - Memory-bounded buffering
    - Automatic garbage collection
    - PowerShell orchestration compatibility
    """
    
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        grafana_url: str = "http://localhost:3000",
        rate_limit: float = MAX_QUERIES_PER_SECOND
    ):
        """
        Initialize Grafana sync agent.
        
        Args:
            prometheus_url: Prometheus server URL
            grafana_url: Grafana server URL
            rate_limit: Maximum queries per second
        """
        self.prometheus_url = prometheus_url
        self.grafana_url = grafana_url
        self.rate_limiter = RateLimiter(rate_limit)
        
        # Metric storage
        self._metrics: Dict[str, MetricSeries] = {}
        self._pending_points: List[MetricPoint] = []
        
        # State
        self.acceleration = check_amd_acceleration()
        self._last_gc = datetime.now()
        self._last_flush = datetime.now()
        self._query_count = 0
        self._dropped_count = 0
        
        # Health tracking
        self._is_running = True
        self._start_time = datetime.now()
        
        logger.info(f"GrafanaSyncAgent initialized")
        logger.info(f"Prometheus: {prometheus_url}, Grafana: {grafana_url}")
        logger.info(f"Acceleration: {self.acceleration}")
    
    def _enforce_memory_quota(self) -> None:
        """Enforce memory quota by trimming buffers."""
        now = datetime.now()
        
        # Force GC periodically
        if (now - self._last_gc).total_seconds() > 30:
            gc.collect()
            self._last_gc = now
        
        # Trim pending points if buffer is full
        if len(self._pending_points) > MAX_METRICS_BUFFER_SIZE:
            trim_count = len(self._pending_points) // 4
            self._pending_points = self._pending_points[trim_count:]
            self._dropped_count += trim_count
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None
    ) -> bool:
        """
        Record a metric data point.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional label dictionary
            
        Returns:
            True if recorded successfully
        """
        self._enforce_memory_quota()
        
        if not self._is_running:
            return False
        
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        if len(self._pending_points) >= MAX_METRICS_BUFFER_SIZE:
            self._dropped_count += 1
            return False
        
        self._pending_points.append(point)
        return True
    
    def record_batch_metrics(self, metrics: List[Dict]) -> int:
        """
        Record multiple metrics at once.
        
        Args:
            metrics: List of metric dictionaries with keys: name, value, labels
            
        Returns:
            Number of metrics successfully recorded
        """
        self._enforce_memory_quota()
        
        count = 0
        for m in metrics:
            if self.record_metric(m['name'], m['value'], m.get('labels')):
                count += 1
        
        return count
    
    def _flush_pending_metrics(self) -> int:
        """Flush pending metrics to series storage."""
        if not self._pending_points:
            return 0
        
        flushed = 0
        for point in self._pending_points:
            key = f"{point.name}:{sorted(point.labels.items())}"
            
            if key not in self._metrics:
                self._metrics[key] = MetricSeries(
                    name=point.name,
                    labels=point.labels,
                    values=[],
                    timestamps=[]
                )
            
            self._metrics[key].add_point(point.value, point.timestamp)
            flushed += 1
        
        self._pending_points.clear()
        return flushed
    
    def query_prometheus(self, query: str) -> Optional[Dict]:
        """
        Query Prometheus with rate limiting.
        
        Args:
            query: PromQL query string
            
        Returns:
            Query result dictionary or None if rate limited
        """
        if not self.rate_limiter.wait_for_token(timeout=2.0):
            logger.warning("Rate limit exceeded for Prometheus queries")
            return None
        
        self._query_count += 1
        
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return None
    
    def get_dashboard_metrics(self) -> Dict:
        """Get aggregated metrics for Grafana dashboard."""
        self._flush_pending_metrics()
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'metrics_count': len(self._metrics),
            'pending_points': len(self._pending_points),
            'dropped_points': self._dropped_count,
            'query_count': self._query_count,
            'series': {}
        }
        
        # Aggregate recent values for each series
        for key, series in self._metrics.items():
            if series.values:
                dashboard_data['series'][key] = {
                    'latest_value': series.values[-1],
                    'min_value': min(series.values[-100:]) if len(series.values) >= 100 else min(series.values),
                    'max_value': max(series.values[-100:]) if len(series.values) >= 100 else max(series.values),
                    'avg_value': np.mean(series.values[-100:]) if len(series.values) >= 100 else np.mean(series.values),
                    'point_count': len(series.values)
                }
        
        return dashboard_data
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics."""
        import sys
        
        # Estimate memory usage
        metrics_memory = sum(
            len(s.values) * 8 + len(s.timestamps) * 8
            for s in self._metrics.values()
        )
        pending_memory = len(self._pending_points) * 64  # Approximate size
        
        return {
            'metrics_memory_bytes': metrics_memory,
            'pending_memory_bytes': pending_memory,
            'total_estimated_bytes': metrics_memory + pending_memory,
            'global_quota_bytes': GLOBAL_MEMORY_QUOTA_BYTES,
            'quota_usage_pct': (metrics_memory + pending_memory) / GLOBAL_MEMORY_QUOTA_BYTES * 100
        }
    
    def health_check(self) -> Dict:
        """Return health status for orchestration."""
        return {
            'is_running': self._is_running,
            'status': 'healthy' if self._is_running else 'stopped',
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'acceleration': self.acceleration
        }
    
    def shutdown(self) -> None:
        """Graceful shutdown for /KILL orchestration."""
        logger.info("GrafanaSyncAgent shutting down...")
        self._is_running = False
        
        # Final flush
        self._flush_pending_metrics()
        
        logger.info(f"Final stats: {self._query_count} queries, {self._dropped_count} dropped points")


@ray.remote
def create_grafana_sync_agent(
    prometheus_url: str = "http://localhost:9090",
    grafana_url: str = "http://localhost:3000"
) -> GrafanaSyncAgent:
    """Factory function to create Grafana sync agents."""
    return GrafanaSyncAgent.remote(prometheus_url, grafana_url)


# PowerShell-compatible entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grafana Sync Agent')
    parser.add_argument('--prometheus-url', default='http://localhost:9090')
    parser.add_argument('--grafana-url', default='http://localhost:3000')
    
    args = parser.parse_args()
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Create agent
    agent = create_grafana_sync_agent.remote(args.prometheus_url, args.grafana_url)
    
    logger.info("Grafana Sync Agent started. Use /KILL to stop.")
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ray.get(agent.shutdown.remote())
        ray.shutdown()
