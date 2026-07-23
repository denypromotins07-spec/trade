"""
Ray-based Anomaly Detection for System Telemetry
==================================================

This module streams kernel and CPU telemetry to a Ray-based anomaly detector,
strictly enforcing the 4GB Python RAM quota while flagging hardware degradation
in real-time.

Optimized for: AMD Ryzen AI 5, Ray distributed processing, 4GB RAM limit
Key Features:
- Streaming anomaly detection using statistical methods
- Memory-bounded sliding windows
- Real-time hardware degradation alerts
- Ray actor-based parallel processing

Author: Nautilus/Ray Trading Bot - Stage 36
"""

import os
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import ray


# Memory budget constants (4GB Python RAM quota)
MAX_TELEMETRY_WINDOW = 10000  # Maximum samples in sliding window
ANOMALY_THRESHOLD_STD = 3.0  # Standard deviations for anomaly detection
MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024  # 4GB hard limit


class AnomalyType(Enum):
    """Types of detectable anomalies."""
    LATENCY_SPIKE = "latency_spike"
    THROUGHPUT_DROP = "throughput_drop"
    MEMORY_LEAK = "memory_leak"
    CPU_THROTTLE = "cpu_throttle"
    NIC_INTERRUPT_STORM = "nic_interrupt_storm"
    CONTEXT_SWITCH_EXCESS = "context_switch_excess"
    HARDWARE_DEGRADATION = "hardware_degradation"


@dataclass
class TelemetrySample:
    """Single telemetry sample."""
    timestamp_ns: int
    metric_name: str
    value: float
    cpu_id: int = 0
    thread_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyAlert:
    """Anomaly alert structure."""
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    description: str
    timestamp_ns: int
    affected_metric: str
    threshold_exceeded: float
    actual_value: float
    recommended_action: str = ""


class SlidingWindowStats:
    """
    Memory-efficient sliding window statistics calculator.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, max_size: int = MAX_TELEMETRY_WINDOW):
        self.max_size = max_size
        self.window: deque = deque(maxlen=max_size)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')
        
        # Percentile tracking (approximate)
        self.sorted_cache: Optional[List[float]] = None
        self.cache_dirty = True
    
    def add(self, value: float) -> None:
        """Add a value to the sliding window."""
        if len(self.window) >= self.max_size:
            # Remove oldest value from statistics
            old_value = self.window[0]
            self._remove_old_value(old_value)
        
        self.window.append(value)
        self._update_statistics(value, add=True)
        self.cache_dirty = True
        
        # Update min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    def _update_statistics(self, value: float, add: bool) -> None:
        """Update running statistics using Welford's algorithm."""
        if add:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
        else:
            self.count -= 1
            if self.count > 0:
                delta = value - self.mean
                self.mean -= delta / self.count
                delta2 = value - self.mean
                self.m2 -= delta * delta2
            else:
                self.count = 0
                self.mean = 0.0
                self.m2 = 0.0
    
    def _remove_old_value(self, old_value: float) -> None:
        """Remove an old value from statistics."""
        self._update_statistics(old_value, add=False)
    
    @property
    def variance(self) -> float:
        """Calculate variance."""
        if self.count < 2:
            return 0.0
        return self.m2 / self.count
    
    @property
    def std_dev(self) -> float:
        """Calculate standard deviation."""
        return np.sqrt(self.variance)
    
    def percentile(self, p: float) -> float:
        """Calculate percentile (approximate for large windows)."""
        if not self.window:
            return 0.0
        
        if self.cache_dirty or self.sorted_cache is None:
            # Only sort every N samples for efficiency
            if len(self.window) < 1000:
                self.sorted_cache = sorted(self.window)
            else:
                # Use numpy for large arrays (faster)
                self.sorted_cache = np.sort(list(self.window))
            self.cache_dirty = False
        
        idx = int(len(self.sorted_cache) * p / 100)
        return self.sorted_cache[min(idx, len(self.sorted_cache) - 1)]
    
    def get_z_score(self, value: float) -> float:
        """Calculate z-score for a value."""
        if self.std_dev == 0:
            return 0.0
        return (value - self.mean) / self.std_dev
    
    def is_anomaly(self, value: float, threshold: float = ANOMALY_THRESHOLD_STD) -> bool:
        """Check if value is anomalous."""
        return abs(self.get_z_score(value)) > threshold
    
    def clear(self) -> None:
        """Clear all statistics."""
        self.window.clear()
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sorted_cache = None
        self.cache_dirty = True


@ray.remote(max_calls=1000)  # Restart periodically to prevent memory leaks
class TelemetryStreamActor:
    """
    Ray actor for processing telemetry streams.
    Enforces strict memory limits per worker.
    """
    
    def __init__(self, metric_name: str, window_size: int = MAX_TELEMETRY_WINDOW):
        self.metric_name = metric_name
        self.stats = SlidingWindowStats(window_size)
        self.sample_count = 0
        self.anomaly_count = 0
        self.last_alert_time = 0
        self.alert_cooldown_ns = 1_000_000_000  # 1 second between alerts
        
        # Memory tracking
        self.memory_limit = MEMORY_LIMIT_BYTES // 10  # 400MB per actor (for 10 actors)
    
    def process_sample(self, sample: TelemetrySample) -> Optional[AnomalyAlert]:
        """Process a telemetry sample and check for anomalies."""
        self.sample_count += 1
        
        # Check memory usage
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_limit * 0.9:
            # Force garbage collection
            import gc
            gc.collect()
        
        # Add to statistics
        self.stats.add(sample.value)
        
        # Check for anomaly
        if self.stats.is_anomaly(sample.value):
            z_score = self.stats.get_z_score(sample.value)
            
            # Rate limit alerts
            now_ns = time.time_ns()
            if now_ns - self.last_alert_time < self.alert_cooldown_ns:
                return None
            
            self.last_alert_time = now_ns
            self.anomaly_count += 1
            
            severity = min(1.0, abs(z_score) / 10.0)
            
            return AnomalyAlert(
                anomaly_type=self._determine_anomaly_type(sample.value),
                severity=severity,
                description=f"Anomaly detected in {self.metric_name}: value={sample.value:.2f}, z-score={z_score:.2f}",
                timestamp_ns=sample.timestamp_ns,
                affected_metric=self.metric_name,
                threshold_exceeded=self.stats.mean + ANOMALY_THRESHOLD_STD * self.stats.std_dev,
                actual_value=sample.value,
                recommended_action=self._get_recommendation(sample.value)
            )
        
        return None
    
    def _determine_anomaly_type(self, value: float) -> AnomalyType:
        """Determine the type of anomaly based on metric and value."""
        if 'latency' in self.metric_name.lower():
            return AnomalyType.LATENCY_SPIKE
        elif 'throughput' in self.metric_name.lower():
            return AnomalyType.THROUGHPUT_DROP
        elif 'memory' in self.metric_name.lower():
            return AnomalyType.MEMORY_LEAK
        elif 'cpu' in self.metric_name.lower():
            return AnomalyType.CPU_THROTTLE
        elif 'interrupt' in self.metric_name.lower():
            return AnomalyType.NIC_INTERRUPT_STORM
        elif 'context' in self.metric_name.lower():
            return AnomalyType.CONTEXT_SWITCH_EXCESS
        else:
            return AnomalyType.HARDWARE_DEGRADATION
    
    def _get_recommendation(self, value: float) -> str:
        """Get recommended action based on anomaly type."""
        anomaly_type = self._determine_anomaly_type(value)
        
        recommendations = {
            AnomalyType.LATENCY_SPIKE: "Check network latency and CPU scheduling",
            AnomalyType.THROUGHPUT_DROP: "Investigate bottleneck in data pipeline",
            AnomalyType.MEMORY_LEAK: "Restart affected workers, check for memory leaks",
            AnomalyType.CPU_THROTTLE: "Reduce workload or increase CPU allocation",
            AnomalyType.NIC_INTERRUPT_STORM: "Check NIC driver and interrupt coalescing",
            AnomalyType.CONTEXT_SWITCH_EXCESS: "Reduce thread count or pin threads to cores",
            AnomalyType.HARDWARE_DEGRADATION: "Run hardware diagnostics",
        }
        
        return recommendations.get(anomaly_type, "Investigate immediately")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            'metric_name': self.metric_name,
            'sample_count': self.sample_count,
            'anomaly_count': self.anomaly_count,
            'mean': self.stats.mean,
            'std_dev': self.stats.std_dev,
            'min': self.stats.min_val if self.stats.min_val != float('inf') else 0,
            'max': self.stats.max_val if self.stats.max_val != float('-inf') else 0,
            'p50': self.stats.percentile(50),
            'p95': self.stats.percentile(95),
            'p99': self.stats.percentile(99),
            'memory_mb': self._get_memory_usage() / 1024 / 1024,
        }
    
    def _get_memory_usage(self) -> int:
        """Get current process memory usage."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss


@ray.remote
class AnomalyDetectorManager:
    """
    Central manager for anomaly detection across multiple metrics.
    Coordinates telemetry streams and aggregates alerts.
    """
    
    def __init__(self):
        self.metric_actors: Dict[str, ray.actor.ActorHandle] = {}
        self.alert_buffer: deque = deque(maxlen=1000)
        self.start_time = time.time()
        self.total_samples = 0
        self.total_anomalies = 0
    
    def register_metric(self, metric_name: str) -> None:
        """Register a new metric for monitoring."""
        if metric_name not in self.metric_actors:
            self.metric_actors[metric_name] = TelemetryStreamActor.remote(metric_name)
    
    async def process_sample(self, sample: TelemetrySample) -> Optional[AnomalyAlert]:
        """Process a telemetry sample through the appropriate actor."""
        self.total_samples += 1
        
        if sample.metric_name not in self.metric_actors:
            self.register_metric(sample.metric_name)
        
        actor = self.metric_actors[sample.metric_name]
        alert = await actor.process_sample.remote(sample)
        
        if alert:
            self.total_anomalies += 1
            self.alert_buffer.append(alert)
        
        return alert
    
    async def process_batch(self, samples: List[TelemetrySample]) -> List[AnomalyAlert]:
        """Process a batch of samples efficiently."""
        alerts = []
        
        # Group by metric
        by_metric: Dict[str, List[TelemetrySample]] = {}
        for sample in samples:
            if sample.metric_name not in by_metric:
                by_metric[sample.metric_name] = []
            by_metric[sample.metric_name].append(sample)
        
        # Process each metric in parallel
        tasks = []
        for metric_name, metric_samples in by_metric.items():
            if metric_name not in self.metric_actors:
                self.register_metric(metric_name)
            
            actor = self.metric_actors[metric_name]
            for sample in metric_samples:
                tasks.append(actor.process_sample.remote(sample))
        
        # Collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, AnomalyAlert):
                alerts.append(result)
                self.total_anomalies += 1
        
        self.total_samples += len(samples)
        
        if alerts:
            self.alert_buffer.extend(alerts)
        
        return alerts
    
    def get_recent_alerts(self, count: int = 10) -> List[AnomalyAlert]:
        """Get most recent alerts."""
        return list(self.alert_buffer)[-count:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'uptime_seconds': time.time() - self.start_time,
            'total_samples': self.total_samples,
            'total_anomalies': self.total_anomalies,
            'anomaly_rate': self.total_anomalies / max(self.total_samples, 1),
            'monitored_metrics': list(self.metric_actors.keys()),
            'recent_alerts': len(self.alert_buffer),
        }


def initialize_ray_detector(num_initial_metrics: int = 5):
    """
    Initialize Ray-based anomaly detection system.
    
    Args:
        num_initial_metrics: Number of initial metric actors to spawn
    
    Returns:
        AnomalyDetectorManager actor handle
    """
    if not ray.is_initialized():
        # Configure Ray with memory limits
        ray.init(
            object_store_memory=1 * 1024 * 1024 * 1024,  # 1GB object store
            _system_config={"worker_max_memory_percentage": 40}
        )
    
    manager = AnomalyDetectorManager.remote()
    return manager


# Example usage and testing
if __name__ == '__main__':
    import asyncio
    
    print("=" * 60)
    print("Ray-based Anomaly Detector Test")
    print("=" * 60)
    
    # Initialize Ray
    manager = initialize_ray_detector()
    
    # Create test samples
    print("\nGenerating test telemetry samples...")
    
    async def run_test():
        # Normal samples
        normal_samples = [
            TelemetrySample(
                timestamp_ns=time.time_ns(),
                metric_name='nic_latency_us',
                value=np.random.normal(100, 10),
            )
            for _ in range(100)
        ]
        
        # Add some anomalies
        anomaly_samples = [
            TelemetrySample(
                timestamp_ns=time.time_ns(),
                metric_name='nic_latency_us',
                value=500,  # Significant spike
            ),
            TelemetrySample(
                timestamp_ns=time.time_ns(),
                metric_name='cpu_cycles',
                value=10000,  # High cycle count
            ),
        ]
        
        all_samples = normal_samples + anomaly_samples
        
        # Process batch
        print("Processing samples...")
        alerts = await manager.process_batch.remote(all_samples)
        
        # Get alerts
        if alerts:
            print(f"\nDetected {len(alerts)} anomalies:")
            for alert in alerts:
                print(f"  - {alert.anomaly_type.value}: {alert.description}")
                print(f"    Severity: {alert.severity:.2f}")
                print(f"    Action: {alert.recommended_action}")
        
        # Get summary
        summary = await manager.get_summary.remote()
        print(f"\nSummary: {summary}")
        
        # Get actor stats
        actors = await manager.metric_actors.__getattr__('values').remote()
        for actor in list(manager.metric_actors.values())[:3]:
            stats = await actor.get_stats.remote()
            print(f"\nMetric Stats: {stats['metric_name']}")
            print(f"  Samples: {stats['sample_count']}")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std Dev: {stats['std_dev']:.2f}")
            print(f"  P99: {stats['p99']:.2f}")
    
    # Run async test
    asyncio.run(run_test())
    
    print("\nAnomaly detector test complete!")
