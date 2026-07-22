"""
Ray Dashboard Extension for Memory Diagnostics

Develops a Ray dashboard extension that visualizes object store spills
and worker memory pressure, alerting the system before Python OOM kills occur.
Integrates with the 4GB RAM quota enforcement across all workers.

Features:
- Real-time object store spill monitoring
- Per-worker memory pressure visualization
- Pre-OOM alerting system
- Integration with Ray's internal metrics
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import threading

import ray
from ray.experimental.internal_kv import _internal_kv_get, _internal_kv_put
from ray.dashboard.modules.dashboard_module import DashboardModule


# =============================================================================
# Configuration
# =============================================================================

PYTHON_RAM_QUOTA = 4 * 1024 * 1024 * 1024  # 4GB per worker
OBJECT_STORE_WARNING_THRESHOLD = 0.8  # 80% of object store capacity
MEMORY_PRESSURE_CRITICAL = 0.95  # 95% triggers emergency GC

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WorkerMemoryStatus:
    """Memory status for a single Ray worker."""
    worker_id: str
    pid: int
    current_bytes: int
    peak_bytes: int
    quota_bytes: int
    usage_percent: float
    gc_count: int
    is_python_worker: bool
    timestamp_ns: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectStoreStatus:
    """Object store status including spill information."""
    used_bytes: int
    available_bytes: int
    total_bytes: int
    usage_percent: float
    spilled_bytes: int
    spill_count: int
    reconstruction_count: int
    timestamp_ns: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryAlert:
    """Memory pressure alert."""
    alert_type: str  # 'warning', 'critical', 'oom_imminent'
    source: str  # 'worker', 'object_store', 'global'
    message: str
    severity: int  # 1-3 (low to high)
    timestamp_ns: int
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Memory Monitor
# =============================================================================

class RayMemoryMonitor:
    """
    Monitors memory usage across Ray cluster with focus on Python quotas.
    """
    
    def __init__(self):
        self.alerts: List[MemoryAlert] = []
        self.max_alerts = 1000
        self.worker_statuses: Dict[str, WorkerMemoryStatus] = {}
        self.object_store_status: Optional[ObjectStoreStatus] = None
        self._lock = threading.Lock()
        
    def collect_worker_metrics(self) -> Dict[str, WorkerMemoryStatus]:
        """Collect memory metrics from all workers."""
        try:
            import psutil
            
            # Get all Ray worker processes
            workers = ray.nodes()
            statuses = {}
            
            for node in workers:
                node_id = node.get("NodeID", "unknown")
                
                # Get node-level memory info
                try:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    
                    status = WorkerMemoryStatus(
                        worker_id=node_id,
                        pid=os.getpid(),
                        current_bytes=mem_info.rss,
                        peak_bytes=getattr(mem_info, 'vms', mem_info.rss),
                        quota_bytes=PYTHON_RAM_QUOTA,
                        usage_percent=(mem_info.rss / PYTHON_RAM_QUOTA) * 100,
                        gc_count=0,  # Would track via gc callbacks
                        is_python_worker=True,
                        timestamp_ns=time.time_ns()
                    )
                    statuses[node_id] = status
                except Exception as e:
                    logger.warning(f"Failed to get metrics for node {node_id}: {e}")
            
            with self._lock:
                self.worker_statuses.update(statuses)
            
            return statuses
            
        except Exception as e:
            logger.error(f"Error collecting worker metrics: {e}")
            return {}
    
    def collect_object_store_metrics(self) -> Optional[ObjectStoreStatus]:
        """Collect object store metrics including spills."""
        try:
            # Ray internal stats
            import ray._raylet as raylet
            
            # Try to get object store stats
            stats = ray.internal.global_state()
            
            if stats and 'ObjectStoreStats' in stats:
                store_stats = stats['ObjectStoreStats']
                
                used = store_stats.get('used_bytes', 0)
                available = store_stats.get('available_bytes', 0)
                total = used + available
                
                status = ObjectStoreStatus(
                    used_bytes=used,
                    available_bytes=available,
                    total_bytes=total,
                    usage_percent=(used / total * 100) if total > 0 else 0,
                    spilled_bytes=store_stats.get('spilled_bytes', 0),
                    spill_count=store_stats.get('spill_count', 0),
                    reconstruction_count=store_stats.get('reconstruction_count', 0),
                    timestamp_ns=time.time_ns()
                )
                
                with self._lock:
                    self.object_store_status = status
                
                return status
            else:
                # Fallback: estimate from ray memory-stats
                result = ray.memory_summary()
                # Parse the summary (format varies by Ray version)
                return self._parse_memory_summary(result)
                
        except Exception as e:
            logger.warning(f"Error collecting object store metrics: {e}")
            return None
    
    def _parse_memory_summary(self, summary: str) -> Optional[ObjectStoreStatus]:
        """Parse ray memory summary string into structured data."""
        try:
            # Simple parsing - would need enhancement for production
            lines = summary.split('\n')
            used = 0
            total = 0
            
            for line in lines:
                if 'Plasma memory' in line:
                    # Extract numbers from line like "Plasma memory usage: 1.5 GiB / 10.0 GiB"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        values = parts[1].split('/')
                        if len(values) == 2:
                            used = self._parse_size(values[0].strip())
                            total = self._parse_size(values[1].strip())
            
            if total > 0:
                return ObjectStoreStatus(
                    used_bytes=int(used),
                    available_bytes=int(total - used),
                    total_bytes=int(total),
                    usage_percent=(used / total * 100),
                    spilled_bytes=0,  # Would need additional parsing
                    spill_count=0,
                    reconstruction_count=0,
                    timestamp_ns=time.time_ns()
                )
        except Exception:
            pass
        
        return None
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '1.5 GiB' to bytes."""
        try:
            size_str = size_str.strip()
            multipliers = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3}
            
            for suffix, mult in multipliers.items():
                if suffix in size_str:
                    value = float(size_str.replace(suffix, '').strip())
                    return int(value * mult)
            
            return int(float(size_str))
        except Exception:
            return 0
    
    def check_and_alert(self) -> List[MemoryAlert]:
        """Check memory status and generate alerts."""
        new_alerts = []
        
        # Check worker memory
        for worker_id, status in self.worker_statuses.items():
            if status.usage_percent >= 95:
                alert = MemoryAlert(
                    alert_type='oom_imminent',
                    source='worker',
                    message=f"Worker {worker_id} at {status.usage_percent:.1f}% memory usage",
                    severity=3,
                    timestamp_ns=time.time_ns(),
                    details=status.to_dict()
                )
                new_alerts.append(alert)
            elif status.usage_percent >= 85:
                alert = MemoryAlert(
                    alert_type='critical',
                    source='worker',
                    message=f"Worker {worker_id} at {status.usage_percent:.1f}% memory usage",
                    severity=2,
                    timestamp_ns=time.time_ns(),
                    details=status.to_dict()
                )
                new_alerts.append(alert)
            elif status.usage_percent >= 70:
                alert = MemoryAlert(
                    alert_type='warning',
                    source='worker',
                    message=f"Worker {worker_id} at {status.usage_percent:.1f}% memory usage",
                    severity=1,
                    timestamp_ns=time.time_ns(),
                    details=status.to_dict()
                )
                new_alerts.append(alert)
        
        # Check object store
        if self.object_store_status:
            if self.object_store_status.usage_percent >= OBJECT_STORE_WARNING_THRESHOLD * 100:
                alert = MemoryAlert(
                    alert_type='critical',
                    source='object_store',
                    message=f"Object store at {self.object_store_status.usage_percent:.1f}% capacity",
                    severity=2,
                    timestamp_ns=time.time_ns(),
                    details=self.object_store_status.to_dict()
                )
                new_alerts.append(alert)
                
                if self.object_store_status.spilled_bytes > 0:
                    alert = MemoryAlert(
                        alert_type='warning',
                        source='object_store',
                        message=f"Object store has spilled {self.object_store_status.spilled_bytes} bytes",
                        severity=1,
                        timestamp_ns=time.time_ns(),
                        details=self.object_store_status.to_dict()
                    )
                    new_alerts.append(alert)
        
        # Store alerts
        with self._lock:
            self.alerts.extend(new_alerts)
            # Trim old alerts
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
        
        return new_alerts
    
    def get_recent_alerts(self, count: int = 10) -> List[MemoryAlert]:
        """Get most recent alerts."""
        with self._lock:
            return self.alerts[-count:]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self.alerts.clear()


# =============================================================================
# Dashboard Module
# =============================================================================

class MemoryDashboardModule(DashboardModule):
    """
    Ray dashboard module for memory diagnostics.
    
    Provides REST endpoints for memory monitoring and visualization.
    """
    
    def __init__(self, dashboard_context):
        super().__init__(dashboard_context)
        self.monitor = RayMemoryMonitor()
        self._collection_thread = None
        self._running = False
    
    async def run(self, server):
        """Start the dashboard module."""
        logger.info("Starting MemoryDashboardModule")
        
        # Start background collection thread
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collect_loop,
            daemon=True
        )
        self._collection_thread.start()
        
        # Register REST endpoints
        server.add_routes([
            # Memory status endpoint
            ray.webserver.routes.Route(
                "/api/memory/status",
                self.handle_memory_status,
                methods=["GET"]
            ),
            # Alerts endpoint
            ray.webserver.routes.Route(
                "/api/memory/alerts",
                self.handle_alerts,
                methods=["GET", "DELETE"]
            ),
            # Object store status
            ray.webserver.routes.Route(
                "/api/memory/object-store",
                self.handle_object_store,
                methods=["GET"]
            ),
            # Force GC endpoint
            ray.webserver.routes.Route(
                "/api/memory/gc",
                self.handle_force_gc,
                methods=["POST"]
            ),
        ])
    
    def stop(self):
        """Stop the dashboard module."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Stopped MemoryDashboardModule")
    
    def _collect_loop(self):
        """Background loop for collecting metrics."""
        while self._running:
            try:
                # Collect metrics
                self.monitor.collect_worker_metrics()
                self.monitor.collect_object_store_metrics()
                
                # Check and generate alerts
                new_alerts = self.monitor.check_and_alert()
                
                if new_alerts:
                    for alert in new_alerts:
                        logger.warning(
                            f"[MEMORY ALERT] {alert.alert_type}: {alert.message}"
                        )
                
                # Store latest status in KV for other components
                self._store_status()
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
            
            # Collect every second
            time.sleep(1)
    
    def _store_status(self):
        """Store current status in Ray internal KV."""
        try:
            status = {
                'workers': {
                    k: v.to_dict() 
                    for k, v in self.monitor.worker_statuses.items()
                },
                'object_store': (
                    self.monitor.object_store_status.to_dict() 
                    if self.monitor.object_store_status else None
                ),
                'alert_count': len(self.monitor.alerts),
                'timestamp_ns': time.time_ns()
            }
            
            _internal_kv_put(
                b"memory_dashboard_status",
                json.dumps(status).encode(),
                overwrite=True
            )
        except Exception as e:
            logger.warning(f"Failed to store status: {e}")
    
    async def handle_memory_status(self, request):
        """Handle GET /api/memory/status"""
        statuses = {
            k: v.to_dict() 
            for k, v in self.monitor.worker_statuses.items()
        }
        return ray.webserver.routes.json_response({
            'success': True,
            'data': statuses
        })
    
    async def handle_alerts(self, request):
        """Handle GET/DELETE /api/memory/alerts"""
        if request.method == 'DELETE':
            self.monitor.clear_alerts()
            return ray.webserver.routes.json_response({
                'success': True,
                'message': 'Alerts cleared'
            })
        
        alerts = [
            a.to_dict() 
            for a in self.monitor.get_recent_alerts(50)
        ]
        return ray.webserver.routes.json_response({
            'success': True,
            'data': alerts
        })
    
    async def handle_object_store(self, request):
        """Handle GET /api/memory/object-store"""
        if self.monitor.object_store_status:
            return ray.webserver.routes.json_response({
                'success': True,
                'data': self.monitor.object_store_status.to_dict()
            })
        
        return ray.webserver.routes.json_response({
            'success': False,
            'error': 'Object store status not available'
        }, status=404)
    
    async def handle_force_gc(self, request):
        """Handle POST /api/memory/gc - Force garbage collection"""
        import gc
        
        collected = gc.collect()
        
        # Also trigger Ray cleanup
        ray.internal.free()
        
        return ray.webserver.routes.json_response({
            'success': True,
            'data': {
                'objects_collected': collected,
                'timestamp_ns': time.time_ns()
            }
        })


# =============================================================================
# Utility Functions
# =============================================================================

def get_memory_status() -> Dict[str, Any]:
    """Get current memory status from dashboard."""
    try:
        data = _internal_kv_get(b"memory_dashboard_status")
        if data:
            return json.loads(data.decode())
    except Exception:
        pass
    
    return {'error': 'Status not available'}


def enforce_worker_quota() -> bool:
    """
    Enforce 4GB quota on current worker.
    
    Returns True if within quota, False if exceeded.
    """
    import gc
    import psutil
    
    process = psutil.Process(os.getpid())
    current = process.memory_info().rss
    
    if current >= PYTHON_RAM_QUOTA * MEMORY_PRESSURE_CRITICAL:
        # Critical - force aggressive GC
        gc.collect()
        ray.internal.free()
        
        # Re-check
        current = process.memory_info().rss
        if current >= PYTHON_RAM_QUOTA * MEMORY_PRESSURE_CRITICAL:
            logger.critical(
                f"Worker at {current/PYTHON_RAM_QUOTA*100:.1f}% - OOM imminent!"
            )
            return False
    
    return True


if __name__ == "__main__":
    # Test the memory monitor
    ray.init(ignore_reinit_error=True)
    
    monitor = RayMemoryMonitor()
    
    # Collect some metrics
    for _ in range(5):
        monitor.collect_worker_metrics()
        monitor.collect_object_store_metrics()
        alerts = monitor.check_and_alert()
        
        if alerts:
            print(f"Generated {len(alerts)} alerts")
        
        time.sleep(0.5)
    
    # Print status
    print("\n=== Worker Status ===")
    for worker_id, status in monitor.worker_statuses.items():
        print(f"Worker {worker_id}: {status.usage_percent:.1f}%")
    
    print("\n=== Object Store ===")
    if monitor.object_store_status:
        print(f"Usage: {monitor.object_store_status.usage_percent:.1f}%")
        print(f"Spilled: {monitor.object_store_status.spilled_bytes} bytes")
    
    print("\n=== Recent Alerts ===")
    for alert in monitor.get_recent_alerts(5):
        print(f"[{alert.severity}] {alert.alert_type}: {alert.message}")
    
    ray.shutdown()
