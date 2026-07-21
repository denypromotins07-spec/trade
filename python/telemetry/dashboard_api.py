"""
Telemetry Dashboard API with FastAPI/WebSocket

Scaffolds a high-performance FastAPI/WebSocket endpoint that streams internal
Rust telemetry to the frontend, strictly rate-limiting payloads to save bandwidth.

Key Features:
- WebSocket streaming for real-time telemetry updates
- Rate limiting to prevent bandwidth exhaustion
- AMD ROCm/DirectML environment checks
- Integration with Rust profiler metrics
- Strict memory limits for Python (4GB quota)
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import OrderedDict
import threading

# FastAPI imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Mock classes for testing
    class FastAPI:
        pass
    class WebSocket:
        pass
    class WebSocketDisconnect(Exception):
        pass

# Check for AMD ROCm/DirectML availability
try:
    import torch
    ROCM_AVAILABLE = torch.cuda.is_available() and torch.version.hip is not None
    DIRECTML_AVAILABLE = False
except ImportError:
    ROCM_AVAILABLE = False
    DIRECTML_AVAILABLE = False


@dataclass
class TelemetryMetric:
    """Single telemetry metric."""
    name: str
    value: float
    timestamp_ms: int
    tags: Dict[str, str] = None


@dataclass
class SystemStats:
    """System-wide statistics."""
    cpu_usage_pct: float
    memory_used_bytes: int
    memory_limit_bytes: int
    network_rx_bytes: int
    network_tx_bytes: int
    active_connections: int
    orders_per_second: float
    latency_p99_us: float


class RateLimiter:
    """Token bucket rate limiter for WebSocket messages."""
    
    def __init__(self, max_messages_per_second: int = 100):
        self.max_rate = max_messages_per_second
        self.tokens = max_messages_per_second
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a token, returns True if allowed."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.max_rate, self.tokens + elapsed * self.max_rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
    
    async def wait_for_token(self, timeout_ms: int = 1000) -> bool:
        """Wait for a token to become available."""
        start = time.time()
        while time.time() - start < timeout_ms / 1000:
            if self.acquire():
                return True
            await asyncio.sleep(0.001)
        return False


class TelemetryCollector:
    """Collects and aggregates telemetry metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: OrderedDict[str, TelemetryMetric] = OrderedDict()
        self.max_history = max_history
        self._lock = threading.Lock()
    
    def add_metric(self, metric: TelemetryMetric):
        """Add a new metric."""
        with self._lock:
            if len(self.metrics) >= self.max_history:
                self.metrics.popitem(last=False)
            self.metrics[metric.name] = metric
    
    def get_latest_metrics(self, count: int = 50) -> List[TelemetryMetric]:
        """Get latest N metrics."""
        with self._lock:
            items = list(self.metrics.values())[-count:]
            return items
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self.metrics.clear()


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = threading.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept and register a new connection."""
        await websocket.accept()
        with self._lock:
            self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a connection."""
        with self._lock:
            self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        with self._lock:
            connections = list(self.active_connections)
        
        disconnected = []
        for conn in connections:
            try:
                await conn.send_json(message)
            except Exception:
                disconnected.append(conn)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        """Send message to specific client."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        with self._lock:
            return len(self.active_connections)


class DashboardAPI:
    """Main dashboard API server."""
    
    def __init__(
        self,
        rate_limit_per_second: int = 100,
        max_memory_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB limit
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for DashboardAPI")
        
        self.app = FastAPI(title="Nautilus/Ray Telemetry Dashboard")
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.rate_limiter = RateLimiter(rate_limit_per_second)
        self.collector = TelemetryCollector()
        self.connection_manager = ConnectionManager()
        self.max_memory_bytes = max_memory_bytes
        self.current_memory_bytes = 0
        
        # Setup routes
        self._setup_routes()
        
        # Background task flag
        self._running = False
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp_ms": int(time.time() * 1000)}
        
        @self.app.get("/metrics")
        async def get_metrics(count: int = 50):
            metrics = self.collector.get_latest_metrics(count)
            return {"metrics": [asdict(m) for m in metrics]}
        
        @self.app.get("/stats")
        async def get_stats():
            stats = self._get_system_stats()
            return asdict(stats)
        
        @self.app.websocket("/ws/telemetry")
        async def websocket_telemetry(websocket: WebSocket):
            await self._handle_telemetry_websocket(websocket)
        
        @self.app.get("/amd-info")
        async def amd_info():
            return {
                "rocm_available": ROCM_AVAILABLE,
                "directml_available": DIRECTML_AVAILABLE,
                "gpu_acceleration_enabled": ROCM_AVAILABLE or DIRECTML_AVAILABLE
            }
    
    def _get_system_stats(self) -> SystemStats:
        """Get current system statistics."""
        # In production, would query actual system metrics
        return SystemStats(
            cpu_usage_pct=0.0,
            memory_used_bytes=self.current_memory_bytes,
            memory_limit_bytes=self.max_memory_bytes,
            network_rx_bytes=0,
            network_tx_bytes=0,
            active_connections=self.connection_manager.get_connection_count(),
            orders_per_second=0.0,
            latency_p99_us=0.0
        )
    
    async def _handle_telemetry_websocket(self, websocket: WebSocket):
        """Handle telemetry WebSocket connection."""
        await self.connection_manager.connect(websocket)
        
        try:
            while True:
                # Wait for rate limit token
                if not await self.rate_limiter.wait_for_token(timeout_ms=100):
                    continue
                
                # Get latest metrics
                metrics = self.collector.get_latest_metrics(10)
                
                if metrics:
                    message = {
                        "type": "telemetry",
                        "timestamp_ms": int(time.time() * 1000),
                        "metrics": [asdict(m) for m in metrics],
                        "stats": asdict(self._get_system_stats())
                    }
                    
                    await self.connection_manager.send_personal(websocket, message)
                
                await asyncio.sleep(0.01)  # 10ms update interval
                
        except WebSocketDisconnect:
            self.connection_manager.disconnect(websocket)
        except Exception as e:
            self.connection_manager.disconnect(websocket)
    
    def ingest_rust_telemetry(self, metrics: List[Dict[str, Any]]):
        """Ingest telemetry from Rust profiler."""
        for m in metrics:
            metric = TelemetryMetric(
                name=m.get("name", "unknown"),
                value=m.get("value", 0.0),
                timestamp_ms=m.get("timestamp_ms", int(time.time() * 1000)),
                tags=m.get("tags", {})
            )
            
            # Check memory limit
            estimated_size = len(json.dumps(asdict(metric)))
            if self.current_memory_bytes + estimated_size > self.max_memory_bytes:
                # Drop oldest metrics to stay under limit
                self.collector.clear()
                self.current_memory_bytes = 0
            
            self.collector.add_metric(metric)
            self.current_memory_bytes += estimated_size
    
    async def start_background_tasks(self):
        """Start background tasks."""
        self._running = True
    
    def stop(self):
        """Stop the API server."""
        self._running = False


def check_amd_environment() -> Dict[str, Any]:
    """Check AMD ROCm/DirectML environment."""
    env_info = {
        "rocm_available": ROCM_AVAILABLE,
        "directml_available": DIRECTML_AVAILABLE,
        "fastapi_available": FASTAPI_AVAILABLE,
        "gpu_acceleration_enabled": ROCM_AVAILABLE or DIRECTML_AVAILABLE,
        "recommendations": []
    }
    
    if ROCM_AVAILABLE:
        env_info["recommendations"].append(
            "ROCm detected - GPU acceleration available for compute-intensive operations"
        )
    
    if DIRECTML_AVAILABLE:
        env_info["recommendations"].append(
            "DirectML detected - Windows GPU acceleration available"
        )
    
    if not FASTAPI_AVAILABLE:
        env_info["recommendations"].append(
            "WARNING: FastAPI not available - install with 'pip install fastapi uvicorn'"
        )
    
    return env_info


# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Check environment
    env = check_amd_environment()
    print(f"Environment: {env}")
    
    if FASTAPI_AVAILABLE:
        # Create API instance
        api = DashboardAPI(rate_limit_per_second=50)
        
        # Run server
        uvicorn.run(
            api.app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
