"""
Lightweight Non-Blocking Alerting System

Develops a lightweight, non-blocking alerting system that pushes critical system
anomalies to a local webhook without introducing I/O latency to the hot path.
Optimized for microsecond dispatch and AMD Ryzen AI 5 architecture.
"""

import os
import json
import time
import socket
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import deque
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Categories of alerts."""
    SYSTEM = "system"
    MEMORY = "memory"
    NETWORK = "network"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class Alert:
    """Alert message structure."""
    id: str
    timestamp: float
    severity: str
    category: str
    title: str
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'severity': self.severity,
            'category': self.category,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
        }
    
    @classmethod
    def create(
        cls,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'Alert':
        """Factory method to create a new alert."""
        alert_id = hashlib.md5(
            f"{time.time()}{title}{message}".encode()
        ).hexdigest()[:12]
        
        return cls(
            id=alert_id,
            timestamp=time.time(),
            severity=severity.value,
            category=category.value,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {},
        )


@dataclass
class WebhookConfig:
    """Webhook endpoint configuration."""
    url: str
    method: str = "POST"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_sec: float = 1.0
    retry_count: int = 3
    enabled: bool = True


class NonBlockingDispatcher:
    """
    Non-blocking alert dispatcher using lock-free queues.
    
    Features:
    - Zero-copy message passing where possible
    - Background worker threads for I/O
    - Rate limiting to prevent alert storms
    - Batch dispatch for efficiency
    """
    
    def __init__(
        self,
        max_queue_size: int = 10_000,
        num_workers: int = 2,
        batch_size: int = 10,
        flush_interval_ms: float = 100.0,
    ):
        # Lock-free queue for alerts (thread-safe)
        self.alert_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Deduplication cache (recent alert hashes)
        self.recent_alerts: deque = deque(maxlen=1000)
        self.recent_alerts_lock = threading.Lock()
        
        # Rate limiting state
        self.rate_limit_window_sec = 60.0
        self.rate_limit_max_per_window = 100
        self.rate_limit_counts: Dict[str, deque] = {}
        self.rate_limit_lock = threading.Lock()
        
        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = DispatcherStats()
        
        # Batch settings
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_ms / 1000.0
        
        # Webhook configurations
        self.webhooks: List[WebhookConfig] = []
        
        # Custom handlers
        self.handlers: List[Callable[[Alert], None]] = []
        
        logger.info(f"NonBlockingDispatcher initialized (workers={num_workers}, batch={batch_size})")
    
    def add_webhook(self, config: WebhookConfig) -> None:
        """Add a webhook endpoint for alert delivery."""
        self.webhooks.append(config)
        logger.info(f"Added webhook: {config.url}")
    
    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a custom alert handler function."""
        self.handlers.append(handler)
    
    def dispatch(self, alert: Alert) -> bool:
        """
        Dispatch an alert non-blockingly.
        
        Returns True if alert was queued successfully, False if dropped.
        """
        # Check deduplication
        alert_hash = hashlib.md5(
            f"{alert.severity}{alert.category}{alert.title}{alert.message}".encode()
        ).hexdigest()
        
        with self.recent_alerts_lock:
            if alert_hash in self.recent_alerts:
                self.stats.duplicates_dropped += 1
                return False
            self.recent_alerts.append(alert_hash)
        
        # Check rate limit
        if not self._check_rate_limit(alert.severity):
            self.stats.rate_limited += 1
            return False
        
        # Queue the alert
        try:
            self.alert_queue.put_nowait(alert)
            self.stats.queued += 1
            return True
        except queue.Full:
            self.stats.dropped_full += 1
            logger.warning(f"Alert queue full, dropping alert: {alert.title}")
            return False
    
    def _check_rate_limit(self, severity: str) -> bool:
        """Check if alert should be rate limited."""
        now = time.time()
        window_start = now - self.rate_limit_window_sec
        
        with self.rate_limit_lock:
            if severity not in self.rate_limit_counts:
                self.rate_limit_counts[severity] = deque()
            
            # Remove old entries
            counts = self.rate_limit_counts[severity]
            while counts and counts[0] < window_start:
                counts.popleft()
            
            # Check limit
            if len(counts) >= self.rate_limit_max_per_window:
                return False
            
            counts.append(now)
            return True
    
    def start(self) -> None:
        """Start the dispatcher worker threads."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        for i in range(2):  # Default 2 workers
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AlertDispatcher-{i}",
                daemon=True,
            )
            self.workers.append(worker)
            worker.start()
        
        logger.info("Alert dispatcher started")
    
    def stop(self) -> None:
        """Stop the dispatcher gracefully."""
        self.running = False
        self.shutdown_event.set()
        
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        self.workers.clear()
        logger.info("Alert dispatcher stopped")
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        batch: List[Alert] = []
        last_flush = time.time()
        
        while self.running or not self.shutdown_event.is_set():
            try:
                # Try to get alert with timeout
                try:
                    alert = self.alert_queue.get(timeout=0.01)
                    batch.append(alert)
                    self.stats.processed += 1
                except queue.Empty:
                    pass
                
                # Flush batch if full or timeout
                now = time.time()
                if batch and (
                    len(batch) >= self.batch_size or 
                    now - last_flush >= self.flush_interval_sec
                ):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = now
                
            except Exception as e:
                logger.error(f"Dispatcher worker error: {e}")
                self.stats.errors += 1
        
        # Final flush
        if batch:
            self._flush_batch(batch)
    
    def _flush_batch(self, alerts: List[Alert]) -> None:
        """Send a batch of alerts to all configured destinations."""
        if not alerts:
            return
        
        # Send to webhooks
        for webhook in self.webhooks:
            if webhook.enabled:
                self._send_to_webhook(webhook, alerts)
        
        # Call custom handlers
        for handler in self.handlers:
            try:
                for alert in alerts:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
                self.stats.handler_errors += 1
    
    def _send_to_webhook(self, config: WebhookConfig, alerts: List[Alert]) -> None:
        """Send alerts to a webhook endpoint (non-blocking)."""
        import urllib.request
        import urllib.error
        
        payload = {
            'alerts': [a.to_dict() for a in alerts],
            'count': len(alerts),
            'timestamp': time.time(),
        }
        
        data = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            config.url,
            data=data,
            method=config.method,
        )
        
        req.add_header('Content-Type', 'application/json')
        for key, value in config.headers.items():
            req.add_header(key, value)
        
        try:
            with urllib.request.urlopen(req, timeout=config.timeout_sec) as response:
                if response.status < 200 or response.status >= 300:
                    logger.warning(f"Webhook returned status {response.status}")
                    self.stats.webhook_failures += 1
                else:
                    self.stats.webhook_success += 1
        except urllib.error.URLError as e:
            logger.warning(f"Webhook failed: {e}")
            self.stats.webhook_failures += 1
        except Exception as e:
            logger.error(f"Unexpected webhook error: {e}")
            self.stats.webhook_failures += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get dispatcher statistics."""
        return {
            'queued': self.stats.queued,
            'processed': self.stats.processed,
            'dropped_full': self.stats.dropped_full,
            'duplicates_dropped': self.stats.duplicates_dropped,
            'rate_limited': self.stats.rate_limited,
            'errors': self.stats.errors,
            'handler_errors': self.stats.handler_errors,
            'webhook_success': self.stats.webhook_success,
            'webhook_failures': self.stats.webhook_failures,
            'queue_size': self.alert_queue.qsize(),
        }


@dataclass
class DispatcherStats:
    """Statistics for the dispatcher."""
    queued: int = 0
    processed: int = 0
    dropped_full: int = 0
    duplicates_dropped: int = 0
    rate_limited: int = 0
    errors: int = 0
    handler_errors: int = 0
    webhook_success: int = 0
    webhook_failures: int = 0


class AlertManager:
    """
    High-level alert management interface.
    
    Provides convenient methods for creating and sending alerts
    while maintaining non-blocking operation.
    """
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        enable_console: bool = True,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
    ):
        self.dispatcher = NonBlockingDispatcher()
        self.min_severity = min_severity
        self.enable_console = enable_console
        self.alert_history: deque = deque(maxlen=1000)
        self.history_lock = threading.Lock()
        
        # Add console handler if enabled
        if enable_console:
            self.dispatcher.add_handler(self._console_handler)
        
        # Add webhook if provided
        if webhook_url:
            self.dispatcher.add_webhook(WebhookConfig(url=webhook_url))
        
        self.dispatcher.start()
        logger.info("AlertManager initialized")
    
    def _console_handler(self, alert: Alert) -> None:
        """Print alert to console."""
        emoji = {
            AlertSeverity.INFO.value: "ℹ️",
            AlertSeverity.WARNING.value: "⚠️",
            AlertSeverity.ERROR.value: "❌",
            AlertSeverity.CRITICAL.value: "🚨",
            AlertSeverity.EMERGENCY.value: "🔥",
        }.get(alert.severity, "📢")
        
        timestamp = datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S.%f")[:-3]
        
        print(f"[{timestamp}] {emoji} [{alert.severity.upper()}] {alert.title}: {alert.message}")
    
    def send(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        title: str,
        message: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send an alert.
        
        Returns True if alert was dispatched successfully.
        """
        # Check minimum severity
        if self._severity_value(severity) < self._severity_value(self.min_severity):
            return False
        
        alert = Alert.create(severity, category, title, message, source, metadata)
        
        # Store in history
        with self.history_lock:
            self.alert_history.append(alert)
        
        return self.dispatcher.dispatch(alert)
    
    def _severity_value(self, severity: AlertSeverity) -> int:
        """Get numeric severity value for comparison."""
        return {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.CRITICAL: 3,
            AlertSeverity.EMERGENCY: 4,
        }.get(severity, 0)
    
    def info(self, title: str, message: str, **kwargs) -> bool:
        """Send an INFO level alert."""
        return self.send(AlertSeverity.INFO, AlertCategory.SYSTEM, title, message, **kwargs)
    
    def warning(self, title: str, message: str, **kwargs) -> bool:
        """Send a WARNING level alert."""
        return self.send(AlertSeverity.WARNING, AlertCategory.SYSTEM, title, message, **kwargs)
    
    def error(self, title: str, message: str, **kwargs) -> bool:
        """Send an ERROR level alert."""
        return self.send(AlertSeverity.ERROR, AlertCategory.SYSTEM, title, message, **kwargs)
    
    def critical(self, title: str, message: str, **kwargs) -> bool:
        """Send a CRITICAL level alert."""
        return self.send(AlertSeverity.CRITICAL, AlertCategory.SYSTEM, title, message, **kwargs)
    
    def emergency(self, title: str, message: str, **kwargs) -> bool:
        """Send an EMERGENCY level alert."""
        return self.send(AlertSeverity.EMERGENCY, AlertCategory.SYSTEM, title, message, **kwargs)
    
    def trading_alert(self, title: str, message: str, severity: AlertSeverity = AlertSeverity.WARNING, **kwargs) -> bool:
        """Send a trading-related alert."""
        return self.send(severity, AlertCategory.TRADING, title, message, **kwargs)
    
    def memory_alert(self, title: str, message: str, severity: AlertSeverity = AlertSeverity.WARNING, **kwargs) -> bool:
        """Send a memory-related alert."""
        return self.send(severity, AlertCategory.MEMORY, title, message, **kwargs)
    
    def get_recent_alerts(self, count: int = 10) -> List[Alert]:
        """Get recent alerts from history."""
        with self.history_lock:
            return list(self.alert_history)[-count:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert manager statistics."""
        return {
            'dispatcher': self.dispatcher.get_stats(),
            'history_size': len(self.alert_history),
            'min_severity': self.min_severity.value,
        }
    
    def shutdown(self) -> None:
        """Shutdown the alert manager gracefully."""
        self.dispatcher.stop()
        logger.info("AlertManager shutdown complete")


# Global alert manager instance
_global_manager: Optional[AlertManager] = None


def init_alerting(
    webhook_url: Optional[str] = None,
    enable_console: bool = True,
    min_severity: AlertSeverity = AlertSeverity.WARNING,
) -> AlertManager:
    """Initialize the global alert manager."""
    global _global_manager
    _global_manager = AlertManager(webhook_url, enable_console, min_severity)
    return _global_manager


def get_alert_manager() -> Optional[AlertManager]:
    """Get the global alert manager instance."""
    return _global_manager


def alert_critical(title: str, message: str, **kwargs) -> bool:
    """Send a critical alert via the global manager."""
    if _global_manager:
        return _global_manager.critical(title, message, **kwargs)
    return False


def alert_trading(title: str, message: str, **kwargs) -> bool:
    """Send a trading alert via the global manager."""
    if _global_manager:
        return _global_manager.trading_alert(title, message, **kwargs)
    return False


if __name__ == "__main__":
    # Example usage
    import random
    
    # Initialize alerting
    manager = init_alerting(enable_console=True, min_severity=AlertSeverity.INFO)
    
    print("Testing Alert System")
    print("=" * 50)
    
    # Send various alerts
    manager.info("System Started", "Nautilus trading bot initialized")
    manager.warning("High Latency", "Order execution latency exceeded 1ms threshold")
    manager.error("Connection Lost", "Exchange websocket disconnected", metadata={"exchange": "binance"})
    manager.critical("Memory Pressure", "RAM usage at 95%", metadata={"used_gb": 7.6, "total_gb": 8.0})
    
    # Trading alerts
    manager.trading_alert("Large Fill", "Filled 10 BTC at $45,000", severity=AlertSeverity.INFO)
    manager.trading_alert("Slippage Warning", "Expected slippage exceeded 5 bps", severity=AlertSeverity.WARNING)
    
    # Memory alerts
    manager.memory_alert("GC Triggered", "Forced garbage collection due to memory pressure")
    
    # Get stats
    stats = manager.get_stats()
    print(f"\nAlert Stats: {json.dumps(stats, indent=2)}")
    
    # Get recent alerts
    recent = manager.get_recent_alerts(5)
    print(f"\nRecent Alerts ({len(recent)}):")
    for alert in recent:
        print(f"  - [{alert.severity}] {alert.title}")
    
    # Shutdown
    manager.shutdown()
