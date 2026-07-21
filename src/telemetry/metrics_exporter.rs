//! Prometheus Metrics Exporter: Non-Blocking UDP Export
//! 
//! Exports Prometheus-compatible metrics via a non-blocking UDP socket,
//! ensuring network I/O never contends with the main Binance WebSocket
//! ingestion thread. Optimized for microsecond latency on AMD Ryzen AI 5.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::net::{SocketAddr, UdpSocket};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Metric type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
}

/// Single metric value
#[derive(Debug, Clone)]
pub struct MetricValue {
    /// Metric name
    pub name: String,
    /// Metric value (f64 for Prometheus compatibility)
    pub value: f64,
    /// Metric type
    pub metric_type: MetricType,
    /// Labels/key-value pairs
    pub labels: HashMap<String, String>,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
}

impl MetricValue {
    /// Create a new counter metric
    pub fn counter(name: &str, value: u64) -> Self {
        Self {
            name: name.to_string(),
            value: value as f64,
            metric_type: MetricType::Counter,
            labels: HashMap::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Create a new gauge metric
    pub fn gauge(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
            metric_type: MetricType::Gauge,
            labels: HashMap::new(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Add a label to the metric
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.to_string(), value.to_string());
        self
    }

    /// Format metric in Prometheus text format
    pub fn format_prometheus(&self) -> String {
        let mut output = String::new();
        
        // Add TYPE comment
        let type_str = match self.metric_type {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
        };
        output.push_str(&format!("# TYPE {} {}\n", self.name, type_str));
        
        // Format labels
        let labels_str: Vec<String> = self.labels
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();
        
        if labels_str.is_empty() {
            output.push_str(&format!("{} {}\n", self.name, self.value));
        } else {
            output.push_str(&format!(
                "{}{{{}}} {}\n",
                self.name,
                labels_str.join(","),
                self.value
            ));
        }
        
        output
    }
}

/// Batch of metrics ready for export
#[derive(Debug, Clone)]
pub struct MetricsBatch {
    /// Metrics in the batch
    pub metrics: Vec<MetricValue>,
    /// Batch sequence number
    pub sequence: u64,
    /// Creation timestamp
    pub created_at: Instant,
}

impl MetricsBatch {
    /// Create a new empty batch
    pub fn new(sequence: u64) -> Self {
        Self {
            metrics: Vec::new(),
            sequence,
            created_at: Instant::now(),
        }
    }

    /// Add a metric to the batch
    pub fn add(&mut self, metric: MetricValue) {
        self.metrics.push(metric);
    }

    /// Format entire batch in Prometheus text format
    pub fn format_batch(&self) -> String {
        let mut output = String::new();
        for metric in &self.metrics {
            output.push_str(&metric.format_prometheus());
        }
        output
    }

    /// Get batch size in bytes when formatted
    pub fn size_bytes(&self) -> usize {
        self.format_batch().len()
    }
}

/// Non-blocking UDP metrics exporter
pub struct PrometheusExporter {
    /// UDP socket for sending metrics
    socket: Option<UdpSocket>,
    /// Target address for Prometheus pushgateway or collector
    target_addr: SocketAddr,
    /// Enabled flag
    enabled: AtomicBool,
    /// Metrics sent counter
    metrics_sent: AtomicU64,
    /// Metrics dropped counter (queue full)
    metrics_dropped: AtomicU64,
    /// Current batch sequence
    batch_sequence: AtomicU64,
    /// Pending batch
    pending_batch: Arc<std::sync::Mutex<MetricsBatch>>,
    /// Maximum batch size before flush
    max_batch_size: usize,
    /// Flush interval
    flush_interval: Duration,
    /// Last flush time
    last_flush: std::sync::Mutex<Instant>,
}

unsafe impl Send for PrometheusExporter {}
unsafe impl Sync for PrometheusExporter {}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    /// 
    /// # Arguments
    /// * `target_addr` - Address of Prometheus pushgateway or metrics collector
    /// * `max_batch_size` - Maximum metrics per batch before automatic flush
    /// * `flush_interval` - Time interval between automatic flushes
    pub fn new(
        target_addr: SocketAddr,
        max_batch_size: usize,
        flush_interval: Duration,
    ) -> Self {
        // Create non-blocking UDP socket
        let socket = std::net::UdpSocket::bind("0.0.0.0:0")
            .ok()
            .and_then(|s| {
                s.set_nonblocking(true).ok()?;
                Some(s)
            });

        Self {
            socket,
            target_addr,
            enabled: AtomicBool::new(false),
            metrics_sent: AtomicU64::new(0),
            metrics_dropped: AtomicU64::new(0),
            batch_sequence: AtomicU64::new(0),
            pending_batch: Arc::new(std::sync::Mutex::new(MetricsBatch::new(0))),
            max_batch_size,
            flush_interval,
            last_flush: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Enable metrics export
    #[inline]
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Disable metrics export
    #[inline]
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Check if exporter is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Record a metric (non-blocking, adds to pending batch)
    #[inline]
    pub fn record(&self, metric: MetricValue) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }

        if let Ok(mut batch) = self.pending_batch.lock() {
            if batch.metrics.len() >= self.max_batch_size {
                // Batch full, try to flush
                drop(batch);
                self.flush();
                
                // Try again with new batch
                if let Ok(mut new_batch) = self.pending_batch.lock() {
                    new_batch.add(metric);
                    self.metrics_sent.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.metrics_dropped.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                batch.add(metric);
                self.metrics_sent.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            self.metrics_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Increment a counter metric
    #[inline]
    pub fn inc_counter(&self, name: &str, value: u64) {
        self.record(MetricValue::counter(name, value));
    }

    /// Set a gauge metric
    #[inline]
    pub fn set_gauge(&self, name: &str, value: f64) {
        self.record(MetricValue::gauge(name, value));
    }

    /// Flush pending batch to UDP socket (non-blocking)
    pub fn flush(&self) -> bool {
        let socket = match &self.socket {
            Some(s) => s,
            None => return false,
        };

        // Swap current batch with new empty one
        let batch = {
            let mut current = self.pending_batch.lock().unwrap();
            let seq = self.batch_sequence.fetch_add(1, Ordering::Relaxed);
            let mut new_batch = MetricsBatch::new(seq + 1);
            std::mem::swap(&mut *current, &mut new_batch);
            new_batch
        };

        if batch.metrics.is_empty() {
            return true;
        }

        // Format and send
        let data = batch.format_batch();
        
        // Non-blocking send
        match socket.send_to(data.as_bytes(), self.target_addr) {
            Ok(_) => {
                *self.last_flush.lock().unwrap() = Instant::now();
                true
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Socket buffer full, drop batch
                self.metrics_dropped.fetch_add(batch.metrics.len() as u64, Ordering::Relaxed);
                true // Consider success - we tried
            }
            Err(_) => {
                self.metrics_dropped.fetch_add(batch.metrics.len() as u64, Ordering::Relaxed);
                false
            }
        }
    }

    /// Auto-flush based on interval
    pub fn maybe_auto_flush(&self) {
        let now = Instant::now();
        let last = *self.last_flush.lock().unwrap();
        
        if now - last >= self.flush_interval {
            self.flush();
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> ExporterStats {
        ExporterStats {
            enabled: self.is_enabled(),
            metrics_sent: self.metrics_sent.load(Ordering::Acquire),
            metrics_dropped: self.metrics_dropped.load(Ordering::Acquire),
            pending_count: self.pending_batch.lock().map(|b| b.metrics.len()).unwrap_or(0),
            batch_sequence: self.batch_sequence.load(Ordering::Acquire),
        }
    }

    /// Reset exporter state
    pub fn reset(&self) {
        self.disable();
        self.metrics_sent.store(0, Ordering::Relaxed);
        self.metrics_dropped.store(0, Ordering::Relaxed);
        self.batch_sequence.store(0, Ordering::Relaxed);
        *self.pending_batch.lock().unwrap() = MetricsBatch::new(0);
        *self.last_flush.lock().unwrap() = Instant::now();
    }
}

/// Exporter statistics
#[derive(Debug, Clone)]
pub struct ExporterStats {
    pub enabled: bool,
    pub metrics_sent: u64,
    pub metrics_dropped: u64,
    pub pending_count: usize,
    pub batch_sequence: u64,
}

/// Background flush thread for periodic exports
pub fn start_background_flush(exporter: Arc<PrometheusExporter>, interval: Duration) {
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(interval);
            exporter.maybe_auto_flush();
        }
    });
}

/// Common trading metrics helpers
pub mod trading_metrics {
    use super::*;

    /// Order book depth metrics
    pub fn record_orderbook_depth(exporter: &PrometheusExporter, symbol: &str, side: &str, depth: u64) {
        exporter.set_gauge(
            &format!("binance_orderbook_depth_{}", symbol),
            depth as f64,
        );
        // Note: labels would be added in production
    }

    /// Trade execution metrics
    pub fn record_trade(exporter: &PrometheusExporter, symbol: &str, side: &str, price: f64, quantity: f64) {
        exporter.inc_counter("binance_trades_total", 1);
        exporter.set_gauge(&format!("binance_trade_price_{}", symbol), price);
        exporter.set_gauge(&format!("binance_trade_quantity_{}", symbol), quantity);
    }

    /// Latency metrics
    pub fn record_latency(exporter: &PrometheusExporter, operation: &str, latency_us: u64) {
        exporter.set_gauge(&format!("nautilus_latency_us_{}", operation), latency_us as f64);
        exporter.inc_counter("nautilus_operations_total", 1);
    }

    /// PnL metrics
    pub fn record_pnl(exporter: &PrometheusExporter, symbol: &str, pnl: f64) {
        exporter.set_gauge(&format!("nautilus_pnl_{}", symbol), pnl);
    }

    /// Queue position metrics
    pub fn record_queue_position(exporter: &PrometheusExporter, symbol: &str, position: u64) {
        exporter.set_gauge(&format!("nautilus_queue_position_{}", symbol), position as f64);
    }

    /// VPIN metrics
    pub fn record_vpin(exporter: &PrometheusExporter, symbol: &str, vpin: f64) {
        exporter.set_gauge(&format!("nautilus_vpin_{}", symbol), vpin);
    }

    /// Hardware telemetry metrics
    pub fn record_cpu_temp(exporter: &PrometheusExporter, core: usize, temp: f32) {
        exporter.set_gauge(&format!("amd_cpu_temp_celsius_core_{}", core), temp as f64);
    }

    /// Memory usage metrics
    pub fn record_memory_usage(exporter: &PrometheusExporter, used_mb: u64, total_mb: u64) {
        exporter.set_gauge("nautilus_memory_used_mb", used_mb as f64);
        exporter.set_gauge("nautilus_memory_total_mb", total_mb as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_metric_value_format() {
        let metric = MetricValue::counter("test_counter", 42)
            .with_label("symbol", "BTCUSDT")
            .with_label("side", "buy");

        let formatted = metric.format_prometheus();
        assert!(formatted.contains("TYPE test_counter counter"));
        assert!(formatted.contains("symbol=\"BTCUSDT\""));
        assert!(formatted.contains("side=\"buy\""));
        assert!(formatted.contains("42"));
    }

    #[test]
    fn test_exporter_creation() {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 9091);
        let exporter = PrometheusExporter::new(addr, 100, Duration::from_secs(1));
        
        assert!(!exporter.is_enabled());
        
        exporter.enable();
        assert!(exporter.is_enabled());
    }

    #[test]
    fn test_record_metrics() {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 9091);
        let exporter = PrometheusExporter::new(addr, 100, Duration::from_secs(1));
        exporter.enable();

        exporter.inc_counter("test_ops", 1);
        exporter.set_gauge("test_value", 3.14);

        let stats = exporter.get_stats();
        assert_eq!(stats.metrics_sent, 2);
        assert_eq!(stats.pending_count, 2);
    }

    #[test]
    fn test_trading_metrics_helpers() {
        let addr = SocketAddr::new(Ipv4Addr::LOCALHOST.into(), 9091);
        let exporter = PrometheusExporter::new(addr, 100, Duration::from_secs(1));
        exporter.enable();

        trading_metrics::record_orderbook_depth(&exporter, "BTCUSDT", "bid", 1000);
        trading_metrics::record_trade(&exporter, "BTCUSDT", "buy", 50000.0, 0.5);
        trading_metrics::record_latency(&exporter, "websocket_msg", 100);

        let stats = exporter.get_stats();
        assert!(stats.metrics_sent >= 3);
    }
}
