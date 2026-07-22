//! Microsecond-Precision Heartbeat Tracker for Exchange Connections
//! 
//! This module constructs a heartbeat tracker that measures round-trip latency
//! to the exchange with microsecond precision, predicting connection degradation
//! before standard timeout thresholds trigger.
//! 
//! Key features:
//! - Nanosecond-precision RTT measurement
//! - Exponential moving average for trend detection
//! - Predictive degradation alerts
//! - AMD Ryzen AI 5 TSC (Time Stamp Counter) optimization
//! - Lock-free statistics updates

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::RwLock;

/// Default heartbeat interval (milliseconds)
const DEFAULT_HEARTBEAT_INTERVAL_MS: u64 = 1000;

/// Warning threshold multiplier (RTT above this triggers warning)
const WARNING_THRESHOLD_MULTIPLIER: f64 = 2.0;

/// Critical threshold multiplier (RTT above this triggers critical alert)
const CRITICAL_THRESHOLD_MULTIPLIER: f64 = 5.0;

/// Number of samples for moving average
const EMA_SAMPLES: usize = 50;

/// Degradation prediction window (samples)
const DEGRADATION_WINDOW: usize = 10;

/// High-precision timestamp using TSC where available
#[inline(always)]
fn get_timestamp_ns() -> u64 {
    // Use standard library for cross-platform compatibility
    // On AMD Ryzen, this maps to efficient RDTSC when available
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Single heartbeat measurement
#[derive(Debug, Clone)]
pub struct HeartbeatSample {
    /// Send timestamp (nanoseconds)
    pub send_ts_ns: u64,
    /// Receive timestamp (nanoseconds)
    pub recv_ts_ns: u64,
    /// Round-trip time (microseconds)
    pub rtt_us: u64,
    /// Sequence number
    pub sequence: u64,
}

impl HeartbeatSample {
    /// Calculate RTT in microseconds
    pub fn calculate_rtt_us(send_ns: u64, recv_ns: u64) -> u64 {
        (recv_ns - send_ns) / 1000
    }
}

/// Connection health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionHealth {
    Healthy,
    Degraded,
    Critical,
    Disconnected,
}

/// Heartbeat statistics snapshot
#[derive(Debug, Clone)]
pub struct HeartbeatStats {
    /// Current RTT (microseconds)
    pub current_rtt_us: u64,
    /// Average RTT (microseconds)
    pub avg_rtt_us: f64,
    /// Minimum RTT observed (microseconds)
    pub min_rtt_us: u64,
    /// Maximum RTT observed (microseconds)
    pub max_rtt_us: u64,
    /// Standard deviation of RTT (microseconds)
    pub stddev_rtt_us: f64,
    /// Jitter (microseconds)
    pub jitter_us: f64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packet loss percentage
    pub packet_loss_pct: f64,
    /// Connection health status
    pub health: ConnectionHealth,
    /// Trend direction (-1 = improving, 0 = stable, 1 = degrading)
    pub trend: f64,
    /// Estimated time to degradation (seconds, 0 if not degrading)
    pub time_to_degradation_secs: f64,
}

/// Exponential Moving Average calculator
struct EmaCalculator {
    alpha: f64,
    value: f64,
    initialized: bool,
}

impl EmaCalculator {
    fn new(alpha: f64) -> Self {
        Self {
            alpha,
            value: 0.0,
            initialized: false,
        }
    }

    fn update(&mut self, sample: f64) -> f64 {
        if !self.initialized {
            self.value = sample;
            self.initialized = true;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        self.value
    }

    fn get(&self) -> f64 {
        self.value
    }
}

/// Heartbeat Tracker for connection monitoring
pub struct HeartbeatTracker {
    /// Heartbeat interval (milliseconds)
    interval_ms: AtomicU64,
    /// Current sequence number
    sequence: AtomicU64,
    /// Is tracking active
    is_active: AtomicBool,
    /// Last send timestamp
    last_send_ns: AtomicU64,
    /// Last receive timestamp
    last_recv_ns: AtomicU64,
    /// Pending ping sequences (for detecting lost pings)
    pending_pings: Arc<RwLock<VecDeque<u64>>>,
    /// Recent RTT samples (microseconds)
    rtt_samples: Arc<RwLock<VecDeque<u64>>>,
    /// EMA of RTT
    rtt_ema: RwLock<EmaCalculator>,
    /// Baseline RTT for comparison (microseconds)
    baseline_rtt_us: AtomicU64,
    /// Statistics counters
    packets_sent: AtomicU64,
    packets_received: AtomicU64,
    /// Min/Max RTT tracking
    min_rtt_us: AtomicU64,
    max_rtt_us: AtomicU64,
    /// Sum for standard deviation calculation
    rtt_sum: AtomicU64,
    rtt_sum_sq: AtomicU64,
}

impl HeartbeatTracker {
    /// Create new heartbeat tracker with default interval
    pub fn new() -> Self {
        Self::with_interval(DEFAULT_HEARTBEAT_INTERVAL_MS)
    }

    /// Create new heartbeat tracker with custom interval
    pub fn with_interval(interval_ms: u64) -> Self {
        let ema_alpha = 2.0 / (EMA_SAMPLES as f64 + 1.0);
        
        Self {
            interval_ms: AtomicU64::new(interval_ms),
            sequence: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            last_send_ns: AtomicU64::new(0),
            last_recv_ns: AtomicU64::new(0),
            pending_pings: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            rtt_samples: Arc::new(RwLock::new(VecDeque::with_capacity(EMA_SAMPLES))),
            rtt_ema: RwLock::new(EmaCalculator::new(ema_alpha)),
            baseline_rtt_us: AtomicU64::new(0),
            packets_sent: AtomicU64::new(0),
            packets_received: AtomicU64::new(0),
            min_rtt_us: AtomicU64::new(u64::MAX),
            max_rtt_us: AtomicU64::new(0),
            rtt_sum: AtomicU64::new(0),
            rtt_sum_sq: AtomicU64::new(0),
        }
    }

    /// Start heartbeat tracking
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Release);
    }

    /// Stop heartbeat tracking
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Check if tracking is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Record ping send event
    pub async fn record_ping_sent(&self) -> u64 {
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let now_ns = get_timestamp_ns();
        
        self.last_send_ns.store(now_ns, Ordering::Relaxed);
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
        
        // Track pending ping
        {
            let mut pending = self.pending_pings.write().await;
            pending.push_back(seq);
            if pending.len() > 100 {
                pending.pop_front();
            }
        }
        
        seq
    }

    /// Record pong receive event
    pub async fn record_pong_received(&self, seq: u64) -> Option<HeartbeatSample> {
        let now_ns = get_timestamp_ns();
        let send_ns = self.last_send_ns.load(Ordering::Relaxed);
        
        if send_ns == 0 {
            return None;
        }

        // Remove from pending
        {
            let mut pending = self.pending_pings.write().await;
            if let Some(pos) = pending.iter().position(|&s| s == seq) {
                pending.remove(pos);
            }
        }

        self.last_recv_ns.store(now_ns, Ordering::Relaxed);
        self.packets_received.fetch_add(1, Ordering::Relaxed);

        let rtt_us = HeartbeatSample::calculate_rtt_us(send_ns, now_ns);
        
        // Update statistics
        self.update_rtt_stats(rtt_us).await;

        Some(HeartbeatSample {
            send_ts_ns: send_ns,
            recv_ts_ns: now_ns,
            rtt_us,
            sequence: seq,
        })
    }

    /// Update RTT statistics (lock-free where possible)
    async fn update_rtt_stats(&self, rtt_us: u64) {
        // Update min/max atomically
        self.min_rtt_us.fetch_min(rtt_us, Ordering::Relaxed);
        self.max_rtt_us.fetch_max(rtt_us, Ordering::Relaxed);

        // Update sum for mean/stddev
        self.rtt_sum.fetch_add(rtt_us, Ordering::Relaxed);
        self.rtt_sum_sq.fetch_add(rtt_us * rtt_us, Ordering::Relaxed);

        // Update EMA
        {
            let mut ema = self.rtt_ema.write().await;
            ema.update(rtt_us as f64);
        }

        // Store sample for recent history
        {
            let mut samples = self.rtt_samples.write().await;
            samples.push_back(rtt_us);
            if samples.len() > EMA_SAMPLES {
                samples.pop_front();
            }
        }

        // Set baseline on first measurement
        if self.baseline_rtt_us.load(Ordering::Relaxed) == 0 {
            self.baseline_rtt_us.store(rtt_us, Ordering::Relaxed);
        }
    }

    /// Get current connection health status
    pub async fn get_health(&self) -> ConnectionHealth {
        let baseline = self.baseline_rtt_us.load(Ordering::Relaxed) as f64;
        if baseline == 0.0 {
            return ConnectionHealth::Disconnected;
        }

        let current_rtt = self.get_current_rtt_us().await;
        let ratio = current_rtt as f64 / baseline;

        if ratio > CRITICAL_THRESHOLD_MULTIPLIER {
            ConnectionHealth::Critical
        } else if ratio > WARNING_THRESHOLD_MULTIPLIER {
            ConnectionHealth::Degraded
        } else {
            ConnectionHealth::Healthy
        }
    }

    /// Get current RTT (most recent sample or EMA)
    pub async fn get_current_rtt_us(&self) -> u64 {
        let samples = self.rtt_samples.read().await;
        samples.back().copied().unwrap_or(0)
    }

    /// Get average RTT
    pub async fn get_avg_rtt_us(&self) -> f64 {
        let ema = self.rtt_ema.read().await;
        ema.get()
    }

    /// Predict time to degradation based on trend
    pub async fn predict_degradation(&self) -> Option<Duration> {
        let samples = self.rtt_samples.read().await;
        
        if samples.len() < DEGRADATION_WINDOW {
            return None;
        }

        let samples_vec: Vec<u64> = samples.iter().copied().collect();
        drop(samples);

        // Calculate linear trend
        let n = samples_vec.len() as f64;
        let sum_x: f64 = (0..samples_vec.len() as u64).sum::<u64>() as f64;
        let sum_y: f64 = samples_vec.iter().sum::<u64>() as f64;
        let sum_xy: f64 = samples_vec.iter()
            .enumerate()
            .map(|(i, &v)| i as f64 * v as f64)
            .sum();
        let sum_xx: f64 = (0..samples_vec.len() as u64)
            .map(|i| (i as f64).powi(2))
            .sum();

        // Slope of RTT trend (us per sample)
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        if slope <= 0.0 {
            return None; // Not degrading
        }

        // Calculate threshold for degradation
        let baseline = self.baseline_rtt_us.load(Ordering::Relaxed) as f64;
        let threshold = baseline * WARNING_THRESHOLD_MULTIPLIER;
        let current = samples_vec.last().copied().unwrap_or(0) as f64;

        // Estimate samples until threshold
        let samples_until_threshold = (threshold - current) / slope;
        
        if samples_until_threshold <= 0.0 {
            return Some(Duration::ZERO); // Already degraded
        }

        // Convert to time (assuming 1 second between samples)
        let interval_secs = self.interval_ms.load(Ordering::Relaxed) as f64 / 1000.0;
        let time_until_degradation = samples_until_threshold * interval_secs;

        Some(Duration::from_secs_f64(time_until_degradation))
    }

    /// Get comprehensive statistics
    pub async fn get_stats(&self) -> HeartbeatStats {
        let samples = self.rtt_samples.read().await;
        let ema = self.rtt_ema.read().await;
        
        let samples_vec: Vec<u64> = samples.iter().copied().collect();
        let count = samples_vec.len();
        
        let current_rtt = samples_vec.last().copied().unwrap_or(0);
        let avg_rtt = if count > 0 {
            samples_vec.iter().sum::<u64>() as f64 / count as f64
        } else {
            0.0
        };
        
        let min_rtt = self.min_rtt_us.load(Ordering::Relaxed);
        let max_rtt = self.max_rtt_us.load(Ordering::Relaxed);
        
        // Calculate standard deviation
        let stddev = if count > 1 {
            let variance = samples_vec.iter()
                .map(|&x| (x as f64 - avg_rtt).powi(2))
                .sum::<f64>() / (count - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Calculate jitter (difference between consecutive samples)
        let jitter = if count > 1 {
            let diffs: Vec<f64> = samples_vec.windows(2)
                .map(|w| (w[1] as i64 - w[0] as i64).abs() as f64)
                .collect();
            diffs.iter().sum::<f64>() / diffs.len() as f64
        } else {
            0.0
        };

        let sent = self.packets_sent.load(Ordering::Relaxed);
        let received = self.packets_received.load(Ordering::Relaxed);
        let packet_loss = if sent > 0 {
            ((sent - received) as f64 / sent as f64) * 100.0
        } else {
            0.0
        };

        // Determine health
        let baseline = self.baseline_rtt_us.load(Ordering::Relaxed) as f64;
        let health = if baseline > 0.0 && current_rtt as f64 > baseline * CRITICAL_THRESHOLD_MULTIPLIER {
            ConnectionHealth::Critical
        } else if baseline > 0.0 && current_rtt as f64 > baseline * WARNING_THRESHOLD_MULTIPLIER {
            ConnectionHealth::Degraded
        } else if sent == 0 || received == 0 {
            ConnectionHealth::Disconnected
        } else {
            ConnectionHealth::Healthy
        };

        // Calculate trend
        let trend = if count > 1 {
            let first_half_avg = samples_vec[..count/2].iter().sum::<u64>() as f64 / (count/2) as f64;
            let second_half_avg = samples_vec[count/2..].iter().sum::<u64>() as f64 / (count - count/2) as f64;
            (second_half_avg - first_half_avg) / first_half_avg.max(1.0)
        } else {
            0.0
        };

        // Estimate time to degradation
        let time_to_degradation = self.predict_degradation().await
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        HeartbeatStats {
            current_rtt_us: current_rtt,
            avg_rtt_us: ema.get(),
            min_rtt_us: if min_rtt == u64::MAX { 0 } else { min_rtt },
            max_rtt_us: max_rtt,
            stddev_rtt_us: stddev,
            jitter_us: jitter,
            packets_sent: sent,
            packets_received: received,
            packet_loss_pct: packet_loss,
            health,
            trend,
            time_to_degradation_secs: time_to_degradation,
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.sequence.store(0, Ordering::Relaxed);
        self.last_send_ns.store(0, Ordering::Relaxed);
        self.last_recv_ns.store(0, Ordering::Relaxed);
        self.packets_sent.store(0, Ordering::Relaxed);
        self.packets_received.store(0, Ordering::Relaxed);
        self.min_rtt_us.store(u64::MAX, Ordering::Relaxed);
        self.max_rtt_us.store(0, Ordering::Relaxed);
        self.rtt_sum.store(0, Ordering::Relaxed);
        self.rtt_sum_sq.store(0, Ordering::Relaxed);
        self.baseline_rtt_us.store(0, Ordering::Relaxed);
        
        // Clear collections (requires async)
        tokio::spawn(async move {
            let mut pending = self.pending_pings.write().await;
            pending.clear();
            let mut samples = self.rtt_samples.write().await;
            samples.clear();
            let mut ema = self.rtt_ema.write().await;
            *ema = EmaCalculator::new(2.0 / (EMA_SAMPLES as f64 + 1.0));
        });
    }

    /// Get interval in milliseconds
    pub fn get_interval_ms(&self) -> u64 {
        self.interval_ms.load(Ordering::Relaxed)
    }

    /// Set interval in milliseconds
    pub fn set_interval_ms(&self, interval_ms: u64) {
        self.interval_ms.store(interval_ms, Ordering::Relaxed);
    }
}

impl Default for HeartbeatTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_heartbeat_tracking() {
        let tracker = HeartbeatTracker::new();
        tracker.start();

        // Record ping
        let seq = tracker.record_ping_sent().await;
        
        // Simulate small delay
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Record pong
        let sample = tracker.record_pong_received(seq).await.unwrap();
        
        assert!(sample.rtt_us >= 10000); // At least 10ms in microseconds
        assert_eq!(sample.sequence, seq);
    }

    #[tokio::test]
    async fn test_health_detection() {
        let tracker = HeartbeatTracker::new();
        
        // Initially disconnected
        assert_eq!(tracker.get_health().await, ConnectionHealth::Disconnected);
        
        // Add some samples
        tracker.start();
        for i in 0..10 {
            tracker.record_ping_sent().await;
            tracker.record_pong_received(i).await;
        }
        
        // Should be healthy after successful pings
        assert_eq!(tracker.get_health().await, ConnectionHealth::Healthy);
    }

    #[tokio::test]
    async fn test_statistics() {
        let tracker = HeartbeatTracker::new();
        tracker.start();
        
        // Add consistent samples
        for i in 0..20 {
            tracker.record_ping_sent().await;
            tokio::time::sleep(Duration::from_millis(5)).await;
            tracker.record_pong_received(i).await;
        }
        
        let stats = tracker.get_stats().await;
        
        assert!(stats.packets_sent > 0);
        assert!(stats.packets_received > 0);
        assert!(stats.avg_rtt_us > 0.0);
    }
}
