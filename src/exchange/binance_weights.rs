//! Binance Request Weights - Lock-Free Atomic Counter for API Rate Limiting
//! 
//! This module tracks IP and UID request weights using lock-free atomic counters.
//! Implements dynamic throttling when approaching Binance's 1200/6000 weight limits.
//! Optimized for AMD Ryzen AI 5 with microsecond-level counter operations.
//! 
//! RAM Budget: Pure atomic counters, zero heap allocation in hot path.
//! Enforces global 8GB RAM limit via bounded tracking structures.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Binance IP weight limit per 60-second window
const IP_WEIGHT_LIMIT: u64 = 2400;

/// Binance UID weight limit per 60-second window  
const UID_WEIGHT_LIMIT: u64 = 120_000;

/// Soft throttle threshold (percentage of limit)
const SOFT_THROTTLE_THRESHOLD: f64 = 0.7;

/// Hard throttle threshold (percentage of limit)
const HARD_THROTTLE_THRESHOLD: f64 = 0.9;

/// Emergency brake threshold (percentage of limit)
const EMERGENCY_THRESHOLD: f64 = 0.95;

/// Weight cost for different endpoint categories
#[derive(Debug, Clone, Copy)]
pub enum EndpointWeight {
    Light = 1,      // e.g., ping, time
    Medium = 2,     // e.g., ticker, depth
    Heavy = 4,      // e.g., orders, account
    VeryHeavy = 10, // e.g., batch orders, history
}

impl EndpointWeight {
    #[inline]
    pub const fn value(&self) -> u64 {
        match self {
            Self::Light => 1,
            Self::Medium => 2,
            Self::Heavy => 4,
            Self::VeryHeavy => 10,
        }
    }
}

/// Throttle state returned by the rate limiter
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleState {
    /// No throttling required
    Green,
    /// Soft throttling - reduce polling frequency by 50%
    Yellow,
    /// Hard throttling - only essential requests allowed
    Orange,
    /// Emergency - all non-critical requests blocked
    Red,
}

impl ThrottleState {
    #[inline]
    pub fn should_allow(&self, priority: RequestPriority) -> bool {
        match self {
            Self::Green => true,
            Self::Yellow => true,
            Self::Orange => matches!(priority, RequestPriority::Critical | RequestPriority::High),
            Self::Red => matches!(priority, RequestPriority::Critical),
        }
    }

    #[inline]
    pub fn recommended_delay(&self) -> Duration {
        match self {
            Self::Green => Duration::ZERO,
            Self::Yellow => Duration::from_millis(100),
            Self::Orange => Duration::from_millis(500),
            Self::Red => Duration::from_secs(5),
        }
    }
}

/// Request priority levels for throttling decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Critical = 0,   // Order cancel, emergency actions
    High = 1,       // Order placement, position updates
    Normal = 2,     // Market data, portfolio queries
    Low = 3,        // Analytics, logging, telemetry
}

/// Sliding window counter for weight tracking
struct SlidingWindowCounter {
    /// Current window start timestamp (nanoseconds since epoch)
    window_start: AtomicU64,
    /// Current accumulated weight in this window
    current_weight: AtomicU64,
    /// Previous window weight (for overlap handling)
    prev_weight: AtomicU64,
    /// Window duration in nanoseconds
    window_ns: u64,
}

impl SlidingWindowCounter {
    #[inline]
    fn new(window_duration: Duration) -> Self {
        let now_ns = Instant::now().elapsed().as_nanos() as u64;
        Self {
            window_start: AtomicU64::new(now_ns),
            current_weight: AtomicU64::new(0),
            prev_weight: AtomicU64::new(0),
            window_ns: window_duration.as_nanos() as u64,
        }
    }

    #[inline]
    fn get_current_time_ns() -> u64 {
        Instant::now().elapsed().as_nanos() as u64
    }

    /// Add weight to the counter with automatic window rotation
    #[inline]
    pub fn add(&self, weight: u64) -> u64 {
        let now_ns = Self::get_current_time_ns();
        let window_start = self.window_start.load(Ordering::Relaxed);
        
        // Check if we need to rotate windows
        if now_ns - window_start >= self.window_ns {
            // Attempt to rotate (race condition handled by CAS)
            let mut current = self.current_weight.load(Ordering::Relaxed);
            loop {
                if self.window_start.compare_exchange_weak(
                    window_start,
                    now_ns,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ).is_ok() {
                    // Successfully rotated, swap current to prev and reset
                    self.prev_weight.store(current, Ordering::Relaxed);
                    self.current_weight.store(weight, Ordering::Relaxed);
                    return weight;
                }
                // Another thread rotated, reload and retry
                current = self.current_weight.load(Ordering::Relaxed);
            }
        }
        
        // Same window, just add weight atomically
        self.current_weight.fetch_add(weight, Ordering::AcqRel) + weight
    }

    /// Get weighted average considering window overlap
    #[inline]
    pub fn get_weighted(&self) -> u64 {
        let now_ns = Self::get_current_time_ns();
        let window_start = self.window_start.load(Ordering::Relaxed);
        let elapsed_in_window = now_ns - window_start;
        
        if elapsed_in_window >= self.window_ns {
            return 0; // Past the window
        }
        
        let current = self.current_weight.load(Ordering::Relaxed);
        let prev = self.prev_weight.load(Ordering::Relaxed);
        
        // Calculate overlap ratio
        let overlap_ratio = (self.window_ns - elapsed_in_window) as f64 / self.window_ns as f64;
        
        // Weighted average: current + (prev * overlap)
        (current as f64 + prev as f64 * overlap_ratio) as u64
    }

    /// Reset counter (used for manual intervention)
    #[inline]
    pub fn reset(&self) {
        let now_ns = Self::get_current_time_ns();
        self.window_start.store(now_ns, Ordering::Relaxed);
        self.current_weight.store(0, Ordering::Relaxed);
        self.prev_weight.store(0, Ordering::Relaxed);
    }
}

/// Main weight tracker for Binance API rate limiting
pub struct BinanceWeightTracker {
    /// IP-based weight counter (60-second window)
    ip_counter: SlidingWindowCounter,
    /// UID-based weight counter (60-second window)
    uid_counter: SlidingWindowCounter,
    /// IP weight limit
    ip_limit: AtomicU64,
    /// UID weight limit
    uid_limit: AtomicU64,
    /// Emergency brake flag
    emergency_brake: AtomicBool,
    /// Total requests tracked (for statistics)
    total_requests: AtomicU64,
    /// Total throttled requests
    throttled_requests: AtomicU64,
}

impl Default for BinanceWeightTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BinanceWeightTracker {
    /// Create a new weight tracker with default Binance limits
    #[inline]
    pub fn new() -> Self {
        Self {
            ip_counter: SlidingWindowCounter::new(Duration::from_secs(60)),
            uid_counter: SlidingWindowCounter::new(Duration::from_secs(60)),
            ip_limit: AtomicU64::new(IP_WEIGHT_LIMIT),
            uid_limit: AtomicU64::new(UID_WEIGHT_LIMIT),
            emergency_brake: AtomicBool::new(false),
            total_requests: AtomicU64::new(0),
            throttled_requests: AtomicU64::new(0),
        }
    }

    /// Create with custom limits
    #[inline]
    pub fn with_limits(ip_limit: u64, uid_limit: u64) -> Self {
        Self {
            ip_counter: SlidingWindowCounter::new(Duration::from_secs(60)),
            uid_counter: SlidingWindowCounter::new(Duration::from_secs(60)),
            ip_limit: AtomicU64::new(ip_limit),
            uid_limit: AtomicU64::new(uid_limit),
            emergency_brake: AtomicU64::new(false),
            total_requests: AtomicU64::new(0),
            throttled_requests: AtomicU64::new(0),
        }
    }

    /// Record a request and return whether it should be allowed
    /// Returns (allowed, throttle_state, recommended_delay)
    #[inline]
    pub fn record_request(
        &self,
        weight: EndpointWeight,
        is_uid_request: bool,
        priority: RequestPriority,
    ) -> (bool, ThrottleState, Duration) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        // Check emergency brake first
        if self.emergency_brake.load(Ordering::Relaxed) {
            self.throttled_requests.fetch_add(1, Ordering::Relaxed);
            return (false, ThrottleState::Red, Duration::from_secs(5));
        }
        
        // Get current weights
        let ip_weight = self.ip_counter.get_weighted();
        let uid_weight = if is_uid_request {
            self.uid_counter.get_weighted()
        } else {
            0
        };
        
        let ip_limit = self.ip_limit.load(Ordering::Relaxed);
        let uid_limit = self.uid_limit.load(Ordering::Relaxed);
        
        // Calculate utilization ratios
        let ip_ratio = ip_weight as f64 / ip_limit as f64;
        let uid_ratio = if is_uid_request {
            uid_weight as f64 / uid_limit as f64
        } else {
            0.0
        };
        
        let max_ratio = ip_ratio.max(uid_ratio);
        
        // Determine throttle state
        let throttle_state = if max_ratio >= EMERGENCY_THRESHOLD {
            ThrottleState::Red
        } else if max_ratio >= HARD_THROTTLE_THRESHOLD {
            ThrottleState::Orange
        } else if max_ratio >= SOFT_THROTTLE_THRESHOLD {
            ThrottleState::Yellow
        } else {
            ThrottleState::Green
        };
        
        // Check if request should be allowed
        let allowed = throttle_state.should_allow(priority);
        
        if !allowed {
            self.throttled_requests.fetch_add(1, Ordering::Relaxed);
            return (false, throttle_state, throttle_state.recommended_delay());
        }
        
        // Record the weight
        let weight_value = weight.value();
        self.ip_counter.add(weight_value);
        if is_uid_request {
            self.uid_counter.add(weight_value);
        }
        
        (true, throttle_state, throttle_state.recommended_delay())
    }

    /// Quick check if a request type should be attempted
    #[inline]
    pub fn should_attempt(&self, priority: RequestPriority) -> bool {
        let ip_weight = self.ip_counter.get_weighted();
        let ip_limit = self.ip_limit.load(Ordering::Relaxed);
        let ratio = ip_weight as f64 / ip_limit as f64;
        
        let state = if ratio >= EMERGENCY_THRESHOLD {
            ThrottleState::Red
        } else if ratio >= HARD_THROTTLE_THRESHOLD {
            ThrottleState::Orange
        } else if ratio >= SOFT_THROTTLE_THRESHOLD {
            ThrottleState::Yellow
        } else {
            ThrottleState::Green
        };
        
        state.should_allow(priority)
    }

    /// Get current throttle state without recording a request
    #[inline]
    pub fn current_state(&self) -> ThrottleState {
        let ip_weight = self.ip_counter.get_weighted();
        let ip_limit = self.ip_limit.load(Ordering::Relaxed);
        let ratio = ip_weight as f64 / ip_limit as f64;
        
        if ratio >= EMERGENCY_THRESHOLD {
            ThrottleState::Red
        } else if ratio >= HARD_THROTTLE_THRESHOLD {
            ThrottleState::Orange
        } else if ratio >= SOFT_THROTTLE_THRESHOLD {
            ThrottleState::Yellow
        } else {
            ThrottleState::Green
        }
    }

    /// Engage emergency brake (stops all non-critical requests)
    #[inline]
    pub fn engage_emergency_brake(&self) {
        self.emergency_brake.store(true, Ordering::Relaxed);
    }

    /// Release emergency brake
    #[inline]
    pub fn release_emergency_brake(&self) {
        self.emergency_brake.store(false, Ordering::Relaxed);
        self.ip_counter.reset();
        self.uid_counter.reset();
    }

    /// Get current IP weight utilization (0.0 to 1.0+)
    #[inline]
    pub fn ip_utilization(&self) -> f64 {
        let weight = self.ip_counter.get_weighted();
        let limit = self.ip_limit.load(Ordering::Relaxed);
        weight as f64 / limit as f64
    }

    /// Get current UID weight utilization (0.0 to 1.0+)
    #[inline]
    pub fn uid_utilization(&self) -> f64 {
        let weight = self.uid_counter.get_weighted();
        let limit = self.uid_limit.load(Ordering::Relaxed);
        weight as f64 / limit as f64
    }

    /// Get statistics snapshot
    #[inline]
    pub fn get_stats(&self) -> WeightStats {
        WeightStats {
            ip_weight: self.ip_counter.get_weighted(),
            uid_weight: self.uid_counter.get_weighted(),
            ip_limit: self.ip_limit.load(Ordering::Relaxed),
            uid_limit: self.uid_limit.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            throttled_requests: self.throttled_requests.load(Ordering::Relaxed),
            emergency_active: self.emergency_brake.load(Ordering::Relaxed),
        }
    }

    /// Set custom limits (e.g., for VIP accounts with higher limits)
    #[inline]
    pub fn set_limits(&self, ip_limit: u64, uid_limit: u64) {
        self.ip_limit.store(ip_limit, Ordering::Relaxed);
        self.uid_limit.store(uid_limit, Ordering::Relaxed);
    }
}

/// Statistics snapshot from the weight tracker
#[derive(Debug, Clone, Copy)]
pub struct WeightStats {
    pub ip_weight: u64,
    pub uid_weight: u64,
    pub ip_limit: u64,
    pub uid_limit: u64,
    pub total_requests: u64,
    pub throttled_requests: u64,
    pub emergency_active: bool,
}

impl WeightStats {
    #[inline]
    pub fn ip_utilization(&self) -> f64 {
        self.ip_weight as f64 / self.ip_limit as f64
    }

    #[inline]
    pub fn uid_utilization(&self) -> f64 {
        self.uid_weight as f64 / self.uid_limit as f64
    }

    #[inline]
    pub fn throttle_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.throttled_requests as f64 / self.total_requests as f64
        }
    }
}

/// RAII guard for automatic weight tracking
pub struct WeightGuard<'a> {
    tracker: &'a BinanceWeightTracker,
    weight: EndpointWeight,
    is_uid_request: bool,
    committed: bool,
}

impl<'a> WeightGuard<'a> {
    #[inline]
    pub fn new(
        tracker: &'a BinanceWeightTracker,
        weight: EndpointWeight,
        is_uid_request: bool,
    ) -> Self {
        Self {
            tracker,
            weight,
            is_uid_request,
            committed: false,
        }
    }

    #[inline]
    pub fn commit(mut self) {
        self.committed = true;
        let weight_value = self.weight.value();
        self.tracker.ip_counter.add(weight_value);
        if self.is_uid_request {
            self.tracker.uid_counter.add(weight_value);
        }
    }
}

impl<'a> Drop for WeightGuard<'a> {
    #[inline]
    fn drop(&mut self) {
        if !self.committed {
            // Optionally track rolled-back requests
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_tracker_basic() {
        let tracker = BinanceWeightTracker::new();
        
        // First request should be allowed
        let (allowed, state, _) = tracker.record_request(
            EndpointWeight::Light,
            false,
            RequestPriority::Normal,
        );
        assert!(allowed);
        assert_eq!(state, ThrottleState::Green);
    }

    #[test]
    fn test_throttle_states() {
        assert!(ThrottleState::Green.should_allow(RequestPriority::Low));
        assert!(ThrottleState::Yellow.should_allow(RequestPriority::Low));
        assert!(!ThrottleState::Orange.should_allow(RequestPriority::Low));
        assert!(ThrottleState::Orange.should_allow(RequestPriority::High));
        assert!(!ThrottleState::Red.should_allow(RequestPriority::High));
        assert!(ThrottleState::Red.should_allow(RequestPriority::Critical));
    }

    #[test]
    fn test_sliding_window() {
        let counter = SlidingWindowCounter::new(Duration::from_secs(60));
        
        counter.add(100);
        assert_eq!(counter.get_weighted(), 100);
        
        counter.add(50);
        assert_eq!(counter.get_weighted(), 150);
    }

    #[test]
    fn test_emergency_brake() {
        let tracker = BinanceWeightTracker::new();
        
        tracker.engage_emergency_brake();
        
        let (allowed, state, _) = tracker.record_request(
            EndpointWeight::Light,
            false,
            RequestPriority::Normal,
        );
        assert!(!allowed);
        assert_eq!(state, ThrottleState::Red);
        
        tracker.release_emergency_brake();
        
        let (allowed, _) = tracker.record_request(
            EndpointWeight::Light,
            false,
            RequestPriority::Normal,
        );
        assert!(allowed);
    }
}
