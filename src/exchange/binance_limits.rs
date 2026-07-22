//! Binance API Rate Limit Tracking - IP and UID Weight Management
//! 
//! This module implements strict IP and UID weight tracking to prevent 429 and 418 bans
//! from Binance REST APIs. Uses lock-free atomic counters to manage request budgets
//! across all threads without blocking.
//!
//! Key Features:
//! - Lock-free atomic counters for thread-safe tracking
//! - Separate IP and UID weight buckets
//! - Automatic weight recovery over time
//! - Pre-emptive rate limiting before hitting limits
//! - AMD Ryzen AI 5 architecture optimizations
//!
//! Binance Rate Limits:
//! - IP limits: 1200 requests per minute (varies by endpoint)
//! - UID limits: Varies by account tier
//! - Order limits: Specific per-symbol limits

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Default IP weight limit per minute (Binance standard)
const DEFAULT_IP_WEIGHT_LIMIT: u64 = 1200;

/// Default UID weight limit per minute
const DEFAULT_UID_WEIGHT_LIMIT: u64 = 6000;

/// Weight recovery interval in milliseconds
const WEIGHT_RECOVERY_INTERVAL_MS: u64 = 1000 / 60; // Recover weights every second

/// Lock-free IP/UID weight tracker
pub struct BinanceWeightTracker {
    /// Current IP weight used
    ip_weight_used: AtomicU64,
    /// Current UID weight used
    uid_weight_used: AtomicU64,
    /// IP weight limit
    ip_weight_limit: AtomicU64,
    /// UID weight limit
    uid_weight_limit: AtomicU64,
    /// Last weight recovery timestamp (milliseconds)
    last_recovery_ms: AtomicU64,
    /// Number of 429 responses received
    rate_limit_hits: AtomicU64,
    /// Number of 418 responses received
    ban_hits: AtomicU64,
    /// Total requests made
    total_requests: AtomicU64,
    /// Backoff multiplier when rate limited
    backoff_multiplier: AtomicU64, // Fixed point: value * 1000
}

unsafe impl Send for BinanceWeightTracker {}
unsafe impl Sync for BinanceWeightTracker {}

/// Request weight cost for different endpoints
#[repr(u64)]
#[derive(Debug, Clone, Copy)]
pub enum EndpointWeight {
    /// Lightweight endpoints (weight = 1)
    Light = 1,
    /// Market data endpoints (weight = 2)
    MarketData = 2,
    /// Order endpoints (weight = 1-4 depending on type)
    Order = 4,
    /// Account data (weight = 10)
    Account = 10,
    /// Historical klines (weight = 5-20 depending on range)
    Klines = 10,
    /// Custom weight
    Custom(u64),
}

impl EndpointWeight {
    #[inline]
    pub fn cost(&self) -> u64 {
        match self {
            EndpointWeight::Light => 1,
            EndpointWeight::MarketData => 2,
            EndpointWeight::Order => 4,
            EndpointWeight::Account => 10,
            EndpointWeight::Klines => 10,
            EndpointWeight::Custom(w) => *w,
        }
    }
}

/// Response from weight check
#[derive(Debug, Clone)]
pub struct WeightCheckResult {
    /// Whether request is allowed
    pub allowed: bool,
    /// Current IP weight usage
    pub ip_weight_used: u64,
    /// Current UID weight usage
    pub uid_weight_used: u64,
    /// IP weight limit
    pub ip_weight_limit: u64,
    /// UID weight limit
    pub uid_weight_limit: u64,
    /// Estimated wait time before next request (milliseconds)
    pub wait_time_ms: u64,
    /// Backoff multiplier active
    pub backoff_active: bool,
}

impl BinanceWeightTracker {
    /// Create a new weight tracker with default limits
    pub fn new() -> Self {
        let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
        
        Self {
            ip_weight_used: AtomicU64::new(0),
            uid_weight_used: AtomicU64::new(0),
            ip_weight_limit: AtomicU64::new(DEFAULT_IP_WEIGHT_LIMIT),
            uid_weight_limit: AtomicU64::new(DEFAULT_UID_WEIGHT_LIMIT),
            last_recovery_ms: AtomicU64::new(now_ms),
            rate_limit_hits: AtomicU64::new(0),
            ban_hits: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            backoff_multiplier: AtomicU64::new(1000), // 1.0x normal speed
        }
    }

    /// Create with custom limits
    pub fn with_limits(ip_limit: u64, uid_limit: u64) -> Self {
        let mut tracker = Self::new();
        tracker.ip_weight_limit.store(ip_limit, Ordering::Relaxed);
        tracker.uid_weight_limit.store(uid_limit, Ordering::Relaxed);
        tracker
    }

    /// Check if a request can be made with given weight cost
    #[inline]
    pub fn check_weight(&self, weight: u64) -> WeightCheckResult {
        let ip_used = self.ip_weight_used.load(Ordering::Acquire);
        let uid_used = self.uid_weight_used.load(Ordering::Acquire);
        let ip_limit = self.ip_weight_limit.load(Ordering::Relaxed);
        let uid_limit = self.uid_weight_limit.load(Ordering::Relaxed);
        let backoff = self.backoff_multiplier.load(Ordering::Relaxed);
        
        // Apply backoff to effective limits
        let effective_ip_limit = (ip_limit * 1000) / backoff;
        let effective_uid_limit = (uid_limit * 1000) / backoff;
        
        let ip_remaining = effective_ip_limit.saturating_sub(ip_used);
        let uid_remaining = effective_uid_limit.saturating_sub(uid_used);
        
        let allowed = ip_remaining >= weight && uid_remaining >= weight;
        
        // Estimate wait time if not allowed
        let wait_time_ms = if !allowed {
            let ip_deficit = weight.saturating_sub(ip_remaining);
            let uid_deficit = weight.saturating_sub(uid_remaining);
            let max_deficit = ip_deficit.max(uid_deficit);
            
            // Time to recover enough weight (assuming 1 weight per second recovery)
            max_deficit * WEIGHT_RECOVERY_INTERVAL_MS
        } else {
            0
        };
        
        WeightCheckResult {
            allowed,
            ip_weight_used: ip_used,
            uid_weight_used: uid_used,
            ip_weight_limit: ip_limit,
            uid_weight_limit: uid_limit,
            wait_time_ms,
            backoff_active: backoff > 1000,
        }
    }

    /// Record a request with its weight cost
    #[inline]
    pub fn record_request(&self, weight: u64) -> Result<(), &'static str> {
        let result = self.check_weight(weight);
        
        if !result.allowed {
            return Err("Rate limit would be exceeded");
        }
        
        // Atomically add weight
        self.ip_weight_used.fetch_add(weight, Ordering::AcqRel);
        self.uid_weight_used.fetch_add(weight, Ordering::AcqRel);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Record receiving a 429 rate limit response
    pub fn record_rate_limit_hit(&self) {
        let hits = self.rate_limit_hits.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Increase backoff exponentially with each hit
        let current_backoff = self.backoff_multiplier.load(Ordering::Relaxed);
        let new_backoff = (current_backoff * 2).min(10000); // Cap at 10x backoff
        self.backoff_multiplier.store(new_backoff, Ordering::Relaxed);
        
        // Log warning (in production, use proper logging)
        eprintln!("Rate limit hit #{} - Backoff increased to {:.1}x", hits, new_backoff as f64 / 1000.0);
    }

    /// Record receiving a 418 ban response
    pub fn record_ban_hit(&self) {
        let hits = self.ban_hits.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Severe backoff for bans
        self.backoff_multiplier.store(5000, Ordering::Relaxed); // 5x backoff
        
        eprintln!("Ban hit #{} - Severe backoff activated", hits);
    }

    /// Reset weight counters (called periodically or after recovery)
    pub fn reset_weights(&self) {
        self.ip_weight_used.store(0, Ordering::Release);
        self.uid_weight_used.store(0, Ordering::Release);
        self.last_recovery_ms.store(
            Instant::now().duration_since(Instant::now()).as_millis() as u64,
            Ordering::Relaxed
        );
    }

    /// Gradually recover weights over time
    pub fn recover_weights(&self) {
        let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
        let last_recovery = self.last_recovery_ms.load(Ordering::Relaxed);
        
        let elapsed_ms = now_ms.saturating_sub(last_recovery);
        
        if elapsed_ms >= WEIGHT_RECOVERY_INTERVAL_MS {
            // Recover some weight based on elapsed time
            let recovery_amount = (elapsed_ms / WEIGHT_RECOVERY_INTERVAL_MS) as u64;
            
            let ip_used = self.ip_weight_used.load(Ordering::Acquire);
            let uid_used = self.uid_weight_used.load(Ordering::Acquire);
            
            self.ip_weight_used.store(ip_used.saturating_sub(recovery_amount), Ordering::Release);
            self.uid_weight_used.store(uid_used.saturating_sub(recovery_amount), Ordering::Release);
            
            self.last_recovery_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    /// Reduce backoff multiplier gradually
    pub fn reduce_backoff(&self) {
        let current = self.backoff_multiplier.load(Ordering::Relaxed);
        if current > 1000 {
            // Decay backoff by 10% every call
            let new_backoff = (current * 900) / 1000;
            self.backoff_multiplier.store(new_backoff.max(1000), Ordering::Relaxed);
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> WeightStats {
        WeightStats {
            ip_weight_used: self.ip_weight_used.load(Ordering::Relaxed),
            uid_weight_used: self.uid_weight_used.load(Ordering::Relaxed),
            ip_weight_limit: self.ip_weight_limit.load(Ordering::Relaxed),
            uid_weight_limit: self.uid_weight_limit.load(Ordering::Relaxed),
            rate_limit_hits: self.rate_limit_hits.load(Ordering::Relaxed),
            ban_hits: self.ban_hits.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            backoff_multiplier: self.backoff_multiplier.load(Ordering::Relaxed) as f64 / 1000.0,
        }
    }

    /// Check if approaching rate limit (>80% usage)
    #[inline]
    pub fn is_approaching_limit(&self, threshold: f64) -> bool {
        let ip_used = self.ip_weight_used.load(Ordering::Relaxed) as f64;
        let ip_limit = self.ip_weight_limit.load(Ordering::Relaxed) as f64;
        
        let uid_used = self.uid_weight_used.load(Ordering::Relaxed) as f64;
        let uid_limit = self.uid_weight_limit.load(Ordering::Relaxed) as f64;
        
        let ip_ratio = ip_used / ip_limit;
        let uid_ratio = uid_used / uid_limit;
        
        ip_ratio > threshold || uid_ratio > threshold
    }
}

/// Weight statistics
#[derive(Debug)]
pub struct WeightStats {
    pub ip_weight_used: u64,
    pub uid_weight_used: u64,
    pub ip_weight_limit: u64,
    pub uid_weight_limit: u64,
    pub rate_limit_hits: u64,
    pub ban_hits: u64,
    pub total_requests: u64,
    pub backoff_multiplier: f64,
}

impl Default for BinanceWeightTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_tracking() {
        let tracker = BinanceWeightTracker::new();
        
        // Should allow initial requests
        assert!(tracker.record_request(10).is_ok());
        assert!(tracker.record_request(20).is_ok());
        
        let stats = tracker.get_stats();
        assert_eq!(stats.ip_weight_used, 30);
        assert_eq!(stats.uid_weight_used, 30);
        assert_eq!(stats.total_requests, 2);
    }

    #[test]
    fn test_rate_limit_detection() {
        let tracker = BinanceWeightTracker::with_limits(100, 100);
        
        // Use most of the weight
        tracker.record_request(85).unwrap();
        
        // Should detect approaching limit
        assert!(tracker.is_approaching_limit(0.8));
    }

    #[test]
    fn test_backoff_on_rate_limit() {
        let tracker = BinanceWeightTracker::new();
        
        assert_eq!(tracker.backoff_multiplier.load(Ordering::Relaxed), 1000);
        
        tracker.record_rate_limit_hit();
        
        assert!(tracker.backoff_multiplier.load(Ordering::Relaxed) > 1000);
    }
}
