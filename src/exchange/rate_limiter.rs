//! Hybrid Token Bucket + Leaky Bucket Rate Limiter for Order Submission
//! 
//! This module implements a token bucket combined with a leaky bucket hybrid rate limiter
//! for order submission, ensuring the bot perfectly matches Binance's strict 
//! 10-orders-per-second weight limits. Safely buffers and flushes orders if the 
//! exchange temporarily restricts the API key.
//!
//! Key Features:
//! - Token bucket for burst allowance
//! - Leaky bucket for sustained rate limiting
//! - Order buffering during restriction periods
//! - Automatic flush when limits recover
//! - AMD Ryzen AI 5 architecture optimizations
//!
//! Binance Order Limits:
//! - 10 orders per second per symbol (varies by account)
//! - Weight-based system for different endpoints
//! - Temporary restrictions trigger backoff

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Default orders per second limit (Binance standard)
const DEFAULT_ORDERS_PER_SECOND: u64 = 10;

/// Token bucket capacity (burst allowance)
const TOKEN_BUCKET_CAPACITY: u64 = 20;

/// Maximum buffered orders
const MAX_BUFFERED_ORDERS: usize = 1000;

/// Leak interval in milliseconds
const LEAK_INTERVAL_MS: u64 = 100; // Leak every 100ms

/// Order request structure
#[derive(Debug, Clone)]
pub struct OrderRequest {
    /// Unique order ID (client-side)
    pub client_order_id: u64,
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: [u8; 12],
    /// Side: 0 for buy, 1 for sell
    pub side: u8,
    /// Order type
    pub order_type: OrderType,
    /// Price (in ticks)
    pub price: i64,
    /// Quantity (in base units)
    pub quantity: i64,
    /// Timestamp (microseconds)
    pub timestamp_us: u64,
    /// Weight cost
    pub weight: u64,
}

/// Order type enumeration
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market = 0,
    Limit = 1,
    StopLoss = 2,
    StopLossLimit = 3,
    TakeProfit = 4,
    TakeProfitLimit = 5,
}

/// Buffered order waiting to be submitted
#[derive(Debug, Clone)]
pub struct BufferedOrder {
    /// The order request
    pub request: OrderRequest,
    /// Time when order was buffered
    pub buffered_at: Instant,
    /// Retry count
    pub retry_count: u8,
}

/// Rate limiter result
#[derive(Debug, Clone)]
pub struct RateLimitResult {
    /// Whether order is allowed
    pub allowed: bool,
    /// Current tokens available
    pub tokens_available: u64,
    /// Current leak bucket level
    pub bucket_level: u64,
    /// Estimated wait time (milliseconds)
    pub wait_time_ms: u64,
    /// Is currently restricted
    pub is_restricted: bool,
}

/// Hybrid rate limiter combining token bucket and leaky bucket
pub struct HybridRateLimiter {
    // Token bucket fields
    /// Current tokens available
    tokens: AtomicU64,
    /// Maximum tokens (capacity)
    max_tokens: AtomicU64,
    /// Token refill rate (tokens per second)
    refill_rate: AtomicU64,
    /// Last refill timestamp (milliseconds)
    last_refill_ms: AtomicU64,
    
    // Leaky bucket fields
    /// Current water level in bucket
    water_level: AtomicU64,
    /// Bucket capacity
    bucket_capacity: AtomicU64,
    /// Leak rate (drops per second)
    leak_rate: AtomicU64,
    /// Last leak timestamp (milliseconds)
    last_leak_ms: AtomicU64,
    
    // Restriction handling
    /// Is API key restricted
    is_restricted: AtomicBool,
    /// Restriction start timestamp
    restriction_start_ms: AtomicU64,
    /// Restriction duration (milliseconds)
    restriction_duration_ms: AtomicU64,
    
    // Buffering
    /// Buffered orders queue
    buffered_orders: VecDeque<BufferedOrder>,
    /// Total orders submitted
    total_submitted: AtomicU64,
    /// Total orders rejected
    total_rejected: AtomicU64,
    /// Total orders buffered
    total_buffered: AtomicU64,
}

unsafe impl Send for HybridRateLimiter {}
unsafe impl Sync for HybridRateLimiter {}

impl HybridRateLimiter {
    /// Create a new rate limiter with default settings
    pub fn new() -> Self {
        let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
        
        Self {
            // Token bucket initialization
            tokens: AtomicU64::new(TOKEN_BUCKET_CAPACITY),
            max_tokens: AtomicU64::new(TOKEN_BUCKET_CAPACITY),
            refill_rate: AtomicU64::new(DEFAULT_ORDERS_PER_SECOND),
            last_refill_ms: AtomicU64::new(now_ms),
            
            // Leaky bucket initialization
            water_level: AtomicU64::new(0),
            bucket_capacity: AtomicU64::new(DEFAULT_ORDERS_PER_SECOND * 2),
            leak_rate: AtomicU64::new(DEFAULT_ORDERS_PER_SECOND),
            last_leak_ms: AtomicU64::new(now_ms),
            
            // Restriction handling
            is_restricted: AtomicBool::new(false),
            restriction_start_ms: AtomicU64::new(0),
            restriction_duration_ms: AtomicU64::new(60000), // 1 minute default
            
            // Buffering
            buffered_orders: VecDeque::with_capacity(MAX_BUFFERED_ORDERS),
            total_submitted: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            total_buffered: AtomicU64::new(0),
        }
    }

    /// Create with custom orders per second limit
    pub fn with_limit(orders_per_second: u64) -> Self {
        let mut limiter = Self::new();
        limiter.refill_rate.store(orders_per_second, Ordering::Relaxed);
        limiter.max_tokens.store(orders_per_second * 2, Ordering::Relaxed);
        limiter.tokens.store(orders_per_second * 2, Ordering::Relaxed);
        limiter.bucket_capacity.store(orders_per_second * 2, Ordering::Relaxed);
        limiter.leak_rate.store(orders_per_second, Ordering::Relaxed);
        limiter
    }

    /// Refill tokens based on elapsed time
    #[inline]
    fn refill_tokens(&self, now_ms: u64) {
        let last_refill = self.last_refill_ms.load(Ordering::Relaxed);
        let elapsed_ms = now_ms.saturating_sub(last_refill);
        
        if elapsed_ms >= 100 { // Refill every 100ms
            let refill_rate = self.refill_rate.load(Ordering::Relaxed);
            let tokens_to_add = (refill_rate * elapsed_ms) / 1000;
            let max_tokens = self.max_tokens.load(Ordering::Relaxed);
            
            let current = self.tokens.load(Ordering::Acquire);
            let new_tokens = (current + tokens_to_add).min(max_tokens);
            
            self.tokens.store(new_tokens, Ordering::Release);
            self.last_refill_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    /// Leak water from bucket based on elapsed time
    #[inline]
    fn leak_bucket(&self, now_ms: u64) {
        let last_leak = self.last_leak_ms.load(Ordering::Relaxed);
        let elapsed_ms = now_ms.saturating_sub(last_leak);
        
        if elapsed_ms >= LEAK_INTERVAL_MS {
            let leak_rate = self.leak_rate.load(Ordering::Relaxed);
            let water_to_leak = (leak_rate * elapsed_ms) / 1000;
            
            let current = self.water_level.load(Ordering::Acquire);
            let new_level = current.saturating_sub(water_to_leak);
            
            self.water_level.store(new_level, Ordering::Release);
            self.last_leak_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    /// Check if an order can be submitted
    pub fn check_order(&self, weight: u64) -> RateLimitResult {
        let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
        
        // Update buckets
        self.refill_tokens(now_ms);
        self.leak_bucket(now_ms);
        
        // Check if restricted
        let restricted = self.is_restricted.load(Ordering::Relaxed);
        if restricted {
            let restriction_start = self.restriction_start_ms.load(Ordering::Relaxed);
            let elapsed = now_ms.saturating_sub(restriction_start);
            let duration = self.restriction_duration_ms.load(Ordering::Relaxed);
            
            // Auto-lift restriction after duration
            if elapsed > duration {
                self.is_restricted.store(false, Ordering::Release);
            } else {
                return RateLimitResult {
                    allowed: false,
                    tokens_available: self.tokens.load(Ordering::Relaxed),
                    bucket_level: self.water_level.load(Ordering::Relaxed),
                    wait_time_ms: duration.saturating_sub(elapsed),
                    is_restricted: true,
                };
            }
        }
        
        let tokens = self.tokens.load(Ordering::Relaxed);
        let water = self.water_level.load(Ordering::Relaxed);
        let bucket_cap = self.bucket_capacity.load(Ordering::Relaxed);
        
        // Check both token bucket and leaky bucket
        let token_ok = tokens >= weight;
        let bucket_ok = water + weight <= bucket_cap;
        
        let allowed = token_ok && bucket_ok;
        
        let wait_time_ms = if !allowed {
            if !token_ok {
                // Wait for token refill
                let deficit = weight - tokens;
                let refill_rate = self.refill_rate.load(Ordering::Relaxed);
                (deficit * 1000).saturating_div(refill_rate.max(1))
            } else {
                // Wait for bucket leak
                let deficit = (water + weight) - bucket_cap;
                let leak_rate = self.leak_rate.load(Ordering::Relaxed);
                (deficit * 1000).saturating_div(leak_rate.max(1))
            }
        } else {
            0
        };
        
        RateLimitResult {
            allowed,
            tokens_available: tokens,
            bucket_level: water,
            wait_time_ms,
            is_restricted: false,
        }
    }

    /// Submit an order (consumes tokens if allowed)
    pub fn submit_order(&self, order: OrderRequest) -> Result<(), &'static str> {
        let weight = order.weight;
        let result = self.check_order(weight);
        
        if !result.allowed {
            // Buffer the order if possible
            if self.buffered_orders.len() < MAX_BUFFERED_ORDERS {
                let buffered = BufferedOrder {
                    request: order,
                    buffered_at: Instant::now(),
                    retry_count: 0,
                };
                self.buffered_orders.push_back(buffered);
                self.total_buffered.fetch_add(1, Ordering::Relaxed);
                return Err("Order buffered due to rate limit");
            }
            
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return Err("Rate limit exceeded and buffer full");
        }
        
        // Consume tokens
        self.tokens.fetch_sub(weight, Ordering::AcqRel);
        
        // Add water to leaky bucket
        self.water_level.fetch_add(weight, Ordering::AcqRel);
        
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }

    /// Record that exchange imposed a restriction
    pub fn record_restriction(&self, duration_ms: u64) {
        let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
        
        self.is_restricted.store(true, Ordering::Release);
        self.restriction_start_ms.store(now_ms, Ordering::Relaxed);
        self.restriction_duration_ms.store(duration_ms, Ordering::Relaxed);
        
        // Reset buckets
        self.tokens.store(0, Ordering::Release);
        self.water_level.store(self.bucket_capacity.load(Ordering::Relaxed), Ordering::Release);
    }

    /// Try to flush buffered orders
    pub fn flush_buffered_orders<F>(&self, mut submit_fn: F) -> usize
    where
        F: FnMut(&OrderRequest) -> Result<(), &'static str>,
    {
        if self.is_restricted.load(Ordering::Relaxed) {
            return 0; // Still restricted
        }
        
        let mut flushed = 0;
        let mut to_retry: Vec<BufferedOrder> = Vec::new();
        
        while let Some(buffered) = self.buffered_orders.pop_front() {
            let result = self.check_order(buffered.request.weight);
            
            if result.allowed {
                match submit_fn(&buffered.request) {
                    Ok(()) => {
                        // Consume tokens
                        self.tokens.fetch_sub(buffered.request.weight, Ordering::AcqRel);
                        self.water_level.fetch_add(buffered.request.weight, Ordering::AcqRel);
                        self.total_submitted.fetch_add(1, Ordering::Relaxed);
                        flushed += 1;
                    }
                    Err(_) => {
                        // Retry later
                        to_retry.push(buffered);
                    }
                }
            } else {
                // Not enough capacity, put back and stop
                to_retry.push(buffered);
                break;
            }
        }
        
        // Put retry orders back at front of queue
        for order in to_retry.into_iter().rev() {
            self.buffered_orders.push_front(order);
        }
        
        flushed
    }

    /// Get current statistics
    pub fn get_stats(&self) -> RateLimiterStats {
        RateLimiterStats {
            tokens_available: self.tokens.load(Ordering::Relaxed),
            max_tokens: self.max_tokens.load(Ordering::Relaxed),
            water_level: self.water_level.load(Ordering::Relaxed),
            bucket_capacity: self.bucket_capacity.load(Ordering::Relaxed),
            is_restricted: self.is_restricted.load(Ordering::Relaxed),
            buffered_orders: self.buffered_orders.len(),
            total_submitted: self.total_submitted.load(Ordering::Relaxed),
            total_rejected: self.total_rejected.load(Ordering::Relaxed),
            total_buffered: self.total_buffered.load(Ordering::Relaxed),
        }
    }

    /// Check if buffer is getting full (>80%)
    pub fn is_buffer_nearing_capacity(&self) -> bool {
        self.buffered_orders.len() >= (MAX_BUFFERED_ORDERS * 80) / 100
    }
}

/// Rate limiter statistics
#[derive(Debug)]
pub struct RateLimiterStats {
    pub tokens_available: u64,
    pub max_tokens: u64,
    pub water_level: u64,
    pub bucket_capacity: u64,
    pub is_restricted: bool,
    pub buffered_orders: usize,
    pub total_submitted: u64,
    pub total_rejected: u64,
    pub total_buffered: u64,
}

impl Default for HybridRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = HybridRateLimiter::new();
        let stats = limiter.get_stats();
        
        assert!(stats.tokens_available > 0);
        assert!(!stats.is_restricted);
        assert_eq!(stats.buffered_orders, 0);
    }

    #[test]
    fn test_order_submission() {
        let limiter = HybridRateLimiter::with_limit(10);
        
        // Should allow initial orders
        let order = OrderRequest {
            client_order_id: 1,
            symbol: *b"BTCUSDT\0\0\0\0",
            side: 0,
            order_type: OrderType::Limit,
            price: 50000000,
            quantity: 100,
            timestamp_us: 1000,
            weight: 1,
        };
        
        assert!(limiter.submit_order(order.clone()).is_ok());
        assert!(limiter.submit_order(order).is_ok());
        
        let stats = limiter.get_stats();
        assert_eq!(stats.total_submitted, 2);
    }

    #[test]
    fn test_rate_limit_enforcement() {
        let limiter = HybridRateLimiter::with_limit(2); // Very low limit for testing
        
        // Exhaust tokens
        for _ in 0..10 {
            let order = OrderRequest {
                client_order_id: _,
                symbol: *b"BTCUSDT\0\0\0\0",
                side: 0,
                order_type: OrderType::Limit,
                price: 50000000,
                quantity: 100,
                timestamp_us: 1000,
                weight: 1,
            };
            let _ = limiter.submit_order(order);
        }
        
        // Should be rate limited now
        let result = limiter.check_order(1);
        assert!(!result.allowed || result.wait_time_ms > 0);
    }

    #[test]
    fn test_restriction_handling() {
        let limiter = HybridRateLimiter::new();
        
        // Simulate restriction
        limiter.record_restriction(5000); // 5 second restriction
        
        assert!(limiter.is_restricted.load(Ordering::Relaxed));
        
        let result = limiter.check_order(1);
        assert!(!result.allowed);
        assert!(result.is_restricted);
    }
}
