//! src/execution/cancel_replace.rs
//!
//! Hyper-Fast Cancel-Replace Logic for Limit Order Management.
//!
//! This module implements atomic cancel-replace operations that update limit
//! order prices while maintaining queue priority during rapid momentum shifts.
//! It utilizes Binance's specific modifyOrder endpoint to minimize round-trip
//! latency compared to separate cancel + new order calls.
//!
//! Features:
//! - Atomic Operations: Single API call for cancel + replace.
//! - Queue Priority Preservation: Maintains time priority when possible.
//! - Price Chase Logic: Aggressively updates prices during momentum.
//! - Latency Optimization: Uses Binance's fastest endpoints.

use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Configuration for cancel-replace behavior.
#[derive(Debug, Clone)]
pub struct CancelReplaceConfig {
    /// Maximum price adjustment per tick (basis points).
    pub max_price_adjustment_bps: u32,
    /// Minimum time between cancel-replace operations (ms).
    pub min_interval_ms: u64,
    /// Number of allowed consecutive replaces before forced cancel.
    pub max_consecutive_replaces: u32,
    /// Use modifyOrder endpoint if available.
    pub use_modify_endpoint: bool,
}

impl Default for CancelReplaceConfig {
    fn default() -> Self {
        Self {
            max_price_adjustment_bps: 50, // 0.5% max adjustment
            min_interval_ms: 100,         // 100ms minimum interval
            max_consecutive_replaces: 5,
            use_modify_endpoint: true,
        }
    }
}

/// Result of a cancel-replace operation.
#[derive(Debug, Clone)]
pub struct CancelReplaceResult {
    pub success: bool,
    pub old_order_id: String,
    pub new_order_id: Option<String>,
    pub old_price: f64,
    pub new_price: f64,
    pub quantity: f64,
    pub latency_us: u64,
    pub reason: Option<String>,
}

/// State tracker for cancel-replace operations.
pub struct CancelReplaceTracker {
    /// Last cancel-replace timestamp.
    last_operation_ns: AtomicU64,
    /// Consecutive replace count.
    consecutive_replaces: AtomicU32,
    /// Total operations counter.
    total_operations: AtomicU64,
    /// Successful operations counter.
    successful_operations: AtomicU64,
    /// Currently processing flag.
    is_processing: AtomicBool,
    /// Configuration.
    config: CancelReplaceConfig,
}

unsafe impl Send for CancelReplaceTracker {}
unsafe impl Sync for CancelReplaceTracker {}

impl CancelReplaceTracker {
    pub fn new(config: CancelReplaceConfig) -> Self {
        Self {
            last_operation_ns: AtomicU64::new(0),
            consecutive_replaces: AtomicU32::new(0),
            total_operations: AtomicU64::new(0),
            successful_operations: AtomicU64::new(0),
            is_processing: AtomicBool::new(false),
            config,
        }
    }

    /// Check if a cancel-replace operation is allowed.
    pub fn can_replace(&self) -> Result<(), ReplaceError> {
        // Check if already processing
        if self.is_processing.load(Ordering::Relaxed) {
            return Err(ReplaceError::AlreadyProcessing);
        }

        // Check cooldown
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let last_op = self.last_operation_ns.load(Ordering::Relaxed);
        let elapsed_ms = (now_ns - last_op) / 1_000_000;

        if elapsed_ms < self.config.min_interval_ms {
            return Err(ReplaceError::CooldownActive(elapsed_ms));
        }

        // Check consecutive replace limit
        let consec = self.consecutive_replaces.load(Ordering::Relaxed);
        if consec >= self.config.max_consecutive_replaces {
            return Err(ReplaceError::MaxConsecutiveReached(consec));
        }

        Ok(())
    }

    /// Mark operation as started.
    pub fn start_operation(&self) -> bool {
        self.is_processing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
    }

    /// Mark operation as completed.
    pub fn complete_operation(&self, success: bool) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.last_operation_ns.store(now_ns, Ordering::Relaxed);
        self.total_operations.fetch_add(1, Ordering::Relaxed);

        if success {
            self.successful_operations.fetch_add(1, Ordering::Relaxed);
            self.consecutive_replaces.fetch_add(1, Ordering::Relaxed);
        } else {
            self.consecutive_replaces.store(0, Ordering::Relaxed);
        }

        self.is_processing.store(false, Ordering::Relaxed);
    }

    /// Reset consecutive counter (called after successful fill).
    pub fn reset_on_fill(&self) {
        self.consecutive_replaces.store(0, Ordering::Relaxed);
    }

    /// Get statistics.
    pub fn get_stats(&self) -> TrackerStats {
        TrackerStats {
            total_operations: self.total_operations.load(Ordering::Relaxed),
            successful_operations: self.successful_operations.load(Ordering::Relaxed),
            consecutive_replaces: self.consecutive_replaces.load(Ordering::Relaxed),
            success_rate: {
                let total = self.total_operations.load(Ordering::Relaxed);
                if total > 0 {
                    self.successful_operations.load(Ordering::Relaxed) as f64 / total as f64
                } else {
                    0.0
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackerStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub consecutive_replaces: u32,
    pub success_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReplaceError {
    AlreadyProcessing,
    CooldownActive(u64),           // ms remaining
    MaxConsecutiveReached(u32),
    InvalidPriceAdjustment(f64),
    NetworkError(String),
    ExchangeRejected(String),
}

/// Cancel-replace engine for order management.
pub struct CancelReplaceEngine {
    tracker: CancelReplaceTracker,
    config: CancelReplaceConfig,
}

impl CancelReplaceEngine {
    pub fn new(config: CancelReplaceConfig) -> Self {
        Self {
            tracker: CancelReplaceTracker::new(config.clone()),
            config,
        }
    }

    /// Calculate optimal new price based on market conditions.
    pub fn calculate_chase_price(
        &self,
        current_price: f64,
        target_price: f64,
        tick_size: f64,
        side: Side,
        urgency: Urgency,
    ) -> Result<f64, ReplaceError> {
        let price_diff = target_price - current_price;
        let price_diff_pct = (price_diff.abs() / current_price) * 10000.0; // basis points

        // Check if adjustment is within limits
        if price_diff_pct > self.config.max_price_adjustment_bps as f64 {
            return Err(ReplaceError::InvalidPriceAdjustment(price_diff_pct));
        }

        // Apply urgency multiplier
        let urgency_factor = match urgency {
            Urgency::Low => 0.5,
            Urgency::Normal => 1.0,
            Urgency::High => 1.5,
            Urgency::Critical => 2.0,
        };

        let adjusted_target = current_price + (price_diff * urgency_factor);

        // Round to tick size
        let rounded_price = (adjusted_target / tick_size).round() * tick_size;

        // Ensure price moves in correct direction
        let final_price = match side {
            Side::Buy => {
                // For buys, we want to increase price to chase
                rounded_price.max(current_price + tick_size)
            }
            Side::Sell => {
                // For sells, we want to decrease price to chase
                rounded_price.min(current_price - tick_size)
            }
        };

        Ok(final_price)
    }

    /// Execute a cancel-replace operation.
    /// 
    /// In production, this would call Binance's modifyOrder endpoint:
    /// POST /fapi/v1/orderModification
    /// 
    /// This is more efficient than separate cancel + new calls because:
    /// 1. Single network round-trip
    /// 2. Atomic operation (no risk of one succeeding and other failing)
    /// 3. Preserves some queue priority benefits
    pub fn execute_cancel_replace(
        &self,
        symbol: &str,
        order_id: &str,
        client_order_id: &str,
        side: Side,
        current_price: f64,
        new_price: f64,
        quantity: f64,
    ) -> Result<CancelReplaceResult, ReplaceError> {
        // Pre-check
        self.tracker.can_replace()?;

        if !self.tracker.start_operation() {
            return Err(ReplaceError::AlreadyProcessing);
        }

        let start_time = SystemTime::now();

        // Simulate the API call timing
        // In production: actual HTTP request to Binance modifyOrder endpoint
        let latency_simulation = Duration::from_micros(500); // ~500us for local processing
        std::thread::sleep(latency_simulation);

        let end_time = SystemTime::now();
        let latency_us = end_time.duration_since(start_time).unwrap().as_micros() as u64;

        // Generate new order ID (in production, this comes from exchange)
        let new_order_id = format!("MOD_{}_{}", order_id, SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis());

        let result = CancelReplaceResult {
            success: true,
            old_order_id: order_id.to_string(),
            new_order_id: Some(new_order_id),
            old_price: current_price,
            new_price,
            quantity,
            latency_us,
            reason: None,
        };

        self.tracker.complete_operation(true);

        Ok(result)
    }

    /// Force cancel without replace (emergency scenario).
    pub fn force_cancel(&self, symbol: &str, order_id: &str) -> Result<bool, ReplaceError> {
        // Emergency cancels bypass cooldown
        let start_time = SystemTime::now();

        // Simulate cancel API call
        std::thread::sleep(Duration::from_micros(300));

        self.tracker.complete_operation(true);
        self.tracker.reset_on_fill(); // Reset on any action

        Ok(true)
    }

    /// Get tracker statistics.
    pub fn get_stats(&self) -> TrackerStats {
        self.tracker.get_stats()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Urgency {
    Low,
    Normal,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_replace_cooldown() {
        let config = CancelReplaceConfig {
            min_interval_ms: 50,
            max_consecutive_replaces: 3,
            ..Default::default()
        };
        let tracker = CancelReplaceTracker::new(config);

        // First check should pass
        assert!(tracker.can_replace().is_ok());

        // Start an operation
        assert!(tracker.start_operation());
        tracker.complete_operation(true);

        // Immediate second check should fail (cooldown)
        assert!(matches!(tracker.can_replace(), Err(ReplaceError::CooldownActive(_))));
    }

    #[test]
    fn test_chase_price_calculation() {
        let engine = CancelReplaceEngine::new(CancelReplaceConfig::default());

        // Buy side: chase upward
        let new_price = engine.calculate_chase_price(
            50000.0,
            50100.0,
            0.01,
            Side::Buy,
            Urgency::Normal,
        ).unwrap();

        assert!(new_price > 50000.0);
        assert!(new_price <= 50100.0);

        // Sell side: chase downward
        let new_price = engine.calculate_chase_price(
            50000.0,
            49900.0,
            0.01,
            Side::Sell,
            Urgency::Normal,
        ).unwrap();

        assert!(new_price < 50000.0);
        assert!(new_price >= 49900.0);
    }

    #[test]
    fn test_max_consecutive_replaces() {
        let config = CancelReplaceConfig {
            max_consecutive_replaces: 2,
            min_interval_ms: 0, // No cooldown for this test
            ..Default::default()
        };
        let tracker = CancelReplaceTracker::new(config);

        // First two should succeed
        for _ in 0..2 {
            assert!(tracker.can_replace().is_ok());
            tracker.start_operation();
            tracker.complete_operation(true);
        }

        // Third should fail
        assert!(matches!(
            tracker.can_replace(),
            Err(ReplaceError::MaxConsecutiveReached(2))
        ));
    }
}
