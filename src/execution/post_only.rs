//! src/execution/post_only.rs
//!
//! Strict Post-Only and Reduce-Only Execution Flags.
//!
//! This module implements execution guards that guarantee the bot only captures
//! maker rebates and never accidentally crosses the spread during volatile flash
//! crashes. It enforces Post-Only (GTX) and Reduce-Only constraints at the
//! order submission layer.
//!
//! Features:
//! - Post-Only Validation: Ensures orders rest on book or reject.
//! - Reduce-Only Logic: Prevents opening new positions when closing.
//! - Spread Crossing Prevention: Validates price against best bid/ask.
//! - Flash Crash Protection: Rejects orders during extreme volatility.

use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Order execution flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionFlags {
    /// Order must rest on book (maker) or be rejected.
    pub post_only: bool,
    /// Order can only reduce existing position.
    pub reduce_only: bool,
    /// Close position on opposite fill (for hedging).
    pub close_on_trigger: bool,
}

impl Default for ExecutionFlags {
    fn default() -> Self {
        Self {
            post_only: false,
            reduce_only: false,
            close_on_trigger: false,
        }
    }
}

impl ExecutionFlags {
    /// Create Post-Only flag set.
    pub fn post_only() -> Self {
        Self {
            post_only: true,
            reduce_only: false,
            close_on_trigger: false,
        }
    }

    /// Create Reduce-Only flag set.
    pub fn reduce_only() -> Self {
        Self {
            post_only: false,
            reduce_only: true,
            close_on_trigger: false,
        }
    }

    /// Create combined Post-Only + Reduce-Only flag set.
    pub fn post_only_reduce_only() -> Self {
        Self {
            post_only: true,
            reduce_only: true,
            close_on_trigger: false,
        }
    }
}

/// Result of execution flag validation.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    Valid,
    WouldCrossSpread { side: Side, limit_price: f64, crossing_price: f64 },
    WouldOpenPosition { side: Side, quantity: f64 },
    InvalidPrice { reason: String },
    VolatilityBlock { current_vol: f64, threshold: f64 },
}

/// Market data snapshot for validation.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub symbol: String,
    pub best_bid: f64,
    pub best_ask: f64,
    pub last_price: f64,
    pub timestamp_ns: u64,
    /// Recent price volatility (standard deviation over window).
    pub recent_volatility: f64,
}

/// Position state for reduce-only validation.
#[derive(Debug, Clone)]
pub struct PositionState {
    pub symbol: String,
    pub quantity: f64,  // Positive = long, negative = short
    pub entry_price: f64,
}

impl PositionState {
    pub fn is_long(&self) -> bool {
        self.quantity > 0.0
    }

    pub fn is_short(&self) -> bool {
        self.quantity < 0.0
    }

    pub fn is_flat(&self) -> bool {
        self.quantity.abs() < 0.0001
    }

    /// Get maximum reducible quantity for a given side.
    pub fn max_reduce_quantity(&self, side: Side) -> f64 {
        match side {
            Side::Buy => {
                // Buying reduces short position
                if self.is_short() {
                    self.quantity.abs()
                } else {
                    0.0
                }
            }
            Side::Sell => {
                // Selling reduces long position
                if self.is_long() {
                    self.quantity
                } else {
                    0.0
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Post-Only / Reduce-Only enforcement engine.
pub struct ExecutionGuard {
    /// Whether guard is currently active.
    is_active: AtomicBool,
    /// Volatility threshold for blocking orders (percentage).
    volatility_threshold: AtomicU64, // Fixed-point * 10000
    /// Last validation timestamp.
    last_validation_ns: AtomicU64,
    /// Total validations performed.
    validation_count: AtomicU64,
    /// Total rejections.
    rejection_count: AtomicU64,
}

unsafe impl Send for ExecutionGuard {}
unsafe impl Sync for ExecutionGuard {}

impl ExecutionGuard {
    pub fn new(volatility_threshold_pct: f64) -> Self {
        Self {
            is_active: AtomicBool::new(true),
            volatility_threshold: AtomicU64::new((volatility_threshold_pct * 100.0) as u64),
            last_validation_ns: AtomicU64::new(0),
            validation_count: AtomicU64::new(0),
            rejection_count: AtomicU64::new(0),
        }
    }

    /// Validate a Post-Only order.
    /// 
    /// For Post-Only Buy: limit_price must be < best_ask
    /// For Post-Only Sell: limit_price must be > best_bid
    /// 
    /// If price would cross, returns error with crossing price.
    pub fn validate_post_only(
        &self,
        side: Side,
        limit_price: f64,
        market: &MarketSnapshot,
    ) -> ValidationResult {
        self.validation_count.fetch_add(1, Ordering::Relaxed);
        self.last_validation_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );

        match side {
            Side::Buy => {
                // Buy limit must be below best ask to be post-only
                if limit_price >= market.best_ask {
                    self.rejection_count.fetch_add(1, Ordering::Relaxed);
                    return ValidationResult::WouldCrossSpread {
                        side,
                        limit_price,
                        crossing_price: market.best_ask,
                    };
                }
            }
            Side::Sell => {
                // Sell limit must be above best bid to be post-only
                if limit_price <= market.best_bid {
                    self.rejection_count.fetch_add(1, Ordering::Relaxed);
                    return ValidationResult::WouldCrossSpread {
                        side,
                        limit_price,
                        crossing_price: market.best_bid,
                    };
                }
            }
        }

        // Check volatility block
        let vol_fp = (market.recent_volatility * 100.0) as u64;
        let threshold = self.volatility_threshold.load(Ordering::Relaxed);
        
        if vol_fp > threshold {
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return ValidationResult::VolatilityBlock {
                current_vol: market.recent_volatility,
                threshold: threshold as f64 / 100.0,
            };
        }

        ValidationResult::Valid
    }

    /// Validate a Reduce-Only order.
    /// 
    /// Reduce-Only orders cannot increase position size.
    /// Buy reduces short, Sell reduces long.
    pub fn validate_reduce_only(
        &self,
        side: Side,
        quantity: f64,
        position: &PositionState,
    ) -> ValidationResult {
        self.validation_count.fetch_add(1, Ordering::Relaxed);

        let max_reduce = position.max_reduce_quantity(side);

        if max_reduce <= 0.0 {
            // No position to reduce
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return ValidationResult::WouldOpenPosition { side, quantity };
        }

        if quantity > max_reduce {
            // Order size exceeds reducible amount
            self.rejection_count.fetch_add(1, Ordering::Relaxed);
            return ValidationResult::WouldOpenPosition {
                side,
                quantity: quantity - max_reduce,
            };
        }

        ValidationResult::Valid
    }

    /// Full validation combining all flags.
    pub fn validate_order(
        &self,
        flags: ExecutionFlags,
        side: Side,
        limit_price: f64,
        quantity: f64,
        market: &MarketSnapshot,
        position: &PositionState,
    ) -> ValidationResult {
        if !self.is_active.load(Ordering::Relaxed) {
            return ValidationResult::Valid;
        }

        // Validate Post-Only if flagged
        if flags.post_only {
            let result = self.validate_post_only(side, limit_price, market);
            if result != ValidationResult::Valid {
                return result;
            }
        }

        // Validate Reduce-Only if flagged
        if flags.reduce_only {
            let result = self.validate_reduce_only(side, quantity, position);
            if result != ValidationResult::Valid {
                return result;
            }
        }

        ValidationResult::Valid
    }

    /// Calculate optimal Post-Only price.
    /// 
    /// For Buy: best_bid + tick_size (or just below best_ask)
    /// For Sell: best_ask - tick_size (or just above best_bid)
    pub fn calculate_post_only_price(
        &self,
        side: Side,
        market: &MarketSnapshot,
        tick_size: f64,
    ) -> f64 {
        match side {
            Side::Buy => {
                // Place at best bid or one tick below best ask
                let target = market.best_ask - tick_size;
                target.max(market.best_bid)
            }
            Side::Sell => {
                // Place at best ask or one tick above best bid
                let target = market.best_bid + tick_size;
                target.min(market.best_ask)
            }
        }
    }

    /// Enable/disable the guard.
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Relaxed);
    }

    /// Update volatility threshold.
    pub fn set_volatility_threshold(&self, threshold_pct: f64) {
        self.volatility_threshold.store(
            (threshold_pct * 100.0) as u64,
            Ordering::Relaxed,
        );
    }

    /// Get statistics.
    pub fn get_stats(&self) -> GuardStats {
        GuardStats {
            is_active: self.is_active.load(Ordering::Relaxed),
            validation_count: self.validation_count.load(Ordering::Relaxed),
            rejection_count: self.rejection_count.load(Ordering::Relaxed),
            rejection_rate: {
                let total = self.validation_count.load(Ordering::Relaxed);
                if total > 0 {
                    self.rejection_count.load(Ordering::Relaxed) as f64 / total as f64
                } else {
                    0.0
                }
            },
            volatility_threshold: self.volatility_threshold.load(Ordering::Relaxed) as f64 / 100.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GuardStats {
    pub is_active: bool,
    pub validation_count: u64,
    pub rejection_count: u64,
    pub rejection_rate: f64,
    pub volatility_threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_only_buy_validation() {
        let guard = ExecutionGuard::new(5.0); // 5% volatility threshold
        
        let market = MarketSnapshot {
            symbol: "BTCUSDT".to_string(),
            best_bid: 49990.0,
            best_ask: 50000.0,
            last_price: 49995.0,
            timestamp_ns: 1000000,
            recent_volatility: 0.02, // 2%
        };

        // Valid post-only buy (below best ask)
        let result = guard.validate_post_only(Side::Buy, 49999.0, &market);
        assert_eq!(result, ValidationResult::Valid);

        // Invalid post-only buy (crosses best ask)
        let result = guard.validate_post_only(Side::Buy, 50000.0, &market);
        assert!(matches!(result, ValidationResult::WouldCrossSpread { .. }));
    }

    #[test]
    fn test_post_only_sell_validation() {
        let guard = ExecutionGuard::new(5.0);
        
        let market = MarketSnapshot {
            symbol: "BTCUSDT".to_string(),
            best_bid: 49990.0,
            best_ask: 50000.0,
            last_price: 49995.0,
            timestamp_ns: 1000000,
            recent_volatility: 0.02,
        };

        // Valid post-only sell (above best bid)
        let result = guard.validate_post_only(Side::Sell, 49991.0, &market);
        assert_eq!(result, ValidationResult::Valid);

        // Invalid post-only sell (crosses best bid)
        let result = guard.validate_post_only(Side::Sell, 49990.0, &market);
        assert!(matches!(result, ValidationResult::WouldCrossSpread { .. }));
    }

    #[test]
    fn test_reduce_only_validation() {
        let guard = ExecutionGuard::new(5.0);

        // Long position of 1.0 BTC
        let position = PositionState {
            symbol: "BTCUSDT".to_string(),
            quantity: 1.0,
            entry_price: 50000.0,
        };

        // Valid reduce-only sell (reduces long)
        let result = guard.validate_reduce_only(Side::Sell, 0.5, &position);
        assert_eq!(result, ValidationResult::Valid);

        // Invalid reduce-only sell (exceeds position)
        let result = guard.validate_reduce_only(Side::Sell, 1.5, &position);
        assert!(matches!(result, ValidationResult::WouldOpenPosition { .. }));

        // Invalid reduce-only buy (would increase long)
        let result = guard.validate_reduce_only(Side::Buy, 0.5, &position);
        assert!(matches!(result, ValidationResult::WouldOpenPosition { .. }));
    }

    #[test]
    fn test_volatility_block() {
        let guard = ExecutionGuard::new(3.0); // 3% threshold
        
        let market = MarketSnapshot {
            symbol: "BTCUSDT".to_string(),
            best_bid: 49000.0,
            best_ask: 51000.0,
            last_price: 50000.0,
            timestamp_ns: 1000000,
            recent_volatility: 0.05, // 5% - exceeds threshold
        };

        // Should be blocked due to high volatility
        let result = guard.validate_post_only(Side::Buy, 49500.0, &market);
        assert!(matches!(result, ValidationResult::VolatilityBlock { .. }));
    }

    #[test]
    fn test_calculate_post_only_price() {
        let guard = ExecutionGuard::new(5.0);
        let tick_size = 0.01;

        let market = MarketSnapshot {
            symbol: "BTCUSDT".to_string(),
            best_bid: 49990.0,
            best_ask: 50000.0,
            last_price: 49995.0,
            timestamp_ns: 1000000,
            recent_volatility: 0.02,
        };

        // Buy: should be just below best ask
        let buy_price = guard.calculate_post_only_price(Side::Buy, &market, tick_size);
        assert!(buy_price < market.best_ask);
        assert!(buy_price >= market.best_bid);

        // Sell: should be just above best bid
        let sell_price = guard.calculate_post_only_price(Side::Sell, &market, tick_size);
        assert!(sell_price > market.best_bid);
        assert!(sell_price <= market.best_ask);
    }
}
