//! # Inventory Penalty Functions for Market Making
//! 
//! This module implements non-linear, exponential inventory penalty functions
//! that aggressively skew quotes when the bot approaches its strict maximum
//! position size limits. Critical for risk management in HFT market making.
//! 
//! ## Architecture Notes:
//! - Pure Rust with zero heap allocations in hot path
//! - Exponential penalty curve for aggressive position limiting
//! - Contiguous memory layout for cache efficiency
//! - Respects 8GB RAM limit with bounded state structures
//! 
//! ## Mathematical Foundation:
//! The penalty function is:
//! penalty(position) = base_penalty * exp(k * |position| / max_position)
//! 
//! Where:
//! - k controls the curvature (higher = more aggressive)
//! - position is signed (positive = long, negative = short)
//! - penalty skews bid/ask asymmetrically

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Fixed-point precision for calculations
const FIXED_POINT_SCALE: i64 = 1_000_000;

/// Inventory penalty configuration
#[derive(Debug, Clone, Copy)]
pub struct InventoryPenaltyConfig {
    /// Maximum allowed position (absolute value), scaled by asset precision
    pub max_position: i64,
    /// Warning threshold (percentage of max, scaled)
    pub warning_threshold: i64,
    /// Base penalty in basis points (scaled)
    pub base_penalty_bps: i64,
    /// Maximum penalty in basis points (scaled)
    pub max_penalty_bps: i64,
    /// Exponential curvature factor (higher = more aggressive)
    /// Scaled by 1000 for fine control (e.g., 3000 = k=3.0)
    pub curvature_k: i64,
    /// Linear penalty coefficient for small positions (scaled)
    pub linear_coeff: i64,
}

impl Default for InventoryPenaltyConfig {
    fn default() -> Self {
        Self {
            max_position: 10_000_000,     // 10 units (scaled)
            warning_threshold: 700_000,   // 70% of max
            base_penalty_bps: 1_000,      // 1 bps base
            max_penalty_bps: 100_000,     // 100 bps max
            curvature_k: 3_000,           // k = 3.0
            linear_coeff: 500,            // Small linear component
        }
    }
}

/// Current inventory state
#[derive(Debug, Clone)]
pub struct InventoryState {
    /// Current net position (signed, scaled)
    pub current_position: i64,
    /// Peak long position in session
    pub peak_long: i64,
    /// Peak short position in session
    pub peak_short: i64,
    /// Total volume traded
    pub total_volume: u64,
    /// Number of penalty adjustments made
    pub adjustment_count: u64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl InventoryState {
    /// Create new inventory state
    pub fn new() -> Self {
        Self {
            current_position: 0,
            peak_long: 0,
            peak_short: 0,
            total_volume: 0,
            adjustment_count: 0,
            last_update: Instant::now(),
        }
    }

    /// Reset state for new trading session
    pub fn reset(&mut self) {
        self.current_position = 0;
        self.peak_long = 0;
        self.peak_short = 0;
        self.total_volume = 0;
        self.adjustment_count = 0;
        self.last_update = Instant::now();
    }
}

impl Default for InventoryState {
    fn default() -> Self {
        Self::new()
    }
}

/// Inventory penalty calculator for dynamic quote skewing
pub struct InventoryPenalty {
    /// Configuration parameters
    config: InventoryPenaltyConfig,
    /// Current inventory state
    state: InventoryState,
    /// Atomic position tracker for lock-free reads
    atomic_position: AtomicI64,
    /// Counter for penalty calculations
    calc_count: AtomicU64,
}

impl InventoryPenalty {
    /// Create a new inventory penalty calculator
    pub fn new(config: InventoryPenaltyConfig) -> Self {
        Self {
            config,
            state: InventoryState::new(),
            atomic_position: AtomicI64::new(0),
            calc_count: AtomicU64::new(0),
        }
    }

    /// Update inventory after a trade
    /// 
    /// # Arguments
    /// * `trade_size` - Signed trade size (positive = buy, negative = sell)
    /// 
    /// # Returns
    /// New position after the trade
    pub fn update_inventory(&mut self, trade_size: i64) -> i64 {
        self.state.current_position += trade_size;
        self.state.total_volume += trade_size.unsigned_abs();
        self.state.adjustment_count += 1;
        self.state.last_update = Instant::now();

        // Update peaks
        if self.state.current_position > self.state.peak_long {
            self.state.peak_long = self.state.current_position;
        }
        if self.state.current_position < self.state.peak_short {
            self.state.peak_short = self.state.current_position;
        }

        // Update atomic for lock-free reads
        self.atomic_position.store(self.state.current_position, Ordering::Release);

        self.state.current_position
    }

    /// Calculate exponential penalty for current position
    /// 
    /// Uses Taylor series approximation for exp() to avoid
    /// floating-point operations and maintain microsecond latency.
    /// 
    /// # Returns
    /// Penalty in basis points (scaled by FIXED_POINT_SCALE)
    pub fn calculate_penalty(&self) -> i64 {
        self.calc_count.fetch_add(1, Ordering::Relaxed);

        let position = self.atomic_position.load(Ordering::Acquire);
        let abs_position = position.unsigned_abs() as i64;
        let max_pos = self.config.max_position;

        // Normalize position to [0, 1] range
        let norm_position = if max_pos == 0 {
            0
        } else {
            (abs_position * FIXED_POINT_SCALE) / max_pos
        };

        // Clamp to 1.0 (100%)
        let norm_clamped = norm_position.min(FIXED_POINT_SCALE);

        // Calculate exponential penalty using Taylor series approximation
        // exp(k * x) ≈ 1 + kx + (kx)^2/2 + (kx)^3/6 + (kx)^4/24
        let k = self.config.curvature_k;
        let kx = (k * norm_clamped) / 1000; // Divide by 1000 to unscale k

        // Taylor series terms (fixed-point arithmetic)
        let term0 = FIXED_POINT_SCALE; // 1.0
        let term1 = kx; // kx
        let term2 = (kx * kx) / (2 * FIXED_POINT_SCALE); // (kx)^2 / 2
        let term3 = (kx * kx * kx) / (6 * FIXED_POINT_SCALE * FIXED_POINT_SCALE); // (kx)^3 / 6
        let term4 = (kx * kx * kx * kx) / (24 * FIXED_POINT_SCALE * FIXED_POINT_SCALE * FIXED_POINT_SCALE);

        let exp_approx = term0 + term1 + term2 + term3 + term4;

        // Apply base penalty and scale
        let raw_penalty = (self.config.base_penalty_bps * exp_approx) / FIXED_POINT_SCALE;

        // Add linear component for small positions
        let linear_penalty = (self.config.linear_coeff * norm_clamped) / FIXED_POINT_SCALE;

        let total_penalty = raw_penalty + linear_penalty;

        // Clamp to maximum
        total_penalty.min(self.config.max_penalty_bps)
    }

    /// Get asymmetric bid skew based on inventory
    /// 
    /// When long (positive position):
    /// - Bid is skewed down (less aggressive buying)
    /// - Ask is less skewed (more aggressive selling)
    /// 
    /// When short (negative position):
    /// - Bid is less skewed (more aggressive buying)
    /// - Ask is skewed up (less aggressive selling)
    /// 
    /// # Arguments
    /// * `base_bid` - Base bid price from pricing model
    /// * `base_ask` - Base ask price from pricing model
    /// 
    /// # Returns
    /// Tuple of (skewed_bid, skewed_ask)
    pub fn get_skewed_quotes(&self, base_bid: i64, base_ask: i64) -> (i64, i64) {
        let position = self.atomic_position.load(Ordering::Acquire);
        let penalty = self.calculate_penalty();

        // Determine direction of skew
        // Long position → skew bid down, ask neutral
        // Short position → skew ask up, bid neutral
        
        let position_sign = if position >= 0 { 1 } else { -1 };
        let abs_position = position.unsigned_abs() as i64;
        
        // Calculate position ratio (0 to 1, scaled)
        let position_ratio = if self.config.max_position == 0 {
            0
        } else {
            (abs_position * FIXED_POINT_SCALE) / self.config.max_position
        }.min(FIXED_POINT_SCALE);

        // Calculate directional penalty
        // Full penalty applied to side that increases position
        let directional_penalty = (penalty * position_ratio) / FIXED_POINT_SCALE;

        // Apply skew
        let skewed_bid = if position > 0 {
            // Long: reduce bid aggressiveness
            base_bid - (directional_penalty / 2)
        } else {
            // Short or flat: keep bid aggressive
            base_bid + (directional_penalty / 4)
        };

        let skewed_ask = if position < 0 {
            // Short: reduce ask aggressiveness
            base_ask + (directional_penalty / 2)
        } else {
            // Long or flat: keep ask aggressive
            base_ask - (directional_penalty / 4)
        };

        (skewed_bid, skewed_ask)
    }

    /// Check if position is in warning zone
    pub fn is_in_warning_zone(&self) -> bool {
        let position = self.atomic_position.load(Ordering::Acquire).unsigned_abs() as i64;
        let threshold = (self.config.max_position * self.config.warning_threshold) / FIXED_POINT_SCALE;
        position >= threshold
    }

    /// Check if position has hit maximum limit
    pub fn is_at_limit(&self) -> bool {
        let position = self.atomic_position.load(Ordering::Acquire).unsigned_abs() as i64;
        position >= self.config.max_position
    }

    /// Get remaining capacity before hitting limit
    pub fn get_remaining_capacity(&self) -> i64 {
        let position = self.atomic_position.load(Ordering::Acquire).unsigned_abs() as i64;
        (self.config.max_position - position).max(0)
    }

    /// Get current position (lock-free)
    pub fn get_current_position(&self) -> i64 {
        self.atomic_position.load(Ordering::Acquire)
    }

    /// Get inventory utilization percentage (scaled)
    pub fn get_utilization(&self) -> i64 {
        let position = self.atomic_position.load(Ordering::Acquire).unsigned_abs() as i64;
        if self.config.max_position == 0 {
            return 0;
        }
        (position * FIXED_POINT_SCALE) / self.config.max_position
    }

    /// Get state reference for monitoring
    pub fn get_state(&self) -> &InventoryState {
        &self.state
    }

    /// Force flatten position (emergency only)
    /// 
    /// Resets internal position tracking without actual trading.
    /// Should only be called during emergency risk management.
    pub fn force_flatten(&mut self) {
        self.state.current_position = 0;
        self.atomic_position.store(0, Ordering::Release);
        self.state.adjustment_count += 1;
        self.state.last_update = Instant::now();
    }
}

impl Drop for InventoryPenalty {
    fn drop(&mut self) {
        // Zero out sensitive state
        self.atomic_position.store(0, Ordering::Release);
        
        // Memory barrier
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = InventoryPenaltyConfig::default();
        assert_eq!(config.max_position, 10_000_000);
        assert_eq!(config.warning_threshold, 700_000); // 70%
    }

    #[test]
    fn test_inventory_update() {
        let config = InventoryPenaltyConfig::default();
        let mut penalty = InventoryPenalty::new(config);

        // Buy 3 units
        let pos = penalty.update_inventory(3_000_000);
        assert_eq!(pos, 3_000_000);

        // Sell 1 unit
        let pos = penalty.update_inventory(-1_000_000);
        assert_eq!(pos, 2_000_000);
    }

    #[test]
    fn test_penalty_increases_with_position() {
        let config = InventoryPenaltyConfig::default();
        let mut penalty = InventoryPenalty::new(config);

        // No position = minimal penalty
        let penalty_0 = penalty.calculate_penalty();

        // Add position
        penalty.update_inventory(5_000_000);
        let penalty_50 = penalty.calculate_penalty();

        // Penalty should increase
        assert!(penalty_50 > penalty_0);
    }

    #[test]
    fn test_quote_skew_long_position() {
        let config = InventoryPenaltyConfig::default();
        let mut penalty = InventoryPenalty::new(config);

        // Establish long position
        penalty.update_inventory(7_000_000); // 70% of max

        let base_bid = 100_000_000;
        let base_ask = 100_100_000;

        let (skewed_bid, skewed_ask) = penalty.get_skewed_quotes(base_bid, base_ask);

        // Long position should skew bid down (less aggressive buying)
        assert!(skewed_bid < base_bid);
    }

    #[test]
    fn test_warning_zone() {
        let config = InventoryPenaltyConfig::default();
        let mut penalty = InventoryPenalty::new(config);

        assert!(!penalty.is_in_warning_zone());

        // Move to 75% of max
        penalty.update_inventory(7_500_000);
        assert!(penalty.is_in_warning_zone());
    }

    #[test]
    fn test_position_limit() {
        let config = InventoryPenaltyConfig::default();
        let mut penalty = InventoryPenalty::new(config);

        // Move to max position
        penalty.update_inventory(config.max_position);
        assert!(penalty.is_at_limit());
    }
}
