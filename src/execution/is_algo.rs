//! Implementation Shortfall (IS) Algorithm
//! 
//! This module implements the Implementation Shortfall execution algorithm, designed to minimize
//! the difference between the decision price and the final execution price while balancing
//! market impact against delay costs. Optimized for microsecond latency on AMD Ryzen AI 5.
//!
//! Key Features:
//! - Dynamic aggression adjustment based on real-time volatility (ATR) and order book depth
//! - Binance-specific maker/taker fee and rebate integration
//! - Lock-free state management for concurrent access
//! - Strict memory allocation within 8GB global limit

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};
use crate::memory::allocator::GlobalMemoryTracker;
use crate::network::tcp_tuning::LatencyMonitor;

/// Configuration for the Implementation Shortfall algorithm
#[derive(Debug, Clone)]
pub struct ISConfig {
    /// Maximum participation rate (0.0 - 1.0)
    pub max_participation_rate: f64,
    /// Aggression factor (higher = more aggressive)
    pub base_aggression: f64,
    /// Volatility sensitivity (how much vol affects aggression)
    pub vol_sensitivity: f64,
    /// Depth sensitivity (how much order book depth affects aggression)
    pub depth_sensitivity: f64,
    /// Minimum order size in quote currency
    pub min_order_size: f64,
    /// Binance maker fee (negative for rebates)
    pub maker_fee: f64,
    /// Binance taker fee
    pub taker_fee: f64,
    /// Target completion time in milliseconds
    pub target_completion_ms: u64,
}

impl Default for ISConfig {
    fn default() -> Self {
        Self {
            max_participation_rate: 0.15,
            base_aggression: 0.5,
            vol_sensitivity: 0.3,
            depth_sensitivity: 0.4,
            min_order_size: 10.0,
            maker_fee: -0.0001, // VIP rebate
            taker_fee: 0.0004,
            target_completion_ms: 60000,
        }
    }
}

/// Real-time state of the IS algorithm
pub struct ISState {
    /// Decision price (price when order was initiated)
    pub decision_price: AtomicU64, // Stored as u64 fixed-point (price * 1e8)
    /// Remaining quantity to execute
    pub remaining_qty: AtomicU64,
    /// Executed quantity
    pub executed_qty: AtomicU64,
    /// Total value executed (for VWAP calculation)
    pub total_value: AtomicU64,
    /// Start time of execution
    pub start_time: Instant,
    /// Active flag
    pub is_active: AtomicBool,
    /// Current aggression level (0-100)
    pub current_aggression: AtomicU64,
}

impl ISState {
    pub fn new(decision_price: u64, initial_qty: u64) -> Self {
        GlobalMemoryTracker::allocate(256).expect("ISState allocation failed");
        
        Self {
            decision_price: AtomicU64::new(decision_price),
            remaining_qty: AtomicU64::new(initial_qty),
            executed_qty: AtomicU64::new(0),
            total_value: AtomicU64::new(0),
            start_time: Instant::now(),
            is_active: AtomicBool::new(true),
            current_aggression: AtomicU64::new(50),
        }
    }

    #[inline]
    pub fn get_decision_price(&self) -> f64 {
        self.decision_price.load(Ordering::Relaxed) as f64 / 1e8
    }

    #[inline]
    pub fn get_remaining_qty(&self) -> u64 {
        self.remaining_qty.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_executed_qty(&self) -> u64 {
        self.executed_qty.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn get_avg_execution_price(&self) -> f64 {
        let executed = self.executed_qty.load(Ordering::Relaxed);
        if executed == 0 {
            return 0.0;
        }
        let total_val = self.total_value.load(Ordering::Relaxed);
        (total_val as f64 / 1e8) / (executed as f64 / 1e8)
    }

    #[inline]
    pub fn update_execution(&self, qty: u64, price: u64) {
        self.executed_qty.fetch_add(qty, Ordering::Relaxed);
        self.remaining_qty.fetch_sub(qty, Ordering::Release);
        
        // Update total value (price * qty in fixed-point)
        let value = (price as u128 * qty as u128) as u64;
        self.total_value.fetch_add(value, Ordering::Relaxed);
    }

    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_qty.load(Ordering::Acquire) == 0 || !self.is_active.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }
}

impl Drop for ISState {
    fn drop(&mut self) {
        GlobalMemoryTracker::deallocate(256);
    }
}

/// Implementation Shortfall Execution Engine
pub struct ISExecutionEngine {
    config: ISConfig,
    latency_monitor: LatencyMonitor,
}

impl ISExecutionEngine {
    pub fn new(config: ISConfig) -> Self {
        Self {
            config,
            latency_monitor: LatencyMonitor::new(),
        }
    }

    /// Calculate dynamic aggression based on volatility and order book depth
    /// Returns aggression level 0.0 - 1.0
    #[inline]
    pub fn calculate_aggression(
        &self,
        current_volatility: f64, // ATR or similar
        bid_depth: f64,          // Quantity at best bid
        ask_depth: f64,          // Quantity at best ask
        is_buy: bool,
        elapsed_ratio: f64,      // 0.0 - 1.0 (elapsed / target)
    ) -> f64 {
        let _latency = self.latency_monitor.start_operation();

        // Base aggression from config
        let mut aggression = self.config.base_aggression;

        // Volatility adjustment: higher vol = more aggressive to avoid adverse selection
        let vol_adjustment = current_volatility * self.config.vol_sensitivity;
        aggression += vol_adjustment;

        // Depth adjustment: lower depth = more aggressive (liquidity might disappear)
        let relevant_depth = if is_buy { ask_depth } else { bid_depth };
        let depth_factor = if relevant_depth > 0.0 {
            1.0 - (relevant_depth / 1000000.0).min(1.0) // Normalize to 1M units
        } else {
            1.0
        };
        let depth_adjustment = depth_factor * self.config.depth_sensitivity;
        aggression += depth_adjustment;

        // Time urgency: as deadline approaches, increase aggression
        let urgency_factor = elapsed_ratio.powi(2); // Quadratic increase
        aggression += urgency_factor * 0.3;

        // Cap aggression between 0.0 and 1.0
        aggression.clamp(0.0, 1.0)
    }

    /// Determine order type and size based on current market conditions
    /// Returns (is_market_order, quantity)
    #[inline]
    pub fn determine_order(
        &self,
        state: &ISState,
        current_price: f64,
        bid_depth: f64,
        ask_depth: f64,
        current_volatility: f64,
        is_buy: bool,
    ) -> (bool, f64) {
        let _latency = self.latency_monitor.start_operation();

        let remaining = state.get_remaining_qty() as f64 / 1e8;
        if remaining < self.config.min_order_size {
            return (false, 0.0);
        }

        let elapsed = state.start_time.elapsed().as_millis() as u64;
        let target = self.config.target_completion_ms;
        let elapsed_ratio = (elapsed as f64) / (target as f64);

        let aggression = self.calculate_aggression(
            current_volatility,
            bid_depth,
            ask_depth,
            is_buy,
            elapsed_ratio,
        );

        // Update state aggression
        state.current_aggression.store((aggression * 100.0) as u64, Ordering::Relaxed);

        // Calculate ideal quantity based on time remaining
        let time_remaining_ratio = 1.0 - elapsed_ratio;
        let ideal_qty = remaining * (1.0 / time_remaining_ratio.max(0.01));

        // Apply participation rate limit
        let market_volume_estimate = if is_buy { ask_depth } else { bid_depth };
        let max_qty = market_volume_estimate * self.config.max_participation_rate;

        let order_qty = ideal_qty.min(max_qty).min(remaining);

        // Decide between limit and market order based on aggression
        // High aggression (>0.7) = market order, otherwise limit order
        let is_market = aggression > 0.7;

        (is_market, order_qty)
    }

    /// Calculate implementation shortfall in basis points
    #[inline]
    pub fn calculate_is_bps(&self, state: &ISState, side: i8) -> f64 {
        let decision_price = state.get_decision_price();
        let avg_exec_price = state.get_avg_execution_price();
        
        if decision_price == 0.0 || avg_exec_price == 0.0 {
            return 0.0;
        }

        let price_diff = if side > 0 {
            // Buy: positive IS means we paid more than decision price
            avg_exec_price - decision_price
        } else {
            // Sell: positive IS means we received less than decision price
            decision_price - avg_exec_price
        };

        (price_diff / decision_price) * 10000.0 // Basis points
    }

    /// Calculate total execution cost including fees
    #[inline]
    pub fn calculate_total_cost(&self, executed_qty: f64, avg_price: f64, is_maker: bool) -> f64 {
        let notional = executed_qty * avg_price;
        let fee_rate = if is_maker {
            self.config.maker_fee
        } else {
            self.config.taker_fee
        };
        notional * fee_rate.abs()
    }

    /// Log execution metrics for post-mortem analysis (SOUL.md)
    pub fn log_metrics(&self, state: &ISState, side: i8, symbol: &str) {
        let is_bps = self.calculate_is_bps(state, side);
        let elapsed_ms = state.start_time.elapsed().as_millis();
        let avg_price = state.get_avg_execution_price();
        let exec_qty = state.get_executed_qty() as f64 / 1e8;
        
        eprintln!(
            "[IS_METRIC] symbol={} side={} is_bps={:.2} elapsed_ms={} avg_price={:.8} qty={:.8}",
            symbol,
            if side > 0 { "BUY" } else { "SELL" },
            is_bps,
            elapsed_ms,
            avg_price,
            exec_qty
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_state_creation() {
        let state = ISState::new(50000_00000000, 1000_00000000); // $50k, 1000 units
        assert_eq!(state.get_decision_price(), 50000.0);
        assert_eq!(state.get_remaining_qty(), 1000_00000000);
        assert!(!state.is_complete());
    }

    #[test]
    fn test_aggression_calculation() {
        let engine = ISExecutionEngine::new(ISConfig::default());
        
        // Low vol, high depth, early time = low aggression
        let agg1 = engine.calculate_aggression(0.01, 500000.0, 500000.0, true, 0.1);
        assert!(agg1 < 0.5);

        // High vol, low depth, late time = high aggression
        let agg2 = engine.calculate_aggression(0.1, 10000.0, 10000.0, true, 0.9);
        assert!(agg2 > 0.7);
    }

    #[test]
    fn test_execution_update() {
        let state = ISState::new(50000_00000000, 1000_00000000);
        state.update_execution(500_00000000, 50001_00000000);
        
        assert_eq!(state.get_executed_qty(), 500_00000000);
        assert_eq!(state.get_remaining_qty(), 500_00000000);
        assert!(state.get_avg_execution_price() > 50000.0);
    }
}
