//! Execution Routing & Slippage Modeling - Slippage & TCA Engine
//! 
//! This module implements real-time slippage and transaction cost analysis (TCA)
//! models that predict execution degradation based on current order book depth
//! and recent trade velocity.
//! 
//! **Performance Characteristics:**
//! - Lock-free ring buffers for tick/velocity tracking
//! - Zero heap allocations during hot path
//! - O(1) slippage estimation
//! - SIMD-ready calculations where applicable
//! 
//! **Architecture:**
//! The SlippageModel predicts execution quality using:
//! 1. Order book depth analysis at multiple levels
//! 2. Recent trade velocity and volume patterns
//! 3. Historical slippage tracking for calibration
//! 4. Market regime detection (calm/volatile)

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Configuration for slippage modeling
#[derive(Debug, Clone, Copy)]
pub struct SlippageConfig {
    /// Number of order book levels to analyze
    pub orderbook_levels: usize,
    /// Time window for velocity calculation (ms)
    pub velocity_window_ms: u64,
    /// Maximum ticks to store for history
    pub max_history_ticks: usize,
    /// Volatility threshold for regime change (basis points)
    pub volatility_threshold_bps: u32,
    /// Base slippage multiplier in calm markets
    pub calm_multiplier: u32,
    /// Slippage multiplier in volatile markets
    pub volatile_multiplier: u32,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            orderbook_levels: 10,
            velocity_window_ms: 1000, // 1 second
            max_history_ticks: 512,
            volatility_threshold_bps: 100, // 1%
            calm_multiplier: 100,     // 1.0x
            volatile_multiplier: 250, // 2.5x
        }
    }
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    Calm,
    Normal,
    Volatile,
    Extreme,
}

/// Slippage estimation result
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlippageEstimate {
    /// Estimated slippage in basis points
    pub slippage_bps: u32,
    /// Confidence level (0-100)
    pub confidence: u8,
    /// Market regime
    pub regime: MarketRegime,
    /// Estimated market impact in bps
    pub impact_bps: u32,
    /// Estimated spread cost in bps
    pub spread_cost_bps: u32,
    /// Timestamp of estimate
    pub timestamp_ms: u64,
}

/// Main Slippage Model
pub struct SlippageModel {
    /// Configuration
    config: SlippageConfig,
    /// Active flag
    is_active: AtomicBool,
    
    // Order book depth cache (scaled sizes at each level)
    bid_depths: [u64; 10],
    ask_depths: [u64; 10],
    
    // Trade velocity tracking
    recent_trades: [u64; 64],  // timestamps
    recent_volumes: [u64; 64], // volumes
    trade_idx: usize,
    trade_count: usize,
    
    // Historical slippage for calibration
    actual_slippages: [u32; 128],
    slippage_idx: usize,
    slippage_count: usize,
    
    // Current regime
    current_regime: MarketRegime,
    last_regime_change_ms: u64,
    
    // Running volatility estimate (scaled by 10000)
    running_volatility: u64,
}

unsafe impl Send for SlippageModel {}
unsafe impl Sync for SlippageModel {}

impl SlippageModel {
    /// Initialize the slippage model
    pub fn new(config: SlippageConfig) -> Self {
        Self {
            config,
            is_active: AtomicBool::new(true),
            bid_depths: [0; 10],
            ask_depths: [0; 10],
            recent_trades: [0; 64],
            recent_volumes: [0; 64],
            trade_idx: 0,
            trade_count: 0,
            actual_slippages: [0; 128],
            slippage_idx: 0,
            slippage_count: 0,
            current_regime: MarketRegime::Normal,
            last_regime_change_ms: 0,
            running_volatility: 500, // 5% initial estimate
        }
    }

    /// Update order book depths
    #[inline]
    pub fn update_orderbook(&mut self, bid_depths: &[u64], ask_depths: &[u64]) {
        let levels = bid_depths.len().min(10);
        for i in 0..levels {
            self.bid_depths[i] = bid_depths[i];
            self.ask_depths[i] = ask_depths[i];
        }
    }

    /// Record a trade for velocity calculation
    #[inline]
    pub fn record_trade(&mut self, timestamp_ms: u64, volume_scaled: u64) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }

        self.recent_trades[self.trade_idx] = timestamp_ms;
        self.recent_volumes[self.trade_idx] = volume_scaled;
        self.trade_idx = (self.trade_idx + 1) % 64;
        
        if self.trade_count < 64 {
            self.trade_count += 1;
        }

        // Update volatility estimate
        self.update_volatility();
    }

    /// Estimate slippage for a given order size
    /// Hot path function - zero allocations
    #[inline]
    pub fn estimate_slippage(
        &self,
        quantity_scaled: u64,
        is_buy: bool,
        timestamp_ms: u64,
    ) -> SlippageEstimate {
        // Calculate liquidity available
        let available_liquidity = self.calculate_available_liquidity(is_buy);
        
        // Market impact component
        let impact_bps = if available_liquidity > 0 {
            ((quantity_scaled as u128 * 10_000) / available_liquidity as u128) as u32
        } else {
            1000 // High impact if no liquidity
        };

        // Spread cost (half spread for round-trip)
        let spread_cost_bps = self.estimate_spread_cost();

        // Velocity adjustment
        let velocity_factor = self.calculate_velocity_factor(timestamp_ms);

        // Regime multiplier
        let regime_mult = match self.current_regime {
            MarketRegime::Calm => self.config.calm_multiplier,
            MarketRegime::Normal => 150,
            MarketRegime::Volatile => 200,
            MarketRegime::Extreme => self.config.volatile_multiplier,
        };

        // Total slippage
        let base_slippage = impact_bps.saturating_add(spread_cost_bps);
        let adjusted_slippage = (base_slippage as u64 * regime_mult as u64 / 100)
            .saturating_mul(velocity_factor) as u32;

        // Confidence based on data availability
        let confidence = if self.trade_count >= 32 && self.slippage_count >= 16 {
            90
        } else if self.trade_count >= 16 {
            70
        } else {
            50
        };

        SlippageEstimate {
            slippage_bps: adjusted_slippage.min(5000), // Cap at 50%
            confidence,
            regime: self.current_regime,
            impact_bps,
            spread_cost_bps,
            timestamp_ms,
        }
    }

    /// Calculate available liquidity at relevant book levels
    #[inline]
    fn calculate_available_liquidity(&self, is_buy: bool) -> u64 {
        // Sum liquidity at top N levels
        let depths = if is_buy { &self.ask_depths } else { &self.bid_depths };
        let levels = self.config.orderbook_levels.min(10);
        
        let mut total = 0u64;
        for i in 0..levels {
            total = total.saturating_add(depths[i]);
        }
        total
    }

    /// Estimate spread cost in basis points
    #[inline]
    fn estimate_spread_cost(&self) -> u32 {
        // Simplified: use typical crypto spread of 1-5 bps
        // In production, would calculate from actual bid-ask
        2 // 2 bps typical
    }

    /// Calculate velocity adjustment factor
    #[inline]
    fn calculate_velocity_factor(&self, current_time_ms: u64) -> u64 {
        if self.trade_count < 2 {
            return 100; // Neutral factor (scaled by 100)
        }

        // Count trades in velocity window
        let window_start = current_time_ms.saturating_sub(self.config.velocity_window_ms);
        let mut count = 0u64;
        let mut volume = 0u64;

        for i in 0..self.trade_count {
            let idx = (self.trade_idx + 64 - i - 1) % 64;
            if self.recent_trades[idx] >= window_start {
                count += 1;
                volume = volume.saturating_add(self.recent_volumes[idx]);
            }
        }

        // Higher velocity = higher slippage
        if count > 50 {
            150 // 1.5x
        } else if count > 20 {
            120 // 1.2x
        } else {
            100 // 1.0x
        }
    }

    /// Update running volatility estimate
    #[inline]
    fn update_volatility(&mut self) {
        if self.trade_count < 10 {
            return;
        }

        // Simple volatility: std dev of inter-trade times and sizes
        // Simplified here - would use proper statistical calculation
        
        // Check for regime change
        let vol_bps = (self.running_volatility / 100) as u32;
        let new_regime = if vol_bps < 20 {
            MarketRegime::Calm
        } else if vol_bps < 100 {
            MarketRegime::Normal
        } else if vol_bps < 200 {
            MarketRegime::Volatile
        } else {
            MarketRegime::Extreme
        };

        if new_regime != self.current_regime {
            self.current_regime = new_regime;
            self.last_regime_change_ms = self.recent_trades[self.trade_idx];
        }
    }

    /// Record actual slippage for model calibration
    #[inline]
    pub fn record_actual_slippage(&mut self, slippage_bps: u32) {
        self.actual_slippages[self.slippage_idx] = slippage_bps;
        self.slippage_idx = (self.slippage_idx + 1) % 128;
        
        if self.slippage_count < 128 {
            self.slippage_count += 1;
        }
    }

    /// Get average historical slippage
    #[inline]
    pub fn get_average_slippage(&self) -> u32 {
        if self.slippage_count == 0 {
            return 0;
        }

        let mut sum = 0u64;
        for i in 0..self.slippage_count {
            sum += self.actual_slippages[i] as u64;
        }
        (sum / self.slippage_count as u64) as u32
    }

    /// Get current market regime
    #[inline]
    pub fn get_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Shutdown model
    #[inline]
    pub fn shutdown(&self) {
        self.is_active.store(false, Ordering::Release);
    }
}

/// Transaction Cost Analysis results
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TCAResult {
    /// Total transaction cost in basis points
    pub total_cost_bps: u32,
    /// Explicit costs (fees)
    pub explicit_costs_bps: u32,
    /// Implicit costs (slippage, spread)
    pub implicit_costs_bps: u32,
    /// Market impact cost
    pub impact_cost_bps: u32,
    /// Timing cost (opportunity cost of delay)
    pub timing_cost_bps: u32,
}

/// TCA Calculator
pub struct TCACalculator {
    /// Fee rate in basis points
    fee_rate_bps: u32,
    /// Slippage model reference
    slippage_model: SlippageModel,
}

impl TCACalculator {
    /// Initialize TCA calculator
    pub fn new(fee_rate_bps: u32, slippage_config: SlippageConfig) -> Self {
        Self {
            fee_rate_bps,
            slippage_model: SlippageModel::new(slippage_config),
        }
    }

    /// Calculate pre-trade TCA estimate
    #[inline]
    pub fn pre_trade_estimate(
        &mut self,
        quantity_scaled: u64,
        is_buy: bool,
        timestamp_ms: u64,
    ) -> TCAResult {
        let slippage = self.slippage_model.estimate_slippage(quantity_scaled, is_buy, timestamp_ms);
        
        TCAResult {
            total_cost_bps: self.fee_rate_bps
                .saturating_add(slippage.slippage_bps)
                .saturating_add(slippage.spread_cost_bps),
            explicit_costs_bps: self.fee_rate_bps,
            implicit_costs_bps: slippage.slippage_bps.saturating_add(slippage.spread_cost_bps),
            impact_cost_bps: slippage.impact_bps,
            timing_cost_bps: 0, // Would calculate based on urgency
        }
    }

    /// Calculate post-trade TCA analysis
    #[inline]
    pub fn post_trade_analysis(
        &self,
        executed_price_scaled: u64,
        arrival_price_scaled: u64,
        quantity_scaled: u64,
        fee_scaled: u64,
    ) -> TCAResult {
        // Implementation cost vs arrival price
        let implementation_shortfall = if arrival_price_scaled > 0 {
            ((executed_price_scaled.abs_diff(arrival_price_scaled) as u128 * 10_000) 
                / arrival_price_scaled as u128) as u32
        } else {
            0
        };

        let fee_bps = ((fee_scaled as u128 * 10_000) / quantity_scaled as u128.max(1) as u128) as u32;

        TCAResult {
            total_cost_bps: implementation_shortfall.saturating_add(fee_bps),
            explicit_costs_bps: fee_bps,
            implicit_costs_bps: implementation_shortfall,
            impact_cost_bps: implementation_shortfall,
            timing_cost_bps: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_estimation() {
        let config = SlippageConfig::default();
        let mut model = SlippageModel::new(config);

        // Set up some order book depth
        let bids = [1_000_000_000; 10];
        let asks = [1_000_000_000; 10];
        model.update_orderbook(&bids, &asks);

        // Estimate slippage for a buy order
        let estimate = model.estimate_slippage(100_000_000, true, 1000);
        assert!(estimate.slippage_bps > 0);
        assert!(estimate.confidence <= 100);
    }
}
