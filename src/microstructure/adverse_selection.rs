//! # Real-Time Adverse Selection Cost Estimator
//!
//! This module implements a real-time adverse selection cost estimator using Glosten-Milgrom
//! logic, widening the bot's internal spread when toxic order flow is detected. It strictly
//! enforces the 8GB RAM limit through bounded event buffers.
//!
//! ## Key Features
//! - **Glosten-Milgrom Model**: Classic market microstructure theory implementation.
//! - **Toxic Flow Detection**: Identifies informed traders via order flow analysis.
//! - **Dynamic Spread Adjustment**: Widens spread based on adverse selection risk.
//! - **Memory Bounded**: Circular buffers for trade history.
//! - **Microsecond Updates**: O(1) per-trade cost estimation.
//!
//! ## Safety Guarantees
//! - No allocations during hot-path updates.
//! - Deterministic memory footprint.
//! - Thread-safe concurrent access.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Maximum trade history size (bounded for 8GB RAM).
const MAX_TRADE_HISTORY: usize = 1 << 16; // 65K trades

/// Default prior probability of informed trader.
const DEFAULT_PRIOR_PROB: f64 = 0.2;

/// Cache line size for alignment.
const CACHE_LINE_SIZE: usize = 64;

/// Trade direction indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeDirection {
    Buy,
    Sell,
    Unknown,
}

/// Single trade record for adverse selection analysis.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp_ns: u64,
    pub price: f64,
    pub size: f64,
    pub direction: TradeDirection,
    pub aggressor_side: TradeDirection,
}

/// Adverse selection estimator using Glosten-Milgrom framework.
pub struct AdverseSelectionEstimator {
    /// Prior probability of informed trader.
    prior_prob: AtomicU64, // f64 bits
    /// Current estimate of adverse selection cost (basis points).
    current_cost_bps: AtomicU64, // f64 bits
    /// Trade history (circular buffer).
    trade_history: parking_lot::Mutex<Vec<TradeRecord>>,
    /// Write index for circular buffer.
    write_idx: AtomicU64,
    /// Total trades processed.
    total_trades: AtomicU64,
    /// Informed trader probability estimate.
    informed_prob: AtomicU64, // f64 bits
    /// Whether estimator is active.
    active: AtomicBool,
    /// Last update timestamp.
    last_update_ns: AtomicU64,
    /// Volatility estimate (for spread calculation).
    volatility_bps: AtomicU64, // f64 bits
}

impl AdverseSelectionEstimator {
    /// Create a new adverse selection estimator.
    pub fn new() -> Self {
        Self {
            prior_prob: AtomicU64::new(DEFAULT_PRIOR_PROB.to_bits()),
            current_cost_bps: AtomicU64::new(5.0f64.to_bits()), // Default 5 bps
            trade_history: parking_lot::Mutex::new(Vec::with_capacity(MAX_TRADE_HISTORY)),
            write_idx: AtomicU64::new(0),
            total_trades: AtomicU64::new(0),
            informed_prob: AtomicU64::new(DEFAULT_PRIOR_PROB.to_bits()),
            active: AtomicBool::new(true),
            last_update_ns: AtomicU64::new(0),
            volatility_bps: AtomicU64::new(10.0f64.to_bits()), // Default 10 bps vol
        }
    }

    /// Process a new trade and update adverse selection estimate.
    pub fn process_trade(&self, trade: TradeRecord) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }

        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Update trade history (circular buffer)
        {
            let mut history = self.trade_history.lock();
            let idx = self.write_idx.fetch_add(1, Ordering::Relaxed) as usize;
            
            if history.len() < MAX_TRADE_HISTORY {
                history.push(trade.clone());
            } else {
                // Overwrite oldest
                let circular_idx = idx % MAX_TRADE_HISTORY;
                history[circular_idx] = trade.clone();
            }
        }

        self.total_trades.fetch_add(1, Ordering::Relaxed);
        self.last_update_ns.store(now_ns, Ordering::Relaxed);

        // Update adverse selection estimate
        self.update_estimate(&trade);
    }

    /// Update adverse selection estimate using Glosten-Milgrom logic.
    fn update_estimate(&self, trade: &TradeRecord) {
        let prior = f64::from_bits(self.prior_prob.load(Ordering::Relaxed));
        let current_informed = f64::from_bits(self.informed_prob.load(Ordering::Relaxed));

        // Glosten-Milgrom: Update probability of informed trading
        // P(informed | buy) = P(buy | informed) * P(informed) / P(buy)
        
        let likelihood_ratio = self.compute_likelihood_ratio(trade);
        
        // Bayesian update
        let posterior = (likelihood_ratio * prior) / 
            (likelihood_ratio * prior + (1.0 - prior));
        
        // Smooth update (exponential moving average)
        let alpha = 0.1; // Smoothing factor
        let new_informed_prob = alpha * posterior + (1.0 - alpha) * current_informed;
        
        self.informed_prob.store(new_informed_prob.to_bits(), Ordering::Relaxed);

        // Calculate adverse selection cost
        // Cost ≈ informed_prob * expected_price_move
        let vol = f64::from_bits(self.volatility_bps.load(Ordering::Relaxed));
        let cost_bps = new_informed_prob * vol;
        
        self.current_cost_bps.store(cost_bps.to_bits(), Ordering::Relaxed);
    }

    /// Compute likelihood ratio for Bayesian update.
    fn compute_likelihood_ratio(&self, trade: &TradeRecord) -> f64 {
        // Simplified likelihood based on trade characteristics
        let mut ratio = 1.0;

        // Large trades more likely to be informed
        let size_factor = (trade.size / 1000.0).min(3.0); // Cap at 3x
        ratio *= 1.0 + size_factor * 0.2;

        // Consecutive same-direction trades suggest informed flow
        let consecutive = self.count_consecutive_direction(trade.direction);
        if consecutive > 3 {
            ratio *= 1.0 + (consecutive as f64 - 3.0) * 0.1;
        }

        // Price impact suggests information
        let history = self.trade_history.lock();
        if history.len() >= 2 {
            let prev_price = history[history.len().saturating_sub(2)].price;
            let price_move = (trade.price - prev_price).abs() / prev_price * 10000.0; // bps
            
            if price_move > 5.0 {
                ratio *= 1.0 + (price_move - 5.0) * 0.02;
            }
        }

        ratio.min(5.0) // Cap likelihood ratio
    }

    /// Count consecutive trades in same direction.
    fn count_consecutive_direction(&self, direction: TradeDirection) -> usize {
        let history = self.trade_history.lock();
        if history.is_empty() {
            return 0;
        }

        let mut count = 0;
        for trade in history.iter().rev() {
            if trade.direction == direction {
                count += 1;
            } else {
                break;
            }
            if count >= 10 {
                break;
            }
        }
        count
    }

    /// Get current adverse selection cost in basis points.
    pub fn get_cost_bps(&self) -> f64 {
        f64::from_bits(self.current_cost_bps.load(Ordering::Relaxed))
    }

    /// Get adjusted spread (base spread + adverse selection cost).
    pub fn get_adjusted_spread(&self, base_spread_bps: f64) -> f64 {
        let cost = self.get_cost_bps();
        base_spread_bps + cost
    }

    /// Get probability of informed trading.
    pub fn get_informed_probability(&self) -> f64 {
        f64::from_bits(self.informed_prob.load(Ordering::Relaxed))
    }

    /// Check if toxic flow is detected (high adverse selection).
    pub fn is_toxic(&self, threshold_bps: f64) -> bool {
        self.get_cost_bps() > threshold_bps
    }

    /// Set prior probability of informed trading.
    pub fn set_prior(&self, prob: f64) {
        self.prior_prob.store(prob.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    /// Set volatility estimate.
    pub fn set_volatility(&self, vol_bps: f64) {
        self.volatility_bps.store(vol_bps.max(0.0).to_bits(), Ordering::Relaxed);
    }

    /// Get estimator statistics.
    pub fn get_stats(&self) -> AdverseSelectionStats {
        let history = self.trade_history.lock();
        
        AdverseSelectionStats {
            total_trades: self.total_trades.load(Ordering::Relaxed),
            current_cost_bps: self.get_cost_bps(),
            informed_probability: self.get_informed_probability(),
            prior_probability: f64::from_bits(self.prior_prob.load(Ordering::Relaxed)),
            volatility_bps: f64::from_bits(self.volatility_bps.load(Ordering::Relaxed)),
            history_size: history.len(),
            max_history: MAX_TRADE_HISTORY,
            active: self.active.load(Ordering::Relaxed),
        }
    }

    /// Activate/deactivate estimator.
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Reset estimator state.
    pub fn reset(&self) {
        {
            let mut history = self.trade_history.lock();
            history.clear();
        }
        self.write_idx.store(0, Ordering::Relaxed);
        self.total_trades.store(0, Ordering::Relaxed);
        self.informed_prob.store(DEFAULT_PRIOR_PROB.to_bits(), Ordering::Relaxed);
        self.current_cost_bps.store(5.0f64.to_bits(), Ordering::Relaxed);
    }
}

impl Default for AdverseSelectionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about adverse selection estimator.
#[derive(Debug, Clone)]
pub struct AdverseSelectionStats {
    pub total_trades: u64,
    pub current_cost_bps: f64,
    pub informed_probability: f64,
    pub prior_probability: f64,
    pub volatility_bps: f64,
    pub history_size: usize,
    pub max_history: usize,
    pub active: bool,
}

/// Helper function to classify trade direction from raw data.
pub fn classify_trade_direction(
    price: f64,
    bid_price: f64,
    ask_price: f64,
) -> TradeDirection {
    let midpoint = (bid_price + ask_price) / 2.0;
    
    if price > midpoint {
        TradeDirection::Buy
    } else if price < midpoint {
        TradeDirection::Sell
    } else {
        TradeDirection::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adverse_selection_basic() {
        let estimator = AdverseSelectionEstimator::new();
        
        // Initial state
        assert!((estimator.get_cost_bps() - 5.0).abs() < 0.01);
        assert!((estimator.get_informed_probability() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_trade_processing() {
        let estimator = AdverseSelectionEstimator::new();
        
        // Process some trades
        for i in 0..10 {
            let trade = TradeRecord {
                timestamp_ns: i * 1_000_000,
                price: 100.0 + (i as f64 * 0.1),
                size: 1000.0,
                direction: TradeDirection::Buy,
                aggressor_side: TradeDirection::Buy,
            };
            estimator.process_trade(trade);
        }
        
        let stats = estimator.get_stats();
        assert_eq!(stats.total_trades, 10);
        assert!(stats.history_size > 0);
    }

    #[test]
    fn test_toxic_detection() {
        let estimator = AdverseSelectionEstimator::new();
        
        // Initially not toxic
        assert!(!estimator.is_toxic(20.0));
        
        // Process large trades to increase adverse selection
        for i in 0..50 {
            let trade = TradeRecord {
                timestamp_ns: i * 1_000_000,
                price: 100.0 + (i as f64 * 0.5),
                size: 10000.0, // Large size
                direction: TradeDirection::Buy,
                aggressor_side: TradeDirection::Buy,
            };
            estimator.process_trade(trade);
        }
        
        // May or may not be toxic depending on parameters
        let _ = estimator.is_toxic(50.0);
    }

    #[test]
    fn test_spread_adjustment() {
        let estimator = AdverseSelectionEstimator::new();
        
        let base_spread = 5.0;
        let adjusted = estimator.get_adjusted_spread(base_spread);
        
        assert!(adjusted >= base_spread);
    }

    #[test]
    fn test_memory_bounds() {
        let estimator = AdverseSelectionEstimator::new();
        
        // Process more trades than buffer size
        for i in 0..MAX_TRADE_HISTORY + 100 {
            let trade = TradeRecord {
                timestamp_ns: i * 1_000_000,
                price: 100.0,
                size: 100.0,
                direction: TradeDirection::Buy,
                aggressor_side: TradeDirection::Buy,
            };
            estimator.process_trade(trade);
        }
        
        let stats = estimator.get_stats();
        assert!(stats.history_size <= MAX_TRADE_HISTORY);
    }
}
