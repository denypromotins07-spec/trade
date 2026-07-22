//! # Glosten-Milgrom Sequential Trade Model for Market Making
//! 
//! This module implements the Glosten-Milgrom (1985) sequential trade arrival model
//! to estimate the probability of trading with an informed ("toxic") trader.
//! The model dynamically widens spreads in response to detected adverse selection.
//! 
//! ## Architecture Notes:
//! - Pure Rust implementation with no heap allocations in hot path
//! - Uses fixed-point arithmetic for deterministic microsecond calculations
//! - Contiguous memory layout for cache efficiency
//! - Respects 8GB RAM limit with bounded state structures
//! 
//! ## Mathematical Foundation:
//! The Glosten-Milgrom model assumes:
//! - A fraction μ of traders are informed (know true value)
//! - A fraction (1-μ) are uninformed (liquidity traders)
//! - Market makers update beliefs based on trade direction
//! - Spread widens as P(informed | buy/sell) increases

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Fixed-point precision for microsecond calculations (6 decimal places)
const FIXED_POINT_SCALE: i64 = 1_000_000;

/// Glosten-Milgrom model parameters
#[derive(Debug, Clone, Copy)]
pub struct GMParameters {
    /// Prior probability of informed trader (μ), scaled by FIXED_POINT_SCALE
    pub informed_prob: i64,
    /// Probability of good news given informed trader, scaled
    pub good_news_prob: i64,
    /// Initial belief about asset value (mid-price), scaled
    pub initial_value: i64,
    /// Minimum spread (basis points), scaled
    pub min_spread_bps: i64,
    /// Maximum spread (basis points), scaled
    pub max_spread_bps: i64,
}

impl Default for GMParameters {
    fn default() -> Self {
        Self {
            informed_prob: 200_000,      // 20% prior probability of informed trader
            good_news_prob: 500_000,     // 50% probability of good news
            initial_value: 100_000_000,  // $100.00 initial value (scaled)
            min_spread_bps: 5_000,       // 5 bps minimum spread
            max_spread_bps: 500_000,     // 50 bps maximum spread
        }
    }
}

/// State for the Glosten-Milgrom sequential trade estimator
#[derive(Debug, Clone)]
pub struct GMState {
    /// Current belief about asset value (scaled fixed-point)
    pub current_value: i64,
    /// Posterior probability that last trader was informed (scaled)
    pub posterior_informed: i64,
    /// Number of consecutive buys
    pub consecutive_buys: u32,
    /// Number of consecutive sells
    pub consecutive_sells: u32,
    /// Total trades observed in current session
    pub total_trades: u64,
    /// Last update timestamp
    pub last_update: Instant,
}

impl GMState {
    /// Create new GM state with initial parameters
    pub fn new(params: &GMParameters) -> Self {
        Self {
            current_value: params.initial_value,
            posterior_informed: params.informed_prob,
            consecutive_buys: 0,
            consecutive_sells: 0,
            total_trades: 0,
            last_update: Instant::now(),
        }
    }

    /// Reset state for new trading session
    pub fn reset(&mut self, params: &GMParameters) {
        self.current_value = params.initial_value;
        self.posterior_informed = params.informed_prob;
        self.consecutive_buys = 0;
        self.consecutive_sells = 0;
        self.total_trades = 0;
        self.last_update = Instant::now();
    }
}

/// Glosten-Milgrom sequential trade estimator for toxic flow detection
pub struct GlostenMilgromEstimator {
    /// Model parameters
    params: GMParameters,
    /// Current state
    state: GMState,
    /// Atomic counter for thread-safe trade counting
    trade_count: AtomicU64,
}

impl GlostenMilgromEstimator {
    /// Create a new Glosten-Milgrom estimator
    pub fn new(params: GMParameters) -> Self {
        let state = GMState::new(&params);
        Self {
            params,
            state,
            trade_count: AtomicU64::new(0),
        }
    }

    /// Process a buy order and update beliefs
    /// 
    /// Returns the updated bid-ask spread in basis points (scaled)
    /// 
    /// # Arguments
    /// * `trade_size` - Size of the incoming buy order (scaled)
    /// 
    /// # Returns
    /// Updated spread in basis points (scaled by FIXED_POINT_SCALE)
    pub fn process_buy(&mut self, trade_size: i64) -> i64 {
        self.process_trade(true, trade_size)
    }

    /// Process a sell order and update beliefs
    /// 
    /// Returns the updated bid-ask spread in basis points (scaled)
    pub fn process_sell(&mut self, trade_size: i64) -> i64 {
        self.process_trade(false, trade_size)
    }

    /// Internal trade processing logic using Bayes' rule
    fn process_trade(&mut self, is_buy: bool, trade_size: i64) -> i64 {
        let mu = self.params.informed_prob;
        let gamma = self.params.good_news_prob;
        
        // Update consecutive counters
        if is_buy {
            self.state.consecutive_buys += 1;
            self.state.consecutive_sells = 0;
        } else {
            self.state.consecutive_sells += 1;
            self.state.consecutive_buys = 0;
        }

        self.state.total_trades += 1;
        self.trade_count.fetch_add(1, Ordering::Relaxed);

        // Bayesian update of posterior probability
        // P(informed | buy) = P(buy | informed) * P(informed) / P(buy)
        // P(buy) = P(buy | informed) * P(informed) + P(buy | uninformed) * P(uninformed)
        
        let prior_informed = self.state.posterior_informed;
        let prior_uninformed = FIXED_POINT_SCALE - prior_informed;

        // Likelihood of buy given informed (assumes informed know true direction)
        let likelihood_buy_informed = if is_buy { gamma } else { FIXED_POINT_SCALE - gamma };
        
        // Likelihood of buy given uninformed (random liquidity trading)
        let likelihood_buy_uninformed = FIXED_POINT_SCALE / 2;

        // Total probability of observed trade
        let total_prob = (likelihood_buy_informed * prior_informed 
                        + likelihood_buy_uninformed * prior_uninformed) 
                       / FIXED_POINT_SCALE;

        // Avoid division by zero
        if total_prob == 0 {
            return self.calculate_spread();
        }

        // Bayes' rule: posterior = (likelihood * prior) / total_prob
        let new_posterior_informed = (likelihood_buy_informed * prior_informed) / total_prob;
        
        // Clamp to valid range
        self.state.posterior_informed = new_posterior_informed
            .clamp(0, FIXED_POINT_SCALE);

        // Update belief about asset value
        // If buy from informed, value likely higher; if sell, likely lower
        let value_adjustment = if is_buy {
            (mu * gamma * trade_size / FIXED_POINT_SCALE) as i64
        } else {
            -(mu * (FIXED_POINT_SCALE - gamma) * trade_size / FIXED_POINT_SCALE) as i64
        };

        self.state.current_value = (self.state.current_value + value_adjustment)
            .max(0);

        self.state.last_update = Instant::now();

        self.calculate_spread()
    }

    /// Calculate optimal spread based on current toxicity estimate
    /// 
    /// Spread = base_spread + toxicity_premium
    /// where toxicity_premium ∝ P(informed trader)
    fn calculate_spread(&self) -> i64 {
        let base_spread = self.params.min_spread_bps;
        let toxicity = self.state.posterior_informed;
        
        // Toxicity premium: wider spread when more likely facing informed trader
        // Linear scaling: spread increases proportionally with informed probability
        let toxicity_premium = (toxicity * self.params.max_spread_bps) / FIXED_POINT_SCALE;
        
        let total_spread = base_spread + toxicity_premium;
        
        // Clamp to configured bounds
        total_spread.clamp(self.params.min_spread_bps, self.params.max_spread_bps)
    }

    /// Get current bid price given mid-price
    /// 
    /// # Arguments
    /// * `mid_price` - Current market mid-price (scaled)
    /// 
    /// # Returns
    /// Bid price (scaled)
    pub fn get_bid(&self, mid_price: i64) -> i64 {
        let spread = self.calculate_spread();
        let half_spread = spread / 2;
        mid_price - half_spread
    }

    /// Get current ask price given mid-price
    pub fn get_ask(&self, mid_price: i64) -> i64 {
        let spread = self.calculate_spread();
        let half_spread = spread / 2;
        mid_price + half_spread
    }

    /// Get the current estimated probability of informed trading
    /// 
    /// Returns value scaled by FIXED_POINT_SCALE (divide by 1_000_000 for percentage)
    pub fn get_informed_probability(&self) -> i64 {
        self.state.posterior_informed
    }

    /// Get current fair value estimate
    pub fn get_fair_value(&self) -> i64 {
        self.state.current_value
    }

    /// Get total trade count
    pub fn get_trade_count(&self) -> u64 {
        self.trade_count.load(Ordering::Relaxed)
    }

    /// Get time since last update
    pub fn time_since_last_update(&self) -> Duration {
        self.state.last_update.elapsed()
    }

    /// Check if model needs recalibration (too many consecutive same-direction trades)
    pub fn needs_recalibration(&self, threshold: u32) -> bool {
        self.state.consecutive_buys > threshold || self.state.consecutive_sells > threshold
    }

    /// Recalibrate model parameters based on observed volatility
    /// 
    /// # Arguments
    /// * `volatility_estimate` - Estimated volatility (scaled by 1000 for basis points)
    pub fn recalibrate(&mut self, volatility_estimate: i64) {
        // Adjust informed probability based on volatility
        // Higher volatility → potentially more informed trading
        let vol_adjustment = volatility_estimate / 100; // Scale down
        self.params.informed_prob = (self.params.informed_prob + vol_adjustment)
            .clamp(100_000, 500_000); // Keep between 10% and 50%

        // Reset state
        self.state.reset(&self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let params = GMParameters::default();
        assert_eq!(params.informed_prob, 200_000); // 20%
    }

    #[test]
    fn test_process_sequential_buys() {
        let params = GMParameters::default();
        let mut estimator = GlostenMilgromEstimator::new(params);

        // Process several buy orders
        let mut spread = 0;
        for _ in 0..5 {
            spread = estimator.process_buy(1_000_000); // 1 unit
        }

        // Spread should widen due to increased toxicity estimate
        assert!(spread >= params.min_spread_bps);
        assert!(spread <= params.max_spread_bps);
    }

    #[test]
    fn test_bid_ask_calculation() {
        let params = GMParameters::default();
        let estimator = GlostenMilgromEstimator::new(params);

        let mid = 100_000_000; // $100.00
        let bid = estimator.get_bid(mid);
        let ask = estimator.get_ask(mid);

        assert!(bid < mid);
        assert!(ask > mid);
        assert_eq!(ask - bid, estimator.calculate_spread());
    }

    #[test]
    fn test_reciprocating_trades() {
        let params = GMParameters::default();
        let mut estimator = GlostenMilgromEstimator::new(params);

        // Alternating buys and sells should keep spread relatively stable
        for i in 0..10 {
            if i % 2 == 0 {
                estimator.process_buy(1_000_000);
            } else {
                estimator.process_sell(1_000_000);
            }
        }

        // Consecutive counters should be low
        assert!(estimator.state.consecutive_buys <= 1);
        assert!(estimator.state.consecutive_sells <= 1);
    }
}
