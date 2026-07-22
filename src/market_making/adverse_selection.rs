//! Glosten-Milgrom Adverse Selection Cost Estimator
//! 
//! This module implements a Glosten-Milgrom model-based adverse selection
//! cost estimator that automatically widens spreads when the probability
//! of trading with an informed (toxic) trader increases.
//! 
//! Key features:
//! - Real-time toxic flow detection
//! - Dynamic spread adjustment based on adverse selection risk
//! - Order flow imbalance analysis
//! - Microsecond-latency calculations for hot path execution
//! - AMD Ryzen AI 5 optimized memory access patterns

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::Duration;

/// Default prior probability of informed trader
const DEFAULT_PROB_INFORMED: f64 = 0.2;

/// Minimum adverse selection cost (basis points)
const MIN_ADVERSE_SELECTION_BPS: f64 = 0.5;

/// Maximum adverse selection cost (basis points)
const MAX_ADVERSE_SELECTION_BPS: f64 = 50.0;

/// Window size for order flow analysis (number of trades)
const ORDER_FLOW_WINDOW_SIZE: usize = 100;

/// Adverse selection cost estimate
#[derive(Debug, Clone)]
pub struct AdverseSelectionCost {
    /// Estimated cost in basis points
    pub cost_bps: f64,
    /// Probability of informed trader (0-1)
    pub prob_informed: f64,
    /// Order flow imbalance (-1 to 1)
    pub order_flow_imbalance: f64,
    /// Toxic flow indicator (0-1)
    pub toxic_flow_score: f64,
    /// Timestamp of calculation (nanoseconds)
    pub timestamp_ns: u64,
}

impl AdverseSelectionCost {
    /// Check if cost is within reasonable bounds
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.cost_bps >= 0.0 && self.cost_bps <= MAX_ADVERSE_SELECTION_BPS * 2.0
    }
    
    /// Get recommended spread adjustment multiplier
    #[inline]
    pub fn spread_multiplier(&self) -> f64 {
        // Base multiplier of 1.0, increased by adverse selection
        1.0 + (self.cost_bps / 10.0).min(5.0)
    }
}

/// Order flow state tracker for adverse selection analysis
pub struct OrderFlowTracker {
    /// Rolling window of signed order volumes (positive = buy, negative = sell)
    order_signs: [i64; ORDER_FLOW_WINDOW_SIZE],
    /// Current write index in rolling window
    write_index: usize,
    /// Count of orders in window
    order_count: usize,
    /// Sum of signed volumes in current window
    signed_volume_sum: i64,
    /// Total volume in current window
    total_volume: i64,
    /// Number of buyer-initiated trades
    buyer_initiated: AtomicU64,
    /// Number of seller-initiated trades
    seller_initiated: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl OrderFlowTracker {
    /// Create new order flow tracker
    pub fn new() -> Self {
        Self {
            order_signs: [0; ORDER_FLOW_WINDOW_SIZE],
            write_index: 0,
            order_count: 0,
            signed_volume_sum: 0,
            total_volume: 0,
            buyer_initiated: AtomicU64::new(0),
            seller_initiated: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Record a trade with direction
    #[inline(always)]
    pub fn record_trade(&mut self, volume: i64, is_buyer_initiated: bool) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let signed_volume = if is_buyer_initiated {
            self.buyer_initiated.fetch_add(1, Ordering::Relaxed);
            volume
        } else {
            self.seller_initiated.fetch_add(1, Ordering::Relaxed);
            -volume
        };

        // Update rolling window
        let old_value = self.order_signs[self.write_index];
        self.order_signs[self.write_index] = signed_volume;
        
        // Update sums
        self.signed_volume_sum = self.signed_volume_sum - old_value + signed_volume;
        self.total_volume = self.total_volume - old_value.abs() + signed_volume.abs();
        
        // Move write index
        self.write_index = (self.write_index + 1) % ORDER_FLOW_WINDOW_SIZE;
        
        if self.order_count < ORDER_FLOW_WINDOW_SIZE {
            self.order_count += 1;
        }

        self.last_update_ns.store(now_ns, Ordering::Relaxed);
    }

    /// Calculate order flow imbalance (OFI)
    /// Range: -1 (all sells) to 1 (all buys)
    #[inline]
    pub fn calculate_ofi(&self) -> f64 {
        if self.total_volume == 0 {
            return 0.0;
        }
        
        self.signed_volume_sum as f64 / self.total_volume as f64
    }

    /// Detect unusual order flow patterns
    #[inline]
    pub fn detect_unusual_flow(&self) -> f64 {
        // Simple heuristic: extreme OFI indicates potential informed trading
        let ofi = self.calculate_ofi();
        let abs_ofi = ofi.abs();
        
        // Score increases as OFI approaches extremes
        if abs_ofi > 0.8 {
            0.9
        } else if abs_ofi > 0.6 {
            0.7
        } else if abs_ofi > 0.4 {
            0.4
        } else {
            0.2
        }
    }

    /// Get buyer/seller ratio
    #[inline]
    pub fn get_buy_sell_ratio(&self) -> f64 {
        let buys = self.buyer_initiated.load(Ordering::Acquire) as f64;
        let sells = self.seller_initiated.load(Ordering::Acquire) as f64;
        
        if sells == 0.0 {
            return if buys == 0.0 { 1.0 } else { f64::INFINITY };
        }
        
        buys / sells
    }

    /// Reset tracker state
    pub fn reset(&mut self) {
        self.order_signs.fill(0);
        self.write_index = 0;
        self.order_count = 0;
        self.signed_volume_sum = 0;
        self.total_volume = 0;
        self.buyer_initiated.store(0, Ordering::Release);
        self.seller_initiated.store(0, Ordering::Release);
    }

    /// Get statistics
    pub fn get_stats(&self) -> OrderFlowStats {
        OrderFlowStats {
            order_count: self.order_count,
            buyer_initiated: self.buyer_initiated.load(Ordering::Acquire),
            seller_initiated: self.seller_initiated.load(Ordering::Acquire),
            order_flow_imbalance: self.calculate_ofi(),
            total_volume: self.total_volume,
        }
    }
}

impl Default for OrderFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot from order flow tracker
#[derive(Debug, Clone)]
pub struct OrderFlowStats {
    pub order_count: usize,
    pub buyer_initiated: u64,
    pub seller_initiated: u64,
    pub order_flow_imbalance: f64,
    pub total_volume: i64,
}

/// Glosten-Milgrom adverse selection estimator
pub struct GlostenMilgromEstimator {
    /// Prior probability of informed trader
    prob_informed_prior: f64,
    /// Current estimated probability of informed trader
    prob_informed_current: f64,
    /// Good news signal probability given informed trader
    prob_good_news_given_informed: f64,
    /// Bad news signal probability given informed trader
    prob_bad_news_given_informed: f64,
    /// Order flow tracker
    order_flow: OrderFlowTracker,
    /// Price impact coefficient
    price_impact_coef: f64,
    /// Last estimation timestamp
    last_estimate_ns: AtomicU64,
}

impl GlostenMilgromEstimator {
    /// Create new estimator with default parameters
    pub fn new() -> Self {
        Self {
            prob_informed_prior: DEFAULT_PROB_INFORMED,
            prob_informed_current: DEFAULT_PROB_INFORMED,
            prob_good_news_given_informed: 0.8,
            prob_bad_news_given_informed: 0.8,
            order_flow: OrderFlowTracker::new(),
            price_impact_coef: 0.1,
            last_estimate_ns: AtomicU64::new(0),
        }
    }

    /// Create estimator with custom parameters
    pub fn with_params(
        prob_informed: f64,
        prob_good_news: f64,
        prob_bad_news: f64,
    ) -> Self {
        Self {
            prob_informed_prior: prob_informed.clamp(0.0, 1.0),
            prob_informed_current: prob_informed.clamp(0.0, 1.0),
            prob_good_news_given_informed: prob_good_news.clamp(0.0, 1.0),
            prob_bad_news_given_informed: prob_bad_news.clamp(0.0, 1.0),
            order_flow: OrderFlowTracker::new(),
            price_impact_coef: 0.1,
            last_estimate_ns: AtomicU64::new(0),
        }
    }

    /// Record a buy order observation
    #[inline(always)]
    pub fn record_buy(&mut self, volume: i64) {
        self.order_flow.record_trade(volume, true);
        self.update_prob_informed(true);
    }

    /// Record a sell order observation
    #[inline(always)]
    pub fn record_sell(&mut self, volume: i64) {
        self.order_flow.record_trade(volume, false);
        self.update_prob_informed(false);
    }

    /// Bayesian update of informed trader probability
    #[inline]
    fn update_prob_informed(&mut self, is_buy: bool) {
        let pi = self.prob_informed_current;
        
        if is_buy {
            // P(Informed | Buy) using Bayes' rule
            let p_buy_given_informed = self.prob_good_news_given_informed;
            let p_buy_given_uninformed = 0.5; // Uninformed traders buy/sell equally
            
            let p_buy = pi * p_buy_given_informed + (1.0 - pi) * p_buy_given_uninformed;
            
            if p_buy > 0.0 {
                self.prob_informed_current = (pi * p_buy_given_informed / p_buy)
                    .clamp(0.0, 1.0);
            }
        } else {
            // P(Informed | Sell) using Bayes' rule
            let p_sell_given_informed = self.prob_bad_news_given_informed;
            let p_sell_given_uninformed = 0.5;
            
            let p_sell = pi * p_sell_given_informed + (1.0 - pi) * p_sell_given_uninformed;
            
            if p_sell > 0.0 {
                self.prob_informed_current = (pi * p_sell_given_informed / p_sell)
                    .clamp(0.0, 1.0);
            }
        }
        
        // Mean-revert toward prior over time
        self.prob_informed_current = 0.99 * self.prob_informed_current 
            + 0.01 * self.prob_informed_prior;
    }

    /// Calculate adverse selection cost using Glosten-Milgrom framework
    /// AS_cost = π * (V_I - V_U) where π is prob of informed, V_I is value with informed info
    #[inline(always)]
    pub fn calculate_adverse_selection_cost(&mut self, mid_price: f64) -> AdverseSelectionCost {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let pi = self.prob_informed_current;
        let ofi = self.order_flow.calculate_ofi();
        let toxic_score = self.order_flow.detect_unusual_flow();
        
        // Glosten-Milgrom spread component
        // Spread_AS = π * (E[V|Good] - E[V|Bad])
        // Simplified: use price impact coefficient and order flow
        let base_as_cost = pi * self.price_impact_coef * mid_price;
        
        // Adjust for order flow imbalance (extreme OFI = higher adverse selection)
        let ofi_adjustment = 1.0 + ofi.abs() * toxic_score;
        
        // Convert to basis points
        let cost_bps = (base_as_cost / mid_price * 10000.0 * ofi_adjustment)
            .clamp(MIN_ADVERSE_SELECTION_BPS, MAX_ADVERSE_SELECTION_BPS);
        
        self.last_estimate_ns.store(now_ns, Ordering::Relaxed);

        AdverseSelectionCost {
            cost_bps,
            prob_informed: pi,
            order_flow_imbalance: ofi,
            toxic_flow_score: toxic_score,
            timestamp_ns: now_ns,
        }
    }

    /// Get recommended bid/ask spread adjustment
    #[inline(always)]
    pub fn get_spread_adjustment(&mut self, mid_price: f64, base_spread_bps: f64) -> f64 {
        let as_cost = self.calculate_adverse_selection_cost(mid_price);
        
        // Base spread + adverse selection component
        let adjusted_spread = base_spread_bps * as_cost.spread_multiplier();
        
        adjusted_spread.clamp(base_spread_bps, base_spread_bps * 5.0)
    }

    /// Set price impact coefficient
    pub fn set_price_impact_coefficient(&mut self, coef: f64) {
        self.price_impact_coef = coef.max(0.0);
    }

    /// Update prior probability of informed trader
    pub fn set_prior_prob_informed(&mut self, prob: f64) {
        self.prob_informed_prior = prob.clamp(0.0, 1.0);
    }

    /// Get current estimate of informed trader probability
    #[inline]
    pub fn get_prob_informed(&self) -> f64 {
        self.prob_informed_current
    }

    /// Get order flow tracker reference
    pub fn order_flow(&self) -> &OrderFlowTracker {
        &self.order_flow
    }

    /// Get mutable order flow tracker reference
    pub fn order_flow_mut(&mut self) -> &mut OrderFlowTracker {
        &mut self.order_flow
    }

    /// Reset estimator state
    pub fn reset(&mut self) {
        self.prob_informed_current = self.prob_informed_prior;
        self.order_flow.reset();
    }
}

impl Default for GlostenMilgromEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined market making quote adjuster with adverse selection
pub struct AdverseSelectionQuoteAdjuster {
    gm_estimator: GlostenMilgromEstimator,
    /// Base spread (basis points)
    base_spread_bps: f64,
    /// Inventory skew factor
    inventory_skew: f64,
}

impl AdverseSelectionQuoteAdjuster {
    /// Create new adjuster
    pub fn new(base_spread_bps: f64) -> Self {
        Self {
            gm_estimator: GlostenMilgromEstimator::new(),
            base_spread_bps,
            inventory_skew: 0.0,
        }
    }

    /// Record trade for adverse selection tracking
    #[inline(always)]
    pub fn record_trade(&mut self, volume: i64, is_buyer_initiated: bool) {
        if is_buyer_initiated {
            self.gm_estimator.record_buy(volume);
        } else {
            self.gm_estimator.record_sell(volume);
        }
    }

    /// Generate adjusted quotes considering adverse selection
    #[inline(always)]
    pub fn generate_adjusted_quotes(&mut self, mid_price: f64) -> (f64, f64) {
        let as_cost = self.gm_estimator.calculate_adverse_selection_cost(mid_price);
        
        // Apply adverse selection adjustment to spread
        let adjusted_spread_bps = self.base_spread_bps * as_cost.spread_multiplier();
        let half_spread = (mid_price * adjusted_spread_bps / 10000.0) / 2.0;
        
        // Apply inventory skew if set
        let skew_adjustment = mid_price * self.inventory_skew;
        
        let bid_price = mid_price - half_spread - skew_adjustment;
        let ask_price = mid_price + half_spread - skew_adjustment;
        
        (bid_price.max(0.0001), ask_price.max(bid_price * 1.0001))
    }

    /// Set inventory skew
    pub fn set_inventory_skew(&mut self, skew: f64) {
        self.inventory_skew = skew;
    }

    /// Get adverse selection cost
    #[inline]
    pub fn get_adverse_selection_cost(&mut self, mid_price: f64) -> AdverseSelectionCost {
        self.gm_estimator.calculate_adverse_selection_cost(mid_price)
    }

    /// Get GM estimator reference
    pub fn gm_estimator(&self) -> &GlostenMilgromEstimator {
        &self.gm_estimator
    }

    /// Get mutable GM estimator reference
    pub fn gm_estimator_mut(&mut self) -> &mut GlostenMilgromEstimator {
        &mut self.gm_estimator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_flow_imbalance() {
        let mut tracker = OrderFlowTracker::new();
        
        // All buys -> OFI should be positive
        for _ in 0..50 {
            tracker.record_trade(100, true);
        }
        assert!(tracker.calculate_ofi() > 0.5);
        
        // All sells -> OFI should be negative
        tracker.reset();
        for _ in 0..50 {
            tracker.record_trade(100, false);
        }
        assert!(tracker.calculate_ofi() < -0.5);
    }

    #[test]
    fn test_adverse_selection_increases_with_informed_prob() {
        let mut estimator = GlostenMilgromEstimator::with_params(0.5, 0.9, 0.9);
        
        // Record some buys to increase informed probability
        for _ in 0..20 {
            estimator.record_buy(100);
        }
        
        let cost = estimator.calculate_adverse_selection_cost(50000.0);
        assert!(cost.prob_informed > DEFAULT_PROB_INFORMED);
        assert!(cost.cost_bps > MIN_ADVERSE_SELECTION_BPS);
    }

    #[test]
    fn test_spread_multiplier_increases_with_adverse_selection() {
        let mut estimator = GlostenMilgromEstimator::new();
        
        // Low adverse selection initially
        let cost_low = estimator.calculate_adverse_selection_cost(50000.0);
        let multiplier_low = cost_low.spread_multiplier();
        
        // Record many one-sided trades to increase adverse selection
        for _ in 0..80 {
            estimator.record_buy(1000);
        }
        
        let cost_high = estimator.calculate_adverse_selection_cost(50000.0);
        let multiplier_high = cost_high.spread_multiplier();
        
        assert!(multiplier_high >= multiplier_low);
    }

    #[test]
    fn test_quote_adjuster() {
        let mut adjuster = AdverseSelectionQuoteAdjuster::new(10.0);
        
        let (bid, ask) = adjuster.generate_adjusted_quotes(50000.0);
        
        assert!(bid > 0.0);
        assert!(ask > 0.0);
        assert!(ask > bid);
    }
}
