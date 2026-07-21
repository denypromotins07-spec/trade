//! `src/market/microstructure.rs`
//! 
//! **Market Microstructure Analysis Engine**
//! 
//! Analyzes bid-ask spread dynamics, tick size impacts, and market depth
//! to identify optimal limit order placement offsets.
//! 
//! **Features:**
//! - Spread analysis and regime detection
//! - Order book imbalance calculations
//! - Optimal limit price offset suggestions
//! - Tick size constraint handling

use crate::data::orderbook::OrderBook;

/// Market microstructure metrics
#[derive(Debug, Clone)]
pub struct MicrostructureMetrics {
    pub spread_bps: f64,           // Spread in basis points
    pub mid_price: f64,
    pub book_imbalance: f64,       // -1.0 (all asks) to 1.0 (all bids)
    pub depth_ratio: f64,          // Bid depth / Ask depth at top levels
    pub effective_spread: f64,     // Realized spread from recent trades
    pub price_impact_coefficient: f64,
}

/// Optimal order placement suggestion
#[derive(Debug, Clone)]
pub struct PlacementSuggestion {
    pub suggested_price: f64,
    pub confidence: f64,
    pub expected_fill_probability: f64,
    pub adverse_selection_risk: f64,
}

/// Market microstructure analyzer
pub struct MicrostructureAnalyzer {
    tick_size: f64,
    lot_size: f64,
    spread_threshold_high: f64,
    spread_threshold_low: f64,
}

impl MicrostructureAnalyzer {
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        Self {
            tick_size,
            lot_size,
            spread_threshold_high: 0.001,  // 10 bps
            spread_threshold_low: 0.0001,  // 1 bps
        }
    }

    /// Analyzes current market microstructure
    #[inline]
    pub fn analyze(&self, book: &OrderBook) -> Option<MicrostructureMetrics> {
        let best_bid = book.get_best_bid()?;
        let best_ask = book.get_best_ask()?;
        
        let mid_price = (best_bid.price + best_ask.price) / 2.0;
        let spread = best_ask.price - best_bid.price;
        let spread_bps = (spread / mid_price) * 10000.0;

        // Calculate book imbalance using top 5 levels
        let bid_depth = book.get_cumulative_bid_volume(5);
        let ask_depth = book.get_cumulative_ask_volume(5);
        
        let total_depth = bid_depth + ask_depth;
        let book_imbalance = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };

        let depth_ratio = if ask_depth > 0.0 {
            bid_depth / ask_depth
        } else {
            f64::INFINITY
        };

        // Estimate price impact coefficient (simplified Kyle's lambda)
        let price_impact_coefficient = spread / (bid_depth + ask_depth).max(0.0001);

        Some(MicrostructureMetrics {
            spread_bps,
            mid_price,
            book_imbalance,
            depth_ratio,
            effective_spread: spread, // Would be calculated from trades in production
            price_impact_coefficient,
        })
    }

    /// Suggests optimal limit order placement
    #[inline]
    pub fn suggest_placement(
        &self,
        book: &OrderBook,
        side: crate::market::matching::Side,
        quantity: f64,
    ) -> Option<PlacementSuggestion> {
        let metrics = self.analyze(book)?;
        let best_bid = book.get_best_bid()?;
        let best_ask = book.get_best_ask()?;

        match side {
            crate::market::matching::Side::Buy => {
                // If book is imbalanced towards asks (negative), we might want to be more aggressive
                let aggressiveness = (-metrics.book_imbalance * 0.5).clamp(-0.3, 0.3);
                
                // Calculate suggested offset from mid
                let offset = spread_for_side(true, metrics.spread_bps, aggressiveness);
                let suggested_price = round_to_tick(metrics.mid_price - offset, self.tick_size);
                
                // Estimate fill probability based on queue position and spread
                let fill_prob = estimate_fill_probability(suggested_price, best_bid.price, metrics.spread_bps);
                
                // Adverse selection risk increases when we're too passive in trending markets
                let adverse_risk = calculate_adverse_selection(metrics.book_imbalance, metrics.spread_bps);

                Some(PlacementSuggestion {
                    suggested_price,
                    confidence: 0.7, // Would be ML-derived in production
                    expected_fill_probability: fill_prob,
                    adverse_selection_risk: adverse_risk,
                })
            }
            crate::market::matching::Side::Sell => {
                let aggressiveness = (metrics.book_imbalance * 0.5).clamp(-0.3, 0.3);
                let offset = spread_for_side(false, metrics.spread_bps, aggressiveness);
                let suggested_price = round_to_tick(metrics.mid_price + offset, self.tick_size);
                
                let fill_prob = estimate_fill_probability(suggested_price, best_ask.price, metrics.spread_bps);
                let adverse_risk = calculate_adverse_selection(-metrics.book_imbalance, metrics.spread_bps);

                Some(PlacementSuggestion {
                    suggested_price,
                    confidence: 0.7,
                    expected_fill_probability: fill_prob,
                    adverse_selection_risk: adverse_risk,
                })
            }
        }
    }

    /// Returns whether the spread is wide enough to justify market making
    #[inline]
    pub fn is_spread_favorable(&self, spread_bps: f64) -> bool {
        spread_bps > self.spread_threshold_low * 10000.0
    }

    /// Checks if a price respects tick size constraints
    #[inline]
    pub fn is_valid_price(&self, price: f64) -> bool {
        (price / self.tick_size).fract() < f64::EPSILON
    }
}

#[inline]
fn spread_for_side(is_buy: bool, spread_bps: f64, aggressiveness: f64) -> f64 {
    let half_spread = (spread_bps / 10000.0) / 2.0;
    if is_buy {
        half_spread * (1.0 - aggressiveness)
    } else {
        half_spread * (1.0 + aggressiveness)
    }
}

#[inline]
fn round_to_tick(price: f64, tick_size: f64) -> f64 {
    (price / tick_size).round() * tick_size
}

#[inline]
fn estimate_fill_probability(order_price: f64, best_price: f64, spread_bps: f64) -> f64 {
    // Simplified model: closer to best price = higher fill probability
    let distance = (order_price - best_price).abs();
    let half_spread = (spread_bps / 10000.0) / 2.0 * best_price;
    
    if distance <= f64::EPSILON {
        0.9 // At best price
    } else if distance >= half_spread * 2.0 {
        0.1 // Far from spread
    } else {
        1.0 - (distance / (half_spread * 2.0))
    }
}

#[inline]
fn calculate_adverse_selection(imbalance: f64, spread_bps: f64) -> f64 {
    // Higher adverse selection when:
    // 1. Strong imbalance (one-sided flow)
    // 2. Wide spreads (uncertainty)
    imbalance.abs() * 0.5 + (spread_bps / 10000.0) * 0.3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_rounding() {
        assert_eq!(round_to_tick(100.123, 0.01), 100.12);
        assert_eq!(round_to_tick(100.127, 0.01), 100.13);
    }
}
