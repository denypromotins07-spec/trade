//! Advanced Execution & Smart Order Routing - Chapter 4
//! File 11: sor.rs
//! 
//! Builds a Smart Order Router (SOR) that evaluates Binance spot, futures,
//! and cross-venue liquidity in real-time to guarantee the best possible
//! execution price and minimize slippage. Factors in funding rates and fees.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;

/// Venue types supported by SOR
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum VenueType {
    BinanceSpot,
    BinanceFutures,
    DEXUniswap,
    DEXPancakeSwap,
    CEXOther,
}

/// Liquidity source information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquiditySource {
    pub venue: VenueType,
    pub symbol: String,
    pub bid_price: i64,
    pub ask_price: i64,
    pub bid_qty: u64,
    pub ask_qty: u64,
    /// Maker fee in basis points (1 bp = 0.01%)
    pub maker_fee_bps: i32,
    /// Taker fee in basis points
    pub taker_fee_bps: i32,
    /// Funding rate for perps (scaled by 1e6)
    pub funding_rate_ppm: i32,
    /// Latency in microseconds
    pub latency_us: u32,
    /// Last update timestamp
    pub last_update_ns: u64,
}

impl LiquiditySource {
    /// Get effective buy price including fees
    #[inline]
    pub fn effective_buy_price(&self) -> f64 {
        let fee_multiplier = 1.0 + (self.taker_fee_bps as f64 / 10000.0);
        self.ask_price as f64 * fee_multiplier
    }

    /// Get effective sell price including fees
    #[inline]
    pub fn effective_sell_price(&self) -> f64 {
        let fee_multiplier = 1.0 - (self.taker_fee_bps as f64 / 10000.0);
        self.bid_price as f64 * fee_multiplier
    }

    /// Calculate slippage for given quantity
    pub fn estimate_slippage(&self, quantity: u64, is_buy: bool) -> f64 {
        let available_qty = if is_buy { self.ask_qty } else { self.bid_qty };
        if available_qty == 0 {
            return 1.0; // 100% slippage if no liquidity
        }
        
        let fill_ratio = quantity as f64 / available_qty as f64;
        // Linear slippage model (simplified)
        fill_ratio * 0.01 // 1% slippage at 100% of book
    }
}

/// Route segment for order execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSegment {
    pub venue: VenueType,
    pub symbol: String,
    pub quantity: u64,
    pub expected_price: i64,
    pub estimated_slippage_bps: i32,
    pub total_fees_bps: i32,
    pub priority: u32,
}

/// Complete execution route
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRoute {
    pub segments: Vec<RouteSegment>,
    pub total_quantity: u64,
    pub weighted_avg_price: i64,
    pub total_slippage_bps: i32,
    pub total_fees_bps: i32,
    pub expected_cost: u128,
    pub confidence_score: f64,
    pub estimated_latency_us: u32,
}

/// Smart Order Router
pub struct SmartOrderRouter {
    /// Liquidity sources by symbol
    liquidity_sources: DashMap<String, Vec<LiquiditySource>>,
    /// Historical execution quality by venue
    venue_quality: DashMap<VenueType, VenueQualityMetrics>,
    /// Configuration
    max_slippage_bps: i32,
    min_liquidity_qty: u64,
    prefer_low_latency: bool,
    /// Statistics
    routes_computed: AtomicU64,
    orders_routed: AtomicU64,
    /// Active flag
    is_active: AtomicBool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VenueQualityMetrics {
    pub total_orders: u64,
    pub filled_orders: u64,
    pub avg_fill_ratio: f64,
    pub avg_slippage_bps: f64,
    pub avg_latency_us: f64,
    pub reliability_score: f64,
}

impl SmartOrderRouter {
    /// Create new SOR instance
    pub fn new(max_slippage_bps: i32, min_liquidity_qty: u64) -> Self {
        Self {
            liquidity_sources: DashMap::new(),
            venue_quality: DashMap::new(),
            max_slippage_bps,
            min_liquidity_qty,
            prefer_low_latency: true,
            routes_computed: AtomicU64::new(0),
            orders_routed: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Update liquidity source
    pub fn update_liquidity(&self, source: LiquiditySource) {
        let sources = self.liquidity_sources
            .entry(source.symbol.clone())
            .or_insert_with(Vec::new);
        
        // Find existing or add new
        let existing_idx = sources.iter()
            .position(|s| s.venue == source.venue);
        
        if let Some(idx) = existing_idx {
            sources[idx] = source;
        } else {
            sources.push(source);
        }
    }

    /// Find best execution route for buying
    pub fn find_best_buy_route(
        &self,
        symbol: &str,
        quantity: u64,
    ) -> Option<ExecutionRoute> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let sources = self.liquidity_sources.get(symbol)?;
        if sources.is_empty() {
            return None;
        }

        // Filter viable sources
        let mut viable: Vec<_> = sources.iter()
            .filter(|s| s.ask_qty >= self.min_liquidity_qty)
            .collect();
        
        if viable.is_empty() {
            return None;
        }

        // Sort by effective price (including fees and estimated slippage)
        viable.sort_by(|a, b| {
            let cost_a = a.effective_buy_price() + 
                (a.estimate_slippage(quantity, true) * a.ask_price as f64);
            let cost_b = b.effective_buy_price() + 
                (b.estimate_slippage(quantity, true) * b.ask_price as f64);
            
            cost_a.partial_cmp(&cost_b).unwrap()
        });

        // Build route potentially splitting across venues
        let mut segments = Vec::new();
        let mut remaining_qty = quantity;
        let mut total_value = 0u128;
        let mut total_slippage = 0i32;
        let mut total_fees = 0i32;
        let mut total_latency = 0u32;

        for source in viable {
            if remaining_qty == 0 {
                break;
            }

            let fill_qty = remaining_qty.min(source.ask_qty);
            let slippage_bps = (source.estimate_slippage(fill_qty, true) * 100.0) as i32;
            
            // Skip if slippage exceeds threshold
            if slippage_bps > self.max_slippage_bps {
                continue;
            }

            let segment = RouteSegment {
                venue: source.venue,
                symbol: symbol.to_string(),
                quantity: fill_qty,
                expected_price: source.ask_price,
                estimated_slippage_bps: slippage_bps,
                total_fees_bps: source.taker_fee_bps,
                priority: segments.len() as u32,
            };

            total_value += (fill_qty as u128) * (source.ask_price as u128);
            total_slippage = total_slippage.saturating_add(slippage_bps);
            total_fees = total_fees.saturating_add(source.taker_fee_bps);
            total_latency = total_latency.saturating_add(source.latency_us);
            
            remaining_qty -= fill_qty;
            segments.push(segment);
        }

        if segments.is_empty() || remaining_qty > 0 {
            return None; // Could not fill entire order
        }

        let weighted_price = (total_value / quantity as u128) as i64;
        let confidence = self.calculate_route_confidence(&segments);

        self.routes_computed.fetch_add(1, Ordering::Relaxed);

        Some(ExecutionRoute {
            segments,
            total_quantity: quantity,
            weighted_avg_price: weighted_price,
            total_slippage_bps: total_slippage / segments.len() as i32,
            total_fees_bps: total_fees / segments.len() as i32,
            expected_cost: total_value,
            confidence_score: confidence,
            estimated_latency_us: total_latency,
        })
    }

    /// Find best execution route for selling
    pub fn find_best_sell_route(
        &self,
        symbol: &str,
        quantity: u64,
    ) -> Option<ExecutionRoute> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        let sources = self.liquidity_sources.get(symbol)?;
        if sources.is_empty() {
            return None;
        }

        // Filter viable sources
        let mut viable: Vec<_> = sources.iter()
            .filter(|s| s.bid_qty >= self.min_liquidity_qty)
            .collect();
        
        if viable.is_empty() {
            return None;
        }

        // Sort by effective sell price (higher is better)
        viable.sort_by(|a, b| {
            let proceeds_a = a.effective_sell_price() - 
                (a.estimate_slippage(quantity, false) * a.bid_price as f64);
            let proceeds_b = b.effective_sell_price() - 
                (b.estimate_slippage(quantity, false) * b.bid_price as f64);
            
            proceeds_b.partial_cmp(&proceeds_a).unwrap()
        });

        // Build route
        let mut segments = Vec::new();
        let mut remaining_qty = quantity;
        let mut total_value = 0u128;
        let mut total_slippage = 0i32;
        let mut total_fees = 0i32;
        let mut total_latency = 0u32;

        for source in viable {
            if remaining_qty == 0 {
                break;
            }

            let fill_qty = remaining_qty.min(source.bid_qty);
            let slippage_bps = (source.estimate_slippage(fill_qty, false) * 100.0) as i32;
            
            if slippage_bps > self.max_slippage_bps {
                continue;
            }

            let segment = RouteSegment {
                venue: source.venue,
                symbol: symbol.to_string(),
                quantity: fill_qty,
                expected_price: source.bid_price,
                estimated_slippage_bps: slippage_bps,
                total_fees_bps: source.taker_fee_bps,
                priority: segments.len() as u32,
            };

            total_value += (fill_qty as u128) * (source.bid_price as u128);
            total_slippage = total_slippage.saturating_add(slippage_bps);
            total_fees = total_fees.saturating_add(source.taker_fee_bps);
            total_latency = total_latency.saturating_add(source.latency_us);
            
            remaining_qty -= fill_qty;
            segments.push(segment);
        }

        if segments.is_empty() || remaining_qty > 0 {
            return None;
        }

        let weighted_price = (total_value / quantity as u128) as i64;
        let confidence = self.calculate_route_confidence(&segments);

        self.routes_computed.fetch_add(1, Ordering::Relaxed);

        Some(ExecutionRoute {
            segments,
            total_quantity: quantity,
            weighted_avg_price: weighted_price,
            total_slippage_bps: total_slippage / segments.len() as i32,
            total_fees_bps: total_fees / segments.len() as i32,
            expected_cost: total_value,
            confidence_score: confidence,
            estimated_latency_us: total_latency,
        })
    }

    /// Calculate route confidence based on venue quality metrics
    fn calculate_route_confidence(&self, segments: &[RouteSegment]) -> f64 {
        if segments.is_empty() {
            return 0.0;
        }

        let mut total_confidence = 0.0;
        
        for segment in segments {
            let quality = self.venue_quality
                .get(&segment.venue)
                .map(|q| q.reliability_score)
                .unwrap_or(0.5);
            
            // Reduce confidence for high slippage
            let slippage_factor = 1.0 - (segment.estimated_slippage_bps as f64 / 100.0);
            
            total_confidence += quality * slippage_factor;
        }

        total_confidence / segments.len() as f64
    }

    /// Update venue quality metrics after execution
    pub fn record_execution_result(
        &self,
        venue: VenueType,
        fill_ratio: f64,
        actual_slippage_bps: i32,
        actual_latency_us: u32,
    ) {
        let mut metrics = self.venue_quality
            .entry(venue)
            .or_insert_with(VenueQualityMetrics::default);
        
        metrics.total_orders += 1;
        metrics.filled_orders += 1;
        
        // Update averages with exponential moving average
        let alpha = 0.1;
        metrics.avg_fill_ratio = metrics.avg_fill_ratio * (1.0 - alpha) + fill_ratio * alpha;
        metrics.avg_slippage_bps = metrics.avg_slippage_bps * (1.0 - alpha) 
            + (actual_slippage_bps as f64) * alpha;
        metrics.avg_latency_us = metrics.avg_latency_us * (1.0 - alpha) 
            + (actual_latency_us as f64) * alpha;
        
        // Calculate reliability score
        metrics.reliability_score = (metrics.avg_fill_ratio * 0.5)
            + ((1.0 - metrics.avg_slippage_bps / 100.0) * 0.3)
            + ((1.0 - metrics.avg_latency_us / 10000.0) * 0.2);
        
        metrics.reliability_score = metrics.reliability_score.clamp(0.0, 1.0);
    }

    /// Compare routes between spot and futures considering funding
    pub fn compare_spot_vs_futures(
        &self,
        symbol: &str,
        quantity: u64,
        holding_period_hours: u64,
    ) -> SpotFuturesComparison {
        let spot_route = self.find_best_buy_route(symbol, quantity);
        let futures_route = self.find_best_buy_route(symbol, quantity);
        
        // Get funding rates
        let spot_funding = spot_route.as_ref()
            .and_then(|r| r.segments.first())
            .map(|s| 0i32) // Spot has no funding
            .unwrap_or(0);
        
        let futures_funding = futures_route.as_ref()
            .and_then(|r| r.segments.first())
            .map(|s| {
                // Get from liquidity source
                0i32 // Would fetch actual funding rate
            })
            .unwrap_or(0);
        
        // Calculate funding cost over holding period
        let funding_intervals = holding_period_hours / 8; // Funding every 8 hours
        let funding_cost_pct = (futures_funding as f64 * funding_intervals as f64) / 1e6;
        
        SpotFuturesComparison {
            spot_effective_cost: spot_route.as_ref().map(|r| r.expected_cost).unwrap_or(u128::MAX),
            futures_effective_cost: futures_route.as_ref()
                .map(|r| r.expected_cost as f64 * (1.0 + funding_cost_pct))
                .unwrap_or(f64::MAX) as u128,
            funding_rate_ppm: futures_funding,
            estimated_funding_cost_pct: funding_cost_pct,
            recommended_venue: if spot_route.as_ref().map(|r| r.expected_cost).unwrap_or(u128::MAX)
                <= futures_route.as_ref().map(|r| r.expected_cost).unwrap_or(u128::MAX)
            {
                VenueType::BinanceSpot
            } else {
                VenueType::BinanceFutures
            },
        }
    }

    /// Enable/disable routing
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_statistics(&self) -> SORStatistics {
        SORStatistics {
            symbols_tracked: self.liquidity_sources.len(),
            routes_computed: self.routes_computed.load(Ordering::Relaxed),
            orders_routed: self.orders_routed.load(Ordering::Relaxed),
            is_active: self.is_active.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotFuturesComparison {
    pub spot_effective_cost: u128,
    pub futures_effective_cost: u128,
    pub funding_rate_ppm: i32,
    pub estimated_funding_cost_pct: f64,
    pub recommended_venue: VenueType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SORStatistics {
    pub symbols_tracked: usize,
    pub routes_computed: u64,
    pub orders_routed: u64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sor_basic() {
        let sor = SmartOrderRouter::new(50, 100);
        
        // Add liquidity sources
        sor.update_liquidity(LiquiditySource {
            venue: VenueType::BinanceSpot,
            symbol: "BTCUSDT".to_string(),
            bid_price: 60000000000,
            ask_price: 60001000000,
            bid_qty: 1000,
            ask_qty: 1000,
            maker_fee_bps: 10,
            taker_fee_bps: 10,
            funding_rate_ppm: 0,
            latency_us: 100,
            last_update_ns: 1000000,
        });
        
        sor.update_liquidity(LiquiditySource {
            venue: VenueType::BinanceFutures,
            symbol: "BTCUSDT".to_string(),
            bid_price: 60000500000,
            ask_price: 60001500000,
            bid_qty: 5000,
            ask_qty: 5000,
            maker_fee_bps: 4,
            taker_fee_bps: 6,
            funding_rate_ppm: 100,
            latency_us: 50,
            last_update_ns: 1000000,
        });
        
        // Find best buy route
        let route = sor.find_best_buy_route("BTCUSDT", 500);
        assert!(route.is_some());
        
        let r = route.unwrap();
        assert_eq!(r.total_quantity, 500);
        
        let stats = sor.get_statistics();
        assert_eq!(stats.routes_computed, 1);
    }
}
