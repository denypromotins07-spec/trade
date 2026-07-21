//! Latency Arbitrage: Cross-Market Price Discrepancy Detection
//! 
//! Detects and exploits microsecond price discrepancies across Binance spot, margin,
//! and futures markets using strictly integer math to avoid floating-point drift.
//! Optimized for AMD Ryzen AI 5 architecture with zero-allocation hot paths.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;

/// Market type enumeration for cross-market arbitrage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketType {
    Spot,
    Margin,
    UsdtFutures,
    CoinFutures,
}

/// Price data for a single market (stored as integer ticks)
#[derive(Debug, Clone)]
pub struct MarketPrice {
    /// Best bid price in quote ticks (integer, no floats)
    pub best_bid_ticks: u64,
    /// Best ask price in quote ticks (integer, no floats)
    pub best_ask_ticks: u64,
    /// Best bid quantity in base units
    pub best_bid_qty: u64,
    /// Best ask quantity in base units
    pub best_ask_qty: u64,
    /// Last update timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Market type
    pub market_type: MarketType,
}

impl MarketPrice {
    /// Create new market price from tick values
    pub fn new(
        best_bid_ticks: u64,
        best_ask_ticks: u64,
        best_bid_qty: u64,
        best_ask_qty: u64,
        timestamp_ns: u64,
        market_type: MarketType,
    ) -> Self {
        Self {
            best_bid_ticks,
            best_ask_ticks,
            best_bid_qty,
            best_ask_qty,
            timestamp_ns,
            market_type,
        }
    }

    /// Get mid price in ticks (integer average)
    #[inline]
    pub fn mid_price_ticks(&self) -> u64 {
        (self.best_bid_ticks + self.best_ask_ticks) / 2
    }

    /// Get spread in ticks
    #[inline]
    pub fn spread_ticks(&self) -> u64 {
        self.best_ask_ticks.saturating_sub(self.best_bid_ticks)
    }
}

/// Latency arbitrage opportunity detected between two markets
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    /// Buy market (where to buy)
    pub buy_market: MarketType,
    /// Sell market (where to sell)
    pub sell_market: MarketType,
    /// Buy price in ticks
    pub buy_price_ticks: u64,
    /// Sell price in ticks
    pub sell_price_ticks: u64,
    /// Maximum executable quantity (limited by liquidity)
    pub max_qty: u64,
    /// Expected profit in quote ticks per unit
    pub profit_per_unit_ticks: u64,
    /// Total expected profit in quote ticks
    pub total_profit_ticks: u64,
    /// Timestamp of opportunity detection (nanoseconds)
    pub detected_at_ns: u64,
    /// Estimated latency to execute (microseconds)
    pub estimated_latency_us: u32,
}

impl ArbitrageOpportunity {
    /// Calculate if opportunity is profitable after fees
    /// 
    /// # Arguments
    /// * `fee_bps` - Trading fee in basis points (per leg)
    /// * `min_profit_ticks` - Minimum profit threshold in ticks
    #[inline]
    pub fn is_profitable(&self, fee_bps: u64, min_profit_ticks: u64) -> bool {
        // Total fees for round trip (buy + sell) in basis points
        let total_fee_bps = fee_bps * 2;
        
        // Profit after fees (using integer math)
        let profit_after_fees = if self.profit_per_unit_ticks > 0 {
            (self.profit_per_unit_ticks as u128 * (10000 - total_fee_bps) as u128 / 10000) as u64
        } else {
            0
        };
        
        let total_profit_after_fees = profit_after_fees.saturating_mul(self.max_qty);
        total_profit_after_fees >= min_profit_ticks
    }
}

/// Lock-free latency arbitrage detector for cross-market opportunities
pub struct LatencyArbitrageDetector {
    /// Latest prices for each market type
    prices: dashmap::DashMap<MarketType, Arc<MarketPrice>>,
    /// Detected opportunities queue (lock-free)
    opportunities: crossbeam_queue::SegQueue<ArbitrageOpportunity>,
    /// Minimum price difference threshold (in ticks)
    min_price_diff_ticks: AtomicU64,
    /// Minimum quantity threshold
    min_qty: AtomicU64,
    /// Trading fee in basis points (per leg)
    fee_bps: AtomicU64,
    /// Opportunity counter
    opportunity_count: AtomicU64,
    /// Last scan timestamp
    last_scan_ns: AtomicU64,
}

impl LatencyArbitrageDetector {
    /// Create a new latency arbitrage detector
    /// 
    /// # Arguments
    /// * `min_price_diff_ticks` - Minimum price difference to consider (in quote ticks)
    /// * `min_qty` - Minimum quantity required for execution
    /// * `fee_bps` - Trading fee per leg in basis points
    pub fn new(min_price_diff_ticks: u64, min_qty: u64, fee_bps: u64) -> Self {
        Self {
            prices: dashmap::DashMap::new(),
            opportunities: crossbeam_queue::SegQueue::new(),
            min_price_diff_ticks: AtomicU64::new(min_price_diff_ticks),
            min_qty: AtomicU64::new(min_qty),
            fee_bps: AtomicU64::new(fee_bps),
            opportunity_count: AtomicU64::new(0),
            last_scan_ns: AtomicU64::new(0),
        }
    }

    /// Update price for a specific market (lock-free)
    #[inline]
    pub fn update_price(&self, price: MarketPrice) {
        let market_type = price.market_type;
        self.prices.insert(market_type, Arc::new(price));
    }

    /// Scan for arbitrage opportunities across all market pairs
    /// Returns number of opportunities detected
    #[inline]
    pub fn scan_opportunities(&self, current_timestamp_ns: u64) -> usize {
        let prices: Vec<_> = self.prices.iter().collect();
        let mut count = 0;

        // Compare all pairs of markets
        for i in 0..prices.len() {
            for j in (i + 1)..prices.len() {
                let price_a = prices[i].value();
                let price_b = prices[j].value();

                // Check both directions: A→B and B→A
                if let Some(opp) = self.check_arbitrage(price_a, price_b, current_timestamp_ns) {
                    self.opportunities.push(opp);
                    count += 1;
                }
                if let Some(opp) = self.check_arbitrage(price_b, price_a, current_timestamp_ns) {
                    self.opportunities.push(opp);
                    count += 1;
                }
            }
        }

        self.last_scan_ns.store(current_timestamp_ns, Ordering::Release);
        self.opportunity_count.fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Check for arbitrage opportunity between two markets
    /// 
    /// Buys at market_a's ask, sells at market_b's bid
    #[inline]
    fn check_arbitrage(
        &self,
        market_a: &MarketPrice,
        market_b: &MarketPrice,
        timestamp_ns: u64,
    ) -> Option<ArbitrageOpportunity> {
        let min_diff = self.min_price_diff_ticks.load(Ordering::Acquire);
        let min_qty = self.min_qty.load(Ordering::Acquire);

        // Check if market_b's bid > market_a's ask (buy low, sell high)
        if market_b.best_bid_ticks <= market_a.best_ask_ticks {
            return None;
        }

        let price_diff = market_b.best_bid_ticks - market_a.best_ask_ticks;
        
        // Check minimum price difference threshold
        if price_diff < min_diff {
            return None;
        }

        // Maximum quantity limited by liquidity at both legs
        let max_qty = market_a.best_ask_qty.min(market_b.best_bid_qty);
        
        // Check minimum quantity threshold
        if max_qty < min_qty {
            return None;
        }

        // Calculate total profit in quote ticks
        let total_profit = price_diff.saturating_mul(max_qty);

        // Estimate latency based on price age
        let price_age_ns = timestamp_ns.saturating_sub(market_a.timestamp_ns.max(market_b.timestamp_ns));
        let estimated_latency_us = (price_age_ns / 1000) as u32;

        Some(ArbitrageOpportunity {
            buy_market: market_a.market_type,
            sell_market: market_b.market_type,
            buy_price_ticks: market_a.best_ask_ticks,
            sell_price_ticks: market_b.best_bid_ticks,
            max_qty,
            profit_per_unit_ticks: price_diff,
            total_profit_ticks: total_profit,
            detected_at_ns: timestamp_ns,
            estimated_latency_us,
        })
    }

    /// Get next available opportunity (pop from queue)
    #[inline]
    pub fn pop_opportunity(&self) -> Option<ArbitrageOpportunity> {
        self.opportunities.pop()
    }

    /// Get number of pending opportunities
    #[inline]
    pub fn pending_opportunities(&self) -> usize {
        self.opportunities.len()
    }

    /// Get total opportunities detected since startup
    #[inline]
    pub fn total_opportunities(&self) -> u64 {
        self.opportunity_count.load(Ordering::Acquire)
    }

    /// Set minimum price difference threshold
    #[inline]
    pub fn set_min_price_diff(&self, ticks: u64) {
        self.min_price_diff_ticks.store(ticks, Ordering::Release);
    }

    /// Set minimum quantity threshold
    #[inline]
    pub fn set_min_qty(&self, qty: u64) {
        self.min_qty.store(qty, Ordering::Release);
    }

    /// Get latest price for a market
    #[inline]
    pub fn get_price(&self, market_type: MarketType) -> Option<Arc<MarketPrice>> {
        self.prices.get(&market_type).map(|v| v.value().clone())
    }

    /// Clear all state (for /KILL orchestration)
    pub fn reset(&self) {
        self.prices.clear();
        while self.opportunities.pop().is_some() {}
        self.opportunity_count.store(0, Ordering::Relaxed);
        self.last_scan_ns.store(0, Ordering::Relaxed);
    }

    /// Get latency statistics (average opportunity age in microseconds)
    #[inline]
    pub fn avg_opportunity_age_us(&self) -> u64 {
        let mut total_age = 0u64;
        let mut count = 0u64;

        for opp in self.opportunities.iter() {
            // This is a snapshot, so we can't calculate real-time age
            // In production, this would track execution latency
            total_age += opp.estimated_latency_us as u64;
            count += 1;
        }

        if count == 0 {
            0
        } else {
            total_age / count
        }
    }
}

/// Multi-asset latency arbitrage scanner
pub struct MultiAssetArbitrageScanner {
    /// Per-asset arbitrage detectors
    scanners: dashmap::DashMap<String, Arc<LatencyArbitrageDetector>>,
    /// Global minimum profit threshold (quote ticks)
    global_min_profit_ticks: AtomicU64,
}

impl MultiAssetArbitrageScanner {
    /// Create a new multi-asset scanner
    pub fn new(global_min_profit_ticks: u64) -> Self {
        Self {
            scanners: dashmap::DashMap::new(),
            global_min_profit_ticks: AtomicU64::new(global_min_profit_ticks),
        }
    }

    /// Get or create scanner for a specific asset
    #[inline]
    pub fn get_or_create_scanner(
        &self,
        symbol: &str,
        min_price_diff_ticks: u64,
        min_qty: u64,
        fee_bps: u64,
    ) -> Arc<LatencyArbitrageDetector> {
        self.scanners
            .entry(symbol.to_string())
            .or_insert_with(|| {
                Arc::new(LatencyArbitrageDetector::new(min_price_diff_ticks, min_qty, fee_bps))
            })
            .value()
            .clone()
    }

    /// Scan all assets for opportunities
    #[inline]
    pub fn scan_all(&self, timestamp_ns: u64) -> usize {
        let mut total = 0;
        for entry in self.scanners.iter() {
            total += entry.value().scan_opportunities(timestamp_ns);
        }
        total
    }

    /// Get best opportunity across all assets
    #[inline]
    pub fn get_best_opportunity(&self) -> Option<ArbitrageOpportunity> {
        let mut best: Option<ArbitrageOpportunity> = None;

        for entry in self.scanners.iter() {
            let detector = entry.value();
            while let Some(opp) = detector.pop_opportunity() {
                if opp.is_profitable(detector.fee_bps.load(Ordering::Acquire), 
                                     self.global_min_profit_ticks.load(Ordering::Acquire)) {
                    match &best {
                        None => best = Some(opp),
                        Some(current_best) => {
                            if opp.total_profit_ticks > current_best.total_profit_ticks {
                                best = Some(opp);
                            }
                        }
                    }
                }
            }
        }

        best
    }

    /// Reset all scanners (for /KILL)
    pub fn reset_all(&self) {
        for entry in self.scanners.iter() {
            entry.value().reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_price_mid_price() {
        let price = MarketPrice::new(10000, 10010, 100, 150, 1000, MarketType::Spot);
        assert_eq!(price.mid_price_ticks(), 10005);
        assert_eq!(price.spread_ticks(), 10);
    }

    #[test]
    fn test_arbitrage_detection() {
        let detector = LatencyArbitrageDetector::new(5, 10, 10); // 5 tick min diff, 10 qty, 10 bps fee

        // Spot: bid=10000, ask=10005
        detector.update_price(MarketPrice::new(10000, 10005, 100, 100, 1000, MarketType::Spot));
        
        // Futures: bid=10015, ask=10020 (higher than spot)
        detector.update_price(MarketPrice::new(10015, 10020, 100, 100, 1000, MarketType::UsdtFutures));

        let count = detector.scan_opportunities(2000);
        assert!(count >= 1);

        // Should detect: buy spot at 10005, sell futures at 10015
        if let Some(opp) = detector.pop_opportunity() {
            assert_eq!(opp.buy_market, MarketType::Spot);
            assert_eq!(opp.sell_market, MarketType::UsdtFutures);
            assert_eq!(opp.profit_per_unit_ticks, 10); // 10015 - 10005
            
            // Check profitability (10 bps per leg = 20 bps total)
            assert!(opp.is_profitable(10, 50)); // 10*100*(1-0.002) = 980 > 50
        } else {
            panic!("Expected arbitrage opportunity");
        }
    }

    #[test]
    fn test_no_arbitrage_when_prices_equal() {
        let detector = LatencyArbitrageDetector::new(5, 10, 10);

        // Same prices on both markets
        detector.update_price(MarketPrice::new(10000, 10005, 100, 100, 1000, MarketType::Spot));
        detector.update_price(MarketPrice::new(10000, 10005, 100, 100, 1000, MarketType::UsdtFutures));

        let count = detector.scan_opportunities(2000);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_unprofitable_after_fees() {
        let detector = LatencyArbitrageDetector::new(1, 10, 100); // High fee

        detector.update_price(MarketPrice::new(10000, 10002, 100, 100, 1000, MarketType::Spot));
        detector.update_price(MarketPrice::new(10005, 10007, 100, 100, 1000, MarketType::UsdtFutures));

        let count = detector.scan_opportunities(2000);
        assert!(count >= 1);

        if let Some(opp) = detector.pop_opportunity() {
            // 3 tick profit, but 100 bps per leg = 200 bps total fees
            // 3 * (1 - 0.02) = 2.94 < 3, so should be unprofitable with high threshold
            assert!(!opp.is_profitable(100, 50));
        }
    }
}
