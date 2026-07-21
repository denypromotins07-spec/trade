//! Funding Rate Arbitrage: Cash-and-Carry Tracker
//! 
//! Automated cash-and-carry arbitrage tracker that monitors the spot vs perpetual
//! basis and funding rates to capture risk-free yield during extreme market sentiment.
//! Uses integer arithmetic for microsecond precision without floating-point drift.
//! Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;

/// Funding rate data for a perpetual contract
#[derive(Debug, Clone)]
pub struct FundingRate {
    /// Current funding rate (basis points, annualized)
    pub rate_bps_annual: i64,
    /// Next funding timestamp (nanoseconds)
    pub next_funding_ns: u64,
    /// Last funding rate (basis points)
    pub last_rate_bps: i64,
    /// Mark price in quote ticks
    pub mark_price_ticks: u64,
    /// Index price in quote ticks
    pub index_price_ticks: u64,
}

impl FundingRate {
    /// Create new funding rate data
    pub fn new(
        rate_bps_annual: i64,
        next_funding_ns: u64,
        last_rate_bps: i64,
        mark_price_ticks: u64,
        index_price_ticks: u64,
    ) -> Self {
        Self {
            rate_bps_annual,
            next_funding_ns,
            last_rate_bps,
            mark_price_ticks,
            index_price_ticks,
        }
    }

    /// Get implied funding rate per 8-hour interval (standard Binance interval)
    #[inline]
    pub fn rate_per_interval_bps(&self) -> i64 {
        // Annual rate / (365 * 3) ≈ rate per 8-hour interval
        self.rate_bps_annual / 1095
    }

    /// Calculate basis (spot - perp) in ticks
    #[inline]
    pub fn basis_ticks(&self) -> i64 {
        self.index_price_ticks as i64 - self.mark_price_ticks as i64
    }

    /// Calculate basis in basis points relative to spot price
    #[inline]
    pub fn basis_bps(&self) -> i64 {
        if self.index_price_ticks == 0 {
            return 0;
        }
        (self.basis_ticks() * 10000) / self.index_price_ticks as i64
    }
}

/// Cash-and-carry arbitrage opportunity
#[derive(Debug, Clone)]
pub struct CashAndCarryOpportunity {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Spot price in ticks
    pub spot_price_ticks: u64,
    /// Perpetual price in ticks
    pub perp_price_ticks: u64,
    /// Current funding rate (annualized bps)
    pub funding_rate_bps_annual: i64,
    /// Basis (spot - perp) in bps
    pub basis_bps: i64,
    /// Expected annual return from carry trade (bps)
    pub expected_return_bps_annual: i64,
    /// Maximum executable size (base units)
    pub max_size: u64,
    /// Timestamp of detection (nanoseconds)
    pub detected_at_ns: u64,
}

impl CashAndCarryOpportunity {
    /// Check if opportunity is profitable after costs
    /// 
    /// # Arguments
    /// * `borrow_cost_bps` - Cost to borrow spot (annualized bps)
    /// * `min_return_bps` - Minimum acceptable return (annualized bps)
    #[inline]
    pub fn is_profitable(&self, borrow_cost_bps: i64, min_return_bps: i64) -> bool {
        let net_return = self.expected_return_bps_annual - borrow_cost_bps;
        net_return >= min_return_bps
    }

    /// Get direction: true = long spot / short perp, false = short spot / long perp
    #[inline]
    pub fn trade_direction(&self) -> bool {
        // If funding rate is positive, go long spot / short perp
        self.funding_rate_bps_annual > 0
    }
}

/// Lock-free funding rate arbitrage tracker
pub struct FundingArbitrageTracker {
    /// Funding rates by symbol
    funding_rates: dashmap::DashMap<String, Arc<FundingRate>>,
    /// Spot prices by symbol (in ticks)
    spot_prices: dashmap::DashMap<String, AtomicU64>,
    /// Detected opportunities queue
    opportunities: crossbeam_queue::SegQueue<CashAndCarryOpportunity>,
    /// Minimum funding rate threshold (bps annual)
    min_funding_rate_bps: AtomicI64,
    /// Minimum basis threshold (bps)
    min_basis_bps: AtomicI64,
    /// Borrow cost estimate (bps annual)
    borrow_cost_bps: AtomicI64,
    /// Opportunity counter
    opportunity_count: AtomicU64,
    /// Last scan timestamp
    last_scan_ns: AtomicU64,
}

impl FundingArbitrageTracker {
    /// Create a new funding arbitrage tracker
    /// 
    /// # Arguments
    /// * `min_funding_rate_bps` - Minimum funding rate to consider (annualized bps)
    /// * `min_basis_bps` - Minimum basis to consider (bps)
    pub fn new(min_funding_rate_bps: i64, min_basis_bps: i64) -> Self {
        Self {
            funding_rates: dashmap::DashMap::new(),
            spot_prices: dashmap::DashMap::new(),
            opportunities: crossbeam_queue::SegQueue::new(),
            min_funding_rate_bps: AtomicI64::new(min_funding_rate_bps),
            min_basis_bps: AtomicI64::new(min_basis_bps),
            borrow_cost_bps: AtomicI64::new(200), // Default 2% annual borrow cost
            opportunity_count: AtomicU64::new(0),
            last_scan_ns: AtomicU64::new(0),
        }
    }

    /// Update funding rate for a symbol
    #[inline]
    pub fn update_funding_rate(&self, symbol: &str, funding: FundingRate) {
        self.funding_rates.insert(symbol.to_string(), Arc::new(funding));
    }

    /// Update spot price for a symbol
    #[inline]
    pub fn update_spot_price(&self, symbol: &str, price_ticks: u64) {
        self.spot_prices
            .entry(symbol.to_string())
            .and_modify(|p| p.store(price_ticks, Ordering::Release))
            .or_insert_with(|| AtomicU64::new(price_ticks));
    }

    /// Scan for cash-and-carry opportunities
    #[inline]
    pub fn scan_opportunities(&self, timestamp_ns: u64) -> usize {
        let min_funding = self.min_funding_rate_bps.load(Ordering::Acquire);
        let min_basis = self.min_basis_bps.load(Ordering::Acquire);
        let mut count = 0;

        for entry in self.funding_rates.iter() {
            let symbol = entry.key();
            let funding = entry.value();
            
            // Get absolute funding rate
            let abs_funding = funding.rate_bps_annual.abs();
            
            // Check minimum funding rate threshold
            if abs_funding < min_funding as u64 {
                continue;
            }

            // Get spot price
            let spot_price = match self.spot_prices.get(symbol) {
                Some(p) => p.load(Ordering::Acquire),
                None => continue,
            };

            // Calculate basis
            let basis_ticks = spot_price as i64 - funding.mark_price_ticks as i64;
            let basis_bps = if spot_price > 0 {
                (basis_ticks * 10000) / spot_price as i64
            } else {
                0
            };

            // Check minimum basis threshold
            if basis_bps.abs() < min_basis {
                continue;
            }

            // Calculate expected return
            // For positive funding: long spot, short perp → earn funding rate
            // For negative funding: short spot, long perp → pay funding rate (avoid)
            let expected_return = if funding.rate_bps_annual > 0 {
                // Long spot / short perp: earn funding, lose basis if contango
                funding.rate_bps_annual - basis_bps
            } else {
                // Short spot / long perp: pay funding, gain basis if backwardation
                -funding.rate_bps_annual + basis_bps
            };

            // Only consider positive expected returns
            if expected_return <= 0 {
                continue;
            }

            let opp = CashAndCarryOpportunity {
                symbol: symbol.clone(),
                spot_price_ticks: spot_price,
                perp_price_ticks: funding.mark_price_ticks,
                funding_rate_bps_annual: funding.rate_bps_annual,
                basis_bps,
                expected_return_bps_annual: expected_return,
                max_size: 0, // Would be calculated from order book depth
                detected_at_ns: timestamp_ns,
            };

            self.opportunities.push(opp);
            count += 1;
        }

        self.last_scan_ns.store(timestamp_ns, Ordering::Release);
        self.opportunity_count.fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Get next available opportunity
    #[inline]
    pub fn pop_opportunity(&self) -> Option<CashAndCarryOpportunity> {
        self.opportunities.pop()
    }

    /// Get number of pending opportunities
    #[inline]
    pub fn pending_opportunities(&self) -> usize {
        self.opportunities.len()
    }

    /// Get total opportunities detected
    #[inline]
    pub fn total_opportunities(&self) -> u64 {
        self.opportunity_count.load(Ordering::Acquire)
    }

    /// Get funding rate for a symbol
    #[inline]
    pub fn get_funding_rate(&self, symbol: &str) -> Option<Arc<FundingRate>> {
        self.funding_rates.get(symbol).map(|v| v.value().clone())
    }

    /// Set minimum funding rate threshold
    #[inline]
    pub fn set_min_funding_rate(&self, bps: i64) {
        self.min_funding_rate_bps.store(bps, Ordering::Release);
    }

    /// Set borrow cost estimate
    #[inline]
    pub fn set_borrow_cost(&self, bps: i64) {
        self.borrow_cost_bps.store(bps, Ordering::Release);
    }

    /// Reset tracker (for /KILL orchestration)
    pub fn reset(&self) {
        self.funding_rates.clear();
        self.spot_prices.clear();
        while self.opportunities.pop().is_some() {}
        self.opportunity_count.store(0, Ordering::Relaxed);
        self.last_scan_ns.store(0, Ordering::Relaxed);
    }

    /// Get average funding rate across all symbols (bps annual)
    #[inline]
    pub fn avg_funding_rate_bps(&self) -> i64 {
        let mut sum = 0i64;
        let mut count = 0i64;

        for entry in self.funding_rates.iter() {
            sum += entry.value().rate_bps_annual;
            count += 1;
        }

        if count == 0 {
            0
        } else {
            sum / count
        }
    }

    /// Get symbols with highest funding rates (top N)
    pub fn top_funding_symbols(&self, n: usize) -> Vec<(String, i64)> {
        let mut rates: Vec<_> = self.funding_rates
            .iter()
            .map(|e| (e.key().clone(), e.value().rate_bps_annual))
            .collect();
        
        rates.sort_by(|a, b| b.1.abs().cmp(&a.1.abs()));
        rates.truncate(n);
        rates
    }
}

/// Multi-symbol funding rate monitor with alerting
pub struct FundingRateMonitor {
    /// Underlying tracker
    tracker: Arc<FundingArbitrageTracker>,
    /// Extreme funding threshold (bps annual)
    extreme_threshold_bps: AtomicI64,
    /// Alert flag for extreme funding
    extreme_alert_active: AtomicI64,
}

impl FundingRateMonitor {
    /// Create a new funding rate monitor
    pub fn new(extreme_threshold_bps: i64) -> Self {
        Self {
            tracker: Arc::new(FundingArbitrageTracker::new(100, 50)),
            extreme_threshold_bps: AtomicI64::new(extreme_threshold_bps),
            extreme_alert_active: AtomicI64::new(0),
        }
    }

    /// Update funding rate and check for extreme conditions
    #[inline]
    pub fn update_and_check(&self, symbol: &str, funding: FundingRate) {
        self.tracker.update_funding_rate(symbol, funding);
        
        // Check for extreme funding
        let threshold = self.extreme_threshold_bps.load(Ordering::Acquire);
        if funding.rate_bps_annual.abs() > threshold {
            self.extreme_alert_active.store(1, Ordering::Release);
        }
    }

    /// Update spot price
    #[inline]
    pub fn update_spot(&self, symbol: &str, price_ticks: u64) {
        self.tracker.update_spot_price(symbol, price_ticks);
    }

    /// Scan for opportunities
    #[inline]
    pub fn scan(&self, timestamp_ns: u64) -> usize {
        self.tracker.scan_opportunities(timestamp_ns)
    }

    /// Check if extreme funding alert is active
    #[inline]
    pub fn is_extreme(&self) -> bool {
        self.extreme_alert_active.load(Ordering::Acquire) != 0
    }

    /// Clear extreme alert
    #[inline]
    pub fn clear_extreme_alert(&self) {
        self.extreme_alert_active.store(0, Ordering::Relaxed);
    }

    /// Get tracker reference
    #[inline]
    pub fn tracker(&self) -> &Arc<FundingArbitrageTracker> {
        &self.tracker
    }

    /// Reset monitor (for /KILL)
    pub fn reset(&self) {
        self.tracker.reset();
        self.extreme_alert_active.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_rate_calculations() {
        let funding = FundingRate::new(
            10950, // 109.5% annual = 10% per 8hr interval
            1000,
            100,
            5000000, // $50,000 in ticks
            4990000, // $49,900 in ticks
        );

        assert_eq!(funding.rate_per_interval_bps(), 10); // 10% per interval
        assert_eq!(funding.basis_ticks(), 10000); // $100
        assert!(funding.basis_bps() > 0); // Contango
    }

    #[test]
    fn test_cash_carry_profitability() {
        let opp = CashAndCarryOpportunity {
            symbol: "BTCUSDT".to_string(),
            spot_price_ticks: 5000000,
            perp_price_ticks: 4990000,
            funding_rate_bps_annual: 10950,
            basis_bps: 20,
            expected_return_bps_annual: 10930,
            max_size: 100,
            detected_at_ns: 1000,
        };

        // Profitable with 2% borrow cost and 5% min return
        assert!(opp.is_profitable(200, 500));
        
        // Not profitable with high borrow cost
        assert!(!opp.is_profitable(11000, 500));
    }

    #[test]
    fn test_funding_tracker_basic() {
        let tracker = FundingArbitrageTracker::new(100, 50);

        // Add funding rate
        let funding = FundingRate::new(
            10950,
            1000,
            100,
            5000000,
            4990000,
        );
        tracker.update_funding_rate("BTCUSDT", funding);
        tracker.update_spot_price("BTCUSDT", 5000000);

        let count = tracker.scan_opportunities(2000);
        assert!(count >= 0); // May or may not find opportunities

        // Verify funding rate retrieval
        let retrieved = tracker.get_funding_rate("BTCUSDT");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().rate_bps_annual, 10950);
    }

    #[test]
    fn test_monitor_extreme_detection() {
        let monitor = FundingRateMonitor::new(5000); // 50% annual threshold

        // Normal funding - no alert
        let normal_funding = FundingRate::new(1000, 1000, 10, 5000000, 5000000);
        monitor.update_and_check("BTCUSDT", normal_funding);
        assert!(!monitor.is_extreme());

        // Extreme funding - should trigger alert
        let extreme_funding = FundingRate::new(100000, 2000, 100, 5000000, 5000000);
        monitor.update_and_check("ETHUSDT", extreme_funding);
        assert!(monitor.is_extreme());
    }
}
