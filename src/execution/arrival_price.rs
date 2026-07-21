//! Arrival Price Benchmarking and VWAP Execution Tracker
//!
//! This module provides institutional-grade execution quality metrics by tracking
//! Arrival Price (price at order initiation) and Volume-Weighted Average Price (VWAP)
//! performance. Essential for SOUL.md post-mortem analysis.
//!
//! Optimized for microsecond latency with lock-free operations and zero allocations in hot path.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::time::{Instant, Duration};
use crate::memory::allocator::GlobalMemoryTracker;

/// Fixed-point precision for price/quantity storage (8 decimal places)
const FIXED_POINT: u64 = 100_000_000;

/// Arrival Price execution tracker state
pub struct ArrivalPriceTracker {
    /// Arrival price (at order initiation) in fixed-point
    arrival_price: AtomicU64,
    /// Total executed quantity in fixed-point
    total_qty: AtomicU64,
    /// Total notional value (price * qty) in fixed-point squared
    total_notional: AtomicU64,
    /// Number of fills
    fill_count: AtomicUsize,
    /// Start timestamp (microseconds since epoch)
    start_ts: AtomicU64,
    /// End timestamp (0 if not complete)
    end_ts: AtomicU64,
    /// Side: 1 for buy, -1 for sell
    side: i8,
    /// Symbol hash for identification
    symbol_hash: u64,
    /// Active flag
    is_active: AtomicBool,
}

impl ArrivalPriceTracker {
    pub fn new(arrival_price: f64, side: i8, symbol_hash: u64) -> Self {
        GlobalMemoryTracker::allocate(128).expect("ArrivalPriceTracker allocation failed");
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self {
            arrival_price: AtomicU64::new((arrival_price * FIXED_POINT as f64) as u64),
            total_qty: AtomicU64::new(0),
            total_notional: AtomicU64::new(0),
            fill_count: AtomicUsize::new(0),
            start_ts: AtomicU64::new(now),
            end_ts: 0,
            side,
            symbol_hash,
            is_active: AtomicBool::new(true),
        }
    }

    /// Record a fill
    #[inline]
    pub fn record_fill(&self, qty: f64, price: f64) {
        let qty_fp = (qty * FIXED_POINT as f64) as u64;
        let price_fp = (price * FIXED_POINT as f64) as u64;
        let notional = ((qty * price) * FIXED_POINT as f64) as u64;

        self.total_qty.fetch_add(qty_fp, Ordering::Relaxed);
        self.total_notional.fetch_add(notional, Ordering::Relaxed);
        self.fill_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current VWAP
    #[inline]
    pub fn get_vwap(&self) -> f64 {
        let qty = self.total_qty.load(Ordering::Relaxed);
        if qty == 0 {
            return 0.0;
        }
        let notional = self.total_notional.load(Ordering::Relaxed);
        (notional as f64 / FIXED_POINT as f64) / (qty as f64 / FIXED_POINT as f64)
    }

    /// Get arrival price
    #[inline]
    pub fn get_arrival_price(&self) -> f64 {
        self.arrival_price.load(Ordering::Relaxed) as f64 / FIXED_POINT as f64
    }

    /// Calculate slippage in basis points relative to arrival price
    #[inline]
    pub fn get_slippage_bps(&self) -> f64 {
        let arrival = self.get_arrival_price();
        let vwap = self.get_vwap();
        
        if arrival == 0.0 || vwap == 0.0 {
            return 0.0;
        }

        let slippage = if self.side > 0 {
            // Buy: positive slippage means we paid more than arrival
            vwap - arrival
        } else {
            // Sell: positive slippage means we received less than arrival
            arrival - vwap
        };

        (slippage / arrival) * 10000.0
    }

    /// Mark execution as complete
    #[inline]
    pub fn complete(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        self.end_ts = now;
        self.is_active.store(false, Ordering::Release);
    }

    /// Get execution duration in microseconds
    #[inline]
    pub fn get_duration_us(&self) -> u64 {
        let start = self.start_ts.load(Ordering::Relaxed);
        let end = if self.end_ts == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64
        } else {
            self.end_ts
        };
        end - start
    }

    /// Get total executed quantity
    #[inline]
    pub fn get_total_qty(&self) -> f64 {
        self.total_qty.load(Ordering::Relaxed) as f64 / FIXED_POINT as f64
    }

    /// Get fill count
    #[inline]
    pub fn get_fill_count(&self) -> usize {
        self.fill_count.load(Ordering::Relaxed)
    }

    /// Log metrics for SOUL.md post-mortem
    pub fn log_metrics(&self, symbol: &str) {
        let arrival = self.get_arrival_price();
        let vwap = self.get_vwap();
        let slippage_bps = self.get_slippage_bps();
        let duration_ms = self.get_duration_us() / 1000;
        let qty = self.get_total_qty();
        let fills = self.get_fill_count();

        eprintln!(
            "[ARRIVAL_METRIC] symbol={} side={} arrival={:.8} vwap={:.8} slippage_bps={:.2} \
             duration_ms={} qty={:.8} fills={}",
            symbol,
            if self.side > 0 { "BUY" } else { "SELL" },
            arrival,
            vwap,
            slippage_bps,
            duration_ms,
            qty,
            fills
        );
    }
}

impl Drop for ArrivalPriceTracker {
    fn drop(&mut self) {
        GlobalMemoryTracker::deallocate(128);
    }
}

/// VWAP Execution Tracker for continuous monitoring
pub struct VWAPTracker {
    /// Target VWAP (from historical or benchmark)
    target_vwap: AtomicU64,
    /// Running sum of price * qty
    running_notional: AtomicU64,
    /// Running sum of qty
    running_qty: AtomicU64,
    /// Target quantity to execute
    target_qty: AtomicU64,
    /// Binance maker fee rate (fixed-point)
    maker_fee_fp: AtomicU64,
    /// Binance taker fee rate (fixed-point)
    taker_fee_fp: AtomicU64,
    /// Is active
    is_active: AtomicBool,
}

impl VWAPTracker {
    pub fn new(target_vwap: f64, target_qty: f64, maker_fee: f64, taker_fee: f64) -> Self {
        GlobalMemoryTracker::allocate(128).expect("VWAPTracker allocation failed");

        Self {
            target_vwap: AtomicU64::new((target_vwap * FIXED_POINT as f64) as u64),
            target_qty: AtomicU64::new((target_qty * FIXED_POINT as f64) as u64),
            running_notional: AtomicU64::new(0),
            running_qty: AtomicU64::new(0),
            maker_fee_fp: AtomicU64::new((maker_fee.abs() * 10000.0) as u64), // Basis points
            taker_fee_fp: AtomicU64::new((taker_fee.abs() * 10000.0) as u64),
            is_active: AtomicBool::new(true),
        }
    }

    /// Record a fill
    #[inline]
    pub fn record_fill(&self, qty: f64, price: f64) {
        let qty_fp = (qty * FIXED_POINT as f64) as u64;
        let notional = ((qty * price) * FIXED_POINT as f64) as u64;

        self.running_qty.fetch_add(qty_fp, Ordering::Relaxed);
        self.running_notional.fetch_add(notional, Ordering::Relaxed);
    }

    /// Get current realized VWAP
    #[inline]
    pub fn get_realized_vwap(&self) -> f64 {
        let qty = self.running_qty.load(Ordering::Relaxed);
        if qty == 0 {
            return 0.0;
        }
        let notional = self.running_notional.load(Ordering::Relaxed);
        (notional as f64 / FIXED_POINT as f64) / (qty as f64 / FIXED_POINT as f64)
    }

    /// Get target VWAP
    #[inline]
    pub fn get_target_vwap(&self) -> f64 {
        self.target_vwap.load(Ordering::Relaxed) as f64 / FIXED_POINT as f64
    }

    /// Get execution progress (0.0 - 1.0)
    #[inline]
    pub fn get_progress(&self) -> f64 {
        let target = self.target_qty.load(Ordering::Relaxed);
        if target == 0 {
            return 0.0;
        }
        let current = self.running_qty.load(Ordering::Relaxed);
        current as f64 / target as f64
    }

    /// Calculate VWAP slippage in basis points
    #[inline]
    pub fn get_vwap_slippage_bps(&self) -> f64 {
        let target = self.get_target_vwap();
        let realized = self.get_realized_vwap();
        
        if target == 0.0 || realized == 0.0 {
            return 0.0;
        }

        ((realized - target) / target) * 10000.0
    }

    /// Estimate total fees based on maker/taker ratio assumption (50/50)
    #[inline]
    pub fn estimate_fees(&self, avg_price: f64) -> f64 {
        let qty = self.get_progress() * (self.target_qty.load(Ordering::Relaxed) as f64 / FIXED_POINT as f64);
        let notional = qty * avg_price;
        
        let maker_rate = self.maker_fee_fp.load(Ordering::Relaxed) as f64 / 10000.0;
        let taker_rate = self.taker_fee_fp.load(Ordering::Relaxed) as f64 / 10000.0;
        
        // Assume 50% maker, 50% taker
        let avg_rate = (maker_rate + taker_rate) / 2.0;
        notional * avg_rate
    }

    /// Check if target quantity reached
    #[inline]
    pub fn is_complete(&self) -> bool {
        let target = self.target_qty.load(Ordering::Acquire);
        let current = self.running_qty.load(Ordering::Acquire);
        current >= target || !self.is_active.load(Ordering::Relaxed)
    }

    /// Deactivate tracker
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Log VWAP metrics
    pub fn log_metrics(&self, symbol: &str) {
        let target = self.get_target_vwap();
        let realized = self.get_realized_vwap();
        let slippage = self.get_vwap_slippage_bps();
        let progress = self.get_progress() * 100.0;
        let fees = self.estimate_fees(realized);

        eprintln!(
            "[VWAP_METRIC] symbol={} target_vwap={:.8} realized_vwap={:.8} slippage_bps={:.2} \
             progress={:.1}% est_fees={:.8}",
            symbol, target, realized, slippage, progress, fees
        );
    }
}

impl Drop for VWAPTracker {
    fn drop(&mut self) {
        GlobalMemoryTracker::deallocate(128);
    }
}

/// Execution quality report for SOUL.md post-mortems
#[derive(Debug)]
pub struct ExecutionQualityReport {
    pub symbol: String,
    pub side: String,
    pub arrival_price: f64,
    pub vwap: f64,
    pub slippage_bps: f64,
    pub duration_ms: u64,
    pub total_qty: f64,
    pub fill_count: usize,
    pub estimated_fees: f64,
    pub timestamp: u64,
}

impl ExecutionQualityReport {
    pub fn generate(tracker: &ArrivalPriceTracker, symbol: &str, avg_price: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side: if tracker.side > 0 { "BUY" } else { "SELL" }.to_string(),
            arrival_price: tracker.get_arrival_price(),
            vwap: tracker.get_vwap(),
            slippage_bps: tracker.get_slippage_bps(),
            duration_ms: tracker.get_duration_us() / 1000,
            total_qty: tracker.get_total_qty(),
            fill_count: tracker.get_fill_count(),
            estimated_fees: (tracker.get_total_qty() * avg_price) * 0.0002, // Avg fee estimate
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Format for SOUL.md logging
    pub fn format_soul(&self) -> String {
        format!(
            "## Execution Quality Report\n\
             - **Symbol**: {}\n\
             - **Side**: {}\n\
             - **Arrival Price**: {:.8}\n\
             - **VWAP**: {:.8}\n\
             - **Slippage**: {:.2} bps\n\
             - **Duration**: {} ms\n\
             - **Quantity**: {:.8}\n\
             - **Fills**: {}\n\
             - **Est. Fees**: {:.8}\n\
             - **Timestamp**: {}",
            self.symbol, self.side, self.arrival_price, self.vwap,
            self.slippage_bps, self.duration_ms, self.total_qty,
            self.fill_count, self.estimated_fees, self.timestamp
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrival_price_tracker() {
        let tracker = ArrivalPriceTracker::new(50000.0, 1, 12345);
        tracker.record_fill(10.0, 50001.0);
        tracker.record_fill(10.0, 50002.0);

        assert_eq!(tracker.get_arrival_price(), 50000.0);
        assert!((tracker.get_vwap() - 50001.5).abs() < 0.01);
        assert!(tracker.get_slippage_bps() > 0.0); // Positive slippage for buy above arrival
    }

    #[test]
    fn test_vwap_tracker() {
        let tracker = VWAPTracker::new(50000.0, 100.0, -0.0001, 0.0004);
        tracker.record_fill(25.0, 50000.0);
        tracker.record_fill(25.0, 50001.0);

        assert!((tracker.get_progress() - 0.5).abs() < 0.01);
        assert!(!tracker.is_complete());
    }

    #[test]
    fn test_execution_quality_report() {
        let tracker = ArrivalPriceTracker::new(50000.0, -1, 67890);
        tracker.record_fill(50.0, 49999.0);
        tracker.complete();

        let report = ExecutionQualityReport::generate(&tracker, "BTCUSDT", 49999.0);
        assert_eq!(report.side, "SELL");
        assert!(report.slippage_bps > 0.0); // Positive slippage for sell below arrival
    }
}
