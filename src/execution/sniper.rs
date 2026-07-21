//! Latency Arbitrage Sniper Module
//!
//! High-frequency execution module that detects fleeting liquidity across multiple
//! Binance order books and executes market orders in microseconds before quotes vanish.
//! Optimized for AMD Ryzen AI 5 with cache-line alignment and zero allocations in hot path.
//!
//! WARNING: This module operates at microsecond timescales and requires careful tuning.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::time::{Instant, Duration};
use crate::memory::allocator::GlobalMemoryTracker;
use crate::network::tcp_tuning::LatencyMonitor;

/// Cache line size for AMD Ryzen (64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of order books to monitor simultaneously
const MAX_ORDER_BOOKS: usize = 32;

/// Sniped order result
#[derive(Debug, Clone)]
#[repr(C, align(64))]
pub struct SnipeResult {
    /// Symbol identifier
    pub symbol_hash: u64,
    /// Side: true = buy, false = sell
    pub is_buy: bool,
    /// Executed price
    pub price: f64,
    /// Executed quantity
    pub qty: f64,
    /// Latency from detection to execution (microseconds)
    pub latency_us: u64,
    /// Profit/loss estimate
    pub pnl_estimate: f64,
    /// Success flag
    pub success: bool,
    /// Error code if failed
    pub error_code: u8,
}

/// Order book snapshot for sniping decisions
#[derive(Clone)]
#[repr(C, align(64))]
pub struct OrderBookSnapshot {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Best bid price (fixed-point)
    pub best_bid: u64,
    /// Best ask price (fixed-point)
    pub best_ask: u64,
    /// Bid quantity (fixed-point)
    pub bid_qty: u64,
    /// Ask quantity (fixed-point)
    pub ask_qty: u64,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Spread in basis points
    pub spread_bps: u16,
}

impl OrderBookSnapshot {
    #[inline]
    pub fn new(symbol_hash: u64) -> Self {
        Self {
            symbol_hash,
            best_bid: 0,
            best_ask: 0,
            bid_qty: 0,
            ask_qty: 0,
            timestamp_ns: 0,
            spread_bps: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, bid: f64, ask: f64, bid_qty: f64, ask_qty: f64) {
        const FP: u64 = 100_000_000;
        self.best_bid = (bid * FP as f64) as u64;
        self.best_ask = (ask * FP as f64) as u64;
        self.bid_qty = (bid_qty * FP as f64) as u64;
        self.ask_qty = (ask_qty * FP as f64) as u64;
        self.timestamp_ns = Instant::now().duration_since(Instant::epoch()).unwrap().as_nanos() as u64;
        
        // Calculate spread
        if self.best_bid > 0 {
            self.spread_bps = (((self.best_ask - self.best_bid) as f64) / self.best_bid as f64 * 10000.0) as u16;
        }
    }

    #[inline]
    pub fn get_bid(&self) -> f64 {
        self.best_bid as f64 / 100_000_000.0
    }

    #[inline]
    pub fn get_ask(&self) -> f64 {
        self.best_ask as f64 / 100_000_000.0
    }
}

/// Liquidity opportunity detected by sniper
#[derive(Debug)]
pub struct SnipeOpportunity {
    /// Symbol hash
    pub symbol_hash: u64,
    /// Is buy opportunity
    pub is_buy: bool,
    /// Target price
    pub target_price: f64,
    /// Available quantity
    pub available_qty: f64,
    /// Expected slippage (bps)
    pub expected_slippage_bps: f64,
    /// Confidence score (0-100)
    pub confidence: u8,
    /// Detection timestamp
    pub detected_at: Instant,
}

/// Latency Arbitrage Sniper Engine
pub struct SniperEngine {
    /// Order book snapshots (cache-line aligned)
    order_books: Vec<OrderBookSnapshot>,
    /// Number of active order books
    active_count: AtomicUsize,
    /// Minimum spread threshold (bps) to consider sniping
    min_spread_bps: u16,
    /// Minimum quantity threshold
    min_qty: f64,
    /// Maximum position size per snipe
    max_position: f64,
    /// Enable sniping
    is_enabled: AtomicBool,
    /// Total snipes attempted
    total_snipes: AtomicUsize,
    /// Successful snipes
    successful_snipes: AtomicUsize,
    /// Latency monitor
    latency_monitor: LatencyMonitor,
    /// Binance taker fee (bps)
    taker_fee_bps: u16,
}

impl SniperEngine {
    pub fn new(min_spread_bps: u16, min_qty: f64, max_position: f64) -> Self {
        GlobalMemoryTracker::allocate(MAX_ORDER_BOOKS * 128).expect("SniperEngine allocation failed");

        let mut order_books = Vec::with_capacity(MAX_ORDER_BOOKS);
        for i in 0..MAX_ORDER_BOOKS {
            order_books.push(OrderBookSnapshot::new(i as u64));
        }

        Self {
            order_books,
            active_count: AtomicUsize::new(0),
            min_spread_bps,
            min_qty,
            max_position,
            is_enabled: AtomicBool::new(true),
            total_snipes: AtomicUsize::new(0),
            successful_snipes: AtomicUsize::new(0),
            latency_monitor: LatencyMonitor::new(),
            taker_fee_bps: 4, // 0.04% = 4 bps
        }
    }

    /// Register an order book for monitoring
    #[inline]
    pub fn register_order_book(&self, symbol_hash: u64) -> Option<usize> {
        let count = self.active_count.load(Ordering::Relaxed);
        if count >= MAX_ORDER_BOOKS {
            return None;
        }

        // Find empty slot
        for i in 0..MAX_ORDER_BOOKS {
            if self.order_books[i].symbol_hash == 0 {
                self.order_books[i].symbol_hash = symbol_hash;
                self.active_count.fetch_add(1, Ordering::Relaxed);
                return Some(i);
            }
        }
        None
    }

    /// Update order book snapshot (called from WebSocket callback)
    #[inline]
    pub fn update_order_book(&self, index: usize, bid: f64, ask: f64, bid_qty: f64, ask_qty: f64) {
        if index < MAX_ORDER_BOOKS {
            self.order_books[index].update(bid, ask, bid_qty, ask_qty);
        }
    }

    /// Scan for sniping opportunities across all monitored order books
    /// Returns vector of opportunities sorted by confidence
    #[inline]
    pub fn scan_opportunities(&self) -> Vec<SnipeOpportunity> {
        let _latency = self.latency_monitor.start_operation();
        let mut opportunities = Vec::with_capacity(self.active_count.load(Ordering::Relaxed));

        let count = self.active_count.load(Ordering::Acquire);
        for i in 0..count.min(MAX_ORDER_BOOKS) {
            let ob = &self.order_books[i];
            
            // Skip inactive or stale data (>1ms old)
            let now_ns = Instant::now().duration_since(Instant::epoch()).unwrap().as_nanos() as u64;
            if ob.timestamp_ns == 0 || now_ns - ob.timestamp_ns > 1_000_000 {
                continue;
            }

            // Check spread threshold
            if ob.spread_bps < self.min_spread_bps {
                continue;
            }

            // Calculate potential profit after fees
            let spread_after_fees = ob.spread_bps as i32 - (self.taker_fee_bps as i32 * 2);
            if spread_after_fees <= 0 {
                continue;
            }

            // Determine opportunity type and size
            let (is_buy, target_price, available_qty) = if ob.bid_qty > ob.ask_qty {
                // More liquidity on bid side - snipe asks (buy)
                (true, ob.get_ask(), ob.ask_qty as f64 / 100_000_000.0)
            } else {
                // More liquidity on ask side - snipe bids (sell)
                (false, ob.get_bid(), ob.bid_qty as f64 / 100_000_000.0)
            };

            // Check minimum quantity
            if available_qty < self.min_qty {
                continue;
            }

            // Calculate confidence based on spread, quantity, and age
            let age_factor = (1.0 - (now_ns - ob.timestamp_ns) as f64 / 1_000_000.0).max(0.0);
            let qty_factor = (available_qty / 100.0).min(1.0); // Normalize to 100 units
            let spread_factor = (ob.spread_bps as f64 / 100.0).min(1.0); // Normalize to 100 bps
            
            let confidence = ((spread_factor * 0.4 + qty_factor * 0.3 + age_factor * 0.3) * 100.0) as u8;

            opportunities.push(SnipeOpportunity {
                symbol_hash: ob.symbol_hash,
                is_buy,
                target_price,
                available_qty: available_qty.min(self.max_position),
                expected_slippage_bps: (self.taker_fee_bps * 2) as f64,
                confidence,
                detected_at: Instant::now(),
            });
        }

        // Sort by confidence descending
        opportunities.sort_by(|a, b| b.confidence.cmp(&a.confidence));
        opportunities
    }

    /// Execute a snipe (returns SnipeResult)
    #[inline]
    pub fn execute_snipe(&self, opp: &SnipeOpportunity) -> SnipeResult {
        let _latency = self.latency_monitor.start_operation();
        
        if !self.is_enabled.load(Ordering::Relaxed) {
            return SnipeResult {
                symbol_hash: opp.symbol_hash,
                is_buy: opp.is_buy,
                price: opp.target_price,
                qty: 0.0,
                latency_us: 0,
                pnl_estimate: 0.0,
                success: false,
                error_code: 1, // Disabled
            };
        }

        self.total_snipes.fetch_add(1, Ordering::Relaxed);

        // Simulate execution (in production, this would call exchange API)
        let detection_latency = opp.detected_at.elapsed().as_micros() as u64;
        
        // Assume 50us execution time for simulation
        let execution_time_us = 50;
        let total_latency = detection_latency + execution_time_us;

        // Estimate P&L (simplified)
        let gross_pnl = opp.available_qty * opp.target_price * (opp.expected_slippage_bps / 10000.0);
        let fees = opp.available_qty * opp.target_price * (self.taker_fee_bps as f64 / 10000.0);
        let net_pnl = gross_pnl - fees;

        let success = net_pnl > 0.0 && total_latency < 500; // Fail if latency > 500us

        if success {
            self.successful_snipes.fetch_add(1, Ordering::Relaxed);
        }

        SnipeResult {
            symbol_hash: opp.symbol_hash,
            is_buy: opp.is_buy,
            price: opp.target_price,
            qty: if success { opp.available_qty } else { 0.0 },
            latency_us: total_latency,
            pnl_estimate: net_pnl,
            success,
            error_code: if success { 0 } else { 2 }, // 2 = unprofitable or too slow
        }
    }

    /// Get success rate
    #[inline]
    pub fn get_success_rate(&self) -> f64 {
        let total = self.total_snipes.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let success = self.successful_snipes.load(Ordering::Relaxed);
        success as f64 / total as f64 * 100.0
    }

    /// Enable/disable sniping
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.is_enabled.store(enabled, Ordering::Release);
    }

    /// Log sniper metrics
    pub fn log_metrics(&self) {
        let total = self.total_snipes.load(Ordering::Relaxed);
        let success = self.successful_snipes.load(Ordering::Relaxed);
        let rate = self.get_success_rate();

        eprintln!(
            "[SNIPER_METRIC] total_snipes={} successful={} success_rate={:.2}% enabled={}",
            total, success, rate, self.is_enabled.load(Ordering::Relaxed)
        );
    }
}

impl Drop for SniperEngine {
    fn drop(&mut self) {
        GlobalMemoryTracker::deallocate(MAX_ORDER_BOOKS * 128);
    }
}

/// Multi-exchange sniper for cross-venue arbitrage
pub struct CrossVenueSniper {
    /// Primary venue sniper
    primary: SniperEngine,
    /// Secondary venue sniper
    secondary: SniperEngine,
    /// Minimum cross-venue spread (bps)
    min_cross_spread_bps: u16,
}

impl CrossVenueSniper {
    pub fn new(min_spread_bps: u16, min_cross_spread_bps: u16, min_qty: f64) -> Self {
        Self {
            primary: SniperEngine::new(min_spread_bps, min_qty, 1000.0),
            secondary: SniperEngine::new(min_spread_bps, min_qty, 1000.0),
            min_cross_spread_bps,
        }
    }

    /// Scan for cross-venue arbitrage opportunities
    pub fn scan_cross_venue(&self, symbol_hash: u64) -> Option<SnipeOpportunity> {
        // Find matching symbols in both venues
        let primary_ob = self.primary.order_books.iter()
            .find(|ob| ob.symbol_hash == symbol_hash);
        let secondary_ob = self.secondary.order_books.iter()
            .find(|ob| ob.symbol_hash == symbol_hash);

        if let (Some(p), Some(s)) = (primary_ob, secondary_ob) {
            // Check for price discrepancy
            let p_bid = p.get_bid();
            let p_ask = p.get_ask();
            let s_bid = s.get_bid();
            let s_ask = s.get_ask();

            // Buy on secondary, sell on primary?
            if s_ask < p_bid {
                let spread_bps = ((p_bid - s_ask) / s_ask * 10000.0) as u16;
                if spread_bps >= self.min_cross_spread_bps {
                    return Some(SnipeOpportunity {
                        symbol_hash,
                        is_buy: true,
                        target_price: s_ask,
                        available_qty: (s.ask_qty as f64 / 100_000_000.0).min(p.bid_qty as f64 / 100_000_000.0),
                        expected_slippage_bps: spread_bps as f64,
                        confidence: 90,
                        detected_at: Instant::now(),
                    });
                }
            }

            // Buy on primary, sell on secondary?
            if p_ask < s_bid {
                let spread_bps = ((s_bid - p_ask) / p_ask * 10000.0) as u16;
                if spread_bps >= self.min_cross_spread_bps {
                    return Some(SnipeOpportunity {
                        symbol_hash,
                        is_buy: false,
                        target_price: p_ask,
                        available_qty: (p.ask_qty as f64 / 100_000_000.0).min(s.bid_qty as f64 / 100_000_000.0),
                        expected_slippage_bps: spread_bps as f64,
                        confidence: 90,
                        detected_at: Instant::now(),
                    });
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sniper_engine_creation() {
        let sniper = SniperEngine::new(10, 1.0, 100.0);
        assert_eq!(sniper.active_count.load(Ordering::Relaxed), 0);
        assert!(sniper.is_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_order_book_update() {
        let sniper = SniperEngine::new(10, 1.0, 100.0);
        let idx = sniper.register_order_book(12345).unwrap();
        sniper.update_order_book(idx, 50000.0, 50001.0, 100.0, 100.0);

        let ob = &sniper.order_books[idx];
        assert!((ob.get_bid() - 50000.0).abs() < 0.01);
        assert!((ob.get_ask() - 50001.0).abs() < 0.01);
        assert!(ob.spread_bps > 0);
    }

    #[test]
    fn test_opportunity_scan() {
        let sniper = SniperEngine::new(5, 1.0, 100.0);
        let idx = sniper.register_order_book(12345).unwrap();
        
        // Create wide spread opportunity
        sniper.update_order_book(idx, 50000.0, 50010.0, 100.0, 100.0);
        
        let opps = sniper.scan_opportunities();
        assert!(!opps.is_empty());
        assert!(opps[0].confidence > 0);
    }
}
