//! Real-Time Greeks Calculator for Portfolio-Wide Risk Management
//!
//! This module calculates Delta, Gamma, Theta, Vega, and Rho for an entire
//! options portfolio in real-time, enabling instantaneous dynamic delta-hedging
//! using underlying spot perpetual contracts.
//!
//! Optimized for microsecond latency with lock-free aggregation and SIMD batch processing.

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use crate::memory::allocator::GlobalMemoryTracker;
use super::black_scholes::{BSParams, BSResult, price_option, price_options_batch};

/// Fixed-point precision for Greek calculations (6 decimal places)
const GREEKS_FP: i64 = 1_000_000;

/// Single option position with Greeks
#[derive(Debug, Clone)]
pub struct OptionPosition {
    /// Unique position ID
    pub id: u64,
    /// Symbol (e.g., "BTC")
    pub symbol: u64, // Hash
    /// Strike price (fixed-point)
    pub strike: u64,
    /// Expiry timestamp (Unix seconds)
    pub expiry_ts: u64,
    /// Position size (positive = long, negative = short)
    pub quantity: i64, // Fixed-point
    /// Option type: true = call, false = put
    pub is_call: bool,
    /// Entry implied volatility (fixed-point)
    pub entry_iv: u64,
    /// Current Black-Scholes result
    pub bs_result: BSResult,
}

impl OptionPosition {
    pub fn new(
        id: u64,
        symbol: u64,
        strike: f64,
        expiry_ts: u64,
        quantity: f64,
        is_call: bool,
        entry_iv: f64,
    ) -> Self {
        const FP: u64 = 100_000_000;
        Self {
            id,
            symbol,
            strike: (strike * FP as f64) as u64,
            expiry_ts,
            quantity: (quantity * 1000.0) as i64, // 3 decimal places for quantity
            is_call,
            entry_iv: (entry_iv * 1000.0) as u64, // 3 decimal places for IV
            bs_result: BSResult {
                call_price: 0.0,
                put_price: 0.0,
                call_delta: 0.0,
                put_delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                call_theta: 0.0,
                put_theta: 0.0,
                rho: 0.0,
            },
        }
    }

    /// Update Greeks based on current market data
    #[inline]
    pub fn update_greeks(&mut self, spot: f64, current_time: u64, rate: f64, yield_: f64) {
        let strike_fp = self.strike as f64 / 100_000_000.0;
        let iv = self.entry_iv as f64 / 1000.0;
        
        // Calculate time to expiry in years
        if current_time >= self.expiry_ts {
            // Expired
            self.bs_result = BSResult {
                call_price: 0.0,
                put_price: 0.0,
                call_delta: 0.0,
                put_delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                call_theta: 0.0,
                put_theta: 0.0,
                rho: 0.0,
            };
            return;
        }
        
        let days_to_expiry = ((self.expiry_ts - current_time) / 86400) as u32;
        let params = BSParams::new(spot, strike_fp, days_to_expiry.max(1), iv, rate, yield_);
        self.bs_result = price_option(&params);
    }

    /// Get position delta (quantity-weighted)
    #[inline]
    pub fn get_delta(&self) -> f64 {
        let option_delta = if self.is_call {
            self.bs_result.call_delta
        } else {
            self.bs_result.put_delta
        };
        (self.quantity as f64 / 1000.0) * option_delta
    }

    /// Get position gamma
    #[inline]
    pub fn get_gamma(&self) -> f64 {
        (self.quantity as f64 / 1000.0) * self.bs_result.gamma
    }

    /// Get position theta (daily P&L from time decay)
    #[inline]
    pub fn get_theta(&self) -> f64 {
        let theta = if self.is_call {
            self.bs_result.call_theta
        } else {
            self.bs_result.put_theta
        };
        (self.quantity as f64 / 1000.0) * theta
    }

    /// Get position vega (P&L per 1% vol move)
    #[inline]
    pub fn get_vega(&self) -> f64 {
        (self.quantity as f64 / 1000.0) * self.bs_result.vega
    }
}

/// Portfolio-wide Greeks aggregator
pub struct PortfolioGreeks {
    /// Total portfolio delta
    total_delta: AtomicI64, // Fixed-point
    /// Total portfolio gamma
    total_gamma: AtomicI64,
    /// Total portfolio theta (daily)
    total_theta: AtomicI64,
    /// Total portfolio vega
    total_vega: AtomicI64,
    /// Number of positions
    position_count: AtomicU64,
    /// Last update timestamp
    last_update_ts: AtomicU64,
    /// Is active
    is_active: AtomicBool,
}

impl PortfolioGreeks {
    pub fn new() -> Self {
        GlobalMemoryTracker::allocate(128).expect("PortfolioGreeks allocation failed");
        
        Self {
            total_delta: AtomicI64::new(0),
            total_gamma: AtomicI64::new(0),
            total_theta: AtomicI64::new(0),
            total_vega: AtomicI64::new(0),
            position_count: AtomicU64::new(0),
            last_update_ts: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Recalculate portfolio Greeks from positions
    #[inline]
    pub fn recalculate(&self, positions: &[OptionPosition], spot: f64) {
        let mut delta_sum: f64 = 0.0;
        let mut gamma_sum: f64 = 0.0;
        let mut theta_sum: f64 = 0.0;
        let mut vega_sum: f64 = 0.0;

        for pos in positions.iter() {
            delta_sum += pos.get_delta();
            gamma_sum += pos.get_gamma();
            theta_sum += pos.get_theta();
            vega_sum += pos.get_vega();
        }

        // Store as fixed-point
        self.total_delta.store((delta_sum * GREEKS_FP as f64) as i64, Ordering::Release);
        self.total_gamma.store((gamma_sum * GREEKS_FP as f64) as i64, Ordering::Relaxed);
        self.total_theta.store((theta_sum * GREEKS_FP as f64) as i64, Ordering::Relaxed);
        self.total_vega.store((vega_sum * GREEKS_FP as f64) as i64, Ordering::Relaxed);
        self.position_count.store(positions.len() as u64, Ordering::Relaxed);
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.last_update_ts.store(now, Ordering::Relaxed);
    }

    /// Get total delta
    #[inline]
    pub fn get_total_delta(&self) -> f64 {
        self.total_delta.load(Ordering::Acquire) as f64 / GREEKS_FP as f64
    }

    /// Get total gamma
    #[inline]
    pub fn get_total_gamma(&self) -> f64 {
        self.total_gamma.load(Ordering::Relaxed) as f64 / GREEKS_FP as f64
    }

    /// Get total theta (daily P&L)
    #[inline]
    pub fn get_total_theta(&self) -> f64 {
        self.total_theta.load(Ordering::Relaxed) as f64 / GREEKS_FP as f64
    }

    /// Get total vega (per 1% vol move)
    #[inline]
    pub fn get_total_vega(&self) -> f64 {
        self.total_vega.load(Ordering::Relaxed) as f64 / GREEKS_FP as f64
    }

    /// Calculate delta-neutral hedge quantity for perpetual contract
    /// Returns positive = buy perp, negative = sell perp
    #[inline]
    pub fn calculate_hedge_quantity(&self, spot: f64, contract_size: f64) -> f64 {
        let portfolio_delta = self.get_total_delta();
        
        // To be delta-neutral, we need opposite delta in the underlying
        // Perpetual contract has delta = 1.0 per unit
        let hedge_delta = -portfolio_delta;
        
        // Convert to contract units
        hedge_delta / contract_size
    }

    /// Check if portfolio exceeds risk limits
    #[inline]
    pub fn check_limits(
        &self,
        max_delta: f64,
        max_gamma: f64,
        max_theta: f64,
        max_vega: f64,
    ) -> Vec<&'static str> {
        let mut breaches = Vec::new();
        
        let delta = self.get_total_delta().abs();
        let gamma = self.get_total_gamma().abs();
        let theta = self.get_total_theta().abs();
        let vega = self.get_total_vega().abs();

        if delta > max_delta {
            breaches.push("DELTA_LIMIT");
        }
        if gamma > max_gamma {
            breaches.push("GAMMA_LIMIT");
        }
        if theta > max_theta {
            breaches.push("THETA_LIMIT");
        }
        if vega > max_vega {
            breaches.push("VEGA_LIMIT");
        }

        breaches
    }

    /// Log portfolio Greeks
    pub fn log_metrics(&self, symbol: &str) {
        eprintln!(
            "[GREEKS] symbol={} delta={:.4} gamma={:.4} theta={:.2} vega={:.2} positions={}",
            symbol,
            self.get_total_delta(),
            self.get_total_gamma(),
            self.get_total_theta(),
            self.get_total_vega(),
            self.position_count.load(Ordering::Relaxed)
        );
    }
}

impl Default for PortfolioGreeks {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PortfolioGreeks {
    fn drop(&mut self) {
        GlobalMemoryTracker::deallocate(128);
    }
}

/// Delta-hedging execution manager
pub struct DeltaHedger {
    /// Target delta (usually 0 for neutral)
    target_delta: f64,
    /// Rebalance threshold (rebalance when delta drifts beyond this)
    rebalance_threshold: f64,
    /// Minimum trade size
    min_trade_size: f64,
    /// Binance perpetual contract sizes (BTC = 0.001, ETH = 0.01, etc.)
    contract_sizes: std::collections::HashMap<u64, f64>,
}

impl DeltaHedger {
    pub fn new(target_delta: f64, rebalance_threshold: f64, min_trade_size: f64) -> Self {
        let mut contract_sizes = std::collections::HashMap::new();
        // Default contract sizes for major cryptos
        contract_sizes.insert(hash_symbol("BTC"), 0.001);
        contract_sizes.insert(hash_symbol("ETH"), 0.01);
        contract_sizes.insert(hash_symbol("SOL"), 0.01);
        
        Self {
            target_delta,
            rebalance_threshold,
            min_trade_size,
            contract_sizes,
        }
    }

    /// Check if rebalancing is needed
    #[inline]
    pub fn needs_rebalance(&self, current_delta: f64) -> bool {
        let drift = (current_delta - self.target_delta).abs();
        drift > self.rebalance_threshold
    }

    /// Calculate hedge order for a symbol
    #[inline]
    pub fn calculate_hedge_order(
        &self,
        symbol_hash: u64,
        portfolio_delta: f64,
        spot: f64,
    ) -> Option<(f64, bool)> {
        // Returns (quantity, is_buy)
        let contract_size = self.contract_sizes.get(&symbol_hash)?;
        
        let hedge_qty = -portfolio_delta / contract_size;
        
        if hedge_qty.abs() < self.min_trade_size {
            return None;
        }

        Some((hedge_qty.abs(), hedge_qty > 0.0))
    }

    /// Execute delta hedge (returns order details)
    pub fn execute_hedge(
        &self,
        greeks: &PortfolioGreeks,
        symbol_hash: u64,
        spot: f64,
    ) -> Option<HedgeOrder> {
        let current_delta = greeks.get_total_delta();
        
        if !self.needs_rebalance(current_delta) {
            return None;
        }

        let (qty, is_buy) = self.calculate_hedge_order(symbol_hash, current_delta, spot)?;

        Some(HedgeOrder {
            symbol_hash,
            quantity: qty,
            is_buy,
            target_delta: self.target_delta,
            current_delta,
            execution_ts: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }
}

/// Hedge order details
#[derive(Debug)]
pub struct HedgeOrder {
    pub symbol_hash: u64,
    pub quantity: f64,
    pub is_buy: bool,
    pub target_delta: f64,
    pub current_delta: f64,
    pub execution_ts: u64,
}

/// Helper function to hash symbol strings
fn hash_symbol(symbol: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    symbol.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_position_greeks() {
        let mut pos = OptionPosition::new(
            1,
            hash_symbol("BTC"),
            50000.0,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() + 86400 * 30, // 30 days
            1.0,
            true,
            0.8,
        );
        
        pos.update_greeks(50000.0, pos.expiry_ts - 86400 * 30, 0.05, 0.0);
        
        // ATM call should have delta ~0.5
        assert!(pos.get_delta().abs() > 0.3);
        assert!(pos.get_delta().abs() < 0.7);
    }

    #[test]
    fn test_portfolio_greeks_aggregation() {
        let greeks = PortfolioGreeks::new();
        
        let mut pos1 = OptionPosition::new(1, 123, 50000.0, 9999999999, 1.0, true, 0.8);
        let mut pos2 = OptionPosition::new(2, 123, 50000.0, 9999999999, -1.0, true, 0.8);
        
        pos1.update_greeks(50000.0, 1000000000, 0.05, 0.0);
        pos2.update_greeks(50000.0, 1000000000, 0.05, 0.0);
        
        let positions = vec![pos1, pos2];
        greeks.recalculate(&positions, 50000.0);
        
        // Long + Short should net to ~0 delta
        assert!(greeks.get_total_delta().abs() < 0.1);
    }

    #[test]
    fn test_delta_hedger() {
        let hedger = DeltaHedger::new(0.0, 0.5, 0.1);
        
        // Large delta should trigger rebalance
        assert!(hedger.needs_rebalance(1.0));
        
        // Small delta should not
        assert!(!hedger.needs_rebalance(0.1));
    }
}
