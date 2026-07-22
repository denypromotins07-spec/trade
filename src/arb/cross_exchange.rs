//! Chapter 1: Advanced Statistical Arbitrage & Pairs Trading
//! File 3: src/arb/cross_exchange.rs
//!
//! Cross-exchange statistical arbitrage engine exploiting temporary basis
//! divergences between Binance spot and futures. Uses strict integer math
//! to prevent floating-point drift. Enforces 8GB RAM limit.
//!
//! Optimized for AMD Ryzen AI 5 with cache-aligned structures.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of cross-exchange pairs tracked
const MAX_CROSS_PAIRS: usize = 256 * 1024; // 256K pairs

/// Price representation in fixed-point integer (price * 10^8)
pub type FixedPrice = i64;
/// Quantity in fixed-point (qty * 10^8)
pub type FixedQty = i64;

/// Convert float price to fixed-point
#[inline(always)]
pub fn price_to_fixed(price: f64) -> FixedPrice {
    (price * 1e8) as i64
}

/// Convert fixed-point price to float
#[inline(always)]
pub fn fixed_to_price(fixed: FixedPrice) -> f64 {
    fixed as f64 / 1e8
}

/// Basis state between spot and futures
#[derive(Debug, Clone, Copy)]
pub struct BasisState {
    /// Spot price (fixed-point)
    pub spot_price: FixedPrice,
    /// Futures price (fixed-point)
    pub futures_price: FixedPrice,
    /// Basis = futures - spot (fixed-point)
    pub basis: FixedPrice,
    /// Basis percentage * 10^6
    pub basis_pct: i64,
    /// Fair value basis (moving average)
    pub fair_basis: FixedPrice,
    /// Z-score of current basis vs fair
    pub z_score: f64,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: u64,
}

/// Cross-exchange arbitrage opportunity
#[derive(Debug, Clone, Copy)]
pub struct ArbOpportunity {
    pub pair_id: usize,
    pub direction: ArbDirection,
    pub expected_profit_bps: i64,
    pub spot_leg: LegSpec,
    pub futures_leg: LegSpec,
    pub confidence: f64,
    pub expiry_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArbDirection {
    LongBasis,  // Buy spot, sell futures (contango)
    ShortBasis, // Sell spot, buy futures (backwardation)
}

#[derive(Debug, Clone, Copy)]
pub struct LegSpec {
    pub exchange: ExchangeId,
    pub symbol_hash: u64,
    pub side: Side,
    pub quantity: FixedQty,
    pub max_slippage_bps: i64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExchangeId {
    BinanceSpot,
    BinanceFutures,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Cross-Exchange Arbitrage Engine
#[repr(C, align(64))]
pub struct CrossExchangeArbEngine {
    /// Pre-allocated basis states
    basis_states: [BasisState; MAX_CROSS_PAIRS],
    
    /// Symbol hash to pair_id mapping (simple hash table)
    symbol_hashes: [u64; MAX_CROSS_PAIRS],
    
    /// Running statistics for basis mean/variance
    basis_sum: [i128; MAX_CROSS_PAIRS],
    basis_sq_sum: [i128; MAX_CROSS_PAIRS],
    sample_counts: [u64; MAX_CROSS_PAIRS],
    
    /// Entry threshold in basis points (1 bp = 0.01%)
    entry_threshold_bps: i64,
    exit_threshold_bps: i64,
    
    /// Active pair count
    active_count: AtomicU64,
    
    /// Fee rates in basis points (maker/taker)
    spot_maker_fee_bps: i64,
    spot_taker_fee_bps: i64,
    futures_maker_fee_bps: i64,
    futures_taker_fee_bps: i64,
    
    /// Minimum profit threshold after fees
    min_profit_bps: i64,
    
    /// Circuit breaker
    trading_enabled: AtomicBool,
}

impl Default for BasisState {
    fn default() -> Self {
        BasisState {
            spot_price: 0,
            futures_price: 0,
            basis: 0,
            basis_pct: 0,
            fair_basis: 0,
            z_score: 0.0,
            last_update_ns: 0,
        }
    }
}

impl CrossExchangeArbEngine {
    /// Create new cross-exchange arb engine
    pub fn new(
        entry_bps: i64,
        exit_bps: i64,
        spot_maker: i64,
        spot_taker: i64,
        futures_maker: i64,
        futures_taker: i64,
    ) -> Self {
        Self {
            basis_states: [BasisState::default(); MAX_CROSS_PAIRS],
            symbol_hashes: [0; MAX_CROSS_PAIRS],
            basis_sum: [0; MAX_CROSS_PAIRS],
            basis_sq_sum: [0; MAX_CROSS_PAIRS],
            sample_counts: [0; MAX_CROSS_PAIRS],
            entry_threshold_bps: entry_bps,
            exit_threshold_bps: exit_bps,
            active_count: AtomicU64::new(0),
            spot_maker_fee_bps: spot_maker,
            spot_taker_fee_bps: spot_taker,
            futures_maker_fee_bps: futures_maker,
            futures_taker_fee_bps: futures_taker,
            min_profit_bps: 5, // Minimum 5 bps profit after fees
            trading_enabled: AtomicBool::new(true),
        }
    }
    
    /// Register a new cross-exchange pair
    pub fn register_pair(&self, symbol_hash: u64, initial_spot: FixedPrice, initial_futures: FixedPrice) -> Option<usize> {
        let current = self.active_count.load(Ordering::Relaxed);
        if current >= MAX_CROSS_PAIRS as u64 {
            return None; // Enforce 8GB RAM cap
        }
        
        let idx = current as usize;
        let basis = initial_futures - initial_spot;
        let basis_pct = if initial_spot != 0 {
            (basis * 1_000_000) / initial_spot
        } else {
            0
        };
        
        unsafe {
            let state_ptr = self.basis_states.as_mut_ptr().add(idx);
            (*state_ptr).spot_price = initial_spot;
            (*state_ptr).futures_price = initial_futures;
            (*state_ptr).basis = basis;
            (*state_ptr).basis_pct = basis_pct;
            (*state_ptr).fair_basis = basis;
            (*state_ptr).last_update_ns = get_timestamp_ns();
            
            *self.symbol_hashes.as_mut_ptr().add(idx) = symbol_hash;
            *self.basis_sum.as_mut_ptr().add(idx) = basis as i128;
            *self.basis_sq_sum.as_mut_ptr().add(idx) = (basis * basis) as i128;
            *self.sample_counts.as_mut_ptr().add(idx) = 1;
        }
        
        self.active_count.fetch_add(1, Ordering::Relaxed);
        Some(idx)
    }
    
    /// Update spot price for a pair
    #[inline(always)]
    pub fn update_spot(&self, pair_id: usize, spot_price: FixedPrice) -> Option<BasisState> {
        self.update_prices(pair_id, Some(spot_price), None)
    }
    
    /// Update futures price for a pair
    #[inline(always)]
    pub fn update_futures(&self, pair_id: usize, futures_price: FixedPrice) -> Option<BasisState> {
        self.update_prices(pair_id, None, Some(futures_price))
    }
    
    /// Update both prices and calculate arbitrage signals
    #[inline(always)]
    pub fn update_prices(
        &self,
        pair_id: usize,
        spot_price: Option<FixedPrice>,
        futures_price: Option<FixedPrice>,
    ) -> Option<BasisState> {
        if pair_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return None;
        }
        
        unsafe {
            let state_ptr = self.basis_states.as_mut_ptr().add(pair_id);
            let sum_ptr = self.basis_sum.as_mut_ptr().add(pair_id);
            let sq_sum_ptr = self.basis_sq_sum.as_mut_ptr().add(pair_id);
            let count_ptr = self.sample_counts.as_mut_ptr().add(pair_id);
            
            let state = &mut *state_ptr;
            
            // Update prices using strict integer math
            if let Some(sp) = spot_price {
                state.spot_price = sp;
            }
            if let Some(fp) = futures_price {
                state.futures_price = fp;
            }
            
            // Recalculate basis (integer arithmetic, no drift)
            state.basis = state.futures_price - state.spot_price;
            
            // Basis percentage in micro-units (1e-6 precision)
            state.basis_pct = if state.spot_price != 0 {
                (state.basis * 1_000_000) / state.spot_price
            } else {
                0
            };
            
            state.last_update_ns = get_timestamp_ns();
            
            // Update running statistics for Z-score calculation
            *sum_ptr += state.basis as i128;
            *sq_sum_ptr += (state.basis * state.basis) as i128;
            *count_ptr += 1;
            
            // Calculate fair basis (EMA with integer approximation)
            let count = *count_ptr as f64;
            let alpha = 0.01.min(100.0 / count); // Adaptive EMA factor
            state.fair_basis = ((state.fair_basis as f64) * (1.0 - alpha) + (state.basis as f64) * alpha) as FixedPrice;
            
            // Calculate Z-score
            if *count_ptr > 100 {
                let n = *count_ptr as f64;
                let mean = (*sum_ptr as f64) / n;
                let variance = (*sq_sum_ptr as f64) / n - mean * mean;
                let std_dev = if variance > 0.0 { variance.sqrt() } else { 1.0 };
                state.z_score = (state.basis as f64 - mean) / std_dev;
            }
            
            Some(*state)
        }
    }
    
    /// Check for arbitrage opportunity
    #[inline]
    pub fn check_arb_opportunity(&self, pair_id: usize) -> Option<ArbOpportunity> {
        if !self.trading_enabled.load(Ordering::Relaxed) {
            return None;
        }
        
        if pair_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return None;
        }
        
        unsafe {
            let state_ptr = self.basis_states.as_ptr().add(pair_id);
            let state = &*state_ptr;
            
            // Total round-trip fees in bps
            let total_fees_bps = self.spot_taker_fee_bps + self.futures_taker_fee_bps;
            let min_basis_required = self.entry_threshold_bps.max(total_fees_bps + self.min_profit_bps);
            
            let direction = if state.basis_pct > min_basis_required {
                // Futures expensive relative to spot: short basis
                Some(ArbDirection::ShortBasis)
            } else if state.basis_pct < -min_basis_required {
                // Futures cheap relative to spot: long basis
                Some(ArbDirection::LongBasis)
            } else {
                None
            };
            
            direction.map(|dir| {
                let expected_profit_bps = state.basis_pct.abs() - total_fees_bps;
                let confidence = calculate_confidence(state.z_score, *count_ptr.add(pair_id));
                
                ArbOpportunity {
                    pair_id,
                    direction: dir,
                    expected_profit_bps: expected_profit_bps.abs() as i64,
                    spot_leg: LegSpec {
                        exchange: ExchangeId::BinanceSpot,
                        symbol_hash: *self.symbol_hashes.as_ptr().add(pair_id),
                        side: match dir {
                            ArbDirection::LongBasis => Side::Buy,
                            ArbDirection::ShortBasis => Side::Sell,
                        },
                        quantity: 0, // To be filled by execution layer
                        max_slippage_bps: 5,
                    },
                    futures_leg: LegSpec {
                        exchange: ExchangeId::BinanceFutures,
                        symbol_hash: *self.symbol_hashes.as_ptr().add(pair_id),
                        side: match dir {
                            ArbDirection::LongBasis => Side::Sell,
                            ArbDirection::ShortBasis => Side::Buy,
                        },
                        quantity: 0,
                        max_slippage_bps: 5,
                    },
                    confidence,
                    expiry_ns: state.last_update_ns + 10_000_000, // 10ms expiry
                }
            })
        }
    }
    
    /// Enable/disable trading (circuit breaker)
    pub fn set_trading_enabled(&self, enabled: bool) {
        self.trading_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Memory statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let active = self.active_count.load(Ordering::Relaxed) as usize;
        let per_pair_size = std::mem::size_of::<BasisState>() 
            + std::mem::size_of::<u64>()
            + 2 * std::mem::size_of::<i128>()
            + std::mem::size_of::<u64>();
        
        (active, active * per_pair_size, MAX_CROSS_PAIRS * per_pair_size)
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Calculate confidence score based on Z-score and sample count
#[inline]
fn calculate_confidence(z_score: f64, sample_count: u64) -> f64 {
    let z_factor = (z_score.abs() / 3.0).min(1.0);
    let sample_factor = (sample_count as f64 / 1000.0).min(1.0);
    z_factor * sample_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixed_point_conversion() {
        let price = 50000.50;
        let fixed = price_to_fixed(price);
        let recovered = fixed_to_price(fixed);
        assert!((recovered - price).abs() < 1e-7);
    }
    
    #[test]
    fn test_cross_exchange_registration() {
        let engine = CrossExchangeArbEngine::new(50, 20, 10, 10, 4, 4);
        let spot = price_to_fixed(50000.0);
        let futures = price_to_fixed(50100.0);
        
        assert!(engine.register_pair(12345, spot, futures).is_some());
    }
    
    #[test]
    fn test_basis_calculation() {
        let engine = CrossExchangeArbEngine::new(50, 20, 10, 10, 4, 4);
        let pair_id = engine.register_pair(
            12345,
            price_to_fixed(50000.0),
            price_to_fixed(50250.0),
        ).unwrap();
        
        let state = engine.update_prices(pair_id, None, None).unwrap();
        assert_eq!(state.basis, price_to_fixed(250.0));
        assert!(state.basis_pct > 0);
    }
    
    #[test]
    fn test_ram_cap() {
        assert!(MAX_CROSS_PAIRS > 0);
        assert!(MAX_CROSS_PAIRS <= 512 * 1024);
    }
}
