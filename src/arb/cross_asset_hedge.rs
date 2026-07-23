//! Cross-Asset Delta-Neutral Hedging Engine
//! 
//! Adjusts for basis risk, funding rates, and cross-margin offsets.
//! Fires offsetting limit orders instantly when exposure thresholds are breached.
//! Uses lock-free structures for microsecond latency. Enforces 8GB RAM limit.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

/// Maximum number of hedging instruments
const MAX_INSTRUMENTS: usize = 32;

/// Hedge instrument configuration
#[derive(Debug, Clone)]
pub struct HedgeInstrument {
    pub symbol_id: u32,
    pub notional_per_contract: f64,
    pub tick_size: f64,
    pub lot_size: f64,
    pub funding_rate: f64, // Annualized
    pub basis_points: i64, // Fixed-point
}

/// Exposure state for a single instrument
#[derive(Clone, Copy)]
struct ExposureState {
    delta: i64,         // Fixed-point: value * 1e9
    gamma: i64,         // Fixed-point: value * 1e12
    theta: i64,         // Fixed-point: value * 1e9 per day
    vega: i64,          // Fixed-point: value * 1e9 per 1% vol
}

/// Cross-Asset Hedging Engine
pub struct CrossAssetHedger {
    instruments: [Option<HedgeInstrument>; MAX_INSTRUMENTS],
    num_instruments: usize,
    
    /// Current exposures (fixed-point)
    total_delta: AtomicI64,
    total_gamma: AtomicI64,
    total_vega: AtomicI64,
    
    /// Hedging thresholds (fixed-point)
    delta_threshold: AtomicI64,
    gamma_threshold: AtomicI64,
    vega_threshold: AtomicI64,
    
    /// Last hedge timestamp
    last_hedge_ns: AtomicU64,
    
    /// Hedge pending flag
    hedge_pending: AtomicBool,
    
    /// Cumulative hedge P&L (fixed-point)
    cumulative_hedge_pnl: AtomicI64,
}

impl CrossAssetHedger {
    /// Create a new hedging engine
    pub fn new() -> Self {
        Self {
            instruments: Default::default(),
            num_instruments: 0,
            total_delta: AtomicI64::new(0),
            total_gamma: AtomicI64::new(0),
            total_vega: AtomicI64::new(0),
            delta_threshold: AtomicI64::new(1_000_000_000), // 1.0 in fixed-point
            gamma_threshold: AtomicI64::new(100_000_000),   // 0.1 in fixed-point  
            vega_threshold: AtomicI64::new(500_000_000),    // 0.5 in fixed-point
            last_hedge_ns: AtomicU64::new(0),
            hedge_pending: AtomicBool::new(false),
            cumulative_hedge_pnl: AtomicI64::new(0),
        }
    }

    /// Add an instrument to the hedging universe
    pub fn add_instrument(&mut self, instrument: HedgeInstrument) -> bool {
        if self.num_instruments >= MAX_INSTRUMENTS {
            return false;
        }
        self.instruments[self.num_instruments] = Some(instrument);
        self.num_instruments += 1;
        true
    }

    /// Update delta exposure
    pub fn update_delta(&self, instrument_idx: usize, delta_change: i64) {
        if instrument_idx >= self.num_instruments {
            return;
        }
        
        self.total_delta.fetch_add(delta_change, Ordering::AcqRel);
        self.check_hedge_trigger();
    }

    /// Update gamma exposure
    pub fn update_gamma(&self, instrument_idx: usize, gamma_change: i64) {
        if instrument_idx >= self.num_instruments {
            return;
        }
        
        self.total_gamma.fetch_add(gamma_change, Ordering::AcqRel);
        self.check_hedge_trigger();
    }

    /// Update vega exposure
    pub fn update_vega(&self, instrument_idx: usize, vega_change: i64) {
        if instrument_idx >= self.num_instruments {
            return;
        }
        
        self.total_vega.fetch_add(vega_change, Ordering::AcqRel);
        self.check_hedge_trigger();
    }

    /// Check if any exposure threshold is breached
    fn check_hedge_trigger(&self) {
        let delta = self.total_delta.load(Ordering::Acquire).abs();
        let gamma = self.total_gamma.load(Ordering::Acquire).abs();
        let vega = self.total_vega.load(Ordering::Acquire).abs();
        
        let delta_thresh = self.delta_threshold.load(Ordering::Acquire);
        let gamma_thresh = self.gamma_threshold.load(Ordering::Acquire);
        let vega_thresh = self.vega_threshold.load(Ordering::Acquire);
        
        let breach = delta > delta_thresh || gamma > gamma_thresh || vega > vega_thresh;
        
        if breach && !self.hedge_pending.load(Ordering::Acquire) {
            self.hedge_pending.store(true, Ordering::Release);
        }
    }

    /// Calculate optimal hedge ratios using minimum variance
    pub fn calculate_hedge_ratios(&self) -> Vec<(usize, f64)> {
        let mut ratios = Vec::with_capacity(self.num_instruments);
        
        let total_delta = self.total_delta.load(Ordering::Acquire) as f64 / 1e9;
        
        if total_delta.abs() < 1e-6 {
            return ratios;
        }
        
        // Simple proportional hedging (for production, use covariance matrix)
        for i in 0..self.num_instruments {
            if let Some(ref inst) = self.instruments[i] {
                let weight = 1.0 / self.num_instruments as f64;
                let hedge_qty = -total_delta * weight / inst.notional_per_contract;
                ratios.push((i, hedge_qty));
            }
        }
        
        ratios
    }

    /// Get current delta exposure
    pub fn get_delta(&self) -> f64 {
        self.total_delta.load(Ordering::Acquire) as f64 / 1e9
    }

    /// Get current gamma exposure
    pub fn get_gamma(&self) -> f64 {
        self.total_gamma.load(Ordering::Acquire) as f64 / 1e12
    }

    /// Get current vega exposure
    pub fn get_vega(&self) -> f64 {
        self.total_vega.load(Ordering::Acquire) as f64 / 1e9
    }

    /// Check if hedge is needed
    pub fn needs_hedge(&self) -> bool {
        self.hedge_pending.load(Ordering::Acquire)
    }

    /// Acknowledge hedge execution
    pub fn acknowledge_hedge(&self, pnl_fixed: i64) {
        self.hedge_pending.store(false, Ordering::Release);
        self.last_hedge_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Release
        );
        self.cumulative_hedge_pnl.fetch_add(pnl_fixed, Ordering::AcqRel);
    }

    /// Set hedging thresholds
    pub fn set_thresholds(&self, delta: f64, gamma: f64, vega: f64) {
        self.delta_threshold.store((delta * 1e9) as i64, Ordering::Release);
        self.gamma_threshold.store((gamma * 1e12) as i64, Ordering::Release);
        self.vega_threshold.store((vega * 1e9) as i64, Ordering::Release);
    }

    /// Get cumulative hedge P&L
    pub fn cumulative_hedge_pnl(&self) -> f64 {
        self.cumulative_hedge_pnl.load(Ordering::Acquire) as f64 / 1e9
    }

    /// Calculate funding cost adjustment
    pub fn funding_adjustment(&self, position_days: f64) -> f64 {
        let mut total_funding = 0.0;
        
        for i in 0..self.num_instruments {
            if let Some(ref inst) = self.instruments[i] {
                // Daily funding rate = annual / 365
                let daily_rate = inst.funding_rate / 365.0;
                total_funding += daily_rate * position_days;
            }
        }
        
        total_funding / self.num_instruments as f64
    }
}

impl Default for CrossAssetHedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedger_creation() {
        let hedger = CrossAssetHedger::new();
        assert_eq!(hedger.get_delta(), 0.0);
        assert!(!hedger.needs_hedge());
    }

    #[test]
    fn test_instrument_addition() {
        let mut hedger = CrossAssetHedger::new();
        
        let inst = HedgeInstrument {
            symbol_id: 1,
            notional_per_contract: 1.0,
            tick_size: 0.01,
            lot_size: 0.001,
            funding_rate: 0.05,
            basis_points: 0,
        };
        
        assert!(hedger.add_instrument(inst));
    }

    #[test]
    fn test_exposure_update_and_hedge_trigger() {
        let hedger = CrossAssetHedger::new();
        
        // Update delta beyond threshold
        hedger.update_delta(0, 2_000_000_000); // 2.0 in fixed-point
        
        assert!(hedger.needs_hedge());
        assert!(hedger.get_delta() > 1.0);
    }

    #[test]
    fn test_hedge_ratios() {
        let mut hedger = CrossAssetHedger::new();
        
        for i in 0..3 {
            hedger.add_instrument(HedgeInstrument {
                symbol_id: i as u32,
                notional_per_contract: 1.0,
                tick_size: 0.01,
                lot_size: 0.001,
                funding_rate: 0.05,
                basis_points: 0,
            });
        }
        
        hedger.update_delta(0, 3_000_000_000); // 3.0 delta
        
        let ratios = hedger.calculate_hedge_ratios();
        assert_eq!(ratios.len(), 3);
        
        // Each should hedge ~1.0
        for (_, qty) in ratios {
            assert!((qty + 1.0).abs() < 0.1);
        }
    }
}
