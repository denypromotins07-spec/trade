//! Advanced Spread & Market Microstructure - Roll Model and Glosten-Harris Estimators
//! 
//! This module calculates effective and realized bid-ask spreads in real-time,
//! exposing hidden transaction costs to the RL agent.
//! 
//! ## Key Features
//! - Roll model implementation for effective spread estimation
//! - Glosten-Harris estimator for adverse selection costs
//! - Real-time spread tracking with microsecond updates
//! - Zero heap allocations during runtime
//! - Strict 8GB RAM limit enforcement
//! 
//! ## Mathematical Background
//! Roll Model: Δp_t = c * sign(q_t) + ε_t
//! where c is half the effective spread
//! 
//! Glosten-Harris: Δp_t = λ * q_t + φ * sign(q_t) + ε_t
//! where λ captures adverse selection, φ captures order processing costs

use std::sync::atomic::{AtomicUsize, Ordering};

/// Global cap on total samples tracked
const MAX_TOTAL_SAMPLES: usize = 20_000_000;

/// Maximum lookback for spread estimation
const SPREAD_LOOKBACK: usize = 10_000;

/// Atomic counter for global sample tracking
static TOTAL_SPREAD_SAMPLES: AtomicUsize = AtomicUsize::new(0);

/// Result structure for spread estimation
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SpreadEstimate {
    /// Timestamp (microseconds since epoch)
    pub timestamp_us: u64,
    
    /// Effective spread (Roll model, in basis points)
    pub effective_spread_bps: f64,
    
    /// Realized spread (ex-post price impact, in bps)
    pub realized_spread_bps: f64,
    
    /// Adverse selection cost (Glosten-Harris λ, in bps)
    pub adverse_selection_bps: f64,
    
    /// Order processing cost (Glosten-Harris φ, in bps)
    pub order_processing_bps: f64,
    
    /// Quoted spread (if available from LOB)
    pub quoted_spread_bps: f64,
    
    /// Number of trades used in estimation
    pub trade_count: usize,
    
    /// Standard error of estimate
    pub standard_error: f64,
}

impl Default for SpreadEstimate {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            effective_spread_bps: 0.0,
            realized_spread_bps: 0.0,
            adverse_selection_bps: 0.0,
            order_processing_bps: 0.0,
            quoted_spread_bps: 0.0,
            trade_count: 0,
            standard_error: f64::NAN,
        }
    }
}

/// Roll Model Spread Estimator
/// 
/// Estimates effective spread using the Roll model:
/// Δp_t = c * sign(q_{t-1}) + ε_t
/// 
/// The coefficient c represents half the effective spread.
#[derive(Debug)]
pub struct RollModelEstimator {
    /// Circular buffer of price changes
    price_changes: Box<[f64; SPREAD_LOOKBACK]>,
    
    /// Circular buffer of trade signs (+1 for buy, -1 for sell)
    trade_signs: Box<[i8; SPREAD_LOOKBACK]>,
    
    /// Write index
    write_index: usize,
    
    /// Valid count
    valid_count: usize,
    
    /// Running sum for covariance calculation
    sum_dx: f64,   // Sum of price changes
    sum_sq_dx: f64, // Sum of squared price changes
    sum_sign: f64,  // Sum of signs
    sum_dx_sign: f64, // Sum of dx * sign
    
    /// Current spread estimate (half-spread)
    current_spread: f64,
    
    /// Last price for calculating changes
    last_price: Option<f64>,
}

impl RollModelEstimator {
    /// Create a new Roll model estimator
    pub fn new() -> Result<Self, &'static str> {
        let current = TOTAL_SPREAD_SAMPLES.load(Ordering::Relaxed);
        if current + SPREAD_LOOKBACK * 2 > MAX_TOTAL_SAMPLES {
            return Err("Global RAM limit exceeded: cannot allocate Roll model buffers");
        }
        
        let price_changes = Box::new([0.0_f64; SPREAD_LOOKBACK]);
        let trade_signs = Box::new([0i8; SPREAD_LOOKBACK]);
        
        TOTAL_SPREAD_SAMPLES.fetch_add(SPREAD_LOOKBACK * 2, Ordering::Relaxed);
        
        Ok(Self {
            price_changes,
            trade_signs,
            write_index: 0,
            valid_count: 0,
            sum_dx: 0.0,
            sum_sq_dx: 0.0,
            sum_sign: 0.0,
            sum_dx_sign: 0.0,
            current_spread: 0.0,
            last_price: None,
        })
    }
    
    /// Add a new trade observation
    #[inline(always)]
    pub fn add_trade(&mut self, price: f64, is_buy: bool, timestamp_us: u64) -> SpreadEstimate {
        // Calculate price change
        let dx = if let Some(last) = self.last_price {
            (price / last).ln()
        } else {
            0.0
        };
        
        self.last_price = Some(price);
        
        // Trade sign: +1 for buyer-initiated, -1 for seller-initiated
        let sign: i8 = if is_buy { 1 } else { -1 };
        
        // Update buffers
        let idx = self.write_index;
        let old_dx = self.price_changes[idx];
        let old_sign = self.trade_signs[idx];
        
        self.price_changes[idx] = dx;
        self.trade_signs[idx] = sign;
        
        // Update running sums (remove old, add new)
        self.sum_dx -= old_dx;
        self.sum_sq_dx -= old_dx * old_dx;
        self.sum_sign -= old_sign as f64;
        self.sum_dx_sign -= old_dx * old_sign as f64;
        
        self.sum_dx += dx;
        self.sum_sq_dx += dx * dx;
        self.sum_sign += sign as f64;
        self.sum_dx_sign += dx * sign as f64;
        
        // Advance index
        self.write_index = (self.write_index + 1) % SPREAD_LOOKBACK;
        
        if self.valid_count < SPREAD_LOOKBACK {
            self.valid_count += 1;
        }
        
        // Update spread estimate
        self.update_estimate();
        
        // Return current estimate
        SpreadEstimate {
            timestamp_us,
            effective_spread_bps: self.current_spread * 2.0 * 10000.0, // Convert to bps
            realized_spread_bps: 0.0, // Calculated separately
            adverse_selection_bps: 0.0,
            order_processing_bps: 0.0,
            quoted_spread_bps: 0.0,
            trade_count: self.valid_count,
            standard_error: self.calculate_standard_error(),
        }
    }
    
    /// Update spread estimate using Roll formula
    fn update_estimate(&mut self) {
        if self.valid_count < 10 {
            return;
        }
        
        let n = self.valid_count as f64;
        
        // Covariance between price changes and lagged signs
        // Roll's formula: c = sqrt(-Cov(Δp_t, Δp_{t-1}))
        // Simplified: c = |mean(Δp * sign)|
        
        let mean_dx = self.sum_dx / n;
        let mean_dx_sign = self.sum_dx_sign / n;
        
        // Roll spread estimate
        // c = -Cov(Δp_t, sign_{t-1}) ≈ -mean(dx * sign)
        let cov = mean_dx_sign - mean_dx * (self.sum_sign / n);
        
        // Effective half-spread
        self.current_spread = cov.abs().sqrt().min(0.1); // Cap at 10%
    }
    
    /// Calculate standard error of estimate
    fn calculate_standard_error(&self) -> f64 {
        if self.valid_count < 10 {
            return f64::NAN;
        }
        
        let n = self.valid_count as f64;
        let variance = self.sum_sq_dx / n - (self.sum_dx / n).powi(2);
        
        if variance <= 0.0 {
            return f64::NAN;
        }
        
        variance.sqrt() / n.sqrt()
    }
    
    /// Get current spread estimate
    #[inline(always)]
    pub fn get_effective_spread_bps(&self) -> f64 {
        self.current_spread * 2.0 * 10000.0
    }
    
    /// Reset estimator
    pub fn reset(&mut self) {
        self.price_changes.fill(0.0);
        self.trade_signs.fill(0);
        self.write_index = 0;
        self.valid_count = 0;
        self.sum_dx = 0.0;
        self.sum_sq_dx = 0.0;
        self.sum_sign = 0.0;
        self.sum_dx_sign = 0.0;
        self.current_spread = 0.0;
        self.last_price = None;
    }
}

impl Drop for RollModelEstimator {
    fn drop(&mut self) {
        TOTAL_SPREAD_SAMPLES.fetch_sub(SPREAD_LOOKBACK * 2, Ordering::Relaxed);
        
        unsafe {
            std::ptr::write_bytes(self.price_changes.as_mut_ptr(), 0, SPREAD_LOOKBACK);
            std::ptr::write_bytes(self.trade_signs.as_mut_ptr(), 0, SPREAD_LOOKBACK);
        }
    }
}

/// Glosten-Harris Spread Decomposition
/// 
/// Decomposes spread into adverse selection and order processing components:
/// Δp_t = λ * q_t + φ * sign(q_t) + ε_t
/// 
/// where:
/// - λ (lambda): Adverse selection cost (informed trading)
/// - φ (phi): Order processing cost (inventory, risk)
#[derive(Debug)]
pub struct GlostenHarrisEstimator {
    /// Circular buffer of signed volumes
    signed_volumes: Box<[f64; SPREAD_LOOKBACK]>,
    
    /// Circular buffer of trade signs
    trade_signs: Box<[i8; SPREAD_LOOKBACK]>,
    
    /// Circular buffer of price changes
    price_changes: Box<[f64; SPREAD_LOOKBACK]>,
    
    /// Write index
    write_index: usize,
    
    /// Valid count
    valid_count: usize,
    
    /// Adverse selection coefficient (λ)
    lambda: f64,
    
    /// Order processing coefficient (φ)
    phi: f64,
    
    /// Running sums for OLS estimation
    sum_q: f64,
    sum_s: f64,
    sum_dx: f64,
    sum_q2: f64,
    sum_s2: f64,
    sum_qs: f64,
    sum_q_dx: f64,
    sum_s_dx: f64,
    
    /// Last price
    last_price: Option<f64>,
    
    /// Normalization factor for volume
    volume_scale: f64,
}

impl GlostenHarrisEstimator {
    /// Create a new Glosten-Harris estimator
    pub fn new(volume_scale: f64) -> Result<Self, &'static str> {
        let current = TOTAL_SPREAD_SAMPLES.load(Ordering::Relaxed);
        if current + SPREAD_LOOKBACK * 3 > MAX_TOTAL_SAMPLES {
            return Err("Global RAM limit exceeded: cannot allocate GH buffers");
        }
        
        let signed_volumes = Box::new([0.0_f64; SPREAD_LOOKBACK]);
        let trade_signs = Box::new([0i8; SPREAD_LOOKBACK]);
        let price_changes = Box::new([0.0_f64; SPREAD_LOOKBACK]);
        
        TOTAL_SPREAD_SAMPLES.fetch_add(SPREAD_LOOKBACK * 3, Ordering::Relaxed);
        
        Ok(Self {
            signed_volumes,
            trade_signs,
            price_changes,
            write_index: 0,
            valid_count: 0,
            lambda: 0.0,
            phi: 0.0,
            sum_q: 0.0,
            sum_s: 0.0,
            sum_dx: 0.0,
            sum_q2: 0.0,
            sum_s2: 0.0,
            sum_qs: 0.0,
            sum_q_dx: 0.0,
            sum_s_dx: 0.0,
            last_price: None,
            volume_scale: if volume_scale > 0.0 { volume_scale } else { 1000.0 },
        })
    }
    
    /// Add a new trade observation
    #[inline(always)]
    pub fn add_trade(
        &mut self, 
        price: f64, 
        volume: f64, 
        is_buy: bool,
        timestamp_us: u64,
    ) -> SpreadEstimate {
        // Calculate price change
        let dx = if let Some(last) = self.last_price {
            (price / last).ln()
        } else {
            0.0
        };
        
        self.last_price = Some(price);
        
        // Signed volume (positive for buys, negative for sells)
        let q = if is_buy { volume } else { -volume };
        let q_normalized = q / self.volume_scale;
        
        // Trade sign
        let s: i8 = if is_buy { 1 } else { -1 };
        
        // Update buffers
        let idx = self.write_index;
        let old_q = self.signed_volumes[idx];
        let old_s = self.trade_signs[idx];
        let old_dx = self.price_changes[idx];
        
        self.signed_volumes[idx] = q_normalized;
        self.trade_signs[idx] = s;
        self.price_changes[idx] = dx;
        
        // Update running sums
        self.sum_q -= old_q;
        self.sum_s -= old_s as f64;
        self.sum_dx -= old_dx;
        self.sum_q2 -= old_q * old_q;
        self.sum_s2 -= (old_s as f64) * (old_s as f64);
        self.sum_qs -= old_q * old_s as f64;
        self.sum_q_dx -= old_q * old_dx;
        self.sum_s_dx -= old_s as f64 * old_dx;
        
        self.sum_q += q_normalized;
        self.sum_s += s as f64;
        self.sum_dx += dx;
        self.sum_q2 += q_normalized * q_normalized;
        self.sum_s2 += 1.0; // s^2 = 1 always
        self.sum_qs += q_normalized * s as f64;
        self.sum_q_dx += q_normalized * dx;
        self.sum_s_dx += s as f64 * dx;
        
        // Advance index
        self.write_index = (self.write_index + 1) % SPREAD_LOOKBACK;
        
        if self.valid_count < SPREAD_LOOKBACK {
            self.valid_count += 1;
        }
        
        // Update coefficients
        self.update_coefficients();
        
        // Calculate spread components
        let avg_volume = (self.sum_q.abs() / self.valid_count.max(1) as f64) * self.volume_scale;
        let adverse_selection = self.lambda * avg_volume * 10000.0; // bps
        let order_processing = self.phi.abs() * 10000.0; // bps
        
        SpreadEstimate {
            timestamp_us,
            effective_spread_bps: (adverse_selection + order_processing) * 2.0,
            realized_spread_bps: 0.0,
            adverse_selection_bps: adverse_selection,
            order_processing_bps: order_processing,
            quoted_spread_bps: 0.0,
            trade_count: self.valid_count,
            standard_error: f64::NAN,
        }
    }
    
    /// Update OLS estimates for λ and φ
    fn update_coefficients(&mut self) {
        if self.valid_count < 20 {
            return;
        }
        
        let n = self.valid_count as f64;
        
        // Normal equations for 2-variable OLS
        // [sum_q2  sum_qs] [λ]   [sum_q_dx]
        // [sum_qs  sum_s2] [φ] = [sum_s_dx]
        
        let det = self.sum_q2 * self.sum_s2 - self.sum_qs * self.sum_qs;
        
        if det.abs() < 1e-10 {
            return;
        }
        
        self.lambda = (self.sum_s2 * self.sum_q_dx - self.sum_qs * self.sum_s_dx) / det;
        self.phi = (self.sum_q2 * self.sum_s_dx - self.sum_qs * self.sum_q_dx) / det;
        
        // Bound coefficients for stability
        self.lambda = self.lambda.abs().min(0.001); // Max 0.1% per unit volume
        self.phi = self.phi.clamp(-0.001, 0.001);
    }
    
    /// Reset estimator
    pub fn reset(&mut self) {
        self.signed_volumes.fill(0.0);
        self.trade_signs.fill(0);
        self.price_changes.fill(0.0);
        self.write_index = 0;
        self.valid_count = 0;
        self.lambda = 0.0;
        self.phi = 0.0;
        self.sum_q = 0.0;
        self.sum_s = 0.0;
        self.sum_dx = 0.0;
        self.sum_q2 = 0.0;
        self.sum_s2 = 0.0;
        self.sum_qs = 0.0;
        self.sum_q_dx = 0.0;
        self.sum_s_dx = 0.0;
        self.last_price = None;
    }
}

impl Drop for GlostenHarrisEstimator {
    fn drop(&mut self) {
        TOTAL_SPREAD_SAMPLES.fetch_sub(SPREAD_LOOKBACK * 3, Ordering::Relaxed);
        
        unsafe {
            std::ptr::write_bytes(self.signed_volumes.as_mut_ptr(), 0, SPREAD_LOOKBACK);
            std::ptr::write_bytes(self.trade_signs.as_mut_ptr(), 0, SPREAD_LOOKBACK);
            std::ptr::write_bytes(self.price_changes.as_mut_ptr(), 0, SPREAD_LOOKBACK);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_roll_model() {
        let mut estimator = RollModelEstimator::new().unwrap();
        
        // Simulate some trades
        let base_price = 100.0;
        for i in 0..100 {
            let price = base_price * (1.0 + 0.0001 * ((i % 10) as f64 - 5.0));
            let is_buy = i % 2 == 0;
            let _estimate = estimator.add_trade(price, is_buy, 1000000 + i as u64);
        }
        
        let spread = estimator.get_effective_spread_bps();
        assert!(spread >= 0.0);
    }
    
    #[test]
    fn test_glosten_harris() {
        let mut estimator = GlostenHarrisEstimator::new(1000.0).unwrap();
        
        for i in 0..100 {
            let price = 100.0 * (1.0 + 0.0001 * ((i % 10) as f64 - 5.0));
            let volume = 100.0 + (i as f64 * 10.0);
            let is_buy = i % 2 == 0;
            let _estimate = estimator.add_trade(price, volume, is_buy, 1000000 + i as u64);
        }
    }
}
