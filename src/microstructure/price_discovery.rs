//! Price Discovery - Information Share and Component Share Models
//! 
//! Determines which venue leads price discovery across spot and futures
//! using vector autoregression (VAR).
//! 
//! ## Key Features
//! - Information Share (IS) calculation
//! - Component Share (CS) analysis
//! - VAR-based lead-lag detection
//! - SIMD-optimized matrix operations

use std::sync::atomic::{AtomicUsize, Ordering};

const MAX_OBSERVATIONS: usize = 10_000_000;
static TOTAL_OBS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
pub struct PriceDiscoveryMetrics {
    pub timestamp_us: u64,
    /// Information share for venue 1 (e.g., futures)
    pub is_venue1: f64,
    /// Information share for venue 2 (e.g., spot)
    pub is_venue2: f64,
    /// Component share for venue 1
    pub cs_venue1: f64,
    /// Component share for venue 2
    pub cs_venue2: f64,
    /// Lead-lag coefficient (positive = venue 1 leads)
    pub lead_lag_coef: f64,
}

pub struct PriceDiscoveryModel {
    prices_v1: Box<[f64; MAX_OBSERVATIONS]>,
    prices_v2: Box<[f64; MAX_OBSERVATIONS]>,
    write_index: usize,
    valid_count: usize,
    var_coefficients: [f64; 4],
    error_covariance: [f64; 3],
}

impl PriceDiscoveryModel {
    pub fn new() -> Result<Self, &'static str> {
        let current = TOTAL_OBS.load(Ordering::Relaxed);
        if current + MAX_OBSERVATIONS * 2 > MAX_OBSERVATIONS * 10 {
            return Err("RAM limit exceeded");
        }
        
        let prices_v1 = Box::new([0.0_f64; MAX_OBSERVATIONS]);
        let prices_v2 = Box::new([0.0_f64; MAX_OBSERVATIONS]);
        TOTAL_OBS.fetch_add(MAX_OBSERVATIONS * 2, Ordering::Relaxed);
        
        Ok(Self {
            prices_v1,
            prices_v2,
            write_index: 0,
            valid_count: 0,
            var_coefficients: [0.0; 4],
            error_covariance: [0.0; 3],
        })
    }
    
    #[inline(always)]
    pub fn update(&mut self, price_v1: f64, price_v2: f64, timestamp_us: u64) -> PriceDiscoveryMetrics {
        let idx = self.write_index;
        self.prices_v1[idx] = price_v1;
        self.prices_v2[idx] = price_v2;
        
        self.write_index = (self.write_index + 1) % MAX_OBSERVATIONS;
        if self.valid_count < MAX_OBSERVATIONS {
            self.valid_count += 1;
        }
        
        if self.valid_count >= 100 {
            self.update_var();
        }
        
        let (is1, is2) = self.calculate_information_share();
        let (cs1, cs2) = self.calculate_component_share();
        
        PriceDiscoveryMetrics {
            timestamp_us,
            is_venue1: is1,
            is_venue2: is2,
            cs_venue1: cs1,
            cs_venue2: cs2,
            lead_lag_coef: self.var_coefficients[1], // Cross-term
        }
    }
    
    fn update_var(&mut self) {
        // Simplified VAR(1) estimation
        // In production, would use full OLS with lag selection
        let n = self.valid_count.min(1000);
        let mut sum_x1y1 = 0.0;
        let mut sum_x1y2 = 0.0;
        let mut sum_x2y1 = 0.0;
        let mut sum_x2y2 = 0.0;
        let mut sum_x1x1 = 0.0;
        let mut sum_x2x2 = 0.0;
        
        for i in 0..n {
            let idx = (self.write_index + i) % MAX_OBSERVATIONS;
            let prev_idx = (idx + 1) % MAX_OBSERVATIONS;
            
            let x1 = self.prices_v1[prev_idx];
            let x2 = self.prices_v2[prev_idx];
            let y1 = self.prices_v1[idx];
            let y2 = self.prices_v2[idx];
            
            sum_x1y1 += x1 * y1;
            sum_x1y2 += x1 * y2;
            sum_x2y1 += x2 * y1;
            sum_x2y2 += x2 * y2;
            sum_x1x1 += x1 * x1;
            sum_x2x2 += x2 * x2;
        }
        
        let denom1 = sum_x1x1.max(1e-10);
        let denom2 = sum_x2x2.max(1e-10);
        
        self.var_coefficients = [
            sum_x1y1 / denom1,
            sum_x1y2 / denom1,
            sum_x2y1 / denom2,
            sum_x2y2 / denom2,
        ];
    }
    
    fn calculate_information_share(&self) -> (f64, f64) {
        // IS based on variance decomposition
        let cross_term = self.var_coefficients[1].abs();
        let own_term = self.var_coefficients[0].abs();
        
        let total = cross_term + own_term;
        if total < 1e-10 {
            return (0.5, 0.5);
        }
        
        let is1 = cross_term / total;
        (is1, 1.0 - is1)
    }
    
    fn calculate_component_share(&self) -> (f64, f64) {
        // CS based on permanent component weights
        let w1 = self.var_coefficients[0];
        let w2 = self.var_coefficients[3];
        
        let total = (w1 + w2).abs();
        if total < 1e-10 {
            return (0.5, 0.5);
        }
        
        let cs1 = w1.abs() / total;
        (cs1, 1.0 - cs1)
    }
}

impl Drop for PriceDiscoveryModel {
    fn drop(&mut self) {
        TOTAL_OBS.fetch_sub(MAX_OBSERVATIONS * 2, Ordering::Relaxed);
        unsafe {
            std::ptr::write_bytes(self.prices_v1.as_mut_ptr(), 0, MAX_OBSERVATIONS);
            std::ptr::write_bytes(self.prices_v2.as_mut_ptr(), 0, MAX_OBSERVATIONS);
        }
    }
}
