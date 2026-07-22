//! Structural Time-Series Decomposition for Tick Data
//!
//! Decomposes tick streams into trend, seasonal, and irregular components
//! using lock-free contiguous arrays. Optimized for zero heap allocations
//! during the hot path with cache-line aligned structures.
//!
//! # Architecture
//! - Trend: Local linear trend model with time-varying slope
//! - Seasonal: Fourier-based seasonal component (crypto has intraday patterns)
//! - Irregular: Gaussian noise component for residual variance
//!
//! # Memory Safety
//! - All buffers pre-allocated at construction
//! - Lock-free ring buffers for streaming updates
//! - 64-byte cache line alignment for AMD Ryzen AI 5

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

/// Maximum decomposition history (bounded for 8GB RAM)
const MAX_HISTORY: usize = 131072; // 128K ticks
/// Maximum harmonic count for seasonal component
const MAX_HARMONICS: usize = 32;

/// Global heap tracker for structural models
static ST_HEAP_USAGE: AtomicUsize = AtomicUsize::new(0);
const ST_HEAP_LIMIT: usize = 64 * 1024 * 1024; // 64MB reserved

/// Cache-line padded atomic for lock-free coordination
#[repr(C, align(64))]
struct CachePaddedAtomic<T> {
    value: T,
    _padding: [u8; 64 - size_of::<T>()],
}

impl<T: Default> Default for CachePaddedAtomic<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            _padding: [0u8; 64 - size_of::<T>()],
        }
    }
}

/// Structural time-series model state
#[repr(C, align(64))]
pub struct StructuralTimeSeries {
    /// Trend level component (μ_t)
    trend_level: [f64; MAX_HISTORY],
    /// Trend slope component (ν_t)
    trend_slope: [f64; MAX_HISTORY],
    /// Seasonal component (γ_t)
    seasonal: [f64; MAX_HISTORY],
    /// Harmonic coefficients for seasonal (sin/cos pairs)
    harmonic_coeffs: [[f64; 2]; MAX_HARMONICS],
    /// Irregular/residual component (ε_t)
    irregular: [f64; MAX_HISTORY],
    /// Observation variance estimate (σ²_ε)
    obs_variance: f64,
    /// Trend variance estimate (σ²_η)
    trend_variance: f64,
    /// Slope variance estimate (σ²_ζ)
    slope_variance: f64,
    /// Seasonal frequencies (radians per tick)
    seasonal_freqs: [f64; MAX_HARMONICS],
    /// Ring buffer head index (lock-free)
    head: CachePaddedAtomic<AtomicUsize>,
    /// Ring buffer tail index (lock-free)
    tail: CachePaddedAtomic<AtomicUsize>,
    /// Active harmonic count
    num_harmonics: usize,
    /// Expected seasonal period (e.g., 1440 for daily at 1-min resolution)
    seasonal_period: usize,
    /// Initialization flag
    initialized: CachePaddedAtomic<AtomicBool>,
}

impl StructuralTimeSeries {
    /// Create a new structural time-series model
    #[inline]
    pub fn new(seasonal_period: usize, num_harmonics: usize) -> Option<Self> {
        if num_harmonics > MAX_HARMONICS || seasonal_period == 0 {
            return None;
        }
        
        let required = size_of::<Self>();
        let current = ST_HEAP_USAGE.load(Ordering::Relaxed);
        if current.checked_add(required).unwrap_or(usize::MAX) > ST_HEAP_LIMIT {
            eprintln!("[StructuralTS] Heap limit exceeded, rejecting model");
            return None;
        }
        
        ST_HEAP_USAGE.fetch_add(required, Ordering::Relaxed);
        
        let mut model = Self {
            trend_level: [0.0; MAX_HISTORY],
            trend_slope: [0.0; MAX_HISTORY],
            seasonal: [0.0; MAX_HISTORY],
            harmonic_coeffs: [[0.0, 0.0]; MAX_HARMONICS],
            irregular: [0.0; MAX_HISTORY],
            obs_variance: 1.0,
            trend_variance: 0.01,
            slope_variance: 0.001,
            seasonal_freqs: [0.0; MAX_HARMONICS],
            head: CachePaddedAtomic::default(),
            tail: CachePaddedAtomic::default(),
            num_harmonics,
            seasonal_period,
            initialized: CachePaddedAtomic::default(),
        };
        
        // Initialize seasonal frequencies
        for k in 0..num_harmonics {
            model.seasonal_freqs[k] = 2.0 * std::f64::consts::PI * (k + 1) as f64 / seasonal_period as f64;
        }
        
        Some(model)
    }
    
    /// Process a new tick observation (lock-free streaming)
    #[inline]
    pub fn update(&mut self, observation: f64) {
        let head = self.head.value.load(Ordering::Acquire);
        let idx = head % MAX_HISTORY;
        
        // Decompose observation: y_t = μ_t + γ_t + ε_t
        let (trend, seasonal, irregular) = self.decompose_step(observation, idx);
        
        // Store components
        self.trend_level[idx] = trend.0;
        self.trend_slope[idx] = trend.1;
        self.seasonal[idx] = seasonal;
        self.irregular[idx] = irregular;
        
        // Update variance estimates using Welford's online algorithm
        self.update_variance_estimates(irregular, idx);
        
        // Advance head pointer (lock-free)
        self.head.value.store(head.wrapping_add(1), Ordering::Release);
        
        // Mark as initialized after first update
        self.initialized.value.store(true, Ordering::Release);
    }
    
    /// Single-step decomposition using local linear trend + Fourier seasonal
    #[inline]
    fn decompose_step(&self, observation: f64, idx: usize) -> ((f64, f64), f64, f64) {
        let prev_idx = if idx == 0 { MAX_HISTORY - 1 } else { idx - 1 };
        
        // Previous state
        let prev_level = self.trend_level[prev_idx];
        let prev_slope = self.trend_slope[prev_idx];
        let prev_seasonal = self.seasonal[prev_idx];
        
        // Prediction step for trend (local linear trend model)
        // μ_t = μ_{t-1} + ν_{t-1} + η_t
        // ν_t = ν_{t-1} + ζ_t
        let pred_level = prev_level + prev_slope;
        let pred_slope = prev_slope;
        
        // Compute seasonal component using harmonic sum
        let mut seasonal_pred = 0.0f64;
        for k in 0..self.num_harmonics {
            let freq = self.seasonal_freqs[k];
            let [a_k, b_k] = self.harmonic_coeffs[k];
            seasonal_pred += a_k * (freq * idx as f64).sin() + b_k * (freq * idx as f64).cos();
        }
        
        // Residual after removing trend and seasonal predictions
        let residual = observation - pred_level - seasonal_pred;
        
        // Kalman-like update for level and slope
        let kalman_gain_level = self.obs_variance / (self.obs_variance + self.trend_variance);
        let kalman_gain_slope = self.slope_variance / (self.obs_variance + self.slope_variance);
        
        let new_level = pred_level + kalman_gain_level * residual;
        let new_slope = pred_slope + kalman_gain_slope * residual;
        
        // Update seasonal harmonics using recursive least squares (simplified)
        self.update_seasonal_harmonics(residual, idx);
        
        // Recompute seasonal with updated harmonics
        let mut seasonal_new = 0.0f64;
        for k in 0..self.num_harmonics {
            let freq = self.seasonal_freqs[k];
            let [a_k, b_k] = self.harmonic_coeffs[k];
            seasonal_new += a_k * (freq * idx as f64).sin() + b_k * (freq * idx as f64).cos();
        }
        
        // Final irregular component
        let irregular = observation - new_level - seasonal_new;
        
        ((new_level, new_slope), seasonal_new, irregular)
    }
    
    /// Update seasonal harmonic coefficients using RLS
    #[inline]
    fn update_seasonal_harmonics(&mut self, residual: f64, idx: usize) {
        let lambda = 0.995; // Forgetting factor for non-stationarity
        
        for k in 0..self.num_harmonics {
            let freq = self.seasonal_freqs[k];
            let x = freq * idx as f64;
            let sin_x = x.sin();
            let cos_x = x.cos();
            
            let [a_k, b_k] = self.harmonic_coeffs[k];
            let pred = a_k * sin_x + b_k * cos_x;
            let error = residual; // Simplified: attribute all residual to seasonal
            
            // LMS-style update (computationally cheaper than full RLS)
            let step_size = 0.001 * lambda;
            self.harmonic_coeffs[k][0] += step_size * error * sin_x;
            self.harmonic_coeffs[k][1] += step_size * error * cos_x;
        }
    }
    
    /// Update variance estimates using exponential moving average
    #[inline]
    fn update_variance_estimates(&mut self, irregular: f64, idx: usize) {
        let alpha = 0.01; // Smoothing parameter
        
        // Update observation variance
        let sq_err = irregular * irregular;
        self.obs_variance = (1.0 - alpha) * self.obs_variance + alpha * sq_err;
        
        // Estimate trend variance from level changes
        let prev_idx = if idx == 0 { MAX_HISTORY - 1 } else { idx - 1 };
        let level_change = self.trend_level[idx] - self.trend_level[prev_idx];
        self.trend_variance = (1.0 - alpha) * self.trend_variance + alpha * level_change * level_change;
        
        // Estimate slope variance
        let slope_change = self.trend_slope[idx] - self.trend_slope[prev_idx];
        self.slope_variance = (1.0 - alpha) * self.slope_variance + alpha * slope_change * slope_change;
    }
    
    /// Get current trend estimate (level + slope projection)
    #[inline]
    pub fn get_trend(&self, steps_ahead: usize) -> f64 {
        let head = self.head.value.load(Ordering::Acquire);
        if head == 0 {
            return 0.0;
        }
        
        let idx = (head - 1) % MAX_HISTORY;
        let level = self.trend_level[idx];
        let slope = self.trend_slope[idx];
        
        level + slope * steps_ahead as f64
    }
    
    /// Get current seasonal component
    #[inline]
    pub fn get_seasonal(&self, steps_ahead: usize) -> f64 {
        let head = self.head.value.load(Ordering::Acquire);
        if head == 0 {
            return 0.0;
        }
        
        let idx = ((head - 1) + steps_ahead) % MAX_HISTORY;
        self.seasonal[idx]
    }
    
    /// Get current irregular (residual) component
    #[inline]
    pub fn get_irregular(&self) -> f64 {
        let head = self.head.value.load(Ordering::Acquire);
        if head == 0 {
            return 0.0;
        }
        
        let idx = (head - 1) % MAX_HISTORY;
        self.irregular[idx]
    }
    
    /// Get estimated observation volatility (standard deviation)
    #[inline]
    pub fn get_volatility(&self) -> f64 {
        self.obs_variance.sqrt()
    }
    
    /// Check if model is initialized
    #[inline]
    pub fn is_initialized(&self) -> bool {
        self.initialized.value.load(Ordering::Acquire)
    }
    
    /// Get number of processed observations
    #[inline]
    pub fn observation_count(&self) -> usize {
        self.head.value.load(Ordering::Acquire)
    }
    
    /// Get heap usage for this instance
    #[inline]
    pub fn heap_usage(&self) -> usize {
        size_of::<Self>()
    }
}

impl Drop for StructuralTimeSeries {
    fn drop(&mut self) {
        ST_HEAP_USAGE.fetch_sub(size_of::<Self>(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_structural_ts_creation() {
        let ts = StructuralTimeSeries::new(1440, 8);
        assert!(ts.is_some());
    }
    
    #[test]
    fn test_decomposition() {
        let mut ts = StructuralTimeSeries::new(100, 4).unwrap();
        
        // Feed some synthetic data with trend + seasonality
        for i in 0..200 {
            let trend = 100.0 + 0.01 * i as f64;
            let seasonal = 5.0 * ((2.0 * std::f64::consts::PI * i as f64) / 50.0).sin();
            let noise = (i % 10) as f64 * 0.1;
            let obs = trend + seasonal + noise;
            
            ts.update(obs);
        }
        
        assert!(ts.is_initialized());
        assert!(ts.observation_count() > 0);
        
        // Trend should be positive (upward trend in data)
        let trend = ts.get_trend(0);
        assert!(trend > 95.0);
    }
    
    #[test]
    fn test_heap_tracking() {
        let initial = ST_HEAP_USAGE.load(Ordering::Relaxed);
        {
            let _ts = StructuralTimeSeries::new(100, 4).unwrap();
            assert!(ST_HEAP_USAGE.load(Ordering::Relaxed) > initial);
        }
        assert_eq!(ST_HEAP_USAGE.load(Ordering::Relaxed), initial);
    }
}
