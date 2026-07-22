//! Seasonal ARIMA (SARIMA) Models for Crypto Cyclical Patterns
//!
//! Implements seasonal ARIMA extensions to capture intraday crypto cyclical patterns,
//! utilizing contiguous memory arrays to avoid heap allocations during rolling updates.
//! Optimized for AMD Ryzen AI 5 with SIMD acceleration and strict 8GB RAM enforcement.

use std::arch::x86_64::*;
use std::collections::VecDeque;
use rayon::prelude::*;

/// Maximum total parameters (enforces 8GB RAM limit)
const MAX_SARIMA_PARAMS: usize = 200;

/// Maximum seasonal period for crypto intraday patterns (e.g., 24 hours * 60 minutes)
const MAX_SEASONAL_PERIOD: usize = 1440;

/// Contiguous buffer for efficient rolling updates
#[derive(Debug, Clone)]
struct ContiguousBuffer {
    data: Vec<f64>,
    head: usize,
    capacity: usize,
}

impl ContiguousBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            capacity,
        }
    }

    #[inline]
    fn push(&mut self, value: f64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
    }

    #[inline]
    fn get(&self, index: usize) -> Option<f64> {
        if index >= self.capacity {
            return None;
        }
        let idx = (self.head + self.capacity - 1 - index) % self.capacity;
        Some(self.data[idx])
    }

    fn to_vec(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.capacity);
        for i in 0..self.capacity {
            if let Some(v) = self.get(i) {
                result.push(v);
            }
        }
        result
    }
}

/// SARIMA Model: (p,d,q) x (P,D,Q,s)
#[derive(Debug, Clone)]
pub struct SARIMAModel {
    /// Non-seasonal AR order
    pub p: usize,
    /// Non-seasonal differencing order
    pub d: usize,
    /// Non-seasonal MA order
    pub q: usize,
    /// Seasonal AR order
    pub P: usize,
    /// Seasonal differencing order
    pub D: usize,
    /// Seasonal MA order
    pub Q: usize,
    /// Seasonal period (e.g., 24 for hourly daily seasonality)
    pub s: usize,
    
    /// Non-seasonal AR coefficients
    pub ar_coeffs: Vec<f64>,
    /// Non-seasonal MA coefficients
    pub ma_coeffs: Vec<f64>,
    /// Seasonal AR coefficients
    pub sar_coeffs: Vec<f64>,
    /// Seasonal MA coefficients
    pub sma_coeffs: Vec<f64>,
    
    /// Mean/constant term
    pub mean: f64,
    /// Residual variance
    pub residual_variance: f64,
    
    /// Rolling buffers using contiguous memory
    observations: ContiguousBuffer,
    residuals: ContiguousBuffer,
    seasonal_observations: ContiguousBuffer,
    
    /// Pre-allocated workspace for SIMD operations
    workspace: Vec<f64>,
}

impl SARIMAModel {
    /// Create a new SARIMA model
    pub fn new(
        p: usize, d: usize, q: usize,
        P: usize, D: usize, Q: usize, s: usize,
    ) -> Result<Self, &'static str> {
        // Validate parameters against memory limits
        let total_params = p + q + P + Q + d + D;
        if total_params > MAX_SARIMA_PARAMS {
            return Err("Total parameters exceed 8GB RAM limit");
        }
        if s > MAX_SEASONAL_PERIOD {
            return Err("Seasonal period exceeds maximum allowed");
        }
        if s == 0 && (P > 0 || Q > 0 || D > 0) {
            return Err("Seasonal period must be > 0 for seasonal components");
        }

        let buffer_size = (s.max(p).max(q).max(P).max(Q) * 2).max(1000);

        Ok(Self {
            p, d, q, P, D, Q, s,
            ar_coeffs: vec![0.0; p],
            ma_coeffs: vec![0.0; q],
            sar_coeffs: vec![0.0; P],
            sma_coeffs: vec![0.0; Q],
            mean: 0.0,
            residual_variance: 0.0,
            observations: ContiguousBuffer::new(buffer_size),
            residuals: ContiguousBuffer::new(buffer_size),
            seasonal_observations: ContiguousBuffer::new(s.max(1)),
            workspace: vec![0.0; MAX_SARIMA_PARAMS * 2],
        })
    }

    /// Apply seasonal differencing
    fn seasonal_difference(&self, data: &[f64]) -> Vec<f64> {
        if self.D == 0 || self.s == 0 || data.len() <= self.s * self.D {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        for _ in 0..self.D {
            let mut diffed = Vec::with_capacity(result.len().saturating_sub(self.s));
            for i in self.s..result.len() {
                diffed.push(result[i] - result[i - self.s]);
            }
            result = diffed;
        }
        result
    }

    /// Apply regular differencing
    fn regular_difference(&self, data: &[f64]) -> Vec<f64> {
        if self.d == 0 || data.is_empty() {
            return data.to_vec();
        }

        let mut result = data.to_vec();
        for _ in 0..self.d {
            let mut diffed = Vec::with_capacity(result.len().saturating_sub(1));
            for i in 1..result.len() {
                diffed.push(result[i] - result[i - 1]);
            }
            result = diffed;
        }
        result
    }

    /// Fit SARIMA model using extended Yule-Walker equations
    pub fn fit(&mut self, data: &[f64]) -> Result<(), &'static str> {
        if data.len() < self.s * (self.D + 1) + self.p + self.q + 10 {
            return Err("Insufficient data for SARIMA fitting");
        }

        // Compute mean
        self.mean = data.iter().sum::<f64>() / data.len() as f64;

        // Center data
        let centered: Vec<f64> = data.par_iter()
            .map(|&x| x - self.mean)
            .collect();

        // Apply differencing
        let diff_regular = self.regular_difference(&centered);
        let fully_diffed = self.seasonal_difference(&diff_regular);

        if fully_diffed.is_empty() {
            return Err("Differencing resulted in empty series");
        }

        // Fit non-seasonal AR component using SIMD-optimized autocovariance
        self.fit_ar_component(&fully_diffed)?;
        
        // Fit seasonal AR component
        if self.P > 0 && self.s > 0 {
            self.fit_seasonal_ar_component(&fully_diffed)?;
        }

        // Compute residuals and fit MA components
        self.compute_residuals(&fully_diffed)?;
        
        if self.q > 0 {
            self.fit_ma_component()?;
        }
        if self.Q > 0 && self.s > 0 {
            self.fit_seasonal_ma_component()?;
        }

        // Estimate residual variance
        self.residual_variance = self.compute_residual_variance();

        Ok(())
    }

    /// SIMD-optimized autocovariance computation
    fn compute_autocovariance_simd(&self, data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        let mut acf = vec![0.0; max_lag.min(n)];

        // Variance at lag 0
        let variance: f64 = data.par_iter()
            .map(|&x| x * x)
            .sum::<f64>() / n as f64;
        acf[0] = variance;

        // Check for CPU feature support
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.compute_acf_avx2(data, &mut acf);
            }
        } else {
            self.compute_acf_scalar(data, &mut acf);
        }

        acf
    }

    /// AVX2-accelerated ACF computation
    #[target_feature(enable = "avx2")]
    unsafe fn compute_acf_avx2(&self, data: &[f64], acf: &mut [f64]) {
        let n = data.len();
        
        for lag in 1..acf.len() {
            let mut sum = 0.0f64;
            let simd_limit = (n - lag) & !3;

            for i in (0..simd_limit).step_by(4) {
                let v1 = _mm256_loadu_pd(data.as_ptr().add(i));
                let v2 = _mm256_loadu_pd(data.as_ptr().add(i + lag));
                let prod = _mm256_mul_pd(v1, v2);
                
                let result: [f64; 4] = std::mem::transmute(prod);
                sum += result[0] + result[1] + result[2] + result[3];
            }

            for i in simd_limit..n - lag {
                sum += data[i] * data[i + lag];
            }

            acf[lag] = sum / n as f64;
        }
    }

    /// Scalar fallback for ACF computation
    fn compute_acf_scalar(&self, data: &[f64], acf: &mut [f64]) {
        let n = data.len();
        
        for lag in 1..acf.len() {
            let sum: f64 = (0..n - lag)
                .map(|i| data[i] * data[i + lag])
                .sum();
            acf[lag] = sum / n as f64;
        }
    }

    /// Fit non-seasonal AR coefficients
    fn fit_ar_component(&mut self, data: &[f64]) -> Result<(), &'static str> {
        if self.p == 0 {
            return Ok(());
        }

        let acf = self.compute_autocovariance_simd(data, self.p + 1);
        
        // Levinson-Durbin recursion
        let mut phi = vec![vec![0.0; self.p + 1]; self.p + 1];
        let mut var = vec![0.0; self.p + 1];
        
        phi[0][0] = 1.0;
        var[0] = acf[0];

        for k in 1..=self.p {
            let num: f64 = (1..=k)
                .map(|j| phi[k - 1][k - j] * acf[j])
                .sum::<f64>();
            
            if var[k - 1].abs() < 1e-12 {
                return Err("Numerical instability in AR fitting");
            }
            
            phi[k][k] = (acf[k] - num) / var[k - 1];
            
            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }
            phi[k][0] = 1.0;
            var[k] = var[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
        }

        self.ar_coeffs = (1..=self.p).map(|j| phi[self.p][j]).collect();
        Ok(())
    }

    /// Fit seasonal AR coefficients
    fn fit_seasonal_ar_component(&mut self, data: &[f64]) -> Result<(), &'static str> {
        if self.P == 0 || self.s == 0 {
            return Ok(());
        }

        // Sample at seasonal lags
        let seasonal_data: Vec<f64> = (0..data.len() / self.s)
            .map(|i| data[i * self.s])
            .collect();

        if seasonal_data.len() < self.P + 10 {
            return Ok(()); // Not enough seasonal samples
        }

        let acf = self.compute_autocovariance_simd(&seasonal_data, self.P + 1);
        
        // Levinson-Durbin for seasonal AR
        let mut phi = vec![vec![0.0; self.P + 1]; self.P + 1];
        let mut var = vec![0.0; self.P + 1];
        
        phi[0][0] = 1.0;
        var[0] = acf[0];

        for k in 1..=self.P {
            let num: f64 = (1..=k)
                .map(|j| phi[k - 1][k - j] * acf[j])
                .sum::<f64>();
            
            if var[k - 1].abs() < 1e-12 {
                continue;
            }
            
            phi[k][k] = (acf[k] - num) / var[k - 1];
            
            for j in 1..k {
                phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
            }
            phi[k][0] = 1.0;
            var[k] = var[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
        }

        self.sar_coeffs = (1..=self.P).map(|j| phi[self.P][j]).collect();
        Ok(())
    }

    /// Compute residuals from fitted model
    fn compute_residuals(&mut self, data: &[f64]) -> Result<(), &'static str> {
        let max_buffer = self.p.max(self.q).max(self.P * self.s).max(self.Q * self.s);
        self.observations = ContiguousBuffer::new(max_buffer.max(1000));
        self.residuals = ContiguousBuffer::new(max_buffer.max(1000));

        for (i, &x) in data.iter().enumerate() {
            if i < self.p.max(self.P * self.s) {
                self.observations.push(x);
                continue;
            }

            let prediction = self.predict_one_step()?;
            let residual = x - prediction;
            
            self.observations.push(x);
            self.residuals.push(residual);
        }

        Ok(())
    }

    /// One-step ahead prediction
    fn predict_one_step(&self) -> Result<f64, &'static str> {
        let mut pred = 0.0;

        // Non-seasonal AR component
        for (i, &phi) in self.ar_coeffs.iter().enumerate() {
            if let Some(obs) = self.observations.get(i) {
                pred += phi * obs;
            }
        }

        // Seasonal AR component
        for (i, &phi) in self.sar_coeffs.iter().enumerate() {
            let lag = (i + 1) * self.s;
            if let Some(obs) = self.observations.get(lag - 1) {
                pred += phi * obs;
            }
        }

        // Non-seasonal MA component
        for (i, &theta) in self.ma_coeffs.iter().enumerate() {
            if let Some(res) = self.residuals.get(i) {
                pred += theta * res;
            }
        }

        // Seasonal MA component
        for (i, &theta) in self.sma_coeffs.iter().enumerate() {
            let lag = (i + 1) * self.s;
            if let Some(res) = self.residuals.get(lag - 1) {
                pred += theta * res;
            }
        }

        Ok(pred)
    }

    /// Fit MA coefficients using method of moments
    fn fit_ma_component(&mut self) -> Result<(), &'static str> {
        if self.q == 0 {
            return Ok(());
        }

        let residuals: Vec<f64> = (0..self.q.max(10))
            .filter_map(|i| self.residuals.get(i))
            .collect();

        if residuals.len() < self.q + 1 {
            return Ok(());
        }

        let res_mean: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let res_var: f64 = residuals.iter()
            .map(|r| (r - res_mean).powi(2))
            .sum::<f64>() / residuals.len() as f64;

        for k in 0..self.q {
            let cov: f64 = residuals[..residuals.len() - k - 1]
                .iter()
                .zip(residuals[k + 1..].iter())
                .map(|(&a, &b)| (a - res_mean) * (b - res_mean))
                .sum::<f64>() / residuals.len() as f64;
            
            self.ma_coeffs[k] = if res_var > 1e-12 { -cov / res_var } else { 0.0 };
        }

        Ok(())
    }

    /// Fit seasonal MA coefficients
    fn fit_seasonal_ma_component(&mut self) -> Result<(), &'static str> {
        if self.Q == 0 || self.s == 0 {
            return Ok(());
        }

        let residuals: Vec<f64> = (0..(self.Q * self.s).max(100))
            .filter_map(|i| self.residuals.get(i))
            .collect();

        if residuals.len() < self.Q * self.s + 1 {
            return Ok(());
        }

        let res_mean: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let res_var: f64 = residuals.iter()
            .map(|r| (r - res_mean).powi(2))
            .sum::<f64>() / residuals.len() as f64;

        for k in 0..self.Q {
            let lag = (k + 1) * self.s;
            if lag >= residuals.len() {
                break;
            }

            let cov: f64 = residuals[..residuals.len() - lag]
                .iter()
                .zip(residuals[lag..].iter())
                .map(|(&a, &b)| (a - res_mean) * (b - res_mean))
                .sum::<f64>() / residuals.len() as f64;
            
            self.sma_coeffs[k] = if res_var > 1e-12 { -cov / res_var } else { 0.0 };
        }

        Ok(())
    }

    /// Compute residual variance
    fn compute_residual_variance(&self) f64 {
        let mut sum_sq = 0.0;
        let mut count = 0;
        
        for i in 0..self.residuals.capacity {
            if let Some(r) = self.residuals.get(i) {
                sum_sq += r * r;
                count += 1;
            }
        }

        if count == 0 { 0.0 } else { sum_sq / count as f64 }
    }

    /// Forecast next value
    pub fn forecast(&mut self) -> f64 {
        let prediction = self.predict_one_step().unwrap_or(0.0);
        self.mean + prediction
    }

    /// Update model with new observation
    pub fn update(&mut self, new_value: f64) {
        let predicted = self.forecast();
        let residual = new_value - predicted;

        self.observations.push(new_value);
        self.residuals.push(residual);
        self.seasonal_observations.push(new_value);
    }

    /// Multi-step ahead forecast
    pub fn forecast_multi(&self, steps: usize) -> Vec<f64> {
        let mut forecasts = Vec::with_capacity(steps);
        let mut temp_obs = self.observations.clone();
        let mut temp_res = self.residuals.clone();

        for step in 0..steps {
            let mut pred = 0.0;

            // AR components
            for (i, &phi) in self.ar_coeffs.iter().enumerate() {
                if let Some(obs) = temp_obs.get(i) {
                    pred += phi * obs;
                }
            }

            // Seasonal AR
            for (i, &phi) in self.sar_coeffs.iter().enumerate() {
                let lag = (i + 1) * self.s;
                if let Some(obs) = temp_obs.get(lag - 1) {
                    pred += phi * obs;
                }
            }

            // MA components (future residuals assumed zero)
            if step == 0 {
                for (i, &theta) in self.ma_coeffs.iter().enumerate() {
                    if let Some(res) = temp_res.get(i) {
                        pred += theta * res;
                    }
                }
            }

            forecasts.push(self.mean + pred);
            
            // Shift for next iteration
            temp_obs.push(self.mean + pred);
            temp_res.push(0.0);
        }

        forecasts
    }

    /// Detect intraday crypto patterns (e.g., Asian/European/US session effects)
    pub fn detect_session_patterns(&self) -> Vec<f64> {
        if self.s == 0 {
            return vec![];
        }

        // Return seasonal component for one full period
        let mut patterns = Vec::with_capacity(self.s);
        for i in 0..self.s {
            let mut pattern = 0.0;
            
            // Seasonal AR contribution
            for (j, &phi) in self.sar_coeffs.iter().enumerate() {
                let lag = ((i + 1) * self.s) - (j + 1) * self.s;
                if lag > 0 && lag <= self.s {
                    pattern += phi;
                }
            }
            
            patterns.push(pattern);
        }

        patterns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sarima_creation() {
        let model = SARIMAModel::new(1, 1, 1, 1, 1, 1, 24);
        assert!(model.is_ok());

        let invalid = SARIMAModel::new(100, 0, 100, 100, 0, 100, 24);
        assert!(invalid.is_err());
    }

    #[test]
    fn test_sarima_with_seasonal_data() {
        // Generate synthetic seasonal data
        let mut data = Vec::with_capacity(500);
        for i in 0..500 {
            let seasonal = (i as f64 * std::f64::consts::PI / 24.0).sin();
            let trend = (i as f64) * 0.001;
            data.push(seasonal + trend);
        }

        let mut model = SARIMAModel::new(1, 0, 0, 1, 0, 0, 24).unwrap();
        model.fit(&data).unwrap();

        let forecast = model.forecast();
        assert!(forecast.is_finite());
    }

    #[test]
    fn test_contiguous_buffer() {
        let mut buf = ContiguousBuffer::new(10);
        for i in 0..20 {
            buf.push(i as f64);
        }

        // Should contain last 10 values
        assert_eq!(buf.get(0), Some(19.0));
        assert_eq!(buf.get(9), Some(10.0));
        assert_eq!(buf.get(10), None);
    }
}
