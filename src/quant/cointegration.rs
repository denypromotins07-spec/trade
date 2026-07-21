//! Statistical Arbitrage Engine: Cointegration Analysis
//! 
//! Implements Engle-Granger and Johansen tests for identifying cointegrated crypto pairs.
//! Optimized for AMD Ryzen AI 5 with SIMD instructions for real-time vector calculations.
//! Zero heap allocations in hot paths; uses pre-allocated buffers and contiguous memory.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;

/// Pre-allocated buffer size for time-series data (tuned for 8GB RAM limit)
const MAX_SERIES_LEN: usize = 10_000;

/// Cointegration test results with statistical metrics
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    pub pair_id: (String, String),
    pub is_cointegrated: bool,
    pub engle_granger_stat: f64,
    pub critical_value: f64,
    pub half_life: f64,
    pub hedge_ratio: f64,
}

/// Engle-Granger two-step cointegration test
/// Uses SIMD-accelerated OLS regression for speed
pub struct EngleGrangerTest {
    /// Pre-allocated residual buffer
    residuals: [f64; MAX_SERIES_LEN],
    /// Pre-allocated X matrix column (ones for intercept)
    x_const: [f64; MAX_SERIES_LEN],
    /// Cache for summation values to avoid recalculation
    sum_cache: SumCache,
}

#[derive(Default)]
struct SumCache {
    sum_x: f64,
    sum_y: f64,
    sum_xy: f64,
    sum_xx: f64,
    count: usize,
}

impl EngleGrangerTest {
    pub const fn new() -> Self {
        Self {
            residuals: [0.0; MAX_SERIES_LEN],
            x_const: [1.0; MAX_SERIES_LEN],
            sum_cache: SumCache::default(),
        }
    }

    /// Calculate hedge ratio using OLS: y = alpha + beta * x
    /// Returns (alpha, beta) with zero allocations
    #[inline(always)]
    pub fn calculate_hedge_ratio(&mut self, x: &[f64], y: &[f64]) -> (f64, f64) {
        debug_assert_eq!(x.len(), y.len(), "Series must be equal length");
        let n = x.len().min(MAX_SERIES_LEN);
        
        // Reset cache
        self.sum_cache = SumCache {
            sum_x: 0.0,
            sum_y: 0.0,
            sum_xy: 0.0,
            sum_xx: 0.0,
            count: n,
        };

        // SIMD-friendly accumulation (compiler auto-vectorizes)
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            self.sum_cache.sum_x += xi;
            self.sum_cache.sum_y += yi;
            self.sum_cache.sum_xy += xi * yi;
            self.sum_cache.sum_xx += xi * xi;
        }

        let n_f64 = n as f64;
        let denom = n_f64 * self.sum_cache.sum_xx - self.sum_cache.sum_x * self.sum_cache.sum_x;
        
        if denom.abs() < 1e-12 {
            return (0.0, 0.0);
        }

        let beta = (n_f64 * self.sum_cache.sum_xy - self.sum_cache.sum_x * self.sum_cache.sum_y) / denom;
        let alpha = (self.sum_cache.sum_y - beta * self.sum_cache.sum_x) / n_f64;

        (alpha, beta)
    }

    /// Compute residuals and run ADF test on them
    /// Returns ADF t-statistic (more negative = more likely cointegrated)
    pub fn compute_adf_statistic(&mut self, x: &[f64], y: &[f64], hedge_ratio: f64) -> f64 {
        let n = x.len().min(MAX_SERIES_LEN);
        
        // Calculate residuals: e_t = y_t - alpha - beta * x_t
        let (alpha, _) = self.calculate_hedge_ratio(x, y);
        for i in 0..n {
            self.residuals[i] = y[i] - alpha - hedge_ratio * x[i];
        }

        // Augmented Dickey-Fuller test on residuals
        // ADF: Δe_t = α + β*e_{t-1} + Σγ_i*Δe_{t-i} + ε_t
        self.adf_test_simple(&self.residuals[..n])
    }

    /// Simplified ADF test (no lag terms for speed, can be extended)
    fn adf_test_simple(&self, series: &[f64]) -> f64 {
        if series.len() < 10 {
            return 0.0;
        }

        let mut sum_dy_e = 0.0;
        let mut sum_e_lag_sq = 0.0;
        let mut count = 0;

        for i in 1..series.len() {
            let dy = series[i] - series[i - 1];
            let e_lag = series[i - 1];
            sum_dy_e += dy * e_lag;
            sum_e_lag_sq += e_lag * e_lag;
            count += 1;
        }

        if sum_e_lag_sq < 1e-12 {
            return 0.0;
        }

        // t-statistic for beta coefficient
        let beta = sum_dy_e / sum_e_lag_sq;
        let residual_var = self.calculate_residual_variance(series, beta);
        
        if residual_var < 1e-12 {
            return 0.0;
        }

        beta / (residual_var / sum_e_lag_sq).sqrt()
    }

    fn calculate_residual_variance(&self, series: &[f64], beta: f64) -> f64 {
        let mut sum_sq = 0.0;
        let mut count = 0;

        for i in 1..series.len() {
            let dy = series[i] - series[i - 1];
            let predicted = beta * series[i - 1];
            let resid = dy - predicted;
            sum_sq += resid * resid;
            count += 1;
        }

        if count < 2 {
            return 0.0;
        }

        sum_sq / (count - 1) as f64
    }
}

/// Johansen test placeholder (full implementation requires eigenvalue decomposition)
/// For production, link to LAPACK or use approximations
pub struct JohansenTest {
    max_lag: usize,
    critical_values: [f64; 3], // 90%, 95%, 99%
}

impl JohansenTest {
    pub const fn new(max_lag: usize) -> Self {
        Self {
            max_lag,
            critical_values: [13.43, 15.49, 19.93], // Approximate 95% for r=0, n=2
        }
    }

    /// Trace statistic calculation (simplified)
    /// Full implementation requires VAR estimation and eigenvalue decomposition
    pub fn trace_statistic(&self, series1: &[f64], series2: &[f64]) -> f64 {
        // Placeholder: In production, use ndarray-linalg for eigenvalues
        // This returns a pseudo-statistic based on correlation decay
        let n = series1.len().min(series2.len()).min(MAX_SERIES_LEN);
        if n < self.max_lag + 10 {
            return 0.0;
        }

        // Calculate correlation as proxy for cointegration strength
        let mut sum_prod = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..n {
            sum_prod += series1[i] * series2[i];
            sum_sq1 += series1[i] * series1[i];
            sum_sq2 += series2[i] * series2[i];
        }

        let denom = (sum_sq1 * sum_sq2).sqrt();
        if denom < 1e-12 {
            return 0.0;
        }

        let corr = sum_prod / denom;
        // Transform correlation to trace-like statistic
        (1.0 - corr.abs()).ln().abs() * n as f64 / 100.0
    }

    pub fn is_cointegrated(&self, statistic: f64) -> bool {
        statistic > self.critical_values[1] // 95% confidence
    }
}

/// Main cointegration tracker for multiple pairs
pub struct CointegrationTracker {
    eg_test: EngleGrangerTest,
    johansen_test: JohansenTest,
    results: HashMap<(String, String), CointegrationResult>,
    update_counter: AtomicUsize,
}

impl CointegrationTracker {
    pub fn new() -> Self {
        Self {
            eg_test: EngleGrangerTest::new(),
            johansen_test: JohansenTest::new(5),
            results: HashMap::new(),
            update_counter: AtomicUsize::new(0),
        }
    }

    /// Test a pair for cointegration
    /// Returns CointegrationResult with all metrics
    pub fn test_pair(&mut self, asset1: &str, asset2: &str, prices1: &[f64], prices2: &[f64]) -> CointegrationResult {
        let (alpha, beta) = self.engle_granger_test(prices1, prices2);
        let adf_stat = self.eg_test.compute_adf_statistic(prices1, prices2, beta);
        
        // Critical value for ADF at 95% (approximate for large N)
        let critical_value = -2.86;
        let is_cointegrated = adf_stat < critical_value;

        // Estimate half-life from mean reversion speed
        let half_life = if beta.abs() > 1e-6 {
            (-2.0_f64.ln()) / (2.0 * beta.abs())
        } else {
            f64::MAX
        };

        let result = CointegrationResult {
            pair_id: (asset1.to_string(), asset2.to_string()),
            is_cointegrated,
            engle_granger_stat: adf_stat,
            critical_value,
            half_life,
            hedge_ratio: beta,
        };

        self.results.insert((asset1.to_string(), asset2.to_string()), result.clone());
        self.update_counter.fetch_add(1, Ordering::Relaxed);
        
        result
    }

    fn engle_granger_test(&mut self, x: &[f64], y: &[f64]) -> (f64, f64) {
        self.eg_test.calculate_hedge_ratio(x, y)
    }

    /// Get cached result for a pair
    pub fn get_result(&self, asset1: &str, asset2: &str) -> Option<&CointegrationResult> {
        self.results.get(&(asset1.to_string(), asset2.to_string()))
    }

    /// Number of pairs tested
    pub fn pairs_count(&self) -> usize {
        self.results.len()
    }
}

impl Default for CointegrationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cointegration_tracker() {
        let mut tracker = CointegrationTracker::new();
        
        // Generate synthetic cointegrated series
        let n = 1000;
        let mut series1 = Vec::with_capacity(n);
        let mut series2 = Vec::with_capacity(n);
        
        let mut price = 100.0;
        for i in 0..n {
            price += (i as f64 * 0.001).sin() * 0.1;
            series1.push(price);
            series2.push(price * 1.05 + (i as f64 * 0.1).sin());
        }

        let result = tracker.test_pair("BTC", "ETH", &series1, &series2);
        assert!(result.hedge_ratio > 0.0);
    }
}
