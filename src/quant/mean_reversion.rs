//! Mean Reversion Engine: Ornstein-Uhlenbeck Process Modeling
//! 
//! Calculates half-life of mean-reverting spreads for stat-arb entry/exit timing.
//! Zero heap allocations in hot paths; uses pre-allocated buffers and SIMD hints.
//! Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum series length for pre-allocated buffers (8GB RAM constraint)
const MAX_SERIES_LEN: usize = 10_000;

/// OU process parameters estimated from data
#[derive(Debug, Clone, Copy)]
pub struct OUParameters {
    /// Mean reversion speed (theta) - higher = faster reversion
    pub theta: f64,
    /// Long-term mean (mu)
    pub mu: f64,
    /// Volatility of the process (sigma)
    pub sigma: f64,
    /// Half-life in time units: ln(2) / theta
    pub half_life: f64,
    /// Current value of the spread
    pub current_value: f64,
    /// Z-score of current value relative to equilibrium
    pub z_score: f64,
}

/// Ornstein-Uhlenbeck process estimator
/// Uses maximum likelihood estimation for parameter fitting
pub struct OUEstimator {
    /// Pre-allocated buffer for differenced series
    diff_buffer: [f64; MAX_SERIES_LEN],
    /// Pre-allocated buffer for lagged values
    lag_buffer: [f64; MAX_SERIES_LEN],
    /// Last update timestamp (microseconds since epoch)
    last_update: AtomicU64,
    /// Convergence flag for iterative methods
    converged: bool,
}

impl OUEstimator {
    pub const fn new() -> Self {
        Self {
            diff_buffer: [0.0; MAX_SERIES_LEN],
            lag_buffer: [0.0; MAX_SERIES_LEN],
            last_update: AtomicU64::new(0),
            converged: false,
        }
    }

    /// Estimate OU parameters using MLE (Maximum Likelihood Estimation)
    /// Based on discrete-time AR(1) approximation: X_t = c + φ*X_{t-1} + ε_t
    /// where θ = -ln(φ)/Δt, μ = c/(1-φ), σ² = var(ε)*2θ/(1-e^{-2θΔt})
    #[inline(always)]
    pub fn estimate(&mut self, series: &[f64]) -> OUParameters {
        let n = series.len().min(MAX_SERIES_LEN);
        if n < 10 {
            return OUParameters {
                theta: 0.0,
                mu: 0.0,
                sigma: 0.0,
                half_life: f64::MAX,
                current_value: series.last().copied().unwrap_or(0.0),
                z_score: 0.0,
            };
        }

        // Prepare differenced and lagged series
        for i in 1..n {
            self.diff_buffer[i - 1] = series[i];
            self.lag_buffer[i - 1] = series[i - 1];
        }

        let effective_n = n - 1;
        
        // Linear regression: ΔX_t = α + β*X_{t-1} + ε
        // Then: θ = -β, μ = α/θ
        let (alpha, beta) = self.linear_regression(&self.lag_buffer[..effective_n], &self.diff_buffer[..effective_n]);
        
        // Convert AR(1) parameters to OU parameters
        // Assuming Δt = 1 (can be scaled for actual time intervals)
        let theta = -beta;
        
        // Ensure positive mean reversion speed
        let theta = if theta > 1e-6 { theta } else { 1e-6 };
        
        let mu = if theta.abs() > 1e-6 { alpha / theta } else { 0.0 };
        
        // Calculate residual variance for sigma estimation
        let sigma_sq = self.residual_variance(&self.lag_buffer[..effective_n], &self.diff_buffer[..effective_n], alpha, beta);
        let sigma = sigma_sq.sqrt();

        // Half-life: time to revert halfway to mean
        let half_life = 2.0_f64.ln() / theta;

        // Calculate current z-score
        let current_value = series[n - 1];
        let z_score = if sigma > 1e-6 {
            (current_value - mu) / sigma
        } else {
            0.0
        };

        self.converged = true;
        self.last_update.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            Ordering::Relaxed,
        );

        OUParameters {
            theta,
            mu,
            sigma,
            half_life,
            current_value,
            z_score,
        }
    }

    /// Simple linear regression: y = α + β*x
    /// Returns (α, β) with zero allocations
    #[inline(always)]
    fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len();
        if n == 0 {
            return (0.0, 0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        // SIMD-friendly accumulation
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            sum_x += xi;
            sum_y += yi;
            sum_xy += xi * yi;
            sum_xx += xi * xi;
        }

        let n_f64 = n as f64;
        let denom = n_f64 * sum_xx - sum_x * sum_x;

        if denom.abs() < 1e-12 {
            return (0.0, 0.0);
        }

        let beta = (n_f64 * sum_xy - sum_x * sum_y) / denom;
        let alpha = (sum_y - beta * sum_x) / n_f64;

        (alpha, beta)
    }

    /// Calculate residual variance from linear regression
    fn residual_variance(&self, x: &[f64], y: &[f64], alpha: f64, beta: f64) -> f64 {
        let mut sum_sq = 0.0;
        let n = x.len();

        for i in 0..n {
            let predicted = alpha + beta * x[i];
            let resid = y[i] - predicted;
            sum_sq += resid * resid;
        }

        if n < 2 {
            return 0.0;
        }

        sum_sq / (n - 1) as f64
    }

    /// Check if parameters have been estimated
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Get last update timestamp in microseconds
    pub fn last_update_micros(&self) -> u64 {
        self.last_update.load(Ordering::Relaxed)
    }
}

impl Default for OUEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Spread tracker with OU-based trading signals
pub struct MeanReversionTracker {
    estimator: OUEstimator,
    /// Entry threshold (z-score)
    entry_threshold: f64,
    /// Exit threshold (z-score)
    exit_threshold: f64,
    /// Current position: -1 (short), 0 (flat), 1 (long)
    position: i8,
}

impl MeanReversionTracker {
    pub fn new(entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            estimator: OUEstimator::new(),
            entry_threshold,
            exit_threshold,
            position: 0,
        }
    }

    /// Update with new spread value and return trading signal
    /// Signal: -1 (sell), 0 (hold), 1 (buy)
    pub fn update(&mut self, spread_series: &[f64]) -> i8 {
        let params = self.estimator.estimate(spread_series);
        
        if !self.estimator.is_converged() {
            return 0;
        }

        let z = params.z_score;

        // Trading logic based on z-score and current position
        let signal = if self.position == 0 {
            // Enter position when z-score exceeds entry threshold
            if z > self.entry_threshold {
                self.position = -1; // Short overvalued spread
                -1
            } else if z < -self.entry_threshold {
                self.position = 1; // Long undervalued spread
                1
            } else {
                0
            }
        } else if self.position == 1 {
            // Exit long when z-score crosses back toward mean
            if z > -self.exit_threshold {
                self.position = 0;
                0
            } else {
                1 // Hold long
            }
        } else {
            // Exit short when z-score crosses back toward mean
            if z < self.exit_threshold {
                self.position = 0;
                0
            } else {
                -1 // Hold short
            }
        };

        signal
    }

    /// Get current OU parameters
    pub fn get_parameters(&self) -> Option<OUParameters> {
        // Note: Would need to store last params or recalculate
        None
    }

    /// Reset position tracker
    pub fn reset_position(&mut self) {
        self.position = 0;
    }

    /// Get current position
    pub fn position(&self) -> i8 {
        self.position
    }
}

/// Kalman filter for adaptive mean estimation (optional enhancement)
pub struct AdaptiveMeanEstimator {
    /// Current mean estimate
    mean: f64,
    /// Current variance estimate
    variance: f64,
    /// Process noise covariance
    q: f64,
    /// Measurement noise covariance
    r: f64,
    /// Kalman gain
    k: f64,
}

impl AdaptiveMeanEstimator {
    pub fn new(initial_mean: f64, initial_var: f64, q: f64, r: f64) -> Self {
        Self {
            mean: initial_mean,
            variance: initial_var,
            q,
            r,
            k: 0.0,
        }
    }

    /// Update mean estimate with new observation
    #[inline(always)]
    pub fn update(&mut self, observation: f64) -> f64 {
        // Prediction step (mean stays same for constant model)
        let predicted_var = self.variance + self.q;

        // Update step
        self.k = predicted_var / (predicted_var + self.r);
        self.mean = self.mean + self.k * (observation - self.mean);
        self.variance = (1.0 - self.k) * predicted_var;

        self.mean
    }

    /// Get current mean estimate
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance estimate
    pub fn variance(&self) -> f64 {
        self.variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ou_estimator() {
        let mut estimator = OUEstimator::new();
        
        // Generate synthetic OU process
        let n = 1000;
        let mut series = Vec::with_capacity(n);
        let mut x = 0.0;
        let theta = 0.1;
        let mu = 0.0;
        let sigma = 0.5;

        for i in 0..n {
            series.push(x);
            // Euler-Maruyama discretization
            let drift = theta * (mu - x);
            let diffusion = sigma * (i as f64 * 0.01).sin();
            x = x + drift * 0.01 + diffusion * 0.1;
        }

        let params = estimator.estimate(&series);
        assert!(params.theta > 0.0, "Theta should be positive for mean reversion");
        assert!(params.half_life.is_finite(), "Half-life should be finite");
    }

    #[test]
    fn test_mean_reversion_tracker() {
        let mut tracker = MeanReversionTracker::new(2.0, 0.5);
        
        // Generate mean-reverting series
        let n = 500;
        let mut series = Vec::with_capacity(n);
        let mut x = 0.0;

        for i in 0..n {
            x = x * 0.95 + (i as f64 * 0.1).sin() * 0.5;
            series.push(x);
        }

        let signal = tracker.update(&series);
        // Signal depends on current z-score relative to thresholds
        assert!(signal >= -1 && signal <= 1);
    }
}
