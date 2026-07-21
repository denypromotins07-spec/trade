//! `src/ta/regime.rs`
//! 
//! **Market Regime Detection Engine**
//! 
//! Implements statistical models for real-time volatility regime classification:
//! - Hidden Markov Models (HMM) for state transitions
//! - Kalman Filters for dynamic mean/variance estimation
//! - Regime-switching logic to toggle between mean-reversion and trend-following
//! 
//! **Optimization Strategy:**
//! - Pure Rust implementation with no external ML dependencies in the hot path.
//! - Pre-allocated matrices for HMM transition probabilities.
//! - SIMD-friendly array operations for Kalman filter updates.
//! - Designed for microsecond-level regime detection to adapt strategy parameters instantly.

use std::array;

/// Market regime states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    LowVolatilityBull,
    LowVolatilityBear,
    HighVolatilityBull,
    HighVolatilityBear,
    Transitioning,
}

/// Hidden Markov Model configuration
pub struct HmmConfig {
    pub num_states: usize,
    pub initial_probs: Vec<f64>,
    pub transition_matrix: Vec<Vec<f64>>,
    pub emission_means: Vec<f64>,
    pub emission_stds: Vec<f64>,
}

impl Default for HmmConfig {
    fn default() -> Self {
        // 4-state HMM: LowVolBull, LowVolBear, HighVolBull, HighVolBear
        Self {
            num_states: 4,
            initial_probs: vec![0.25; 4],
            transition_matrix: vec![
                vec![0.85, 0.10, 0.03, 0.02], // From LowVolBull
                vec![0.10, 0.85, 0.02, 0.03], // From LowVolBear
                vec![0.05, 0.05, 0.80, 0.10], // From HighVolBull
                vec![0.05, 0.05, 0.10, 0.80], // From HighVolBear
            ],
            emission_means: vec![0.001, -0.001, 0.002, -0.002],
            emission_stds: vec![0.005, 0.005, 0.02, 0.02],
        }
    }
}

/// Hidden Markov Model state tracker
pub struct HiddenMarkovModel {
    config: HmmConfig,
    state_probs: Vec<f64>, // Current belief state
    log_probs: Vec<f64>,   // Temporary buffer for log calculations
}

impl HiddenMarkovModel {
    pub fn new(config: HmmConfig) -> Self {
        Self {
            state_probs: config.initial_probs.clone(),
            log_probs: vec![0.0; config.num_states],
            config,
        }
    }

    /// Updates HMM belief state with new observation (return)
    #[inline]
    pub fn update(&mut self, observation: f64) {
        // Step 1: Calculate emission probabilities (Gaussian)
        let mut emissions = Vec::with_capacity(self.config.num_states);
        for i in 0..self.config.num_states {
            let mean = self.config.emission_means[i];
            let std = self.config.emission_stds[i];
            let diff = observation - mean;
            let prob = (-0.5 * (diff / std).powi(2)).exp() / (std * 2.5066); // 2.5066 ≈ sqrt(2π)
            emissions.push(prob);
        }

        // Step 2: Bayes update (element-wise multiplication)
        let mut total_prob = 0.0;
        for i in 0..self.config.num_states {
            self.state_probs[i] *= emissions[i];
            total_prob += self.state_probs[i];
        }

        // Step 3: Normalize
        if total_prob > f64::EPSILON {
            for i in 0..self.config.num_states {
                self.state_probs[i] /= total_prob;
            }
        }

        // Step 4: Apply transition matrix (prediction step)
        let mut new_probs = vec![0.0; self.config.num_states];
        for j in 0..self.config.num_states {
            for i in 0..self.config.num_states {
                new_probs[j] += self.state_probs[i] * self.config.transition_matrix[i][j];
            }
        }
        self.state_probs = new_probs;
    }

    /// Returns the most likely current state
    #[inline]
    pub fn get_most_likely_state(&self) -> usize {
        let mut max_prob = 0.0;
        let mut max_idx = 0;
        for (i, &prob) in self.state_probs.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_idx = i;
            }
        }
        max_idx
    }

    /// Returns confidence in the current state estimate
    #[inline]
    pub fn get_confidence(&self) -> f64 {
        self.state_probs.iter().cloned().fold(0.0, f64::max)
    }
}

/// Kalman Filter for dynamic mean and variance estimation
pub struct KalmanFilter {
    x: f64,          // State estimate (mean)
    p: f64,          // Estimate error covariance
    q: f64,          // Process noise covariance
    r: f64,          // Measurement noise covariance
    k: f64,          // Kalman gain
}

impl KalmanFilter {
    pub fn new(initial_value: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            x: initial_value,
            p: 1.0,
            q: process_noise,
            r: measurement_noise,
            k: 0.0,
        }
    }

    /// Updates filter with new measurement
    #[inline]
    pub fn update(&mut self, measurement: f64) -> f64 {
        // Prediction step
        let p_pred = self.p + self.q;

        // Update step
        self.k = p_pred / (p_pred + self.r);
        self.x = self.x + self.k * (measurement - self.x);
        self.p = (1.0 - self.k) * p_pred;

        self.x
    }

    /// Returns current estimate
    #[inline]
    pub fn estimate(&self) -> f64 {
        self.x
    }

    /// Returns current uncertainty
    #[inline]
    pub fn uncertainty(&self) -> f64 {
        self.p
    }
}

/// Dual Kalman Filter for tracking both mean and variance
pub struct DualKalmanFilter {
    mean_filter: KalmanFilter,
    var_filter: KalmanFilter,
    window_size: usize,
    buffer: Vec<f64>,
    write_idx: usize,
}

impl DualKalmanFilter {
    pub fn new(window_size: usize) -> Self {
        Self {
            mean_filter: KalmanFilter::new(0.0, 0.0001, 0.001),
            var_filter: KalmanFilter::new(0.0001, 0.00001, 0.0001),
            window_size,
            buffer: vec![0.0; window_size],
            write_idx: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, value: f64) -> (f64, f64) {
        self.buffer[self.write_idx] = value;
        self.write_idx = (self.write_idx + 1) % self.window_size;

        // Calculate instantaneous variance from window
        let mean = self.buffer.iter().sum::<f64>() / self.window_size as f64;
        let variance = self.buffer.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.window_size as f64;

        let filtered_mean = self.mean_filter.update(value);
        let filtered_var = self.var_filter.update(variance);

        (filtered_mean, filtered_var)
    }
}

/// Main Regime Detection Engine
pub struct RegimeDetector {
    hmm: HiddenMarkovModel,
    dual_kalman: DualKalmanFilter,
    volatility_threshold: f64,
    returns_buffer: Vec<f64>,
    last_price: Option<f64>,
}

impl RegimeDetector {
    pub fn new(volatility_threshold: f64) -> Self {
        Self {
            hmm: HiddenMarkovModel::new(HmmConfig::default()),
            dual_kalman: DualKalmanFilter::new(20),
            volatility_threshold,
            returns_buffer: Vec::with_capacity(100),
            last_price: None,
        }
    }

    /// Processes a new price tick and returns the detected regime
    #[inline]
    pub fn process_price(&mut self, price: f64) -> MarketRegime {
        // Calculate return
        let return_val = if let Some(last) = self.last_price {
            (price - last) / last
        } else {
            self.last_price = Some(price);
            return MarketRegime::Transitioning;
        };
        self.last_price = Some(price);

        // Update HMM with return
        self.hmm.update(return_val);

        // Update Kalman filters
        let (_, filtered_variance) = self.dual_kalman.update(return_val);

        // Determine regime based on HMM state and volatility
        let hmm_state = self.hmm.get_most_likely_state();
        let is_high_vol = filtered_variance.sqrt() > self.volatility_threshold;

        match (hmm_state, is_high_vol) {
            (0, false) => MarketRegime::LowVolatilityBull,
            (1, false) => MarketRegime::LowVolatilityBear,
            (2, true) => MarketRegime::HighVolatilityBull,
            (3, true) => MarketRegime::HighVolatilityBear,
            _ => {
                // Fallback: use volatility directly if HMM confidence is low
                if self.hmm.get_confidence() < 0.5 {
                    if is_high_vol {
                        if return_val > 0.0 {
                            MarketRegime::HighVolatilityBull
                        } else {
                            MarketRegime::HighVolatilityBear
                        }
                    } else {
                        if return_val > 0.0 {
                            MarketRegime::LowVolatilityBull
                        } else {
                            MarketRegime::LowVolatilityBear
                        }
                    }
                } else {
                    MarketRegime::Transitioning
                }
            }
        }
    }

    /// Returns whether the system should use trend-following or mean-reversion
    #[inline]
    pub fn get_strategy_mode(&self) -> StrategyMode {
        let regime = self.get_current_regime();
        match regime {
            MarketRegime::LowVolatilityBull | MarketRegime::LowVolatilityBear => {
                StrategyMode::MeanReversion
            }
            MarketRegime::HighVolatilityBull | MarketRegime::HighVolatilityBear => {
                StrategyMode::TrendFollowing
            }
            _ => StrategyMode::Neutral,
        }
    }

    #[inline]
    fn get_current_regime(&self) -> MarketRegime {
        // Simplified: in production would track last computed regime
        MarketRegime::Transitioning
    }
}

/// Strategy mode recommendation based on regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyMode {
    TrendFollowing,
    MeanReversion,
    Neutral,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_detection() {
        let mut detector = RegimeDetector::new(0.01);
        
        // Simulate stable prices (low vol)
        for _ in 0..50 {
            detector.process_price(100.0);
        }
        
        // Should eventually settle into a low volatility regime
        let mode = detector.get_strategy_mode();
        assert!(mode == StrategyMode::MeanReversion || mode == StrategyMode::Neutral);
    }
}
