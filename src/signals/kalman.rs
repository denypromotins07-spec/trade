//! Multi-Asset Kalman Filter for Dynamic Beta Hedging
//! 
//! Real-time state estimation with O(1) covariance updates using contiguous memory.
//! Pre-allocated matrices for microsecond performance on AMD Ryzen AI 5.
//! Zero heap allocations during runtime hot path.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of assets in portfolio
const MAX_ASSETS: usize = 32;

/// Maximum state dimension
const MAX_STATE_DIM: usize = MAX_ASSETS + 1; // +1 for market factor

/// Kalman filter state estimate
#[derive(Debug, Clone)]
pub struct KalmanState {
    /// State vector (betas + alpha)
    pub state: [f64; MAX_STATE_DIM],
    /// State dimension (actual used size)
    pub dim: usize,
    /// Covariance trace (uncertainty measure)
    pub uncertainty: f64,
    /// Last update timestamp (microseconds)
    pub timestamp_us: u64,
}

/// Kalman filter for single asset beta estimation
pub struct SingleAssetKalman {
    /// State: [beta, alpha]
    state: [f64; 2],
    /// Covariance matrix (2x2, stored as flat array)
    covariance: [f64; 4],
    /// Process noise covariance
    process_noise: [f64; 4],
    /// Measurement noise variance
    measurement_noise: f64,
    /// Kalman gain (temporary)
    kalman_gain: [f64; 2],
    /// Update counter
    update_count: AtomicU64,
}

impl SingleAssetKalman {
    /// Create new Kalman filter with initial parameters
    pub fn new(initial_beta: f64, initial_alpha: f64, process_var: f64, measurement_var: f64) -> Self {
        Self {
            state: [initial_beta, initial_alpha],
            covariance: [1.0, 0.0, 0.0, 1.0], // Identity
            process_noise: [process_var, 0.0, 0.0, process_var],
            measurement_noise: measurement_var,
            kalman_gain: [0.0; 2],
            update_count: AtomicU64::new(0),
        }
    }

    /// Update state with new observation (O(1) operations)
    #[inline(always)]
    pub fn update(&mut self, market_return: f64, asset_return: f64) -> KalmanState {
        // Observation model: y = H * x + v
        // where H = [market_return, 1] and x = [beta, alpha]
        
        let h = [market_return, 1.0];
        
        // Predicted measurement
        let y_pred = h[0] * self.state[0] + h[1] * self.state[1];
        
        // Innovation (measurement residual)
        let innovation = asset_return - y_pred;
        
        // Innovation covariance: S = H * P * H' + R
        let s = h[0] * (self.covariance[0] * h[0] + self.covariance[1] * h[1])
              + h[1] * (self.covariance[2] * h[0] + self.covariance[3] * h[1])
              + self.measurement_noise;
        
        if s < 1e-12 {
            return self.get_current_state();
        }
        
        // Kalman gain: K = P * H' / S
        self.kalman_gain[0] = (self.covariance[0] * h[0] + self.covariance[1] * h[1]) / s;
        self.kalman_gain[1] = (self.covariance[2] * h[0] + self.covariance[3] * h[1]) / s;
        
        // Update state: x = x + K * innovation
        self.state[0] += self.kalman_gain[0] * innovation;
        self.state[1] += self.kalman_gain[1] * innovation;
        
        // Update covariance: P = (I - K * H) * P
        let kh_00 = self.kalman_gain[0] * h[0];
        let kh_01 = self.kalman_gain[0] * h[1];
        let kh_10 = self.kalman_gain[1] * h[0];
        let kh_11 = self.kalman_gain[1] * h[1];
        
        let i_kh_00 = 1.0 - kh_00;
        let i_kh_01 = -kh_01;
        let i_kh_10 = -kh_10;
        let i_kh_11 = 1.0 - kh_11;
        
        let p00 = i_kh_00 * self.covariance[0] + i_kh_01 * self.covariance[2];
        let p01 = i_kh_00 * self.covariance[1] + i_kh_01 * self.covariance[3];
        let p10 = i_kh_10 * self.covariance[0] + i_kh_11 * self.covariance[2];
        let p11 = i_kh_10 * self.covariance[1] + i_kh_11 * self.covariance[3];
        
        self.covariance[0] = p00;
        self.covariance[1] = p01;
        self.covariance[2] = p10;
        self.covariance[3] = p11;
        
        // Ensure symmetry
        self.covariance[1] = (self.covariance[1] + self.covariance[2]) / 2.0;
        self.covariance[2] = self.covariance[1];
        
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        self.get_current_state()
    }

    /// Get current state estimate
    fn get_current_state(&self) -> KalmanState {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        KalmanState {
            state: {
                let mut s = [0.0; MAX_STATE_DIM];
                s[0] = self.state[0];
                s[1] = self.state[1];
                s
            },
            dim: 2,
            uncertainty: self.covariance[0] + self.covariance[3],
            timestamp_us: timestamp,
        }
    }

    /// Get current beta estimate
    pub fn beta(&self) -> f64 {
        self.state[0]
    }

    /// Get current alpha estimate
    pub fn alpha(&self) -> f64 {
        self.state[1]
    }

    /// Get beta standard error
    pub fn beta_std_error(&self) -> f64 {
        self.covariance[0].sqrt()
    }

    /// Number of updates performed
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Reset filter to initial state
    pub fn reset(&mut self, initial_beta: f64, initial_alpha: f64) {
        self.state = [initial_beta, initial_alpha];
        self.covariance = [1.0, 0.0, 0.0, 1.0];
        self.update_count.store(0, Ordering::Relaxed);
    }
}

/// Multi-asset Kalman filter for portfolio beta hedging
pub struct MultiAssetKalman {
    /// Individual filters for each asset
    filters: [Option<SingleAssetKalman>; MAX_ASSETS],
    /// Asset names
    asset_names: [Option<String>; MAX_ASSETS],
    /// Number of active assets
    num_assets: usize,
    /// Market factor filter
    market_filter: SingleAssetKalman,
}

impl MultiAssetKalman {
    pub fn new() -> Self {
        // Initialize array with None
        let mut filters: [Option<SingleAssetKalman>; MAX_ASSETS] = Default::default();
        let mut asset_names: [Option<String>; MAX_ASSETS] = Default::default();
        
        Self {
            filters,
            asset_names,
            num_assets: 0,
            market_filter: SingleAssetKalman::new(1.0, 0.0, 0.001, 0.01),
        }
    }

    /// Add an asset to track
    pub fn add_asset(&mut self, name: &str, initial_beta: f64) -> Option<usize> {
        if self.num_assets >= MAX_ASSETS {
            return None;
        }

        let idx = self.num_assets;
        self.filters[idx] = Some(SingleAssetKalman::new(initial_beta, 0.0, 0.0001, 0.001));
        self.asset_names[idx] = Some(name.to_string());
        self.num_assets += 1;
        
        Some(idx)
    }

    /// Update all assets with new returns
    pub fn update_all(&mut self, market_return: f64, asset_returns: &[f64]) -> Vec<KalmanState> {
        let mut states = Vec::with_capacity(self.num_assets);
        
        for i in 0..self.num_assets.min(asset_returns.len()) {
            if let Some(ref mut filter) = self.filters[i] {
                let state = filter.update(market_return, asset_returns[i]);
                states.push(state);
            }
        }
        
        states
    }

    /// Update single asset
    pub fn update_asset(&mut self, idx: usize, market_return: f64, asset_return: f64) -> Option<KalmanState> {
        if idx >= self.num_assets {
            return None;
        }
        
        self.filters[idx].as_mut().map(|f| f.update(market_return, asset_return))
    }

    /// Calculate optimal hedge ratios for portfolio
    pub fn calculate_hedge_ratios(&self, portfolio_weights: &[f64]) -> Vec<f64> {
        let mut hedge_ratios = Vec::with_capacity(self.num_assets);
        
        for i in 0..self.num_assets {
            let weight = portfolio_weights.get(i).copied().unwrap_or(0.0);
            let beta = self.filters[i].as_ref().map(|f| f.beta()).unwrap_or(1.0);
            hedge_ratios.push(weight * beta);
        }
        
        hedge_ratios
    }

    /// Get total portfolio beta
    pub fn portfolio_beta(&self, weights: &[f64]) -> f64 {
        let mut total_beta = 0.0;
        
        for i in 0..self.num_assets {
            let weight = weights.get(i).copied().unwrap_or(0.0);
            let beta = self.filters[i].as_ref().map(|f| f.beta()).unwrap_or(1.0);
            total_beta += weight * beta;
        }
        
        total_beta
    }

    /// Adjust portfolio to target beta (e.g., beta-neutral)
    pub fn calculate_adjustment(&self, current_weights: &[f64], target_beta: f64) -> Vec<f64> {
        let current_beta = self.portfolio_beta(current_weights);
        let beta_diff = target_beta - current_beta;
        
        // Simple proportional adjustment
        let mut adjustments = Vec::with_capacity(self.num_assets);
        let total_weight: f64 = current_weights.iter().take(self.num_assets).sum();
        
        for i in 0..self.num_assets {
            let weight = current_weights.get(i).copied().unwrap_or(0.0);
            let beta = self.filters[i].as_ref().map(|f| f.beta()).unwrap_or(1.0);
            
            // Adjustment proportional to weight and inverse of beta
            let adj = if beta.abs() > 1e-6 {
                (beta_diff * weight / total_weight) / beta
            } else {
                0.0
            };
            
            adjustments.push(adj);
        }
        
        adjustments
    }

    /// Get number of tracked assets
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }

    /// Get asset name by index
    pub fn asset_name(&self, idx: usize) -> Option<&str> {
        if idx < self.num_assets {
            self.asset_names[idx].as_deref()
        } else {
            None
        }
    }
}

impl Default for MultiAssetKalman {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive Kalman filter with dynamic noise estimation
pub struct AdaptiveKalman {
    base_filter: SingleAssetKalman,
    /// Rolling window for innovation analysis
    innovations: [f64; 100],
    innovation_head: usize,
    innovation_count: usize,
    /// Adaptation threshold
    adaptation_threshold: f64,
}

impl AdaptiveKalman {
    pub fn new(initial_beta: f64, adaptation_threshold: f64) -> Self {
        Self {
            base_filter: SingleAssetKalman::new(initial_beta, 0.0, 0.0001, 0.001),
            innovations: [0.0; 100],
            innovation_head: 0,
            innovation_count: 0,
            adaptation_threshold,
        }
    }

    /// Update with adaptive noise tuning
    pub fn update_adaptive(&mut self, market_return: f64, asset_return: f64) -> KalmanState {
        // Calculate innovation before update
        let h = [market_return, 1.0];
        let y_pred = h[0] * self.base_filter.state[0] + h[1] * self.base_filter.state[1];
        let innovation = asset_return - y_pred;
        
        // Store innovation
        self.innovations[self.innovation_head] = innovation;
        self.innovation_head = (self.innovation_head + 1) % 100;
        if self.innovation_count < 100 {
            self.innovation_count += 1;
        }
        
        // Check if noise characteristics have changed
        if self.innovation_count >= 20 {
            let innov_var = self.calculate_innovation_variance();
            let current_noise = self.base_filter.measurement_noise;
            
            // Adapt measurement noise if significant change detected
            if (innov_var - current_noise).abs() > self.adaptation_threshold * current_noise {
                self.base_filter.measurement_noise = innov_var.max(1e-6);
            }
        }
        
        self.base_filter.update(market_return, asset_return)
    }

    fn calculate_innovation_variance(&self) -> f64 {
        if self.innovation_count < 2 {
            return self.base_filter.measurement_noise;
        }
        
        let mean: f64 = self.innovations[..self.innovation_count].iter().sum::<f64>() 
            / self.innovation_count as f64;
        
        let variance: f64 = self.innovations[..self.innovation_count]
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.innovation_count - 1) as f64;
        
        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_asset_kalman() {
        let mut kf = SingleAssetKalman::new(1.0, 0.0, 0.0001, 0.001);
        
        // Simulate correlated returns
        for i in 0..100 {
            let market_ret = (i as f64 * 0.01).sin() * 0.02;
            let asset_ret = 1.2 * market_ret + 0.001 + (i as f64 * 0.03).cos() * 0.005;
            kf.update(market_ret, asset_ret);
        }
        
        // Beta should converge near 1.2
        let beta = kf.beta();
        assert!(beta > 1.0 && beta < 1.4, "Beta should be near 1.2, got {}", beta);
    }

    #[test]
    fn test_multi_asset_kalman() {
        let mut kf = MultiAssetKalman::new();
        
        kf.add_asset("BTC", 1.0);
        kf.add_asset("ETH", 1.2);
        kf.add_asset("SOL", 1.5);
        
        let market_ret = 0.01;
        let asset_rets = [0.012, 0.015, 0.018];
        
        let states = kf.update_all(market_ret, &asset_rets);
        assert_eq!(states.len(), 3);
        
        let portfolio_beta = kf.portfolio_beta(&[0.4, 0.4, 0.2]);
        assert!(portfolio_beta > 0.0);
    }
}
