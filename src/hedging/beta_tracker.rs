//! Hedging - Beta Tracker
//! 
//! Implements a rolling OLS and Kalman filter beta tracker to dynamically hedge
//! spot portfolios against perpetual futures with zero-allocation memory updates.
//! Optimized for AMD Ryzen AI 5 microsecond latency in the Rust hot path.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};

/// Maximum number of assets to track
const MAX_ASSETS: usize = 100;

/// Fixed-point scale factor (10^9)
const FP_SCALE: i64 = 1_000_000_000;

/// Rolling window size for OLS
const ROLLING_WINDOW: usize = 252; // ~1 trading year of daily data

/// Asset beta tracking state
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct BetaState {
    /// Asset identifier hash
    pub asset_hash: u64,
    /// Current beta estimate (fixed-point)
    pub beta_fp: i64,
    /// Beta standard error (fixed-point)
    pub beta_se_fp: i64,
    /// R-squared of the regression (fixed-point)
    pub r_squared_fp: i64,
    /// Rolling sum of X (market returns)
    pub sum_x: i64,
    /// Rolling sum of Y (asset returns)
    pub sum_y: i64,
    /// Rolling sum of X*X
    pub sum_xx: i64,
    /// Rolling sum of X*Y
    pub sum_xy: i64,
    /// Rolling sum of Y*Y
    pub sum_yy: i64,
    /// Number of observations in window
    pub n_obs: usize,
    /// Last update timestamp
    pub last_update_ns: u64,
}

impl BetaState {
    pub const fn new(asset_hash: u64) -> Self {
        Self {
            asset_hash,
            beta_fp: FP_SCALE, // Default beta = 1.0
            beta_se_fp: 0,
            r_squared_fp: 0,
            sum_x: 0,
            sum_y: 0,
            sum_xx: 0,
            sum_xy: 0,
            sum_yy: 0,
            n_obs: 0,
            last_update_ns: 0,
        }
    }

    /// Update beta using rolling OLS (zero allocation)
    #[inline(always)]
    pub fn update_ols(&mut self, market_ret: i64, asset_ret: i64) {
        // Returns are in fixed-point nanobasis points
        
        if self.n_obs >= ROLLING_WINDOW {
            // Remove oldest observation (simplified: decay old sums)
            let decay_factor = FP_SCALE - FP_SCALE / ROLLING_WINDOW as i64;
            self.sum_x = (self.sum_x * decay_factor).wrapping_div(FP_SCALE);
            self.sum_y = (self.sum_y * decay_factor).wrapping_div(FP_SCALE);
            self.sum_xx = (self.sum_xx * decay_factor).wrapping_div(FP_SCALE);
            self.sum_xy = (self.sum_xy * decay_factor).wrapping_div(FP_SCALE);
            self.sum_yy = (self.sum_yy * decay_factor).wrapping_div(FP_SCALE);
        }

        // Add new observation
        self.sum_x = self.sum_x.wrapping_add(market_ret);
        self.sum_y = self.sum_y.wrapping_add(asset_ret);
        self.sum_xx = self.sum_xx.wrapping_add(market_ret.wrapping_mul(market_ret).wrapping_div(FP_SCALE));
        self.sum_xy = self.sum_xy.wrapping_add(market_ret.wrapping_mul(asset_ret).wrapping_div(FP_SCALE));
        self.sum_yy = self.sum_yy.wrapping_add(asset_ret.wrapping_mul(asset_ret).wrapping_div(FP_SCALE));
        
        self.n_obs = (self.n_obs + 1).min(ROLLING_WINDOW);

        // Calculate beta
        self.recalculate_beta();
    }

    /// Recalculate beta from accumulated sums
    #[inline(always)]
    fn recalculate_beta(&mut self) {
        if self.n_obs < 2 {
            return;
        }

        let n = self.n_obs as i64;
        
        // Mean calculations
        let mean_x = self.sum_x.wrapping_div(n);
        let mean_y = self.sum_y.wrapping_div(n);

        // Variance and covariance
        let var_x = self.sum_xx.wrapping_sub(mean_x.wrapping_mul(self.sum_x));
        let cov_xy = self.sum_xy.wrapping_sub(mean_x.wrapping_mul(self.sum_y));
        let var_y = self.sum_yy.wrapping_sub(mean_y.wrapping_mul(self.sum_y));

        if var_x == 0 {
            self.beta_fp = FP_SCALE;
            self.beta_se_fp = 0;
            self.r_squared_fp = 0;
            return;
        }

        // Beta = Cov(X,Y) / Var(X)
        self.beta_fp = cov_xy.wrapping_mul(FP_SCALE).wrapping_div(var_x);

        // Standard error of beta (simplified)
        let residual_var = var_y.wrapping_sub(cov_xy.wrapping_mul(cov_xy).wrapping_div(var_x));
        if residual_var > 0 && var_x > 0 {
            // SE(beta) = sqrt(residual_var / ((n-2) * var_x))
            // Simplified: use approximation
            self.beta_se_fp = residual_var
                .wrapping_mul(FP_SCALE)
                .wrapping_div(var_x)
                .wrapping_div(n.max(3) - 2);
        }

        // R-squared = (Cov(X,Y))^2 / (Var(X) * Var(Y))
        if var_y > 0 {
            self.r_squared_fp = cov_xy
                .wrapping_mul(cov_xy)
                .wrapping_mul(FP_SCALE)
                .wrapping_div(var_x)
                .wrapping_div(var_y);
        }
    }

    /// Get current beta value
    #[inline(always)]
    pub fn beta(&self) -> f64 {
        self.beta_fp as f64 / FP_SCALE as f64
    }

    /// Get beta in fixed-point
    #[inline(always)]
    pub fn beta_fp(&self) -> i64 {
        self.beta_fp
    }
}

/// Kalman filter state for adaptive beta estimation
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct KalmanBetaState {
    /// Asset identifier hash
    pub asset_hash: u64,
    /// State estimate (beta)
    pub x_hat: i64,
    /// Error covariance
    pub p: i64,
    /// Process noise covariance
    pub q: i64,
    /// Measurement noise covariance
    pub r: i64,
    /// Kalman gain
    pub k: i64,
    /// Last measurement
    pub last_z: i64,
    /// Valid flag
    pub valid: AtomicBool,
}

impl KalmanBetaState {
    pub const fn new(asset_hash: u64) -> Self {
        Self {
            asset_hash,
            x_hat: FP_SCALE, // Initial beta = 1.0
            p: FP_SCALE,
            q: 1_000_000,    // Small process noise
            r: 100_000_000,  // Measurement noise
            k: 0,
            last_z: 0,
            valid: AtomicBool::new(false),
        }
    }

    /// Predict step (state transition)
    #[inline(always)]
    pub fn predict(&mut self) {
        // State prediction: x_hat_minus = x_hat (random walk model)
        // Covariance prediction: P_minus = P + Q
        self.p = self.p.wrapping_add(self.q);
    }

    /// Update step with new measurement
    #[inline(always)]
    pub fn update(&mut self, market_ret: i64, asset_ret: i64) {
        // Measurement: z = asset_ret / market_ret (approximate beta observation)
        let z = if market_ret.abs() > 1_000_000 {
            asset_ret.wrapping_mul(FP_SCALE).wrapping_div(market_ret)
        } else {
            self.x_hat // Skip update if market ret too small
        };

        self.last_z = z;

        // Kalman gain: K = P / (P + R)
        let denom = self.p.wrapping_add(self.r);
        if denom > 0 {
            self.k = self.p.wrapping_mul(FP_SCALE).wrapping_div(denom);
        }

        // State update: x_hat = x_hat + K * (z - x_hat)
        let innovation = z.wrapping_sub(self.x_hat);
        self.x_hat = self.x_hat.wrapping_add(
            self.k.wrapping_mul(innovation).wrapping_div(FP_SCALE)
        );

        // Covariance update: P = (1 - K) * P
        self.p = (FP_SCALE - self.k)
            .wrapping_mul(self.p)
            .wrapping_div(FP_SCALE);

        self.valid.store(true, Ordering::Release);
    }

    /// Combined predict and update
    #[inline(always)]
    pub fn filter(&mut self, market_ret: i64, asset_ret: i64) -> i64 {
        self.predict();
        self.update(market_ret, asset_ret);
        self.x_hat
    }

    /// Get current beta estimate
    #[inline(always)]
    pub fn beta(&self) -> i64 {
        self.x_hat
    }
}

/// Main beta tracker managing multiple assets
#[repr(C, align(64))]
pub struct BetaTracker {
    /// OLS-based beta states
    ols_states: [Option<BetaState>; MAX_ASSETS],
    /// Kalman filter states
    kalman_states: [Option<KalmanBetaState>; MAX_ASSETS],
    /// Number of tracked assets
    asset_count: usize,
    /// Market benchmark return (latest)
    market_return: AtomicI64,
    /// Update counter
    update_count: AtomicU64,
    /// Use Kalman filter (true) or OLS (false)
    use_kalman: AtomicBool,
}

impl BetaTracker {
    pub const fn new() -> Self {
        Self {
            ols_states: [None; MAX_ASSETS],
            kalman_states: [None; MAX_ASSETS],
            asset_count: 0,
            market_return: AtomicI64::new(0),
            update_count: AtomicU64::new(0),
            use_kalman: AtomicBool::new(false),
        }
    }

    /// Add an asset to track
    #[inline(always)]
    pub fn add_asset(&mut self, asset_hash: u64) -> bool {
        if self.asset_count >= MAX_ASSETS {
            return false;
        }
        
        self.ols_states[self.asset_count] = Some(BetaState::new(asset_hash));
        self.kalman_states[self.asset_count] = Some(KalmanBetaState::new(asset_hash));
        self.asset_count += 1;
        true
    }

    /// Set market benchmark return
    #[inline(always)]
    pub fn set_market_return(&self, ret: i64) {
        self.market_return.store(ret, Ordering::Release);
    }

    /// Update beta for a specific asset
    #[inline(always)]
    pub fn update_asset(&mut self, asset_hash: u64, asset_ret: i64, timestamp_ns: u64) -> Option<i64> {
        let market_ret = self.market_return.load(Ordering::Acquire);
        
        for i in 0..self.asset_count {
            if let Some(state) = &mut self.ols_states[i] {
                if state.asset_hash == asset_hash {
                    state.update_ols(market_ret, asset_ret);
                    state.last_update_ns = timestamp_ns;
                    
                    if self.use_kalman.load(Ordering::Relaxed) {
                        if let Some(k_state) = &mut self.kalman_states[i] {
                            k_state.filter(market_ret, asset_ret);
                            return Some(k_state.beta());
                        }
                    }
                    return Some(state.beta_fp);
                }
            }
        }
        None
    }

    /// Get current beta for an asset
    #[inline(always)]
    pub fn get_beta(&self, asset_hash: u64) -> Option<i64> {
        for i in 0..self.asset_count {
            if let Some(state) = &self.ols_states[i] {
                if state.asset_hash == asset_hash {
                    if self.use_kalman.load(Ordering::Relaxed) {
                        if let Some(k_state) = &self.kalman_states[i] {
                            return Some(k_state.beta());
                        }
                    }
                    return Some(state.beta_fp);
                }
            }
        }
        None
    }

    /// Get all betas for portfolio hedging calculation
    #[inline(always)]
    pub fn get_all_betas(&self) -> impl Iterator<Item = (u64, i64)> + '_ {
        (0..self.asset_count).filter_map(move |i| {
            if let Some(state) = &self.ols_states[i] {
                let beta = if self.use_kalman.load(Ordering::Relaxed) {
                    self.kalman_states[i].map(|k| k.beta()).unwrap_or(state.beta_fp)
                } else {
                    state.beta_fp
                };
                Some((state.asset_hash, beta))
            } else {
                None
            }
        })
    }

    /// Toggle between OLS and Kalman filter
    #[inline(always)]
    pub fn set_use_kalman(&self, use_kalman: bool) {
        self.use_kalman.store(use_kalman, Ordering::Release);
    }

    /// Get update statistics
    #[inline(always)]
    pub fn stats(&self) -> (usize, u64) {
        (self.asset_count, self.update_count.load(Ordering::Relaxed))
    }
}

/// Hedge ratio calculator for delta-neutral positioning
#[repr(C, align(64))]
pub struct HedgeCalculator {
    /// Portfolio value in nanodollars
    portfolio_value_ns: AtomicI64,
    /// Current hedge ratio (fixed-point)
    hedge_ratio_fp: AtomicI64,
    /// Target delta (fixed-point, 0 = neutral)
    target_delta_fp: AtomicI64,
    /// Current delta exposure
    current_delta_fp: AtomicI64,
}

impl HedgeCalculator {
    pub const fn new() -> Self {
        Self {
            portfolio_value_ns: AtomicI64::new(0),
            hedge_ratio_fp: AtomicI64::new(FP_SCALE),
            target_delta_fp: AtomicI64::new(0),
            current_delta_fp: AtomicI64::new(0),
        }
    }

    /// Calculate hedge quantity needed for delta neutrality
    #[inline(always)]
    pub fn calc_hedge_qty(
        &self,
        asset_beta: i64,
        spot_qty: i64,
        futures_price: i64,
        spot_price: i64,
    ) -> i64 {
        // Hedge ratio = -beta * (spot_qty * spot_price) / (futures_price)
        let spot_notional = spot_qty.wrapping_mul(spot_price);
        let hedge_notional = spot_notional
            .wrapping_mul(asset_beta)
            .wrapping_div(FP_SCALE);
        
        // Negate for short hedge
        -hedge_notional.wrapping_div(futures_price)
    }

    /// Update portfolio delta
    #[inline(always)]
    pub fn update_delta(&self, delta: i64) {
        self.current_delta_fp.store(delta, Ordering::Release);
    }

    /// Check if rebalancing is needed
    #[inline(always)]
    pub fn needs_rebalance(&self, threshold_fp: i64) -> bool {
        let current = self.current_delta_fp.load(Ordering::Acquire).abs();
        let target = self.target_delta_fp.load(Ordering::Acquire).abs();
        current.wrapping_sub(target).abs() > threshold_fp
    }
}

/// Get current time in nanoseconds
#[inline(always)]
fn get_time_ns() -> u64 {
    use std::time::Instant;
    static START: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_tracker_ols() {
        let mut tracker = BetaTracker::new();
        tracker.add_asset(0xBTC);
        
        // Simulate correlated returns
        for i in 0..100 {
            let market_ret = 1_000_000 + (i as i64 * 100_000); // 0.1% + drift
            let asset_ret = market_ret * 2; // Beta ~ 2.0
            
            tracker.set_market_return(market_ret);
            tracker.update_asset(0xBTC, asset_ret, get_time_ns());
        }
        
        let beta = tracker.get_beta(0xBTC).unwrap();
        // Beta should be close to 2.0 (2_000_000_000 in fixed-point)
        assert!(beta > 1_500_000_000 && beta < 2_500_000_000);
    }

    #[test]
    fn test_kalman_filter() {
        let mut kalman = KalmanBetaState::new(0xETH);
        
        // Feed consistent measurements
        for _ in 0..50 {
            kalman.filter(1_000_000_000, 1_500_000_000); // Beta ~ 1.5
        }
        
        let beta = kalman.beta();
        // Should converge toward 1.5
        assert!(beta > 1_000_000_000 && beta < 2_000_000_000);
    }

    #[test]
    fn test_hedge_calculation() {
        let calc = HedgeCalculator::new();
        
        // Long 1 BTC spot at $100k, beta = 1.0
        let hedge_qty = calc.calc_hedge_qty(
            FP_SCALE,           // beta = 1.0
            1_000_000_000,      // 1 BTC in nanounits
            100_000_000_000,    // Futures price
            100_000_000_000,    // Spot price
        );
        
        // Should short approximately 1 BTC equivalent
        assert!(hedge_qty < 0);
        assert!(hedge_qty.abs() > 900_000_000); // Close to 1 BTC
    }
}
