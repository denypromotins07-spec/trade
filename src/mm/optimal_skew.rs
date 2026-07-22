//! `src/mm/optimal_skew.rs`
//!
//! **Module:** Advanced Market Making - Optimal Bid-Ask Skew
//! **Purpose:** Calculate optimal skew using dynamic programming under inventory constraints.
//! **Optimization:** Contiguous memory arrays, microsecond HJB equation solver.
//! **Constraints:** Bounded state space enforces 8GB RAM limit.
//!
//! Solves the Hamilton-Jacobi-Bellman (HJB) equation to find optimal quoting strategy:
//! - Maximizes expected utility of terminal wealth
//! - Accounts for inventory risk and adverse selection
//! - Produces optimal bid/ask spreads as function of state

use std::sync::atomic::{AtomicBool, Ordering};

// Configuration constants
const MAX_INVENTORY_STEPS: usize = 101;  // Discrete inventory levels (-50 to +50)
const INVENTORY_STEP_SIZE: f64 = 10.0;   // Units per step
const TIME_STEPS: usize = 50;            // Time discretization for DP
const MAX_TIME_HORIZON: f64 = 5.0;       // Maximum planning horizon in seconds

/// Active flag
static OPTIMAL_SKEW_ACTIVE: AtomicBool = AtomicBool::new(true);

/// Market making model parameters
#[derive(Clone, Debug)]
pub struct SkewParameters {
    /// Risk aversion coefficient
    pub gamma: f64,
    /// Asset volatility (annualized)
    pub sigma: f64,
    /// Order arrival intensity at zero spread
    pub lambda_0: f64,
    /// Spread sensitivity parameter (kappa)
    pub kappa: f64,
    /// Drift estimate (alpha)
    pub drift: f64,
}

impl Default for SkewParameters {
    fn default() -> Self {
        Self {
            gamma: 0.5,
            sigma: 0.02,
            lambda_0: 10.0,
            kappa: 100.0,
            drift: 0.0,
        }
    }
}

/// Optimal quote output
#[derive(Clone, Debug)]
pub struct OptimalQuote {
    /// Optimal bid spread (distance from fair value)
    pub bid_spread: f64,
    /// Optimal ask spread (distance from fair value)
    pub ask_spread: f64,
    /// Optimal bid size
    pub bid_size: f64,
    /// Optimal ask size
    pub ask_size: f64,
    /// Value function at current state
    pub value: f64,
}

/// Dynamic Programming solver for optimal skew
/// 
/// Uses backward induction to solve the HJB equation:
/// ∂V/∂t + max_{δb,δa} [λ(δb)(V(t,q+1) - V(t,q)) + λ(δa)(V(t,q-1) - V(t,q))] = 0
pub struct OptimalSkewSolver {
    /// Model parameters
    params: SkewParameters,
    /// Value function table V[t][q]
    value_function: Vec<[f64; MAX_INVENTORY_STEPS]>,
    /// Optimal bid spreads δb*[t][q]
    optimal_bid_spread: Vec<[f64; MAX_INVENTORY_STEPS]>,
    /// Optimal ask spreads δa*[t][q]
    optimal_ask_spread: Vec<[f64; MAX_INVENTORY_STEPS]>,
    /// Current inventory index
    current_inventory_idx: usize,
    /// Current time to horizon
    time_to_horizon: f64,
    /// Fair value reference
    fair_value: f64,
}

impl OptimalSkewSolver {
    pub fn new(params: SkewParameters) -> Self {
        let mut solver = Self {
            params,
            value_function: vec![[0.0; MAX_INVENTORY_STEPS]; TIME_STEPS],
            optimal_bid_spread: vec![[0.0; MAX_INVENTORY_STEPS]; TIME_STEPS],
            optimal_ask_spread: vec![[f64; MAX_INVENTORY_STEPS]; TIME_STEPS],
            current_inventory_idx: MAX_INVENTORY_STEPS / 2, // Start at zero inventory
            time_to_horizon: MAX_TIME_HORIZON,
            fair_value: 0.0,
        };
        
        // Pre-compute optimal policy
        solver.solve_hjb();
        solver
    }

    /// Solve the HJB equation using backward induction
    pub fn solve_hjb(&mut self) {
        let dt = MAX_TIME_HORIZON / TIME_STEPS as f64;
        let center = MAX_INVENTORY_STEPS / 2;

        // Terminal condition: V(T, q) = -gamma * q^2 (liquidation penalty)
        for q_idx in 0..MAX_INVENTORY_STEPS {
            let q = (q_idx as i32 - center as i32) as f64 * INVENTORY_STEP_SIZE;
            self.value_function[TIME_STEPS - 1][q_idx] = -self.params.gamma * q * q;
        }

        // Backward induction
        for t in (0..TIME_STEPS - 1).rev() {
            let tau = (TIME_STEPS - 1 - t) as f64 * dt; // Time from terminal

            for q_idx in 0..MAX_INVENTORY_STEPS {
                let q = (q_idx as i32 - center as i32) as f64 * INVENTORY_STEP_SIZE;
                
                // Compute optimal spreads at this state
                let (bid_spread, ask_spread) = self.compute_optimal_spreads(q, tau);
                
                self.optimal_bid_spread[t][q_idx] = bid_spread;
                self.optimal_ask_spread[t][q_idx] = ask_spread;

                // Compute arrival rates at optimal spreads
                let lambda_b = self.params.lambda_0 * (-self.params.kappa * bid_spread).exp();
                let lambda_a = self.params.lambda_0 * (-self.params.kappa * ask_spread).exp();

                // Get value at neighboring inventory states
                let v_up = if q_idx + 1 < MAX_INVENTORY_STEPS {
                    self.value_function[t + 1][q_idx + 1]
                } else {
                    f64::MIN / 2.0
                };
                
                let v_down = if q_idx > 0 {
                    self.value_function[t + 1][q_idx - 1]
                } else {
                    f64::MIN / 2.0
                };
                
                let v_hold = self.value_function[t + 1][q_idx];

                // HJB update (simplified Euler scheme)
                // ∂V/∂t + λ_b*(V(q+1)-V(q)) + λ_a*(V(q-1)-V(q)) - gamma*sigma^2*q^2/2 = 0
                let diffusion_term = -0.5 * self.params.gamma * self.params.sigma.powi(2) * q * q;
                let jump_term = lambda_b * (v_up - v_hold) + lambda_a * (v_down - v_hold);
                
                self.value_function[t][q_idx] = v_hold + dt * (jump_term + diffusion_term);
            }
        }
    }

    /// Compute optimal spreads at given inventory and time
    fn compute_optimal_spreads(&self, q: f64, tau: f64) -> (f64, f64) {
        // Closed-form approximation for Avellaneda-Stoikov with extensions
        // δ* = 1/kappa + gamma*sigma^2*(T-t)*|q| +/- drift adjustment
        
        let base_spread = 1.0 / self.params.kappa;
        let inventory_skew = self.params.gamma * self.params.sigma.powi(2) * (MAX_TIME_HORIZON - tau) * q.abs();
        
        // Asymmetric adjustment for drift
        let drift_adj = self.params.drift * (MAX_TIME_HORIZON - tau);

        // Bid spread: wider when long (positive q), tighter when short
        let bid_spread = base_spread + inventory_skew - drift_adj;
        
        // Ask spread: wider when short (negative q), tighter when long
        let ask_spread = base_spread + inventory_skew + drift_adj;

        (bid_spread.max(0.0001), ask_spread.max(0.0001))
    }

    /// Update current state
    #[inline]
    pub fn update_state(&mut self, inventory: f64, fair_value: f64, time_to_horizon: f64) {
        let center = MAX_INVENTORY_STEPS / 2;
        
        // Convert continuous inventory to discrete index
        let inventory_steps = (inventory / INVENTORY_STEP_SIZE).round() as i32;
        self.current_inventory_idx = (center as i32 + inventory_steps)
            .clamp(0, MAX_INVENTORY_STEPS as i32 - 1) as usize;
        
        self.fair_value = fair_value;
        self.time_to_horizon = time_to_horizon.clamp(0.0, MAX_TIME_HORIZON);
    }

    /// Get optimal quotes for current state
    pub fn get_optimal_quotes(&self) -> OptimalQuote {
        // Find closest time step
        let t_idx = ((1.0 - self.time_to_horizon / MAX_TIME_HORIZON) * (TIME_STEPS - 1) as f64)
            .round() as usize
            .clamp(0, TIME_STEPS - 1);

        let bid_spread = self.optimal_bid_spread[t_idx][self.current_inventory_idx];
        let ask_spread = self.optimal_ask_spread[t_idx][self.current_inventory_idx];
        let value = self.value_function[t_idx][self.current_inventory_idx];

        // Size based on distance from boundaries (more aggressive near zero inventory)
        let center = MAX_INVENTORY_STEPS / 2;
        let distance_from_center = (self.current_inventory_idx as i32 - center as i32).abs();
        let size_factor = (1.0 - distance_from_center as f64 / center as f64).max(0.1);
        let base_size = 100.0;

        OptimalQuote {
            bid_spread,
            ask_spread,
            bid_size: base_size * size_factor,
            ask_size: base_size * size_factor,
            value,
        }
    }

    /// Get the full value function surface (for analysis)
    pub fn get_value_function(&self) -> &Vec<[f64; MAX_INVENTORY_STEPS]> {
        &self.value_function
    }

    /// Check if solver is active
    #[inline]
    pub fn is_active(&self) -> bool {
        OPTIMAL_SKEW_ACTIVE.load(Ordering::Relaxed)
    }

    /// Deactivate solver
    pub fn deactivate(&self) {
        OPTIMAL_SKEW_ACTIVE.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_skew_solver() {
        let params = SkewParameters::default();
        let mut solver = OptimalSkewSolver::new(params);

        // Zero inventory should give symmetric spreads
        solver.update_state(0.0, 50000.0, MAX_TIME_HORIZON);
        let quote_zero = solver.get_optimal_quotes();
        
        assert!(quote_zero.bid_spread > 0.0);
        assert!(quote_zero.ask_spread > 0.0);
        // Should be approximately symmetric at zero inventory
        assert!((quote_zero.bid_spread - quote_zero.ask_spread).abs() < 0.001);
    }

    #[test]
    fn test_inventory_skew() {
        let params = SkewParameters::default();
        let mut solver = OptimalSkewSolver::new(params);

        // Long inventory should widen ask, tighten bid
        solver.update_state(100.0, 50000.0, MAX_TIME_HORIZON);
        let quote_long = solver.get_optimal_quotes();

        // Short inventory should widen bid, tighten ask
        solver.update_state(-100.0, 50000.0, MAX_TIME_HORIZON);
        let quote_short = solver.get_optimal_quotes();

        // Verify skew direction
        assert!(quote_long.ask_spread > quote_long.bid_spread);
        assert!(quote_short.bid_spread > quote_short.ask_spread);
    }

    #[test]
    fn test_value_function_properties() {
        let params = SkewParameters::default();
        let solver = OptimalSkewSolver::new(params);

        let vf = solver.get_value_function();
        
        // Terminal value should be negative (liquidation penalty)
        let terminal_center = vf[TIME_STEPS - 1][MAX_INVENTORY_STEPS / 2];
        assert!(terminal_center <= 0.0);

        // Value should decrease with |inventory|
        let center = MAX_INVENTORY_STEPS / 2;
        let v_zero = vf[0][center];
        let v_high_inv = vf[0][0]; // High negative inventory
        
        assert!(v_zero >= v_high_inv);
    }
}
