//! Nautilus/Ray Bot - Stage 15: Monte Carlo Stress Testing Engine
//! Module: src/risk/stress_tester.rs
//!
//! Description:
//!     Real-time Monte Carlo stress testing engine that injects historical black-swan
//!     scenarios (FTX collapse, LUNA crash) into the current portfolio state.
//!     Provides instant risk assessment during market turbulence.
//!
//! Constraints:
//!     - Latency: Microsecond-level scenario injection.
//!     - Architecture: AMD Ryzen AI 5 (SIMD optimized).
//!     - Memory: Pre-allocated buffers, zero heap allocation during hot path.

use std::collections::HashMap;

// Configuration Constants
const MAX_SCENARIOS: usize = 100;
const MAX_ASSETS: usize = 50;
const MONTE_CARLO_SIMULATIONS: usize = 10000;

/// Historical black-swan event definitions.
#[derive(Debug, Clone)]
pub struct BlackSwanEvent {
    pub name: &'static str,
    pub date: &'static str,
    pub asset_shocks: HashMap<&'static str, f64>, // Asset -> Shock percentage
    pub correlation_matrix: [[f64; MAX_ASSETS]; MAX_ASSETS],
    pub duration_days: u32,
}

/// Result of a stress test simulation.
#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub event_name: String,
    pub portfolio_loss_pct: f64,
    var_95: f64,
    var_99: f64,
    max_drawdown: f64,
    recovery_time_days: u32,
    breach_probability: f64,
}

/// High-performance stress testing engine.
pub struct StressTester {
    events: Vec<BlackSwanEvent>,
    current_portfolio: Vec<f64>, // Weights
    asset_prices: Vec<f64>,      // Current prices
    results_cache: Vec<StressTestResult>,
    is_running: bool,
}

impl StressTester {
    pub fn new() -> Self {
        Self {
            events: Self::load_historical_events(),
            current_portfolio: vec![0.0; MAX_ASSETS],
            asset_prices: vec![1.0; MAX_ASSETS],
            results_cache: Vec::with_capacity(MAX_SCENARIOS),
            is_running: false,
        }
    }

    /// Load predefined historical black-swan events.
    fn load_historical_events() -> Vec<BlackSwanEvent> {
        vec![
            BlackSwanEvent {
                name: "LUNA Collapse",
                date: "2022-05-09",
                asset_shocks: HashMap::from([
                    ("LUNA", -0.9999),
                    ("UST", -0.95),
                    ("BTC", -0.45),
                    ("ETH", -0.55),
                ]),
                correlation_matrix: [[0.0; MAX_ASSETS]; MAX_ASSETS],
                duration_days: 7,
            },
            BlackSwanEvent {
                name: "FTX Collapse",
                date: "2022-11-08",
                asset_shocks: HashMap::from([
                    ("FTT", -0.95),
                    ("BTC", -0.20),
                    ("ETH", -0.25),
                    ("SOL", -0.40),
                ]),
                correlation_matrix: [[0.0; MAX_ASSETS]; MAX_ASSETS],
                duration_days: 14,
            },
            BlackSwanEvent {
                name: "COVID Crash",
                date: "2020-03-12",
                asset_shocks: HashMap::from([
                    ("BTC", -0.50),
                    ("ETH", -0.55),
                    ("SPY", -0.30),
                    ("GOLD", -0.10),
                ]),
                correlation_matrix: [[0.0; MAX_ASSETS]; MAX_ASSETS],
                duration_days: 30,
            },
            BlackSwanEvent {
                name: "Flash Crash 2010",
                date: "2010-05-06",
                asset_shocks: HashMap::from([
                    ("ES_FUTURES", -0.09),
                    ("SPY", -0.07),
                ]),
                correlation_matrix: [[0.0; MAX_ASSETS]; MAX_ASSETS],
                duration_days: 1,
            },
        ]
    }

    /// Set current portfolio weights.
    pub fn set_portfolio(&mut self, weights: Vec<f64>) {
        self.current_portfolio = weights;
        self.current_portfolio.resize(MAX_ASSETS, 0.0);
    }

    /// Set current asset prices for PnL calculation.
    pub fn set_prices(&mut self, prices: Vec<f64>) {
        self.asset_prices = prices;
        self.asset_prices.resize(MAX_ASSETS, 1.0);
    }

    /// Run Monte Carlo simulation for a specific black-swan event.
    pub fn run_simulation(&mut self, event_idx: usize) -> StressTestResult {
        if event_idx >= self.events.len() {
            return self.create_default_result();
        }

        let event = &self.events[event_idx];
        let mut losses = Vec::with_capacity(MONTE_CARLO_SIMULATIONS);

        // Run Monte Carlo simulations
        for sim in 0..MONTE_CARLO_SIMULATIONS {
            let portfolio_return = self.simulate_scenario(event, sim);
            losses.push(-portfolio_return); // Convert return to loss
        }

        // Sort losses for VaR calculation
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate statistics
        let var_95_idx = (losses.len() as f64 * 0.95) as usize;
        let var_99_idx = (losses.len() as f64 * 0.99) as usize;
        
        let var_95 = losses[var_95_idx.min(losses.len() - 1)];
        let var_99 = losses[var_99_idx.min(losses.len() - 1)];
        let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

        let result = StressTestResult {
            event_name: event.name.to_string(),
            portfolio_loss_pct: avg_loss * 100.0,
            var_95: var_95 * 100.0,
            var_99: var_99 * 100.0,
            max_drawdown: max_loss * 100.0,
            recovery_time_days: event.duration_days,
            breach_probability: self.calculate_breach_probability(&losses),
        };

        self.results_cache.push(result.clone());
        result
    }

    /// Simulate portfolio return under a black-swan scenario.
    fn simulate_scenario(&self, event: &BlackSwanEvent, seed: usize) -> f64 {
        let mut total_return = 0.0;
        
        // Apply shocks with some random variation based on seed
        for (asset_name, shock) in &event.asset_shocks {
            if let Some(idx) = self.get_asset_index(asset_name) {
                let weight = self.current_portfolio[idx];
                
                // Add randomness: shock ± 10% based on seed
                let random_factor = 1.0 + ((seed % 100) as f64 - 50.0) / 500.0;
                let adjusted_shock = shock * random_factor;
                
                total_return += weight * adjusted_shock;
            }
        }

        // Add correlation effects (simplified)
        let correlation_effect = self.apply_correlation_effects(seed);
        total_return += correlation_effect;

        total_return
    }

    /// Apply correlation matrix effects during stress.
    fn apply_correlation_effects(&self, seed: usize) -> f64 {
        // Simplified correlation impact
        // In production: use full correlation matrix multiplication
        ((seed % 100) as f64 - 50.0) / 1000.0
    }

    /// Calculate probability of breaching risk limits.
    fn calculate_breach_probability(&self, losses: &[f64]) -> f64 {
        let threshold = 0.10; // 10% loss threshold
        let breaches = losses.iter().filter(|&&l| l > threshold).count();
        breaches as f64 / losses.len() as f64
    }

    /// Get asset index by name (simplified mapping).
    fn get_asset_index(&self, name: &str) -> Option<usize> {
        // Simplified asset mapping for demo
        match name {
            "BTC" => Some(0),
            "ETH" => Some(1),
            "SOL" => Some(2),
            "LUNA" => Some(3),
            "UST" => Some(4),
            "FTT" => Some(5),
            "SPY" => Some(6),
            "GOLD" => Some(7),
            _ => None,
        }
    }

    /// Run all historical stress tests.
    pub fn run_all_stress_tests(&mut self) -> Vec<StressTestResult> {
        self.results_cache.clear();
        (0..self.events.len())
            .map(|i| self.run_simulation(i))
            .collect()
    }

    /// Get worst-case scenario from cached results.
    pub fn get_worst_case(&self) -> Option<&StressTestResult> {
        self.results_cache.iter()
            .max_by(|a, b| a.max_drawdown.partial_cmp(&b.max_drawdown).unwrap())
    }

    /// Create default result for invalid scenarios.
    fn create_default_result(&self) -> StressTestResult {
        StressTestResult {
            event_name: "Unknown".to_string(),
            portfolio_loss_pct: 0.0,
            var_95: 0.0,
            var_99: 0.0,
            max_drawdown: 0.0,
            recovery_time_days: 0,
            breach_probability: 0.0,
        }
    }

    /// Check if stress test is currently running.
    #[inline]
    pub fn is_busy(&self) -> bool {
        self.is_running
    }
}

/// SIMD-accelerated portfolio loss calculation.
#[target_feature(enable = "avx2")]
unsafe fn simd_calculate_portfolio_loss(
    weights: &[f64],
    shocks: &[f64]
) -> f64 {
    // Placeholder for AVX2 implementation
    // In production: use std::arch::x86_64::_mm256_* functions
    weights.iter()
        .zip(shocks.iter())
        .map(|(w, s)| w * s)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luna_stress_test() {
        let mut tester = StressTester::new();
        
        // Set up portfolio with crypto exposure
        tester.set_portfolio(vec![
            0.4,  // BTC
            0.3,  // ETH
            0.1,  // SOL
            0.1,  // LUNA (will be hit hard)
            0.1,  // Other
        ]);

        // Run LUNA collapse simulation (index 0)
        let result = tester.run_simulation(0);
        
        assert!(result.portfolio_loss_pct > 0.0);
        assert_eq!(result.event_name, "LUNA Collapse");
    }

    #[test]
    fn test_ftx_stress_test() {
        let mut tester = StressTester::new();
        
        tester.set_portfolio(vec![
            0.5,  // BTC
            0.3,  // ETH
            0.1,  // SOL
            0.1,  // FTT
        ]);

        let result = tester.run_simulation(1); // FTX is index 1
        
        assert!(result.portfolio_loss_pct > 0.0);
        assert_eq!(result.event_name, "FTX Collapse");
    }

    #[test]
    fn test_all_stress_tests() {
        let mut tester = StressTester::new();
        tester.set_portfolio(vec![0.5, 0.3, 0.2]);
        
        let results = tester.run_all_stress_tests();
        
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.portfolio_loss_pct >= 0.0));
    }

    #[test]
    fn test_worst_case_identification() {
        let mut tester = StressTester::new();
        tester.set_portfolio(vec![0.6, 0.4]);
        
        tester.run_all_stress_tests();
        let worst = tester.get_worst_case();
        
        assert!(worst.is_some());
    }
}
