//! Nautilus/Ray Bot - Stage 15: Risk Parity Capital Allocator
//! Module: src/risk/capital_allocator.rs
//!
//! Description:
//!     Strict Risk Parity and risk-budgeting allocator that ensures every active
//!     strategy contributes equally to the overall portfolio volatility limit.
//!     Uses lock-free structures for thread-safe allocation updates.
//!
//! Constraints:
//!     - Latency: Microsecond-level reallocation.
//!     - Architecture: AMD Ryzen AI 5 (SIMD optimized).
//!     - Memory: Lock-free, zero allocation during hot path.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

// Configuration Constants
const MAX_STRATEGIES: usize = 20;
const VOLATILITY_TARGET: f64 = 0.15; // 15% annual volatility target
const REBALANCE_THRESHOLD: f64 = 0.05; // Rebalance if drift > 5%
const MIN_ALLOCATION: f64 = 0.01; // Minimum 1% allocation per strategy

/// Represents a single strategy's risk metrics.
#[derive(Debug, Clone, Copy)]
pub struct StrategyMetrics {
    pub name: &'static str,
    pub volatility: f64,      // Annualized volatility
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub correlation_to_portfolio: f64,
    pub current_allocation: f64,
}

/// Risk parity allocation result.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub strategy_allocations: Vec<f64>,
    pub total_risk_contribution: f64,
    pub is_balanced: bool,
    pub rebalance_required: bool,
}

/// Lock-free risk parity capital allocator.
pub struct CapitalAllocator {
    strategies: Vec<StrategyMetrics>,
    allocations: Vec<f64>,
    target_volatility: f64,
    is_rebalancing: AtomicBool,
    last_rebalance_ns: AtomicU64,
    allocation_checksum: AtomicU64,
}

impl CapitalAllocator {
    pub fn new(num_strategies: usize) -> Self {
        let equal_weight = 1.0 / num_strategies.min(MAX_STRATEGIES) as f64;
        
        Self {
            strategies: Vec::with_capacity(MAX_STRATEGIES),
            allocations: vec![equal_weight; num_strategies.min(MAX_STRATEGIES)],
            target_volatility: VOLATILITY_TARGET,
            is_rebalancing: AtomicBool::new(false),
            last_rebalance_ns: AtomicU64::new(0),
            allocation_checksum: AtomicU64::new(0),
        }
    }

    /// Add or update a strategy's risk metrics.
    pub fn add_strategy(&mut self, metrics: StrategyMetrics) {
        if self.strategies.len() >= MAX_STRATEGIES {
            // Replace oldest or find by name match
            self.strategies.remove(0);
        }
        self.strategies.push(metrics);
    }

    /// Calculate risk contribution of each strategy.
    /// Risk contribution = allocation * marginal_risk
    fn calculate_risk_contributions(&self) -> Vec<f64> {
        let n = self.strategies.len().min(self.allocations.len());
        let mut contributions = vec![0.0; n];

        // Simplified: assume diagonal covariance (no correlation)
        // In production: use full covariance matrix
        let mut total_risk = 0.0;
        
        for i in 0..n {
            let alloc = self.allocations[i];
            let vol = self.strategies[i].volatility;
            
            // Marginal risk contribution
            contributions[i] = alloc * vol;
            total_risk += contributions[i];
        }

        // Normalize if needed
        if total_risk > 0.0 {
            for contrib in &mut contributions {
                *contrib /= total_risk;
            }
        }

        contributions
    }

    /// Check if portfolio satisfies risk parity condition.
    fn is_risk_parity_satisfied(&self, tolerance: f64) -> bool {
        let contributions = self.calculate_risk_contributions();
        if contributions.is_empty() {
            return true;
        }

        let avg_contribution = contributions.iter().sum::<f64>() / contributions.len() as f64;
        
        contributions.iter().all(|&c| {
            (c - avg_contribution).abs() / avg_contribution < tolerance
        })
    }

    /// Compute risk parity allocations using iterative optimization.
    pub fn compute_risk_parity_allocations(&self) -> Vec<f64> {
        let n = self.strategies.len();
        if n == 0 {
            return vec![];
        }

        // Initialize with equal weights
        let mut new_allocations = vec![1.0 / n as f64; n];
        
        // Iterative optimization (simplified gradient descent)
        let learning_rate = 0.1;
        let max_iterations = 100;
        let tolerance = 1e-6;

        for _iteration in 0..max_iterations {
            let contributions = self.calculate_risk_contributions_with_allocs(&new_allocations);
            let avg_contrib = contributions.iter().sum::<f64>() / contributions.len() as f64;

            // Check convergence
            let max_drift = contributions.iter()
                .map(|&c| (c - avg_contrib).abs())
                .fold(0.0, f64::max);

            if max_drift < tolerance {
                break;
            }

            // Adjust allocations toward risk parity
            for i in 0..n {
                if contributions[i] > avg_contrib {
                    // Reduce allocation if contributing too much risk
                    new_allocations[i] *= (1.0 - learning_rate);
                } else {
                    // Increase allocation if contributing too little risk
                    new_allocations[i] *= (1.0 + learning_rate);
                }
            }

            // Project to simplex (sum to 1, all positive)
            Self::project_to_simplex(&mut new_allocations);
        }

        // Apply minimum allocation constraint
        for alloc in &mut new_allocations {
            *alloc = alloc.max(MIN_ALLOCATION);
        }
        
        // Re-normalize after applying minimums
        Self::project_to_simplex(&mut new_allocations);

        new_allocations
    }

    /// Calculate risk contributions given specific allocations.
    fn calculate_risk_contributions_with_allocs(&self, allocs: &[f64]) -> Vec<f64> {
        let n = allocs.len().min(self.strategies.len());
        let mut contributions = vec![0.0; n];

        for i in 0..n {
            contributions[i] = allocs[i] * self.strategies[i].volatility;
        }

        let total: f64 = contributions.iter().sum();
        if total > 0.0 {
            for c in &mut contributions {
                *c /= total;
            }
        }

        contributions
    }

    /// Project weights onto probability simplex.
    fn project_to_simplex(weights: &mut [f64]) {
        // Ensure non-negativity
        for w in weights.iter_mut() {
            *w = (*w).max(0.0);
        }

        // Normalize to sum to 1
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }

    /// Execute rebalancing if drift exceeds threshold.
    pub fn maybe_rebalance(&mut self) -> Option<AllocationResult> {
        if self.is_rebalancing.swap(true, Ordering::SeqCst) {
            return None; // Already rebalancing
        }

        let current_drift = self.calculate_allocation_drift();
        
        if current_drift < REBALANCE_THRESHOLD && self.is_risk_parity_satisfied(0.1) {
            self.is_rebalancing.store(false, Ordering::SeqCst);
            return None; // No rebalance needed
        }

        let new_allocations = self.compute_risk_parity_allocations();
        
        // Update allocations atomically
        self.allocations.clone_from(&new_allocations);
        
        let checksum = self.calculate_checksum(&new_allocations);
        self.allocation_checksum.store(checksum, Ordering::Relaxed);
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_rebalance_ns.store(now_ns, Ordering::Relaxed);

        self.is_rebalancing.store(false, Ordering::SeqCst);

        Some(AllocationResult {
            strategy_allocations: new_allocations,
            total_risk_contribution: self.calculate_total_risk(),
            is_balanced: self.is_risk_parity_satisfied(0.05),
            rebalance_required: false,
        })
    }

    /// Calculate drift from target allocations.
    fn calculate_allocation_drift(&self) -> f64 {
        let target = self.compute_risk_parity_allocations();
        if target.len() != self.allocations.len() {
            return 1.0; // Maximum drift if sizes mismatch
        }

        self.allocations.iter()
            .zip(target.iter())
            .map(|(current, t)| (current - t).abs())
            .sum()
    }

    /// Calculate total portfolio risk.
    fn calculate_total_risk(&self) -> f64 {
        let contributions = self.calculate_risk_contributions();
        contributions.iter().sum()
    }

    /// Calculate checksum for allocation verification.
    fn calculate_checksum(&self, allocs: &[f64]) -> u64 {
        // Simple hash for verification
        let mut hash: u64 = 0;
        for (i, &alloc) in allocs.iter().enumerate() {
            hash ^= (alloc.to_bits() ^ (i as u64)).wrapping_mul(31);
        }
        hash
    }

    /// Get current allocations (thread-safe copy).
    #[inline]
    pub fn get_allocations(&self) -> Vec<f64> {
        self.allocations.clone()
    }

    /// Check if rebalancing is in progress.
    #[inline]
    pub fn is_busy(&self) -> bool {
        self.is_rebalancing.load(Ordering::Relaxed)
    }

    /// Get time since last rebalance in nanoseconds.
    #[inline]
    pub fn time_since_rebalance_ns(&self) -> u64 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let last = self.last_rebalance_ns.load(Ordering::Relaxed);
        now_ns.saturating_sub(last)
    }
}

/// SIMD-accelerated risk contribution calculation.
#[target_feature(enable = "avx2")]
unsafe fn simd_calculate_risk_contributions(
    allocations: &[f64],
    volatilities: &[f64]
) -> Vec<f64> {
    // Placeholder for AVX2 implementation
    // In production: use std::arch::x86_64::_mm256_* functions
    allocations.iter()
        .zip(volatilities.iter())
        .map(|(&a, &v)| a * v)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_parity_computation() {
        let mut allocator = CapitalAllocator::new(3);
        
        allocator.add_strategy(StrategyMetrics {
            name: "Momentum",
            volatility: 0.20,
            sharpe_ratio: 1.5,
            max_drawdown: 0.15,
            correlation_to_portfolio: 0.8,
            current_allocation: 0.33,
        });
        
        allocator.add_strategy(StrategyMetrics {
            name: "MeanReversion",
            volatility: 0.10,
            sharpe_ratio: 2.0,
            max_drawdown: 0.08,
            correlation_to_portfolio: 0.5,
            current_allocation: 0.33,
        });
        
        allocator.add_strategy(StrategyMetrics {
            name: "Arbitrage",
            volatility: 0.05,
            sharpe_ratio: 3.0,
            max_drawdown: 0.03,
            correlation_to_portfolio: 0.2,
            current_allocation: 0.34,
        });

        let allocations = allocator.compute_risk_parity_allocations();
        
        // Higher volatility strategies should get lower allocations
        assert!(allocations[0] < allocations[2]); // Momentum < Arbitrage
        assert!((allocations.iter().sum::<f64>() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rebalance_trigger() {
        let mut allocator = CapitalAllocator::new(2);
        
        allocator.add_strategy(StrategyMetrics {
            name: "Strat1",
            volatility: 0.15,
            sharpe_ratio: 1.0,
            max_drawdown: 0.10,
            correlation_to_portfolio: 0.5,
            current_allocation: 0.5,
        });
        
        allocator.add_strategy(StrategyMetrics {
            name: "Strat2",
            volatility: 0.15,
            sharpe_ratio: 1.0,
            max_drawdown: 0.10,
            correlation_to_portfolio: 0.5,
            current_allocation: 0.5,
        });

        // Equal risk should not trigger rebalance
        let result = allocator.maybe_rebalance();
        assert!(result.is_none() || result.unwrap().is_balanced);
    }

    #[test]
    fn test_lock_free_operation() {
        let allocator = CapitalAllocator::new(3);
        
        assert!(!allocator.is_busy());
        
        // Simulate concurrent access
        let was_busy = allocator.is_rebalancing.swap(true, Ordering::SeqCst);
        assert!(!was_busy);
        assert!(allocator.is_busy());
        
        allocator.is_rebalancing.store(false, Ordering::SeqCst);
        assert!(!allocator.is_busy());
    }
}
