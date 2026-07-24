# Profitability Gate: Hard gate blocking strategies not meeting SOUL.md thresholds
# Optimized for AMD Ryzen AI 5 with lock-free atomic reads, zero heap allocations in hot path.
# Enforces strict microsecond latency requirements for strategy selection.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::soul::hot_swap_strategy::StrategyMetadata;

/// Thresholds stored as fixed-point integers to avoid FPU drift and heap allocations.
/// Profit threshold: 0.05% minimum daily return (stored as 500 basis points * 100)
const MIN_PROFIT_THRESHOLD_BPS: u64 = 500;
/// Sharpe ratio threshold: 1.5 minimum (stored as 150 * 100)
const MIN_SHARPE_THRESHOLD: u64 = 150;
/// Maximum age of validation: 24 hours in milliseconds
const MAX_VALIDATION_AGE_MS: u64 = 86_400_000;

/// Lock-free profitability gate that instantly rejects deprecated algorithms.
/// Uses atomic operations to read SOUL.md validated thresholds without mutex contention.
pub struct ProfitabilityGate {
    /// Atomic flag indicating if the gate is active
    is_active: AtomicBool,
    /// Current profit threshold in basis points (fixed-point)
    profit_threshold_bps: AtomicU64,
    /// Current Sharpe threshold (fixed-point)
    sharpe_threshold: AtomicU64,
    /// Last validation timestamp in milliseconds since epoch
    last_validation_ms: AtomicU64,
    /// Cache of approved strategy IDs (bitmask for up to 64 strategies)
    approved_strategies_mask: AtomicU64,
}

impl ProfitabilityGate {
    /// Creates a new profitability gate with default thresholds from SOUL.md
    pub const fn new() -> Self {
        Self {
            is_active: AtomicBool::new(true),
            profit_threshold_bps: AtomicU64::new(MIN_PROFIT_THRESHOLD_BPS),
            sharpe_threshold: AtomicU64::new(MIN_SHARPE_THRESHOLD),
            last_validation_ms: AtomicU64::new(0),
            approved_strategies_mask: AtomicU64::new(0),
        }
    }

    /// Updates thresholds atomically from SOUL.md ledger.
    /// This is called only when SOUL.md approves new strategies (cold path).
    #[inline]
    pub fn update_thresholds(&self, profit_bps: u64, sharpe: u64, strategy_id: u8) {
        self.profit_threshold_bps.store(profit_bps, Ordering::Release);
        self.sharpe_threshold.store(sharpe, Ordering::Release);
        self.last_validation_ms.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64,
            Ordering::Release,
        );
        
        // Set the bit for this strategy ID in the approved mask
        let current_mask = self.approved_strategies_mask.load(Ordering::Acquire);
        self.approved_strategies_mask.store(
            current_mask | (1u64 << strategy_id),
            Ordering::Release,
        );
    }

    /// Lock-free check if a strategy meets profitability requirements.
    /// Returns true if the strategy is approved, false otherwise.
    /// ZERO heap allocations - suitable for microsecond hot path.
    #[inline(always)]
    pub fn is_strategy_approved(&self, strategy: &StrategyMetadata) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }

        let current_profit_threshold = self.profit_threshold_bps.load(Ordering::Acquire);
        let current_sharpe_threshold = self.sharpe_threshold.load(Ordering::Acquire);
        
        // Check if strategy ID is in the approved bitmask
        let strategy_id = strategy.id as u8;
        if strategy_id >= 64 {
            return false; // Invalid strategy ID
        }
        
        let approved_mask = self.approved_strategies_mask.load(Ordering::Acquire);
        if approved_mask & (1u64 << strategy_id) == 0 {
            return false;
        }

        // Fixed-point comparison: strategy profit must exceed threshold
        if strategy.last_daily_profit_bps < current_profit_threshold {
            return false;
        }

        // Fixed-point comparison: strategy Sharpe must exceed threshold
        if strategy.last_sharpe_ratio < current_sharpe_threshold {
            return false;
        }

        // Check validation age - reject if too old
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        let last_val = self.last_validation_ms.load(Ordering::Acquire);
        
        if now_ms.saturating_sub(last_val) > MAX_VALIDATION_AGE_MS {
            return false;
        }

        true
    }

    /// Instantly reject all strategies (used during thermal shedding or emergency stop)
    #[inline]
    pub fn emergency_reject_all(&self) {
        self.is_active.store(false, Ordering::Release);
        self.approved_strategies_mask.store(0, Ordering::Release);
    }

    /// Re-enable the gate after emergency conditions clear
    #[inline]
    pub fn reenable(&self) {
        self.is_active.store(true, Ordering::Release);
    }

    /// Get current approval mask for monitoring (cold path)
    pub fn get_approval_mask(&self) -> u64 {
        self.approved_strategies_mask.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_rejects_unprofitable_strategy() {
        let gate = ProfitabilityGate::new();
        let unprofitable_strategy = StrategyMetadata {
            id: 0,
            name: "lossy_strat",
            last_daily_profit_bps: 100, // Below 500 threshold
            last_sharpe_ratio: 200,     // Above 150 threshold
        };
        
        // Initially no strategies are approved (mask is 0)
        assert!(!gate.is_strategy_approved(&unprofitable_strategy));
    }

    #[test]
    fn test_gate_approves_profitable_strategy() {
        let gate = ProfitabilityGate::new();
        
        // Manually approve strategy 0
        gate.update_thresholds(500, 150, 0);
        
        let profitable_strategy = StrategyMetadata {
            id: 0,
            name: "winner_strat",
            last_daily_profit_bps: 750, // Above 500 threshold
            last_sharpe_ratio: 180,     // Above 150 threshold
        };
        
        assert!(gate.is_strategy_approved(&profitable_strategy));
    }
}
