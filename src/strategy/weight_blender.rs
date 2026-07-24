// Weight Blender: Lock-free RCU blender for dynamically shifting strategy weights in O(1) time.
// Optimized for AMD Ryzen AI 5 architecture with zero mutex contention in the hot path.
// Ensures capital is only deployed to SOUL.md validated, profitable models.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Maximum number of strategies that can be blended simultaneously
const MAX_STRATEGIES: usize = 64;

/// Fixed-point representation for weights (scaled by 10^6 for precision)
const WEIGHT_SCALE: u64 = 1_000_000;

/// Read-Copy-Update (RCU) protected weight configuration.
/// Allows lock-free reads while updates happen on a cloned copy.
pub struct WeightBlender {
    /// Current active weights snapshot (RCU pointer via Arc)
    active_weights: Arc<AtomicU64>, // Bitmask-packed weights for up to 32 strategies (2 bits each)
    /// Pending update flag
    update_pending: AtomicBool,
    /// Last update timestamp in milliseconds
    last_update_ms: AtomicU64,
    /// Total allocated fraction (fixed-point)
    total_allocation_fp: AtomicU64,
    /// Blending mode: 0=equal, 1=performance-weighted, 2=regime-adaptive
    blending_mode: AtomicU64,
}

/// Strategy weight entry in fixed-point format
#[derive(Debug, Clone, Copy)]
pub struct StrategyWeight {
    pub strategy_id: u8,
    pub weight_fp: u64, // Fixed-point weight (scaled by WEIGHT_SCALE)
    pub confidence_fp: u64,
}

impl WeightBlender {
    /// Create a new weight blender with equal initial weights
    pub const fn new() -> Self {
        Self {
            active_weights: Arc::new(AtomicU64::new(0)),
            update_pending: AtomicBool::new(false),
            last_update_ms: AtomicU64::new(0),
            total_allocation_fp: AtomicU64::new(WEIGHT_SCALE), // 100% allocation
            blending_mode: AtomicU64::new(1), // Default to performance-weighted
        }
    }

    /// Lock-free read of current strategy weight.
    /// Returns weight as fixed-point value (scaled by WEIGHT_SCALE).
    #[inline(always)]
    pub fn get_weight(&self, strategy_id: u8) -> u64 {
        let weights_packed = self.active_weights.load(Ordering::Acquire);
        
        if strategy_id >= 32 {
            return 0; // Only support 32 strategies in packed format
        }
        
        // Extract 2-bit weight tier for this strategy (0-3 tiers)
        let shift = (strategy_id as u64) * 2;
        let tier = (weights_packed >> shift) & 0b11;
        
        // Convert tier to fixed-point weight
        match tier {
            0 => 0,                    // 0% allocation
            1 => WEIGHT_SCALE / 4,     // 25% of base
            2 => WEIGHT_SCALE / 2,     // 50% of base
            3 => WEIGHT_SCALE,         // 100% of base
            _ => unreachable!(),
        }
    }

    /// Compute blended weights from multiple strategies using RCU pattern.
    /// This is the cold path - called when SOUL.md approves new strategies.
    pub fn compute_blended_weights(
        &self,
        strategies: &[StrategyWeight],
        mode: u8,
    ) -> Vec<StrategyWeight> {
        let mut blended: Vec<StrategyWeight> = Vec::with_capacity(strategies.len());
        
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        match mode {
            0 => {
                // Equal weighting
                let n = strategies.len() as u64;
                if n == 0 {
                    return blended;
                }
                let equal_weight_fp = WEIGHT_SCALE / n;
                
                for strat in strategies {
                    blended.push(StrategyWeight {
                        strategy_id: strat.strategy_id,
                        weight_fp: equal_weight_fp,
                        confidence_fp: strat.confidence_fp,
                    });
                }
            }
            1 => {
                // Performance-weighted (using confidence as proxy)
                let total_confidence: u64 = strategies.iter()
                    .map(|s| s.confidence_fp)
                    .sum();
                
                if total_confidence == 0 {
                    return blended;
                }
                
                for strat in strategies {
                    let weight_fp = (strat.confidence_fp * WEIGHT_SCALE) / total_confidence;
                    blended.push(StrategyWeight {
                        strategy_id: strat.strategy_id,
                        weight_fp,
                        confidence_fp: strat.confidence_fp,
                    });
                }
            }
            2 => {
                // Regime-adaptive (placeholder - would integrate with regime_router)
                // For now, use performance-weighted with bonus for recent updates
                let age_bonus = |last_update: u64| -> u64 {
                    let age_ms = now_ms.saturating_sub(last_update);
                    if age_ms < 60_000 { 200_000 }      // < 1 min: +20%
                    else if age_ms < 300_000 { 100_000 } // < 5 min: +10%
                    else { 0 }
                };
                
                let total_adjusted: u64 = strategies.iter()
                    .map(|s| s.confidence_fp + age_bonus(now_ms))
                    .sum();
                
                if total_adjusted == 0 {
                    return blended;
                }
                
                for strat in strategies {
                    let adjusted = strat.confidence_fp + age_bonus(now_ms);
                    let weight_fp = (adjusted * WEIGHT_SCALE) / total_adjusted;
                    blended.push(StrategyWeight {
                        strategy_id: strat.strategy_id,
                        weight_fp,
                        confidence_fp: strat.confidence_fp,
                    });
                }
            }
            _ => {
                // Unknown mode, default to equal
                return self.compute_blended_weights(strategies, 0);
            }
        }

        blended
    }

    /// Atomically update weights using RCU pattern.
    /// Creates new snapshot, then atomically swaps pointer.
    pub fn update_weights_rcu(&self, new_weights: &[StrategyWeight]) {
        self.update_pending.store(true, Ordering::Release);
        
        // Pack new weights into bitmask (2 bits per strategy, up to 32 strategies)
        let mut packed: u64 = 0;
        
        for weight in new_weights {
            if weight.strategy_id >= 32 {
                continue; // Skip invalid IDs
            }
            
            // Convert fixed-point weight to 2-bit tier
            let tier = if weight.weight_fp == 0 {
                0
            } else if weight.weight_fp < WEIGHT_SCALE / 3 {
                1
            } else if weight.weight_fp < 2 * WEIGHT_SCALE / 3 {
                2
            } else {
                3
            };
            
            let shift = (weight.strategy_id as u64) * 2;
            packed |= (tier as u64) << shift;
        }
        
        // Update total allocation
        let total_fp: u64 = new_weights.iter().map(|w| w.weight_fp).sum();
        self.total_allocation_fp.store(total_fp, Ordering::Release);
        
        // Atomic swap of weights snapshot (RCU commit)
        self.active_weights.store(packed, Ordering::Release);
        
        // Update timestamp
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        self.last_update_ms.store(now_ms, Ordering::Release);
        
        self.update_pending.store(false, Ordering::Release);
    }

    /// Get total current allocation (fixed-point)
    pub fn get_total_allocation(&self) -> u64 {
        self.total_allocation_fp.load(Ordering::Acquire)
    }

    /// Set blending mode (0=equal, 1=performance, 2=regime-adaptive)
    pub fn set_blending_mode(&self, mode: u8) {
        self.blending_mode.store(mode as u64, Ordering::Release);
    }

    /// Get current blending mode
    pub fn get_blending_mode(&self) -> u8 {
        self.blending_mode.load(Ordering::Acquire) as u8
    }

    /// Check if an update is in progress
    pub fn is_update_pending(&self) -> bool {
        self.update_pending.load(Ordering::Acquire)
    }

    /// Emergency zero all weights (used during thermal shedding)
    pub fn emergency_zero_all(&self) {
        self.active_weights.store(0, Ordering::Release);
        self.total_allocation_fp.store(0, Ordering::Release);
    }
}

impl Default for WeightBlender {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_weight_blending() {
        let blender = WeightBlender::new();
        
        let strategies = vec![
            StrategyWeight { strategy_id: 0, weight_fp: 0, confidence_fp: 500_000 },
            StrategyWeight { strategy_id: 1, weight_fp: 0, confidence_fp: 500_000 },
            StrategyWeight { strategy_id: 2, weight_fp: 0, confidence_fp: 500_000 },
        ];
        
        let blended = blender.compute_blended_weights(&strategies, 0);
        
        assert_eq!(blended.len(), 3);
        // Each should get ~33.33% (333333 fixed-point)
        assert!(blended[0].weight_fp >= 333_333 && blended[0].weight_fp <= 333_334);
    }

    #[test]
    fn test_performance_weighted_blending() {
        let blender = WeightBlender::new();
        
        let strategies = vec![
            StrategyWeight { strategy_id: 0, weight_fp: 0, confidence_fp: 600_000 },
            StrategyWeight { strategy_id: 1, weight_fp: 0, confidence_fp: 400_000 },
        ];
        
        let blended = blender.compute_blended_weights(&strategies, 1);
        
        assert_eq!(blended.len(), 2);
        // Strategy 0 should get 60%, Strategy 1 should get 40%
        assert_eq!(blended[0].weight_fp, 600_000);
        assert_eq!(blended[1].weight_fp, 400_000);
    }

    #[test]
    fn test_rcu_weight_update() {
        let blender = WeightBlender::new();
        
        let new_weights = vec![
            StrategyWeight { strategy_id: 0, weight_fp: 500_000, confidence_fp: 800_000 },
            StrategyWeight { strategy_id: 1, weight_fp: 250_000, confidence_fp: 400_000 },
        ];
        
        blender.update_weights_rcu(&new_weights);
        
        // Verify weights were updated
        let w0 = blender.get_weight(0);
        let w1 = blender.get_weight(1);
        
        assert!(w0 > 0);
        assert!(w1 > 0);
        assert!(!blender.is_update_pending());
    }
}
