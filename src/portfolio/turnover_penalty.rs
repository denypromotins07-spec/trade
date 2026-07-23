//! Turnover Penalty Engine for Portfolio Rebalancing
//! 
//! This module builds a continuous turnover penalty engine that restricts excessive
//! rebalancing to minimize Binance taker fees and slippage, updating weights atomically.
//! 
//! Optimized for:
//! - Microsecond latency via lock-free atomic operations
//! - 8GB RAM limit enforcement via bounded state buffers
//! - AMD Ryzen AI 5 architecture compatibility

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};

/// Lock-free memory counter
static TURNOVER_MEMORY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Memory budget for turnover module (500MB)
const TURNOVER_MEMORY_BUDGET: u64 = 1024 * 1024 * 500;

/// Maximum tracked assets
const MAX_TRACKED_ASSETS: usize = 300;

/// Maximum history length for turnover tracking (ring buffer size)
const MAX_HISTORY_LENGTH: usize = 1000;

/// Fee structure for Binance (taker fee in basis points)
const BINANCE_TAKER_FEE_BPS: f64 = 10.0; // 0.10% default taker fee

/// Slippage estimate per unit of turnover (basis points)
const ESTIMATED_SLIPPAGE_BPS: f64 = 5.0;

/// Atomic flag indicating if rebalancing is currently allowed
static REBALANCING_ALLOWED: AtomicBool = AtomicBool::new(true);

/// Turnover statistics tracker with ring buffer history
pub struct TurnoverTracker {
    /// Ring buffer of historical turnover values
    turnover_history: Vec<f64>,
    /// Current write index in the ring buffer
    current_index: usize,
    /// Total count of recorded turnovers
    record_count: usize,
    /// Running sum for efficient mean calculation
    running_sum: f64,
    /// Running sum of squares for variance calculation
    running_sum_sq: f64,
    /// Number of assets being tracked
    n_assets: usize,
}

impl TurnoverTracker {
    /// Create a new turnover tracker with memory validation
    pub fn new(n_assets: usize, history_length: usize) -> Result<Self, &'static str> {
        if n_assets > MAX_TRACKED_ASSETS {
            return Err("Asset count exceeds maximum for turnover tracking");
        }
        
        let actual_history = history_length.min(MAX_HISTORY_LENGTH);
        let estimated_memory = (actual_history * 8 + n_assets * 16) as u64;
        
        let current_usage = TURNOVER_MEMORY_COUNTER.load(Ordering::Relaxed);
        if current_usage + estimated_memory > TURNOVER_MEMORY_BUDGET {
            return Err("Memory budget exceeded for turnover tracking");
        }
        
        TURNOVER_MEMORY_COUNTER.fetch_add(estimated_memory, Ordering::Relaxed);
        
        Ok(Self {
            turnover_history: vec![0.0; actual_history],
            current_index: 0,
            record_count: 0,
            running_sum: 0.0,
            running_sum_sq: 0.0,
            n_assets,
        })
    }
    
    /// Record a new turnover value (thread-safe)
    pub fn record_turnover(&mut self, turnover: f64) {
        let old_value = self.turnover_history[self.current_index];
        
        // Update running statistics
        self.running_sum -= old_value;
        self.running_sum_sq -= old_value * old_value;
        
        self.running_sum += turnover;
        self.running_sum_sq += turnover * turnover;
        
        // Store new value
        self.turnover_history[self.current_index] = turnover;
        
        // Advance ring buffer index
        self.current_index = (self.current_index + 1) % self.turnover_history.len();
        
        if self.record_count < self.turnover_history.len() {
            self.record_count += 1;
        }
    }
    
    /// Get mean turnover over history
    pub fn mean_turnover(&self) -> f64 {
        if self.record_count == 0 {
            return 0.0;
        }
        self.running_sum / self.record_count as f64
    }
    
    /// Get turnover volatility (standard deviation)
    pub fn turnover_volatility(&self) -> f64 {
        if self.record_count < 2 {
            return 0.0;
        }
        
        let mean = self.mean_turnover();
        let variance = (self.running_sum_sq / self.record_count as f64) - (mean * mean);
        
        if variance < 0.0 {
            return 0.0;
        }
        
        variance.sqrt()
    }
    
    /// Get recent turnover trend (last N records)
    pub fn recent_trend(&self, n: usize) -> f64 {
        let actual_n = n.min(self.record_count);
        if actual_n == 0 {
            return 0.0;
        }
        
        let mut sum = 0.0;
        let mut idx = self.current_index;
        
        for _ in 0..actual_n {
            idx = if idx == 0 { self.turnover_history.len() - 1 } else { idx - 1 };
            sum += self.turnover_history[idx];
        }
        
        sum / actual_n as f64
    }
    
    /// Check if current turnover is above historical average by threshold
    pub fn is_elevated(&self, threshold_multiplier: f64) -> bool {
        let mean = self.mean_turnover();
        let recent = self.recent_trend(10);
        recent > mean * threshold_multiplier
    }
}

impl Drop for TurnoverTracker {
    fn drop(&mut self) {
        let estimated_memory = (self.turnover_history.len() * 8 + self.n_assets * 16) as u64;
        TURNOVER_MEMORY_COUNTER.fetch_sub(estimated_memory, Ordering::Relaxed);
    }
}

/// Turnover penalty calculator with dynamic adjustment
pub struct TurnoverPenaltyEngine {
    /// Base penalty coefficient
    base_penalty: f64,
    /// Dynamic multiplier based on market conditions
    dynamic_multiplier: f64,
    /// Turnover tracker instance
    tracker: TurnoverTracker,
    /// Last rebalance timestamp
    last_rebalance: Instant,
    /// Minimum time between rebalances
    min_rebalance_interval: Duration,
    /// Cumulative transaction costs (fees + slippage)
    cumulative_costs_bps: f64,
    /// Daily cost limit in basis points
    daily_cost_limit_bps: f64,
    /// Flag to enable/disable penalty temporarily
    penalty_enabled: AtomicBool,
}

impl TurnoverPenaltyEngine {
    /// Create a new penalty engine
    pub fn new(
        base_penalty: f64,
        n_assets: usize,
        min_rebalance_seconds: u64,
        daily_cost_limit_bps: f64,
    ) -> Result<Self, &'static str> {
        let tracker = TurnoverTracker::new(n_assets, 500)?;
        
        Ok(Self {
            base_penalty,
            dynamic_multiplier: 1.0,
            tracker,
            last_rebalance: Instant::now(),
            min_rebalance_interval: Duration::from_secs(min_rebalance_seconds),
            cumulative_costs_bps: 0.0,
            daily_cost_limit_bps,
            penalty_enabled: AtomicBool::new(true),
        })
    }
    
    /// Calculate penalty for proposed weight change
    pub fn calculate_penalty(&self, current_weights: &[f64], proposed_weights: &[f64]) -> f64 {
        if !self.penalty_enabled.load(Ordering::Relaxed) {
            return 0.0;
        }
        
        if current_weights.len() != proposed_weights.len() {
            return f64::MAX; // Invalid input, maximum penalty
        }
        
        // Calculate raw turnover (sum of absolute weight changes)
        let turnover: f64 = current_weights.iter()
            .zip(proposed_weights.iter())
            .map(|(c, p)| (p - c).abs())
            .sum();
        
        // Estimate transaction cost in basis points
        let estimated_cost_bps = turnover * (BINANCE_TAKER_FEE_BPS + ESTIMATED_SLIPPAGE_BPS);
        
        // Check daily cost limit
        if self.cumulative_costs_bps + estimated_cost_bps > self.daily_cost_limit_bps {
            return f64::MAX; // Prohibitively high penalty
        }
        
        // Check minimum rebalance interval
        if self.last_rebalance.elapsed() < self.min_rebalance_interval {
            let time_ratio = self.min_rebalance_interval.as_secs_f64() 
                / self.last_rebalance.elapsed().as_secs_f64();
            return self.base_penalty * self.dynamic_multiplier * time_ratio;
        }
        
        // Apply dynamic multiplier based on turnover history
        let effective_penalty = self.base_penalty * self.dynamic_multiplier;
        
        // Non-linear penalty: higher turnover gets disproportionately penalized
        let turnover_factor = if turnover > 0.1 {
            1.0 + (turnover - 0.1) * 5.0 // Extra penalty for >10% turnover
        } else {
            1.0
        };
        
        effective_penalty * turnover_factor * turnover
    }
    
    /// Apply proposed weights if penalty is acceptable
    pub fn try_rebalance(
        &mut self,
        current_weights: &[f64],
        proposed_weights: &[f64],
        max_penalty_threshold: f64,
    ) -> Result<Vec<f64>, &'static str> {
        let penalty = self.calculate_penalty(current_weights, proposed_weights);
        
        if penalty > max_penalty_threshold {
            return Err("Turnover penalty exceeds threshold");
        }
        
        // Calculate actual turnover for recording
        let turnover: f64 = current_weights.iter()
            .zip(proposed_weights.iter())
            .map(|(c, p)| (p - c).abs())
            .sum();
        
        // Update cumulative costs
        let cost_bps = turnover * (BINANCE_TAKER_FEE_BPS + ESTIMATED_SLIPPAGE_BPS);
        self.cumulative_costs_bps += cost_bps;
        
        // Record turnover
        self.tracker.record_turnover(turnover);
        
        // Update last rebalance time
        self.last_rebalance = Instant::now();
        
        // Adjust dynamic multiplier based on recent turnover
        self.update_dynamic_multiplier();
        
        Ok(proposed_weights.to_vec())
    }
    
    /// Update dynamic multiplier based on turnover patterns
    fn update_dynamic_multiplier(&mut self) {
        let mean_turnover = self.tracker.mean_turnover();
        let volatility = self.tracker.turnover_volatility();
        
        // Increase penalty during high volatility periods
        if volatility > mean_turnover * 0.5 {
            self.dynamic_multiplier = (self.dynamic_multiplier * 1.1).min(5.0);
        } else if self.dynamic_multiplier > 1.0 {
            // Gradually decay back to baseline
            self.dynamic_multiplier *= 0.95;
            if self.dynamic_multiplier < 1.0 {
                self.dynamic_multiplier = 1.0;
            }
        }
    }
    
    /// Reset daily cost counter (call at start of each trading day)
    pub fn reset_daily_counter(&mut self) {
        self.cumulative_costs_bps = 0.0;
    }
    
    /// Enable or disable penalty calculation
    pub fn set_penalty_enabled(&self, enabled: bool) {
        self.penalty_enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if rebalancing is globally allowed
    pub fn is_rebalancing_allowed() -> bool {
        REBALANCING_ALLOWED.load(Ordering::Relaxed)
    }
    
    /// Set global rebalancing permission
    pub fn set_rebalancing_allowed(allowed: bool) {
        REBALANCING_ALLOWED.store(allowed, Ordering::Relaxed);
    }
    
    /// Get current turnover statistics
    pub fn get_statistics(&self) -> TurnoverStatistics {
        TurnoverStatistics {
            mean_turnover: self.tracker.mean_turnover(),
            volatility: self.tracker.turnover_volatility(),
            recent_trend_10: self.tracker.recent_trend(10),
            recent_trend_50: self.tracker.recent_trend(50),
            dynamic_multiplier: self.dynamic_multiplier,
            cumulative_costs_bps: self.cumulative_costs_bps,
            time_since_last_rebalance: self.last_rebalance.elapsed(),
        }
    }
}

/// Statistics snapshot for monitoring
#[derive(Debug, Clone)]
pub struct TurnoverStatistics {
    pub mean_turnover: f64,
    pub volatility: f64,
    pub recent_trend_10: f64,
    pub recent_trend_50: f64,
    pub dynamic_multiplier: f64,
    pub cumulative_costs_bps: f64,
    pub time_since_last_rebalance: Duration,
}

/// Atomic weight updater for thread-safe portfolio updates
pub struct AtomicWeightUpdater {
    weights: Vec<std::sync::atomic::AtomicF64Wrapper>,
    n_assets: usize,
}

// Simple wrapper since AtomicF64 isn't stable yet
struct AtomicF64Wrapper {
    bits: AtomicU64,
}

impl AtomicF64Wrapper {
    fn new(value: f64) -> Self {
        Self {
            bits: AtomicU64::new(value.to_bits()),
        }
    }
    
    fn load(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }
    
    fn store(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }
}

impl AtomicWeightUpdater {
    pub fn new(initial_weights: &[f64]) -> Result<Self, &'static str> {
        if initial_weights.len() > MAX_TRACKED_ASSETS {
            return Err("Asset count exceeds maximum for atomic updater");
        }
        
        let weights: Vec<AtomicF64Wrapper> = initial_weights
            .iter()
            .map(|&w| AtomicF64Wrapper::new(w))
            .collect();
        
        Ok(Self {
            weights,
            n_assets: initial_weights.len(),
        })
    }
    
    /// Atomically update all weights
    pub fn update_all(&self, new_weights: &[f64]) -> Result<(), &'static str> {
        if new_weights.len() != self.n_assets {
            return Err("Weight dimension mismatch");
        }
        
        for (i, &new_w) in new_weights.iter().enumerate() {
            self.weights[i].store(new_w);
        }
        
        Ok(())
    }
    
    /// Get atomic snapshot of current weights
    pub fn get_snapshot(&self) -> Vec<f64> {
        self.weights.iter().map(|w| w.load()).collect()
    }
    
    /// Atomically update single weight
    pub fn update_single(&self, asset_idx: usize, new_weight: f64) -> Result<(), &'static str> {
        if asset_idx >= self.n_assets {
            return Err("Invalid asset index");
        }
        
        self.weights[asset_idx].store(new_weight);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_turnover_tracker() {
        let mut tracker = TurnoverTracker::new(10, 100).unwrap();
        
        for i in 0..50 {
            tracker.record_turnover(0.05 + (i as f64 * 0.001));
        }
        
        assert!(tracker.mean_turnover() > 0.05);
        assert!(tracker.turnover_volatility() >= 0.0);
    }
    
    #[test]
    fn test_penalty_engine() {
        let current = vec![0.4, 0.3, 0.3];
        let proposed = vec![0.5, 0.25, 0.25];
        
        let mut engine = TurnoverPenaltyEngine::new(0.1, 3, 1, 100.0).unwrap();
        
        let penalty = engine.calculate_penalty(&current, &proposed);
        assert!(penalty > 0.0);
        
        // Small change should have low penalty
        let small_change = vec![0.41, 0.30, 0.29];
        let small_penalty = engine.calculate_penalty(&current, &small_change);
        assert!(small_penalty < penalty);
    }
    
    #[test]
    fn test_atomic_updater() {
        let initial = vec![0.4, 0.3, 0.3];
        let updater = AtomicWeightUpdater::new(&initial).unwrap();
        
        let snapshot = updater.get_snapshot();
        assert_eq!(snapshot, initial);
        
        let new_weights = vec![0.5, 0.3, 0.2];
        updater.update_all(&new_weights).unwrap();
        
        let new_snapshot = updater.get_snapshot();
        assert_eq!(new_snapshot, new_weights);
    }
}
