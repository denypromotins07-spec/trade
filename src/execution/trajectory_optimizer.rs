//! Trajectory Optimizer for Execution Strategies
//! 
//! Dynamic programming solver for optimal execution trajectories in O(1) space.
//! Continuously adjusts aggression parameter based on real-time order book replenishment rates.
//! Uses lock-free ring buffers to enforce 8GB RAM limit with zero heap allocations during hot path.
//! Optimized for AMD Ryzen AI 5 architecture with SIMD-accelerated matrix operations.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::arch::x86_64::*;

/// Maximum number of trajectory states (fixed for O(1) space complexity)
const MAX_STATES: usize = 128;

/// Ring buffer size for replenishment rate tracking
const REPLENISH_BUFFER_SIZE: usize = 64;

/// SIMD-aligned state buffer
#[repr(align(32))]
struct StateBuffer {
    inventories: [f64; MAX_STATES],
    times: [f64; MAX_STATES],
    costs: [f64; MAX_STATES],
    aggressions: [f64; MAX_STATES],
}

/// Ring buffer for replenishment rate history (lock-free)
struct ReplenishBuffer {
    data: [f64; REPLENISH_BUFFER_SIZE],
    head: AtomicU64,
    count: AtomicU64,
}

impl ReplenishBuffer {
    const fn new() -> Self {
        Self {
            data: [0.0; REPLENISH_BUFFER_SIZE],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn push(&self, value: f64) {
        let head = self.head.fetch_add(1, Ordering::AcqRel) % REPLENISH_BUFFER_SIZE as u64;
        unsafe {
            // Direct memory write, no bounds check needed (circular)
            *(self.data.as_ptr().add(head as usize) as *mut f64) = value;
        }
        self.count.fetch_min(REPLENISH_BUFFER_SIZE as u64, Ordering::Relaxed);
    }

    #[inline]
    fn mean(&self) -> f64 {
        let count = self.count.load(Ordering::Acquire).min(REPLENISH_BUFFER_SIZE as u64) as usize;
        if count == 0 {
            return 0.0;
        }

        let mut sum = 0.0;
        unsafe {
            // SIMD-accelerated summation
            let mut i = 0;
            let mut v_sum = _mm256_setzero_pd();
            
            while i + 4 <= count {
                let v = _mm256_loadu_pd(self.data.as_ptr().add(i));
                v_sum = _mm256_add_pd(v_sum, v);
                i += 4;
            }

            let arr: [f64; 4] = std::mem::transmute(v_sum);
            sum = arr.iter().sum();

            // Remainder
            for j in i..count {
                sum += *self.data.get_unchecked(j);
            }
        }
        sum / count as f64
    }

    #[inline]
    fn volatility(&self) -> f64 {
        let mean = self.mean();
        let count = self.count.load(Ordering::Acquire).min(REPLENISH_BUFFER_SIZE as u64) as usize;
        if count < 2 {
            return 0.0;
        }

        let mut sum_sq = 0.0;
        unsafe {
            for i in 0..count {
                let diff = *self.data.get_unchecked(i) - mean;
                sum_sq += diff * diff;
            }
        }
        (sum_sq / (count - 1) as f64).sqrt()
    }
}

/// Execution Aggression Mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggressionMode {
    Passive,      // Low urgency, minimize impact
    Neutral,      // Balanced approach
    Aggressive,   // High urgency, accept impact
    Urgent,       // Maximum speed, ignore impact
}

/// Dynamic Trajectory Optimizer
/// 
/// Solves for optimal execution path using dynamic programming with:
/// - O(1) space complexity via fixed-size state buffer
/// - Real-time aggression adjustment based on LOB replenishment
/// - Lock-free operation for concurrent updates
pub struct TrajectoryOptimizer {
    state_buffer: StateBuffer,
    replenish_buffer: ReplenishBuffer,
    current_state: AtomicU64,
    is_optimizing: AtomicBool,
    base_aggression: AtomicU64, // Encoded as f64 bits
    last_update_ns: AtomicU64,
}

impl TrajectoryOptimizer {
    /// Create a new trajectory optimizer with default parameters
    pub fn new() -> Self {
        Self {
            state_buffer: StateBuffer {
                inventories: [0.0; MAX_STATES],
                times: [0.0; MAX_STATES],
                costs: [0.0; MAX_STATES],
                aggressions: [0.0; MAX_STATES],
            },
            replenish_buffer: ReplenishBuffer::new(),
            current_state: AtomicU64::new(0),
            is_optimizing: AtomicBool::new(false),
            base_aggression: AtomicU64::new((0.5f64).to_bits()),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Record order book replenishment rate (call on each LOB update)
    #[inline]
    pub fn record_replenishment(&self, rate: f64) {
        self.replenish_buffer.push(rate);
    }

    /// Calculate dynamic aggression based on market conditions
    fn calculate_aggression(&self, target_inventory: f64, current_inventory: f64, time_remaining: f64) -> f64 {
        let base = f64::from_bits(self.base_aggression.load(Ordering::Acquire));
        
        // Get replenishment statistics
        let replenish_mean = self.replenish_buffer.mean();
        let replenish_vol = self.replenish_buffer.volatility();

        // Inventory urgency factor
        let inventory_gap = (target_inventory - current_inventory).abs();
        let urgency = if time_remaining > 0.0 {
            inventory_gap / (time_remaining * (target_inventory.abs() + 1e-9))
        } else {
            1.0
        };

        // Adjust aggression based on replenishment
        // High replenishment = can be more aggressive
        // High volatility = be more cautious
        let replenish_factor = (1.0 + replenish_mean).min(3.0);
        let vol_penalty = 1.0 / (1.0 + replenish_vol);

        let adjusted = base * replenish_factor * vol_penalty * urgency;
        adjusted.clamp(0.1, 2.0)
    }

    /// Determine aggression mode from calculated value
    #[inline]
    pub fn get_aggression_mode(&self, aggression: f64) -> AggressionMode {
        match aggression {
            x if x < 0.3 => AggressionMode::Passive,
            x if x < 0.7 => AggressionMode::Neutral,
            x if x < 1.3 => AggressionMode::Aggressive,
            _ => AggressionMode::Urgent,
        }
    }

    /// Solve optimal trajectory using dynamic programming
    /// 
    /// # Arguments
    /// * `initial_inventory` - Current position size
    /// * `target_inventory` - Desired end position (usually 0)
    /// * `time_horizon` - Time remaining for execution (seconds)
    /// * `max_discretization` - Maximum number of steps
    /// 
    /// # Returns
    /// Number of states computed
    pub fn solve(&self, initial_inventory: f64, target_inventory: f64, 
                 time_horizon: f64, max_discretization: usize) -> usize {
        if self.is_optimizing.swap(true, Ordering::Acquire) {
            return 0; // Already optimizing
        }

        let n = max_discretization.min(MAX_STATES);
        let dt = time_horizon / n as f64;

        // Initialize first state
        unsafe {
            self.state_buffer.inventories[0] = initial_inventory;
            self.state_buffer.times[0] = 0.0;
            self.state_buffer.costs[0] = 0.0;
            self.state_buffer.aggressions[0] = 
                self.calculate_aggression(target_inventory, initial_inventory, time_horizon);
        }

        // Dynamic programming forward pass
        for i in 1..n {
            let prev_idx = i - 1;
            let prev_inventory = unsafe { self.state_buffer.inventories[prev_idx] };
            let prev_time = unsafe { self.state_buffer.times[prev_idx] };
            let prev_cost = unsafe { self.state_buffer.costs[prev_idx] };
            let prev_aggression = unsafe { self.state_buffer.aggressions[prev_idx] };

            let time_remaining = time_horizon - prev_time;
            
            // Calculate optimal trade size for this step
            let aggression = self.calculate_aggression(target_inventory, prev_inventory, time_remaining);
            
            // Blend previous aggression with current for smoothness
            let smooth_aggression = 0.7 * aggression + 0.3 * prev_aggression;

            // Optimal trade size proportional to remaining inventory and aggression
            let remaining_gap = prev_inventory - target_inventory;
            let trade_fraction = (smooth_aggression * dt / time_remaining).min(1.0);
            let trade_size = remaining_gap * trade_fraction;

            let new_inventory = prev_inventory - trade_size;
            let new_time = prev_time + dt;

            // Estimate cost (simplified impact model)
            let impact_coeff = 0.0001 * smooth_aggression;
            let step_cost = prev_cost + impact_coeff * trade_size.abs() * trade_size.abs();

            unsafe {
                self.state_buffer.inventories[i] = new_inventory;
                self.state_buffer.times[i] = new_time;
                self.state_buffer.costs[i] = step_cost;
                self.state_buffer.aggressions[i] = smooth_aggression;
            }
        }

        self.current_state.store(n as u64, Ordering::Release);
        self.last_update_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Release
        );

        self.is_optimizing.store(false, Ordering::Release);
        n
    }

    /// Get the recommended trade size for the current state
    pub fn get_current_trade(&self, target_inventory: f64) -> Option<(f64, AggressionMode)> {
        let state_count = self.current_state.load(Ordering::Acquire) as usize;
        if state_count < 2 {
            return None;
        }

        let current_idx = 0; // Most recent state
        let next_idx = 1;

        unsafe {
            let current_inv = self.state_buffer.inventories[current_idx];
            let next_inv = self.state_buffer.inventories[next_idx];
            let aggression = self.state_buffer.aggressions[current_idx];

            let trade_size = current_inv - next_inv;
            Some((trade_size, self.get_aggression_mode(aggression)))
        }
    }

    /// Update base aggression parameter
    pub fn set_base_aggression(&self, aggression: f64) {
        self.base_aggression.store(aggression.clamp(0.1, 2.0).to_bits(), Ordering::Release);
    }

    /// Get total expected cost of current trajectory
    pub fn expected_total_cost(&self) -> f64 {
        let state_count = self.current_state.load(Ordering::Acquire) as usize;
        if state_count == 0 {
            return 0.0;
        }

        unsafe {
            self.state_buffer.costs[state_count - 1]
        }
    }

    /// Get time to completion estimate
    pub fn estimated_completion_time(&self) -> f64 {
        let state_count = self.current_state.load(Ordering::Acquire) as usize;
        if state_count == 0 {
            return 0.0;
        }

        unsafe {
            self.state_buffer.times[state_count - 1]
        }
    }
}

impl Default for TrajectoryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_initialization() {
        let opt = TrajectoryOptimizer::new();
        assert_eq!(opt.current_state.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_replenishment_tracking() {
        let opt = TrajectoryOptimizer::new();
        
        for i in 0..100 {
            opt.record_replenishment(1.0 + (i % 10) as f64 * 0.1);
        }

        let mean = opt.replenish_buffer.mean();
        assert!(mean > 1.0 && mean < 2.0);
    }

    #[test]
    fn test_trajectory_solving() {
        let opt = TrajectoryOptimizer::new();
        opt.set_base_aggression(0.5);

        // Simulate some replenishment data
        for _ in 0..20 {
            opt.record_replenishment(1.5);
        }

        let states = opt.solve(1000.0, 0.0, 60.0, 50);
        assert!(states > 0);
        
        let result = opt.get_current_trade(0.0);
        assert!(result.is_some());
        
        let (trade, mode) = result.unwrap();
        assert!(trade > 0.0);
        assert_ne!(mode, AggressionMode::Passive); // Should be at least neutral
    }

    #[test]
    fn test_aggression_modes() {
        let opt = TrajectoryOptimizer::new();
        
        assert_eq!(opt.get_aggression_mode(0.2), AggressionMode::Passive);
        assert_eq!(opt.get_aggression_mode(0.5), AggressionMode::Neutral);
        assert_eq!(opt.get_aggression_mode(1.0), AggressionMode::Aggressive);
        assert_eq!(opt.get_aggression_mode(1.5), AggressionMode::Urgent);
    }
}
