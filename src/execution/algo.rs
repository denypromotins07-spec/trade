//! Execution Routing & Slippage Modeling - Adaptive TWAP/VWAP Algorithms
//! 
//! This module codes adaptive TWAP and VWAP execution algorithms that dynamically
//! adjust participation rates to hide the bot's footprint from predatory high-frequency
//! market makers.
//! 
//! **Performance Characteristics:**
//! - Lock-free state management
//! - Zero heap allocations during execution
//! - O(1) slice calculation
//! - Pre-allocated buffers for all data
//! 
//! **Architecture:**
//! The AlgoExecutor implements:
//! 1. TWAP (Time-Weighted Average Price) - Equal slices over time
//! 2. VWAP (Volume-Weighted Average Price) - Volume-proportional slices
//! 3. Adaptive participation - Adjust based on market conditions
//! 4. Stealth mode - Randomize timing and sizes to avoid detection

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Algorithm type selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlgoType {
    /// Time-Weighted Average Price
    Twap,
    /// Volume-Weighted Average Price
    Vwap,
    /// Implementation Shortfall
    Is,
    /// POV (Percentage of Volume)
    Pov,
}

/// Configuration for algorithmic execution
#[derive(Debug, Clone, Copy)]
pub struct AlgoConfig {
    /// Algorithm type
    pub algo_type: AlgoType,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Minimum slice size (scaled by 1e8)
    pub min_slice_size_scaled: u64,
    /// Maximum slice size (scaled by 1e8)
    pub max_slice_size_scaled: u64,
    /// Target participation rate (basis points, e.g., 1000 = 10%)
    pub target_participation_bps: u32,
    /// Maximum participation rate
    pub max_participation_bps: u32,
    /// Enable stealth mode (randomization)
    pub stealth_mode: bool,
    /// Randomization factor for stealth (0-100, percentage variance)
    pub stealth_variance_pct: u32,
    /// Aggressiveness level (1-5, higher = more aggressive)
    pub aggressiveness: u8,
}

impl Default for AlgoConfig {
    fn default() -> Self {
        Self {
            algo_type: AlgoType::Vwap,
            duration_ms: 3_600_000, // 1 hour
            min_slice_size_scaled: 100_000_000, // 1 unit
            max_slice_size_scaled: 10_000_000_000, // 100 units
            target_participation_bps: 500, // 5%
            max_participation_bps: 2000, // 20%
            stealth_mode: true,
            stealth_variance_pct: 20, // ±20% randomization
            aggressiveness: 3,
        }
    }
}

/// Current state of algorithm execution
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AlgoState {
    /// Remaining quantity to execute (scaled by 1e8)
    pub remaining_scaled: u64,
    /// Quantity already executed (scaled by 1e8)
    pub executed_scaled: u64,
    /// Number of slices executed
    pub slices_executed: u32,
    /// Total planned slices
    pub total_slices: u32,
    /// Start timestamp (ms)
    pub start_time_ms: u64,
    /// Last slice timestamp (ms)
    pub last_slice_time_ms: u64,
    /// Average execution price (scaled by 1e8)
    pub avg_price_scaled: u64,
    /// Current participation rate (basis points)
    pub current_participation_bps: u32,
    /// Whether algorithm is complete
    pub is_complete: bool,
    /// Whether algorithm is active
    pub is_active: bool,
}

/// Main Algorithmic Executor
pub struct AlgoExecutor {
    /// Configuration
    config: AlgoConfig,
    /// Current state
    state: AlgoState,
    /// VWAP volume profile (pre-allocated, 60 buckets for 1-hour)
    volume_profile: [u32; 60],
    /// Volume profile bucket count
    profile_buckets: usize,
    /// Random seed for stealth mode
    random_state: u64,
}

unsafe impl Send for AlgoExecutor {}
unsafe impl Sync for AlgoExecutor {}

impl AlgoExecutor {
    /// Initialize the algorithmic executor
    pub fn new(config: AlgoConfig) -> Self {
        // Initialize default VWAP profile (uniform distribution)
        let mut volume_profile = [16; 60]; // Equal weight per minute
        
        Self {
            config,
            state: AlgoState {
                remaining_scaled: 0,
                executed_scaled: 0,
                slices_executed: 0,
                total_slices: 0,
                start_time_ms: 0,
                last_slice_time_ms: 0,
                avg_price_scaled: 0,
                current_participation_bps: 0,
                is_complete: false,
                is_active: false,
            },
            volume_profile,
            profile_buckets: 60,
            random_state: 12345,
        }
    }

    /// Set custom VWAP volume profile
    pub fn set_volume_profile(&mut self, profile: &[u32]) {
        let len = profile.len().min(60);
        for i in 0..len {
            self.volume_profile[i] = profile[i];
        }
        self.profile_buckets = len;
    }

    /// Start a new algorithmic order
    #[inline]
    pub fn start(&mut self, quantity_scaled: u64, side_is_buy: bool, timestamp_ms: u64) {
        self.state.remaining_scaled = quantity_scaled;
        self.state.executed_scaled = 0;
        self.state.slices_executed = 0;
        self.state.start_time_ms = timestamp_ms;
        self.state.last_slice_time_ms = timestamp_ms;
        self.state.is_complete = false;
        self.state.is_active = true;
        self.state.avg_price_scaled = 0;

        // Calculate total slices based on algorithm type
        self.state.total_slices = self.calculate_total_slices(quantity_scaled);
    }

    /// Calculate the next slice to execute
    /// Returns the quantity for the next slice (scaled by 1e8)
    #[inline]
    pub fn calculate_next_slice(&mut self, current_time_ms: u64, market_volume_scaled: u64) -> u64 {
        if !self.state.is_active || self.state.is_complete {
            return 0;
        }

        if self.state.remaining_scaled == 0 {
            self.state.is_complete = true;
            self.state.is_active = false;
            return 0;
        }

        let elapsed_ms = current_time_ms.saturating_sub(self.state.start_time_ms);
        let remaining_ms = self.config.duration_ms.saturating_sub(elapsed_ms);

        // Base slice calculation depends on algorithm type
        let base_slice = match self.config.algo_type {
            AlgoType::Twap => self.calculate_twap_slice(elapsed_ms),
            AlgoType::Vwap => self.calculate_vwap_slice(elapsed_ms, market_volume_scaled),
            AlgoType::Is => self.calculate_is_slice(elapsed_ms, remaining_ms),
            AlgoType::Pov => self.calculate_pov_slice(market_volume_scaled),
        };

        // Apply constraints
        let constrained_slice = base_slice
            .max(self.config.min_slice_size_scaled)
            .min(self.config.max_slice_size_scaled)
            .min(self.state.remaining_scaled);

        // Apply stealth randomization
        let final_slice = if self.config.stealth_mode {
            self.apply_stealth(constrained_slice)
        } else {
            constrained_slice
        };

        // Update participation rate tracking
        if market_volume_scaled > 0 {
            self.state.current_participation_bps = 
                ((final_slice as u128 * 10_000) / market_volume_scaled as u128) as u32;
        }

        final_slice
    }

    /// Record an executed slice
    #[inline]
    pub fn record_fill(&mut self, quantity_scaled: u64, price_scaled: u64) {
        self.state.executed_scaled = self.state.executed_scaled.saturating_add(quantity_scaled);
        self.state.remaining_scaled = self.state.remaining_scaled.saturating_sub(quantity_scaled);
        self.state.slices_executed += 1;

        // Update average price
        let total_value = (self.state.avg_price_scaled as u128)
            .saturating_mul((self.state.executed_scaled - quantity_scaled) as u128)
            .saturating_add((price_scaled as u128).saturating_mul(quantity_scaled as u128));
        
        if self.state.executed_scaled > 0 {
            self.state.avg_price_scaled = 
                (total_value / self.state.executed_scaled as u128) as u64;
        }

        // Check completion
        if self.state.remaining_scaled == 0 || 
           self.state.slices_executed >= self.state.total_slices {
            self.state.is_complete = true;
            self.state.is_active = false;
        }
    }

    /// Calculate TWAP slice (equal distribution over time)
    #[inline]
    fn calculate_twap_slice(&self, elapsed_ms: u64) -> u64 {
        let total_slices = self.state.total_slices.max(1);
        let initial_qty = self.state.executed_scaled + self.state.remaining_scaled;
        initial_qty / total_slices as u64
    }

    /// Calculate VWAP slice (volume-weighted)
    #[inline]
    fn calculate_vwap_slice(&mut self, elapsed_ms: u64, market_volume_scaled: u64) -> u64 {
        if self.profile_buckets == 0 || self.config.duration_ms == 0 {
            return self.calculate_twap_slice(elapsed_ms);
        }

        // Determine current bucket
        let bucket_idx = ((elapsed_ms * self.profile_buckets as u64) / self.config.duration_ms) as usize;
        let bucket = bucket_idx.min(self.profile_buckets - 1);

        // Get volume weight for this bucket
        let volume_weight = self.volume_profile[bucket] as u64;
        let total_weight: u64 = self.volume_profile[..self.profile_buckets].iter().sum();

        if total_weight == 0 {
            return self.calculate_twap_slice(elapsed_ms);
        }

        // Calculate slice proportional to volume profile
        let initial_qty = self.state.executed_scaled + self.state.remaining_scaled;
        let target_slice = (initial_qty as u128 * volume_weight as u128 / total_weight as u128) as u64;

        // Also consider actual market volume (participation)
        let participation_slice = (market_volume_scaled as u128 
            * self.config.target_participation_bps as u128 / 10_000) as u64;

        // Take minimum of profile-based and participation-based
        target_slice.min(participation_slice)
    }

    /// Calculate Implementation Shortfall slice (aggressive early, passive late)
    #[inline]
    fn calculate_is_slice(&self, elapsed_ms: u64, remaining_ms: u64) -> u64 {
        let total_ms = self.config.duration_ms;
        if total_ms == 0 {
            return self.state.remaining_scaled;
        }

        // Aggressiveness determines how much to front-load
        let urgency_factor = match self.config.aggressiveness {
            1 => 20,  // Very passive
            2 => 40,
            3 => 60,  // Neutral
            4 => 80,
            5 => 100, // Very aggressive
            _ => 60,
        };

        let initial_qty = self.state.executed_scaled + self.state.remaining_scaled;
        
        // Front-load based on urgency
        let time_ratio = (elapsed_ms as u128 * 100 / total_ms as u128) as u64;
        let target_ratio = (time_ratio as u128 * urgency_factor as u128 / 100) 
            + ((100 - urgency_factor) as u128 * 50 / 100);
        
        let target_executed = (initial_qty as u128 * target_ratio / 100) as u64;
        target_executed.saturating_sub(self.state.executed_scaled)
    }

    /// Calculate POV (Percentage of Volume) slice
    #[inline]
    fn calculate_pov_slice(&self, market_volume_scaled: u64) -> u64 {
        (market_volume_scaled as u128 * self.config.target_participation_bps as u128 / 10_000) as u64
    }

    /// Apply stealth randomization to slice size
    #[inline]
    fn apply_stealth(&mut self, base_slice: u64) -> u64 {
        // Simple LCG random number generator
        self.random_state = self.random_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let random_factor = (self.random_state % 100) as u32;

        // Apply variance
        let variance = if random_factor < 50 {
            // Below 50: reduce slice
            100 - ((50 - random_factor) * self.config.stealth_variance_pct / 50)
        } else {
            // Above 50: increase slice
            100 + ((random_factor - 50) * self.config.stealth_variance_pct / 50)
        };

        (base_slice as u128 * variance as u128 / 100) as u64
    }

    /// Calculate total number of slices for the order
    #[inline]
    fn calculate_total_slices(&self, quantity_scaled: u64) -> u32 {
        let avg_slice = (self.config.min_slice_size_scaled + self.config.max_slice_size_scaled) / 2;
        (quantity_scaled / avg_slice.max(1)).max(1).min(1000) as u32
    }

    /// Get current execution state
    #[inline]
    pub fn get_state(&self) -> AlgoState {
        self.state
    }

    /// Get progress percentage (0-100)
    #[inline]
    pub fn get_progress_pct(&self) -> u8 {
        let initial = self.state.executed_scaled + self.state.remaining_scaled;
        if initial == 0 {
            return 0;
        }
        ((self.state.executed_scaled as u128 * 100) / initial as u128) as u8
    }

    /// Cancel the algorithm
    #[inline]
    pub fn cancel(&mut self) {
        self.state.is_active = false;
        self.state.is_complete = true;
    }

    /// Pause the algorithm
    #[inline]
    pub fn pause(&mut self) {
        self.state.is_active = false;
    }

    /// Resume the algorithm
    #[inline]
    pub fn resume(&mut self) {
        if !self.state.is_complete {
            self.state.is_active = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_execution() {
        let config = AlgoConfig {
            algo_type: AlgoType::Twap,
            duration_ms: 3600000,
            ..Default::default()
        };
        let mut executor = AlgoExecutor::new(config);

        // Start a 100-unit order
        executor.start(100_000_000_000, true, 1000);

        // Calculate first slice
        let slice = executor.calculate_next_slice(1000, 1_000_000_000);
        assert!(slice > 0);

        // Record fill
        executor.record_fill(slice, 50_000_000_000);
        assert_eq!(executor.get_progress_pct() > 0, true);
    }
}
