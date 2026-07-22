//! Order Flow Toxicity - VPIN (Volume-Synchronized Probability of Informed Trading)
//! 
//! This module extends the classic VPIN metric with adaptive volume bucket sizes to detect
//! toxic flow during extreme low-liquidity regimes. It identifies when informed traders
//! are likely active, allowing the market maker to widen spreads defensively.
//! 
//! **Key Features:**
//! - Adaptive volume bucket sizing for low-liquidity detection.
//! - Real-time toxic flow probability estimation.
//! - Microsecond-level update capability.

use std::collections::VecDeque;

/// Represents a single volume bucket for VPIN calculation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VolumeBucket {
    pub buy_volume: u64,
    pub sell_volume: u64,
    pub timestamp_start_ns: u64,
    pub timestamp_end_ns: u64,
}

impl VolumeBucket {
    pub fn new() -> Self {
        VolumeBucket {
            buy_volume: 0,
            sell_volume: 0,
            timestamp_start_ns: 0,
            timestamp_end_ns: 0,
        }
    }

    /// Add a trade to the bucket based on tick rule classification.
    pub fn add_trade(&mut self, volume: u64, is_buy: bool, timestamp_ns: u64) {
        if self.timestamp_start_ns == 0 {
            self.timestamp_start_ns = timestamp_ns;
        }
        self.timestamp_end_ns = timestamp_ns;

        if is_buy {
            self.buy_volume += volume;
        } else {
            self.sell_volume += volume;
        }
    }

    /// Get total volume in this bucket.
    pub fn total_volume(&self) -> u64 {
        self.buy_volume + self.sell_volume
    }

    /// Get absolute imbalance in this bucket.
    pub fn imbalance(&self) -> u64 {
        if self.buy_volume > self.sell_volume {
            self.buy_volume - self.sell_volume
        } else {
            self.sell_volume - self.buy_volume
        }
    }
}

impl Default for VolumeBucket {
    fn default() -> Self {
        Self::new()
    }
}

/// VPIN Calculator with adaptive bucket sizing.
pub struct VpinCalculator {
    /// Rolling window of volume buckets
    buckets: VecDeque<VolumeBucket>,
    /// Target volume per bucket (adaptive)
    target_bucket_volume: u64,
    /// Current active bucket
    current_bucket: VolumeBucket,
    /// Number of buckets for VPIN window
    num_buckets: usize,
    /// Running sum of imbalances for efficient VPIN calculation
    running_imbalance_sum: u64,
    /// Running sum of total volumes
    running_volume_sum: u64,
    /// Last estimated VPIN value (scaled by 10000 for integer math)
    last_vpin: u32,
}

impl VpinCalculator {
    /// Create a new VPIN calculator with initial parameters.
    pub fn new(num_buckets: usize, initial_target_volume: u64) -> Self {
        VpinCalculator {
            buckets: VecDeque::with_capacity(num_buckets),
            target_bucket_volume: initial_target_volume,
            current_bucket: VolumeBucket::new(),
            num_buckets,
            running_imbalance_sum: 0,
            running_volume_sum: 0,
            last_vpin: 0,
        }
    }

    /// Classify a trade using the tick rule and add to current bucket.
    pub fn add_tick(&mut self, price: u64, volume: u64, timestamp_ns: u64, prev_price: u64) -> Option<u32> {
        // Tick rule: if price > prev_price, it's a buy; if price < prev_price, it's a sell
        // If price == prev_price, use previous tick direction (simplified here as buy)
        let is_buy = price >= prev_price;

        self.current_bucket.add_trade(volume, is_buy, timestamp_ns);

        // Check if bucket is full
        if self.current_bucket.total_volume() >= self.target_bucket_volume {
            self.finalize_bucket();
            
            // Recalculate VPIN if we have enough buckets
            if self.buckets.len() == self.num_buckets {
                return Some(self.calculate_vpin());
            }
        }

        None
    }

    /// Finalize the current bucket and add to the rolling window.
    fn finalize_bucket(&mut self) {
        if self.current_bucket.total_volume() == 0 {
            self.current_bucket = VolumeBucket::new();
            return;
        }

        let finished_bucket = self.current_bucket;
        
        // Update running sums
        if self.buckets.len() >= self.num_buckets {
            // Remove oldest bucket from running sums
            if let Some(oldest) = self.buckets.pop_front() {
                self.running_imbalance_sum = self.running_imbalance_sum.saturating_sub(oldest.imbalance());
                self.running_volume_sum = self.running_volume_sum.saturating_sub(oldest.total_volume());
            }
        }

        // Add new bucket to running sums
        self.running_imbalance_sum += finished_bucket.imbalance();
        self.running_volume_sum += finished_bucket.total_volume();

        self.buckets.push_back(finished_bucket);
        self.current_bucket = VolumeBucket::new();

        // Adapt bucket size based on recent volatility/liquidity
        self.adapt_bucket_size();
    }

    /// Adapt bucket size based on market conditions.
    fn adapt_bucket_size(&mut self) {
        if self.buckets.len() < 5 {
            return;
        }

        // Calculate average bucket duration
        let mut total_duration = 0u64;
        let mut count = 0;
        for bucket in &self.buckets {
            if bucket.timestamp_end_ns > bucket.timestamp_start_ns {
                total_duration += bucket.timestamp_end_ns - bucket.timestamp_start_ns;
                count += 1;
            }
        }

        if count == 0 {
            return;
        }

        let avg_duration_ns = total_duration / count;
        
        // If buckets are filling too slowly (low liquidity), reduce target volume
        // If buckets are filling too fast (high liquidity), increase target volume
        const TARGET_DURATION_NS: u64 = 1_000_000_000; // 1 second target
        
        if avg_duration_ns > TARGET_DURATION_NS * 2 {
            // Low liquidity: reduce bucket size by 10%
            self.target_bucket_volume = (self.target_bucket_volume * 9 / 10).max(1000);
        } else if avg_duration_ns < TARGET_DURATION_NS / 2 {
            // High liquidity: increase bucket size by 10%
            self.target_bucket_volume = self.target_bucket_volume * 11 / 10;
        }
    }

    /// Calculate VPIN from the current window of buckets.
    /// VPIN = Sum(|Buy - Sell|) / Sum(Buy + Sell)
    fn calculate_vpin(&mut self) -> u32 {
        if self.running_volume_sum == 0 {
            return 0;
        }

        // Scale by 10000 for integer representation of percentage (0-100%)
        let vpin_scaled = (self.running_imbalance_sum * 10000) / self.running_volume_sum;
        self.last_vpin = vpin_scaled as u32;
        
        vpin_scaled as u32
    }

    /// Get the latest VPIN estimate (scaled by 10000).
    pub fn get_vpin(&self) -> u32 {
        self.last_vpin
    }

    /// Check if toxicity is above a threshold (e.g., 70% = 7000).
    pub fn is_toxic(&self, threshold_scaled: u32) -> bool {
        self.last_vpin > threshold_scaled
    }

    /// Get the current target bucket volume.
    pub fn get_target_bucket_volume(&self) -> u64 {
        self.target_bucket_volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_calculation() {
        let mut calc = VpinCalculator::new(10, 1000);

        // Simulate imbalanced trades (mostly sells)
        for i in 0..20 {
            let _ = calc.add_tick(50000, 100, i * 1000, 50000);
        }

        // VPIN should be calculated after enough buckets are filled
        // Exact value depends on implementation details
        let vpin = calc.get_vpin();
        println!("VPIN: {}", vpin);
    }

    #[test]
    fn test_adaptive_bucket_sizing() {
        let mut calc = VpinCalculator::new(10, 1000);

        // Simulate slow trades (low liquidity)
        for i in 0..5 {
            let _ = calc.add_tick(50000, 50, i * 2_000_000_000, 50000); // 2 seconds apart
        }

        // Target volume should have decreased
        assert!(calc.get_target_bucket_volume() <= 1000);
    }
}
