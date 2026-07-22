//! Nautilus/Ray Bot - Stage 15: Order Flow Toxicity Detector (VPIN)
//! Module: src/orderflow/toxicity.rs
//!
//! Description:
//!     Advanced Volume-Synchronized Probability of Informed Trading (VPIN) implementation.
//!     Dynamically adjusts spreads when toxic, informed order flow is detected.
//!     Operates purely in the Rust hot path for instant alerts.
//!
//! Constraints:
//!     - Latency: Microsecond-level VPIN calculation.
//!     - Architecture: AMD Ryzen AI 5 (SIMD optimized).
//!     - Memory: Zero heap allocation during hot path.

use std::collections::VecDeque;

// Configuration Constants
const VPIN_BUCKET_SIZE: u64 = 1000; // Volume per bucket in base units
const VPIN_WINDOW_SIZE: usize = 50; // Number of buckets for rolling calculation
const TOXICITY_THRESHOLD: f64 = 0.8; // VPIN > 0.8 indicates toxic flow
const SPREAD_ADJUSTMENT_FACTOR: f64 = 2.5; // Multiplier for spread widening

/// Represents a single volume bucket for VPIN calculation.
#[derive(Debug, Clone, Copy)]
pub struct VolumeBucket {
    pub buy_volume: u64,
    pub sell_volume: u64,
    pub imbalance: i64, // buy - sell
}

impl VolumeBucket {
    pub fn new() -> Self {
        Self {
            buy_volume: 0,
            sell_volume: 0,
            imbalance: 0,
        }
    }

    #[inline]
    pub fn add_trade(&mut self, volume: u64, is_buy: bool) {
        if is_buy {
            self.buy_volume = self.buy_volume.saturating_add(volume);
        } else {
            self.sell_volume = self.sell_volume.saturating_add(volume);
        }
        self.imbalance = self.buy_volume as i64 - self.sell_volume as i64;
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.buy_volume + self.sell_volume >= VPIN_BUCKET_SIZE
    }

    #[inline]
    pub fn total_volume(&self) -> u64 {
        self.buy_volume + self.sell_volume
    }
}

/// High-performance VPIN calculator with lock-free ring buffer.
pub struct VpinDetector {
    buckets: VecDeque<VolumeBucket>,
    current_bucket: VolumeBucket,
    cumulative_volume: u64,
    vpin_value: f64,
    is_toxic: bool,
}

impl VpinDetector {
    pub fn new() -> Self {
        Self {
            buckets: VecDeque::with_capacity(VPIN_WINDOW_SIZE),
            current_bucket: VolumeBucket::new(),
            cumulative_volume: 0,
            vpin_value: 0.0,
            is_toxic: false,
        }
    }

    /// Process a trade and update VPIN metrics.
    /// Returns true if toxicity state changed.
    #[inline]
    pub fn process_trade(&mut self, volume: u64, is_buy: bool) -> bool {
        self.current_bucket.add_trade(volume, is_buy);
        self.cumulative_volume = self.cumulative_volume.saturating_add(volume);

        if self.current_bucket.is_full() {
            self.buckets.push_back(self.current_bucket);
            self.current_bucket = VolumeBucket::new();

            // Maintain window size
            if self.buckets.len() > VPIN_WINDOW_SIZE {
                self.buckets.pop_front();
            }

            // Recalculate VPIN
            let old_vpin = self.vpin_value;
            self.vpin_value = self.calculate_vpin();
            self.is_toxic = self.vpin_value > TOXICITY_THRESHOLD;

            return self.is_toxic != (old_vpin > TOXICITY_THRESHOLD);
        }

        false
    }

    /// Calculate VPIN using the Easley-Prado method.
    /// Optimized for SIMD where possible.
    fn calculate_vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let mut sum_abs_imbalance: u64 = 0;
        let mut total_volume: u64 = 0;

        for bucket in &self.buckets {
            sum_abs_imbalance = sum_abs_imbalance.saturating_add(bucket.imbalance.unsigned_abs());
            total_volume = total_volume.saturating_add(bucket.total_volume());
        }

        if total_volume == 0 {
            return 0.0;
        }

        sum_abs_imbalance as f64 / total_volume as f64
    }

    /// Get current VPIN value.
    #[inline]
    pub fn vpin(&self) -> f64 {
        self.vpin_value
    }

    /// Check if current flow is toxic.
    #[inline]
    pub fn is_toxic(&self) -> bool {
        self.is_toxic
    }

    /// Calculate recommended spread adjustment based on toxicity.
    #[inline]
    pub fn get_spread_adjustment(&self) -> f64 {
        if self.is_toxic {
            SPREAD_ADJUSTMENT_FACTOR * self.vpin_value
        } else {
            1.0
        }
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_bucket = VolumeBucket::new();
        self.cumulative_volume = 0;
        self.vpin_value = 0.0;
        self.is_toxic = false;
    }
}

/// SIMD-accelerated batch VPIN calculation for historical analysis.
/// Uses AVX2 instructions available on AMD Ryzen AI 5.
#[target_feature(enable = "avx2")]
unsafe fn simd_vpin_batch(buy_volumes: &[u64], sell_volumes: &[u64]) -> f64 {
    // Placeholder for explicit AVX2 intrinsics implementation
    // In production: use std::arch::x86_64::_mm256_* functions
    
    let mut sum_abs: u64 = 0;
    let mut total: u64 = 0;
    
    for (buy, sell) in buy_volumes.iter().zip(sell_volumes.iter()) {
        let imbalance = (*buy as i64 - *sell as i64).unsigned_abs();
        sum_abs = sum_abs.saturating_add(imbalance);
        total = total.saturating_add(*buy + *sell);
    }
    
    if total == 0 {
        0.0
    } else {
        sum_abs as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_calculation() {
        let mut detector = VpinDetector::new();
        
        // Simulate balanced flow
        for _ in 0..VPIN_BUCKET_SIZE / 2 {
            detector.process_trade(1, true);
            detector.process_trade(1, false);
        }
        
        assert!(detector.vpin() < 0.5);
        assert!(!detector.is_toxic());
    }

    #[test]
    fn test_toxic_flow_detection() {
        let mut detector = VpinDetector::new();
        
        // Simulate heavily skewed sell flow (informed selling)
        for _ in 0..VPIN_BUCKET_SIZE * VPIN_WINDOW_SIZE as u64 {
            detector.process_trade(1, false);
        }
        
        assert!(detector.vpin() > TOXICITY_THRESHOLD);
        assert!(detector.is_toxic());
        assert!(detector.get_spread_adjustment() > 1.0);
    }

    #[test]
    fn test_spread_adjustment() {
        let mut detector = VpinDetector::new();
        
        // Normal market
        assert_eq!(detector.get_spread_adjustment(), 1.0);
        
        // Toxic market simulation
        for _ in 0..VPIN_BUCKET_SIZE * VPIN_WINDOW_SIZE as u64 {
            detector.process_trade(1, false);
        }
        
        let adjustment = detector.get_spread_adjustment();
        assert!(adjustment > SPREAD_ADJUSTMENT_FACTOR);
    }
}
