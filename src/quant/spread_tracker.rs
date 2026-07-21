//! Spread Tracker: Lock-free Rolling Window Z-Score Computation
//! 
//! Real-time z-score calculation for asset spreads with zero heap allocations.
//! Uses circular buffers and Welford's algorithm for numerically stable variance.
//! Optimized for AMD Ryzen AI 5 with SIMD-friendly memory layouts.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::marker::PhantomData;

/// Maximum window size for rolling statistics (tuned for 8GB RAM limit)
const MAX_WINDOW_SIZE: usize = 10_000;

/// Thread-safe spread tracker with lock-free updates
pub struct SpreadTracker {
    /// Circular buffer for spread values
    buffer: [f64; MAX_WINDOW_SIZE],
    /// Current write position in circular buffer
    head: AtomicUsize,
    /// Number of elements currently in buffer
    count: AtomicUsize,
    /// Running mean (Welford's algorithm)
    mean: f64,
    /// Running sum of squared differences (Welford's M2)
    m2: f64,
    /// Window size for rolling calculations
    window_size: usize,
}

impl SpreadTracker {
    /// Create a new spread tracker with specified window size
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.min(MAX_WINDOW_SIZE);
        Self {
            buffer: [0.0; MAX_WINDOW_SIZE],
            head: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            mean: 0.0,
            m2: 0.0,
            window_size,
        }
    }

    /// Add a new spread value and update rolling statistics
    /// Returns the updated z-score
    #[inline(always)]
    pub fn push(&mut self, value: f64) -> f64 {
        let idx = self.head.load(Ordering::Relaxed);
        let count = self.count.load(Ordering::Relaxed);
        
        // Get the old value being replaced (if buffer is full)
        let old_value = if count >= self.window_size {
            Some(self.buffer[idx])
        } else {
            None
        };

        // Store new value in circular buffer
        self.buffer[idx] = value;

        // Update head pointer
        let new_head = (idx + 1) % self.window_size;
        self.head.store(new_head, Ordering::Relaxed);

        // Update count if not yet at window size
        if count < self.window_size {
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        // Update running statistics using Welford's online algorithm
        if let Some(old_val) = old_value {
            // Remove old value contribution
            self.remove_sample(old_val);
        }
        
        // Add new value contribution
        self.add_sample(value);

        // Calculate and return z-score
        self.calculate_z_score(value)
    }

    /// Add a sample to running statistics (Welford's algorithm)
    #[inline(always)]
    fn add_sample(&mut self, value: f64) {
        let count = self.count.load(Ordering::Relaxed) as f64;
        let delta = value - self.mean;
        self.mean += delta / count;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Remove a sample from running statistics (for sliding window)
    #[inline(always)]
    fn remove_sample(&mut self, value: f64) {
        let count = self.count.load(Ordering::Relaxed) as f64;
        if count <= 1.0 {
            self.mean = 0.0;
            self.m2 = 0.0;
            return;
        }

        let delta = value - self.mean;
        self.mean = (self.mean * count - value) / (count - 1.0);
        let delta2 = value - self.mean;
        self.m2 -= delta * delta2;

        // Ensure M2 doesn't go negative due to floating point errors
        if self.m2 < 0.0 {
            self.m2 = 0.0;
        }
    }

    /// Calculate z-score for a given value
    #[inline(always)]
    pub fn calculate_z_score(&self, value: f64) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return 0.0;
        }

        let variance = self.m2 / (count - 1) as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-12 {
            return 0.0;
        }

        (value - self.mean) / std_dev
    }

    /// Get current mean of the spread
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get current variance of the spread
    pub fn variance(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count < 2 {
            return 0.0;
        }
        self.m2 / (count - 1) as f64
    }

    /// Get current standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get number of samples in window
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if window is full
    pub fn is_full(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= self.window_size
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.head.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.mean = 0.0;
        self.m2 = 0.0;
    }

    /// Get all values in the current window (for debugging/analysis)
    pub fn get_values(&self) -> Vec<f64> {
        let count = self.count.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        
        let mut values = Vec::with_capacity(count);
        for i in 0..count {
            let idx = (head + i) % self.window_size;
            values.push(self.buffer[idx]);
        }
        values
    }
}

impl Default for SpreadTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Multi-spread tracker for monitoring multiple pairs simultaneously
pub struct MultiSpreadTracker {
    trackers: Vec<SpreadTracker>,
    pair_names: Vec<(String, String)>,
}

impl MultiSpreadTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            trackers: Vec::new(),
            pair_names: Vec::new(),
        }
    }

    /// Add a new spread pair to track
    pub fn add_pair(&mut self, asset1: &str, asset2: &str, window_size: usize) -> usize {
        let idx = self.trackers.len();
        self.trackers.push(SpreadTracker::new(window_size));
        self.pair_names.push((asset1.to_string(), asset2.to_string()));
        idx
    }

    /// Update spread for a specific pair and return z-score
    pub fn update_spread(&mut self, pair_idx: usize, spread: f64) -> Option<f64> {
        self.trackers.get_mut(pair_idx).map(|t| t.push(spread))
    }

    /// Get z-score for a specific pair
    pub fn get_z_score(&self, pair_idx: usize, current_spread: f64) -> Option<f64> {
        self.trackers.get(pair_idx).map(|t| t.calculate_z_score(current_spread))
    }

    /// Get all current z-scores
    pub fn get_all_z_scores(&self, spreads: &[f64]) -> Vec<f64> {
        self.trackers
            .iter()
            .zip(spreads.iter())
            .map(|(t, &s)| t.calculate_z_score(s))
            .collect()
    }

    /// Find pairs with extreme z-scores (potential trading opportunities)
    pub fn find_extremes(&self, spreads: &[f64], threshold: f64) -> Vec<(usize, f64)> {
        self.trackers
            .iter()
            .zip(spreads.iter())
            .enumerate()
            .filter_map(|(i, (t, &s))| {
                let z = t.calculate_z_score(s);
                if z.abs() > threshold {
                    Some((i, z))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Spread calculator for two price series
pub struct SpreadCalculator {
    /// Type of spread: 'ratio' or 'difference'
    spread_type: SpreadType,
    /// Hedge ratio for ratio spreads
    hedge_ratio: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum SpreadType {
    Difference,
    Ratio,
}

impl SpreadCalculator {
    pub const fn new(spread_type: SpreadType, hedge_ratio: f64) -> Self {
        Self {
            spread_type,
            hedge_ratio,
        }
    }

    /// Calculate spread between two prices
    #[inline(always)]
    pub fn calculate(&self, price1: f64, price2: f64) -> f64 {
        match self.spread_type {
            SpreadType::Difference => price1 - self.hedge_ratio * price2,
            SpreadType::Ratio => {
                if price2 < 1e-12 {
                    0.0
                } else {
                    price1 / price2
                }
            }
        }
    }

    /// Calculate spread for entire series
    pub fn calculate_series(&self, prices1: &[f64], prices2: &[f64]) -> Vec<f64> {
        let n = prices1.len().min(prices2.len());
        let mut spreads = Vec::with_capacity(n);
        
        for i in 0..n {
            spreads.push(self.calculate(prices1[i], prices2[i]));
        }
        
        spreads
    }
}

/// Trading signal generator based on spread z-scores
#[derive(Debug, Clone, Copy)]
pub struct SpreadSignal {
    pub pair_idx: usize,
    pub z_score: f64,
    pub action: SignalAction,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalAction {
    Long,
    Short,
    Close,
    Hold,
}

pub struct SpreadSignalGenerator {
    entry_threshold: f64,
    exit_threshold: f64,
    current_positions: Vec<i8>, // -1: short, 0: flat, 1: long
}

impl SpreadSignalGenerator {
    pub fn new(entry_threshold: f64, exit_threshold: f64, num_pairs: usize) -> Self {
        Self {
            entry_threshold,
            exit_threshold,
            current_positions: vec![0; num_pairs],
        }
    }

    /// Generate trading signals based on z-scores
    pub fn generate_signals(&mut self, z_scores: &[f64]) -> Vec<SpreadSignal> {
        z_scores
            .iter()
            .enumerate()
            .map(|(i, &z)| {
                let action = self.determine_action(i, z);
                let confidence = self.calculate_confidence(z);
                
                SpreadSignal {
                    pair_idx: i,
                    z_score: z,
                    action,
                    confidence,
                }
            })
            .collect()
    }

    fn determine_action(&mut self, idx: usize, z: f64) -> SignalAction {
        let position = self.current_positions[idx];

        if position == 0 {
            if z > self.entry_threshold {
                self.current_positions[idx] = -1;
                SignalAction::Short
            } else if z < -self.entry_threshold {
                self.current_positions[idx] = 1;
                SignalAction::Long
            } else {
                SignalAction::Hold
            }
        } else if position == 1 {
            if z > -self.exit_threshold {
                self.current_positions[idx] = 0;
                SignalAction::Close
            } else {
                SignalAction::Hold
            }
        } else {
            if z < self.exit_threshold {
                self.current_positions[idx] = 0;
                SignalAction::Close
            } else {
                SignalAction::Hold
            }
        }
    }

    fn calculate_confidence(&self, z: f64) -> f64 {
        // Confidence increases with |z| up to a saturation point
        let abs_z = z.abs();
        if abs_z < self.entry_threshold {
            0.0
        } else {
            (1.0 - (-2.0 * (abs_z - self.entry_threshold)).exp()).min(1.0)
        }
    }

    /// Get current positions
    pub fn positions(&self) -> &[i8] {
        &self.current_positions
    }

    /// Reset all positions
    pub fn reset_positions(&mut self) {
        for pos in &mut self.current_positions {
            *pos = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_tracker() {
        let mut tracker = SpreadTracker::new(100);
        
        // Push some values
        for i in 0..50 {
            let z = tracker.push(i as f64);
            // Z-score should be computable after first few samples
            if i >= 2 {
                assert!(z.is_finite());
            }
        }

        assert_eq!(tracker.count(), 50);
        assert!(!tracker.is_full());
    }

    #[test]
    fn test_sliding_window() {
        let mut tracker = SpreadTracker::new(10);
        
        // Fill the window
        for i in 0..15 {
            tracker.push(i as f64);
        }

        // Should be full and have exactly 10 elements
        assert!(tracker.is_full());
        assert_eq!(tracker.count(), 10);
    }

    #[test]
    fn test_multi_spread_tracker() {
        let mut multi = MultiSpreadTracker::new(100);
        
        let idx1 = multi.add_pair("BTC", "ETH", 50);
        let idx2 = multi.add_pair("ETH", "SOL", 50);
        
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 1);

        let z1 = multi.update_spread(idx1, 0.5).unwrap();
        let z2 = multi.update_spread(idx2, 0.3).unwrap();
        
        assert!(z1.is_finite());
        assert!(z2.is_finite());
    }
}
