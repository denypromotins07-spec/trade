//! Branching Ratio Calculator for Hawkes Processes
//! 
//! Calculates real-time branching ratios to detect critical market states
//! and impending flash crashes before they materialize on the order book.
//! 
//! The branching ratio n = sum(alpha_ij) indicates system stability:
//! - n < 1: Stable (subcritical)
//! - n ≈ 1: Critical state (impending instability)
//! - n > 1: Unstable (supercritical, flash crash risk)
//! 
//! Memory: Bounded sliding windows to enforce 8GB RAM limit
//! Latency: Microsecond updates using incremental computation

use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum window size for branching ratio estimation (bounded memory)
const MAX_WINDOW_SIZE: usize = 100_000;

/// Threshold for critical state detection
const CRITICAL_THRESHOLD: f64 = 0.95;

/// Flash crash warning threshold
const FLASH_CRASH_THRESHOLD: f64 = 1.0;

/// Time window for recent branching ratio in nanoseconds
const RECENT_WINDOW_NS: u64 = 1_000_000_000; // 1 second

/// Branching ratio measurement sample
#[derive(Debug, Clone)]
pub struct BranchingSample {
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Overall branching ratio
    pub branching_ratio: f64,
    /// Per-dimension branching ratios
    pub dimension_ratios: Array1<f64>,
    /// Spectral radius of excitation matrix
    pub spectral_radius: f64,
}

/// Real-time branching ratio calculator
pub struct BranchingRatioCalculator {
    /// Sliding window of recent branching ratio samples
    samples: VecDeque<BranchingSample>,
    /// Current excitation matrix estimate
    excitation_estimate: Array2<f64>,
    /// Running sum for incremental updates
    running_sum: f64,
    /// Number of samples in current window
    sample_count: usize,
    /// Atomic flag for critical state
    is_critical: AtomicBool,
    /// Counter for critical state duration
    critical_duration_ns: AtomicU64,
    /// Last update timestamp
    last_update_ns: u64,
    /// Decay factor for exponential weighting
    decay_factor: f64,
}

impl BranchingRatioCalculator {
    /// Create new branching ratio calculator
    pub fn new(dimensions: usize, decay_factor: f64) -> Self {
        assert!(decay_factor > 0.0 && decay_factor <= 1.0);
        
        Self {
            samples: VecDeque::with_capacity(MAX_WINDOW_SIZE),
            excitation_estimate: Array2::zeros((dimensions, dimensions)),
            running_sum: 0.0,
            sample_count: 0,
            is_critical: AtomicBool::new(false),
            critical_duration_ns: AtomicU64::new(0),
            last_update_ns: 0,
            decay_factor,
        }
    }
    
    /// Update excitation matrix estimate from observed events
    pub fn update_excitation_estimate(
        &mut self,
        event_counts: &Array2<u64>,
        total_events: u64,
    ) {
        if total_events == 0 {
            return;
        }
        
        let scale = 1.0 / total_events as f64;
        
        for i in 0..event_counts.nrows() {
            for j in 0..event_counts.ncols() {
                let old_val = self.excitation_estimate[[i, j]];
                let new_val = event_counts[[i, j]] as f64 * scale;
                
                // Exponential moving average update
                self.excitation_estimate[[i, j]] = 
                    self.decay_factor * new_val + (1.0 - self.decay_factor) * old_val;
            }
        }
    }
    
    /// Calculate branching ratio from excitation matrix
    /// Returns (overall_ratio, per_dimension_ratios, spectral_radius)
    pub fn calculate_branching_ratio(&self) -> (f64, Array1<f64>, f64) {
        let dimensions = self.excitation_estimate.nrows();
        
        // Per-dimension branching ratios (sum of incoming excitations)
        let mut dim_ratios = Array1::zeros(dimensions);
        let mut total_ratio = 0.0;
        
        for j in 0..dimensions {
            let mut col_sum = 0.0;
            for i in 0..dimensions {
                col_sum += self.excitation_estimate[[i, j]];
            }
            dim_ratios[j] = col_sum;
            total_ratio += col_sum;
        }
        
        // Average branching ratio
        let avg_ratio = total_ratio / dimensions as f64;
        
        // Spectral radius approximation (power iteration would be more accurate but slower)
        // Using Frobenius norm as upper bound for spectral radius
        let mut frobenius_norm_sq = 0.0;
        for i in 0..dimensions {
            for j in 0..dimensions {
                let val = self.excitation_estimate[[i, j]];
                frobenius_norm_sq += val * val;
            }
        }
        let spectral_radius = frobenius_norm_sq.sqrt();
        
        (avg_ratio, dim_ratios, spectral_radius)
    }
    
    /// Record new branching ratio sample with timestamp
    pub fn record_sample(
        &mut self,
        timestamp_ns: u64,
        branching_ratio: f64,
        dimension_ratios: Array1<f64>,
        spectral_radius: f64,
    ) {
        let sample = BranchingSample {
            timestamp_ns,
            branching_ratio,
            dimension_ratios,
            spectral_radius,
        };
        
        // Remove old samples outside window
        while let Some(front) = self.samples.front() {
            if timestamp_ns.saturating_sub(front.timestamp_ns) > RECENT_WINDOW_NS {
                if let Some(removed) = self.samples.pop_front() {
                    self.running_sum -= removed.branching_ratio;
                    self.sample_count = self.sample_count.saturating_sub(1);
                }
            } else {
                break;
            }
        }
        
        // Add new sample
        if self.samples.len() >= MAX_WINDOW_SIZE {
            if let Some(removed) = self.samples.pop_front() {
                self.running_sum -= removed.branching_ratio;
                self.sample_count = self.sample_count.saturating_sub(1);
            }
        }
        
        self.running_sum += branching_ratio;
        self.sample_count += 1;
        self.samples.push_back(sample);
        self.last_update_ns = timestamp_ns;
        
        // Update critical state
        self.update_critical_state(branching_ratio, timestamp_ns);
    }
    
    /// Update critical state detection
    fn update_critical_state(&mut self, current_ratio: f64, timestamp_ns: u64) {
        let was_critical = self.is_critical.load(Ordering::Relaxed);
        let is_now_critical = current_ratio >= CRITICAL_THRESHOLD;
        
        self.is_critical.store(is_now_critical, Ordering::Relaxed);
        
        if is_now_critical {
            if was_critical {
                // Continue counting critical duration
                let prev_duration = self.critical_duration_ns.load(Ordering::Relaxed);
                self.critical_duration_ns.store(
                    prev_duration.saturating_add(timestamp_ns.saturating_sub(self.last_update_ns)),
                    Ordering::Relaxed,
                );
            } else {
                // Just entered critical state
                self.critical_duration_ns.store(0, Ordering::Relaxed);
            }
        } else {
            // Reset critical duration when leaving critical state
            self.critical_duration_ns.store(0, Ordering::Relaxed);
        }
    }
    
    /// Get current branching ratio statistics
    pub fn get_statistics(&self) -> BranchingStatistics {
        let (avg_ratio, dim_ratios, spectral_radius) = self.calculate_branching_ratio();
        
        let window_ratio = if self.sample_count > 0 {
            self.running_sum / self.sample_count as f64
        } else {
            avg_ratio
        };
        
        let is_critical = self.is_critical.load(Ordering::Relaxed);
        let critical_duration_ns = self.critical_duration_ns.load(Ordering::Relaxed);
        
        // Calculate variance if we have enough samples
        let variance = if self.sample_count > 1 {
            let mean = window_ratio;
            let mut sum_sq_diff = 0.0;
            for sample in &self.samples {
                let diff = sample.branching_ratio - mean;
                sum_sq_diff += diff * diff;
            }
            sum_sq_diff / (self.sample_count - 1) as f64
        } else {
            0.0
        };
        
        BranchingStatistics {
            instantaneous_ratio: avg_ratio,
            window_average_ratio: window_ratio,
            dimension_ratios: dim_ratios,
            spectral_radius,
            variance,
            is_critical,
            critical_duration_ns,
            sample_count: self.sample_count,
        }
    }
    
    /// Check if flash crash is imminent
    pub fn is_flash_crash_imminent(&self) -> bool {
        let stats = self.get_statistics();
        stats.instantaneous_ratio >= FLASH_CRASH_THRESHOLD ||
        (stats.is_critical && stats.critical_duration_ns > 500_000_000) // Critical for > 500ms
    }
    
    /// Get early warning signal strength (0.0 to 1.0)
    pub fn warning_signal_strength(&self) -> f64 {
        let stats = self.get_statistics();
        
        if stats.instantaneous_ratio >= FLASH_CRASH_THRESHOLD {
            return 1.0;
        }
        
        if stats.instantaneous_ratio >= CRITICAL_THRESHOLD {
            // Linear interpolation between critical and flash crash thresholds
            let normalized = (stats.instantaneous_ratio - CRITICAL_THRESHOLD) 
                / (FLASH_CRASH_THRESHOLD - CRITICAL_THRESHOLD);
            return normalized.min(1.0);
        }
        
        // Below critical threshold but still provide some signal
        (stats.instantaneous_ratio / CRITICAL_THRESHOLD).min(1.0) * 0.5
    }
    
    /// Reset calculator state
    pub fn reset(&mut self) {
        self.samples.clear();
        self.running_sum = 0.0;
        self.sample_count = 0;
        self.is_critical.store(false, Ordering::Relaxed);
        self.critical_duration_ns.store(0, Ordering::Relaxed);
        self.excitation_estimate.fill(0.0);
    }
}

/// Branching ratio statistics snapshot
#[derive(Debug, Clone)]
pub struct BranchingStatistics {
    /// Instantaneous branching ratio from current excitation matrix
    pub instantaneous_ratio: f64,
    /// Average branching ratio over recent window
    pub window_average_ratio: f64,
    /// Per-dimension branching ratios
    pub dimension_ratios: Array1<f64>,
    /// Spectral radius of excitation matrix
    pub spectral_radius: f64,
    /// Variance of branching ratio in recent window
    pub variance: f64,
    /// Whether system is in critical state
    pub is_critical: bool,
    /// Duration of current critical state in nanoseconds
    pub critical_duration_ns: u64,
    /// Number of samples in window
    pub sample_count: usize,
}

impl BranchingStatistics {
    /// Get risk level classification
    pub fn risk_level(&self) -> RiskLevel {
        if self.instantaneous_ratio >= FLASH_CRASH_THRESHOLD {
            RiskLevel::Critical
        } else if self.is_critical {
            RiskLevel::High
        } else if self.instantaneous_ratio >= CRITICAL_THRESHOLD * 0.8 {
            RiskLevel::Elevated
        } else {
            RiskLevel::Normal
        }
    }
    
    /// Generate alert message if needed
    pub fn generate_alert(&self) -> Option<String> {
        match self.risk_level() {
            RiskLevel::Critical => {
                Some(format!(
                    "CRITICAL: Flash crash imminent! Branching ratio {:.3} exceeds 1.0. \
                     Critical duration: {:.2}ms",
                    self.instantaneous_ratio,
                    self.critical_duration_ns as f64 / 1_000_000.0
                ))
            }
            RiskLevel::High => {
                Some(format!(
                    "HIGH RISK: Market in critical state. Branching ratio {:.3}. \
                     Duration: {:.2}ms",
                    self.instantaneous_ratio,
                    self.critical_duration_ns as f64 / 1_000_000.0
                ))
            }
            RiskLevel::Elevated => {
                Some(format!(
                    "ELEVATED: Approaching critical state. Branching ratio {:.3}",
                    self.instantaneous_ratio
                ))
            }
            RiskLevel::Normal => None,
        }
    }
}

/// Risk level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Normal,
    Elevated,
    High,
    Critical,
}

/// Multi-timescale branching ratio analyzer
pub struct MultiTimescaleAnalyzer {
    /// Fast timescale calculator (milliseconds)
    fast_calculator: BranchingRatioCalculator,
    /// Medium timescale calculator (seconds)
    medium_calculator: BranchingRatioCalculator,
    /// Slow timescale calculator (minutes)
    slow_calculator: BranchingRatioCalculator,
}

impl MultiTimescaleAnalyzer {
    /// Create analyzer with multiple timescales
    pub fn new(dimensions: usize) -> Self {
        Self {
            fast_calculator: BranchingRatioCalculator::new(dimensions, 0.9),  // Fast decay
            medium_calculator: BranchingRatioCalculator::new(dimensions, 0.5), // Medium decay
            slow_calculator: BranchingRatioCalculator::new(dimensions, 0.1),   // Slow decay
        }
    }
    
    /// Update all timescale calculators
    pub fn update(
        &mut self,
        timestamp_ns: u64,
        event_counts: &Array2<u64>,
        total_events: u64,
    ) {
        // Update excitation estimates
        self.fast_calculator.update_excitation_estimate(event_counts, total_events);
        self.medium_calculator.update_excitation_estimate(event_counts, total_events);
        self.slow_calculator.update_excitation_estimate(event_counts, total_events);
        
        // Calculate and record branching ratios for each timescale
        for calculator in [&mut self.fast_calculator, &mut self.medium_calculator, &mut self.slow_calculator] {
            let (ratio, dim_ratios, spectral_radius) = calculator.calculate_branching_ratio();
            calculator.record_sample(timestamp_ns, ratio, dim_ratios, spectral_radius);
        }
    }
    
    /// Get comprehensive analysis across all timescales
    pub fn get_comprehensive_analysis(&self) -> MultiTimescaleAnalysis {
        MultiTimescaleAnalysis {
            fast: self.fast_calculator.get_statistics(),
            medium: self.medium_calculator.get_statistics(),
            slow: self.slow_calculator.get_statistics(),
            flash_crash_imminent: self.fast_calculator.is_flash_crash_imminent(),
            warning_strength: self.fast_calculator.warning_signal_strength(),
        }
    }
    
    /// Check if any timescale indicates critical state
    pub fn any_critical(&self) -> bool {
        self.fast_calculator.get_statistics().is_critical ||
        self.medium_calculator.get_statistics().is_critical ||
        self.slow_calculator.get_statistics().is_critical
    }
}

/// Multi-timescale analysis result
#[derive(Debug, Clone)]
pub struct MultiTimescaleAnalysis {
    pub fast: BranchingStatistics,
    pub medium: BranchingStatistics,
    pub slow: BranchingStatistics,
    pub flash_crash_imminent: bool,
    pub warning_strength: f64,
}

impl MultiTimescaleAnalysis {
    /// Get regime classification based on timescale divergence
    pub fn regime_type(&self) -> RegimeType {
        let fast_slow_diff = self.fast.instantaneous_ratio - self.slow.instantaneous_ratio;
        
        if fast_slow_diff > 0.3 {
            RegimeType::Turbulent  // Fast much higher than slow
        } else if fast_slow_diff < -0.1 {
            RegimeType::Stabilizing // Fast lower than slow (recovering)
        } else if self.fast.instantaneous_ratio > 0.8 {
            RegimeType::Stressed   // Both high
        } else {
            RegimeType::Normal
        }
    }
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegimeType {
    Normal,
    Stressed,
    Turbulent,
    Stabilizing,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_branching_ratio_calculation() {
        let mut calc = BranchingRatioCalculator::new(4, 0.5);
        
        // Set up excitation matrix with known branching ratio
        let mut excitation = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                excitation[[i, j]] = 0.2; // Each element contributes 0.2
            }
        }
        calc.excitation_estimate = excitation;
        
        let (avg_ratio, _, spectral_radius) = calc.calculate_branching_ratio();
        
        // With 0.2 in each cell of 4x4 matrix, column sums are 0.8
        // Average should be around 0.8
        assert!(avg_ratio > 0.7 && avg_ratio < 0.9);
    }
    
    #[test]
    fn test_critical_state_detection() {
        let mut calc = BranchingRatioCalculator::new(4, 0.9);
        
        // Simulate approaching critical state
        let base_time = 1_000_000_000_000u64;
        for i in 0..100 {
            let timestamp = base_time + i as u64 * 10_000_000; // 10ms intervals
            
            // Gradually increase branching ratio
            let ratio = 0.8 + (i as f64 / 100.0) * 0.3; // Goes from 0.8 to 1.1
            
            calc.record_sample(
                timestamp,
                ratio,
                Array1::from_vec(vec![ratio / 4.0; 4]),
                ratio,
            );
        }
        
        let stats = calc.get_statistics();
        assert!(stats.is_critical || stats.instantaneous_ratio >= CRITICAL_THRESHOLD);
    }
    
    #[test]
    fn test_memory_bounded_window() {
        let mut calc = BranchingRatioCalculator::new(4, 0.5);
        let base_time = 1_000_000_000_000u64;
        
        // Add more samples than window capacity
        for i in 0..MAX_WINDOW_SIZE + 1000 {
            calc.record_sample(
                base_time + i as u64 * 1_000_000,
                0.5,
                Array1::from_vec(vec![0.125; 4]),
                0.5,
            );
        }
        
        // Window should not exceed capacity
        assert!(calc.samples.len() <= MAX_WINDOW_SIZE);
    }
    
    #[test]
    fn test_multi_timescale_analyzer() {
        let mut analyzer = MultiTimescaleAnalyzer::new(4);
        let base_time = 1_000_000_000_000u64;
        
        let mut event_counts = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                event_counts[[i, j]] = 100;
            }
        }
        
        analyzer.update(base_time, &event_counts, 1600);
        
        let analysis = analyzer.get_comprehensive_analysis();
        
        // All timescales should have valid statistics
        assert!(analysis.fast.sample_count > 0);
        assert!(analysis.medium.sample_count > 0);
        assert!(analysis.slow.sample_count > 0);
    }
}
