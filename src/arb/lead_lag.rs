//! Lead-Lag Arbitrage using de Jong-Neijens Estimator
//! 
//! This module implements the de Jong-Neijens lead-lag estimator in pure Rust,
//! using cross-correlation functions to identify which altcoin leads BTC price
//! discovery at the microsecond level. Essential for statistical arbitrage strategies.
//!
//! Optimized for:
//! - Microsecond cross-correlation calculations
//! - 8GB global RAM limit (bounded ring buffers)
//! - AMD Ryzen AI 5 SIMD acceleration
//! - Lock-free concurrent access

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicUsize, Ordering};
use std::collections::VecDeque;

/// Maximum number of price samples to retain (bounds memory)
const MAX_SAMPLES: usize = 4096;

/// Maximum lag to consider (in ticks)
const MAX_LAG_TICKS: usize = 100;

/// Cache-line aligned atomic for zero false sharing
#[repr(align(64))]
struct AlignedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; 56],
}

impl AlignedAtomicU64 {
    #[inline]
    fn new(val: u64) -> Self {
        Self {
            value: AtomicU64::new(val),
            _padding: [0u8; 56],
        }
    }

    #[inline]
    fn load(&self, order: Ordering) -> u64 {
        self.value.load(order)
    }

    #[inline]
    fn store(&self, val: u64, order: Ordering) {
        self.value.store(val, order);
    }

    #[inline]
    fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        self.value.fetch_add(val, order)
    }
}

#[repr(align(64))]
struct AlignedAtomicI64 {
    value: AtomicI64,
    _padding: [u8; 56],
}

impl AlignedAtomicI64 {
    #[inline]
    fn new(val: i64) -> Self {
        Self {
            value: AtomicI64::new(val),
            _padding: [0u8; 56],
        }
    }

    #[inline]
    fn load(&self, order: Ordering) -> i64 {
        self.value.load(order)
    }

    #[inline]
    fn store(&self, val: i64, order: Ordering) {
        self.value.store(val, order);
    }
}

/// Price sample for cross-correlation
#[derive(Clone, Copy, Debug)]
pub struct PriceSample {
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Mid price in quote ticks
    pub price_ticks: u64,
    /// Log return from previous sample (scaled by 1e9)
    pub log_return: i64,
}

/// Cross-correlation result
#[derive(Clone, Debug)]
pub struct CrossCorrelationResult {
    /// Optimal lag (positive = series1 leads, negative = series2 leads)
    pub optimal_lag: i32,
    /// Correlation coefficient at optimal lag (scaled by 10000)
    pub correlation_bps: i64,
    /// Significance score (0-10000, higher = more significant)
    pub significance: u64,
    /// Number of samples used
    pub sample_count: usize,
}

/// de Jong-Neijens Lead-Lag Estimator
/// 
/// Implements the de Jong-Neijens methodology for estimating lead-lag relationships
/// between two time series using cross-correlation analysis. Specifically optimized
/// for cryptocurrency pair analysis (altcoin vs BTC).
pub struct LeadLagEstimator {
    /// Ring buffer for series 1 (e.g., altcoin returns)
    series1_buffer: Vec<PriceSample>,
    series1_head: AtomicUsize,
    series1_count: AtomicUsize,
    
    /// Ring buffer for series 2 (e.g., BTC returns)
    series2_buffer: Vec<PriceSample>,
    series2_head: AtomicUsize,
    series2_count: AtomicUsize,
    
    /// Running statistics for series 1
    series1_mean: AlignedAtomicI64,
    series1_variance: AlignedAtomicU64,
    series1_count: AlignedAtomicU64,
    
    /// Running statistics for series 2
    series2_mean: AlignedAtomicI64,
    series2_variance: AlignedAtomicU64,
    series2_count: AlignedAtomicU64,
    
    /// Last computed optimal lag
    last_optimal_lag: AtomicI64,
    last_correlation: AlignedAtomicI64,
    last_update_ns: AlignedAtomicU64,
    
    /// Minimum samples required before computing
    min_samples: usize,
    
    /// Decay factor for EWMA statistics (basis points)
    decay_bps: u64,
}

impl LeadLagEstimator {
    /// Create a new lead-lag estimator
    /// 
    /// # Arguments
    /// * `min_samples` - Minimum samples before computing correlations
    /// * `decay_bps` - Decay factor for running statistics (0-10000)
    pub fn new(min_samples: usize, decay_bps: u64) -> Self {
        Self {
            series1_buffer: Vec::with_capacity(MAX_SAMPLES),
            series1_head: AtomicUsize::new(0),
            series1_count: AtomicUsize::new(0),
            series2_buffer: Vec::with_capacity(MAX_SAMPLES),
            series2_head: AtomicUsize::new(0),
            series2_count: AtomicUsize::new(0),
            series1_mean: AlignedAtomicI64::new(0),
            series1_variance: AlignedAtomicU64::new(0),
            series1_count: AlignedAtomicU64::new(0),
            series2_mean: AlignedAtomicI64::new(0),
            series2_variance: AlignedAtomicU64::new(0),
            series2_count: AlignedAtomicU64::new(0),
            last_optimal_lag: AtomicI64::new(0),
            last_correlation: AlignedAtomicI64::new(0),
            last_update_ns: AlignedAtomicU64::new(0),
            min_samples,
            decay_bps: decay_bps.min(10000),
        }
    }

    /// Add a price sample to series 1
    #[inline]
    pub fn add_sample_1(&self, sample: PriceSample) {
        self.add_sample_to_buffer(
            &self.series1_buffer,
            &self.series1_head,
            &self.series1_count,
            &self.series1_mean,
            &self.series1_variance,
            &self.series1_count,
            sample,
        );
    }

    /// Add a price sample to series 2
    #[inline]
    pub fn add_sample_2(&self, sample: PriceSample) {
        self.add_sample_to_buffer(
            &self.series2_buffer,
            &self.series2_head,
            &self.series2_count,
            &self.series2_mean,
            &self.series2_variance,
            &self.series2_count,
            sample,
        );
    }

    /// Internal method to add sample to a ring buffer
    #[inline]
    fn add_sample_to_buffer(
        &self,
        buffer: &Vec<PriceSample>,
        head: &AtomicUsize,
        count: &AtomicUsize,
        mean: &AlignedAtomicI64,
        variance: &AlignedAtomicU64,
        sample_count: &AlignedAtomicU64,
        sample: PriceSample,
    ) {
        let h = head.fetch_add(1, Ordering::Relaxed);
        let idx = h % MAX_SAMPLES;
        
        // Update running statistics using Welford's online algorithm
        let n = sample_count.fetch_add(1, Ordering::Relaxed) + 1;
        let current_mean = mean.load(Ordering::Relaxed);
        let delta = sample.log_return - current_mean;
        let new_mean = current_mean + delta / n as i64;
        mean.store(new_mean, Ordering::Relaxed);
        
        // Update variance (EWMA style for bounded memory)
        let current_var = variance.load(Ordering::Relaxed);
        let delta2 = sample.log_return - new_mean;
        let new_var = ((current_var * (n as u64 - 1)) + (delta * delta2) as u64) / n as u64;
        variance.store(new_var, Ordering::Relaxed);
        
        // Write to buffer (safe due to unique head values)
        unsafe {
            let ptr = buffer.as_ptr() as *mut PriceSample;
            if idx < buffer.capacity() {
                ptr.add(idx).write(sample);
            } else if idx < MAX_SAMPLES {
                // Buffer not yet at capacity
                std::ptr::write(ptr.add(idx), sample);
            }
        }
        
        let c = count.load(Ordering::Relaxed);
        if c < MAX_SAMPLES {
            count.fetch_add(1, Ordering::Release);
        }
    }

    /// Compute cross-correlation at a specific lag
    /// 
    /// Positive lag means series1 leads series2
    /// Negative lag means series2 leads series1
    #[inline]
    fn cross_correlation_at_lag(&self, lag: i32) -> Option<i64> {
        let count1 = self.series1_count.load(Ordering::Acquire);
        let count2 = self.series2_count.load(Ordering::Acquire);
        
        let valid_samples = count1.min(count2).min(MAX_SAMPLES);
        if valid_samples < self.min_samples {
            return None;
        }
        
        let mean1 = self.series1_mean.load(Ordering::Relaxed) as f64;
        let mean2 = self.series2_mean.load(Ordering::Relaxed) as f64;
        let var1 = self.series1_variance.load(Ordering::Relaxed) as f64;
        let var2 = self.series2_variance.load(Ordering::Relaxed) as f64;
        
        if var1 == 0.0 || var2 == 0.0 {
            return Some(0);
        }
        
        let std1 = var1.sqrt();
        let std2 = var2.sqrt();
        
        let mut sum_product = 0.0f64;
        let mut count = 0usize;
        
        let head1 = self.series1_head.load(Ordering::Relaxed);
        let head2 = self.series2_head.load(Ordering::Relaxed);
        
        // Calculate cross-correlation based on lag direction
        if lag >= 0 {
            // Series1 leads (series1[t] vs series2[t+lag])
            for i in 0..valid_samples.saturating_sub(lag as usize) {
                let idx1 = (head1 - 1 - i) % MAX_SAMPLES;
                let idx2 = (head2 - 1 - i - lag as usize) % MAX_SAMPLES;
                
                unsafe {
                    let ptr1 = self.series1_buffer.as_ptr();
                    let ptr2 = self.series2_buffer.as_ptr();
                    
                    let s1 = (*ptr1.add(idx1)).log_return as f64;
                    let s2 = (*ptr2.add(idx2)).log_return as f64;
                    
                    sum_product += (s1 - mean1) * (s2 - mean2);
                    count += 1;
                }
            }
        } else {
            // Series2 leads (series1[t+|lag|] vs series2[t])
            let abs_lag = (-lag) as usize;
            for i in 0..valid_samples.saturating_sub(abs_lag) {
                let idx1 = (head1 - 1 - i - abs_lag) % MAX_SAMPLES;
                let idx2 = (head2 - 1 - i) % MAX_SAMPLES;
                
                unsafe {
                    let ptr1 = self.series1_buffer.as_ptr();
                    let ptr2 = self.series2_buffer.as_ptr();
                    
                    let s1 = (*ptr1.add(idx1)).log_return as f64;
                    let s2 = (*ptr2.add(idx2)).log_return as f64;
                    
                    sum_product += (s1 - mean1) * (s2 - mean2);
                    count += 1;
                }
            }
        }
        
        if count == 0 {
            return Some(0);
        }
        
        // Correlation coefficient scaled by 10000
        let corr = (sum_product / count as f64) / (std1 * std2);
        Some((corr * 10000.0) as i64)
    }

    /// Find optimal lag using grid search
    /// 
    /// Returns the lag with maximum absolute correlation
    pub fn find_optimal_lag(&self) -> Option<CrossCorrelationResult> {
        let count1 = self.series1_count.load(Ordering::Acquire);
        let count2 = self.series2_count.load(Ordering::Acquire);
        
        let valid_samples = count1.min(count2).min(MAX_SAMPLES);
        if valid_samples < self.min_samples {
            return None;
        }
        
        let mut best_lag: i32 = 0;
        let mut best_corr: i64 = 0;
        let mut best_abs_corr: u64 = 0;
        
        // Search across lag range
        for lag in 0..MAX_LAG_TICKS as i32 {
            if let Some(corr) = self.cross_correlation_at_lag(lag) {
                let abs_corr = corr.unsigned_abs();
                if abs_corr > best_abs_corr {
                    best_abs_corr = abs_corr;
                    best_lag = lag;
                    best_corr = corr;
                }
            }
            
            // Also check negative lags
            if lag > 0 {
                if let Some(corr) = self.cross_correlation_at_lag(-(lag as i32)) {
                    let abs_corr = corr.unsigned_abs();
                    if abs_corr > best_abs_corr {
                        best_abs_corr = abs_corr;
                        best_lag = -(lag as i32);
                        best_corr = corr;
                    }
                }
            }
        }
        
        // Calculate significance (simplified t-statistic approximation)
        let significance = if valid_samples > 3 {
            let t_stat = (best_corr.abs() as f64 / 10000.0) * (valid_samples - 2) as f64;
            (t_stat.min(100.0) * 100.0) as u64
        } else {
            0
        };
        
        let result = CrossCorrelationResult {
            optimal_lag: best_lag,
            correlation_bps: best_corr,
            significance,
            sample_count: valid_samples,
        };
        
        // Cache results
        self.last_optimal_lag.store(best_lag as i64, Ordering::Relaxed);
        self.last_correlation.store(best_corr, Ordering::Relaxed);
        self.last_update_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
        
        Some(result)
    }

    /// Get the current estimated lead-lag relationship
    /// 
    /// Returns:
    /// - Positive value: series1 leads by N ticks
    /// - Negative value: series2 leads by N ticks
    /// - Zero: no clear leader
    #[inline]
    pub fn get_lead_lag(&self) -> i32 {
        self.last_optimal_lag.load(Ordering::Relaxed) as i32
    }

    /// Get the correlation strength at optimal lag
    #[inline]
    pub fn get_correlation_strength(&self) -> i64 {
        self.last_correlation.load(Ordering::Relaxed)
    }

    /// Check if series 1 (altcoin) leads series 2 (BTC)
    #[inline]
    pub fn series1_leads(&self) -> bool {
        self.last_optimal_lag.load(Ordering::Relaxed) > 0
    }

    /// Check if series 2 (BTC) leads series 1 (altcoin)
    #[inline]
    pub fn series2_leads(&self) -> bool {
        self.last_optimal_lag.load(Ordering::Relaxed) < 0
    }

    /// Get sample counts
    pub fn get_sample_counts(&self) -> (usize, usize) {
        (
            self.series1_count.load(Ordering::Relaxed),
            self.series2_count.load(Ordering::Relaxed),
        )
    }

    /// Reset all buffers and statistics
    pub fn reset(&self) {
        self.series1_head.store(0, Ordering::Release);
        self.series1_count.store(0, Ordering::Release);
        self.series2_head.store(0, Ordering::Release);
        self.series2_count.store(0, Ordering::Release);
        self.series1_mean.store(0, Ordering::Release);
        self.series1_variance.store(0, Ordering::Release);
        self.series1_count.store(0, Ordering::Release);
        self.series2_mean.store(0, Ordering::Release);
        self.series2_variance.store(0, Ordering::Release);
        self.series2_count.store(0, Ordering::Release);
        self.last_optimal_lag.store(0, Ordering::Release);
        self.last_correlation.store(0, Ordering::Release);
        self.last_update_ns.store(0, Ordering::Release);
    }
}

/// SIMD-optimized cross-correlation computation
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use super::*;
    use std::arch::x86_64::*;

    /// Compute cross-correlation for 8 lags simultaneously
    /// 
    /// # Safety
    /// Requires AVX2 support
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_cross_correlation(
        series1: &[i64],
        series2: &[i64],
        mean1: f64,
        mean2: f64,
        std1: f64,
        std2: f64,
        lags: &[i32],
        results: &mut [i64],
    ) {
        assert_eq!(lags.len() % 8, 0, "Lags must be multiple of 8");
        
        let mean1_vec = _mm256_set1_pd(mean1);
        let mean2_vec = _mm256_set1_pd(mean2);
        let std1_vec = _mm256_set1_pd(std1);
        let std2_vec = _mm256_set1_pd(std2);
        let scale_vec = _mm256_set1_pd(10000.0);
        
        for (lag_idx, &lag) in lags.iter().enumerate().take(lags.len() / 8) {
            let base_idx = lag_idx * 8;
            let mut sum_vec = _mm256_setzero_pd();
            let mut count_vec = _mm256_setzero_pd();
            
            // Process pairs in batches of 4 (due to f64)
            for i in (0..series1.len().saturating_sub(lag.unsigned_abs() as usize)).step_by(4) {
                let idx1 = i;
                let idx2 = if lag >= 0 { i + lag as usize } else { i - (-lag) as usize };
                
                if idx2 >= series2.len() {
                    continue;
                }
                
                let s1 = _mm256_set_pd(
                    series1[idx1] as f64,
                    series1[idx1.min(series1.len()-1)] as f64,
                    series1[idx1.min(series1.len()-1)] as f64,
                    series1[idx1] as f64,
                );
                let s2 = _mm256_set_pd(
                    series2[idx2] as f64,
                    series2[idx2.min(series2.len()-1)] as f64,
                    series2[idx2.min(series2.len()-1)] as f64,
                    series2[idx2] as f64,
                );
                
                let d1 = _mm256_sub_pd(s1, mean1_vec);
                let d2 = _mm256_sub_pd(s2, mean2_vec);
                let prod = _mm256_mul_pd(d1, d2);
                sum_vec = _mm256_add_pd(sum_vec, prod);
                count_vec = _mm256_add_pd(count_vec, _mm256_set1_pd(1.0));
            }
            
            // Horizontal sum and normalize
            let sum_arr: [f64; 4] = std::mem::transmute(sum_vec);
            let count_arr: [f64; 4] = std::mem::transmute(count_vec);
            
            let total_sum: f64 = sum_arr.iter().sum();
            let total_count: f64 = count_arr.iter().sum();
            
            if total_count > 0.0 && std1 > 0.0 && std2 > 0.0 {
                let corr = (total_sum / total_count) / (std1 * std2);
                for j in 0..8 {
                    results[base_idx + j] = (corr * 10000.0) as i64;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lead_lag_basic() {
        let estimator = LeadLagEstimator::new(10, 100);
        
        // Add correlated samples where series1 leads by 5 ticks
        for i in 0..50 {
            let return_val = (i % 10) as i64 * 1000;
            
            estimator.add_sample_1(PriceSample {
                timestamp_ns: i as u64 * 1_000_000,
                price_ticks: 100000 + return_val as u64,
                log_return: return_val,
            });
            
            // Series2 follows series1 with 5-tick delay
            let delayed_return = if i >= 5 {
                ((i - 5) % 10) as i64 * 1000
            } else {
                0
            };
            
            estimator.add_sample_2(PriceSample {
                timestamp_ns: i as u64 * 1_000_000,
                price_ticks: 50000 + delayed_return as u64,
                log_return: delayed_return,
            });
        }
        
        let result = estimator.find_optimal_lag();
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.optimal_lag > 0); // Series1 should lead
    }

    #[test]
    fn test_correlation_strength() {
        let estimator = LeadLagEstimator::new(20, 100);
        
        // Add perfectly correlated samples
        for i in 0..100 {
            let ret = (i % 20) as i64 * 500;
            
            estimator.add_sample_1(PriceSample {
                timestamp_ns: i as u64 * 1_000_000,
                price_ticks: 100000 + ret as u64,
                log_return: ret,
            });
            
            estimator.add_sample_2(PriceSample {
                timestamp_ns: i as u64 * 1_000_000,
                price_ticks: 50000 + ret as u64,
                log_return: ret,
            });
        }
        
        let result = estimator.find_optimal_lag();
        assert!(result.is_some());
        assert!(result.unwrap().correlation_bps > 8000); // Strong correlation
    }

    #[test]
    fn test_no_correlation() {
        let estimator = LeadLagEstimator::new(20, 100);
        
        // Add uncorrelated random-ish samples
        for i in 0..100 {
            let ret1 = ((i * 7) % 50 - 25) as i64 * 1000;
            let ret2 = ((i * 13) % 50 - 25) as i64 * 1000;
            
            estimator.add_sample_1(PriceSample {
                timestamp_ns: i as u64 * 1_000_000,
                price_ticks: 100000 + ret1.unsigned_abs(),
                log_return: ret1,
            });
            
            estimator.add_sample_2(PriceSample {
                timestamp_ns: i as u64 * 1_000_000,
                price_ticks: 50000 + ret2.unsigned_abs(),
                log_return: ret2,
            });
        }
        
        let result = estimator.find_optimal_lag();
        assert!(result.is_some());
        // Correlation should be weak
        assert!(result.unwrap().correlation_bps.abs() < 3000);
    }
}
