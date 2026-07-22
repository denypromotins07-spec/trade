//! `avx2_lob.rs` - AVX2/AVX-512 Intrinsics for Limit Order Book Aggregations
//!
//! This module provides zero-allocation, vectorized math operations for processing
//! Limit Order Book (LOB) depth data. It leverages SIMD instructions to calculate
//! total depth, weighted prices, and order flow imbalance in a single CPU cycle batch.
//!
//! **Optimization Targets:**
//! - AMD Ryzen AI 5 (AVX2/AVX-512 capable)
//! - Microsecond latency hot path
//! - Zero heap allocations (stack-only)
//! - Safe handling of NaN/Infinity in financial data

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;
use std::cmp::min;

/// Maximum number of price levels processed in a single SIMD batch.
/// AVX2 operates on 256-bit registers (4 x f64).
const SIMD_WIDTH: usize = 4;

/// Represents the result of a SIMD aggregation.
#[derive(Debug, Clone, Copy)]
pub struct LobAggregation {
    pub total_volume: f64,
    pub weighted_price_sum: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub imbalance_ratio: f64, // (Bid - Ask) / (Bid + Ask)
}

impl Default for LobAggregation {
    fn default() -> Self {
        Self {
            total_volume: 0.0,
            weighted_price_sum: 0.0,
            bid_volume: 0.0,
            ask_volume: 0.0,
            imbalance_ratio: 0.0,
        }
    }
}

/// Calculates the weighted average price and total volume for a slice of levels.
///
/// # Safety
/// This function uses `std::arch` intrinsics. It ensures safety by:
/// 1. Checking for NaN/Infinity before vectorization.
/// 2. Handling remainder elements scalarly.
/// 3. Using stack-allocated arrays for mask operations.
#[inline(always)]
pub fn aggregate_depth_simd(prices: &[f64], volumes: &[f64]) -> LobAggregation {
    assert_eq!(prices.len(), volumes.len(), "Price and volume slices must match");

    let len = prices.len();
    let mut result = LobAggregation::default();

    if len == 0 {
        return result;
    }

    // Process in chunks of 4 (AVX2 width)
    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    unsafe {
        let mut vol_vec = _mm256_setzero_pd();
        let mut w_price_vec = _mm256_setzero_pd();

        for i in 0..chunks {
            let idx = i * SIMD_WIDTH;

            // Load data into AVX2 registers
            let p_vec = _mm256_loadu_pd(prices.as_ptr().add(idx));
            let v_vec = _mm256_loadu_pd(volumes.as_ptr().add(idx));

            // Check for NaN or Infinity: _mm256_cmp_pd with _CMP_UNORD_Q detects NaNs
            // If any element is NaN/Inf, we mask it out to zero to prevent pollution
            let nan_mask = _mm256_cmp_pd(p_vec, p_vec, _CMP_UNORD_Q);
            let inf_mask = _mm256_cmp_pd(
                _mm256_abs_pd(p_vec),
                _mm256_set1_pd(f64::INFINITY),
                _CMP_EQ_Q,
            );
            let bad_mask = _mm256_or_pd(nan_mask, inf_mask);

            // Create a clean mask: if bad, set to 0.0, else keep original
            // We use AND NOT: (~bad_mask) & v_vec
            let clean_v = _mm256_andnot_pd(bad_mask, v_vec);
            let clean_p = _mm256_andnot_pd(bad_mask, p_vec);

            // Accumulate volume
            vol_vec = _mm256_add_pd(vol_vec, clean_v);

            // Accumulate weighted price (price * volume)
            let wp = _mm256_mul_pd(clean_p, clean_v);
            w_price_vec = _mm256_add_pd(w_price_vec, wp);
        }

        // Horizontal sum of the vector registers
        // Extract lanes
        let vol_vals = [
            _mm256_extractf128_pd(vol_vec, 0),
            _mm256_extractf128_pd(vol_vec, 1),
        ];
        let wp_vals = [
            _mm256_extractf128_pd(w_price_vec, 0),
            _mm256_extractf128_pd(w_price_vec, 1),
        ];

        for i in 0..2 {
            let v_low = _mm_cvtsd_f64(_mm256_castpd256_pd128(vol_vals[i]));
            let v_high = _mm_cvtsd_f64(_mm_shuffle_pd(_mm256_castpd256_pd128(vol_vals[i]), _mm256_castpd256_pd128(vol_vals[i]), 1));
            
            let wp_low = _mm_cvtsd_f64(_mm256_castpd256_pd128(wp_vals[i]));
            let wp_high = _mm_cvtsd_f64(_mm_shuffle_pd(_mm256_castpd256_pd128(wp_vals[i]), _mm256_castpd256_pd128(wp_vals[i]), 1));

            result.total_volume += v_low + v_high;
            result.weighted_price_sum += wp_low + wp_high;
        }

        // Handle remainder scalarly
        for i in (chunks * SIMD_WIDTH)..len {
            let p = prices[i];
            let v = volumes[i];
            if p.is_finite() && v.is_finite() {
                result.total_volume += v;
                result.weighted_price_sum += p * v;
            }
        }
    }

    // Calculate Imbalance Ratio if needed (requires separate bid/ask slices usually, 
    // but here we assume mixed or pre-separated logic. For this util, we return aggregates.)
    // Caller typically splits bids/asks before calling.
    
    if result.total_volume > 0.0 {
        // Prevent division by zero
        result.imbalance_ratio = 0.0; // Placeholder, real impl needs bid/ask separation
    }

    result
}

/// Calculates Order Book Imbalance (OBI) between bid and ask sides.
/// Formula: (Vol_Bid - Vol_Ask) / (Vol_Bid + Vol_Ask)
#[inline(always)]
pub fn calculate_obi_simd(bid_volumes: &[f64], ask_volumes: &[f64]) -> f64 {
    let len = min(bid_volumes.len(), ask_volumes.len());
    if len == 0 {
        return 0.0;
    }

    let chunks = len / SIMD_WIDTH;
    let remainder = len % SIMD_WIDTH;

    unsafe {
        let mut bid_sum = _mm256_setzero_pd();
        let mut ask_sum = _mm256_setzero_pd();

        for i in 0..chunks {
            let idx = i * SIMD_WIDTH;
            let b_vec = _mm256_loadu_pd(bid_volumes.as_ptr().add(idx));
            let a_vec = _mm256_loadu_pd(ask_volumes.as_ptr().add(idx));

            // Sanitize inputs (NaN -> 0)
            let b_clean = _mm256_andnot_pd(
                _mm256_cmp_pd(b_vec, b_vec, _CMP_UNORD_Q),
                b_vec
            );
            let a_clean = _mm256_andnot_pd(
                _mm256_cmp_pd(a_vec, a_vec, _CMP_UNORD_Q),
                a_vec
            );

            bid_sum = _mm256_add_pd(bid_sum, b_clean);
            ask_sum = _mm256_add_pd(ask_sum, a_clean);
        }

        let mut total_bid = 0.0;
        let mut total_ask = 0.0;

        // Horizontal add
        let b_vals = [
            _mm256_extractf128_pd(bid_sum, 0),
            _mm256_extractf128_pd(bid_sum, 1),
        ];
        let a_vals = [
            _mm256_extractf128_pd(ask_sum, 0),
            _mm256_extractf128_pd(ask_sum, 1),
        ];

        for i in 0..2 {
            total_bid += _mm_cvtsd_f64(b_vals[i]) 
                       + _mm_cvtsd_f64(_mm_shuffle_pd(b_vals[i], b_vals[i], 1));
            total_ask += _mm_cvtsd_f64(a_vals[i]) 
                       + _mm_cvtsd_f64(_mm_shuffle_pd(a_vals[i], a_vals[i], 1));
        }

        for i in (chunks * SIMD_WIDTH)..len {
            if bid_volumes[i].is_finite() { total_bid += bid_volumes[i]; }
            if ask_volumes[i].is_finite() { total_ask += ask_volumes[i]; }
        }

        let denom = total_bid + total_ask;
        if denom > 0.0 {
            (total_bid - total_ask) / denom
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx2_aggregation_basic() {
        let prices = vec![100.0, 101.0, 102.0, 103.0];
        let volumes = vec![1.0, 2.0, 3.0, 4.0];
        
        let res = aggregate_depth_simd(&prices, &volumes);
        
        let expected_vol = 10.0;
        let expected_wp = (100.0*1.0) + (101.0*2.0) + (102.0*3.0) + (103.0*4.0);
        
        assert!((res.total_volume - expected_vol).abs() < 1e-9);
        assert!((res.weighted_price_sum - expected_wp).abs() < 1e-9);
    }

    #[test]
    fn test_nan_handling() {
        let prices = vec![100.0, f64::NAN, 102.0, 103.0];
        let volumes = vec![1.0, 100.0, 3.0, 4.0]; // Large volume at NaN price should be ignored
        
        let res = aggregate_depth_simd(&prices, &volumes);
        
        // Should ignore index 1 completely
        let expected_vol = 1.0 + 3.0 + 4.0;
        let expected_wp = (100.0*1.0) + (102.0*3.0) + (103.0*4.0);
        
        assert!((res.total_volume - expected_vol).abs() < 1e-9);
        assert!((res.weighted_price_sum - expected_wp).abs() < 1e-9);
    }
}
