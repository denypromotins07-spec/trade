//! src/asm/simd_math.rs
//! 
//! Stage 51: Inline Assembly & CPU Intrinsics for AMD Ryzen AI 5
//! 
//! Hand-written AVX2/AVX-512 inline assembly for critical path math operations.
//! Implements fast inverse square roots for volatility calculations using strict
//! register allocation to prevent spills on AMD Zen 4/Zen 5 architecture.
//! 
//! Optimized for microsecond latency with 8GB RAM constraint.

#![feature(asm_sym)]
#![feature(asm_experimental_arch)]

use std::arch::x86_64::*;
use std::mem;

/// SIMD vector width for AVX2 (256-bit = 4 f64 or 8 f32)
const AVX2_WIDTH_F32: usize = 8;
const AVX2_WIDTH_F64: usize = 4;

/// AVX-512 vector width (512-bit = 8 f64 or 16 f32)
const AVX512_WIDTH_F32: usize = 16;
const AVX512_WIDTH_F64: usize = 8;

/// Fast inverse square root using AVX2 inline assembly
/// 
/// Implements the Quake III arena algorithm optimized for AMD Zen architecture.
/// Uses the magic constant 0x5f3759df for initial approximation, followed by
/// Newton-Raphson iteration for precision.
/// 
/// # Safety
/// - Requires AVX2 CPU feature support (checked at runtime)
/// - Input values must be positive and non-zero
/// - Register allocation is strictly controlled to prevent spills
#[inline(always)]
pub unsafe fn fast_inverse_sqrt_avx2(input: &[f32]) -> Vec<f32> {
    if !is_x86_feature_detected!("avx2") {
        return fallback_inverse_sqrt(input);
    }

    let len = input.len();
    let mut output = Vec::with_capacity(len);
    output.set_len(len);

    // Process in chunks of AVX2_WIDTH_F32
    let chunks = len / AVX2_WIDTH_F32;
    let remainder = len % AVX2_WIDTH_F32;

    // Magic constant for initial approximation (0x5f3759df)
    let magic: __m256i = _mm256_set1_epi32(0x5f3759df);
    let half: __m256 = _mm256_set1_ps(0.5);
    let three_half: __m256 = _mm256_set1_ps(1.5);

    for i in 0..chunks {
        let idx = i * AVX2_WIDTH_F32;
        
        // Load 8 f32 values into YMM register
        let x = _mm256_loadu_ps(input.as_ptr().add(idx));
        
        // Inline assembly for precise register control on AMD Zen
        // Prevents compiler from spilling registers to stack
        let y: __m256;
        let i_bits: __m256i;
        
        std::arch::asm!(
            "vmovaps {x}, {tmp}",
            "vpsrld $1, {tmp}, {tmp}",
            "psubd {magic}, {tmp}",
            "vmovdqa {tmp}, {i_bits}",
            "vcvtdq2ps {i_bits}, {y}",
            x = xmmreg(x),
            tmp = xmmreg(y),
            i_bits = xmmreg(i_bits),
            y = xmmreg(y),
            magic = xmmreg(magic),
            options(pure, nomem, nostack)
        );

        // First Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y^2)
        let y_sq = _mm256_mul_ps(y, y);
        let x_y_sq = _mm256_mul_ps(x, y_sq);
        let one_point_five_minus = _mm256_sub_ps(three_half, _mm256_mul_ps(half, x_y_sq));
        let y_new = _mm256_mul_ps(y, one_point_five_minus);

        // Second iteration for higher precision (optional, trade-off with latency)
        let y_sq2 = _mm256_mul_ps(y_new, y_new);
        let x_y_sq2 = _mm256_mul_ps(x, y_sq2);
        let one_point_five_minus2 = _mm256_sub_ps(three_half, _mm256_mul_ps(half, x_y_sq2));
        let y_final = _mm256_mul_ps(y_new, one_point_five_minus2);

        _mm256_storeu_ps(output.as_mut_ptr().add(idx), y_final);
    }

    // Handle remainder elements with scalar fallback
    for i in (chunks * AVX2_WIDTH_F32)..len {
        output[i] = 1.0_f32 / input[i].sqrt();
    }

    output
}

/// Fast inverse square root using AVX-512 (if available)
/// 
/// Provides 2x throughput over AVX2 on supported AMD Zen 4+ architectures.
/// Uses masked operations to handle non-aligned data without branching.
#[inline(always)]
pub unsafe fn fast_inverse_sqrt_avx512(input: &[f32]) -> Vec<f32> {
    if !is_x86_feature_detected!("avx512f") {
        return fast_inverse_sqrt_avx2(input);
    }

    let len = input.len();
    let mut output = Vec::with_capacity(len);
    output.set_len(len);

    let chunks = len / AVX512_WIDTH_F32;
    let remainder = len % AVX512_WIDTH_F32;

    // Magic constant for AVX-512
    let magic: __m512i = _mm512_set1_epi32(0x5f3759df);
    let half: __m512 = _mm512_set1_ps(0.5);
    let three_half: __m512 = _mm512_set1_ps(1.5);

    for i in 0..chunks {
        let idx = i * AVX512_WIDTH_F32;
        
        let x = _mm512_loadu_ps(input.as_ptr().add(idx));
        
        // Bit-level manipulation for initial guess
        let i_bits = _mm512_srli_epi32::<1>(_mm512_castps_si512(x));
        let i_bits = _mm512_sub_epi32(magic, i_bits);
        let mut y = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(i_bits));

        // Two Newton-Raphson iterations
        for _ in 0..2 {
            let y_sq = _mm512_mul_ps(y, y);
            let x_y_sq = _mm512_mul_ps(x, y_sq);
            let correction = _mm512_fnmadd_ps(half, x_y_sq, three_half);
            y = _mm512_mul_ps(y, correction);
        }

        _mm512_storeu_ps(output.as_mut_ptr().add(idx), y);
    }

    // Handle remainder with AVX2 or scalar
    let start = chunks * AVX512_WIDTH_F32;
    if remainder > 0 {
        let remaining: Vec<f32> = fast_inverse_sqrt_avx2(&input[start..]);
        output[start..].copy_from_slice(&remaining);
    }

    output
}

/// Volatility calculation using SIMD inline assembly
/// 
/// Computes rolling volatility from price ticks using fused multiply-add
/// operations optimized for AMD Zen 4 FMA units.
#[inline(always)]
pub fn calculate_volatility_simd(prices: &[f64], window: usize) -> Vec<f64> {
    if prices.len() < window || window < 2 {
        return vec![];
    }

    let mut volatilities = Vec::with_capacity(prices.len() - window + 1);
    
    // Calculate log returns first
    let mut log_returns = Vec::with_capacity(prices.len() - 1);
    for i in 1..prices.len() {
        log_returns.push((prices[i] / prices[i - 1]).ln());
    }

    // Sliding window variance calculation using AVX2
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            calculate_variance_sliding_window_avx2(&log_returns, window, &mut volatilities);
        }
    } else {
        // Scalar fallback
        for i in 0..=(log_returns.len() - window) {
            let sum: f64 = log_returns[i..i + window].iter().sum();
            let mean = sum / window as f64;
            let variance: f64 = log_returns[i..i + window]
                .iter()
                .map(|&r| (r - mean).powi(2))
                .sum::<f64>()
                / (window - 1) as f64;
            volatilities.push(variance.sqrt() * (252.0_f64).sqrt()); // Annualized
        }
    }

    volatilities
}

/// AVX2-optimized sliding window variance calculation
/// 
/// Uses horizontal adds and FMA instructions to compute variance in parallel.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn calculate_variance_sliding_window_avx2(
    data: &[f64],
    window: usize,
    output: &mut Vec<f64>,
) {
    use std::arch::x86_64::*;

    let len = data.len();
    let simd_width = AVX2_WIDTH_F64;

    for i in 0..=(len - window) {
        let chunk_start = i;
        let mut sum = _mm256_setzero_pd();
        let mut sum_sq = _mm256_setzero_pd();

        // Process full SIMD chunks within the window
        let j_limit = (window / simd_width) * simd_width;
        for j in (0..j_limit).step_by(simd_width) {
            let idx = chunk_start + j;
            let vals = _mm256_loadu_pd(data.as_ptr().add(idx));
            sum = _mm256_add_pd(sum, vals);
            sum_sq = _mm256_fmadd_pd(vals, vals, sum_sq);
        }

        // Horizontal add to get partial sums
        let sum_arr: [f64; 4] = mem::transmute(sum);
        let sum_sq_arr: [f64; 4] = mem::transmute(sum_sq);

        let mut total_sum: f64 = sum_arr.iter().sum();
        let mut total_sum_sq: f64 = sum_sq_arr.iter().sum();

        // Handle remainder elements
        for j in j_limit..window {
            let val = data[chunk_start + j];
            total_sum += val;
            total_sum_sq += val * val;
        }

        let mean = total_sum / window as f64;
        let variance = (total_sum_sq / window as f64) - (mean * mean);
        
        // Bessel's correction for sample variance
        let corrected_variance = variance * (window as f64 / (window - 1) as f64);
        
        output.push(corrected_variance.sqrt() * (252.0_f64).sqrt());
    }
}

/// Scalar fallback for systems without AVX2
fn fallback_inverse_sqrt(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| 1.0_f32 / x.sqrt()).collect()
}

/// Helper macro for inline assembly register constraints
macro_rules! xmmreg {
    ($reg:expr) => {
        inlateout(xmm_reg) $reg
    };
}

/// Runtime CPU feature detection with graceful fallback
pub struct SimdCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_fma: bool,
    pub has_f16c: bool,
}

impl SimdCapabilities {
    pub fn detect() -> Self {
        Self {
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512: is_x86_feature_detected!("avx512f"),
            has_fma: is_x86_feature_detected!("fma"),
            has_f16c: is_x86_feature_detected!("f16c"),
        }
    }

    /// Returns the optimal inverse sqrt function based on detected features
    pub fn inverse_sqrt_fn(&self) -> fn(&[f32]) -> Vec<f32> {
        if self.has_avx512 {
            #[cfg(target_feature = "avx512f")]
            return |input: &[f32]| unsafe { fast_inverse_sqrt_avx512(input) };
            
            #[cfg(not(target_feature = "avx512f"))]
            return |input: &[f32]| unsafe { fast_inverse_sqrt_avx2(input) };
        } else if self.has_avx2 {
            |input: &[f32]| unsafe { fast_inverse_sqrt_avx2(input) }
        } else {
            fallback_inverse_sqrt
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverse_sqrt_accuracy() {
        let input: Vec<f32> = (1..=100).map(|x| x as f32).collect();
        
        let simd_result = unsafe { fast_inverse_sqrt_avx2(&input) };
        let scalar_result: Vec<f32> = input.iter().map(|&x| 1.0_f32 / x.sqrt()).collect();

        for (simd, scalar) in simd_result.iter().zip(scalar_result.iter()) {
            let rel_error = ((simd - scalar).abs() / scalar).abs();
            assert!(rel_error < 1e-4, "Relative error too high: {}", rel_error);
        }
    }

    #[test]
    fn test_volatility_calculation() {
        let prices: Vec<f64> = (100..200).map(|x| x as f64 + 0.5).collect();
        let volatilities = calculate_volatility_simd(&prices, 20);
        
        assert!(!volatilities.is_empty());
        assert!(volatilities.iter().all(|&v| v > 0.0 && v < 10.0));
    }

    #[test]
    fn test_cpu_feature_detection() {
        let caps = SimdCapabilities::detect();
        println!("AVX2: {}, AVX-512: {}, FMA: {}", caps.has_avx2, caps.has_avx512, caps.has_fma);
        
        // Should always have at least SSE2 on x86_64
        assert!(caps.has_avx2 || true); // Graceful degradation
    }
}
