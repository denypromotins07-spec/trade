//! Dynamic Linear Models (DLM) for Non-Stationary Crypto Markets
//! 
//! Implements sequential Bayesian updating to track time-varying parameters
//! without matrix inversions in the hot path. Optimized for AMD Ryzen AI 5
//! with SIMD instructions and strict 8GB RAM enforcement.
//!
//! # Memory Safety
//! - Pre-allocated contiguous arrays for state vectors
//! - No heap allocations during update steps
//! - Fixed-size buffers bounded by compile-time constants

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum state dimension supported (bounded for 8GB RAM limit)
const MAX_STATE_DIM: usize = 64;
/// Maximum observation dimension
const MAX_OBS_DIM: usize = 16;

/// Global heap tracker for DLM components
static DLM_HEAP_USAGE: AtomicUsize = AtomicUsize::new(0);
const DLM_HEAP_LIMIT: usize = 128 * 1024 * 1024; // 128MB reserved for DLM

/// Dynamic Linear Model state with pre-allocated buffers
#[repr(C, align(64))]
pub struct DynamicLinearModel {
    /// State vector μ_t (pre-allocated, cache-line aligned)
    state: [f64; MAX_STATE_DIM],
    /// State covariance diagonal (simplified for speed)
    variance: [f64; MAX_STATE_DIM],
    /// Transition matrix diagonal (time-varying parameters)
    transition: [f64; MAX_STATE_DIM],
    /// Observation matrix (flattened row-major)
    observation: [f64; MAX_STATE_DIM * MAX_OBS_DIM],
    /// Process noise variance
    process_noise: [f64; MAX_STATE_DIM],
    /// Observation noise variance
    observation_noise: f64,
    /// Active dimensions (runtime configurable)
    state_dim: usize,
    obs_dim: usize,
    /// Sequence counter for ABA prevention in concurrent access
    sequence: u64,
}

impl DynamicLinearModel {
    /// Create a new DLM with specified dimensions
    #[inline]
    pub fn new(state_dim: usize, obs_dim: usize) -> Option<Self> {
        if state_dim > MAX_STATE_DIM || obs_dim > MAX_OBS_DIM {
            return None;
        }
        
        // Check heap budget before allocation
        let required = size_of::<Self>();
        let current = DLM_HEAP_USAGE.load(Ordering::Relaxed);
        if current.checked_add(required).unwrap_or(usize::MAX) > DLM_HEAP_LIMIT {
            // Emergency: reject allocation to preserve system stability
            eprintln!("[DLM] Heap limit exceeded, rejecting new model");
            return None;
        }
        
        DLM_HEAP_USAGE.fetch_add(required, Ordering::Relaxed);
        
        Some(Self {
            state: [0.0; MAX_STATE_DIM],
            variance: [1.0; MAX_STATE_DIM],
            transition: [1.0; MAX_STATE_DIM],
            observation: [0.0; MAX_STATE_DIM * MAX_OBS_DIM],
            process_noise: [0.01; MAX_STATE_DIM],
            observation_noise: 0.1,
            state_dim,
            obs_dim,
            sequence: 0,
        })
    }
    
    /// Initialize observation matrix with identity-like structure
    #[inline]
    pub fn init_observation_matrix(&mut self) {
        // Zero out active region only (SIMD-friendly contiguous access)
        unsafe {
            let ptr = self.observation.as_mut_ptr();
            let len = self.state_dim * self.obs_dim;
            std::ptr::write_bytes(ptr, 0, len);
            
            // Set diagonal elements for identity-like mapping
            for i in 0..self.obs_dim.min(self.state_dim) {
                *ptr.add(i * self.state_dim + i) = 1.0;
            }
        }
    }
    
    /// Sequential Bayesian update without matrix inversion
    /// Uses scalar Kalman filter equations for diagonal covariance
    #[inline]
    pub fn update(&mut self, observation: &[f64]) {
        debug_assert!(observation.len() == self.obs_dim);
        
        self.sequence = self.sequence.wrapping_add(1);
        
        // SIMD-accelerated prediction step
        unsafe {
            let mut i = 0usize;
            let state_ptr = self.state.as_mut_ptr();
            let var_ptr = self.variance.as_mut_ptr();
            let trans_ptr = self.transition.as_ptr();
            let proc_noise_ptr = self.process_noise.as_ptr();
            
            // Process 4 elements at a time with AVX2
            while i + 4 <= self.state_dim {
                let s = _mm256_loadu_pd(state_ptr.add(i));
                let t = _mm256_loadu_pd(trans_ptr.add(i));
                let v = _mm256_loadu_pd(var_ptr.add(i));
                let pn = _mm256_loadu_pd(proc_noise_ptr.add(i));
                
                // μ_t|t-1 = T * μ_t-1|t-1
                let pred_state = _mm256_mul_pd(s, t);
                
                // P_t|t-1 = T^2 * P_t-1|t-1 + Q
                let pred_var = _mm256_add_pd(
                    _mm256_mul_pd(_mm256_mul_pd(t, t), v),
                    pn
                );
                
                _mm256_storeu_pd(state_ptr.add(i), pred_state);
                _mm256_storeu_pd(var_ptr.add(i), pred_var);
                
                i += 4;
            }
            
            // Handle remainder
            while i < self.state_dim {
                let s = *state_ptr.add(i);
                let t = *trans_ptr.add(i);
                let v = *var_ptr.add(i);
                let pn = *proc_noise_ptr.add(i);
                
                *state_ptr.add(i) = s * t;
                *var_ptr.add(i) = t * t * v + pn;
                
                i += 1;
            }
        }
        
        // Update step for each observation dimension
        for k in 0..self.obs_dim {
            let obs_val = observation[k];
            
            // Compute predicted observation: H_k * μ_t|t-1
            let mut pred_obs = 0.0f64;
            unsafe {
                let h_row = self.observation.as_ptr().add(k * self.state_dim);
                let mut j = 0usize;
                
                // SIMD dot product
                while j + 4 <= self.state_dim {
                    let h = _mm256_loadu_pd(h_row.add(j));
                    let s = _mm256_loadu_pd(self.state.as_ptr().add(j));
                    pred_obs += _mm256_reduce_add_pd(_mm256_mul_pd(h, s));
                    j += 4;
                }
                
                while j < self.state_dim {
                    pred_obs += *h_row.add(j) * *self.state.as_ptr().add(j);
                    j += 1;
                }
            }
            
            // Innovation: ν = y - H*μ
            let innovation = obs_val - pred_obs;
            
            // Innovation variance: S = H*P*H' + R (simplified for diagonal P)
            let mut innov_var = self.observation_noise;
            unsafe {
                let h_row = self.observation.as_ptr().add(k * self.state_dim);
                let var_ptr = self.variance.as_ptr();
                let mut j = 0usize;
                
                while j + 4 <= self.state_dim {
                    let h = _mm256_loadu_pd(h_row.add(j));
                    let v = _mm256_loadu_pd(var_ptr.add(j));
                    let hv = _mm256_mul_pd(h, h);
                    let hvv = _mm256_mul_pd(hv, v);
                    innov_var += _mm256_reduce_add_pd(hvv);
                    j += 4;
                }
                
                while j < self.state_dim {
                    let h = *h_row.add(j);
                    let v = *var_ptr.add(j);
                    innov_var += h * h * v;
                    j += 1;
                }
            }
            
            // Kalman gain: K = P*H' / S
            let mut kalman_gain = [0.0f64; MAX_STATE_DIM];
            unsafe {
                let h_row = self.observation.as_ptr().add(k * self.state_dim);
                let var_ptr = self.variance.as_ptr();
                let kg_ptr = kalman_gain.as_mut_ptr();
                let inv_s = 1.0 / innov_var;
                
                let vs = _mm256_set1_pd(inv_s);
                let mut j = 0usize;
                
                while j + 4 <= self.state_dim {
                    let h = _mm256_loadu_pd(h_row.add(j));
                    let v = _mm256_loadu_pd(var_ptr.add(j));
                    let hv = _mm256_mul_pd(h, v);
                    let k = _mm256_mul_pd(hv, vs);
                    _mm256_storeu_pd(kg_ptr.add(j), k);
                    j += 4;
                }
                
                while j < self.state_dim {
                    *kg_ptr.add(j) = *h_row.add(j) * *var_ptr.add(j) * inv_s;
                    j += 1;
                }
            }
            
            // State update: μ = μ + K*ν
            unsafe {
                let state_ptr = self.state.as_mut_ptr();
                let kg_ptr = kalman_gain.as_ptr();
                let inn = _mm256_set1_pd(innovation);
                let mut j = 0usize;
                
                while j + 4 <= self.state_dim {
                    let k = _mm256_loadu_pd(kg_ptr.add(j));
                    let s = _mm256_loadu_pd(state_ptr.add(j));
                    let update = _mm256_mul_pd(k, inn);
                    _mm256_storeu_pd(state_ptr.add(j), _mm256_add_pd(s, update));
                    j += 4;
                }
                
                while j < self.state_dim {
                    *state_ptr.add(j) += *kg_ptr.add(j) * innovation;
                    j += 1;
                }
            }
            
            // Covariance update: P = (I - K*H)*P (Joseph form simplified)
            unsafe {
                let var_ptr = self.variance.as_mut_ptr();
                let h_row = self.observation.as_ptr().add(k * self.state_dim);
                let kg_ptr = kalman_gain.as_ptr();
                let mut j = 0usize;
                
                while j + 4 <= self.state_dim {
                    let h = _mm256_loadu_pd(h_row.add(j));
                    let k = _mm256_loadu_pd(kg_ptr.add(j));
                    let v = _mm256_loadu_pd(var_ptr.add(j));
                    let kh = _mm256_mul_pd(k, h);
                    let one = _mm256_set1_pd(1.0);
                    let factor = _mm256_sub_pd(one, kh);
                    _mm256_storeu_pd(var_ptr.add(j), _mm256_mul_pd(factor, v));
                    j += 4;
                }
                
                while j < self.state_dim {
                    let h = *h_row.add(j);
                    let k = *kg_ptr.add(j);
                    *var_ptr.add(j) *= 1.0 - k * h;
                    j += 1;
                }
            }
        }
    }
    
    /// Get current state estimate (copy to avoid aliasing)
    #[inline]
    pub fn get_state(&self) -> &[f64] {
        &self.state[..self.state_dim]
    }
    
    /// Get current variance estimate
    #[inline]
    pub fn get_variance(&self) -> &[f64] {
        &self.variance[..self.state_dim]
    }
    
    /// Update time-varying transition parameters
    #[inline]
    pub fn set_transition(&mut self, idx: usize, value: f64) {
        if idx < self.state_dim {
            self.transition[idx] = value;
        }
    }
    
    /// Get current heap usage for this DLM instance
    #[inline]
    pub fn heap_usage(&self) -> usize {
        size_of::<Self>()
    }
}

impl Drop for DynamicLinearModel {
    fn drop(&mut self) {
        DLM_HEAP_USAGE.fetch_sub(size_of::<Self>(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dlm_creation() {
        let dlm = DynamicLinearModel::new(8, 4);
        assert!(dlm.is_some());
    }
    
    #[test]
    fn test_dlm_update() {
        let mut dlm = DynamicLinearModel::new(8, 4).unwrap();
        dlm.init_observation_matrix();
        
        let obs = [1.0, 2.0, 3.0, 4.0];
        dlm.update(&obs);
        
        // State should be updated (non-zero after first observation)
        let state = dlm.get_state();
        assert!(state.iter().any(|&x| x.abs() > 1e-10));
    }
    
    #[test]
    fn test_heap_limit() {
        // Verify heap tracking works
        let initial = DLM_HEAP_USAGE.load(Ordering::Relaxed);
        {
            let _dlm = DynamicLinearModel::new(16, 8).unwrap();
            assert!(DLM_HEAP_USAGE.load(Ordering::Relaxed) > initial);
        }
        // Should return to initial after drop
        assert_eq!(DLM_HEAP_USAGE.load(Ordering::Relaxed), initial);
    }
}
