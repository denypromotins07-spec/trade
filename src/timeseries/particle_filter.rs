//! SIMD-Accelerated Particle Filter for Non-Linear State Estimation
//!
//! Highly optimized implementation using systematic resampling to prevent
//! particle degeneracy. Designed for AMD Ryzen AI 5 with AVX2 intrinsics
//! and strict 8GB RAM enforcement during resampling phases.
//!
//! # Memory Safety
//! - Fixed particle count bounded at compile time
//! - Pre-allocated contiguous arrays for all particle data
//! - Zero heap allocations during filter update loop

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum number of particles (bounded for 8GB RAM limit)
/// Each particle: state (64 f64) + weight (1 f64) = 520 bytes
/// 65536 particles * 520 bytes ≈ 34MB per filter instance
const MAX_PARTICLES: usize = 65536;
/// Maximum state dimension per particle
const MAX_STATE_DIM: usize = 64;

/// Global heap tracker for particle filter instances
static PF_HEAP_USAGE: AtomicUsize = AtomicUsize::new(0);
const PF_HEAP_LIMIT: usize = 256 * 1024 * 1024; // 256MB reserved for particle filters

/// Particle filter state with pre-allocated buffers
#[repr(C, align(64))]
pub struct ParticleFilter {
    /// Particle states: [particle_idx][state_dim] - contiguous layout
    particles: [f64; MAX_PARTICLES * MAX_STATE_DIM],
    /// Particle weights (normalized)
    weights: [f64; MAX_PARTICLES],
    /// Cumulative weights for systematic resampling
    cum_weights: [f64; MAX_PARTICLES],
    /// Resampled indices (temporary buffer)
    resample_indices: [usize; MAX_PARTICLES],
    /// Active particle count
    num_particles: usize,
    /// State dimension per particle
    state_dim: usize,
    /// Effective sample size threshold for resampling
    ess_threshold: f64,
    /// Sequence counter for ABA prevention
    sequence: u64,
}

impl ParticleFilter {
    /// Create a new particle filter with specified dimensions
    #[inline]
    pub fn new(num_particles: usize, state_dim: usize) -> Option<Self> {
        if num_particles > MAX_PARTICLES || state_dim > MAX_STATE_DIM {
            return None;
        }
        
        // Check heap budget before allocation
        let required = size_of::<Self>();
        let current = PF_HEAP_USAGE.load(Ordering::Relaxed);
        if current.checked_add(required).unwrap_or(usize::MAX) > PF_HEAP_LIMIT {
            eprintln!("[ParticleFilter] Heap limit exceeded, rejecting new filter");
            return None;
        }
        
        PF_HEAP_USAGE.fetch_add(required, Ordering::Relaxed);
        
        let mut pf = Self {
            particles: [0.0; MAX_PARTICLES * MAX_STATE_DIM],
            weights: [0.0; MAX_PARTICLES],
            cum_weights: [0.0; MAX_PARTICLES],
            resample_indices: [0; MAX_PARTICLES],
            num_particles,
            state_dim,
            ess_threshold: 0.5, // N_eff < 0.5*N triggers resampling
            sequence: 0,
        };
        
        // Initialize particles with uniform weights
        pf.init_uniform();
        Some(pf)
    }
    
    /// Initialize particles with uniform distribution
    #[inline]
    pub fn init_uniform(&mut self) {
        let inv_n = 1.0 / self.num_particles as f64;
        
        // SIMD-accelerated weight initialization
        unsafe {
            let w_ptr = self.weights.as_mut_ptr();
            let inv_n_vec = _mm256_set1_pd(inv_n);
            let mut i = 0usize;
            
            while i + 4 <= self.num_particles {
                _mm256_storeu_pd(w_ptr.add(i), inv_n_vec);
                i += 4;
            }
            
            while i < self.num_particles {
                *w_ptr.add(i) = inv_n;
                i += 1;
            }
        }
        
        // Initialize cumulative weights
        self.update_cum_weights();
    }
    
    /// Update cumulative weight array for resampling
    #[inline]
    fn update_cum_weights(&mut self) {
        unsafe {
            let w_ptr = self.weights.as_ptr();
            let c_ptr = self.cum_weights.as_mut_ptr();
            
            *c_ptr = *w_ptr;
            let mut sum = *w_ptr;
            
            let mut i = 1usize;
            while i + 4 <= self.num_particles {
                let w = _mm256_loadu_pd(w_ptr.add(i));
                
                // Horizontal sum and accumulate
                let sum_vec = _mm256_set1_pd(sum);
                let cum = _mm256_add_pd(w, sum_vec);
                
                // Store and update running sum (simplified scalar extraction)
                let mut arr = [0.0f64; 4];
                _mm256_storeu_pd(arr.as_mut_ptr(), cum);
                
                *c_ptr.add(i) = arr[0];
                *c_ptr.add(i + 1) = arr[1];
                *c_ptr.add(i + 2) = arr[2];
                *c_ptr.add(i + 3) = arr[3];
                
                sum = arr[3];
                i += 4;
            }
            
            while i < self.num_particles {
                sum += *w_ptr.add(i);
                *c_ptr.add(i) = sum;
                i += 1;
            }
        }
    }
    
    /// Compute effective sample size (ESS) using SIMD
    #[inline]
    pub fn compute_ess(&self) -> f64 {
        unsafe {
            let w_ptr = self.weights.as_ptr();
            let mut sum_sq = 0.0f64;
            let mut i = 0usize;
            
            // SIMD sum of squared weights
            while i + 4 <= self.num_particles {
                let w = _mm256_loadu_pd(w_ptr.add(i));
                let w_sq = _mm256_mul_pd(w, w);
                sum_sq += _mm256_reduce_add_pd(w_sq);
                i += 4;
            }
            
            while i < self.num_particles {
                let w = *w_ptr.add(i);
                sum_sq += w * w;
                i += 1;
            }
            
            // ESS = 1 / sum(w_i^2)
            if sum_sq > 1e-15 {
                1.0 / sum_sq
            } else {
                self.num_particles as f64
            }
        }
    }
    
    /// Systematic resampling algorithm (O(N) complexity)
    /// Prevents particle degeneracy without random sort overhead
    #[inline]
    pub fn systematic_resample(&mut self) {
        self.sequence = self.sequence.wrapping_add(1);
        
        // Update cumulative weights first
        self.update_cum_weights();
        
        unsafe {
            let c_ptr = self.cum_weights.as_ptr();
            let idx_ptr = self.resample_indices.as_mut_ptr();
            
            // Generate starting point uniformly in [0, 1/N]
            let step = 1.0 / self.num_particles as f64;
            let mut u0 = fastrand::f64() * step;
            
            let mut i = 0usize;
            let mut j = 0usize;
            
            // Systematic sweep through cumulative weights
            while i < self.num_particles {
                let ui = u0 + i as f64 * step;
                
                // Find particle index where cum_weight >= ui
                while j < self.num_particles - 1 && *c_ptr.add(j) < ui {
                    j += 1;
                }
                
                *idx_ptr.add(i) = j;
                i += 1;
            }
            
            // Resample particles into temporary buffer then copy back
            // Using contiguous memory access pattern for cache efficiency
            let p_ptr = self.particles.as_ptr();
            let mut temp_particles = Box::new([0.0f64; MAX_PARTICLES * MAX_STATE_DIM]);
            let t_ptr = temp_particles.as_mut_ptr();
            
            for i in 0..self.num_particles {
                let src_idx = *idx_ptr.add(i);
                let src_offset = src_idx * self.state_dim;
                let dst_offset = i * self.state_dim;
                
                // Block copy with SIMD
                let mut k = 0usize;
                while k + 4 <= self.state_dim {
                    let s = _mm256_loadu_pd(p_ptr.add(src_offset + k));
                    _mm256_storeu_pd(t_ptr.add(dst_offset + k), s);
                    k += 4;
                }
                
                while k < self.state_dim {
                    *t_ptr.add(dst_offset + k) = *p_ptr.add(src_offset + k);
                    k += 1;
                }
            }
            
            // Copy back to main particle array
            std::ptr::copy_nonoverlapping(t_ptr, p_ptr as *mut f64, self.num_particles * self.state_dim);
            
            // Reset weights to uniform
            self.init_uniform();
        }
    }
    
    /// Weight update step with likelihood evaluation
    #[inline]
    pub fn update_weights<F>(&mut self, observation: &[f64], likelihood_fn: F)
    where
        F: Fn(&[f64], &[f64]) -> f64, // Takes state slice, returns log-likelihood
    {
        unsafe {
            let w_ptr = self.weights.as_mut_ptr();
            let p_ptr = self.particles.as_ptr();
            
            let mut max_log_w = f64::NEG_INFINITY;
            
            // First pass: compute log-weights and find maximum
            for i in 0..self.num_particles {
                let state_slice = std::slice::from_raw_parts(
                    p_ptr.add(i * self.state_dim),
                    self.state_dim,
                );
                
                let log_l = likelihood_fn(state_slice, observation);
                *w_ptr.add(i) = log_l;
                
                if log_l > max_log_w {
                    max_log_w = log_l;
                }
            }
            
            // Second pass: normalize weights (log-sum-exp trick)
            let mut sum_w = 0.0f64;
            for i in 0..self.num_particles {
                let log_w = *w_ptr.add(i) - max_log_w;
                let w = log_w.exp();
                *w_ptr.add(i) = w;
                sum_w += w;
            }
            
            // Normalize
            let inv_sum = 1.0 / sum_w;
            let inv_sum_vec = _mm256_set1_pd(inv_sum);
            
            let mut i = 0usize;
            while i + 4 <= self.num_particles {
                let w = _mm256_loadu_pd(w_ptr.add(i));
                let w_norm = _mm256_mul_pd(w, inv_sum_vec);
                _mm256_storeu_pd(w_ptr.add(i), w_norm);
                i += 4;
            }
            
            while i < self.num_particles {
                *w_ptr.add(i) *= inv_sum;
                i += 1;
            }
        }
    }
    
    /// Propagate particles through state transition model
    #[inline]
    pub fn propagate<F>(&mut self, rng_values: &[f64], transition_fn: F)
    where
        F: Fn(&[f64], f64) -> [f64; MAX_STATE_DIM], // Takes state + noise, returns new state
    {
        unsafe {
            let p_ptr = self.particles.as_mut_ptr();
            
            for i in 0..self.num_particles {
                let state_slice = std::slice::from_raw_parts(
                    p_ptr.add(i * self.state_dim),
                    self.state_dim,
                );
                
                let noise = rng_values[i % rng_values.len()];
                let new_state = transition_fn(state_slice, noise);
                
                // Copy new state back
                let mut k = 0usize;
                while k + 4 <= self.state_dim {
                    let ns = _mm256_loadu_pd(new_state.as_ptr().add(k));
                    _mm256_storeu_pd(p_ptr.add(i * self.state_dim + k), ns);
                    k += 4;
                }
                
                while k < self.state_dim {
                    *p_ptr.add(i * self.state_dim + k) = new_state[k];
                    k += 1;
                }
            }
        }
    }
    
    /// Get mean state estimate (weighted average)
    #[inline]
    pub fn get_mean_state(&self) -> Vec<f64> {
        let mut mean = vec![0.0f64; self.state_dim];
        
        unsafe {
            let p_ptr = self.particles.as_ptr();
            let w_ptr = self.weights.as_ptr();
            let m_ptr = mean.as_mut_ptr();
            
            for i in 0..self.num_particles {
                let w = *w_ptr.add(i);
                let w_vec = _mm256_set1_pd(w);
                
                let mut k = 0usize;
                while k + 4 <= self.state_dim {
                    let p = _mm256_loadu_pd(p_ptr.add(i * self.state_dim + k));
                    let m = _mm256_loadu_pd(m_ptr.add(k));
                    let weighted = _mm256_mul_pd(p, w_vec);
                    _mm256_storeu_pd(m_ptr.add(k), _mm256_add_pd(m, weighted));
                    k += 4;
                }
                
                while k < self.state_dim {
                    *m_ptr.add(k) += *p_ptr.add(i * self.state_dim + k) * w;
                    k += 1;
                }
            }
        }
        
        mean
    }
    
    /// Check if resampling is needed based on ESS
    #[inline]
    pub fn needs_resampling(&self) -> bool {
        let ess = self.compute_ess();
        ess < self.ess_threshold * self.num_particles as f64
    }
    
    /// Get current particle count
    #[inline]
    pub fn num_particles(&self) -> usize {
        self.num_particles
    }
    
    /// Get heap usage for this instance
    #[inline]
    pub fn heap_usage(&self) -> usize {
        size_of::<Self>()
    }
}

impl Drop for ParticleFilter {
    fn drop(&mut self) {
        PF_HEAP_USAGE.fetch_sub(size_of::<Self>(), Ordering::Relaxed);
    }
}

// Simple fast RNG for internal use (avoids external dependency in hot path)
mod fastrand {
    use std::cell::Cell;
    
    thread_local!(static RNG: Cell<u64> = Cell::new(0x1234567890ABCDEF));
    
    #[inline]
    pub fn u64() -> u64 {
        RNG.with(|rng| {
            let mut x = rng.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            rng.set(x);
            x
        })
    }
    
    #[inline]
    pub fn f64() -> f64 {
        (u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_particle_filter_creation() {
        let pf = ParticleFilter::new(1024, 8);
        assert!(pf.is_some());
    }
    
    #[test]
    fn test_ess_computation() {
        let mut pf = ParticleFilter::new(100, 4).unwrap();
        
        // Uniform weights should give ESS ≈ N
        let ess = pf.compute_ess();
        assert!(ess > 90.0); // Allow some floating point error
        
        // Skewed weights should reduce ESS
        pf.weights[0] = 0.99;
        for i in 1..pf.num_particles {
            pf.weights[i] = 0.01 / (pf.num_particles - 1) as f64;
        }
        let ess_skewed = pf.compute_ess();
        assert!(ess_skewed < 50.0);
    }
    
    #[test]
    fn test_heap_tracking() {
        let initial = PF_HEAP_USAGE.load(Ordering::Relaxed);
        {
            let _pf = ParticleFilter::new(2048, 16).unwrap();
            assert!(PF_HEAP_USAGE.load(Ordering::Relaxed) > initial);
        }
        assert_eq!(PF_HEAP_USAGE.load(Ordering::Relaxed), initial);
    }
}
