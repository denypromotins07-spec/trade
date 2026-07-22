//! Smart Limit Order Placement & Queue Jumping Logic
//! 
//! This module implements microsecond queue-jumping strategies by analyzing L3 order book data.
//! It places limit orders just ahead of large resting walls to improve execution priority.
//! Optimized for AMD Ryzen AI 5 with SIMD instructions to avoid heap allocations.
//!
//! # Safety
//! All operations are lock-free and use pre-allocated buffers to respect the 8GB RAM limit.

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU64, Ordering};
use crate::orderbook::{Level, OrderBookSnapshot};

/// Maximum depth levels to analyze for queue jumping (compile-time constant)
const MAX_DEPTH: usize = 50;

/// SIMD-aligned buffer for order book depths (avoids heap allocation)
#[repr(C, align(64))]
pub struct DepthBuffer {
    pub prices: [u64; MAX_DEPTH],
    pub sizes: [u64; MAX_DEPTH],
    pub valid_len: AtomicU64,
}

impl Default for DepthBuffer {
    fn default() -> Self {
        Self {
            prices: [0; MAX_DEPTH],
            sizes: [0; MAX_DEPTH],
            valid_len: AtomicU64::new(0),
        }
    }
}

/// Queue jumping engine for smart limit order placement
pub struct QueueJumper {
    /// Pre-allocated depth buffer (stack-allocated, no heap)
    depth_buffer: DepthBuffer,
    /// Minimum wall size to consider for queue jumping (in base currency * 1e8)
    min_wall_size: u64,
    /// Offset in ticks to place order ahead of wall
    jump_offset_ticks: u64,
    /// Cache line padding to prevent false sharing
    _padding: [u8; 64 - (std::mem::size_of::<DepthBuffer>() % 64)],
}

impl QueueJumper {
    /// Create a new queue jumper with configured parameters
    #[inline]
    pub const fn new(min_wall_size: u64, jump_offset_ticks: u64) -> Self {
        Self {
            depth_buffer: DepthBuffer {
                prices: [0; MAX_DEPTH],
                sizes: [0; MAX_DEPTH],
                valid_len: AtomicU64::new(0),
            },
            min_wall_size,
            jump_offset_ticks,
            _padding: [0; 64],
        }
    }

    /// Analyze order book and find optimal queue jump position using SIMD
    /// Returns (price, size) tuple for optimal limit order placement
    /// 
    /// # Arguments
    /// * `snapshot` - Current L3 order book snapshot
    /// * `is_bid` - true for bid side, false for ask side
    /// 
    /// # Safety
    /// Uses unsafe SIMD intrinsics but validates all inputs first
    #[inline]
    pub fn find_queue_jump_position(&self, snapshot: &OrderBookSnapshot, is_bid: bool) -> Option<(u64, u64)> {
        let levels = if is_bid { &snapshot.bids } else { &snapshot.asks };
        let len = std::cmp::min(levels.len(), MAX_DEPTH);
        
        if len == 0 {
            return None;
        }

        // Load data into SIMD-aligned buffer (no heap allocation)
        self.depth_buffer.valid_len.store(len as u64, Ordering::Relaxed);
        for i in 0..len {
            self.depth_buffer.prices[i] = levels[i].price;
            self.depth_buffer.sizes[i] = levels[i].size;
        }

        // SIMD-accelerated wall detection
        unsafe {
            self.detect_wall_simd(len, is_bid)
        }
    }

    /// SIMD-accelerated wall detection algorithm
    /// Scans for large resting orders that can be front-run
    #[target_feature(enable = "avx2")]
    unsafe fn detect_wall_simd(&self, len: usize, is_bid: bool) -> Option<(u64, u64)> {
        if len < 4 {
            // Fallback to scalar for small depths
            return self.detect_wall_scalar(len, is_bid);
        }

        let valid_len = self.depth_buffer.valid_len.load(Ordering::Relaxed) as usize;
        let mut best_price = 0u64;
        let mut best_size = 0u64;
        let mut found_wall = false;

        // Process 4 levels at a time using AVX2
        let chunks = valid_len / 4;
        for chunk_idx in 0..chunks {
            let base_idx = chunk_idx * 4;
            
            // Load 4 sizes into SIMD register
            let sizes_ptr = self.depth_buffer.sizes[base_idx..].as_ptr() as *const __m256i;
            let sizes = _mm256_load_si256(sizes_ptr);
            
            // Load min_wall_size replicated across SIMD lanes
            let wall_ptr = std::array::from_fn(|_| self.min_wall_size).as_ptr() as *const __m256i;
            let walls = _mm256_load_si256(wall_ptr);
            
            // Compare sizes >= min_wall_size
            let cmp_result = _mm256_cmpgt_epi64(sizes, walls);
            
            // Extract mask to check if any wall detected
            let mask = _mm256_movemask_epi8(cmp_result);
            
            if mask != 0 {
                // Wall detected, find first one
                for i in 0..4 {
                    let idx = base_idx + i;
                    if idx < valid_len && self.depth_buffer.sizes[idx] >= self.min_wall_size {
                        let wall_price = self.depth_buffer.prices[idx];
                        best_price = if is_bid {
                            wall_price.saturating_add(self.jump_offset_ticks)
                        } else {
                            wall_price.saturating_sub(self.jump_offset_ticks)
                        };
                        best_size = self.depth_buffer.sizes[idx] / 4; // Take 25% of wall
                        found_wall = true;
                        break;
                    }
                }
                if found_wall {
                    break;
                }
            }
        }

        if found_wall {
            Some((best_price, best_size))
        } else {
            // Check remaining elements
            self.detect_wall_scalar(valid_len, is_bid)
        }
    }

    /// Scalar fallback for small depths or non-AVX2 systems
    #[inline]
    fn detect_wall_scalar(&self, len: usize, is_bid: bool) -> Option<(u64, u64)> {
        for i in 0..len {
            if self.depth_buffer.sizes[i] >= self.min_wall_size {
                let wall_price = self.depth_buffer.prices[i];
                let jump_price = if is_bid {
                    wall_price.saturating_add(self.jump_offset_ticks)
                } else {
                    wall_price.saturating_sub(self.jump_offset_ticks)
                };
                let jump_size = self.depth_buffer.sizes[i] / 4;
                return Some((jump_price, jump_size));
            }
        }
        None
    }

    /// Calculate optimal order size based on wall depth and risk parameters
    #[inline]
    pub fn calculate_position_size(&self, wall_size: u64, volatility: u64) -> u64 {
        // Reduce size proportionally to volatility
        let vol_factor = 10000u64.saturating_sub(volatility);
        let base_size = wall_size / 4;
        base_size.checked_mul(vol_factor).unwrap_or(0) / 10000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_jumper_creation() {
        let jumper = QueueJumper::new(1000000, 5);
        assert_eq!(jumper.min_wall_size, 1000000);
        assert_eq!(jumper.jump_offset_ticks, 5);
    }

    #[test]
    fn test_position_size_calculation() {
        let jumper = QueueJumper::new(1000000, 5);
        let size = jumper.calculate_position_size(4000000, 2000);
        assert!(size > 0);
        assert!(size <= 1000000);
    }
}
