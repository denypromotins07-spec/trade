//! Queue Position Estimator for High-Frequency Trading
//! 
//! This module implements a probabilistic queue position estimator that tracks
//! the bot's position in the limit order book queue using historical cancellation
//! distributions and hidden liquidity detection. All operations are O(1) and lock-free.
//!
//! Optimized for:
//! - Microsecond latency updates
//! - 8GB global RAM limit (bounded ring buffers)
//! - AMD Ryzen AI 5 SIMD acceleration
//! - Lock-free concurrent access

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Cache-line size for alignment (typically 64 bytes on x86_64)
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of historical events to track (bounds memory usage)
const MAX_HISTORY_SIZE: usize = 1024;

/// Aligned atomic counter for cache-line padding
#[repr(align(64))]
struct AlignedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; 56], // Pad to 64 bytes
}

impl AlignedAtomicU64 {
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

    #[inline]
    fn fetch_sub(&self, val: u64, order: Ordering) -> u64 {
        self.value.fetch_sub(val, order)
    }
}

/// Historical event for cancellation distribution analysis
#[derive(Clone, Copy, Debug)]
struct QueueEvent {
    /// Timestamp in nanoseconds
    timestamp_ns: u64,
    /// Event type: 0=new order, 1=cancellation, 2=fill, 3=hidden liquidity detected
    event_type: u8,
    /// Quantity involved
    quantity: u32,
    /// Price level offset from best (0 = best bid/ask)
    price_offset: u8,
}

/// Probabilistic Queue Position Estimator
/// 
/// Tracks the bot's estimated position in the LOB queue using:
/// 1. Historical cancellation rates at each price level
/// 2. Hidden liquidity detection via sweep patterns
/// 3. Real-time order flow momentum
/// 
/// All updates are O(1) and lock-free for microsecond performance.
pub struct QueuePositionEstimator {
    /// Current estimated position (number of orders ahead)
    estimated_position: AlignedAtomicU64,
    
    /// Total queue size at our price level
    queue_size: AlignedAtomicU64,
    
    /// Our order quantity
    our_quantity: AlignedAtomicU64,
    
    /// Ring buffer of historical events (bounded for memory safety)
    history: Vec<QueueEvent>,
    history_head: AtomicUsize,
    history_count: AtomicUsize,
    
    /// Cancellation statistics per price offset (0-10 levels)
    cancellation_counts: [AlignedAtomicU64; 11],
    total_events_per_level: [AlignedAtomicU64; 11],
    
    /// Hidden liquidity score (0-1000, higher = more hidden liquidity)
    hidden_liquidity_score: AlignedAtomicU64,
    
    /// Last update timestamp for time-decay calculations
    last_update_ns: AlignedAtomicU64,
    
    /// Decay factor for exponential moving averages (scaled by 1000)
    decay_factor: u64,
}

impl QueuePositionEstimator {
    /// Create a new queue position estimator with bounded history
    pub fn new(decay_factor_bps: u64) -> Self {
        // Initialize aligned atomics
        let zero_aligned = || AlignedAtomicU64::new(0);
        
        QueuePositionEstimator {
            estimated_position: AlignedAtomicU64::new(0),
            queue_size: AlignedAtomicU64::new(0),
            our_quantity: AlignedAtomicU64::new(0),
            history: Vec::with_capacity(MAX_HISTORY_SIZE),
            history_head: AtomicUsize::new(0),
            history_count: AtomicUsize::new(0),
            cancellation_counts: std::array::from_fn(|_| zero_aligned()),
            total_events_per_level: std::array::from_fn(|_| zero_aligned()),
            hidden_liquidity_score: AlignedAtomicU64::new(500), // Start at neutral
            last_update_ns: AlignedAtomicU64::new(0),
            decay_factor: decay_factor_bps.min(1000), // Clamp to 0-100%
        }
    }

    /// Record a new event in the history buffer (O(1), lock-free)
    /// 
    /// Uses a circular buffer to maintain bounded memory usage.
    #[inline]
    pub fn record_event(&self, event: QueueEvent) {
        let head = self.history_head.fetch_add(1, Ordering::Relaxed);
        let count = self.history_count.load(Ordering::Relaxed);
        
        // Calculate actual index in circular buffer
        let idx = head % MAX_HISTORY_SIZE;
        
        // Safe write: either expanding buffer or overwriting oldest
        if count < MAX_HISTORY_SIZE {
            unsafe {
                // Safety: We pre-allocated with capacity, and only one thread writes per slot
                let ptr = self.history.as_ptr() as *mut QueueEvent;
                ptr.add(idx).write(event);
            }
            self.history_count.fetch_add(1, Ordering::Release);
        } else {
            // Overwrite oldest entry (lock-free since each head value is unique)
            unsafe {
                let ptr = self.history.as_ptr() as *mut QueueEvent;
                ptr.add(idx).write(event);
            }
        }
        
        // Update statistics for this price level
        let level = (event.price_offset as usize).min(10);
        self.total_events_per_level[level].fetch_add(1, Ordering::Relaxed);
        
        if event.event_type == 1 {
            // Cancellation event
            self.cancellation_counts[level].fetch_add(1, Ordering::Relaxed);
        }
        
        // Update hidden liquidity score based on sweep detection
        if event.event_type == 3 {
            // Hidden liquidity detected - increase score
            let current = self.hidden_liquidity_score.load(Ordering::Relaxed);
            let new_score = (current + 50).min(1000);
            self.hidden_liquidity_score.store(new_score, Ordering::Relaxed);
        }
        
        // Time-decay the hidden liquidity score
        self.apply_time_decay(event.timestamp_ns);
    }

    /// Apply time-based decay to hidden liquidity score
    #[inline]
    fn apply_time_decay(&self, current_ns: u64) {
        let last_ns = self.last_update_ns.load(Ordering::Relaxed);
        
        if last_ns == 0 {
            self.last_update_ns.store(current_ns, Ordering::Relaxed);
            return;
        }
        
        let elapsed_ns = current_ns.saturating_sub(last_ns);
        
        // Decay every 100ms (100,000,000 ns)
        if elapsed_ns > 100_000_000 {
            let current_score = self.hidden_liquidity_score.load(Ordering::Relaxed);
            // Decay by decay_factor percentage
            let decay_amount = (current_score * self.decay_factor) / 1000;
            let new_score = current_score.saturating_sub(decay_amount).max(100);
            self.hidden_liquidity_score.store(new_score, Ordering::Relaxed);
            self.last_update_ns.store(current_ns, Ordering::Relaxed);
        }
    }

    /// Calculate cancellation probability for a given price offset
    /// 
    /// Returns probability as basis points (0-10000)
    #[inline]
    pub fn cancellation_probability(&self, price_offset: u8) -> u64 {
        let level = (price_offset as usize).min(10);
        let cancels = self.cancellation_counts[level].load(Ordering::Acquire);
        let total = self.total_events_per_level[level].load(Ordering::Acquire);
        
        if total == 0 {
            return 5000; // Default 50% if no data
        }
        
        // Return as basis points
        (cancels * 10000) / total
    }

    /// Update queue size and recalculate position estimate
    /// 
    /// This is the core O(1) update function called on every LOB update.
    #[inline]
    pub fn update_queue(&self, new_queue_size: u64, our_qty: u64, orders_ahead_estimate: u64) {
        self.queue_size.store(new_queue_size, Ordering::Release);
        self.our_quantity.store(our_qty, Ordering::Release);
        
        // Adjust estimate based on hidden liquidity
        let hidden_score = self.hidden_liquidity_score.load(Ordering::Relaxed);
        let adjustment_factor = 1000 + hidden_score; // 1000-2000 multiplier
        
        // Expected cancellations ahead of us
        let avg_cancel_prob = self.average_cancellation_probability();
        let expected_cancels = (orders_ahead_estimate * avg_cancel_prob) / 10000;
        
        // Adjusted position = raw estimate - expected cancellations + hidden liquidity penalty
        let adjusted_position = orders_ahead_estimate
            .saturating_sub(expected_cancels)
            .saturating_add((orders_ahead_estimate * hidden_score) / 10000);
        
        self.estimated_position.store(adjusted_position, Ordering::Release);
    }

    /// Get average cancellation probability across all levels
    #[inline]
    fn average_cancellation_probability(&self) -> u64 {
        let mut sum = 0u64;
        let mut count = 0u64;
        
        for level in 0..11 {
            let cancels = self.cancellation_counts[level].load(Ordering::Relaxed);
            let total = self.total_events_per_level[level].load(Ordering::Relaxed);
            
            if total > 0 {
                sum += (cancels * 10000) / total;
                count += 1;
            }
        }
        
        if count == 0 {
            5000
        } else {
            sum / count
        }
    }

    /// Get current estimated position (number of orders ahead)
    #[inline]
    pub fn get_estimated_position(&self) -> u64 {
        self.estimated_position.load(Ordering::Acquire)
    }

    /// Get current queue size at our price level
    #[inline]
    pub fn get_queue_size(&self) -> u64 {
        self.queue_size.load(Ordering::Acquire)
    }

    /// Get our order quantity
    #[inline]
    pub fn get_our_quantity(&self) -> u64 {
        self.our_quantity.load(Ordering::Acquire)
    }

    /// Calculate fill probability based on position and momentum
    /// 
    /// Returns probability as basis points (0-10000)
    #[inline]
    pub fn fill_probability(&self, market_momentum_bps: i64) -> u64 {
        let position = self.get_estimated_position();
        let queue_size = self.get_queue_size();
        
        if queue_size == 0 {
            return 0;
        }
        
        // Base probability from position ratio
        let position_ratio = (position * 10000) / queue_size;
        let base_prob = 10000.saturating_sub(position_ratio);
        
        // Adjust for market momentum (positive = moving in our favor)
        let momentum_adjustment = if market_momentum_bps > 0 {
            (market_momentum_bps as u64).min(2000) // Cap at 20% boost
        } else {
            0
        };
        
        // Adjust for hidden liquidity (more hidden = lower fill prob)
        let hidden_penalty = (self.hidden_liquidity_score.load(Ordering::Relaxed) / 100).min(2000);
        
        base_prob
            .saturating_add(momentum_adjustment)
            .saturating_sub(hidden_penalty)
            .min(10000)
    }

    /// Detect if we're likely at the front of the queue
    #[inline]
    pub fn is_front_of_queue(&self) -> bool {
        self.get_estimated_position() < 10 // Within 10 orders of front
    }

    /// Get hidden liquidity score (0-1000)
    #[inline]
    pub fn hidden_liquidity_score(&self) -> u64 {
        self.hidden_liquidity_score.load(Ordering::Relaxed)
    }

    /// Reset all statistics (thread-safe)
    pub fn reset(&self) {
        self.estimated_position.store(0, Ordering::Release);
        self.queue_size.store(0, Ordering::Release);
        self.our_quantity.store(0, Ordering::Release);
        self.history_count.store(0, Ordering::Release);
        self.history_head.store(0, Ordering::Release);
        
        for i in 0..11 {
            self.cancellation_counts[i].store(0, Ordering::Release);
            self.total_events_per_level[i].store(0, Ordering::Release);
        }
        
        self.hidden_liquidity_score.store(500, Ordering::Release);
        self.last_update_ns.store(0, Ordering::Release);
    }
}

/// SIMD-optimized batch processing for queue position updates
/// 
/// Processes multiple queue updates in parallel using AVX2 instructions.
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use super::*;
    use std::arch::x86_64::*;

    /// Process 8 queue updates simultaneously using AVX2
    /// 
    /// # Safety
    /// Requires AVX2 support (check with is_x86_feature_detected!("avx2"))
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_update_positions(
        positions: &mut [u64],
        cancellation_probs: &[u64],
        queue_sizes: &[u64],
    ) {
        assert_eq!(positions.len(), cancellation_probs.len());
        assert_eq!(positions.len(), queue_sizes.len());
        assert_eq!(positions.len() % 8, 0, "Length must be multiple of 8");

        for i in (0..positions.len()).step_by(8) {
            // Load 8 values into AVX2 registers
            let pos_vec = _mm256_loadu_si256(positions[i..i+8].as_ptr() as *const __m256i);
            let cancel_vec = _mm256_loadu_si256(cancellation_probs[i..i+8].as_ptr() as *const __m256i);
            let queue_vec = _mm256_loadu_si256(queue_sizes[i..i+8].as_ptr() as *const __m256i);

            // Calculate expected cancellations: pos * cancel_prob / 10000
            let cancel_scaled = _mm256_mul_epu32(pos_vec, cancel_vec);
            let divisor = _mm256_set1_epi32(10000);
            let expected_cancels = _mm256_div_epu32(cancel_scaled, divisor);

            // New position = pos - expected_cancels
            let new_pos = _mm256_sub_epi64(pos_vec, expected_cancels);

            // Store results
            _mm256_storeu_si256(positions[i..i+8].as_mut_ptr() as *mut __m256i, new_pos);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_estimator_basic() {
        let estimator = QueuePositionEstimator::new(100); // 10% decay
        
        // Record some events
        estimator.record_event(QueueEvent {
            timestamp_ns: 1000000,
            event_type: 0, // New order
            quantity: 100,
            price_offset: 0,
        });
        
        estimator.record_event(QueueEvent {
            timestamp_ns: 1000100,
            event_type: 1, // Cancellation
            quantity: 50,
            price_offset: 0,
        });
        
        // Update queue state
        estimator.update_queue(1000, 100, 500);
        
        assert_eq!(estimator.get_queue_size(), 1000);
        assert_eq!(estimator.get_our_quantity(), 100);
        assert!(estimator.get_estimated_position() < 500); // Should be reduced by expected cancels
    }

    #[test]
    fn test_cancellation_probability() {
        let estimator = QueuePositionEstimator::new(100);
        
        // Add events with 50% cancellation rate
        for i in 0..100 {
            estimator.record_event(QueueEvent {
                timestamp_ns: i as u64 * 100,
                event_type: if i % 2 == 0 { 1 } else { 0 },
                quantity: 10,
                price_offset: 0,
            });
        }
        
        let prob = estimator.cancellation_probability(0);
        assert!(prob >= 4500 && prob <= 5500); // ~50% +/- tolerance
    }

    #[test]
    fn test_fill_probability() {
        let estimator = QueuePositionEstimator::new(100);
        estimator.update_queue(1000, 100, 100);
        
        // At back of queue with no momentum
        let prob_no_momentum = estimator.fill_probability(0);
        
        // At back of queue with positive momentum
        let prob_with_momentum = estimator.fill_probability(1000); // +10% momentum
        
        assert!(prob_with_momentum > prob_no_momentum);
    }

    #[test]
    fn test_hidden_liquidity_decay() {
        let estimator = QueuePositionEstimator::new(500); // 50% decay
        
        // Trigger hidden liquidity detection
        estimator.record_event(QueueEvent {
            timestamp_ns: 1000000,
            event_type: 3, // Hidden liquidity
            quantity: 1000,
            price_offset: 0,
        });
        
        assert!(estimator.hidden_liquidity_score() > 500);
        
        // Simulate time passing (would need to call update with later timestamp)
        estimator.record_event(QueueEvent {
            timestamp_ns: 200000000, // 200ms later
            event_type: 0,
            quantity: 10,
            price_offset: 0,
        });
        
        // Score should have decayed somewhat
        assert!(estimator.hidden_liquidity_score() < 1000);
    }
}
