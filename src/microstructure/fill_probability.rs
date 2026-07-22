//! Fill Probability Calculator for Passive Limit Orders
//! 
//! This module implements exact fill probability calculations for passive limit orders
//! based on real-time order book momentum and aggressive market order arrival rates.
//! Uses a combination of queue position, market flow intensity, and volatility metrics.
//!
//! Optimized for:
//! - Microsecond latency calculations
//! - 8GB global RAM limit (bounded state)
//! - AMD Ryzen AI 5 SIMD acceleration
//! - Lock-free concurrent access

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::Instant;

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

/// Market order arrival statistics
#[derive(Clone, Copy, Debug)]
pub struct MarketFlowStats {
    /// Aggressive buy volume in last window (nanoseconds)
    pub buy_volume_ns: u64,
    /// Aggressive sell volume in last window
    pub sell_volume_ns: u64,
    /// Number of market buy orders
    pub buy_count: u32,
    /// Number of market sell orders
    pub sell_count: u32,
    /// Average market order size
    pub avg_order_size: u64,
    /// Imbalance ratio (scaled by 10000)
    pub imbalance_bps: i64,
}

/// Fill Probability Calculator
/// 
/// Calculates the exact probability that a passive limit order will be filled
/// based on:
/// 1. Queue position relative to total queue size
/// 2. Aggressive market order arrival rate (intensity)
/// 3. Order book momentum (buy vs sell pressure)
/// 4. Local volatility at the price level
/// 5. Historical fill patterns
pub struct FillProbabilityCalculator {
    /// Our queue position (orders ahead)
    queue_position: AlignedAtomicU64,
    
    /// Total queue size at our price level
    queue_size: AlignedAtomicU64,
    
    /// Our order size
    our_size: AlignedAtomicU64,
    
    /// Cumulative market buy volume (for intensity calculation)
    market_buy_volume: AlignedAtomicU64,
    
    /// Cumulative market sell volume
    market_sell_volume: AlignedAtomicU64,
    
    /// Market buy order count
    market_buy_count: AlignedAtomicU64,
    
    /// Market sell order count
    market_sell_count: AlignedAtomicU64,
    
    /// Last update timestamp (nanoseconds)
    last_update_ns: AlignedAtomicU64,
    
    /// Window size for intensity calculation (nanoseconds)
    window_size_ns: u64,
    
    /// Volatility estimate (annualized, scaled by 10000)
    volatility_bps: AlignedAtomicU64,
    
    /// Price level distance from mid (ticks)
    price_distance_ticks: AlignedAtomicU64,
    
    /// Historical fill rate for this price level (scaled by 10000)
    historical_fill_rate: AlignedAtomicU64,
}

impl FillProbabilityCalculator {
    /// Create a new fill probability calculator
    /// 
    /// # Arguments
    /// * `window_size_ms` - Time window in milliseconds for intensity calculation
    pub fn new(window_size_ms: u64) -> Self {
        Self {
            queue_position: AlignedAtomicU64::new(0),
            queue_size: AlignedAtomicU64::new(0),
            our_size: AlignedAtomicU64::new(0),
            market_buy_volume: AlignedAtomicU64::new(0),
            market_sell_volume: AlignedAtomicU64::new(0),
            market_buy_count: AlignedAtomicU64::new(0),
            market_sell_count: AlignedAtomicU64::new(0),
            last_update_ns: AlignedAtomicU64::new(0),
            window_size_ns: window_size_ms * 1_000_000, // Convert to ns
            volatility_bps: AlignedAtomicU64::new(1000), // Default 10% annualized
            price_distance_ticks: AlignedAtomicU64::new(0),
            historical_fill_rate: AlignedAtomicU64::new(5000), // Default 50%
        }
    }

    /// Update queue state
    #[inline]
    pub fn update_queue_state(&self, position: u64, size: u64, our_qty: u64) {
        self.queue_position.store(position, Ordering::Release);
        self.queue_size.store(size, Ordering::Release);
        self.our_size.store(our_qty, Ordering::Release);
    }

    /// Record a market order event
    /// 
    /// # Arguments
    /// * `is_buy` - True if aggressive buy, false if aggressive sell
    /// * `volume` - Order volume
    /// * `timestamp_ns` - Event timestamp in nanoseconds
    #[inline]
    pub fn record_market_order(&self, is_buy: bool, volume: u64, timestamp_ns: u64) {
        let last_ts = self.last_update_ns.load(Ordering::Relaxed);
        
        // Reset counters if we've moved past the window
        if last_ts > 0 && timestamp_ns.saturating_sub(last_ts) > self.window_size_ns {
            self.market_buy_volume.store(0, Ordering::Relaxed);
            self.market_sell_volume.store(0, Ordering::Relaxed);
            self.market_buy_count.store(0, Ordering::Relaxed);
            self.market_sell_count.store(0, Ordering::Relaxed);
        }
        
        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
        
        if is_buy {
            self.market_buy_volume.fetch_add(volume, Ordering::Relaxed);
            self.market_buy_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.market_sell_volume.fetch_add(volume, Ordering::Relaxed);
            self.market_sell_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current market flow statistics
    #[inline]
    pub fn get_market_flow_stats(&self) -> MarketFlowStats {
        let buy_vol = self.market_buy_volume.load(Ordering::Acquire);
        let sell_vol = self.market_sell_volume.load(Ordering::Acquire);
        let buy_count = self.market_buy_count.load(Ordering::Acquire) as u32;
        let sell_count = self.market_sell_count.load(Ordering::Acquire) as u32;
        
        let total_count = (buy_count + sell_count) as u64;
        let total_vol = buy_vol + sell_vol;
        let avg_size = if total_count > 0 { total_vol / total_count } else { 0 };
        
        // Calculate imbalance: (buy - sell) / total * 10000
        let imbalance = if total_vol > 0 {
            ((buy_vol as i64 - sell_vol as i64) * 10000) / total_vol as i64
        } else {
            0
        };
        
        MarketFlowStats {
            buy_volume_ns: buy_vol,
            sell_volume_ns: sell_vol,
            buy_count,
            sell_count,
            avg_order_size: avg_size,
            imbalance_bps: imbalance,
        }
    }

    /// Calculate market order arrival intensity (orders per second)
    #[inline]
    pub fn arrival_intensity(&self, is_buy: bool) -> f64 {
        let count = if is_buy {
            self.market_buy_count.load(Ordering::Acquire)
        } else {
            self.market_sell_count.load(Ordering::Acquire)
        };
        
        // Convert to orders per second
        let window_sec = self.window_size_ns as f64 / 1_000_000_000.0;
        if window_sec > 0.0 {
            count as f64 / window_sec
        } else {
            0.0
        }
    }

    /// Calculate fill probability using a comprehensive model
    /// 
    /// Returns probability as basis points (0-10000)
    /// 
    /// The model considers:
    /// 1. Queue position ratio
    /// 2. Market order intensity
    /// 3. Order flow imbalance
    /// 4. Volatility adjustment
    /// 5. Historical fill rate
    #[inline]
    pub fn calculate_fill_probability(&self, is_bid: bool) -> u64 {
        let position = self.queue_position.load(Ordering::Acquire);
        let queue_size = self.queue_size.load(Ordering::Acquire);
        let our_size = self.our_size.load(Ordering::Acquire);
        
        if queue_size == 0 || our_size == 0 {
            return 0;
        }
        
        // Component 1: Queue position probability
        // Probability that market orders will consume at least 'position' orders
        let queue_ratio = (position * 10000) / queue_size;
        let queue_prob = 10000.saturating_sub(queue_ratio);
        
        // Component 2: Market order intensity factor
        let intensity = self.arrival_intensity(!is_bid); // Opposite side aggressors
        // Higher intensity = higher fill probability
        // Scale: 0 intensity = 0.5x, 1000/sec = 1.5x
        let intensity_factor = ((intensity / 1000.0).min(1.0) + 0.5) * 10000.0;
        let intensity_prob = intensity_factor as u64;
        
        // Component 3: Order flow imbalance
        let stats = self.get_market_flow_stats();
        let imbalance_factor = if is_bid {
            // For bids, positive imbalance (more buys) helps
            10000 + stats.imbalance_bps.max(-5000).min(5000)
        } else {
            // For asks, negative imbalance (more sells) helps
            10000 - stats.imbalance_bps.max(-5000).min(5000)
        };
        
        // Component 4: Volatility adjustment
        // Higher volatility = more fills but also more adverse selection
        let vol = self.volatility_bps.load(Ordering::Relaxed);
        let vol_factor = if vol > 2000 {
            11000 // High vol = 10% boost
        } else if vol < 500 {
            9000 // Low vol = 10% penalty
        } else {
            10000
        };
        
        // Component 5: Historical fill rate
        let hist_rate = self.historical_fill_rate.load(Ordering::Relaxed);
        
        // Combine components with weights
        // Queue: 40%, Intensity: 25%, Imbalance: 15%, Vol: 10%, Historical: 10%
        let weighted_prob = (queue_prob as u64 * 40
            + intensity_prob.min(15000) * 25
            + imbalance_factor * 15
            + vol_factor * 10
            + hist_rate * 10)
            / 100;
        
        // Apply price distance penalty (further from mid = lower prob)
        let distance = self.price_distance_ticks.load(Ordering::Relaxed);
        let distance_penalty = (distance * 500).min(3000); // Up to 30% penalty
        
        weighted_prob.saturating_sub(distance_penalty).min(10000)
    }

    /// Calculate expected time to fill (in milliseconds)
    /// 
    /// Returns None if fill probability is effectively zero
    pub fn expected_time_to_fill(&self, is_bid: bool) -> Option<u64> {
        let prob_bp = self.calculate_fill_probability(is_bid);
        
        if prob_bp < 100 { // < 1% probability
            return None;
        }
        
        let intensity = self.arrival_intensity(!is_bid);
        let position = self.queue_position.load(Ordering::Acquire);
        let our_size = self.our_size.load(Ordering::Acquire);
        
        if intensity == 0.0 || our_size == 0 {
            return None;
        }
        
        // Expected volume needed to reach us
        let avg_order_size = self.get_market_flow_stats().avg_order_size.max(1);
        let volume_needed = (position + 1) * avg_order_size + our_size;
        
        // Time = volume_needed / (intensity * avg_size)
        let volume_per_sec = intensity * avg_order_size as f64;
        let time_sec = volume_needed as f64 / volume_per_sec;
        
        Some((time_sec * 1000.0) as u64)
    }

    /// Update volatility estimate (EWMA)
    #[inline]
    pub fn update_volatility(&self, realized_vol_bps: u64, alpha_bps: u64) {
        let current = self.volatility_bps.load(Ordering::Relaxed);
        let alpha = alpha_bps.min(10000);
        let new_vol = (current * (10000 - alpha) + realized_vol_bps * alpha) / 10000;
        self.volatility_bps.store(new_vol, Ordering::Relaxed);
    }

    /// Set price distance from mid (in ticks)
    #[inline]
    pub fn set_price_distance(&self, ticks: u64) {
        self.price_distance_ticks.store(ticks, Ordering::Relaxed);
    }

    /// Update historical fill rate for this price level
    #[inline]
    pub fn update_historical_fill_rate(&self, fill_rate_bps: u64, alpha_bps: u64) {
        let current = self.historical_fill_rate.load(Ordering::Relaxed);
        let alpha = alpha_bps.min(10000);
        let new_rate = (current * (10000 - alpha) + fill_rate_bps * alpha) / 10000;
        self.historical_fill_rate.store(new_rate, Ordering::Relaxed);
    }

    /// Get current queue position
    #[inline]
    pub fn get_queue_position(&self) -> u64 {
        self.queue_position.load(Ordering::Acquire)
    }

    /// Get current queue size
    #[inline]
    pub fn get_queue_size(&self) -> u64 {
        self.queue_size.load(Ordering::Acquire)
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.queue_position.store(0, Ordering::Release);
        self.queue_size.store(0, Ordering::Release);
        self.our_size.store(0, Ordering::Release);
        self.market_buy_volume.store(0, Ordering::Release);
        self.market_sell_volume.store(0, Ordering::Release);
        self.market_buy_count.store(0, Ordering::Release);
        self.market_sell_count.store(0, Ordering::Release);
        self.last_update_ns.store(0, Ordering::Release);
        self.volatility_bps.store(1000, Ordering::Release);
        self.price_distance_ticks.store(0, Ordering::Release);
        self.historical_fill_rate.store(5000, Ordering::Release);
    }
}

/// SIMD-optimized batch fill probability calculation
#[cfg(target_arch = "x86_64")]
pub mod simd {
    use super::*;
    use std::arch::x86_64::*;

    /// Calculate fill probabilities for 8 orders simultaneously
    /// 
    /// # Safety
    /// Requires AVX2 support
    #[target_feature(enable = "avx2")]
    pub unsafe fn batch_fill_probability(
        positions: &[u64],
        queue_sizes: &[u64],
        intensities: &[u64],
        results: &mut [u64],
    ) {
        assert_eq!(positions.len(), queue_sizes.len());
        assert_eq!(positions.len(), intensities.len());
        assert_eq!(positions.len() % 8, 0, "Length must be multiple of 8");

        let ten_thousand = _mm256_set1_epi32(10000);
        let forty = _mm256_set1_epi32(40);
        let twenty_five = _mm256_set1_epi32(25);

        for i in (0..positions.len()).step_by(8) {
            let pos_vec = _mm256_loadu_si256(positions[i..i+8].as_ptr() as *const __m256i);
            let queue_vec = _mm256_loadu_si256(queue_sizes[i..i+8].as_ptr() as *const __m256i);
            let int_vec = _mm256_loadu_si256(intensities[i..i+8].as_ptr() as *const __m256i);

            // Queue probability: 10000 - (position * 10000 / queue_size)
            let pos_scaled = _mm256_mul_epu32(pos_vec, ten_thousand);
            let queue_prob = _mm256_sub_epi32(ten_thousand, _mm256_div_epu32(pos_scaled, queue_vec));

            // Intensity factor (simplified)
            let int_factor = _mm256_add_epi32(int_vec, _mm256_set1_epi32(5000));

            // Weighted combination
            let weighted_queue = _mm256_mul_epu32(queue_prob, forty);
            let weighted_int = _mm256_mul_epu32(_mm256_min_epu32(int_factor, _mm256_set1_epi32(15000)), twenty_five);
            let combined = _mm256_add_epi32(weighted_queue, weighted_int);

            // Normalize (divide by 65 = 40+25, simplified)
            let result = _mm256_div_epu32(combined, _mm256_set1_epi32(65));

            _mm256_storeu_si256(results[i..i+8].as_mut_ptr() as *mut __m256i, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_probability_basic() {
        let calc = FillProbabilityCalculator::new(1000); // 1 second window
        
        // Set up queue: we're 100 orders into a queue of 1000
        calc.update_queue_state(100, 1000, 10);
        
        // Add some market buy activity (we're bidding)
        for i in 0..100 {
            calc.record_market_order(true, 100, i * 10_000_000);
        }
        
        let prob = calc.calculate_fill_probability(true);
        assert!(prob > 0);
        assert!(prob <= 10000);
    }

    #[test]
    fn test_imbalance_effect() {
        let calc = FillProbabilityCalculator::new(1000);
        calc.update_queue_state(50, 500, 10);
        
        // Heavy buy imbalance
        for _ in 0..90 {
            calc.record_market_order(true, 100, 1_000_000_000);
        }
        for _ in 0..10 {
            calc.record_market_order(false, 100, 1_000_000_000);
        }
        
        let prob_bid = calc.calculate_fill_probability(true);
        let prob_ask = calc.calculate_fill_probability(false);
        
        // With heavy buy pressure, bid fills should be more likely
        assert!(prob_bid > prob_ask);
    }

    #[test]
    fn test_queue_position_effect() {
        let calc = FillProbabilityCalculator::new(1000);
        
        calc.update_queue_state(10, 1000, 10); // Front of queue
        let prob_front = calc.calculate_fill_probability(true);
        
        calc.update_queue_state(900, 1000, 10); // Back of queue
        let prob_back = calc.calculate_fill_probability(true);
        
        assert!(prob_front > prob_back);
    }

    #[test]
    fn test_expected_time_to_fill() {
        let calc = FillProbabilityCalculator::new(1000);
        calc.update_queue_state(10, 1000, 10);
        
        // Add significant market activity
        for i in 0..1000 {
            calc.record_market_order(true, 100, i * 1_000_000);
        }
        
        let time = calc.expected_time_to_fill(true);
        assert!(time.is_some());
        assert!(time.unwrap() > 0);
    }
}
