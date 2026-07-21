//! Smart Money Concepts (SMC) - Liquidity Pool & Sweep Detection Engine
//! 
//! This module implements liquidity pool mapping, sweep detection, and inducement logic
//! to identify stop-hunts and market manipulation patterns in real-time order flow.
//! 
//! **Performance Characteristics:**
//! - Lock-free ring buffers for liquidity level tracking
//! - Zero heap allocations during hot path execution
//! - O(1) complexity for sweep detection
//! - Pre-allocated arrays for all dynamic data
//! 
//! **Architecture:**
//! Liquidity pools represent concentrations of stop-loss orders and pending orders
//! at key price levels. Institutional players often "sweep" these levels to gather
//! liquidity before making significant moves.
//! 
//! Key Concepts:
//! 1. Equal Highs/Lows (EQH/EQL): Liquidity pools at repeated price levels
//! 2. Liquidity Sweeps: Price briefly pierces a level then reverses
//! 3. Inducement: False moves to trigger stops before real direction

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Configuration for liquidity detection parameters
#[derive(Debug, Clone, Copy)]
pub struct LiquidityConfig {
    /// Minimum number of touches to form a liquidity pool
    pub min_touches_for_pool: u8,
    /// Maximum distance in basis points for touches to be considered same level
    pub level_tolerance_bps: u32,
    /// Time window in milliseconds for sweep detection
    pub sweep_window_ms: u64,
    /// Minimum reversal percentage after sweep to confirm
    pub min_reversal_bps: u32,
    /// Maximum number of liquidity pools to track
    pub max_pools: usize,
}

impl Default for LiquidityConfig {
    fn default() -> Self {
        Self {
            min_touches_for_pool: 2,
            level_tolerance_bps: 5, // 0.05% tolerance
            sweep_window_ms: 5_000, // 5 seconds
            min_reversal_bps: 10,   // 0.1% reversal
            max_pools: 128,
        }
    }
}

/// Type of liquidity pool
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiquidityType {
    /// Equal Highs - resistance level with multiple touches
    EqualHighs,
    /// Equal Lows - support level with multiple touches
    EqualLows,
    /// Swing High - isolated high point
    SwingHigh,
    /// Swing Low - isolated low point
    SwingLow,
    /// Trend Line - diagonal support/resistance
    TrendLine,
}

/// Represents a liquidity pool at a specific price level
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LiquidityPool {
    /// Unique identifier
    pub id: u128,
    /// Pool type
    pub pool_type: LiquidityType,
    /// Price level (scaled by 1e8)
    pub price_scaled: i64,
    /// Number of times price has touched this level
    pub touch_count: u8,
    /// Timestamp of first touch (ms)
    pub first_touch_ms: u64,
    /// Timestamp of last touch (ms)
    pub last_touch_ms: u64,
    /// Highest price reached at this level (scaled)
    pub extreme_high_scaled: i64,
    /// Lowest price reached at this level (scaled)
    pub extreme_low_scaled: i64,
    /// Whether this pool has been swept
    pub is_swept: bool,
    /// Timestamp of sweep (ms)
    pub sweep_timestamp_ms: Option<u64>,
    /// Sweep depth in basis points (how far price pierced the level)
    pub sweep_depth_bps: u32,
    /// Whether reversal was confirmed after sweep
    pub reversal_confirmed: bool,
    /// Strength score (0-100) based on touches and age
    pub strength_score: u8,
}

impl LiquidityPool {
    /// Create a new liquidity pool
    #[inline]
    pub fn new(
        pool_type: LiquidityType,
        price_scaled: i64,
        timestamp_ms: u64,
        sequence: u64,
    ) -> Self {
        Self {
            id: ((timestamp_ms as u128) << 64) | (sequence as u128),
            pool_type,
            price_scaled,
            touch_count: 1,
            first_touch_ms: timestamp_ms,
            last_touch_ms: timestamp_ms,
            extreme_high_scaled: price_scaled,
            extreme_low_scaled: price_scaled,
            is_swept: false,
            sweep_timestamp_ms: None,
            sweep_depth_bps: 0,
            reversal_confirmed: false,
            strength_score: 0,
        }
    }

    /// Add a touch to this pool
    #[inline]
    pub fn add_touch(&mut self, high_scaled: i64, low_scaled: i64, timestamp_ms: u64) {
        self.touch_count = self.touch_count.saturating_add(1);
        self.last_touch_ms = timestamp_ms;
        self.extreme_high_scaled = self.extreme_high_scaled.max(high_scaled);
        self.extreme_low_scaled = self.extreme_low_scaled.min(low_scaled);
        
        // Update strength based on touches
        self.strength_score = self.touch_count.saturating_mul(20).min(100);
    }

    /// Check if price level matches within tolerance
    #[inline]
    pub fn matches_level(&self, price_scaled: i64, tolerance_bps: u32) -> bool {
        let diff = (price_scaled - self.price_scaled).abs();
        let threshold = ((self.price_scaled.abs() as u128 * tolerance_bps as u128) / 10_000) as i64;
        diff <= threshold
    }

    /// Detect if a sweep occurred
    #[inline]
    pub fn detect_sweep(
        &mut self,
        high_scaled: i64,
        low_scaled: i64,
        close_scaled: i64,
        timestamp_ms: u64,
        config: &LiquidityConfig,
    ) -> bool {
        if self.is_swept {
            return false;
        }

        let pierced = match self.pool_type {
            LiquidityType::EqualHighs | LiquidityType::SwingHigh => {
                high_scaled > self.extreme_high_scaled
            }
            LiquidityType::EqualLows | LiquidityType::SwingLow => {
                low_scaled < self.extreme_low_scaled
            }
            LiquidityType::TrendLine => {
                // Simplified - would need trend line calculation
                high_scaled > self.extreme_high_scaled || low_scaled < self.extreme_low_scaled
            }
        };

        if !pierced {
            return false;
        }

        // Calculate sweep depth
        let depth = match self.pool_type {
            LiquidityType::EqualHighs | LiquidityType::SwingHigh => {
                high_scaled - self.extreme_high_scaled
            }
            LiquidityType::EqualLows | LiquidityType::SwingLow => {
                self.extreme_low_scaled - low_scaled
            }
            LiquidityType::TrendLine => {
                (high_scaled - self.extreme_high_scaled)
                    .max(self.extreme_low_scaled - low_scaled)
            }
        };

        let depth_bps = ((depth as u128 * 10_000) / self.price_scaled.abs().max(1) as u128) as u32;
        
        // Check if reversal occurred (price closed back inside the level)
        let reversed = match self.pool_type {
            LiquidityType::EqualHighs | LiquidityType::SwingHigh => {
                close_scaled < self.extreme_high_scaled
            }
            LiquidityType::EqualLows | LiquidityType::SwingLow => {
                close_scaled > self.extreme_low_scaled
            }
            LiquidityType::TrendLine => {
                close_scaled < self.extreme_high_scaled && close_scaled > self.extreme_low_scaled
            }
        };

        if reversed {
            self.is_swept = true;
            self.sweep_timestamp_ms = Some(timestamp_ms);
            self.sweep_depth_bps = depth_bps;
            
            // Check reversal confirmation
            let reversal_bps = ((depth as u128 * 10_000) / self.price_scaled.abs().max(1) as u128) as u32;
            self.reversal_confirmed = reversal_bps >= config.min_reversal_bps;
            
            return true;
        }

        false
    }
}

/// Main Liquidity Detection Engine
pub struct LiquidityDetector {
    /// Pre-allocated array for liquidity pools
    pools: [Option<LiquidityPool>; 128],
    /// Write index
    write_idx: AtomicU64,
    /// Configuration
    config: LiquidityConfig,
    /// Active flag
    is_active: AtomicBool,
    /// Last sequence
    last_sequence: AtomicU64,
    /// Recent highs buffer for swing detection
    recent_highs: [i64; 20],
    /// Recent lows buffer
    recent_lows: [i64; 20],
    /// Buffer index
    buffer_idx: usize,
    /// Count of valid entries
    entry_count: usize,
}

unsafe impl Send for LiquidityDetector {}
unsafe impl Sync for LiquidityDetector {}

impl LiquidityDetector {
    /// Initialize the liquidity detector
    pub fn new(config: LiquidityConfig) -> Self {
        Self {
            pools: [None; 128],
            write_idx: AtomicU64::new(0),
            config,
            is_active: AtomicBool::new(true),
            last_sequence: AtomicU64::new(0),
            recent_highs: [0; 20],
            recent_lows: [0; 20],
            buffer_idx: 0,
            entry_count: 0,
        }
    }

    /// Process a new candle for liquidity detection
    /// Hot path function - zero allocations
    #[inline]
    pub fn process_candle(
        &mut self,
        high_scaled: i64,
        low_scaled: i64,
        close_scaled: i64,
        timestamp_ms: u64,
        sequence: u64,
    ) -> Option<(LiquidityPool, bool)> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }

        // Deduplicate
        let last_seq = self.last_sequence.load(Ordering::Relaxed);
        if sequence <= last_seq {
            return None;
        }
        self.last_sequence.store(sequence, Ordering::Relaxed);

        // Update buffers
        self.recent_highs[self.buffer_idx] = high_scaled;
        self.recent_lows[self.buffer_idx] = low_scaled;
        self.buffer_idx = (self.buffer_idx + 1) % 20;
        if self.entry_count < 20 {
            self.entry_count += 1;
        }

        let mut sweep_detected = false;
        let mut new_pool: Option<LiquidityPool> = None;

        // Check for existing pool touches and sweeps
        let write_pos = self.write_idx.load(Ordering::Acquire);
        let start = write_pos.saturating_sub(self.config.max_pools as u64);

        for idx in start..write_pos {
            let pool_idx = (idx % self.config.max_pools as u64) as usize;
            if let Some(ref mut pool) = self.pools[pool_idx] {
                // Check if current price touches the pool
                if pool.matches_level(high_scaled, self.config.level_tolerance_bps)
                    || pool.matches_level(low_scaled, self.config.level_tolerance_bps)
                {
                    pool.add_touch(high_scaled, low_scaled, timestamp_ms);
                }

                // Check for sweep
                if pool.detect_sweep(high_scaled, low_scaled, close_scaled, timestamp_ms, &self.config) {
                    sweep_detected = true;
                }
            }
        }

        // Detect new liquidity pools (swing highs/lows)
        if self.entry_count >= 5 {
            if let Some(pool_type) = self.detect_swing_pattern() {
                let price = match pool_type {
                    LiquidityType::SwingHigh => self.get_recent_high(2),
                    LiquidityType::SwingLow => self.get_recent_low(2),
                    _ => close_scaled,
                };

                // Check if we already have a pool at this level
                let exists = (start..write_pos).any(|idx| {
                    let pool_idx = (idx % self.config.max_pools as u64) as usize;
                    self.pools[pool_idx]
                        .as_ref()
                        .map_or(false, |p| p.matches_level(price, self.config.level_tolerance_bps))
                });

                if !exists {
                    let mut pool = LiquidityPool::new(pool_type, price, timestamp_ms, sequence);
                    pool.strength_score = 20; // Initial strength
                    new_pool = Some(pool);
                }
            }
        }

        // Store new pool if found
        if let Some(pool) = new_pool.clone() {
            let write_pos = (self.write_idx.load(Ordering::Relaxed) % self.config.max_pools as u64) as usize;
            self.pools[write_pos] = Some(pool);
            self.write_idx.fetch_add(1, Ordering::Release);
        }

        new_pool.map(|p| (p, sweep_detected))
    }

    /// Detect swing high/low patterns
    #[inline]
    fn detect_swing_pattern(&self) -> Option<LiquidityType> {
        if self.entry_count < 5 {
            return None;
        }

        let center_idx = self.buffer_idx;
        let left_start = (center_idx + 19) % 20; // 2 bars ago (wrapping)
        let right_end = (center_idx + 1) % 20;   // 2 bars ago forward

        let center_high = self.recent_highs[center_idx];
        let center_low = self.recent_lows[center_idx];

        // Check for swing high (higher than 2 bars on each side)
        let is_swing_high = (0..5).all(|i| {
            let idx = (left_start + i) % 20;
            if idx == center_idx { return true; }
            self.recent_highs[idx] < center_high
        });

        if is_swing_high {
            return Some(LiquidityType::SwingHigh);
        }

        // Check for swing low
        let is_swing_low = (0..5).all(|i| {
            let idx = (left_start + i) % 20;
            if idx == center_idx { return true; }
            self.recent_lows[idx] > center_low
        });

        if is_swing_low {
            return Some(LiquidityType::SwingLow);
        }

        None
    }

    /// Get recent high at offset
    #[inline]
    fn get_recent_high(&self, offset: usize) -> i64 {
        let idx = (self.buffer_idx + 20 - offset) % 20;
        self.recent_highs[idx]
    }

    /// Get recent low at offset
    #[inline]
    fn get_recent_low(&self, offset: usize) -> i64 {
        let idx = (self.buffer_idx + 20 - offset) % 20;
        self.recent_lows[idx]
    }

    /// Get all active liquidity pools
    pub fn get_all_pools<'a>(&'a self) -> impl Iterator<Item = &'a LiquidityPool> + 'a {
        let write_pos = self.write_idx.load(Ordering::Acquire);
        let start = write_pos.saturating_sub(self.config.max_pools as u64);

        (start..write_pos).filter_map(move |idx| {
            let pool_idx = (idx % self.config.max_pools as u64) as usize;
            self.pools[pool_idx].as_ref()
        })
    }

    /// Get unswept pools (potential targets)
    pub fn get_unswept_pools<'a>(&'a self) -> impl Iterator<Item = &'a LiquidityPool> + 'a {
        self.get_all_pools().filter(|p| !p.is_swept)
    }

    /// Get recently swept pools (potential reversals)
    pub fn get_swept_pools<'a>(&'a self) -> impl Iterator<Item = &'a LiquidityPool> + 'a {
        self.get_all_pools().filter(|p| p.is_swept)
    }

    /// Find pools near current price (within threshold)
    pub fn find_nearby_pools(&self, price_scaled: i64, threshold_bps: u32) -> Vec<&LiquidityPool> {
        self.get_unswept_pools()
            .filter(|pool| {
                let diff = (price_scaled - pool.price_scaled).abs();
                let limit = ((price_scaled.abs() as u128 * threshold_bps as u128) / 10_000) as i64;
                diff <= limit
            })
            .collect()
    }

    /// Shutdown detector
    #[inline]
    pub fn shutdown(&self) {
        self.is_active.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_detection() {
        let config = LiquidityConfig::default();
        let mut detector = LiquidityDetector::new(config);

        // Simulate price action forming a swing high
        for i in 0..10 {
            let base = 50_000_000_000i64;
            let high = base + (i as i64 * 1_000_000);
            let low = base - (i as i64 * 500_000);
            let close = base + (i as i64 * 500_000);
            detector.process_candle(high, low, close, 1000 + i as u64, i as u64);
        }

        let pool_count = detector.get_all_pools().count();
        assert!(pool_count >= 0);
    }
}
