//! # Trailing Stop-Loss Module
//! 
//! Implements volatility-adjusted, tick-level trailing stop-loss algorithms using
//! Average True Range (ATR) and Chandelier Exits, updating strictly within lock-free memory bounds.
//! 
//! ## Features
//! - ATR-based dynamic stop calculation
//! - Chandelier Exit integration for trend following
//! - Lock-free updates for microsecond latency
//! - Optimized for AMD Ryzen AI 5 architecture

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Configuration for trailing stop parameters
#[derive(Debug, Clone)]
pub struct TrailingStopConfig {
    /// ATR multiplier for stop distance (e.g., 2.0 = 2x ATR)
    pub atr_multiplier: f64,
    /// Period for ATR calculation (number of bars)
    pub atr_period: usize,
    /// Chandelier Exit period (typically 22)
    pub chandelier_period: usize,
    /// Minimum stop distance in ticks (to prevent too-tight stops)
    pub min_stop_ticks: i64,
    /// Trail only when profitable (lock-in gains mode)
    pub trail_only_profitable: bool,
}

impl Default for TrailingStopConfig {
    fn default() -> Self {
        Self {
            atr_multiplier: 3.0,
            atr_period: 14,
            chandelier_period: 22,
            min_stop_ticks: 10,
            trail_only_profitable: true,
        }
    }
}

/// Position direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Tick data structure (optimized for cache locality)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Tick {
    /// Timestamp in microseconds
    pub timestamp: u64,
    /// Price in integer ticks (avoid floating point)
    pub price: i64,
    /// Volume
    pub volume: u64,
    /// High for the tick period
    pub high: i64,
    /// Low for the tick period
    pub low: i64,
}

/// Ring buffer for ATR calculation (lock-free, fixed size)
struct AtrBuffer {
    data: Box<[i64]>, // Stores true ranges
    head: AtomicU64,
    count: AtomicU64,
    capacity: usize,
}

impl AtrBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![0; capacity].into_boxed_slice(),
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
            capacity,
        }
    }
    
    #[inline]
    fn push(&self, value: i64) {
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        let index = (head as usize) % self.capacity;
        self.data[index] = value;
        
        // Update count (capped at capacity)
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count < self.capacity as u64 {
            self.count.store(current_count + 1, Ordering::Relaxed);
        }
    }
    
    #[inline]
    fn average(&self) -> f64 {
        let count = self.count.load(Ordering::Acquire) as usize;
        if count == 0 {
            return 0.0;
        }
        
        let mut sum: i64 = 0;
        for i in 0..count {
            sum += self.data[i];
        }
        
        sum as f64 / count as f64
    }
}

/// High-performance Trailing Stop engine
pub struct TrailingStop {
    /// Current stop price (in ticks)
    stop_price: AtomicI64,
    /// Entry price
    entry_price: AtomicI64,
    /// Position side
    side: PositionSide,
    /// Whether stop is active
    is_active: AtomicBool,
    /// ATR buffer for long positions
    atr_long: AtrBuffer,
    /// ATR buffer for short positions  
    atr_short: AtrBuffer,
    /// Previous close for true range calculation
    prev_close_long: AtomicI64,
    prev_close_short: AtomicI64,
    /// Highest price since entry (for long)
    highest_price: AtomicI64,
    /// Lowest price since entry (for short)
    lowest_price: AtomicI64,
    /// Configuration
    config: TrailingStopConfig,
    /// Last update timestamp
    last_update: AtomicU64,
}

impl TrailingStop {
    /// Create a new trailing stop for a position
    pub fn new(side: PositionSide, entry_price: i64, config: TrailingStopConfig) -> Self {
        let atr_size = config.atr_period.max(config.chandelier_period);
        
        Self {
            stop_price: AtomicI64::new(entry_price),
            entry_price: AtomicI64::new(entry_price),
            side,
            is_active: AtomicBool::new(true),
            atr_long: AtrBuffer::new(atr_size),
            atr_short: AtrBuffer::new(atr_size),
            prev_close_long: AtomicI64::new(entry_price),
            prev_close_short: AtomicI64::new(entry_price),
            highest_price: AtomicI64::new(entry_price),
            lowest_price: AtomicI64::new(entry_price),
            config,
            last_update: AtomicU64::new(0),
        }
    }
    
    /// Wrap in Arc for shared access
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }
    
    /// Update with new tick data (called every tick)
    #[inline]
    pub fn update(&self, tick: &Tick) -> Option<i64> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }
        
        match self.side {
            PositionSide::Long => self.update_long(tick),
            PositionSide::Short => self.update_short(tick),
        }
    }
    
    /// Update logic for long positions
    fn update_long(&self, tick: &Tick) -> Option<i64> {
        // Calculate true range
        let prev_close = self.prev_close_long.load(Ordering::Relaxed);
        let true_range = (tick.high - tick.low).max(
            (tick.high - prev_close).abs().max((tick.low - prev_close).abs())
        );
        
        // Update ATR buffer
        self.atr_long.push(true_range);
        self.prev_close_long.store(tick.price, Ordering::Relaxed);
        
        // Update highest price
        let current_high = self.highest_price.load(Ordering::Relaxed);
        if tick.high > current_high {
            self.highest_price.store(tick.high, Ordering::Relaxed);
        }
        
        // Calculate new stop based on ATR
        let atr = self.atr_long.average();
        let atr_distance = (atr * self.config.atr_multiplier) as i64;
        let atr_stop = tick.price - atr_distance.max(self.config.min_stop_ticks);
        
        // Calculate Chandelier Exit (highest high - 3*ATR over period)
        // Simplified: use current highest
        let chandelier_stop = current_high - ((atr * 3.0) as i64);
        
        // Use the higher (tighter) stop
        let new_stop = atr_stop.max(chandelier_stop);
        
        // Get current stop
        let current_stop = self.stop_price.load(Ordering::Acquire);
        
        // For long: stop can only move up (trail higher)
        if new_stop > current_stop {
            // Check if we should trail only when profitable
            if self.config.trail_only_profitable {
                let entry = self.entry_price.load(Ordering::Relaxed);
                if tick.price <= entry {
                    // Not profitable yet, don't trail down
                    self.last_update.store(tick.timestamp, Ordering::Relaxed);
                    return Some(current_stop);
                }
            }
            
            self.stop_price.store(new_stop, Ordering::Release);
            self.last_update.store(tick.timestamp, Ordering::Relaxed);
            Some(new_stop)
        } else {
            self.last_update.store(tick.timestamp, Ordering::Relaxed);
            Some(current_stop)
        }
    }
    
    /// Update logic for short positions
    fn update_short(&self, tick: &Tick) -> Option<i64> {
        // Calculate true range
        let prev_close = self.prev_close_short.load(Ordering::Relaxed);
        let true_range = (tick.high - tick.low).max(
            (tick.high - prev_close).abs().max((tick.low - prev_close).abs())
        );
        
        // Update ATR buffer
        self.atr_short.push(true_range);
        self.prev_close_short.store(tick.price, Ordering::Relaxed);
        
        // Update lowest price
        let current_low = self.lowest_price.load(Ordering::Relaxed);
        if tick.low < current_low || current_low == 0 {
            self.lowest_price.store(tick.low, Ordering::Relaxed);
        }
        
        // Calculate new stop based on ATR
        let atr = self.atr_short.average();
        let atr_distance = (atr * self.config.atr_multiplier) as i64;
        let atr_stop = tick.price + atr_distance.max(self.config.min_stop_ticks);
        
        // Calculate Chandelier Exit for short
        let chandelier_stop = current_low + ((atr * 3.0) as i64);
        
        // Use the lower (tighter) stop
        let new_stop = atr_stop.min(chandelier_stop);
        
        // Get current stop
        let current_stop = self.stop_price.load(Ordering::Acquire);
        
        // For short: stop can only move down (trail lower)
        if new_stop < current_stop {
            if self.config.trail_only_profitable {
                let entry = self.entry_price.load(Ordering::Relaxed);
                if tick.price >= entry {
                    self.last_update.store(tick.timestamp, Ordering::Relaxed);
                    return Some(current_stop);
                }
            }
            
            self.stop_price.store(new_stop, Ordering::Release);
            self.last_update.store(tick.timestamp, Ordering::Relaxed);
            Some(new_stop)
        } else {
            self.last_update.store(tick.timestamp, Ordering::Relaxed);
            Some(current_stop)
        }
    }
    
    /// Check if stop has been hit
    #[inline]
    pub fn check_hit(&self, tick: &Tick) -> bool {
        if !self.is_active.load(Ordering::Acquire) {
            return false;
        }
        
        let stop = self.stop_price.load(Ordering::Acquire);
        
        match self.side {
            PositionSide::Long => tick.low <= stop,
            PositionSide::Short => tick.high >= stop,
        }
    }
    
    /// Deactivate the stop (position closed)
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }
    
    /// Get current stop price
    #[inline]
    pub fn get_stop(&self) -> i64 {
        self.stop_price.load(Ordering::Acquire)
    }
    
    /// Get unrealized PnL in ticks
    #[inline]
    pub fn unrealized_pnl(&self, current_price: i64) -> i64 {
        let entry = self.entry_price.load(Ordering::Relaxed);
        match self.side {
            PositionSide::Long => current_price - entry,
            PositionSide::Short => entry - current_price,
        }
    }
    
    /// Reset for new position
    pub fn reset(&self, side: PositionSide, entry_price: i64) {
        self.side = side;
        self.entry_price.store(entry_price, Ordering::Release);
        self.stop_price.store(entry_price, Ordering::Release);
        self.is_active.store(true, Ordering::Release);
        self.highest_price.store(entry_price, Ordering::Release);
        self.lowest_price.store(entry_price, Ordering::Release);
        // Note: ATR buffers are preserved for continuity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_long_trailing_stop() {
        let config = TrailingStopConfig::default();
        let ts = TrailingStop::new(PositionSide::Long, 10000, config);
        
        // Simulate upward price movement
        let ticks = vec![
            Tick { timestamp: 1, price: 10000, volume: 100, high: 10050, low: 9950 },
            Tick { timestamp: 2, price: 10100, volume: 100, high: 10150, low: 10050 },
            Tick { timestamp: 3, price: 10200, volume: 100, high: 10250, low: 10150 },
        ];
        
        for tick in &ticks {
            ts.update(tick);
        }
        
        // Stop should have trailed up
        let final_stop = ts.get_stop();
        assert!(final_stop > 10000, "Stop should trail up for long position");
        
        // Verify PnL calculation
        let pnl = ts.unrealized_pnl(10200);
        assert_eq!(pnl, 200);
    }
    
    #[test]
    fn test_stop_hit_detection() {
        let config = TrailingStopConfig {
            atr_multiplier: 1.0,
            min_stop_ticks: 50,
            ..Default::default()
        };
        let ts = TrailingStop::new(PositionSide::Long, 10000, config);
        
        // Move price up to establish stop
        let up_tick = Tick { timestamp: 1, price: 10100, volume: 100, high: 10100, low: 10000 };
        ts.update(&up_tick);
        
        let stop = ts.get_stop();
        
        // Simulate stop hit
        let hit_tick = Tick { timestamp: 2, price: stop - 10, volume: 100, high: stop + 10, low: stop - 10 };
        assert!(ts.check_hit(&hit_tick));
    }
}
