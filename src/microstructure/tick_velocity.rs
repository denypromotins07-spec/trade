//! # Tick Velocity Calculator
//! 
//! Computes real-time tick velocity and acceleration using lock-free exponential
//! moving averages, feeding momentum signals directly to the execution router.
//! 
//! Optimized for AMD Ryzen AI 5 with strict 8GB RAM limit enforcement via ring buffers.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::time::{Duration, Instant};

/// Ring buffer size for tick history (bounded for memory limits)
const TICK_HISTORY_SIZE: usize = 256_000; // ~256K ticks max

/// Lock-free atomic ring buffer for tick data
pub struct TickRingBuffer {
    /// Timestamps in nanoseconds
    timestamps: Box<[AtomicU64; TICK_HISTORY_SIZE]>,
    /// Prices in ticks (signed for direction)
    prices: Box<[AtomicI64; TICK_HISTORY_SIZE]>,
    /// Volumes (scaled integers)
    volumes: Box<[AtomicU64; TICK_HISTORY_SIZE]>,
    /// Head pointer (next write position)
    head: AtomicU64,
    /// Count of ticks stored
    count: AtomicU64,
}

impl TickRingBuffer {
    pub fn new() -> Self {
        Self {
            timestamps: Box::new(std::array::from_fn(|_| AtomicU64::new(0))),
            prices: Box::new(std::array::from_fn(|_| AtomicI64::new(0))),
            volumes: Box::new(std::array::from_fn(|_| AtomicU64::new(0))),
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Push a new tick (lock-free)
    #[inline]
    pub fn push(&self, timestamp_ns: u64, price: i64, volume: u64) {
        let head = self.head.fetch_add(1, Ordering::AcqRel);
        let idx = (head as usize) % TICK_HISTORY_SIZE;

        self.timestamps[idx].store(timestamp_ns, Ordering::Release);
        self.prices[idx].store(price, Ordering::Release);
        self.volumes[idx].store(volume, Ordering::Release);

        // Update count with wraparound handling
        let count = self.count.load(Ordering::Acquire);
        if count < TICK_HISTORY_SIZE as u64 {
            self.count.fetch_add(1, Ordering::Release);
        }
    }

    /// Get tick at relative position (0 = latest, -1 = previous, etc.)
    #[inline]
    pub fn get_tick(&self, offset: usize) -> Option<(u64, i64, u64)> {
        let count = self.count.load(Ordering::Acquire);
        if offset >= count as usize {
            return None;
        }

        let head = self.head.load(Ordering::Acquire);
        let idx = ((head as usize).wrapping_sub(offset + 1)) % TICK_HISTORY_SIZE;

        Some((
            self.timestamps[idx].load(Ordering::Acquire),
            self.prices[idx].load(Ordering::Acquire),
            self.volumes[idx].load(Ordering::Acquire),
        ))
    }

    /// Current number of ticks stored
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire) as usize
    }

    /// Memory footprint in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        TICK_HISTORY_SIZE * (std::mem::size_of::<u64>() * 2 + std::mem::size_of::<i64>())
    }
}

impl Default for TickRingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Tick velocity and acceleration calculator
pub struct TickVelocityCalculator {
    /// Ring buffer for tick history
    buffer: TickRingBuffer,
    /// EMA of tick velocity (ticks per second)
    velocity_ema: f64,
    /// EMA of tick acceleration (velocity change per second)
    acceleration_ema: f64,
    /// Last tick timestamp for delta calculation
    last_timestamp_ns: u64,
    /// Last tick price
    last_price: i64,
    /// EMA decay factor for velocity
    velocity_alpha: f64,
    /// EMA decay factor for acceleration
    acceleration_alpha: f64,
    /// Volume-weighted velocity component
    volume_weighted_velocity: f64,
}

impl TickVelocityCalculator {
    pub fn new(velocity_alpha: f64, acceleration_alpha: f64) -> Self {
        Self {
            buffer: TickRingBuffer::new(),
            velocity_ema: 0.0,
            acceleration_ema: 0.0,
            last_timestamp_ns: 0,
            last_price: 0,
            velocity_alpha,
            acceleration_alpha,
            volume_weighted_velocity: 0.0,
        }
    }

    /// Process a new tick and return velocity signal
    pub fn update(&mut self, timestamp_ns: u64, price: i64, volume: u64) -> TickVelocitySignal {
        // Calculate time delta
        let time_delta_ns = if self.last_timestamp_ns > 0 {
            timestamp_ns.saturating_sub(self.last_timestamp_ns)
        } else {
            1_000_000_000 // Default to 1 second for first tick
        };

        // Calculate price change (tick direction)
        let price_change = price - self.last_price;

        // Calculate instantaneous velocity (ticks per second)
        let time_delta_sec = time_delta_ns as f64 / 1_000_000_000.0;
        let instant_velocity = if time_delta_sec > 0.0 {
            price_change as f64 / time_delta_sec
        } else {
            0.0
        };

        // Calculate volume-weighted velocity
        let volume_factor = (volume as f64).sqrt() / 100.0; // Normalize volume impact
        let volume_weighted_instant = instant_velocity * volume_factor;

        // Update EMAs
        let prev_velocity = self.velocity_ema;
        self.velocity_ema = self.velocity_alpha * instant_velocity
            + (1.0 - self.velocity_alpha) * self.velocity_ema;

        let instant_acceleration = if self.last_timestamp_ns > 0 && time_delta_sec > 0.0 {
            (instant_velocity - prev_velocity) / time_delta_sec
        } else {
            0.0
        };

        self.acceleration_ema = self.acceleration_alpha * instant_acceleration
            + (1.0 - self.acceleration_alpha) * self.acceleration_ema;

        // Update volume-weighted velocity EMA
        self.volume_weighted_velocity = self.velocity_alpha * volume_weighted_instant
            + (1.0 - self.velocity_alpha) * self.volume_weighted_velocity;

        // Store in ring buffer
        self.buffer.push(timestamp_ns, price, volume);

        // Detect velocity regime changes
        let velocity_regime = self.classify_velocity_regime();
        let acceleration_sign_change = (prev_velocity - self.velocity_ema) * self.acceleration_ema < 0.0;

        // Update state
        self.last_timestamp_ns = timestamp_ns;
        self.last_price = price;

        TickVelocitySignal {
            instantaneous_velocity: instant_velocity,
            velocity_ema: self.velocity_ema,
            acceleration_ema: self.acceleration_ema,
            volume_weighted_velocity: self.volume_weighted_velocity,
            velocity_regime,
            acceleration_sign_change,
            tick_direction: price_change.signum() as i8,
            time_delta_ms: time_delta_ns as f64 / 1_000_000.0,
            timestamp_ns,
            buffer_memory_bytes: self.buffer.memory_bytes(),
        }
    }

    /// Classify current velocity regime
    fn classify_velocity_regime(&self) -> VelocityRegime {
        let abs_velocity = self.velocity_ema.abs();
        let abs_acceleration = self.acceleration_ema.abs();

        match (abs_velocity, abs_acceleration) {
            (v, _) if v < 10.0 => VelocityRegime::Stagnant,      // < 10 ticks/sec
            (v, a) if v < 100.0 && a < 50.0 => VelocityRegime::Normal,  // 10-100 ticks/sec
            (v, a) if v >= 100.0 && a < 100.0 => VelocityRegime::HighMomentum,  // Fast but stable
            (_, a) if a >= 100.0 => VelocityRegime::Accelerating,  // Rapid acceleration
            _ => VelocityRegime::Normal,
        }
    }

    /// Get reference to tick buffer
    pub fn buffer(&self) -> &TickRingBuffer {
        &self.buffer
    }

    /// Calculate tick imbalance over recent window
    pub fn tick_imbalance(&self, window_ticks: usize) -> f64 {
        let mut buy_volume: u64 = 0;
        let mut sell_volume: u64 = 0;

        for i in 0..window_ticks {
            if let Some((_, price, volume)) = self.buffer.get_tick(i) {
                if i == 0 {
                    continue; // Skip current tick
                }
                if let Some((_, prev_price, _)) = self.buffer.get_tick(i - 1) {
                    if price > prev_price {
                        buy_volume = buy_volume.saturating_add(volume);
                    } else if price < prev_price {
                        sell_volume = sell_volume.saturating_add(volume);
                    }
                }
            }
        }

        let total = buy_volume.saturating_add(sell_volume);
        if total == 0 {
            return 0.0;
        }

        (buy_volume as i128 - sell_volume as i128) as f64 / total as f64
    }

    /// Verify memory compliance
    pub fn verify_memory_limit(&self, global_used_bytes: usize) -> bool {
        const GLOBAL_LIMIT_BYTES: usize = 8 * 1024 * 1024 * 1024; // 8GB
        global_used_bytes <= GLOBAL_LIMIT_BYTES
    }
}

/// Velocity regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityRegime {
    /// Very low activity (< 10 ticks/sec)
    Stagnant,
    /// Normal trading activity (10-100 ticks/sec)
    Normal,
    /// High momentum (> 100 ticks/sec)
    HighMomentum,
    /// Rapidly accelerating (potential breakout/crash)
    Accelerating,
}

/// Signal output from tick velocity calculation
#[derive(Debug, Clone)]
pub struct TickVelocitySignal {
    /// Instantaneous velocity (ticks per second)
    pub instantaneous_velocity: f64,
    /// EMA-smoothed velocity
    pub velocity_ema: f64,
    /// EMA-smoothed acceleration
    pub acceleration_ema: f64,
    /// Volume-weighted velocity
    pub volume_weighted_velocity: f64,
    /// Current velocity regime
    pub velocity_regime: VelocityRegime,
    /// Whether acceleration changed sign
    pub acceleration_sign_change: bool,
    /// Direction of last tick (-1, 0, 1)
    pub tick_direction: i8,
    /// Time since last tick in milliseconds
    pub time_delta_ms: f64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Buffer memory usage in bytes
    pub buffer_memory_bytes: usize,
}

impl TickVelocitySignal {
    /// Generate momentum signal for execution router
    /// Returns: -1 (bearish), 0 (neutral), 1 (bullish)
    pub fn momentum_signal(&self, velocity_threshold: f64) -> i8 {
        if self.velocity_ema > velocity_threshold {
            1
        } else if self.velocity_ema < -velocity_threshold {
            -1
        } else {
            0
        }
    }

    /// Check if regime indicates potential breakout
    pub fn is_breakout_regime(&self) -> bool {
        matches!(self.velocity_regime, VelocityRegime::Accelerating | VelocityRegime::HighMomentum)
    }

    /// Get recommended order urgency based on velocity
    pub fn recommended_urgency(&self) -> OrderUrgency {
        match self.velocity_regime {
            VelocityRegime::Accelerating => OrderUrgency::Immediate,
            VelocityRegime::HighMomentum => OrderUrgency::High,
            VelocityRegime::Normal => OrderUrgency::Normal,
            VelocityRegime::Stagnant => OrderUrgency::Low,
        }
    }
}

/// Order urgency recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderUrgency {
    /// Execute immediately (market order)
    Immediate,
    /// High priority (aggressive limit)
    High,
    /// Normal priority
    Normal,
    /// Low priority (passive limit)
    Low,
}

/// Configuration for tick velocity calculator
#[derive(Debug, Clone)]
pub struct TickVelocityConfig {
    /// EMA alpha for velocity
    pub velocity_alpha: f64,
    /// EMA alpha for acceleration
    pub acceleration_alpha: f64,
    /// Window for tick imbalance calculation
    pub imbalance_window: usize,
    /// Velocity threshold for signal generation
    pub signal_threshold: f64,
}

impl Default for TickVelocityConfig {
    fn default() -> Self {
        Self {
            velocity_alpha: 0.1,
            acceleration_alpha: 0.2,
            imbalance_window: 50,
            signal_threshold: 50.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_velocity_basic() {
        let mut calc = TickVelocityCalculator::new(0.1, 0.2);

        // Simulate rapid upward ticks
        let base_time = 1_000_000_000_000u64;
        for i in 0..10 {
            let ts = base_time + (i as u64 * 10_000_000); // 10ms intervals
            let price = 1000 + i as i64;
            calc.update(ts, price, 100);
        }

        // Should detect positive velocity
        let signal = calc.update(base_time + 100_000_000, 1015, 100);
        assert!(signal.velocity_ema > 0.0, "Should detect upward velocity");
    }

    #[test]
    fn test_tick_ring_buffer() {
        let buffer = TickRingBuffer::new();
        
        // Push some ticks
        for i in 0..100 {
            buffer.push(i * 1_000_000, i as i64, 100);
        }

        assert_eq!(buffer.len(), 100);
        
        // Get latest tick
        let (ts, price, vol) = buffer.get_tick(0).unwrap();
        assert_eq!(price, 99);
    }

    #[test]
    fn test_memory_bounds() {
        let calc = TickVelocityCalculator::new(0.1, 0.2);
        let mem = calc.buffer().memory_bytes();
        
        println!("Tick buffer memory: {} bytes", mem);
        assert!(mem > 0);
        
        // Verify within reasonable bounds
        assert!(mem < 10 * 1024 * 1024); // Less than 10MB
    }

    #[test]
    fn test_velocity_regime_classification() {
        let mut calc = TickVelocityCalculator::new(0.1, 0.2);
        
        // Initial state should be stagnant
        let signal = calc.update(1_000_000_000_000, 1000, 100);
        assert_eq!(signal.velocity_regime, VelocityRegime::Stagnant);
    }
}
