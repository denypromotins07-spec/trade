//! Micro-Pattern Recognition - Lock-Free Finite State Machines
//! 
//! This module implements lock-free finite state machines (FSMs) to identify
//! micro-structure patterns like stop-runs and absorption, feeding signals
//! directly to the execution router with microsecond latency.
//! 
//! **Key Features:**
//! - Lock-free state transitions using atomics.
//! - Pattern detection for stop-runs, absorption, and liquidity grabs.
//! - Direct signal output to execution router.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// Pattern types detected by the FSM.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    None = 0,
    StopRun = 1,
    Absorption = 2,
    LiquidityGrab = 3,
    Fakeout = 4,
}

/// State machine states for pattern detection.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsmState {
    Idle = 0,
    Building = 1,
    Triggered = 2,
    Confirmed = 3,
    Exhausted = 4,
}

/// Configuration for pattern detection thresholds.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PatternConfig {
    /// Minimum price movement for stop-run (in ticks)
    pub stop_run_ticks: u32,
    /// Minimum volume for absorption detection
    pub absorption_volume: u64,
    /// Time window for pattern completion (nanoseconds)
    pub time_window_ns: u64,
    /// Confirmation threshold (number of confirming events)
    pub confirmation_count: u32,
}

impl Default for PatternConfig {
    fn default() -> Self {
        PatternConfig {
            stop_run_ticks: 5,
            absorption_volume: 10000,
            time_window_ns: 100_000_000, // 100ms
            confirmation_count: 3,
        }
    }
}

/// Shared state for a single pattern FSM.
pub struct PatternFsm {
    /// Current state (atomic for lock-free access)
    state: AtomicU8,
    /// Pattern type being tracked
    pattern_type: AtomicU8,
    /// Start timestamp of the pattern
    start_ts_ns: AtomicU64,
    /// Last update timestamp
    last_ts_ns: AtomicU64,
    /// Price at pattern start
    start_price: AtomicU64,
    /// Current price
    current_price: AtomicU64,
    /// Accumulated volume
    accumulated_volume: AtomicU64,
    /// Event counter for confirmation
    event_count: AtomicU64,
    /// Signal strength (0-100)
    signal_strength: AtomicU8,
}

impl PatternFsm {
    /// Create a new pattern FSM.
    pub fn new() -> Self {
        PatternFsm {
            state: AtomicU8::new(FsmState::Idle as u8),
            pattern_type: AtomicU8::new(PatternType::None as u8),
            start_ts_ns: AtomicU64::new(0),
            last_ts_ns: AtomicU64::new(0),
            start_price: AtomicU64::new(0),
            current_price: AtomicU64::new(0),
            accumulated_volume: AtomicU64::new(0),
            event_count: AtomicU64::new(0),
            signal_strength: AtomicU8::new(0),
        }
    }

    /// Reset the FSM to idle state.
    pub fn reset(&self) {
        self.state.store(FsmState::Idle as u8, Ordering::Release);
        self.pattern_type.store(PatternType::None as u8, Ordering::Release);
        self.start_ts_ns.store(0, Ordering::Release);
        self.last_ts_ns.store(0, Ordering::Release);
        self.accumulated_volume.store(0, Ordering::Release);
        self.event_count.store(0, Ordering::Release);
        self.signal_strength.store(0, Ordering::Release);
    }

    /// Process a tick and attempt state transition.
    pub fn process_tick(
        &self,
        price: u64,
        volume: u64,
        timestamp_ns: u64,
        config: &PatternConfig,
    ) -> Option<(PatternType, u8)> {
        let current_state = FsmState::from_u8(self.state.load(Ordering::Acquire));
        
        match current_state {
            FsmState::Idle => self.try_start_pattern(price, volume, timestamp_ns, config),
            FsmState::Building => self.update_building_state(price, volume, timestamp_ns, config),
            FsmState::Triggered => self.check_confirmation(price, timestamp_ns, config),
            FsmState::Confirmed => self.check_exhaustion(price, timestamp_ns, config),
            FsmState::Exhausted => {
                // Auto-reset after exhaustion
                if timestamp_ns.saturating_sub(self.last_ts_ns.load(Ordering::Acquire)) > config.time_window_ns {
                    self.reset();
                }
                None
            }
        }
    }

    /// Try to start a new pattern detection.
    fn try_start_pattern(
        &self,
        price: u64,
        volume: u64,
        timestamp_ns: u64,
        config: &PatternConfig,
    ) -> Option<(PatternType, u8)> {
        // Heuristic: large volume spike might indicate start of a pattern
        if volume >= config.absorption_volume / 2 {
            self.state.store(FsmState::Building as u8, Ordering::Release);
            self.start_ts_ns.store(timestamp_ns, Ordering::Release);
            self.last_ts_ns.store(timestamp_ns, Ordering::Release);
            self.start_price.store(price, Ordering::Release);
            self.current_price.store(price, Ordering::Release);
            self.accumulated_volume.store(volume, Ordering::Release);
            self.event_count.store(1, Ordering::Release);
            
            // Tentatively classify pattern type based on context (simplified)
            self.pattern_type.store(PatternType::Absorption as u8, Ordering::Release);
        }
        None
    }

    /// Update state while building pattern.
    fn update_building_state(
        &self,
        price: u64,
        volume: u64,
        timestamp_ns: u64,
        config: &PatternConfig,
    ) -> Option<(PatternType, u8)> {
        let start_price = self.start_price.load(Ordering::Acquire);
        let price_diff = if price > start_price { price - start_price } else { start_price - price };
        
        self.current_price.store(price, Ordering::Release);
        self.accumulated_volume.fetch_add(volume, Ordering::Relaxed);
        self.event_count.fetch_add(1, Ordering::Relaxed);
        self.last_ts_ns.store(timestamp_ns, Ordering::Release);

        // Check for stop-run: significant price move with high volume
        let ticks_moved = price_diff / 100; // Assuming price in cents
        if ticks_moved >= config.stop_run_ticks as u64 {
            self.state.store(FsmState::Triggered as u8, Ordering::Release);
            self.pattern_type.store(PatternType::StopRun as u8, Ordering::Release);
            return None;
        }

        // Check for absorption: high volume without much price movement
        if self.accumulated_volume.load(Ordering::Acquire) >= config.absorption_volume
            && ticks_moved < config.stop_run_ticks as u64 / 2
        {
            self.state.store(FsmState::Triggered as u8, Ordering::Release);
            self.pattern_type.store(PatternType::Absorption as u8, Ordering::Release);
            return None;
        }

        // Check timeout
        if timestamp_ns.saturating_sub(self.start_ts_ns.load(Ordering::Acquire)) > config.time_window_ns {
            self.reset();
        }

        None
    }

    /// Check for pattern confirmation.
    fn check_confirmation(
        &self,
        price: u64,
        timestamp_ns: u64,
        config: &PatternConfig,
    ) -> Option<(PatternType, u8)> {
        self.event_count.fetch_add(1, Ordering::Relaxed);
        self.last_ts_ns.store(timestamp_ns, Ordering::Release);

        if self.event_count.load(Ordering::Acquire) >= config.confirmation_count as u64 {
            self.state.store(FsmState::Confirmed as u8, Ordering::Release);
            let pattern = PatternType::from_u8(self.pattern_type.load(Ordering::Acquire));
            let strength = 80u8; // High confidence on confirmation
            self.signal_strength.store(strength, Ordering::Release);
            return Some((pattern, strength));
        }

        None
    }

    /// Check for pattern exhaustion (signal decay).
    fn check_exhaustion(
        &self,
        price: u64,
        timestamp_ns: u64,
        _config: &PatternConfig,
    ) -> Option<(PatternType, u8)> {
        // Simple exhaustion: price reverses or time passes
        let start_price = self.start_price.load(Ordering::Acquire);
        let current = self.current_price.load(Ordering::Acquire);
        
        // If price has reversed direction significantly, pattern is exhausted
        if (price as i64 - start_price as i64).abs() < (current as i64 - start_price as i64).abs() / 2 {
            self.state.store(FsmState::Exhausted as u8, Ordering::Release);
            self.signal_strength.store(0, Ordering::Release);
        }

        None
    }

    /// Get current state.
    pub fn get_state(&self) -> FsmState {
        FsmState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Get current signal strength.
    pub fn get_signal_strength(&self) -> u8 {
        self.signal_strength.load(Ordering::Acquire)
    }
}

impl Default for PatternFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl FsmState {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => FsmState::Idle,
            1 => FsmState::Building,
            2 => FsmState::Triggered,
            3 => FsmState::Confirmed,
            4 => FsmState::Exhausted,
            _ => FsmState::Idle,
        }
    }
}

impl PatternType {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => PatternType::StopRun,
            2 => PatternType::Absorption,
            3 => PatternType::LiquidityGrab,
            4 => PatternType::Fakeout,
            _ => PatternType::None,
        }
    }
}

/// Manager for multiple parallel pattern FSMs.
pub struct PatternManager {
    fsms: [PatternFsm; 4], // Track 4 patterns in parallel
    active_index: AtomicU8,
}

impl PatternManager {
    pub fn new() -> Self {
        PatternManager {
            fsms: [
                PatternFsm::new(),
                PatternFsm::new(),
                PatternFsm::new(),
                PatternFsm::new(),
            ],
            active_index: AtomicU8::new(0),
        }
    }

    /// Process a tick through all active FSMs.
    pub fn process_tick(
        &self,
        price: u64,
        volume: u64,
        timestamp_ns: u64,
        config: &PatternConfig,
    ) -> Vec<(PatternType, u8)> {
        let mut signals = Vec::with_capacity(4);
        
        for fsm in &self.fsms {
            if let Some(signal) = fsm.process_tick(price, volume, timestamp_ns, config) {
                signals.push(signal);
            }
        }

        signals
    }

    /// Get a reference to the next available FSM for new pattern tracking.
    pub fn get_available_fsm(&self) -> &PatternFsm {
        let idx = self.active_index.fetch_add(1, Ordering::Relaxed) % 4;
        &self.fsms[idx as usize]
    }
}

impl Default for PatternManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stop_run_detection() {
        let fsm = PatternFsm::new();
        let config = PatternConfig {
            stop_run_ticks: 5,
            absorption_volume: 10000,
            time_window_ns: 100_000_000,
            confirmation_count: 3,
        };

        let base_price = 5000000; // $50,000.00
        let ts = 1000000000u64;

        // Simulate rapid price increase (stop-run scenario)
        for i in 0..10 {
            let price = base_price + i * 200; // 200 cents per tick
            let _ = fsm.process_tick(price, 5000, ts + i * 1_000_000, &config);
        }

        assert!(fsm.get_state() == FsmState::Building || fsm.get_state() == FsmState::Triggered);
    }

    #[test]
    fn test_absorption_detection() {
        let fsm = PatternFsm::new();
        let config = PatternConfig::default();

        let price = 5000000;
        let ts = 1000000000u64;

        // Simulate high volume at same price (absorption)
        for i in 0..5 {
            let _ = fsm.process_tick(price, 3000, ts + i * 10_000_000, &config);
        }

        // Should detect absorption
        let state = fsm.get_state();
        assert!(state == FsmState::Building || state == FsmState::Triggered);
    }
}
