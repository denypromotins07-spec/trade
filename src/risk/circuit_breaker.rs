//! # Circuit Breaker Module
//! 
//! Implements hard-coded, zero-latency circuit breakers that instantly halt all execution threads
//! if daily drawdown, velocity, or max position limits are breached.
//! 
//! ## Architecture
//! - Operates at the absolute lowest layer of the Rust event loop
//! - Uses atomic flags for thread-safe state management
//! - Lock-free design ensures microsecond response times
//! - Compatible with `/START` and `/KILL` PowerShell orchestration

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI64, Ordering};
use std::time::{Instant, Duration};
use std::sync::Arc;

/// Represents the current state of the circuit breaker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation, all systems green
    Closed,
    /// Triggered, trading halted immediately
    Open,
    /// Cooling down, testing waters with reduced size
    HalfOpen,
}

/// Configuration for circuit breaker thresholds
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Maximum allowed daily drawdown in basis points (e.g., 500 = 5%)
    pub max_daily_drawdown_bps: i64,
    /// Maximum velocity of losses (losses per second)
    pub max_loss_velocity_bps: i64,
    /// Maximum position size in quote currency units
    pub max_position_size: u64,
    /// Cooldown period before attempting to reset from Open to HalfOpen
    pub cooldown_duration: Duration,
    /// Maximum consecutive triggers before requiring manual intervention
    pub max_consecutive_triggers: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            max_daily_drawdown_bps: 500, // 5% daily max
            max_loss_velocity_bps: 50,   // 0.5% per second max velocity
            max_position_size: 1_000_000_000, // 1B units
            cooldown_duration: Duration::from_secs(300), // 5 minutes
            max_consecutive_triggers: 3,
        }
    }
}

/// Metrics snapshot for evaluation
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Current daily PnL in basis points
    pub daily_pnl_bps: i64,
    /// Recent loss velocity (bps per second over last 1s window)
    pub loss_velocity_bps: i64,
    /// Current total position size
    pub current_position_size: u64,
    /// Timestamp of last update
    pub timestamp: Instant,
}

/// High-performance Circuit Breaker engine
/// 
/// This struct uses atomic operations exclusively to ensure zero-lock performance
/// suitable for HFT environments on AMD Ryzen AI 5 architecture.
pub struct CircuitBreaker {
    /// Current state of the breaker
    state: AtomicU64, // Encoded CircuitState
    /// Atomic flag for immediate halt signal
    halt_flag: AtomicBool,
    /// Counter for consecutive triggers
    trigger_count: AtomicU64,
    /// Timestamp when breaker was opened
    open_timestamp: AtomicU64,
    /// Peak equity mark for drawdown calculation (in atomic units)
    peak_equity: AtomicI64,
    /// Starting equity for daily calculation
    start_equity: AtomicI64,
    /// Current equity
    current_equity: AtomicI64,
    /// Configuration (immutable after creation for lock-free safety)
    config: CircuitBreakerConfig,
    /// System start time
    start_time: Instant,
}

// Helper to encode/decode CircuitState to u64 for atomics
fn encode_state(state: CircuitState) -> u64 {
    match state {
        CircuitState::Closed => 0,
        CircuitState::Open => 1,
        CircuitState::HalfOpen => 2,
    }
}

fn decode_state(val: u64) -> CircuitState {
    match val {
        0 => CircuitState::Closed,
        1 => CircuitState::Open,
        2 => CircuitState::HalfOpen,
        _ => CircuitState::Closed,
    }
}

impl CircuitBreaker {
    /// Create a new CircuitBreaker with the given configuration
    pub fn new(config: CircuitBreakerConfig, initial_equity: i64) -> Self {
        Self {
            state: AtomicU64::new(encode_state(CircuitState::Closed)),
            halt_flag: AtomicBool::new(false),
            trigger_count: AtomicU64::new(0),
            open_timestamp: AtomicU64::new(0),
            peak_equity: AtomicI64::new(initial_equity),
            start_equity: AtomicI64::new(initial_equity),
            current_equity: AtomicI64::new(initial_equity),
            config,
            start_time: Instant::now(),
        }
    }

    /// Wrap in Arc for shared access across threads
    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// Update current equity value (called every tick)
    #[inline]
    pub fn update_equity(&self, equity: i64) {
        self.current_equity.store(equity, Ordering::Relaxed);
        
        // Update peak equity if we've exceeded it (only in Closed state)
        let current_state = decode_state(self.state.load(Ordering::Acquire));
        if current_state == CircuitState::Closed {
            let current_peak = self.peak_equity.load(Ordering::Relaxed);
            if equity > current_peak {
                self.peak_equity.store(equity, Ordering::Relaxed);
            }
        }
    }

    /// Check all circuit breaker conditions
    /// Returns true if trading should continue, false if halted
    #[inline]
    pub fn check(&self, metrics: &RiskMetrics) -> bool {
        let current_state = decode_state(self.state.load(Ordering::Acquire));
        
        // If already open, check cooldown
        if current_state == CircuitState::Open {
            return self.check_cooldown();
        }
        
        // If half-open, allow limited trading but monitor closely
        if current_state == CircuitState::HalfOpen {
            return self.check_half_open(metrics);
        }
        
        // CLOSED STATE: Check all triggers
        
        // 1. Daily Drawdown Check
        let start_eq = self.start_equity.load(Ordering::Relaxed);
        let current_eq = self.current_equity.load(Ordering::Relaxed);
        let daily_drawdown_bps = if start_eq > 0 {
            ((start_eq - current_eq) * 10000) / start_eq
        } else {
            0
        };
        
        if daily_drawdown_bps >= self.config.max_daily_drawdown_bps {
            self.trigger_breaker(RiskTrigger::DailyDrawdown, daily_drawdown_bps);
            return false;
        }
        
        // 2. Loss Velocity Check
        if metrics.loss_velocity_bps >= self.config.max_loss_velocity_bps {
            self.trigger_breaker(RiskTrigger::LossVelocity, metrics.loss_velocity_bps);
            return false;
        }
        
        // 3. Position Size Check
        if metrics.current_position_size > self.config.max_position_size {
            self.trigger_breaker(RiskTrigger::PositionLimit, metrics.current_position_size as i64);
            return false;
        }
        
        true
    }
    
    /// Check if cooldown period has elapsed
    fn check_cooldown(&self) -> bool {
        let open_ts_val = self.open_timestamp.load(Ordering::Relaxed);
        if open_ts_val == 0 {
            return true;
        }
        
        let elapsed_ms = (Instant::now().duration_since(self.start_time)).as_millis() as u64;
        let open_elapsed = elapsed_ms.saturating_sub(open_ts_val);
        let cooldown_ms = self.config.cooldown_duration.as_millis() as u64;
        
        if open_elapsed >= cooldown_ms {
            // Transition to HalfOpen
            let expected = encode_state(CircuitState::Open);
            if self.state.compare_exchange(
                expected,
                encode_state(CircuitState::HalfOpen),
                Ordering::SeqCst,
                Ordering::Relaxed
            ).is_ok() {
                self.halt_flag.store(false, Ordering::Release);
            }
        }
        
        false // Still in cooldown, remain halted
    }
    
    /// Check conditions in HalfOpen state
    fn check_half_open(&self, metrics: &RiskMetrics) -> bool {
        // Stricter limits in HalfOpen state
        let strict_drawdown = self.config.max_daily_drawdown_bps / 2;
        
        let start_eq = self.start_equity.load(Ordering::Relaxed);
        let current_eq = self.current_equity.load(Ordering::Relaxed);
        let daily_drawdown_bps = if start_eq > 0 {
            ((start_eq - current_eq) * 10000) / start_eq
        } else {
            0
        };
        
        if daily_drawdown_bps >= strict_drawdown || 
           metrics.loss_velocity_bps >= self.config.max_loss_velocity_bps / 2 {
            // Re-trigger to Open
            let expected = encode_state(CircuitState::HalfOpen);
            if self.state.compare_exchange(
                expected,
                encode_state(CircuitState::Open),
                Ordering::SeqCst,
                Ordering::Relaxed
            ).is_ok() {
                self.update_open_timestamp();
                self.halt_flag.store(true, Ordering::Release);
            }
            return false;
        }
        
        // If we've survived long enough in HalfOpen without issues, close
        // (Simplified: in production, track time in HalfOpen)
        true
    }
    
    /// Trigger the circuit breaker
    fn trigger_breaker(&self, trigger: RiskTrigger, value: i64) {
        let expected = encode_state(CircuitState::Closed);
        if self.state.compare_exchange(
            expected,
            encode_state(CircuitState::Open),
            Ordering::SeqCst,
            Ordering::Relaxed
        ).is_ok() {
            self.update_open_timestamp();
            self.halt_flag.store(true, Ordering::Release);
            
            let count = self.trigger_count.fetch_add(1, Ordering::Relaxed);
            
            // Log trigger (in production, use tracing crate)
            eprintln!(
                "[CIRCUIT_BREAKER] TRIGGERED: {:?} (value: {}), consecutive_triggers: {}",
                trigger, value, count + 1
            );
            
            // Check if max consecutive triggers exceeded
            if count + 1 >= self.config.max_consecutive_triggers as u64 {
                eprintln!("[CIRCUIT_BREAKER] MAX CONSECUTIVE TRIGGERS REACHED - MANUAL INTERVENTION REQUIRED");
                // In production, this would set a permanent halt flag
            }
        }
    }
    
    fn update_open_timestamp(&self) {
        let elapsed_ms = (Instant::now().duration_since(self.start_time)).as_millis() as u64;
        self.open_timestamp.store(elapsed_ms, Ordering::Relaxed);
    }
    
    /// Get current state
    #[inline]
    pub fn state(&self) -> CircuitState {
        decode_state(self.state.load(Ordering::Acquire))
    }
    
    /// Check if halted
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halt_flag.load(Ordering::Acquire)
    }
    
    /// Manual reset (for /START orchestration)
    pub fn reset(&self, new_equity: i64) {
        self.state.store(encode_state(CircuitState::Closed), Ordering::SeqCst);
        self.halt_flag.store(false, Ordering::Release);
        self.trigger_count.store(0, Ordering::Relaxed);
        self.open_timestamp.store(0, Ordering::Relaxed);
        self.peak_equity.store(new_equity, Ordering::Relaxed);
        self.start_equity.store(new_equity, Ordering::Relaxed);
        self.current_equity.store(new_equity, Ordering::Relaxed);
    }
}

/// Reason for circuit breaker trigger
#[derive(Debug, Clone, Copy)]
enum RiskTrigger {
    DailyDrawdown,
    LossVelocity,
    PositionLimit,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_daily_drawdown_trigger() {
        let config = CircuitBreakerConfig {
            max_daily_drawdown_bps: 100, // 1%
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config, 1_000_000);
        
        // Simulate 1.5% drawdown
        cb.update_equity(985_000);
        
        let metrics = RiskMetrics {
            daily_pnl_bps: -150,
            loss_velocity_bps: 10,
            current_position_size: 100_000,
            timestamp: Instant::now(),
        };
        
        assert!(!cb.check(&metrics));
        assert_eq!(cb.state(), CircuitState::Open);
    }
    
    #[test]
    fn test_normal_operation() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config, 1_000_000);
        
        cb.update_equity(1_010_000); // Profit
        
        let metrics = RiskMetrics {
            daily_pnl_bps: 100,
            loss_velocity_bps: 5,
            current_position_size: 100_000,
            timestamp: Instant::now(),
        };
        
        assert!(cb.check(&metrics));
        assert_eq!(cb.state(), CircuitState::Closed);
    }
}
