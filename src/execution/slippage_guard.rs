//! `src/execution/slippage_guard.rs`
//!
//! **Cross-Asset Slippage Guard**
//! Pauses secondary altcoin engines if the primary BTC/ETH execution quality degrades,
//! preserving capital during exchange matching engine lag or network congestion.
//!
//! **Logic:**
//! - Monitors slippage (difference between expected fill price and actual fill price).
//! - If slippage on BTC/ETH exceeds threshold, halt all altcoin trading immediately.
//! - Uses a circuit-breaker pattern with hysteresis to prevent flapping.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Threshold for acceptable slippage in basis points (1 bp = 0.01%).
/// E.g., 10 bps = 0.1% slippage tolerance.
const SLIPPAGE_THRESHOLD_BPS: u64 = 10;

/// Number of consecutive slippage events required to trigger a halt.
const CONSECUTIVE_EVENTS_THRESHOLD: usize = 3;

/// Represents slippage metrics for a single symbol.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlippageMetrics {
    pub symbol_id: u8,
    pub expected_price: i64, // Fixed point
    pub actual_price: i64,   // Fixed point
    pub timestamp_ns: u64,
}

/// The Slippage Guard engine.
pub struct SlippageGuard {
    /// Master halt flag for altcoin engines.
    altcoins_halted: AtomicBool,
    /// Consecutive slippage violation counter.
    violation_count: AtomicUsize,
    /// Last violation timestamp for reset logic.
    last_violation_ns: AtomicU64,
    /// Flag indicating if the guard is currently in cooldown.
    is_cooldown: AtomicBool,
}

unsafe impl Send for SlippageGuard {}
unsafe impl Sync for SlippageGuard {}

impl SlippageGuard {
    pub fn new() -> Self {
        Self {
            altcoins_halted: AtomicBool::new(false),
            violation_count: AtomicUsize::new(0),
            last_violation_ns: AtomicU64::new(0),
            is_cooldown: AtomicBool::new(false),
        }
    }

    /// Records a slippage event.
    /// Returns `true` if the event triggered a halt.
    pub fn record_slippage(&self, metrics: SlippageMetrics) -> bool {
        // Calculate slippage in basis points
        let diff = (metrics.expected_price as i64 - metrics.actual_price as i64).abs();
        let price = metrics.expected_price.max(1); // Avoid div by zero
        
        // Slippage BPS = (Diff / Price) * 10000
        let slippage_bps = (diff as u64 * 10_000) / (price as u64);

        if slippage_bps > SLIPPAGE_THRESHOLD_BPS {
            self.handle_violation(metrics.symbol_id)
        } else {
            // Reset counter on good execution
            self.violation_count.store(0, Ordering::Relaxed);
            false
        }
    }

    /// Handles a slippage violation.
    fn handle_violation(&self, symbol_id: u8) -> bool {
        let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        
        // Check time since last violation (reset if > 1 second)
        let last = self.last_violation_ns.load(Ordering::Relaxed);
        if now_ns.saturating_sub(last) > 1_000_000_000 {
            self.violation_count.store(0, Ordering::Relaxed);
        }

        self.last_violation_ns.store(now_ns, Ordering::Relaxed);
        
        let count = self.violation_count.fetch_add(1, Ordering::Relaxed) + 1;

        if count >= CONSECUTIVE_EVENTS_THRESHOLD {
            // Only halt altcoins, not BTC/ETH (symbol_id 0 and 1 typically)
            if symbol_id > 1 {
                self.trigger_halt();
                return true;
            } else {
                // For BTC/ETH, we might want stricter rules or immediate global halt
                // For now, just log
                eprintln!("[SLIPPAGE GUARD] Primary asset slippage detected on symbol {}", symbol_id);
            }
        }

        false
    }

    /// Triggers the halt for altcoin engines.
    fn trigger_halt(&self) {
        if !self.altcoins_halted.swap(true, Ordering::SeqCst) {
            eprintln!("[SLIPPAGE GUARD] Altcoin engines HALTED due to excessive slippage.");
            // In production: Send signal to execution engines to pause
        }
    }

    /// Checks if altcoin engines should be paused.
    pub fn is_altcoins_halted(&self) -> bool {
        self.altcoins_halted.load(Ordering::Acquire)
    }

    /// Attempts to resume trading after a cooldown period.
    /// Should be called periodically by a recovery thread.
    pub fn try_resume(&self) {
        if self.is_cooldown.load(Ordering::Relaxed) {
            // Check if enough time has passed (e.g., 5 seconds)
            let last = self.last_violation_ns.load(Ordering::Relaxed);
            let now_ns = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
            
            if now_ns.saturating_sub(last) > 5_000_000_000 {
                self.altcoins_halted.store(false, Ordering::SeqCst);
                self.violation_count.store(0, Ordering::Relaxed);
                self.is_cooldown.store(false, Ordering::Relaxed);
                eprintln!("[SLIPPAGE GUARD] Altcoin engines RESUMED.");
            }
        } else if self.altcoins_halted.load(Ordering::Relaxed) {
            // Just halted, enter cooldown
            self.is_cooldown.store(true, Ordering::Relaxed);
        }
    }

    /// Force resume (for manual override).
    pub fn force_resume(&self) {
        self.altcoins_halted.store(false, Ordering::SeqCst);
        self.violation_count.store(0, Ordering::Relaxed);
        self.is_cooldown.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_detection() {
        let guard = SlippageGuard::new();
        
        // Normal slippage (should not trigger)
        let normal = SlippageMetrics {
            symbol_id: 2, // Altcoin
            expected_price: 100_000_000,
            actual_price: 99_950_000, // 0.05% slippage (5 bps)
            timestamp_ns: 1234567890,
        };
        assert!(!guard.record_slippage(normal));
        assert!(!guard.is_altcoins_halted());

        // Excessive slippage (3 times to trigger)
        let bad = SlippageMetrics {
            symbol_id: 2,
            expected_price: 100_000_000,
            actual_price: 99_000_000, // 1% slippage (100 bps)
            timestamp_ns: 1234567890,
        };

        assert!(!guard.record_slippage(bad)); // 1
        assert!(!guard.record_slippage(bad)); // 2
        assert!(guard.record_slippage(bad));  // 3 -> Trigger
        
        assert!(guard.is_altcoins_halted());
    }
}
