//! Guerrilla Execution Algorithm for Liquidity Taking
//!
//! Implements aggressive liquidity taking in small slices while hiding
//! the parent order's true intent from HFT sniffers. Uses randomized
//! timing and sizing to avoid detection patterns.
//!
//! # Key Features
//! - Randomized child order sizes (guerrilla tactics)
//! - Time-varying execution tempo
//! - Multi-venue order splitting
//! - Anti-sniffing obfuscation

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Maximum child orders per parent (bounded for memory)
const MAX_CHILD_ORDERS: usize = 1024;

/// Global tracker for guerrilla executions
static GUERRILLA_EXEC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Guerrilla execution state
#[repr(C, align(64))]
pub struct GuerrillaExecutor {
    /// Parent order ID
    parent_id: u64,
    /// Total quantity to execute
    total_quantity: f64,
    /// Remaining quantity
    remaining_qty: f64,
    /// Executed quantity
    executed_qty: f64,
    /// Child order counter
    child_count: usize,
    /// Maximum child order size
    max_child_size: f64,
    /// Minimum child order size
    min_child_size: f64,
    /// Randomization seed for deterministic replay
    seed: u64,
    /// Execution start time
    start_time: Instant,
    /// Target completion time (nanoseconds)
    target_duration_ns: u64,
    /// Active flag
    is_active: CachePaddedAtomicBool,
    /// Last execution time
    last_exec_time: Instant,
    /// Minimum interval between child orders (nanoseconds)
    min_interval_ns: u64,
    /// Venue rotation index
    venue_index: usize,
    /// Number of venues
    num_venues: usize,
}

/// Cache-line padded atomic bool for lock-free flags
#[repr(C, align(64))]
struct CachePaddedAtomicBool {
    value: AtomicBool,
    _padding: [u8; 63],
}

impl CachePaddedAtomicBool {
    fn new(val: bool) -> Self {
        Self {
            value: AtomicBool::new(val),
            _padding: [0u8; 63],
        }
    }
    
    #[inline]
    fn load(&self) -> bool {
        self.value.load(Ordering::Acquire)
    }
    
    #[inline]
    fn store(&self, val: bool) {
        self.value.store(val, Ordering::Release);
    }
}

impl GuerrillaExecutor {
    /// Create a new guerrilla executor
    #[inline]
    pub fn new(
        parent_id: u64,
        total_quantity: f64,
        max_child_size: f64,
        min_child_size: f64,
        target_duration_ms: u64,
        seed: u64,
    ) -> Self {
        GUERRILLA_EXEC_COUNT.fetch_add(1, Ordering::Relaxed);
        
        Self {
            parent_id,
            total_quantity,
            remaining_qty: total_quantity,
            executed_qty: 0.0,
            child_count: 0,
            max_child_size,
            min_child_size,
            seed,
            start_time: Instant::now(),
            target_duration_ns: target_duration_ms * 1_000_000,
            is_active: CachePaddedAtomicBool::new(true),
            last_exec_time: Instant::now(),
            min_interval_ns: 100_000, // 100 microseconds minimum
            venue_index: 0,
            num_venues: 3, // Default: 3 venues for splitting
        }
    }
    
    /// Check if ready to send next child order
    #[inline]
    pub fn should_execute(&self) -> bool {
        if !self.is_active.load() || self.remaining_qty <= 0.0 {
            return false;
        }
        
        let elapsed = self.last_exec_time.elapsed().as_nanos() as u64;
        elapsed >= self.min_interval_ns
    }
    
    /// Generate next child order size using pseudo-random guerrilla pattern
    #[inline]
    pub fn next_child_size(&mut self) -> Option<f64> {
        if !self.is_active.load() || self.remaining_qty <= 0.0 {
            return None;
        }
        
        // Check tempo constraints
        if !self.should_execute() {
            return None;
        }
        
        // Check if we've exceeded target duration
        let elapsed_ns = self.start_time.elapsed().as_nanos() as u64;
        if elapsed_ns > self.target_duration_ns {
            // Rush to complete - use larger sizes
            let rush_size = self.remaining_qty.min(self.max_child_size);
            self.execute_child(rush_size);
            return Some(rush_size);
        }
        
        // Guerrilla tactic: randomize size within bounds
        let size = self.randomize_child_size();
        
        // Ensure we don't exceed remaining
        let actual_size = size.min(self.remaining_qty);
        
        self.execute_child(actual_size);
        Some(actual_size)
    }
    
    /// Execute a child order (updates internal state)
    #[inline]
    fn execute_child(&mut self, size: f64) {
        self.executed_qty += size;
        self.remaining_qty -= size;
        self.child_count += 1;
        self.last_exec_time = Instant::now();
        
        // Rotate venue for splitting
        self.venue_index = (self.venue_index + 1) % self.num_venues;
        
        // Deactivate if complete
        if self.remaining_qty <= 0.0 {
            self.is_active.store(false);
        }
    }
    
    /// Randomize child order size using XORShift for speed
    #[inline]
    fn randomize_child_size(&self) -> f64 {
        // XORShift-based pseudo-random number generator
        let mut x = self.seed.wrapping_add(self.child_count as u64);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        
        // Normalize to [0, 1)
        let normalized = (x & 0xFFFF_FFFF) as f64 / u32::MAX as f64;
        
        // Map to [min_child_size, max_child_size]
        let range = self.max_child_size - self.min_child_size;
        self.min_child_size + normalized * range
    }
    
    /// Get current venue index for order routing
    #[inline]
    pub fn current_venue(&self) -> usize {
        self.venue_index
    }
    
    /// Get execution progress (0.0 to 1.0)
    #[inline]
    pub fn progress(&self) -> f64 {
        self.executed_qty / self.total_quantity
    }
    
    /// Get remaining quantity
    #[inline]
    pub fn remaining(&self) -> f64 {
        self.remaining_qty
    }
    
    /// Get executed quantity
    #[inline]
    pub fn executed(&self) -> f64 {
        self.executed_qty
    }
    
    /// Get child order count
    #[inline]
    pub fn child_count(&self) -> usize {
        self.child_count
    }
    
    /// Check if execution is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        !self.is_active.load()
    }
    
    /// Cancel remaining execution
    #[inline]
    pub fn cancel(&mut self) {
        self.is_active.store(false);
    }
    
    /// Get average child size
    #[inline]
    pub fn avg_child_size(&self) -> f64 {
        if self.child_count == 0 {
            return 0.0;
        }
        self.executed_qty / self.child_count as f64
    }
    
    /// Estimate time to completion based on current tempo
    #[inline]
    pub fn estimated_completion_ms(&self) -> u64 {
        if self.child_count == 0 || self.executed_qty <= 0.0 {
            return self.target_duration_ns / 1_000_000;
        }
        
        let elapsed_ms = self.start_time.elapsed().as_millis() as u64;
        let progress = self.progress();
        
        if progress < 0.001 {
            return self.target_duration_ns / 1_000_000;
        }
        
        (elapsed_ms as f64 / progress) as u64
    }
}

impl Drop for GuerrillaExecutor {
    fn drop(&mut self) {
        GUERRILLA_EXEC_COUNT.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Get global count of active guerrilla executions
#[inline]
pub fn active_guerrilla_count() -> usize {
    GUERRILLA_EXEC_COUNT.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_guerrilla_creation() {
        let executor = GuerrillaExecutor::new(
            12345,
            100.0,
            10.0,
            1.0,
            5000,
            0xDEADBEEF,
        );
        
        assert_eq!(executor.total_quantity, 100.0);
        assert_eq!(executor.remaining(), 100.0);
        assert!(!executor.is_complete());
    }
    
    #[test]
    fn test_guerrilla_execution() {
        let mut executor = GuerrillaExecutor::new(
            12345,
            100.0,
            10.0,
            1.0,
            5000,
            0xDEADBEEF,
        );
        
        // Execute several child orders
        let mut total_executed = 0.0;
        for _ in 0..20 {
            std::thread::sleep(Duration::from_micros(150));
            if let Some(size) = executor.next_child_size() {
                total_executed += size;
            }
        }
        
        assert!(total_executed > 0.0);
        assert!(executor.executed() > 0.0);
        assert!(executor.remaining() < 100.0);
    }
    
    #[test]
    fn test_guerrilla_completion() {
        let mut executor = GuerrillaExecutor::new(
            12345,
            50.0,
            10.0,
            5.0,
            1000,
            0xCAFE,
        );
        
        // Force execution until complete
        while !executor.is_complete() {
            std::thread::sleep(Duration::from_micros(150));
            executor.next_child_size();
        }
        
        assert!(executor.is_complete());
        assert!(executor.remaining() <= 0.0);
        assert!((executor.executed() - 50.0).abs() < 0.01);
    }
}
