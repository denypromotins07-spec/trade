//! Branch Prediction Optimizer - PGO and Branch Hints
//! 
//! This module applies advanced branch prediction hinting using `likely` and
//! `unlikely` compiler macros, combined with Profile-Guided Optimization (PGO)
//! support for the main event loop. It helps the CPU's branch predictor make
//! correct predictions more often, reducing pipeline stalls.
//! 
//! **Key Features:**
//! - likely/unlikely macros for branch hints
//! - PGO instrumentation support
//! - Branch statistics collection
//! - Hot/cold path separation

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Branch hint: mark condition as likely to be true.
#[inline]
pub fn likely(b: bool) -> bool {
    #[cfg(feature = "nightly")]
    {
        std::intrinsics::likely(b)
    }
    #[cfg(not(feature = "nightly"))]
    {
        b
    }
}

/// Branch hint: mark condition as unlikely to be true.
#[inline]
pub fn unlikely(b: bool) -> bool {
    #[cfg(feature = "nightly")]
    {
        std::intrinsics::unlikely(b)
    }
    #[cfg(not(feature = "nightly"))]
    {
        b
    }
}

/// Branch statistics collector for profiling.
pub struct BranchStats {
    /// Total times branch was executed
    total_count: AtomicU64,
    /// Times branch was taken
    taken_count: AtomicU64,
    /// Branch name for identification
    name: &'static str,
}

impl BranchStats {
    /// Create new branch stats collector.
    pub const fn new(name: &'static str) -> Self {
        BranchStats {
            total_count: AtomicU64::new(0),
            taken_count: AtomicU64::new(0),
            name,
        }
    }

    /// Record a branch execution.
    #[inline]
    pub fn record(&self, taken: bool) {
        self.total_count.fetch_add(1, Ordering::Relaxed);
        if taken {
            self.taken_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get the taken rate (0.0 - 1.0).
    pub fn taken_rate(&self) -> f64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let taken = self.taken_count.load(Ordering::Relaxed);
        taken as f64 / total as f64
    }

    /// Check if branch is predictable (>90% or <10% taken rate).
    pub fn is_predictable(&self) -> bool {
        let rate = self.taken_rate();
        rate > 0.9 || rate < 0.1
    }

    /// Reset statistics.
    pub fn reset(&self) {
        self.total_count.store(0, Ordering::Relaxed);
        self.taken_count.store(0, Ordering::Relaxed);
    }

    /// Get branch name.
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// PGO (Profile-Guided Optimization) instrumentation helper.
pub struct PgoInstrumentation {
    /// Enable/disable instrumentation
    enabled: AtomicBool,
    /// Instrumentation counter
    counter: AtomicU64,
}

impl PgoInstrumentation {
    pub const fn new() -> Self {
        PgoInstrumentation {
            enabled: AtomicBool::new(false),
            counter: AtomicU64::new(0),
        }
    }

    /// Enable instrumentation for PGO profiling run.
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
    }

    /// Disable instrumentation for production runs.
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Record an instrumentation point.
    #[inline]
    pub fn record(&self) {
        if self.enabled.load(Ordering::Relaxed) {
            self.counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get counter value for PGO data export.
    pub fn get_counter(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Export PGO data (would write to file in production).
    pub fn export_pgo_data(&self) -> Vec<u8> {
        // In production, this would serialize counters to a .profraw file
        // for LLVM's pgo-use to consume
        self.counter.load(Ordering::Relaxed).to_le_bytes().to_vec()
    }
}

impl Default for PgoInstrumentation {
    fn default() -> Self {
        Self::new()
    }
}

/// Event type for the trading system event loop.
#[derive(Debug, Clone, Copy)]
pub enum EventType {
    Tick,
    OrderUpdate,
    Cancel,
    Fill,
    Signal,
    RiskCheck,
}

/// Optimized event loop with branch prediction hints.
pub struct OptimizedEventLoop {
    /// PGO instrumentation
    pgo: PgoInstrumentation,
    /// Branch stats for key decision points
    tick_branch_stats: BranchStats,
    order_branch_stats: BranchStats,
    risk_check_stats: BranchStats,
    /// Event counter
    events_processed: AtomicU64,
}

impl OptimizedEventLoop {
    pub const fn new() -> Self {
        OptimizedEventLoop {
            pgo: PgoInstrumentation::new(),
            tick_branch_stats: BranchStats::new("tick_branch"),
            order_branch_stats: BranchStats::new("order_branch"),
            risk_check_stats: BranchStats::new("risk_check"),
            events_processed: AtomicU64::new(0),
        }
    }

    /// Process an event with optimized branching.
    #[inline]
    pub fn process_event(&self, event_type: EventType, data: u64) -> bool {
        // PGO instrumentation point
        self.pgo.record();

        match event_type {
            EventType::Tick => {
                // Ticks are the most common event - mark as likely
                if likely(true) {
                    self.tick_branch_stats.record(true);
                    self.process_tick(data);
                }
            }
            EventType::OrderUpdate => {
                // Order updates are less common
                if unlikely(false) {
                    self.order_branch_stats.record(true);
                    self.process_order_update(data);
                } else {
                    self.order_branch_stats.record(false);
                }
            }
            EventType::RiskCheck => {
                // Risk checks should rarely fail - mark failure as unlikely
                let passed = self.check_risk(data);
                if unlikely(!passed) {
                    self.risk_check_stats.record(true); // Risk check failed
                    return false;
                }
                self.risk_check_stats.record(false);
            }
            _ => {
                // Other events
                self.process_other(event_type, data);
            }
        }

        self.events_processed.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Process tick event (hot path).
    #[inline(always)]
    fn process_tick(&self, _data: u64) {
        // Hot path - keep minimal
        // In production: update order book, check signals, etc.
    }

    /// Process order update (cold path).
    #[inline(never)]
    fn process_order_update(&self, _data: u64) {
        // Cold path - can be larger function
        // In production: update positions, recalculate risk, etc.
    }

    /// Check risk limits (should usually pass).
    #[inline]
    fn check_risk(&self, _data: u64) -> bool {
        // Risk check - should almost always pass
        // Mark failure path as unlikely at call site
        true
    }

    /// Process other event types.
    #[inline(never)]
    fn process_other(&self, _event_type: EventType, _data: u64) {
        // Cold path for rare events
    }

    /// Get branch prediction statistics.
    pub fn get_branch_stats(&self) -> Vec<(&str, f64)> {
        vec![
            (self.tick_branch_stats.name(), self.tick_branch_stats.taken_rate()),
            (self.order_branch_stats.name(), self.order_branch_stats.taken_rate()),
            (self.risk_check_stats.name(), self.risk_check_stats.taken_rate()),
        ]
    }

    /// Get total events processed.
    pub fn get_events_processed(&self) -> u64 {
        self.events_processed.load(Ordering::Relaxed)
    }

    /// Enable PGO instrumentation.
    pub fn enable_pgo(&self) {
        self.pgo.enable();
    }

    /// Disable PGO instrumentation.
    pub fn disable_pgo(&self) {
        self.pgo.disable();
    }

    /// Export PGO data.
    pub fn export_pgo(&self) -> Vec<u8> {
        self.pgo.export_pgo_data()
    }
}

impl Default for OptimizedEventLoop {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro for marking cold code paths.
#[macro_export]
macro_rules! cold_path {
    ($($tt:tt)*) => {{
        #[cold]
        #[inline(never)]
        fn cold_code() {
            $($tt)*
        }
        cold_code();
    }};
}

/// Macro for branch prediction with statistics.
#[macro_export]
macro_rules! likely_with_stats {
    ($cond:expr, $stats:expr) => {{
        let result = $cond;
        $stats.record(result);
        if likely(result) {
            true
        } else {
            false
        }
    }};
}

#[macro_export]
macro_rules! unlikely_with_stats {
    ($cond:expr, $stats:expr) => {{
        let result = $cond;
        $stats.record(result);
        if unlikely(result) {
            true
        } else {
            false
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_stats() {
        let stats = BranchStats::new("test");
        
        // Simulate 90% taken
        for _ in 0..90 {
            stats.record(true);
        }
        for _ in 0..10 {
            stats.record(false);
        }
        
        assert!((stats.taken_rate() - 0.9).abs() < 0.01);
        assert!(stats.is_predictable());
    }

    #[test]
    fn test_unpredictable_branch() {
        let stats = BranchStats::new("unpredictable");
        
        // Simulate 50/50 branch
        for i in 0..100 {
            stats.record(i % 2 == 0);
        }
        
        assert!((stats.taken_rate() - 0.5).abs() < 0.01);
        assert!(!stats.is_predictable());
    }

    #[test]
    fn test_pgo_instrumentation() {
        let pgo = PgoInstrumentation::new();
        
        assert!(!pgo.enabled.load(Ordering::Relaxed));
        
        pgo.enable();
        for _ in 0..100 {
            pgo.record();
        }
        
        assert_eq!(pgo.get_counter(), 100);
        
        let data = pgo.export_pgo_data();
        assert_eq!(data.len(), 8); // u64 in little-endian
    }

    #[test]
    fn test_optimized_event_loop() {
        let loop_ = OptimizedEventLoop::new();
        
        // Process mostly ticks (common case)
        for _ in 0..100 {
            loop_.process_event(EventType::Tick, 0);
        }
        
        // Process few order updates (rare)
        for _ in 0..5 {
            loop_.process_event(EventType::OrderUpdate, 0);
        }
        
        assert_eq!(loop_.get_events_processed(), 105);
        
        let stats = loop_.get_branch_stats();
        assert_eq!(stats.len(), 3);
    }

    #[test]
    fn test_likely_unlikely() {
        assert!(likely(true));
        assert!(!likely(false));
        assert!(unlikely(true));
        assert!(!unlikely(false));
    }
}
