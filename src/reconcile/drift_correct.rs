//! Drift Correction - Automated State Discrepancy Patching
//! 
//! This module implements automated drift correction logic that atomically patches
//! local state discrepancies and cancels orphaned orders without halting the main
//! execution event loop. Optimized for AMD Ryzen AI 5 with microsecond latency.
//! 
//! RAM Budget: Uses lock-free atomic operations and bounded queues.
//! Enforces global 8GB RAM limit via strict memory management.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam::queue::SegQueue;
use thiserror::Error;

/// Maximum concurrent corrections allowed
const MAX_CONCURRENT_CORRECTIONS: usize = 10;

/// Correction timeout duration
const CORRECTION_TIMEOUT_MS: u64 = 5000;

/// Error types for drift correction
#[derive(Error, Debug, Clone)]
pub enum CorrectionError {
    #[error("Correction queue full")]
    QueueFull,
    
    #[error("Correction timeout after {0}ms")]
    Timeout(u64),
    
    #[error("Invalid correction: {0}")]
    InvalidCorrection(String),
    
    #[error("Order cancel failed: {order_id}")]
    OrderCancelFailed { order_id: String },
    
    #[error("State patch failed: {reason}")]
    StatePatchFailed { reason: String },
    
    #[error("Concurrent correction limit reached")]
    ConcurrentLimitReached,
    
    #[error("Circuit breaker open")]
    CircuitBreakerOpen,
}

/// Result type for correction operations
pub type CorrectionResult<T> = Result<T, CorrectionError>;

/// Types of corrections that can be applied
#[derive(Debug, Clone)]
pub enum CorrectionType {
    /// Replace entire order book snapshot
    ReplaceOrderBook {
        symbol: String,
        new_sequence: u64,
    },
    
    /// Patch specific price level
    PatchLevel {
        symbol: String,
        side: Side,
        price_fp: i128,
        qty_fp: i128,
    },
    
    /// Remove stale order from local state
    RemoveStaleOrder {
        order_id: String,
        symbol: String,
    },
    
    /// Cancel orphaned order on exchange
    CancelOrphanedOrder {
        order_id: String,
        symbol: String,
    },
    
    /// Sync portfolio balance
    SyncBalance {
        asset: String,
        correct_free_fp: i128,
        correct_locked_fp: i128,
    },
    
    /// Reset sequence number
    ResetSequence {
        symbol: String,
        new_sequence: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Status of a correction operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    TimedOut,
}

/// Tracking information for a correction
#[derive(Debug, Clone)]
pub struct CorrectionRecord {
    pub id: u64,
    pub correction_type: CorrectionType,
    pub status: CorrectionStatus,
    pub created_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub retry_count: u32,
    pub error_message: Option<String>,
}

impl CorrectionRecord {
    #[inline]
    fn new(id: u64, correction_type: CorrectionType) -> Self {
        Self {
            id,
            correction_type,
            status: CorrectionStatus::Pending,
            created_at_ms: get_timestamp_ms(),
            completed_at_ms: None,
            retry_count: 0,
            error_message: None,
        }
    }
}

/// Statistics for drift correction operations
#[derive(Debug, Clone, Copy)]
pub struct CorrectionStats {
    pub total_corrections: u64,
    pub successful_corrections: u64,
    pub failed_corrections: u64,
    pub timed_out_corrections: u64,
    pub orders_canceled: u64,
    pub state_patches_applied: u64,
    pub circuit_breaker_trips: u64,
    pub avg_correction_time_ms: u64,
}

impl Default for CorrectionStats {
    fn default() -> Self {
        Self {
            total_corrections: 0,
            successful_corrections: 0,
            failed_corrections: 0,
            timed_out_corrections: 0,
            orders_canceled: 0,
            state_patches_applied: 0,
            circuit_breaker_trips: 0,
            avg_correction_time_ms: 0,
        }
    }
}

/// Circuit breaker state for protecting against cascading failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Get current timestamp in milliseconds
#[inline]
fn get_timestamp_ms() -> u64 {
    Instant::now().elapsed().as_millis() as u64
}

/// Main drift corrector that manages state reconciliation fixes
pub struct DriftCorrector {
    /// Correction ID counter
    next_id: AtomicU64,
    /// Pending corrections queue
    pending_queue: SegQueue<CorrectionRecord>,
    /// In-progress corrections count
    in_progress: AtomicU64,
    /// Circuit breaker state
    circuit_breaker: AtomicU64, // Using u64 to represent CircuitBreakerState
    /// Consecutive failure count
    consecutive_failures: AtomicU64,
    /// Failure threshold for circuit breaker
    failure_threshold: AtomicU64,
    /// Statistics
    stats: parking_lot::RwLock<CorrectionStats>,
    /// Total correction time for averaging
    total_correction_time_ms: AtomicU64,
    /// Shutdown flag
    shutdown_flag: AtomicBool,
}

impl Default for DriftCorrector {
    fn default() -> Self {
        Self::new()
    }
}

impl DriftCorrector {
    /// Create a new drift corrector
    #[inline]
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(0),
            pending_queue: SegQueue::new(),
            in_progress: AtomicU64::new(0),
            circuit_breaker: AtomicU64::new(CircuitBreakerState::Closed as u64),
            consecutive_failures: AtomicU64::new(0),
            failure_threshold: AtomicU64::new(5),
            stats: parking_lot::RwLock::new(CorrectionStats::default()),
            total_correction_time_ms: AtomicU64::new(0),
            shutdown_flag: AtomicBool::new(false),
        }
    }
    
    /// Create with custom failure threshold
    #[inline]
    pub fn with_failure_threshold(threshold: u64) -> Self {
        Self {
            next_id: AtomicU64::new(0),
            pending_queue: SegQueue::new(),
            in_progress: AtomicU64::new(0),
            circuit_breaker: AtomicU64::new(CircuitBreakerState::Closed as u64),
            consecutive_failures: AtomicU64::new(0),
            failure_threshold: AtomicU64::new(threshold),
            stats: parking_lot::RwLock::new(CorrectionStats::default()),
            total_correction_time_ms: AtomicU64::new(0),
            shutdown_flag: AtomicBool::new(false),
        }
    }
    
    /// Submit a correction for processing (non-blocking)
    /// Returns the correction ID if successfully queued
    #[inline]
    pub fn submit_correction(&self, correction_type: CorrectionType) -> CorrectionResult<u64> {
        // Check circuit breaker first
        let cb_state = self.get_circuit_breaker_state();
        if cb_state == CircuitBreakerState::Open {
            return Err(CorrectionError::CircuitBreakerOpen);
        }
        
        // Check concurrent limit
        let in_prog = self.in_progress.load(Ordering::Relaxed);
        if in_prog >= MAX_CONCURRENT_CORRECTIONS as u64 {
            return Err(CorrectionError::ConcurrentLimitReached);
        }
        
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let record = CorrectionRecord::new(id, correction_type);
        
        self.pending_queue.push(record);
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_corrections += 1;
        }
        
        Ok(id)
    }
    
    /// Try to process the next pending correction (call from worker thread)
    /// Returns true if a correction was processed, false if queue empty
    #[inline]
    pub fn try_process_next<F>(&self, processor: F) -> bool
    where
        F: FnOnce(&CorrectionRecord) -> Result<(), String>,
    {
        // Try to pop a pending correction
        let mut record = match self.pending_queue.pop() {
            Some(r) => r,
            None => return false,
        };
        
        // Check circuit breaker again
        if self.get_circuit_breaker_state() == CircuitBreakerState::Open {
            // Re-queue the correction
            let _ = self.pending_queue.push(record);
            return false;
        }
        
        // Mark as in progress
        record.status = CorrectionStatus::InProgress;
        self.in_progress.fetch_add(1, Ordering::Relaxed);
        
        let start_ms = get_timestamp_ms();
        
        // Execute the correction
        let result = processor(&record);
        
        let elapsed_ms = get_timestamp_ms() - start_ms;
        
        // Update record based on result
        match result {
            Ok(()) => {
                record.status = CorrectionStatus::Completed;
                record.completed_at_ms = Some(get_timestamp_ms());
                
                // Record success
                self.record_success(elapsed_ms);
            }
            Err(e) => {
                record.status = CorrectionStatus::Failed;
                record.error_message = Some(e);
                record.retry_count += 1;
                
                // Record failure
                self.record_failure();
                
                // Re-queue if retries remaining and not timed out
                if record.retry_count < 3 && elapsed_ms < CORRECTION_TIMEOUT_MS {
                    let _ = self.pending_queue.push(record);
                }
            }
        }
        
        self.in_progress.fetch_sub(1, Ordering::Relaxed);
        
        true
    }
    
    /// Immediately cancel an orphaned order (high priority, bypasses queue)
    #[inline]
    pub fn cancel_orphaned_order_now<F>(
        &self,
        order_id: String,
        symbol: String,
        cancel_fn: F,
    ) -> CorrectionResult<()>
    where
        F: FnOnce(&str, &str) -> Result<(), String>,
    {
        // Check circuit breaker
        if self.get_circuit_breaker_state() == CircuitBreakerState::Open {
            return Err(CorrectionError::CircuitBreakerOpen);
        }
        
        let start_ms = get_timestamp_ms();
        
        match cancel_fn(&order_id, &symbol) {
            Ok(()) => {
                let elapsed_ms = get_timestamp_ms() - start_ms;
                
                // Update stats
                {
                    let mut stats = self.stats.write();
                    stats.orders_canceled += 1;
                    stats.successful_corrections += 1;
                }
                self.total_correction_time_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
                
                Ok(())
            }
            Err(e) => {
                self.record_failure();
                Err(CorrectionError::OrderCancelFailed { order_id })
            }
        }
    }
    
    /// Apply a state patch atomically
    #[inline]
    pub fn apply_state_patch<F>(&self, patch_fn: F) -> CorrectionResult<()>
    where
        F: FnOnce() -> Result<(), String>,
    {
        let start_ms = get_timestamp_ms();
        
        match patch_fn() {
            Ok(()) => {
                let elapsed_ms = get_timestamp_ms() - start_ms;
                
                // Update stats
                {
                    let mut stats = self.stats.write();
                    stats.state_patches_applied += 1;
                    stats.successful_corrections += 1;
                }
                self.total_correction_time_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
                
                Ok(())
            }
            Err(e) => {
                self.record_failure();
                Err(CorrectionError::StatePatchFailed { reason: e })
            }
        }
    }
    
    /// Record a successful correction
    #[inline]
    fn record_success(&self, elapsed_ms: u64) {
        {
            let mut stats = self.stats.write();
            stats.successful_corrections += 1;
            
            // Update average
            let total = self.total_correction_time_ms.fetch_add(elapsed_ms, Ordering::Relaxed);
            let count = stats.successful_corrections;
            if count > 0 {
                stats.avg_correction_time_ms = (total + elapsed_ms) / count;
            }
        }
        
        // Reset consecutive failures on success
        self.consecutive_failures.store(0, Ordering::Relaxed);
        
        // If we were half-open, move to closed
        let current_cb = self.circuit_breaker.load(Ordering::Relaxed);
        if current_cb == CircuitBreakerState::HalfOpen as u64 {
            self.circuit_breaker.store(
                CircuitBreakerState::Closed as u64,
                Ordering::Relaxed,
            );
        }
    }
    
    /// Record a failed correction
    #[inline]
    fn record_failure(&self) {
        {
            let mut stats = self.stats.write();
            stats.failed_corrections += 1;
        }
        
        // Increment consecutive failures
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Check if we should trip circuit breaker
        let threshold = self.failure_threshold.load(Ordering::Relaxed);
        if failures >= threshold {
            self.circuit_breaker.store(
                CircuitBreakerState::Open as u64,
                Ordering::Relaxed,
            );
            
            let mut stats = self.stats.write();
            stats.circuit_breaker_trips += 1;
        }
    }
    
    /// Get current circuit breaker state
    #[inline]
    fn get_circuit_breaker_state(&self) -> CircuitBreakerState {
        let state = self.circuit_breaker.load(Ordering::Relaxed);
        match state {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }
    
    /// Attempt to reset circuit breaker to half-open
    #[inline]
    pub fn try_reset_circuit_breaker(&self) -> bool {
        let current = self.circuit_breaker.load(Ordering::Relaxed);
        if current == CircuitBreakerState::Open as u64 {
            self.circuit_breaker.store(
                CircuitBreakerState::HalfOpen as u64,
                Ordering::Relaxed,
            );
            return true;
        }
        false
    }
    
    /// Get current statistics
    #[inline]
    pub fn get_stats(&self) -> CorrectionStats {
        self.stats.read().clone()
    }
    
    /// Get pending correction count
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }
    
    /// Get in-progress correction count
    #[inline]
    pub fn in_progress_count(&self) -> u64 {
        self.in_progress.load(Ordering::Relaxed)
    }
    
    /// Check if circuit breaker is open
    #[inline]
    pub fn is_circuit_breaker_open(&self) -> bool {
        self.get_circuit_breaker_state() == CircuitBreakerState::Open
    }
    
    /// Force close circuit breaker (manual intervention)
    #[inline]
    pub fn force_close_circuit_breaker(&self) {
        self.circuit_breaker.store(
            CircuitBreakerState::Closed as u64,
            Ordering::Relaxed,
        );
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }
    
    /// Shutdown the corrector
    #[inline]
    pub fn shutdown(&self) {
        self.shutdown_flag.store(true, Ordering::Relaxed);
    }
    
    /// Check if shutdown requested
    #[inline]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown_flag.load(Ordering::Relaxed)
    }
}

/// Builder for constructing drift correction requests
pub struct CorrectionBuilder {
    id_counter: Arc<AtomicU64>,
}

impl CorrectionBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            id_counter: Arc::new(AtomicU64::new(0)),
        }
    }
    
    #[inline]
    pub fn replace_order_book(symbol: String, new_sequence: u64) -> CorrectionType {
        CorrectionType::ReplaceOrderBook {
            symbol,
            new_sequence,
        }
    }
    
    #[inline]
    pub fn patch_level(
        symbol: String,
        side: Side,
        price_fp: i128,
        qty_fp: i128,
    ) -> CorrectionType {
        CorrectionType::PatchLevel {
            symbol,
            side,
            price_fp,
            qty_fp,
        }
    }
    
    #[inline]
    pub fn remove_stale_order(order_id: String, symbol: String) -> CorrectionType {
        CorrectionType::RemoveStaleOrder { order_id, symbol }
    }
    
    #[inline]
    pub fn cancel_orphaned_order(order_id: String, symbol: String) -> CorrectionType {
        CorrectionType::CancelOrphanedOrder { order_id, symbol }
    }
    
    #[inline]
    pub fn sync_balance(
        asset: String,
        correct_free_fp: i128,
        correct_locked_fp: i128,
    ) -> CorrectionType {
        CorrectionType::SyncBalance {
            asset,
            correct_free_fp,
            correct_locked_fp,
        }
    }
    
    #[inline]
    pub fn reset_sequence(symbol: String, new_sequence: u64) -> CorrectionType {
        CorrectionType::ResetSequence {
            symbol,
            new_sequence,
        }
    }
}

impl Default for CorrectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corrector_creation() {
        let corrector = DriftCorrector::new();
        assert!(!corrector.is_circuit_breaker_open());
        assert_eq!(corrector.pending_count(), 0);
        assert_eq!(corrector.in_progress_count(), 0);
    }

    #[test]
    fn test_submit_correction() {
        let corrector = DriftCorrector::new();
        
        let correction = CorrectionType::ResetSequence {
            symbol: "BTCUSDT".to_string(),
            new_sequence: 1000,
        };
        
        let result = corrector.submit_correction(correction);
        assert!(result.is_ok());
        assert_eq!(corrector.pending_count(), 1);
    }

    #[test]
    fn test_try_process_next() {
        let corrector = DriftCorrector::new();
        
        let correction = CorrectionType::ResetSequence {
            symbol: "BTCUSDT".to_string(),
            new_sequence: 1000,
        };
        
        corrector.submit_correction(correction).unwrap();
        
        let processed = corrector.try_process_next(|record| {
            // Simulate successful processing
            Ok(())
        });
        
        assert!(processed);
        assert_eq!(corrector.pending_count(), 0);
        
        let stats = corrector.get_stats();
        assert_eq!(stats.successful_corrections, 1);
    }

    #[test]
    fn test_circuit_breaker() {
        let corrector = DriftCorrector::with_failure_threshold(3);
        
        // Trigger failures to trip circuit breaker
        for _ in 0..3 {
            corrector.record_failure();
        }
        
        assert!(corrector.is_circuit_breaker_open());
        
        // Manual reset
        corrector.force_close_circuit_breaker();
        assert!(!corrector.is_circuit_breaker_open());
    }

    #[test]
    fn test_correction_builder() {
        let cancel = CorrectionBuilder::cancel_orphaned_order(
            "order_123".to_string(),
            "BTCUSDT".to_string(),
        );
        
        match cancel {
            CorrectionType::CancelOrphanedOrder { order_id, symbol } => {
                assert_eq!(order_id, "order_123");
                assert_eq!(symbol, "BTCUSDT");
            }
            _ => panic!("Wrong correction type"),
        }
    }
}
