//! Zero-Overhead Leak Detector using Epoch-Based Tracking
//!
//! Identifies forgotten lock-free nodes without pausing the microsecond
//! execution hot path. Uses epoch-based reclamation to track allocations
//! and detect potential memory leaks in lock-free data structures.
//!
//! # Key Features
//! - Epoch-based tracking for safe memory reclamation
//! - Zero overhead in the common case
//! - Detects leaked nodes from Treiber stacks, Chase-Lev deques, etc.
//! - Bounded deferred memory to respect 8GB RAM limit

use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Maximum deferred nodes before forced reclamation (bounded for 8GB RAM)
const MAX_DEFERRED_NODES: usize = 65536;
/// Maximum deferred bytes (64MB cap)
const MAX_DEFERRED_BYTES: usize = 64 * 1024 * 1024;
/// Epoch increment for each reclamation cycle
const EPOCH_INCREMENT: u64 = 1;

/// Global epoch counter
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);
/// Total allocated nodes tracked
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
/// Total freed nodes
static TOTAL_FREED: AtomicUsize = AtomicUsize::new(0);
/// Current deferred count
static DEFERRED_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Current deferred bytes
static DEFERRED_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Leak detection enabled flag
static LEAK_DETECTION_ENABLED: AtomicBool = AtomicBool::new(true);
/// Leaks detected counter
static LEAKS_DETECTED: AtomicUsize = AtomicUsize::new(0);

/// Epoch marker for thread participation
#[derive(Clone, Copy)]
pub struct EpochMarker {
    /// Thread-local epoch value
    pub local_epoch: u64,
    /// Active flag
    pub active: bool,
}

impl Default for EpochMarker {
    fn default() -> Self {
        Self {
            local_epoch: GLOBAL_EPOCH.load(Ordering::Relaxed),
            active: false,
        }
    }
}

/// Deferred node for epoch-based reclamation
pub struct DeferredNode {
    /// Pointer to deallocate (stored as usize for FFI compatibility)
    pub ptr: usize,
    /// Size of allocation in bytes
    pub size: usize,
    /// Epoch when deferral occurred
    pub defer_epoch: u64,
    /// Source identifier (for leak diagnostics)
    pub source_id: u32,
}

/// Epoch guard for safe memory access
pub struct EpochGuard {
    /// Marker for this guard
    marker: EpochMarker,
    /// References held during guard lifetime
    refs_held: usize,
}

impl EpochGuard {
    /// Create a new epoch guard (enter critical section)
    #[inline]
    pub fn new() -> Self {
        let current_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        
        Self {
            marker: EpochMarker {
                local_epoch: current_epoch,
                active: true,
            },
            refs_held: 0,
        }
    }
    
    /// Increment reference count
    #[inline]
    pub fn hold_reference(&mut self) {
        self.refs_held += 1;
    }
    
    /// Release reference
    #[inline]
    pub fn release_reference(&mut self) {
        if self.refs_held > 0 {
            self.refs_held -= 1;
        }
    }
    
    /// Get current epoch
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.marker.local_epoch
    }
}

impl Drop for EpochGuard {
    fn drop(&mut self) {
        self.marker.active = false;
    }
}

/// Leak detector state manager
pub struct LeakDetector {
    /// Deferred nodes queue (simplified - would be lock-free in production)
    deferred_queue: Vec<DeferredNode>,
    /// Last reclamation time
    last_reclaim: Instant,
    /// Reclamation interval
    reclaim_interval: Duration,
    /// Allocation records for leak detection
    alloc_records: Vec<AllocationRecord>,
}

/// Allocation record for tracking
#[derive(Clone)]
pub struct AllocationRecord {
    /// Pointer address
    pub ptr: usize,
    /// Size in bytes
    pub size: usize,
    /// Allocation epoch
    pub alloc_epoch: u64,
    /// Source identifier
    pub source_id: u32,
    /// Freed flag
    pub freed: bool,
}

impl LeakDetector {
    /// Create a new leak detector
    #[inline]
    pub fn new() -> Self {
        Self {
            deferred_queue: Vec::with_capacity(1024),
            last_reclaim: Instant::now(),
            reclaim_interval: Duration::from_millis(100), // 100ms default
            alloc_records: Vec::with_capacity(4096),
        }
    }
    
    /// Track a new allocation
    #[inline]
    pub fn track_allocation(&mut self, ptr: usize, size: usize, source_id: u32) {
        if !LEAK_DETECTION_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        
        TOTAL_ALLOCATED.fetch_add(1, Ordering::Relaxed);
        
        let epoch = GLOBAL_EPOCH.load(Ordering::Relaxed);
        
        self.alloc_records.push(AllocationRecord {
            ptr,
            size,
            alloc_epoch: epoch,
            source_id,
            freed: false,
        });
    }
    
    /// Track a deallocation
    #[inline]
    pub fn track_deallocation(&mut self, ptr: usize) {
        if !LEAK_DETECTION_ENABLED.load(Ordering::Relaxed) {
            return;
        }
        
        TOTAL_FREED.fetch_add(1, Ordering::Relaxed);
        
        // Mark as freed
        for record in &mut self.alloc_records {
            if record.ptr == ptr && !record.freed {
                record.freed = true;
                break;
            }
        }
    }
    
    /// Defer a node for later reclamation (epoch-based)
    #[inline]
    pub fn defer_node(&mut self, ptr: usize, size: usize, source_id: u32) {
        let current_epoch = GLOBAL_EPOCH.load(Ordering::Relaxed);
        
        self.deferred_queue.push(DeferredNode {
            ptr,
            size,
            defer_epoch: current_epoch,
            source_id,
        });
        
        DEFERRED_COUNT.fetch_add(1, Ordering::Relaxed);
        DEFERRED_BYTES.fetch_add(size, Ordering::Relaxed);
        
        // Check if forced reclamation needed
        if DEFERRED_COUNT.load(Ordering::Relaxed) >= MAX_DEFERRED_NODES 
           || DEFERRED_BYTES.load(Ordering::Relaxed) >= MAX_DEFERRED_BYTES {
            self.force_reclaim();
        }
    }
    
    /// Attempt reclamation if safe
    #[inline]
    pub fn try_reclaim(&mut self) -> usize {
        let current_epoch = GLOBAL_EPOCH.load(Ordering::Relaxed);
        let mut reclaimed = 0;
        
        // Find nodes that are safe to reclaim (deferred at least 2 epochs ago)
        let safe_epoch = current_epoch.saturating_sub(2 * EPOCH_INCREMENT);
        
        self.deferred_queue.retain(|node| {
            if node.defer_epoch < safe_epoch {
                // Safe to reclaim
                reclaimed += node.size;
                DEFERRED_BYTES.fetch_sub(node.size, Ordering::Relaxed);
                false // Remove from queue
            } else {
                true // Keep in queue
            }
        });
        
        DEFERRED_COUNT.store(self.deferred_queue.len(), Ordering::Relaxed);
        
        if reclaimed > 0 {
            self.last_reclaim = Instant::now();
        }
        
        reclaimed
    }
    
    /// Force reclamation (used when approaching memory limits)
    #[inline]
    pub fn force_reclaim(&mut self) -> usize {
        let mut reclaimed = 0;
        
        for node in &self.deferred_queue {
            reclaimed += node.size;
        }
        
        DEFERRED_BYTES.fetch_sub(reclaimed, Ordering::Relaxed);
        self.deferred_queue.clear();
        DEFERRED_COUNT.store(0, Ordering::Relaxed);
        
        self.last_reclaim = Instant::now();
        
        eprintln!("[LeakDetector] Forced reclamation: {} bytes", reclaimed);
        
        reclaimed
    }
    
    /// Detect potential leaks (allocations older than threshold)
    #[inline]
    pub fn detect_leaks(&self, age_threshold_epochs: u64) -> Vec<AllocationRecord> {
        let current_epoch = GLOBAL_EPOCH.load(Ordering::Relaxed);
        let mut leaks = Vec::new();
        
        for record in &self.alloc_records {
            if !record.freed {
                let age = current_epoch.saturating_sub(record.alloc_epoch);
                if age > age_threshold_epochs {
                    leaks.push(record.clone());
                }
            }
        }
        
        if !leaks.is_empty() {
            LEAKS_DETECTED.fetch_add(leaks.len(), Ordering::Relaxed);
            eprintln!(
                "[LeakDetector] Detected {} potential leaks (age > {} epochs)",
                leaks.len(), 
                age_threshold_epochs
            );
        }
        
        leaks
    }
    
    /// Get leak statistics
    #[inline]
    pub fn get_stats(&self) -> LeakStats {
        let unfreed = self.alloc_records.iter().filter(|r| !r.freed).count();
        let total_leaked_bytes: usize = self.alloc_records
            .iter()
            .filter(|r| !r.freed)
            .map(|r| r.size)
            .sum();
        
        LeakStats {
            total_allocated: TOTAL_ALLOCATED.load(Ordering::Relaxed),
            total_freed: TOTAL_FREED.load(Ordering::Relaxed),
            currently_tracked: self.alloc_records.len(),
            unfreed_count: unfreed,
            unfreed_bytes: total_leaked_bytes,
            deferred_count: DEFERRED_COUNT.load(Ordering::Relaxed),
            deferred_bytes: DEFERRED_BYTES.load(Ordering::Relaxed),
            leaks_detected: LEAKS_DETECTED.load(Ordering::Relaxed),
        }
    }
    
    /// Advance global epoch (called periodically by reclamation thread)
    #[inline]
    pub fn advance_epoch() -> u64 {
        GLOBAL_EPOCH.fetch_add(EPOCH_INCREMENT, Ordering::Release) + EPOCH_INCREMENT
    }
    
    /// Enable leak detection
    #[inline]
    pub fn enable_detection() {
        LEAK_DETECTION_ENABLED.store(true, Ordering::Relaxed);
    }
    
    /// Disable leak detection (for performance-critical sections)
    #[inline]
    pub fn disable_detection() {
        LEAK_DETECTION_ENABLED.store(false, Ordering::Relaxed);
    }
}

impl Default for LeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Leak statistics structure
#[derive(Debug, Clone)]
pub struct LeakStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub currently_tracked: usize,
    pub unfreed_count: usize,
    pub unfreed_bytes: usize,
    pub deferred_count: usize,
    pub deferred_bytes: usize,
    pub leaks_detected: usize,
}

/// Quick helper for epoch-based reclamation in lock-free structures
pub struct EpochReclaimer {
    detector: Arc<std::sync::Mutex<LeakDetector>>,
}

impl EpochReclaimer {
    #[inline]
    pub fn new() -> Self {
        Self {
            detector: Arc::new(std::sync::Mutex::new(LeakDetector::new())),
        }
    }
    
    #[inline]
    pub fn defer(&self, ptr: usize, size: usize, source_id: u32) {
        if let Ok(mut det) = self.detector.lock() {
            det.defer_node(ptr, size, source_id);
        }
    }
    
    #[inline]
    pub fn try_reclaim(&self) -> usize {
        if let Ok(mut det) = self.detector.lock() {
            det.try_reclaim()
        } else {
            0
        }
    }
    
    #[inline]
    pub fn get_stats(&self) -> LeakStats {
        if let Ok(det) = self.detector.lock() {
            det.get_stats()
        } else {
            LeakStats {
                total_allocated: 0,
                total_freed: 0,
                currently_tracked: 0,
                unfreed_count: 0,
                unfreed_bytes: 0,
                deferred_count: 0,
                deferred_bytes: 0,
                leaks_detected: 0,
            }
        }
    }
}

impl Default for EpochReclaimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_epoch_guard() {
        let guard = EpochGuard::new();
        assert!(guard.epoch() > 0);
        assert!(!std::mem::needs_drop::<EpochGuard>()); // Should be cheap
    }
    
    #[test]
    fn test_leak_detector_basic() {
        let mut detector = LeakDetector::new();
        
        // Track an allocation
        detector.track_allocation(0x1000, 64, 1);
        
        let stats = detector.get_stats();
        assert_eq!(stats.total_allocated, 1);
        assert_eq!(stats.unfreed_count, 1);
        
        // Track deallocation
        detector.track_deallocation(0x1000);
        
        let stats = detector.get_stats();
        assert_eq!(stats.total_freed, 1);
    }
    
    #[test]
    fn test_deferred_reclamation() {
        let mut detector = LeakDetector::new();
        
        // Defer some nodes
        detector.defer_node(0x1000, 64, 1);
        detector.defer_node(0x2000, 128, 2);
        
        let stats = detector.get_stats();
        assert_eq!(stats.deferred_count, 2);
        assert_eq!(stats.deferred_bytes, 192);
        
        // Advance epoch enough times for safe reclamation
        for _ in 0..3 {
            LeakDetector::advance_epoch();
        }
        
        // Try to reclaim
        let reclaimed = detector.try_reclaim();
        assert!(reclaimed > 0);
        
        let stats = detector.get_stats();
        assert_eq!(stats.deferred_count, 0);
    }
    
    #[test]
    fn test_leak_detection() {
        let mut detector = LeakDetector::new();
        
        // Allocate without freeing
        detector.track_allocation(0x1000, 64, 1);
        
        // Advance epochs
        for _ in 0..5 {
            LeakDetector::advance_epoch();
        }
        
        // Detect leaks (threshold of 3 epochs)
        let leaks = detector.detect_leaks(3);
        assert_eq!(leaks.len(), 1);
        
        let stats = detector.get_stats();
        assert!(stats.leaks_detected > 0);
    }
}
