//! Custom Global Allocator with Heap Tracking
//!
//! Wraps the global allocator to track exact heap usage per thread,
//! triggering emergency flush if any component approaches the 8GB RAM limit.
//!
//! # Features
//! - Per-thread allocation tracking
//! - Emergency memory pressure handling
//! - Graceful degradation of non-essential caches
//! - Lock-free statistics collection

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::cell::RefCell;
use std::ptr::NonNull;

/// Global 8GB RAM limit
const GLOBAL_RAM_LIMIT: usize = 8 * 1024 * 1024 * 1024;
/// Warning threshold (85% of limit)
const WARNING_THRESHOLD: usize = (GLOBAL_RAM_LIMIT as f64 * 0.85) as usize;
/// Critical threshold (95% of limit)
const CRITICAL_THRESHOLD: usize = (GLOBAL_RAM_LIMIT as f64 * 0.95) as usize;

/// Total heap usage counter (atomic for lock-free access)
static TOTAL_HEAP_USAGE: AtomicUsize = AtomicUsize::new(0);
/// Peak heap usage
static PEAK_HEAP_USAGE: AtomicUsize = AtomicUsize::new(0);
/// Allocation count
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Deallocation count
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Emergency mode flag
static EMERGENCY_MODE: AtomicBool = AtomicBool::new(false);
/// GC trigger count
static GC_TRIGGERS: AtomicUsize = AtomicUsize::new(0);

/// Thread-local allocation tracker
thread_local! {
    static THREAD_ALLOC_BYTES: RefCell<usize> = RefCell::new(0);
}

/// Nautilus custom allocator wrapper
pub struct NautilusAllocator;

#[global_allocator]
pub static GLOBAL_ALLOCATOR: NautilusAllocator = NautilusAllocator;

unsafe impl GlobalAlloc for NautilusAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        
        // Check if we're in emergency mode
        if EMERGENCY_MODE.load(Ordering::Relaxed) {
            // In emergency mode, try to allocate but prepare for failure
            let ptr = System.alloc(layout);
            if !ptr.is_null() {
                self.track_allocation(size);
            }
            return ptr;
        }
        
        // Check if allocation would exceed critical threshold
        let current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
        if current.saturating_add(size) > CRITICAL_THRESHOLD {
            // Trigger emergency GC
            emergency_gc();
            
            // Re-check after GC
            let new_current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
            if new_current.saturating_add(size) > CRITICAL_THRESHOLD {
                // Still over limit - return null to signal OOM
                return std::ptr::null_mut();
            }
        }
        
        let ptr = System.alloc(layout);
        
        if !ptr.is_null() {
            self.track_allocation(size);
        }
        
        ptr
    }
    
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        self.track_deallocation(size);
        System.dealloc(ptr, layout);
    }
    
    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        
        if EMERGENCY_MODE.load(Ordering::Relaxed) {
            let ptr = System.alloc_zeroed(layout);
            if !ptr.is_null() {
                self.track_allocation(size);
            }
            return ptr;
        }
        
        let current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
        if current.saturating_add(size) > CRITICAL_THRESHOLD {
            emergency_gc();
            
            let new_current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
            if new_current.saturating_add(size) > CRITICAL_THRESHOLD {
                return std::ptr::null_mut();
            }
        }
        
        let ptr = System.alloc_zeroed(layout);
        
        if !ptr.is_null() {
            self.track_allocation(size);
        }
        
        ptr
    }
}

impl NautilusAllocator {
    #[inline]
    fn track_allocation(&self, size: usize) {
        TOTAL_HEAP_USAGE.fetch_add(size, Ordering::Relaxed);
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        
        // Update peak
        let mut current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
        let mut peak = PEAK_HEAP_USAGE.load(Ordering::Relaxed);
        
        while current > peak {
            match PEAK_HEAP_USAGE.compare_exchange_weak(
                peak, 
                current, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
        
        // Update thread-local counter
        THREAD_ALLOC_BYTES.with(|bytes| {
            let mut b = bytes.borrow_mut();
            *b = b.saturating_add(size);
        });
        
        // Check thresholds
        if current > WARNING_THRESHOLD && !EMERGENCY_MODE.load(Ordering::Relaxed) {
            eprintln!(
                "[HeapTracker] WARNING: Heap usage at {:.1}% ({:.2}GB / 8GB)",
                (current as f64 / GLOBAL_RAM_LIMIT as f64) * 100.0,
                current as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        
        if current > CRITICAL_THRESHOLD {
            EMERGENCY_MODE.store(true, Ordering::Relaxed);
            eprintln!(
                "[HeapTracker] CRITICAL: Entering emergency mode at {:.1}%",
                (current as f64 / GLOBAL_RAM_LIMIT as f64) * 100.0
            );
        }
    }
    
    #[inline]
    fn track_deallocation(&self, size: usize) {
        TOTAL_HEAP_USAGE.fetch_sub(size, Ordering::Relaxed);
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        
        // Update thread-local counter
        THREAD_ALLOC_BYTES.with(|bytes| {
            let mut b = bytes.borrow_mut();
            *b = b.saturating_sub(size);
        });
        
        // Exit emergency mode if below warning threshold
        let current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
        if current < WARNING_THRESHOLD {
            EMERGENCY_MODE.store(false, Ordering::Relaxed);
        }
    }
}

/// Trigger emergency garbage collection
#[inline]
fn emergency_gc() {
    GC_TRIGGERS.fetch_add(1, Ordering::Relaxed);
    
    // Notify system to flush non-essential caches
    flush_non_essential_caches();
    
    // On Python side, this would trigger gc.collect()
    // For now, just log
    eprintln!("[HeapTracker] Emergency GC triggered (count: {})", 
              GC_TRIGGERS.load(Ordering::Relaxed));
}

/// Flush non-essential caches to free memory
#[inline]
pub fn flush_non_essential_caches() {
    // This would integrate with the actual cache systems
    // For now, it's a hook for the broader system
    
    // Signal to other components to release cached data
    eprintln!("[HeapTracker] Flushing non-essential caches...");
}

/// Get current total heap usage in bytes
#[inline]
pub fn get_heap_usage() -> usize {
    TOTAL_HEAP_USAGE.load(Ordering::Relaxed)
}

/// Get peak heap usage in bytes
#[inline]
pub fn get_peak_usage() -> usize {
    PEAK_HEAP_USAGE.load(Ordering::Relaxed)
}

/// Get allocation statistics
#[inline]
pub fn get_alloc_stats() -> (usize, usize, usize, usize) {
    (
        TOTAL_HEAP_USAGE.load(Ordering::Relaxed),
        PEAK_HEAP_USAGE.load(Ordering::Relaxed),
        ALLOC_COUNT.load(Ordering::Relaxed),
        DEALLOC_COUNT.load(Ordering::Relaxed),
    )
}

/// Check if in emergency mode
#[inline]
pub fn is_emergency_mode() -> bool {
    EMERGENCY_MODE.load(Ordering::Relaxed)
}

/// Get heap usage as percentage of limit
#[inline]
pub fn get_usage_percent() -> f64 {
    let current = TOTAL_HEAP_USAGE.load(Ordering::Relaxed);
    (current as f64 / GLOBAL_RAM_LIMIT as f64) * 100.0
}

/// Get thread-local allocation count
#[inline]
pub fn get_thread_alloc_bytes() -> usize {
    THREAD_ALLOC_BYTES.with(|bytes| *bytes.borrow())
}

/// Reset statistics (for testing)
#[inline]
pub fn reset_stats() {
    TOTAL_HEAP_USAGE.store(0, Ordering::Relaxed);
    PEAK_HEAP_USAGE.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    DEALLOC_COUNT.store(0, Ordering::Relaxed);
    GC_TRIGGERS.store(0, Ordering::Relaxed);
    EMERGENCY_MODE.store(false, Ordering::Relaxed);
}

/// Memory budget guard for scoped allocations
pub struct MemoryBudgetGuard {
    budget: usize,
    allocated: usize,
}

impl MemoryBudgetGuard {
    /// Create a new memory budget guard
    #[inline]
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            allocated: 0,
        }
    }
    
    /// Try to allocate from budget
    #[inline]
    pub fn try_allocate(&mut self, size: usize) -> bool {
        if self.allocated + size <= self.budget {
            self.allocated += size;
            true
        } else {
            false
        }
    }
    
    /// Get remaining budget
    #[inline]
    pub fn remaining(&self) -> usize {
        self.budget - self.allocated
    }
    
    /// Get usage percentage
    #[inline]
    pub fn usage_percent(&self) -> f64 {
        (self.allocated as f64 / self.budget as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocator_basic() {
        reset_stats();
        
        // Allocate some memory
        let _vec: Vec<u8> = vec![0u8; 1024];
        
        let (usage, peak, allocs, deallocs) = get_alloc_stats();
        
        assert!(usage >= 1024);
        assert!(peak >= 1024);
        assert!(allocs > 0);
    }
    
    #[test]
    fn test_memory_budget_guard() {
        let mut guard = MemoryBudgetGuard::new(4096);
        
        assert!(guard.try_allocate(1024));
        assert_eq!(guard.remaining(), 3072);
        assert!((guard.usage_percent() - 25.0).abs() < 0.01);
        
        assert!(guard.try_allocate(2048));
        assert_eq!(guard.remaining(), 1024);
        
        assert!(!guard.try_allocate(2048)); // Would exceed budget
    }
    
    #[test]
    fn test_usage_percent() {
        reset_stats();
        
        // Should start near 0%
        assert!(get_usage_percent() < 1.0);
    }
}
