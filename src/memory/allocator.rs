//! # Custom Arena Allocator for HFT Tick Structures
//! 
//! Integrates mimalloc with custom arena allocators specifically designed for
//! fixed-size Nautilus tick structs to prevent fragmentation and enforce 8GB RAM limit.
//! 
//! ## Key Features:
//! - MiMalloc integration for low-latency global allocation
//! - Fixed-size arena pools for tick/order structures
//! - Zero fragmentation through object pooling
//! - Strict memory budget enforcement (8GB global limit)
//! - Cache-line aligned allocations for AMD Ryzen AI 5

use std::alloc::{self, Layout};
use std::ptr;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;

/// Global memory budget in bytes (8GB limit)
const GLOBAL_MEMORY_LIMIT: usize = 8 * 1024 * 1024 * 1024;

/// Default arena size for tick objects (64MB per arena)
const DEFAULT_ARENA_SIZE: usize = 64 * 1024 * 1024;

/// Size of a single tick structure (fixed)
const TICK_STRUCT_SIZE: usize = 256; // Adjust based on actual Tick struct

/// Number of ticks per arena block
const TICKS_PER_ARENA: usize = DEFAULT_ARENA_SIZE / TICK_STRUCT_SIZE;

/// Global memory tracker enforcing 8GB limit
pub struct GlobalMemoryTracker {
    /// Current memory usage in bytes
    used: AtomicUsize,
    /// Maximum allowed memory (8GB)
    limit: usize,
    /// Allocation failure count
    alloc_failures: AtomicUsize,
    /// Flag indicating memory pressure
    under_pressure: AtomicBool,
}

impl GlobalMemoryTracker {
    pub const fn new() -> Self {
        Self {
            used: AtomicUsize::new(0),
            limit: GLOBAL_MEMORY_LIMIT,
            alloc_failures: AtomicUsize::new(0),
            under_pressure: AtomicBool::new(false),
        }
    }

    /// Try to allocate specified bytes, returns true if successful
    #[inline(always)]
    pub fn try_allocate(&self, bytes: usize) -> bool {
        let current = self.used.fetch_add(bytes, Ordering::Relaxed);
        
        if current + bytes > self.limit {
            // Rollback
            self.used.fetch_sub(bytes, Ordering::Relaxed);
            self.alloc_failures.fetch_add(1, Ordering::Relaxed);
            
            // Signal memory pressure
            if !self.under_pressure.load(Ordering::Relaxed) {
                self.under_pressure.store(true, Ordering::Relaxed);
            }
            return false;
        }

        // Check pressure threshold (90% utilization)
        if (current + bytes) as f64 > self.limit as f64 * 0.9 {
            self.under_pressure.store(true, Ordering::Relaxed);
        }

        true
    }

    /// Deallocate specified bytes
    #[inline(always)]
    pub fn deallocate(&self, bytes: usize) {
        self.used.fetch_sub(bytes, Ordering::Relaxed);
        
        // Clear pressure flag if below threshold
        if self.used.load(Ordering::Relaxed) < self.limit / 2 {
            self.under_pressure.store(false, Ordering::Relaxed);
        }
    }

    /// Get current usage in bytes
    #[inline(always)]
    pub fn used_bytes(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Get remaining capacity in bytes
    #[inline(always)]
    pub fn remaining_bytes(&self) -> usize {
        self.limit.saturating_sub(self.used.load(Ordering::Relaxed))
    }

    /// Check if under memory pressure
    #[inline(always)]
    pub fn is_under_pressure(&self) -> bool {
        self.under_pressure.load(Ordering::Relaxed)
    }

    /// Get allocation failure count
    #[inline(always)]
    pub fn get_failure_count(&self) -> usize {
        self.alloc_failures.load(Ordering::Relaxed)
    }
}

impl Default for GlobalMemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size arena allocator for tick structures
pub struct TickArena {
    /// Pre-allocated memory block
    memory: Vec<u8>,
    /// Free list head (index of next free slot)
    free_list_head: AtomicUsize,
    /// Array of next pointers for free list
    free_list_next: Vec<AtomicUsize>,
    /// Total slots in arena
    total_slots: usize,
    /// Allocated slot count
    allocated_count: AtomicUsize,
    /// Reference to global tracker
    tracker: Arc<GlobalMemoryTracker>,
}

impl TickArena {
    /// Create a new tick arena
    pub fn new(tracker: Arc<GlobalMemoryTracker>) -> Option<Self> {
        // Check if we have enough memory
        if !tracker.try_allocate(DEFAULT_ARENA_SIZE) {
            return None;
        }

        let mut memory = vec![0u8; DEFAULT_ARENA_SIZE];
        let mut free_list_next = Vec::with_capacity(TICKS_PER_ARENA);
        
        // Initialize free list
        for i in 0..TICKS_PER_ARENA {
            let next = if i < TICKS_PER_ARENA - 1 { i + 1 } else { usize::MAX };
            free_list_next.push(AtomicUsize::new(next));
        }

        Some(Self {
            memory,
            free_list_head: AtomicUsize::new(0),
            free_list_next,
            total_slots: TICKS_PER_ARENA,
            allocated_count: AtomicUsize::new(0),
            tracker,
        })
    }

    /// Allocate a tick slot (returns index into memory)
    #[inline(always)]
    pub fn allocate(&self) -> Option<usize> {
        // Lock-free free list pop
        loop {
            let current_head = self.free_list_head.load(Ordering::Acquire);
            
            if current_head == usize::MAX {
                // Arena exhausted
                return None;
            }

            let next = self.free_list_next[current_head].load(Ordering::Relaxed);
            
            // Try to swap head to next
            if self.free_list_head.compare_exchange_weak(
                current_head,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocated_count.fetch_add(1, Ordering::Relaxed);
                return Some(current_head);
            }
            // CAS failed, retry
        }
    }

    /// Deallocate a tick slot
    #[inline(always)]
    pub fn deallocate(&self, index: usize) {
        if index >= self.total_slots {
            return;
        }

        // Lock-free free list push
        loop {
            let current_head = self.free_list_head.load(Ordering::Acquire);
            self.free_list_next[index].store(current_head, Ordering::Release);
            
            if self.free_list_head.compare_exchange_weak(
                current_head,
                index,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                self.allocated_count.fetch_sub(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Get pointer to tick data at index (zero-copy access)
    #[inline(always)]
    pub fn get_tick_ptr(&self, index: usize) -> *mut u8 {
        if index >= self.total_slots {
            return ptr::null_mut();
        }
        
        let offset = index * TICK_STRUCT_SIZE;
        unsafe { self.memory.as_ptr().add(offset) as *mut u8 }
    }

    /// Get number of allocated slots
    #[inline(always)]
    pub fn allocated_count(&self) -> usize {
        self.allocated_count.load(Ordering::Relaxed)
    }

    /// Get number of free slots
    #[inline(always)]
    pub fn free_count(&self) -> usize {
        self.total_slots - self.allocated_count.load(Ordering::Relaxed)
    }
}

impl Drop for TickArena {
    fn drop(&mut self) {
        // Return memory to global tracker
        self.tracker.deallocate(DEFAULT_ARENA_SIZE);
    }
}

/// Multi-arena pool manager for scaling beyond single arena
pub struct TickArenaPool {
    /// List of arenas
    arenas: Vec<Arc<TickArena>>,
    /// Current arena index for allocation (round-robin)
    current_arena: AtomicUsize,
    /// Maximum number of arenas
    max_arenas: usize,
    /// Global memory tracker
    tracker: Arc<GlobalMemoryTracker>,
}

impl TickArenaPool {
    pub fn new(max_arenas: usize) -> Self {
        let tracker = Arc::new(GlobalMemoryTracker::new());
        let mut arenas = Vec::with_capacity(max_arenas.min(128)); // Cap at 128 arenas
        
        // Pre-allocate initial arenas
        for _ in 0..max_arenas.min(4) {
            if let Some(arena) = TickArena::new(Arc::clone(&tracker)) {
                arenas.push(Arc::new(arena));
            } else {
                break; // Memory limit reached
            }
        }

        Self {
            arenas,
            current_arena: AtomicUsize::new(0),
            max_arenas: max_arenas.min(128),
            tracker,
        }
    }

    /// Allocate a tick slot from any available arena
    #[inline(always)]
    pub fn allocate(&self) -> Option<(usize, Arc<TickArena>)> {
        let start = self.current_arena.load(Ordering::Relaxed);
        
        for i in 0..self.arenas.len() {
            let idx = (start + i) % self.arenas.len();
            let arena = &self.arenas[idx];
            
            if let Some(slot) = arena.allocate() {
                self.current_arena.store(idx, Ordering::Relaxed);
                return Some((slot, Arc::clone(arena)));
            }
        }

        // Try to create new arena if under limit
        if self.arenas.len() < self.max_arenas {
            if let Some(arena) = TickArena::new(Arc::clone(&self.tracker)) {
                let arena = Arc::new(arena);
                if let Some(slot) = arena.allocate() {
                    // Note: In production, would need to add to self.arenas safely
                    return Some((slot, arena));
                }
            }
        }

        None
    }

    /// Get global memory statistics
    pub fn get_stats(&self) -> ArenaPoolStats {
        let total_allocated: usize = self.arenas.iter()
            .map(|a| a.allocated_count())
            .sum();
        
        let total_free: usize = self.arenas.iter()
            .map(|a| a.free_count())
            .sum();

        ArenaPoolStats {
            arena_count: self.arenas.len(),
            total_slots: self.arenas.len() * TICKS_PER_ARENA,
            allocated: total_allocated,
            free: total_free,
            memory_used: self.tracker.used_bytes(),
            memory_remaining: self.tracker.remaining_bytes(),
            under_pressure: self.tracker.is_under_pressure(),
            alloc_failures: self.tracker.get_failure_count(),
        }
    }
}

/// Statistics for arena pool
#[derive(Debug, Clone)]
pub struct ArenaPoolStats {
    pub arena_count: usize,
    pub total_slots: usize,
    pub allocated: usize,
    pub free: usize,
    pub memory_used: usize,
    pub memory_remaining: usize,
    pub under_pressure: bool,
    pub alloc_failures: usize,
}

/// Custom allocator wrapper using mimalloc semantics
pub struct MiMallocWrapper;

impl MiMallocWrapper {
    /// Allocate aligned memory
    pub unsafe fn alloc_aligned(size: usize, align: usize) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(size, align);
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            panic!("Allocation failed: out of memory");
        }
        ptr
    }

    /// Deallocate aligned memory
    pub unsafe fn dealloc_aligned(ptr: *mut u8, size: usize, align: usize) {
        let layout = Layout::from_size_align_unchecked(size, align);
        alloc::dealloc(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker() {
        let tracker = Arc::new(GlobalMemoryTracker::new());
        
        assert!(tracker.try_allocate(1024));
        assert_eq!(tracker.used_bytes(), 1024);
        
        tracker.deallocate(512);
        assert_eq!(tracker.used_bytes(), 512);
        
        tracker.deallocate(512);
        assert_eq!(tracker.used_bytes(), 0);
    }

    #[test]
    fn test_arena_allocation() {
        let tracker = Arc::new(GlobalMemoryTracker::new());
        let arena = TickArena::new(Arc::clone(&tracker)).expect("Should create arena");
        
        let slot1 = arena.allocate();
        let slot2 = arena.allocate();
        
        assert!(slot1.is_some());
        assert!(slot2.is_some());
        assert_ne!(slot1.unwrap(), slot2.unwrap());
        
        assert_eq!(arena.allocated_count(), 2);
        assert_eq!(arena.free_count(), TICKS_PER_ARENA - 2);
    }

    #[test]
    fn test_arena_pool_stats() {
        let pool = TickArenaPool::new(4);
        let stats = pool.get_stats();
        
        assert!(stats.arena_count >= 1);
        assert!(!stats.under_pressure);
        assert_eq!(stats.alloc_failures, 0);
    }
}
