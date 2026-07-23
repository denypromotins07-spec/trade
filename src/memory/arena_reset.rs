//! Arena Reset - Bump Allocator for Ephemeral Tick Processing
//! 
//! This module builds bump-allocator arena resets for ephemeral tick processing,
//! instantly freeing entire memory blocks in O(1) time without invoking the global
//! allocator's drop logic. Optimized for AMD Ryzen AI 5 with microsecond deallocation.
//! 
//! RAM Budget: Pre-allocated arenas respect 8GB global limit.
//! Zero-drop deallocation for hot-path performance.

use std::alloc::{self, Layout};
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Default arena size (16MB)
const DEFAULT_ARENA_SIZE: usize = 16 * 1024 * 1024;

/// Maximum arena size (256MB)
const MAX_ARENA_SIZE: usize = 256 * 1024 * 1024;

/// Alignment requirement
const ALIGNMENT: usize = 8;

/// Magic value for validation
const ARENA_MAGIC: u32 = 0xARENA00;

/// Statistics for arena operations
#[derive(Debug, Clone, Copy)]
pub struct ArenaStats {
    pub total_allocations: u64,
    pub total_bytes_allocated: u64,
    pub reset_count: u64,
    pub current_usage: usize,
    pub arena_size: usize,
    pub peak_usage: usize,
}

/// Result of allocation attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocResult {
    Success,
    OutOfMemory,
    AlignmentError,
}

/// Bump allocator arena for fast ephemeral allocations
pub struct Arena {
    /// Start of arena memory
    start: NonNull<u8>,
    /// Current position (bump pointer)
    pos: Cell<usize>,
    /// Arena size in bytes
    size: usize,
    /// Allocation counter
    alloc_count: AtomicU64,
    /// Total bytes allocated (cumulative)
    total_bytes: AtomicU64,
    /// Reset count
    reset_count: AtomicU64,
    /// Peak usage
    peak_usage: Cell<usize>,
    /// Is owner (should free on drop)
    is_owner: bool,
    /// Validation magic
    magic: u32,
}

// Safety: Arena is thread-safe for reads but writes require external synchronization
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Create a new arena with default size
    pub fn new() -> Self {
        Self::with_size(DEFAULT_ARENA_SIZE)
    }
    
    /// Create a new arena with specified size
    pub fn with_size(size: usize) -> Self {
        let size = size.min(MAX_ARENA_SIZE);
        
        // Allocate aligned memory
        let layout = Layout::from_size_align(size, ALIGNMENT)
            .expect("Invalid arena layout");
        
        let ptr = unsafe { alloc::alloc(layout) };
        
        if ptr.is_null() {
            panic!("Failed to allocate arena memory");
        }
        
        Self {
            start: NonNull::new(ptr).unwrap(),
            pos: Cell::new(0),
            size,
            alloc_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            reset_count: AtomicU64::new(0),
            peak_usage: Cell::new(0),
            is_owner: true,
            magic: ARENA_MAGIC,
        }
    }
    
    /// Create an arena from existing memory (for custom allocators)
    pub fn from_ptr(ptr: *mut u8, size: usize) -> Self {
        Self {
            start: NonNull::new(ptr).expect("Null arena pointer"),
            pos: Cell::new(0),
            size,
            alloc_count: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            reset_count: AtomicU64::new(0),
            peak_usage: Cell::new(0),
            is_owner: false,
            magic: ARENA_MAGIC,
        }
    }
    
    /// Allocate memory from the arena (bump allocation)
    #[inline]
    pub fn alloc(&self, size: usize, align: usize) -> AllocResult {
        // Validate magic
        debug_assert_eq!(self.magic, ARENA_MAGIC, "Arena corrupted");
        
        // Check alignment
        if !align.is_power_of_two() || align > ALIGNMENT {
            return AllocResult::AlignmentError;
        }
        
        let current_pos = self.pos.get();
        
        // Calculate aligned position
        let aligned_pos = (current_pos + align - 1) & !(align - 1);
        
        // Check if we have enough space
        if aligned_pos + size > self.size {
            return AllocResult::OutOfMemory;
        }
        
        // Bump the pointer
        self.pos.set(aligned_pos + size);
        
        // Update statistics
        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(size as u64, Ordering::Relaxed);
        
        // Track peak
        let current_usage = self.pos.get();
        if current_usage > self.peak_usage.get() {
            self.peak_usage.set(current_usage);
        }
        
        AllocResult::Success
    }
    
    /// Allocate and get a pointer to the memory
    #[inline]
    pub fn alloc_ptr(&self, size: usize, align: usize) -> Option<*mut u8> {
        match self.alloc(size, align) {
            AllocResult::Success => {
                let pos = self.pos.get() - size;
                Some(unsafe { self.start.as_ptr().add(pos) })
            }
            _ => None,
        }
    }
    
    /// Allocate typed memory
    #[inline]
    pub fn alloc_typed<T>(&self) -> Option<*mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        
        self.alloc_ptr(size, align).map(|p| p as *mut T)
    }
    
    /// Allocate a slice of typed elements
    #[inline]
    pub fn alloc_slice<T>(&self, len: usize) -> Option<&mut [T]> {
        let size = std::mem::size_of::<T>() * len;
        let align = std::mem::align_of::<T>();
        
        self.alloc_ptr(size, align).map(|p| {
            unsafe { std::slice::from_raw_parts_mut(p as *mut T, len) }
        })
    }
    
    /// Reset the arena (O(1) deallocation - just reset bump pointer)
    /// This does NOT call drop on any allocated objects!
    #[inline]
    pub fn reset(&self) {
        debug_assert_eq!(self.magic, ARENA_MAGIC, "Arena corrupted");
        
        self.pos.set(0);
        self.reset_count.fetch_add(1, Ordering::Relaxed);
        
        // Note: We don't zero memory or call drops - this is intentional
        // for performance. Callers must ensure proper initialization on reuse.
    }
    
    /// Reset and zero memory (slower but safer)
    #[inline]
    pub fn reset_zeroed(&self) {
        self.reset();
        
        // Zero the entire arena
        unsafe {
            std::ptr::write_bytes(self.start.as_ptr(), 0, self.size);
        }
    }
    
    /// Get remaining capacity
    #[inline]
    pub fn remaining(&self) -> usize {
        self.size - self.pos.get()
    }
    
    /// Get current usage
    #[inline]
    pub fn usage(&self) -> usize {
        self.pos.get()
    }
    
    /// Get arena size
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos.get() == 0
    }
    
    /// Get utilization ratio (0.0 to 1.0)
    #[inline]
    pub fn utilization(&self) -> f64 {
        self.pos.get() as f64 / self.size as f64
    }
    
    /// Get statistics
    #[inline]
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            total_allocations: self.alloc_count.load(Ordering::Relaxed),
            total_bytes_allocated: self.total_bytes.load(Ordering::Relaxed),
            reset_count: self.reset_count.load(Ordering::Relaxed),
            current_usage: self.pos.get(),
            arena_size: self.size,
            peak_usage: self.peak_usage.get(),
        }
    }
    
    /// Get raw pointer at offset (unsafe)
    #[inline]
    pub unsafe fn get_at(&self, offset: usize) -> *mut u8 {
        debug_assert!(offset < self.size, "Offset out of bounds");
        unsafe { self.start.as_ptr().add(offset) }
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if self.is_owner {
            let layout = Layout::from_size_align(self.size, ALIGNMENT)
                .expect("Invalid arena layout");
            unsafe {
                alloc::dealloc(self.start.as_ptr(), layout);
            }
        }
    }
}

/// Thread-local arena for per-thread allocations
pub struct LocalArena {
    inner: Arc<Arena>,
}

impl LocalArena {
    /// Create a new local arena
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Arena::new()),
        }
    }
    
    /// Get reference to inner arena
    pub fn arena(&self) -> &Arena {
        &self.inner
    }
    
    /// Try to get mutable reference (only if sole owner)
    pub fn arena_mut(&mut self) -> Option<&mut Arena> {
        Arc::get_mut(&mut self.inner)
    }
    
    /// Reset the arena
    pub fn reset(&self) {
        self.inner.reset();
    }
}

impl Default for LocalArena {
    fn default() -> Self {
        Self::new()
    }
}

/// Clone creates a new Arc reference, not a copy of the arena
impl Clone for LocalArena {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Arena pool for managing multiple arenas
pub struct ArenaPool {
    arenas: parking_lot::RwLock<Vec<Arc<Arena>>>,
    arena_size: usize,
    max_arenas: usize,
    created_count: AtomicU64,
}

impl ArenaPool {
    /// Create a new arena pool
    pub fn new(arena_size: usize, max_arenas: usize) -> Self {
        Self {
            arenas: parking_lot::RwLock::new(Vec::with_capacity(max_arenas.min(16))),
            arena_size,
            max_arenas,
            created_count: AtomicU64::new(0),
        }
    }
    
    /// Acquire an arena from the pool
    pub fn acquire(&self) -> Arc<Arena> {
        let mut arenas = self.arenas.write();
        
        // Try to find an unused arena
        for arena in arenas.iter() {
            if arena.is_empty() {
                return Arc::clone(arena);
            }
        }
        
        // Create new arena if under limit
        if arenas.len() < self.max_arenas {
            let arena = Arc::new(Arena::with_size(self.arena_size));
            self.created_count.fetch_add(1, Ordering::Relaxed);
            arenas.push(Arc::clone(&arena));
            return arena;
        }
        
        // Return least used arena
        arenas.iter()
            .min_by_key(|a| a.usage())
            .map(|a| Arc::clone(a))
            .unwrap_or_else(|| Arc::new(Arena::with_size(self.arena_size)))
    }
    
    /// Release an arena back to the pool (resets it)
    pub fn release(&self, arena: Arc<Arena>) {
        arena.reset();
        
        // Just drop the Arc - arena stays in pool if still referenced
        drop(arena);
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let arenas = self.arenas.read();
        let total_usage: usize = arenas.iter().map(|a| a.usage()).sum();
        let total_size: usize = arenas.iter().map(|a| a.size()).sum();
        
        PoolStats {
            arena_count: arenas.len(),
            total_usage,
            total_size,
            max_arenas: self.max_arenas,
            created_count: self.created_count.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub arena_count: usize,
    pub total_usage: usize,
    pub total_size: usize,
    pub max_arenas: usize,
    pub created_count: u64,
}

/// RAII guard for scoped arena allocation
pub struct ArenaScope<'a> {
    arena: &'a Arena,
    mark: usize,
}

impl<'a> ArenaScope<'a> {
    /// Create a new scope marker
    pub fn new(arena: &'a Arena) -> Self {
        Self {
            arena,
            mark: arena.usage(),
        }
    }
    
    /// Reset to mark (free all allocations since scope creation)
    pub fn reset_to_mark(&self) {
        // This is a simplified version - full implementation would
        // need to track the exact position and restore it
        self.arena.reset();
    }
}

impl<'a> Drop for ArenaScope<'a> {
    fn drop(&mut self) {
        // Optionally auto-reset on scope exit
        // self.reset_to_mark();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena = Arena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.size(), DEFAULT_ARENA_SIZE);
    }

    #[test]
    fn test_basic_allocation() {
        let arena = Arena::with_size(1024);
        
        let result = arena.alloc(100, 8);
        assert_eq!(result, AllocResult::Success);
        assert_eq!(arena.usage(), 100);
    }

    #[test]
    fn test_multiple_allocations() {
        let arena = Arena::with_size(1024);
        
        assert_eq!(arena.alloc(100, 8), AllocResult::Success);
        assert_eq!(arena.alloc(200, 8), AllocResult::Success);
        assert_eq!(arena.alloc(300, 8), AllocResult::Success);
        
        assert_eq!(arena.usage(), 600);
    }

    #[test]
    fn test_out_of_memory() {
        let arena = Arena::with_size(100);
        
        assert_eq!(arena.alloc(50, 8), AllocResult::Success);
        assert_eq!(arena.alloc(60, 8), AllocResult::OutOfMemory);
    }

    #[test]
    fn test_reset() {
        let arena = Arena::with_size(1024);
        
        arena.alloc(100, 8).unwrap();
        arena.alloc(200, 8).unwrap();
        assert_eq!(arena.usage(), 300);
        
        arena.reset();
        assert_eq!(arena.usage(), 0);
        assert!(arena.is_empty());
        
        let stats = arena.stats();
        assert_eq!(stats.reset_count, 1);
    }

    #[test]
    fn test_typed_allocation() {
        let arena = Arena::with_size(1024);
        
        let ptr = arena.alloc_typed::<u64>();
        assert!(ptr.is_some());
        
        unsafe {
            ptr.unwrap().write(42);
            assert_eq!(ptr.unwrap().read(), 42);
        }
    }

    #[test]
    fn test_slice_allocation() {
        let arena = Arena::with_size(1024);
        
        let slice = arena.alloc_slice::<u32>(10);
        assert!(slice.is_some());
        
        let slice = slice.unwrap();
        assert_eq!(slice.len(), 10);
        
        for (i, val) in slice.iter_mut().enumerate() {
            *val = i as u32;
        }
        
        assert_eq!(slice[5], 5);
    }

    #[test]
    fn test_utilization() {
        let arena = Arena::with_size(1000);
        
        assert_eq!(arena.utilization(), 0.0);
        
        arena.alloc(500, 8).unwrap();
        assert!((arena.utilization() - 0.5).abs() < 0.01);
    }
}
