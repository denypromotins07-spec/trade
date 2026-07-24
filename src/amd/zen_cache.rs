//! AMD Zen Architecture Cache Line Alignment and False-Sharing Prevention
//!
//! This module implements strict 64-byte cache line alignment and false-sharing
//! prevention macros for all hot-path structs, optimized specifically for
//! AMD Zen 4/Zen 5 L1/L2 cache topologies.
//!
//! Key features:
//! - 64-byte cache line alignment (Zen architecture optimal)
//! - False-sharing prevention via padding
//! - Cache-aware data structure layout
//! - L1/L2 topology-specific optimizations
//!
//! Author: Elite Quantitative Software Engineering Team
//! Stage: 49 - AMD Zen Architecture Tuning

use std::alloc::{self, Layout};
use std::ptr;

// =============================================================================
// AMD Zen Cache Topology Constants
// =============================================================================

/// AMD Zen 4/Zen 5 cache line size (64 bytes)
/// This is critical for preventing false sharing between cores
pub const ZEN_CACHE_LINE_SIZE: usize = 64;

/// L1 Data Cache size per core (Zen 4: 48KB, Zen 5: 64KB)
pub const ZEN_L1D_SIZE: usize = 48 * 1024;

/// L2 Cache size per core complex (Zen 4: 1MB, Zen 5: 2MB)
pub const ZEN_L2_SIZE: usize = 1024 * 1024;

/// L3 Cache size per CCD (Core Complex Die)
pub const ZEN_L3_PER_CCD: usize = 32 * 1024 * 1024;

/// Optimal data alignment for Zen architecture
#[repr(align(64))]
pub struct CacheAligned<T> {
    data: T,
}

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned wrapper
    #[inline(always)]
    pub fn new(data: T) -> Self {
        Self { data }
    }

    /// Get reference to inner data
    #[inline(always)]
    pub fn get(&self) -> &T {
        &self.data
    }

    /// Get mutable reference to inner data
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consume wrapper and return inner data
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.data
    }
}

/// Ensure CacheAligned is always 64-byte aligned
#[test]
fn test_cache_aligned_alignment() {
    assert!(std::mem::align_of::<CacheAligned<u8>>() >= ZEN_CACHE_LINE_SIZE);
}

// =============================================================================
// False Sharing Prevention Macros
// =============================================================================

/// Macro to add cache line padding before a field
/// Prevents false sharing by ensuring the field starts on a new cache line
#[macro_export]
macro_rules! cache_pad_before {
    ($field:ident, $type:ty) => {
        #[doc(hidden)]
        _pad_before_$field: [u8; $crate::amd::zen_cache::ZEN_CACHE_LINE_SIZE],
        $field: $type,
    };
}

/// Macro to add cache line padding after a field
/// Prevents false sharing by ensuring the field ends on a cache line boundary
#[macro_export]
macro_rules! cache_pad_after {
    ($field:ident, $type:ty) => {
        $field: $type,
        #[doc(hidden)]
        _pad_after_$field: [u8; $crate::amd::zen_cache::ZEN_CACHE_LINE_SIZE],
    };
}

/// Macro to isolate a field on its own cache line
/// Provides both before and after padding for complete isolation
#[macro_export]
macro_rules! cache_isolate {
    ($field:ident, $type:ty) => {
        #[doc(hidden)]
        _pad_before_$field: [u8; $crate::amd::zen_cache::ZEN_CACHE_LINE_SIZE],
        $field: $type,
        #[doc(hidden)]
        _pad_after_$field: [u8; $crate::amd::zen_cache::ZEN_CACHE_LINE_SIZE],
    };
}

// =============================================================================
// Cache-Aligned Atomic Types for Lock-Free Programming
// =============================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Cache-line aligned atomic counter for lock-free programming
/// Prevents false sharing when multiple counters are accessed by different cores
#[repr(C)]
#[repr(align(64))]
pub struct CacheAlignedAtomicU64 {
    value: AtomicU64,
    // Padding ensures this struct occupies exactly one cache line
    _padding: [u8; 56], // 64 - 8 (AtomicU64 size) = 56
}

impl CacheAlignedAtomicU64 {
    /// Create a new cache-aligned atomic counter
    #[inline(always)]
    pub fn new(value: u64) -> Self {
        Self {
            value: AtomicU64::new(value),
            _padding: [0u8; 56],
        }
    }

    /// Load value with specified ordering
    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u64 {
        self.value.load(order)
    }

    /// Store value with specified ordering
    #[inline(always)]
    pub fn store(&self, val: u64, order: Ordering) {
        self.value.store(val, order);
    }

    /// Atomically add and return previous value
    #[inline(always)]
    pub fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        self.value.fetch_add(val, order)
    }

    /// Atomically subtract and return previous value
    #[inline(always)]
    pub fn fetch_sub(&self, val: u64, order: Ordering) -> u64 {
        self.value.fetch_sub(val, order)
    }

    /// Compare-and-swap operation
    #[inline(always)]
    pub fn compare_exchange(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        self.value.compare_exchange(current, new, success, failure)
    }
}

/// Cache-line aligned atomic usize
#[repr(C)]
#[repr(align(64))]
pub struct CacheAlignedAtomicUsize {
    value: AtomicUsize,
    _padding: [u8; 56], // 64 - 8 (AtomicUsize on 64-bit) = 56
}

impl CacheAlignedAtomicUsize {
    #[inline(always)]
    pub fn new(value: usize) -> Self {
        Self {
            value: AtomicUsize::new(value),
            _padding: [0u8; 56],
        }
    }

    #[inline(always)]
    pub fn load(&self, order: Ordering) -> usize {
        self.value.load(order)
    }

    #[inline(always)]
    pub fn store(&self, val: usize, order: Ordering) {
        self.value.store(val, order);
    }

    #[inline(always)]
    pub fn fetch_add(&self, val: usize, order: Ordering) -> usize {
        self.value.fetch_add(val, order)
    }

    #[inline(always)]
    pub fn fetch_sub(&self, val: usize, order: Ordering) -> usize {
        self.value.fetch_sub(val, order)
    }
}

// =============================================================================
// Multi-Core Counter Array with False Sharing Prevention
// =============================================================================

/// Array of cache-aligned counters, one per logical core
/// Each counter is isolated on its own cache line to prevent false sharing
pub struct PerCoreCounter {
    counters: Vec<CacheAlignedAtomicU64>,
    num_cores: usize,
}

impl PerCoreCounter {
    /// Create a new per-core counter array
    pub fn new(num_cores: usize) -> Self {
        let mut counters = Vec::with_capacity(num_cores);
        
        for _ in 0..num_cores {
            counters.push(CacheAlignedAtomicU64::new(0));
        }
        
        Self {
            counters,
            num_cores,
        }
    }

    /// Get counter for specific core
    #[inline(always)]
    pub fn get_core_counter(&self, core_id: usize) -> Option<&CacheAlignedAtomicU64> {
        self.counters.get(core_id)
    }

    /// Increment counter for specific core
    #[inline(always)]
    pub fn increment_core(&self, core_id: usize, value: u64) {
        if let Some(counter) = self.counters.get(core_id) {
            counter.fetch_add(value, Ordering::Relaxed);
        }
    }

    /// Get total sum across all cores
    pub fn total_sum(&self) -> u64 {
        self.counters
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }
}

// =============================================================================
// Cache-Aware Memory Allocator
// =============================================================================

/// Allocate memory aligned to cache line boundary
/// Ensures allocated data doesn't straddle cache lines
pub fn allocate_cache_aligned<T>(size: usize) -> *mut T {
    let layout = Layout::from_size_align(
        size * std::mem::size_of::<T>(),
        ZEN_CACHE_LINE_SIZE,
    ).expect("Invalid layout");

    unsafe {
        let ptr = alloc::alloc(layout);
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        ptr as *mut T
    }
}

/// Free cache-aligned allocated memory
pub unsafe fn free_cache_aligned<T>(ptr: *mut T, size: usize) {
    let layout = Layout::from_size_align(
        size * std::mem::size_of::<T>(),
        ZEN_CACHE_LINE_SIZE,
    ).expect("Invalid layout");

    alloc::dealloc(ptr as *mut u8, layout);
}

// =============================================================================
// Hot-Path Struct Templates for Order Book Data
// =============================================================================

/// Cache-optimized order book level structure
/// Aligned to prevent false sharing in multi-threaded matching engine
#[repr(C)]
#[repr(align(64))]
pub struct OrderBookLevel {
    pub price: u64,           // 8 bytes
    pub bid_volume: u64,      // 8 bytes
    pub ask_volume: u64,      // 8 bytes
    pub bid_order_count: u32, // 4 bytes
    pub ask_order_count: u32, // 4 bytes
    pub timestamp_ns: u64,    // 8 bytes
    // Padding to reach 64 bytes: 64 - 40 = 24
    _padding: [u8; 24],
}

impl OrderBookLevel {
    #[inline(always)]
    pub fn new(price: u64) -> Self {
        Self {
            price,
            bid_volume: 0,
            ask_volume: 0,
            bid_order_count: 0,
            ask_order_count: 0,
            timestamp_ns: 0,
            _padding: [0u8; 24],
        }
    }

    /// Verify struct size is exactly one cache line
    #[inline(always)]
    pub const fn verify_size() -> bool {
        std::mem::size_of::<OrderBookLevel>() == ZEN_CACHE_LINE_SIZE
    }
}

#[test]
fn test_order_book_level_size() {
    assert_eq!(std::mem::size_of::<OrderBookLevel>(), ZEN_CACHE_LINE_SIZE);
}

/// Cache-isolated trade execution record
/// Each trade record is on its own cache line for parallel processing
#[repr(C)]
#[repr(align(64))]
pub struct TradeRecord {
    pub trade_id: u64,        // 8 bytes
    pub price: u64,           // 8 bytes
    pub volume: u64,          // 8 bytes
    pub side: u8,             // 1 byte (0=bid, 1=ask)
    pub maker_order_id: u64,  // 8 bytes
    pub taker_order_id: u64,  // 8 bytes
    pub timestamp_ns: u64,    // 8 bytes
    // Padding: 64 - 53 = 11
    _padding: [u8; 11],
}

impl TradeRecord {
    #[inline(always)]
    pub fn new(trade_id: u64, price: u64, volume: u64, side: u8) -> Self {
        Self {
            trade_id,
            price,
            volume,
            side,
            maker_order_id: 0,
            taker_order_id: 0,
            timestamp_ns: 0,
            _padding: [0u8; 11],
        }
    }
}

// =============================================================================
// Ring Buffer with Cache-Aware Sizing
// =============================================================================

/// Cache-aware ring buffer for tick data
/// Size is chosen to fit exactly in L2 cache for optimal performance
pub struct TickRingBuffer<T> {
    data: Vec<CacheAligned<T>>,
    capacity: usize,
    head: CacheAlignedAtomicUsize,
    tail: CacheAlignedAtomicUsize,
}

impl<T: Default + Clone> TickRingBuffer<T> {
    /// Create a new ring buffer with capacity optimized for L2 cache
    pub fn with_l2_optimized_capacity() -> Self {
        // Calculate optimal capacity to fit in L2 cache
        // Reserve space for metadata and leave room for other data
        let available_l2 = ZEN_L2_SIZE / 4; // Use 1/4 of L2
        let element_size = std::mem::size_of::<CacheAligned<T>>();
        let capacity = available_l2 / element_size;
        
        Self::with_capacity(capacity)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        
        for _ in 0..capacity {
            data.push(CacheAligned::new(T::default()));
        }
        
        Self {
            data,
            capacity,
            head: CacheAlignedAtomicUsize::new(0),
            tail: CacheAlignedAtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn push(&self, item: T) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        
        // Check if buffer is full
        if tail.wrapping_sub(head) >= self.capacity {
            // Buffer full, overwrite oldest
            let overwritten = self.data[tail % self.capacity].get().clone();
            self.data[tail % self.capacity] = CacheAligned::new(item);
            self.tail.store(tail.wrapping_add(1), Ordering::Relaxed);
            Some(overwritten)
        } else {
            self.data[tail % self.capacity] = CacheAligned::new(item);
            self.tail.store(tail.wrapping_add(1), Ordering::Relaxed);
            None
        }
    }

    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        if head == tail {
            return None; // Buffer empty
        }
        
        let item = self.data[head % self.capacity].get().clone();
        self.head.store(head.wrapping_add(1), Ordering::Relaxed);
        Some(item)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        tail.wrapping_sub(head)
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// =============================================================================
// Compile-Time Cache Line Assertions
// =============================================================================

/// Macro to assert at compile time that a type is cache-line aligned
#[macro_export]
macro_rules! assert_cache_aligned {
    ($type:ty) => {
        const _: () = {
            assert!(
                std::mem::align_of::<$type>() >= $crate::amd::zen_cache::ZEN_CACHE_LINE_SIZE,
                concat!(
                    "Type ",
                    stringify!($type),
                    " is not cache-line aligned!"
                ),
            );
        };
    };
}

/// Macro to assert at compile time that a type fits in a single cache line
#[macro_export]
macro_rules! assert_fits_in_cache_line {
    ($type:ty) => {
        const _: () = {
            assert!(
                std::mem::size_of::<$type>() <= $crate::amd::zen_cache::ZEN_CACHE_LINE_SIZE,
                concat!(
                    "Type ",
                    stringify!($type),
                    " does not fit in a single cache line!"
                ),
            );
        };
    };
}

// Example usage with compile-time verification
assert_cache_aligned!(OrderBookLevel);
assert_fits_in_cache_line!(OrderBookLevel);
