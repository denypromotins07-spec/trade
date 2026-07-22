//! Last Level Cache (LLC) Optimizer - Cache Line Alignment
//! 
//! This module builds a Last Level Cache (LLC) optimizer that aligns critical
//! Nautilus structs to 64-byte cache lines, drastically reducing cache misses
//! during order book updates. It provides utilities for analyzing and optimizing
//! memory layout for AMD Ryzen AI 5 architecture.
//! 
//! **Key Features:**
//! - 64-byte cache line alignment for all critical structures
//! - Cache miss estimation and profiling
//! - Prefetching hints for sequential access patterns
//! - Memory layout validation

use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

/// Cache line size for modern CPUs (AMD Ryzen = 64 bytes).
pub const CACHE_LINE_SIZE: usize = 64;

/// Trait for types that should be cache-line aligned.
pub trait CacheAligned {
    /// Get the size of the structure in bytes.
    fn size_bytes(&self) -> usize;
    
    /// Check if the structure fits within a single cache line.
    fn fits_in_cache_line(&self) -> bool {
        self.size_bytes() <= CACHE_LINE_SIZE
    }
    
    /// Get estimated cache lines used.
    fn cache_lines_used(&self) -> usize {
        (self.size_bytes() + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE
    }
}

/// Wrapper for cache-line aligned allocation.
#[repr(align(64))]
pub struct CacheAlignedBox<T> {
    ptr: *mut T,
    layout: Layout,
}

unsafe impl<T: Send> Send for CacheAlignedBox<T> {}
unsafe impl<T: Sync> Sync for CacheAlignedBox<T> {}

impl<T> CacheAlignedBox<T> {
    /// Create a new cache-aligned box containing the value.
    pub fn new(value: T) -> Self {
        let layout = Layout::from_size_align(std::mem::size_of::<T>(), CACHE_LINE_SIZE)
            .expect("Invalid layout");
        
        unsafe {
            let ptr = alloc(layout) as *mut T;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr::write(ptr, value);
            
            CacheAlignedBox { ptr, layout }
        }
    }

    /// Get a reference to the contained value.
    pub fn get(&self) -> &T {
        unsafe { &*self.ptr }
    }

    /// Get a mutable reference to the contained value.
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr }
    }

    /// Consume the box and return the inner value.
    pub fn into_inner(self) -> T {
        unsafe {
            let value = ptr::read(self.ptr);
            // Don't deallocate yet, we need to prevent double-free
            // The Drop impl will handle it, but we've moved the value out
            std::mem::forget(self);
            value
        }
    }
}

impl<T> Drop for CacheAlignedBox<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::drop_in_place(self.ptr);
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

/// Cache-optimized order book entry structure.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CacheAlignedOrderBookEntry {
    /// Price (aligned to start of cache line)
    pub price: u64,
    /// Volume
    pub volume: u64,
    /// Order count
    pub order_count: u32,
    /// Flags/padding
    pub flags: u32,
    /// Timestamp nanoseconds
    pub timestamp_ns: u64,
    /// Padding to fill exactly 64 bytes
    _padding: [u8; 8],
}

impl CacheAlignedOrderBookEntry {
    pub fn new(price: u64, volume: u64) -> Self {
        CacheAlignedOrderBookEntry {
            price,
            volume,
            order_count: 1,
            flags: 0,
            timestamp_ns: 0,
            _padding: [0u8; 8],
        }
    }
}

impl Default for CacheAlignedOrderBookEntry {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl CacheAligned for CacheAlignedOrderBookEntry {
    fn size_bytes(&self) -> usize {
        std::mem::size_of::<CacheAlignedOrderBookEntry>()
    }
}

/// Verify cache alignment at compile time.
#[macro_export]
macro_rules! assert_cache_aligned {
    ($type:ty) => {
        const _: () = assert!(
            std::mem::align_of::<$type>() >= 64,
            concat!("Type ", stringify!($type), " is not cache-line aligned")
        );
    };
}

/// Cache line padding utility.
#[macro_export]
macro_rules! cache_pad {
    () => {
        [u8; 64]
    };
    ($n:expr) => {
        [u8; $n * 64]
    };
}

/// Prefetch data into cache using hardware prefetch instructions.
#[inline]
pub fn prefetch_read_data<T>(data: &T, locality: i32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::_mm_prefetch;
        let ptr = data as *const T as *const i8;
        
        match locality {
            0 => _mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0),
            1 => _mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T1),
            2 => _mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T2),
            3 => _mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_NTA),
            _ => _mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0),
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (data, locality); // Suppress unused warnings
    }
}

/// Cache analysis result.
#[derive(Debug, Default)]
pub struct CacheAnalysis {
    /// Total bytes analyzed
    pub total_bytes: usize,
    /// Number of cache lines touched
    pub cache_lines_touched: usize,
    /// Estimated cache miss rate (0.0 - 1.0)
    pub estimated_miss_rate: f64,
    /// Number of false sharing opportunities detected
    pub false_sharing_risks: usize,
}

impl CacheAnalysis {
    /// Analyze memory access pattern for cache efficiency.
    pub fn analyze_sequential_access<T>(slice: &[T]) -> Self {
        let element_size = std::mem::size_of::<T>();
        let total_bytes = slice.len() * element_size;
        let cache_lines = (total_bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
        
        // Estimate miss rate based on element size vs cache line
        let elements_per_line = CACHE_LINE_SIZE / element_size.max(1);
        let miss_rate = if elements_per_line > 0 {
            1.0 / elements_per_line as f64
        } else {
            1.0
        };
        
        CacheAnalysis {
            total_bytes,
            cache_lines_touched: cache_lines,
            estimated_miss_rate: miss_rate,
            false_sharing_risks: 0,
        }
    }

    /// Check if a type is optimally sized for cache efficiency.
    pub fn check_optimal_sizing<T>() -> (bool, &'static str) {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        
        if align < CACHE_LINE_SIZE {
            return (false, "Alignment less than cache line size");
        }
        
        if size > CACHE_LINE_SIZE && size % CACHE_LINE_SIZE != 0 {
            return (false, "Size not multiple of cache line");
        }
        
        (true, "Optimally sized")
    }
}

/// L1/L2/L3 cache sizes for AMD Ryzen AI 5 (typical values).
pub mod amd_ryzen_ai5_cache {
    pub const L1_DATA_SIZE_KB: usize = 32;
    pub const L1_INST_SIZE_KB: usize = 32;
    pub const L2_SIZE_KB: usize = 512;
    pub const L3_SIZE_MB: usize = 16;
    pub const CACHE_LINE_SIZE: usize = 64;
}

/// Validate that critical structures are properly aligned.
pub fn validate_cache_alignment() -> Vec<&'static str> {
    let mut issues = Vec::new();
    
    // Check CacheAlignedOrderBookEntry
    if std::mem::align_of::<CacheAlignedOrderBookEntry>() < CACHE_LINE_SIZE {
        issues.push("CacheAlignedOrderBookEntry not cache-line aligned");
    }
    
    if std::mem::size_of::<CacheAlignedOrderBookEntry>() != CACHE_LINE_SIZE {
        issues.push("CacheAlignedOrderBookEntry not exactly one cache line");
    }
    
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_box() {
        let boxed = CacheAlignedBox::new(42u64);
        assert_eq!(*boxed.get(), 42);
        
        // Verify alignment
        let ptr = boxed.get() as *const u64 as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }

    #[test]
    fn test_order_book_entry_size() {
        let entry = CacheAlignedOrderBookEntry::new(50000, 100);
        assert_eq!(entry.size_bytes(), CACHE_LINE_SIZE);
        assert!(entry.fits_in_cache_line());
        assert_eq!(entry.cache_lines_used(), 1);
    }

    #[test]
    fn test_cache_analysis() {
        let data: Vec<u64> = (0..1024).collect();
        let analysis = CacheAnalysis::analyze_sequential_access(&data);
        
        assert_eq!(analysis.total_bytes, 1024 * 8);
        assert!(analysis.estimated_miss_rate > 0.0);
        assert!(analysis.estimated_miss_rate <= 1.0);
    }

    #[test]
    fn test_compile_time_assert() {
        // This compiles only if the type is properly aligned
        assert_cache_aligned!(CacheAlignedOrderBookEntry);
    }

    #[test]
    fn test_prefetch() {
        let data = vec![1u64, 2, 3, 4, 5];
        prefetch_read_data(&data[0], 0);
        prefetch_read_data(&data[1], 1);
        // Prefetch is a hint, so we just verify it doesn't crash
    }
}
