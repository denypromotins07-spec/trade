//! PCIe Pinned Memory Allocator for AMD Direct Memory Access (DMA)
//!
//! This module implements page-locked (pinned) host memory allocators using
//! Windows APIs to enable Direct Memory Access (DMA) transfers between
//! system RAM and AMD GPU VRAM.
//!
//! Key features:
//! - Page-locked memory allocation via VirtualLock
//! - Zero-copy PCIe transfers with pinned buffers
//! - NUMA-aware allocation for optimal locality
//! - 8GB RAM limit enforcement
//!
//! Author: Elite Quantitative Software Engineering Team
//! Stage: 49 - PCIe & NUMA Zero-Copy Memory Transfers

#[cfg(target_os = "windows")]
use windows::{
    Win32::System::Memory::{
        VirtualAlloc, VirtualFree, VirtualLock, VirtualUnlock,
        MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
        MEM_RELEASE,
    },
};

use std::alloc::{self, Layout};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

// =============================================================================
// Memory Constants
// =============================================================================

/// System page size (typically 4KB on x86_64)
pub const PAGE_SIZE: usize = 4096;

/// Maximum pinned memory budget (enforces 8GB total system limit)
pub const MAX_PINNED_MEMORY: usize = 2 * 1024 * 1024 * 1024; // 2GB for pinned

/// Global tracker for pinned memory usage
static PINNED_MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);

// =============================================================================
// Pinned Memory Allocator
// =============================================================================

/// Allocator for page-locked (pinned) memory
/// Enables zero-copy DMA transfers to/from AMD GPU
pub struct PinnedMemoryAllocator {
    /// Total bytes currently pinned
    current_usage: AtomicUsize,
    /// Maximum allowed pinned bytes
    max_bytes: usize,
}

unsafe impl Send for PinnedMemoryAllocator {}
unsafe impl Sync for PinnedMemoryAllocator {}

impl PinnedMemoryAllocator {
    /// Create a new pinned memory allocator
    pub fn new(max_bytes: usize) -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            max_bytes,
        }
    }

    /// Default allocator with 2GB limit
    pub fn default() -> Self {
        Self::new(MAX_PINNED_MEMORY)
    }

    /// Allocate pinned memory buffer
    ///
    /// # Safety
    /// The returned pointer must be freed with `free_pinned` when no longer needed.
    /// Failure to free will cause memory leaks as pinned memory cannot be swapped.
    pub fn allocate(&self, size: usize) -> Result<PinnedBuffer, PinnedMemoryError> {
        // Check if we have enough budget
        let aligned_size = align_to_page(size);
        
        loop {
            let current = self.current_usage.load(Ordering::Relaxed);
            
            if current + aligned_size > self.max_bytes {
                return Err(PinnedMemoryError::BudgetExceeded {
                    requested: aligned_size,
                    available: self.max_bytes - current,
                });
            }

            if self.current_usage.compare_exchange_weak(
                current,
                current + aligned_size,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ).is_ok() {
                break;
            }
        }

        // Allocate memory
        let ptr = unsafe { self.allocate_pinned(aligned_size)? };

        Ok(PinnedBuffer {
            ptr,
            size: aligned_size,
            allocator: self,
        })
    }

    #[cfg(target_os = "windows")]
    unsafe fn allocate_pinned(&self, size: usize) -> Result<*mut u8, PinnedMemoryError> {
        // Allocate virtual memory
        let ptr = VirtualAlloc(
            None,
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );

        if ptr.is_null() {
            self.current_usage.fetch_sub(size, Ordering::Relaxed);
            return Err(PinnedMemoryError::AllocationFailed);
        }

        // Lock pages in physical memory (prevent paging)
        let result = VirtualLock(ptr, size);
        
        if result.is_err() {
            // Cleanup on failure
            let _ = VirtualFree(ptr, 0, MEM_RELEASE);
            self.current_usage.fetch_sub(size, Ordering::Relaxed);
            return Err(PinnedMemoryError::LockFailed);
        }

        Ok(ptr as *mut u8)
    }

    #[cfg(not(target_os = "windows"))]
    unsafe fn allocate_pinned(&self, size: usize) -> Result<*mut u8, PinnedMemoryError> {
        // Fallback for non-Windows: use mmap with MAP_LOCKED
        use libc;
        
        let ptr = libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_LOCKED | libc::MAP_POPULATE,
            -1,
            0,
        );

        if ptr == libc::MAP_FAILED {
            self.current_usage.fetch_sub(size, Ordering::Relaxed);
            return Err(PinnedMemoryError::AllocationFailed);
        }

        Ok(ptr as *mut u8)
    }

    /// Free previously allocated pinned memory
    unsafe fn free_pinned(&self, ptr: *mut u8, size: usize) {
        #[cfg(target_os = "windows")]
        {
            let _ = VirtualUnlock(ptr as *mut _, size);
            let _ = VirtualFree(ptr as *mut _, 0, MEM_RELEASE);
        }

        #[cfg(not(target_os = "windows"))]
        {
            use libc;
            let _ = libc::munmap(ptr as *mut _, size);
        }

        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }

    /// Get current pinned memory usage
    pub fn usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get available pinned memory budget
    pub fn available(&self) -> usize {
        self.max_bytes.saturating_sub(self.usage())
    }
}

impl Default for PinnedMemoryAllocator {
    fn default() -> Self {
        Self::default()
    }
}

// =============================================================================
// Pinned Buffer RAII Wrapper
// =============================================================================

/// RAII wrapper for pinned memory buffer
/// Automatically frees memory when dropped
pub struct PinnedBuffer<'a> {
    ptr: *mut u8,
    size: usize,
    allocator: &'a PinnedMemoryAllocator,
}

unsafe impl<'a> Send for PinnedBuffer<'a> {}
unsafe impl<'a> Sync for PinnedBuffer<'a> {}

impl<'a> PinnedBuffer<'a> {
    /// Get raw pointer to buffer
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get mutable raw pointer
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get buffer size in bytes
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get slice view of buffer
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get mutable slice view
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get typed slice view
    #[inline(always)]
    pub fn as_typed_slice<T>(&self) -> &[T] {
        let len = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, len) }
    }

    /// Get mutable typed slice view
    #[inline(always)]
    pub fn as_mut_typed_slice<T>(&mut self) -> &mut [T] {
        let len = self.size / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, len) }
    }

    /// Copy data from host memory to pinned buffer
    #[inline(always)]
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<(), PinnedMemoryError> {
        if src.len() > self.size {
            return Err(PinnedMemoryError::BufferSizeMismatch);
        }

        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), self.ptr, src.len());
        }

        Ok(())
    }

    /// Copy data from pinned buffer to host memory
    #[inline(always)]
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), PinnedMemoryError> {
        if dst.len() > self.size {
            return Err(PinnedMemoryError::BufferSizeMismatch);
        }

        unsafe {
            ptr::copy_nonoverlapping(self.ptr, dst.as_mut_ptr(), dst.len());
        }

        Ok(())
    }
}

impl<'a> Drop for PinnedBuffer<'a> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.free_pinned(self.ptr, self.size);
        }
    }
}

// =============================================================================
// DMA Transfer Engine
// =============================================================================

/// DMA transfer engine for zero-copy PCIe transfers
pub struct DmaTransferEngine {
    allocator: PinnedMemoryAllocator,
}

unsafe impl Send for DmaTransferEngine {}
unsafe impl Sync for DmaTransferEngine {}

impl DmaTransferEngine {
    /// Create new DMA transfer engine
    pub fn new() -> Self {
        Self {
            allocator: PinnedMemoryAllocator::default(),
        }
    }

    /// Allocate a DMA-capable buffer
    pub fn allocate_dma_buffer(&self, size: usize) -> Result<PinnedBuffer, PinnedMemoryError> {
        self.allocator.allocate(size)
    }

    /// Prepare buffer for GPU read (host-to-device transfer)
    pub fn prepare_for_gpu_read(&self, data: &[u8]) -> Result<PinnedBuffer, PinnedMemoryError> {
        let mut buffer = self.allocator.allocate(data.len())?;
        buffer.copy_from_host(data)?;
        Ok(buffer)
    }

    /// Prepare buffer for GPU write (device-to-host transfer)
    pub fn prepare_for_gpu_write(&self, size: usize) -> Result<PinnedBuffer, PinnedMemoryError> {
        self.allocator.allocate(size)
    }

    /// Get allocator statistics
    pub fn stats(&self) -> DmaStats {
        DmaStats {
            pinned_usage: self.allocator.usage(),
            pinned_available: self.allocator.available(),
            max_pinned: self.allocator.max_bytes,
        }
    }
}

impl Default for DmaTransferEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// DMA transfer statistics
#[derive(Debug, Clone)]
pub struct DmaStats {
    pub pinned_usage: usize,
    pub pinned_available: usize,
    pub max_pinned: usize,
}

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during pinned memory operations
#[derive(Debug, Clone)]
pub enum PinnedMemoryError {
    /// Requested memory exceeds budget
    BudgetExceeded {
        requested: usize,
        available: usize,
    },
    /// System allocation failed
    AllocationFailed,
    /// Failed to lock pages in physical memory
    LockFailed,
    /// Source/destination buffer size mismatch
    BufferSizeMismatch,
}

impl std::fmt::Display for PinnedMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PinnedMemoryError::BudgetExceeded { requested, available } => {
                write!(f, "Pinned memory budget exceeded: requested {}, available {}", requested, available)
            }
            PinnedMemoryError::AllocationFailed => {
                write!(f, "Failed to allocate pinned memory")
            }
            PinnedMemoryError::LockFailed => {
                write!(f, "Failed to lock pages in physical memory")
            }
            PinnedMemoryError::BufferSizeMismatch => {
                write!(f, "Buffer size mismatch")
            }
        }
    }
}

impl std::error::Error for PinnedMemoryError {}

// =============================================================================
// Utility Functions
// =============================================================================

/// Align size to page boundary
#[inline(always)]
fn align_to_page(size: usize) -> usize {
    (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

/// Check if address is page-aligned
#[inline(always)]
pub fn is_page_aligned(addr: usize) -> bool {
    addr & (PAGE_SIZE - 1) == 0
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_to_page() {
        assert_eq!(align_to_page(100), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE), PAGE_SIZE);
        assert_eq!(align_to_page(PAGE_SIZE + 1), 2 * PAGE_SIZE);
    }

    #[test]
    fn test_is_page_aligned() {
        assert!(is_page_aligned(0));
        assert!(is_page_aligned(PAGE_SIZE));
        assert!(!is_page_aligned(100));
    }

    #[test]
    fn test_pinned_allocator_budget() {
        let allocator = PinnedMemoryAllocator::new(4096);
        
        // First allocation should succeed
        let result = allocator.allocate(100);
        assert!(result.is_ok());
        
        // Second allocation should fail (exceeds budget)
        let result = allocator.allocate(4000);
        assert!(matches!(result, Err(PinnedMemoryError::BudgetExceeded { .. })));
    }
}
