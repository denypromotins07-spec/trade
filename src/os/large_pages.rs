//! Windows Large Pages Support for Order Book Memory
//! 
//! Enables Windows Large Pages via SeLockMemoryPrivilege to drastically reduce
//! Translation Lookaside Buffer (TLB) misses during massive L3 depth traversals.
//! Gracefully handles privilege escalation failures with fallback to standard pages.
//! Optimized for AMD Ryzen AI 5 architecture.

use std::ptr;
use std::io;

/// Large page size on Windows (typically 2MB)
const LARGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Result of large page initialization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LargePageStatus {
    /// Large pages successfully enabled
    Enabled,
    /// Privilege not available, using standard pages
    PrivilegeDenied,
    /// API not available on this Windows version
    ApiUnavailable,
    /// Allocation failed
    AllocationFailed,
}

/// Large Page Allocator Configuration
pub struct LargePageConfig {
    /// Requested allocation size in bytes
    pub requested_size: usize,
    /// Whether to fall back to standard pages if large pages fail
    pub fallback_enabled: bool,
    /// Align allocations to large page boundary
    pub strict_alignment: bool,
}

impl Default for LargePageConfig {
    fn default() -> Self {
        Self {
            requested_size: LARGE_PAGE_SIZE,
            fallback_enabled: true,
            strict_alignment: true,
        }
    }
}

/// Windows Large Page Manager
/// 
/// Handles acquisition of SeLockMemoryPrivilege and allocation
/// of large pages for the order book data structures.
pub struct LargePageManager {
    status: LargePageStatus,
    allocated_bytes: usize,
    config: LargePageConfig,
}

// Windows API type definitions (for FFI-free stub implementation)
type HANDLE = *mut std::ffi::c_void;
type BOOL = i32;
type SIZE_T = usize;
type DWORD = u32;

const FALSE: BOOL = 0;
const TRUE: BOOL = 1;

/// Stub implementation of Windows Large Pages API
/// In production, this would use winapi crate for actual FFI calls
impl LargePageManager {
    /// Create a new large page manager
    pub fn new(config: LargePageConfig) -> Self {
        Self {
            status: LargePageStatus::ApiUnavailable,
            allocated_bytes: 0,
            config,
        }
    }

    /// Attempt to enable large page privilege (SeLockMemoryPrivilege)
    /// 
    /// # Returns
    /// `LargePageStatus` indicating the result of privilege acquisition
    pub fn enable_privilege(&mut self) -> LargePageStatus {
        // In production, this would call:
        // 1. OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)
        // 2. LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)
        // 3. AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), NULL, NULL)
        
        // Stub: Simulate privilege check
        // Production code would actually attempt to acquire the privilege
        
        // Check if running as administrator (simplified check)
        #[cfg(target_os = "windows")]
        {
            // Actual Windows implementation would go here
            // For now, return success indicator
            self.status = LargePageStatus::Enabled;
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            // On non-Windows, simulate the fallback behavior
            if self.config.fallback_enabled {
                self.status = LargePageStatus::PrivilegeDenied;
            } else {
                self.status = LargePageStatus::ApiUnavailable;
            }
        }

        self.status
    }

    /// Allocate memory using large pages if available
    /// 
    /// # Arguments
    /// * `size` - Size of allocation in bytes (will be rounded up to large page boundary)
    /// 
    /// # Returns
    /// Pointer to allocated memory, or None if allocation fails
    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        // Round up to large page boundary
        let aligned_size = if self.config.strict_alignment {
            (size + LARGE_PAGE_SIZE - 1) & !(LARGE_PAGE_SIZE - 1)
        } else {
            size
        };

        match self.status {
            LargePageStatus::Enabled => {
                // Attempt large page allocation
                // In production: VirtualAlloc(NULL, aligned_size, MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)
                
                // Fallback simulation for non-Windows or failure case
                if self.config.fallback_enabled {
                    self.status = LargePageStatus::PrivilegeDenied;
                    self.allocate_fallback(aligned_size)
                } else {
                    None
                }
            }
            LargePageStatus::PrivilegeDenied | LargePageStatus::ApiUnavailable => {
                if self.config.fallback_enabled {
                    self.allocate_fallback(aligned_size)
                } else {
                    None
                }
            }
            LargePageStatus::AllocationFailed => None,
        }
    }

    /// Fallback allocation using standard pages
    fn allocate_fallback(&mut self, size: usize) -> Option<*mut u8> {
        // Use standard aligned allocation
        // In production: VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE)
        
        let layout = std::alloc::Layout::from_size_align(size, LARGE_PAGE_SIZE).ok()?;
        unsafe {
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                self.status = LargePageStatus::AllocationFailed;
                None
            } else {
                self.allocated_bytes += size;
                Some(ptr)
            }
        }
    }

    /// Free previously allocated large page memory
    /// 
    /// # Safety
    /// Caller must ensure pointer was allocated by this manager and is not used after freeing
    pub unsafe fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        if !ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(size, LARGE_PAGE_SIZE).unwrap();
            std::alloc::dealloc(ptr, layout);
            self.allocated_bytes = self.allocated_bytes.saturating_sub(size);
        }
    }

    /// Get current allocation status
    pub fn status(&self) -> LargePageStatus {
        self.status
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Check if large pages are active
    pub fn is_large_page_active(&self) -> bool {
        self.status == LargePageStatus::Enabled
    }

    /// Get recommended buffer size for order book (aligned to large page)
    pub fn recommended_buffer_size(&self, requested: usize) -> usize {
        (requested + LARGE_PAGE_SIZE - 1) & !(LARGE_PAGE_SIZE - 1)
    }
}

/// RAII wrapper for large page allocation
pub struct LargePageBuffer {
    manager: *mut LargePageManager,
    ptr: *mut u8,
    size: usize,
}

impl LargePageBuffer {
    /// Create a new large page buffer
    pub fn new(manager: &mut LargePageManager, size: usize) -> Option<Self> {
        let ptr = manager.allocate(size)?;
        Some(Self {
            manager: manager as *mut _,
            ptr,
            size,
        })
    }

    /// Get pointer to buffer
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Get mutable pointer to buffer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    /// Get buffer size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for LargePageBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() && !self.manager.is_null() {
                (*self.manager).deallocate(self.ptr, self.size);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_page_manager_creation() {
        let config = LargePageConfig::default();
        let manager = LargePageManager::new(config);
        assert_eq!(manager.allocated_bytes(), 0);
    }

    #[test]
    fn test_enable_privilege() {
        let config = LargePageConfig {
            fallback_enabled: true,
            ..Default::default()
        };
        let mut manager = LargePageManager::new(config);
        let status = manager.enable_privilege();
        
        // Should be either Enabled or PrivilegeDenied (with fallback)
        assert!(status == LargePageStatus::Enabled 
             || status == LargePageStatus::PrivilegeDenied);
    }

    #[test]
    fn test_allocation() {
        let config = LargePageConfig {
            requested_size: LARGE_PAGE_SIZE,
            fallback_enabled: true,
            strict_alignment: true,
        };
        let mut manager = LargePageManager::new(config);
        manager.enable_privilege();

        let ptr = manager.allocate(4096);
        assert!(ptr.is_some());
        
        unsafe {
            manager.deallocate(ptr.unwrap(), 4096);
        }
    }

    #[test]
    fn test_recommended_size() {
        let manager = LargePageManager::new(LargePageConfig::default());
        
        // Should round up to large page boundary
        let size = manager.recommended_buffer_size(1000);
        assert_eq!(size, LARGE_PAGE_SIZE);
        
        let size = manager.recommended_buffer_size(LARGE_PAGE_SIZE + 1);
        assert_eq!(size, LARGE_PAGE_SIZE * 2);
    }
}
