// =============================================================================
// Nautilus/Ray Bot - Stage 53: Pre-Warm Routine
// File: src/boot/pre_warm.rs
// Purpose: Flood L1/L2 caches and pre-allocate memory arenas during /START
//          to ensure zero page faults on the first tick.
// Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
// Constraints: 8GB RAM Limit, 4GB Python Quota, Microsecond Latency Focus
// =============================================================================

use std::alloc::{GlobalAlloc, System};
use std::hint;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Total system RAM limit enforced by the bot
const SYSTEM_RAM_LIMIT_GB: usize = 8;
/// Reserved RAM for Python/Ray workers (strict quota)
const PYTHON_RAM_QUOTA_GB: usize = 4;
/// Cache line size for AMD Zen architecture
const CACHE_LINE_SIZE: usize = 64;

/// Memory Arena descriptor
pub struct MemoryArena {
    pub ptr: *mut u8,
    pub size: usize,
    pub initialized: bool,
}

unsafe impl Send for MemoryArena {}
unsafe impl Sync for MemoryArena {}

/// Pre-warming manager for boot-to-trade optimization
pub struct PreWarmManager {
    /// L1 Data Cache size (typically 32KB per core on Zen 4)
    l1_cache_size: usize,
    /// L2 Cache size (typically 1MB per core on Zen 4)
    l2_cache_size: usize,
    /// L3 Cache slice size (varies by CCD)
    l3_cache_size: usize,
    /// Allocated arenas
    arenas: Vec<MemoryArena>,
    /// Total allocated memory
    allocated_bytes: AtomicUsize,
}

impl PreWarmManager {
    pub fn new() -> Self {
        // AMD Ryzen AI 5 Cache Topology (Approximate)
        Self {
            l1_cache_size: 32 * 1024,       // 32 KB
            l2_cache_size: 1024 * 1024,     // 1 MB
            l3_cache_size: 16 * 1024 * 1024,// 16 MB (shared per CCD)
            arenas: Vec::with_capacity(16),
            allocated_bytes: AtomicUsize::new(0),
        }
    }

    /// Execute the full pre-warming sequence
    pub fn execute_pre_warm(&mut self) -> Result<(), String> {
        log::info!("=== INITIATING PRE-WARM SEQUENCE ===");
        
        let max_memory = (SYSTEM_RAM_LIMIT_GB - PYTHON_RAM_QUOTA_GB) * 1024 * 1024 * 1024;
        log::info!("Available memory for Rust engine: {} GB", max_memory / (1024 * 1024 * 1024));
        
        // Step 1: Touch L1 Cache lines
        self.warm_l1_cache()?;
        
        // Step 2: Touch L2 Cache lines
        self.warm_l2_cache()?;
        
        // Step 3: Allocate and touch main Order Book Arena
        self.allocate_main_arena(max_memory)?;
        
        // Step 4: Prefetch critical data structures
        self.prefetch_critical_structs();
        
        log::info!("=== PRE-WARM SEQUENCE COMPLETE ===");
        log::info!("Total memory pre-allocated: {} MB", self.allocated_bytes.load(Ordering::Relaxed) / (1024 * 1024));
        
        Ok(())
    }

    /// Warm up L1 cache by accessing a contiguous block
    fn warm_l1_cache(&self) -> Result<(), String> {
        log::debug!("Warming L1 Cache ({} bytes)...", self.l1_cache_size);
        
        let mut buffer = vec![0u8; self.l1_cache_size];
        
        // Write pattern to force allocation and TLB fill
        for i in 0..buffer.len() {
            buffer[i] = (i % 256) as u8;
        }
        
        // Read pattern to bring into L1
        let mut sum: u64 = 0;
        for i in 0..buffer.len() {
            sum += buffer[i] as u64;
        }
        
        // Prevent compiler from optimizing away
        hint::black_box(sum);
        drop(buffer);
        
        log::debug!("L1 Cache warmed.");
        Ok(())
    }

    /// Warm up L2 cache
    fn warm_l2_cache(&self) -> Result<(), String> {
        log::debug!("Warming L2 Cache ({} bytes)...", self.l2_cache_size);
        
        let mut buffer = vec![0u8; self.l2_cache_size];
        
        // Strided access to ensure all cache lines are touched
        let stride = CACHE_LINE_SIZE;
        for i in (0..buffer.len()).step_by(stride) {
            buffer[i] = 0xAA;
        }
        
        // Reverse pass
        for i in (0..buffer.len()).rev().step_by(stride) {
            buffer[i] = 0x55;
        }
        
        hint::black_box(&buffer);
        drop(buffer);
        
        log::debug!("L2 Cache warmed.");
        Ok(())
    }

    /// Allocate the main order book arena with Huge Pages if possible
    fn allocate_main_arena(&mut self, max_bytes: usize) -> Result<(), String> {
        log::info!("Allocating main order book arena...");
        
        // Reserve 50% of available Rust memory for the main book
        let arena_size = (max_bytes as f64 * 0.5) as usize;
        
        // Align to 2MB boundary for Huge Page compatibility
        let aligned_size = ((arena_size + 0x1FFFFF) & !0x1FFFFF);
        
        unsafe {
            // Attempt to allocate using standard allocator (Huge Page hinting requires OS specific calls)
            let layout = std::alloc::Layout::from_size_align(aligned_size, 0x200000)?;
            let ptr = System.alloc(layout);
            
            if ptr.is_null() {
                return Err("Failed to allocate main arena".to_string());
            }
            
            // TOUCH EVERY PAGE to ensure physical mapping and zero-filling
            // This prevents page faults during live trading
            let slice = std::slice::from_raw_parts_mut(ptr, aligned_size);
            
            // Use non-temporal stores if available to avoid polluting cache with init data
            // But for pre-warm, we WANT it in cache initially
            for i in (0..aligned_size).step_by(CACHE_LINE_SIZE) {
                ptr::write_volatile(slice.as_mut_ptr().add(i), 0x01);
            }
            
            // Memory barrier to ensure writes are complete
            std::sync::atomic::fence(Ordering::SeqCst);
            
            self.arenas.push(MemoryArena {
                ptr,
                size: aligned_size,
                initialized: true,
            });
            
            self.allocated_bytes.fetch_add(aligned_size, Ordering::Relaxed);
        }
        
        log::info!("Main arena allocated: {} MB", aligned_size / (1024 * 1024));
        Ok(())
    }

    /// Prefetch critical structures
    fn prefetch_critical_structs(&self) {
        log::debug!("Prefetching critical data structures...");
        
        // In a real implementation, this would issue PREFETCHT0 instructions
        // for the head of the order book, matching engine state, etc.
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Example: Prefetch address of the main arena
            if let Some(arena) = self.arenas.first() {
                // _mm_prefetch((const char*)arena->ptr, _MM_HINT_T0);
                // Intrinsics require std::arch::x86_64
                use std::arch::x86_64::_mm_prefetch;
                use std::arch::x86_64::_MM_HINT_T0;
                
                _mm_prefetch(arena.ptr as *const i8, _MM_HINT_T0);
            }
        }
        
        log::debug!("Critical structures prefetched.");
    }

    /// Cleanup arenas on shutdown
    pub fn cleanup(&mut self) {
        log::warn!("Cleaning up pre-warmed arenas...");
        
        unsafe {
            for arena in self.arenas.drain(..) {
                if !arena.ptr.is_null() {
                    let layout = std::alloc::Layout::from_size_align_unchecked(
                        arena.size, 
                        0x200000
                    );
                    System.dealloc(arena.ptr, layout);
                }
            }
        }
        
        self.allocated_bytes.store(0, Ordering::Relaxed);
        log::info!("Memory cleanup complete.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prewarm_allocation() {
        let mut manager = PreWarmManager::new();
        // Use smaller size for test
        if let Err(e) = manager.allocate_main_arena(10 * 1024 * 1024) {
            panic!("Allocation failed: {}", e);
        }
        assert!(manager.allocated_bytes.load(Ordering::Relaxed) > 0);
        manager.cleanup();
    }
}
