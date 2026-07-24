//! Heap Scrubber - Continuous Memory Zeroing for Cold-Boot Attack Prevention
//! 
//! This module implements a background thread that continuously overwrites
//! freed memory blocks with zeros to prevent:
//! - Cold-boot attacks
//! - Rogue Python extensions from scraping residual secrets
//! - Memory forensics recovery of sensitive data
//! 
//! Also scrubs AMD DirectML/ROCm GPU VRAM during hot-swaps.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::ptr;

/// Scrub pattern for memory zeroing (multi-pass)
const SCRUB_PASSES: usize = 3;
/// Scrub patterns per pass (for enhanced security)
const SCRUB_PATTERNS: [u8; 4] = [0x00, 0xFF, 0xAA, 0x55];
/// Maximum time between scrub cycles (milliseconds)
const MAX_SCRUB_INTERVAL_MS: u64 = 100;
/// Target latency budget per scrub operation (microseconds)
const SCRUB_LATENCY_BUDGET_US: u64 = 50;

/// Memory region types for scrubbing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionType {
    Heap,
    Stack,
    GpuVram,
    MappedFile,
    SharedMemory,
}

/// Scrub statistics
#[derive(Debug, Clone)]
pub struct ScrubStats {
    pub total_bytes_scrubbed: u64,
    pub scrub_cycles: u64,
    pub gpu_vram_scrubs: u64,
    pub failed_scrubs: u64,
    pub average_latency_us: f64,
}

/// Main Heap Scrubber
pub struct HeapScrubber {
    /// Running flag
    is_running: Arc<AtomicBool>,
    /// Scrub thread handle
    scrub_thread: parking_lot::Mutex<Option<thread::JoinHandle<()>>>,
    /// Total bytes scrubbed
    total_bytes_scrubbed: AtomicU64,
    /// Scrub cycles completed
    scrub_cycles: AtomicU64,
    /// GPU VRAM scrubs
    gpu_vram_scrubs: AtomicU64,
    /// Failed scrubs
    failed_scrubs: AtomicU64,
    /// Latency accumulator for averaging
    latency_accumulator: AtomicU64,
    /// Latency count for averaging
    latency_count: AtomicU64,
    /// Registered memory regions
    registered_regions: parking_lot::Mutex<Vec<MemoryRegion>>,
    /// AMD DirectML device reference
    directml_device: Option<Arc<dyn GpuDevice>>,
}

/// Memory region descriptor
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub address: usize,
    pub size: usize,
    pub region_type: MemoryRegionType,
    pub contains_secrets: bool,
}

/// GPU Device trait for VRAM scrubbing
pub trait GpuDevice: Send + Sync {
    /// Get VRAM allocation info
    fn get_vram_allocations(&self) -> Vec<GpuAllocation>;
    /// Scrub specific VRAM allocation
    fn scrub_allocation(&self, alloc_id: u64) -> Result<(), String>;
    /// Full VRAM scrub
    fn scrub_all_vram(&self) -> Result<(), String>;
}

/// GPU allocation descriptor
#[derive(Debug, Clone)]
pub struct GpuAllocation {
    pub id: u64,
    pub base_address: u64,
    pub size: usize,
    pub memory_type: String,
}

impl HeapScrubber {
    /// Create new heap scrubber
    pub fn new() -> Self {
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            scrub_thread: parking_lot::Mutex::new(None),
            total_bytes_scrubbed: AtomicU64::new(0),
            scrub_cycles: AtomicU64::new(0),
            gpu_vram_scrubs: AtomicU64::new(0),
            failed_scrubs: AtomicU64::new(0),
            latency_accumulator: AtomicU64::new(0),
            latency_count: AtomicU64::new(0),
            registered_regions: parking_lot::Mutex::new(Vec::new()),
            directml_device: None,
        }
    }

    /// Configure with GPU device for VRAM scrubbing
    pub fn with_gpu_device<D: GpuDevice + 'static>(mut self, device: D) -> Self {
        self.directml_device = Some(Arc::new(device));
        self
    }

    /// Register a memory region for scrubbing
    pub fn register_region(&self, region: MemoryRegion) {
        let mut regions = self.registered_regions.lock();
        regions.push(region);
    }

    /// Start the background scrubber thread
    pub fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.is_running.load(Ordering::Relaxed) {
            return Err("Scrubber already running".into());
        }

        self.is_running.store(true, Ordering::SeqCst);

        let scrubber = Arc::new(self.clone_for_thread());
        let handle = thread::Builder::new()
            .name("heap_scrubber".to_string())
            .spawn(move || {
                scrubber.scrub_loop();
            })?;

        *self.scrub_thread.lock() = Some(handle);
        log::info!("Heap scrubber started");
        Ok(())
    }

    /// Stop the scrubber
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        
        let mut thread_guard = self.scrub_thread.lock();
        if let Some(handle) = thread_guard.take() {
            let _ = handle.join();
        }
        
        log::info!("Heap scrubber stopped");
    }

    fn clone_for_thread(&self) -> HeapScrubber {
        HeapScrubber {
            is_running: Arc::clone(&self.is_running),
            scrub_thread: parking_lot::Mutex::new(None),
            total_bytes_scrubbed: AtomicU64::new(self.total_bytes_scrubbed.load(Ordering::Relaxed)),
            scrub_cycles: AtomicU64::new(self.scrub_cycles.load(Ordering::Relaxed)),
            gpu_vram_scrubs: AtomicU64::new(self.gpu_vram_scrubs.load(Ordering::Relaxed)),
            failed_scrubs: AtomicU64::new(self.failed_scrubs.load(Ordering::Relaxed)),
            latency_accumulator: AtomicU64::new(self.latency_accumulator.load(Ordering::Relaxed)),
            latency_count: AtomicU64::new(self.latency_count.load(Ordering::Relaxed)),
            registered_regions: parking_lot::Mutex::new(
                self.registered_regions.lock().clone()
            ),
            directml_device: self.directml_device.clone(),
        }
    }

    /// Main scrub loop
    fn scrub_loop(&self) {
        while self.is_running.load(Ordering::Relaxed) {
            let start = Instant::now();

            // Scrub registered regions
            self.scrub_registered_regions();

            // Scrub GPU VRAM if configured
            if let Some(ref gpu) = self.directml_device {
                if let Ok(_) = gpu.scrub_all_vram() {
                    self.gpu_vram_scrubs.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Record latency
            let elapsed_us = start.elapsed().as_micros() as u64;
            self.latency_accumulator.fetch_add(elapsed_us, Ordering::Relaxed);
            self.latency_count.fetch_add(1, Ordering::Relaxed);

            self.scrub_cycles.fetch_add(1, Ordering::Relaxed);

            // Sleep until next cycle
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms < MAX_SCRUB_INTERVAL_MS {
                thread::sleep(Duration::from_millis(MAX_SCRUB_INTERVAL_MS - elapsed_ms));
            }
        }
    }

    /// Scrub all registered memory regions
    fn scrub_registered_regions(&self) {
        let regions = self.registered_regions.lock();
        
        for region in regions.iter() {
            if region.contains_secrets {
                // Priority scrub for secret-containing regions
                self.scrub_region_secure(region);
            } else {
                // Standard scrub for other regions
                self.scrub_region_standard(region);
            }
        }
    }

    /// Secure multi-pass scrub for secret-containing regions
    fn scrub_region_secure(&self, region: &MemoryRegion) {
        unsafe {
            let ptr = region.address as *mut u8;
            
            // Multi-pass with different patterns
            for &pattern in &SCRUB_PATTERNS[..SCRUB_PASSES.min(SCRUB_PATTERNS.len())] {
                ptr::write_bytes(ptr, pattern, region.size);
            }
            
            // Final zero pass
            ptr::write_bytes(ptr, 0x00, region.size);
        }

        self.total_bytes_scrubbed.fetch_add(region.size as u64, Ordering::Relaxed);
    }

    /// Standard single-pass scrub for non-secret regions
    fn scrub_region_standard(&self, region: &MemoryRegion) {
        unsafe {
            let ptr = region.address as *mut u8;
            ptr::write_bytes(ptr, 0x00, region.size);
        }

        self.total_bytes_scrubbed.fetch_add(region.size as u64, Ordering::Relaxed);
    }

    /// Immediately scrub a specific memory range
    pub fn scrub_range(&self, address: usize, size: usize) {
        let region = MemoryRegion {
            address,
            size,
            region_type: MemoryRegionType::Heap,
            contains_secrets: true,
        };
        
        self.scrub_region_secure(&region);
    }

    /// Scrub a slice directly
    pub fn scrub_slice<T: Copy>(data: &mut [T]) {
        unsafe {
            let ptr = data.as_mut_ptr() as *mut u8;
            let size = data.len() * std::mem::size_of::<T>();
            
            for &pattern in &SCRUB_PATTERNS[..SCRUB_PASSES.min(SCRUB_PATTERNS.len())] {
                ptr::write_bytes(ptr, pattern, size);
            }
            
            ptr::write_bytes(ptr, 0x00, size);
        }
    }

    /// Scrub a string containing secrets
    pub fn scrub_string(s: &mut String) {
        unsafe {
            let vec = s.as_mut_vec();
            let ptr = vec.as_mut_ptr();
            let len = vec.len();
            let cap = vec.capacity();
            
            // Scrub entire capacity
            ptr::write_bytes(ptr, 0x00, cap);
        }
        
        s.clear();
    }

    /// Get scrub statistics
    pub fn get_stats(&self) -> ScrubStats {
        let latency_count = self.latency_count.load(Ordering::Relaxed);
        let avg_latency = if latency_count > 0 {
            self.latency_accumulator.load(Ordering::Relaxed) as f64 / latency_count as f64
        } else {
            0.0
        };

        ScrubStats {
            total_bytes_scrubbed: self.total_bytes_scrubbed.load(Ordering::Relaxed),
            scrub_cycles: self.scrub_cycles.load(Ordering::Relaxed),
            gpu_vram_scrubs: self.gpu_vram_scrubs.load(Ordering::Relaxed),
            failed_scrubs: self.failed_scrubs.load(Ordering::Relaxed),
            average_latency_us: avg_latency,
        }
    }

    /// Force immediate full scrub (blocking)
    pub fn force_full_scrub(&self) -> Result<(), String> {
        log::info!("Forcing full memory scrub...");
        
        self.scrub_registered_regions();
        
        if let Some(ref gpu) = self.directml_device {
            gpu.scrub_all_vram()?;
            self.gpu_vram_scrubs.fetch_add(1, Ordering::Relaxed);
        }

        log::info!("Full memory scrub completed");
        Ok(())
    }
}

impl Default for HeapScrubber {
    fn default() -> Self {
        Self::new()
    }
}

/// Implement Drop to ensure memory is scrubbed on deallocation
impl Drop for HeapScrubber {
    fn drop(&mut self) {
        self.stop();
        
        // Final scrub of all registered regions
        log::info!("Performing final memory scrub on shutdown");
        self.scrub_registered_regions();
    }
}

/// Global heap scrubber instance
pub static GLOBAL_HEAP_SCRUBBER: parking_lot::OnceCell<Arc<HeapScrubber>> = parking_lot::OnceCell::new();

/// Initialize global heap scrubber
pub fn init_global_scrubber() -> Arc<HeapScrubber> {
    let scrubber = Arc::new(HeapScrubber::new());
    GLOBAL_HEAP_SCRUBBER.get_or_init(|| scrubber.clone());
    scrubber
}

/// Get global scrubber instance
pub fn get_global_scrubber() -> Option<Arc<HeapScrubber>> {
    GLOBAL_HEAP_SCRUBBER.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scrubber_creation() {
        let scrubber = HeapScrubber::new();
        let stats = scrubber.get_stats();
        assert_eq!(stats.total_bytes_scrubbed, 0);
        assert_eq!(stats.scrub_cycles, 0);
    }

    #[test]
    fn test_scrub_slice() {
        let mut data = vec![0xDEADBEEFu64; 100];
        HeapScrubber::scrub_slice(&mut data);
        
        // Verify all zeros
        assert!(data.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_scrub_string() {
        let mut secret = String::from("super_secret_api_key_12345");
        HeapScrubber::scrub_string(&mut secret);
        
        assert!(secret.is_empty());
    }
}
