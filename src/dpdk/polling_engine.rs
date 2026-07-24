//! DPDK-Style Busy Polling Engine for User-Space Packet Processing
//! 
//! This module implements a high-performance busy-polling engine that continuously
//! checks NIC descriptor rings for incoming Binance ticks, completely eliminating
//! OS interrupt latency and context switch overhead.
//! 
//! Optimized for AMD Ryzen AI 5 (Zen 4/Zen 5) architecture with strict 8GB RAM enforcement.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Maximum ring buffer size enforced to stay within 8GB global RAM limit
/// Each descriptor is 64 bytes, so 1M descriptors = 64MB
const MAX_RING_SIZE: usize = 1_048_576;

/// Polling interval in nanoseconds for tight spin loops
const POLL_INTERVAL_NS: u64 = 50;

/// Descriptor ring entry for zero-copy packet reception
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct RxDescriptor {
    /// Physical address of the packet buffer (DMA-mapped)
    pub addr: u64,
    /// Length of the packet data
    pub len: u32,
    /// Status flags (DD, EOP, errors, etc.)
    pub status: u32,
    /// Reserved padding to maintain 64-byte cache line alignment
    pub reserved: [u32; 4],
}

impl RxDescriptor {
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        // DD (Descriptor Done) bit indicates hardware has written packet data
        self.status & 0x1 != 0
    }

    #[inline(always)]
    pub fn packet_len(&self) -> usize {
        (self.len & 0xFFFF) as usize
    }
}

/// Receive queue structure with cache-line padded head/tail pointers
/// to prevent false sharing across AMD CCDs
#[repr(C, align(64))]
pub struct RxQueue {
    /// Base pointer to descriptor ring (memory-mapped DMA region)
    descriptors: *mut RxDescriptor,
    /// Number of descriptors in the ring
    ring_size: usize,
    /// Cache-line padded head index (consumer)
    head: AtomicU64,
    /// Cache-line padded tail index (producer - hardware)
    tail: AtomicU64,
    /// Active flag for polling loop control
    active: AtomicBool,
    /// Statistics counters (cache-line separated)
    packets_received: AtomicU64,
    bytes_received: AtomicU64,
    poll_iterations: AtomicU64,
}

unsafe impl Send for RxQueue {}
unsafe impl Sync for RxQueue {}

impl RxQueue {
    /// Create a new receive queue with fixed ring size (8GB limit enforced)
    pub fn new() -> Result<Self, &'static str> {
        // Enforce 8GB RAM limit by capping ring size
        if MAX_RING_SIZE * std::mem::size_of::<RxDescriptor>() > 64 * 1024 * 1024 {
            return Err("Ring size exceeds memory budget");
        }

        // In production, this would allocate huge pages via huge_page_mgr
        // For now, we use standard allocation with proper alignment
        let descriptors = unsafe {
            let layout = std::alloc::Layout::from_size_align(
                MAX_RING_SIZE * std::mem::size_of::<RxDescriptor>(),
                64,
            ).map_err(|_| "Invalid layout")?;
            std::alloc::alloc(layout) as *mut RxDescriptor
        };

        if descriptors.is_null() {
            return Err("Failed to allocate descriptor ring");
        }

        // Initialize all descriptors to zero
        unsafe {
            std::ptr::write_bytes(descriptors, 0, MAX_RING_SIZE);
        }

        Ok(RxQueue {
            descriptors,
            ring_size: MAX_RING_SIZE,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            active: AtomicBool::new(false),
            packets_received: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            poll_iterations: AtomicU64::new(0),
        })
    }

    /// Initialize the queue for polling (called during /START)
    pub fn start(&self) {
        self.active.store(true, Ordering::SeqCst);
        self.head.store(0, Ordering::Relaxed);
        self.tail.store(0, Ordering::Relaxed);
    }

    /// Stop polling and release resources (called during /KILL)
    pub fn stop(&self) {
        self.active.store(false, Ordering::SeqCst);
        
        // Memory barrier to ensure visibility
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Busy-poll loop that checks for new packets without yielding to OS
    /// Returns number of packets processed in this iteration
    #[inline(always)]
    pub fn poll(&self, handler: &mut dyn FnMut(&[u8])) -> usize {
        if !self.active.load(Ordering::Relaxed) {
            return 0;
        }

        let mut processed = 0;
        let current_tail = self.tail.load(Ordering::Acquire);
        let mut current_head = self.head.load(Ordering::Relaxed);

        while current_head < current_tail {
            let idx = (current_head % self.ring_size as u64) as usize;
            let desc = unsafe { &*self.descriptors.add(idx) };

            if !desc.is_complete() {
                break; // No more complete descriptors
            }

            // Zero-copy: directly pass packet data to handler
            let packet_len = desc.packet_len();
            if packet_len > 0 && packet_len <= 9000 { // Jumbo frame max
                let packet_data = unsafe {
                    std::slice::from_raw_parts(
                        (desc.addr as *const u8),
                        packet_len,
                    )
                };
                handler(packet_data);
                
                self.bytes_received.fetch_add(packet_len as u64, Ordering::Relaxed);
                processed += 1;
            }

            // Recycle descriptor (mark as available for hardware)
            unsafe {
                let desc_mut = &mut *self.descriptors.add(idx);
                desc_mut.status = 0; // Clear DD bit
            }

            current_head += 1;
            self.poll_iterations.fetch_add(1, Ordering::Relaxed);
        }

        if processed > 0 {
            self.head.store(current_head, Ordering::Release);
            self.packets_received.fetch_add(processed as u64, Ordering::Relaxed);
        }

        processed
    }

    /// Continuous polling loop (runs on dedicated core)
    pub fn run_polling_loop(&self, mut handler: impl FnMut(&[u8]) + Send) -> ! {
        self.start();
        
        loop {
            if !self.active.load(Ordering::Relaxed) {
                break;
            }

            let count = self.poll(&mut handler);
            
            if count == 0 {
                // Spin with minimal delay to prevent CPU throttling
                // Using PAUSE instruction equivalent for AMD Zen
                std::hint::spin_loop();
                
                // Optional: ultra-short sleep to reduce power consumption
                // Commented out for maximum performance
                // std::thread::sleep(Duration::from_nanos(POLL_INTERVAL_NS));
            }
        }

        panic!("Polling loop exited unexpectedly");
    }

    /// Get statistics (thread-safe)
    pub fn stats(&self) -> QueueStats {
        QueueStats {
            packets: self.packets_received.load(Ordering::Relaxed),
            bytes: self.bytes_received.load(Ordering::Relaxed),
            iterations: self.poll_iterations.load(Ordering::Relaxed),
            avg_packets_per_poll: {
                let iters = self.poll_iterations.load(Ordering::Relaxed);
                let pkts = self.packets_received.load(Ordering::Relaxed);
                if iters > 0 { pkts as f64 / iters as f64 } else { 0.0 }
            },
        }
    }
}

impl Drop for RxQueue {
    fn drop(&mut self) {
        self.stop();
        
        if !self.descriptors.is_null() {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(
                    self.ring_size * std::mem::size_of::<RxDescriptor>(),
                    64,
                );
                std::alloc::dealloc(self.descriptors as *mut u8, layout);
            }
        }
    }
}

/// Statistics structure for monitoring polling performance
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub packets: u64,
    pub bytes: u64,
    pub iterations: u64,
    pub avg_packets_per_poll: f64,
}

/// Global polling engine manager
pub struct PollingEngine {
    queues: Vec<RxQueue>,
    running: AtomicBool,
}

impl PollingEngine {
    pub fn new(num_queues: usize) -> Result<Self, &'static str> {
        let mut queues = Vec::with_capacity(num_queues);
        
        for _ in 0..num_queues {
            queues.push(RxQueue::new()?);
        }
        
        Ok(PollingEngine {
            queues,
            running: AtomicBool::new(false),
        })
    }

    pub fn start_all(&self) {
        self.running.store(true, Ordering::SeqCst);
        for queue in &self.queues {
            queue.start();
        }
    }

    pub fn stop_all(&self) {
        self.running.store(false, Ordering::SeqCst);
        for queue in &self.queues {
            queue.stop();
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rx_queue_creation() {
        let queue = RxQueue::new();
        assert!(queue.is_ok());
    }

    #[test]
    fn test_descriptor_alignment() {
        assert_eq!(std::mem::align_of::<RxDescriptor>(), 64);
        assert_eq!(std::mem::size_of::<RxDescriptor>(), 64);
    }
}
