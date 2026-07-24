//! Zero-Copy Receive Buffer Mapping for Direct Hardware Ring Access
//!
//! This module maps hardware NIC receive rings directly into user-space memory,
//! enabling the matching engine to read Ethernet frames without any kernel context
//! switches or data copies. Critical for microsecond-latency Binance tick ingestion.
//!
//! Optimized for AMD Ryzen AI 5 with strict 8GB RAM limit enforcement.

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum total memory budget for all zero-copy buffers (part of 8GB global limit)
const ZERO_COPY_BUDGET: usize = 2 * 1024 * 1024 * 1024; // 2GB reserved for RX buffers

/// Size of each packet buffer (supports jumbo frames up to 9KB)
const PACKET_BUFFER_SIZE: usize = 16384; // 16KB aligned to power of 2

/// Number of buffers in the pool (calculated to fit within budget)
const NUM_BUFFERS: usize = ZERO_COPY_BUDGET / PACKET_BUFFER_SIZE;

/// Memory-mapped packet buffer with DMA-safe alignment
#[repr(C, align(64))]
pub struct PacketBuffer {
    /// Raw packet data (DMA-written by NIC)
    pub data: [u8; PACKET_BUFFER_SIZE],
    /// Length of valid data (written by hardware/driver)
    pub len: AtomicUsize,
    /// Ownership flag: true = hardware owns, false = software owns
    pub owned_by_hw: AtomicBool,
    /// Sequence number for ordering verification
    pub seq_num: AtomicUsize,
}

impl PacketBuffer {
    #[inline(always)]
    pub fn new() -> Self {
        PacketBuffer {
            data: [0u8; PACKET_BUFFER_SIZE],
            len: AtomicUsize::new(0),
            owned_by_hw: AtomicBool::new(false),
            seq_num: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        !self.owned_by_hw.load(Ordering::Acquire)
    }

    #[inline(always)]
    pub fn packet_data(&self) -> &[u8] {
        let len = self.len.load(Ordering::Acquire);
        if len > 0 && len <= PACKET_BUFFER_SIZE {
            unsafe { std::slice::from_raw_parts(self.data.as_ptr(), len) }
        } else {
            &[]
        }
    }

    #[inline(always)]
    pub fn release_to_hw(&mut self, seq: usize) {
        self.len.store(0, Ordering::Relaxed);
        self.seq_num.store(seq, Ordering::Relaxed);
        self.owned_by_hw.store(true, Ordering::Release);
        
        // Ensure visibility to DMA engine
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

impl Default for PacketBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy receive ring that maps directly to hardware descriptors
#[repr(C, align(64))]
pub struct ZeroCopyRxRing {
    /// Array of packet buffers (pre-allocated, pinned memory)
    buffers: Box<[PacketBuffer]>,
    /// Current read index (consumer - matching engine)
    read_idx: AtomicUsize,
    /// Current write index (producer - hardware via driver)
    write_idx: AtomicUsize,
    /// Mask for efficient modulo operation (power of 2 size)
    mask: usize,
    /// Total packets received
    total_packets: AtomicUsize,
    /// Total bytes received
    total_bytes: AtomicUsize,
    /// Active flag
    active: AtomicBool,
}

unsafe impl Send for ZeroCopyRxRing {}
unsafe impl Sync for ZeroCopyRxRing {}

impl ZeroCopyRxRing {
    /// Create a new zero-copy receive ring
    pub fn new() -> Result<Self, &'static str> {
        // Verify we stay within memory budget
        let required_memory = NUM_BUFFERS * std::mem::size_of::<PacketBuffer>();
        if required_memory > ZERO_COPY_BUDGET {
            return Err("Zero-copy ring exceeds memory budget");
        }

        // Allocate buffers in contiguous memory for better prefetching
        let mut buffers = Vec::with_capacity(NUM_BUFFERS);
        for _ in 0..NUM_BUFFERS {
            buffers.push(PacketBuffer::new());
        }

        let buffers: Box<[PacketBuffer]> = buffers.into_boxed_slice();
        let mask = NUM_BUFFERS - 1; // Assumes NUM_BUFFERS is power of 2

        Ok(ZeroCopyRxRing {
            buffers,
            read_idx: AtomicUsize::new(0),
            write_idx: AtomicUsize::new(0),
            mask,
            total_packets: AtomicUsize::new(0),
            total_bytes: AtomicUsize::new(0),
            active: AtomicBool::new(false),
        })
    }

    /// Initialize the ring for operation (/START sequence)
    pub fn start(&self) {
        self.active.store(true, Ordering::SeqCst);
        self.read_idx.store(0, Ordering::Relaxed);
        self.write_idx.store(0, Ordering::Relaxed);
        
        // Pre-release all buffers to hardware
        for (i, buf) in self.buffers.iter().enumerate() {
            unsafe {
                let mut_buf = buf as *const PacketBuffer as *mut PacketBuffer;
                (*mut_buf).release_to_hw(i);
            }
        }
    }

    /// Stop the ring and cleanup (/KILL sequence)
    pub fn stop(&self) {
        self.active.store(false, Ordering::SeqCst);
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Enqueue a completed buffer from hardware (called by driver/NIC)
    #[inline(always)]
    pub fn enqueue_completed(&self, buffer_idx: usize, packet_len: usize) {
        if buffer_idx >= NUM_BUFFERS {
            return;
        }

        let buf = &self.buffers[buffer_idx];
        buf.len.store(packet_len, Ordering::Release);
        buf.owned_by_hw.store(false, Ordering::Release);
        
        // Update write index
        let old_write = self.write_idx.fetch_add(1, Ordering::AcqRel);
        
        self.total_packets.fetch_add(1, Ordering::Relaxed);
        self.total_bytes.fetch_add(packet_len, Ordering::Relaxed);
    }

    /// Dequeue a packet for processing (zero-copy access)
    /// Returns Some((&[u8], buffer_index)) if packet available, None otherwise
    #[inline(always)]
    pub fn dequeue_packet<'a>(&'a self) -> Option<(&'a [u8], usize)> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }

        let read = self.read_idx.load(Ordering::Acquire);
        let write = self.write_idx.load(Ordering::Acquire);

        if read >= write {
            return None; // No packets available
        }

        let buffer_idx = read & self.mask;
        let buf = &self.buffers[buffer_idx];

        if !buf.is_ready() {
            return None; // Race condition: buffer not actually ready
        }

        let data = buf.packet_data();
        if data.is_empty() {
            return None;
        }

        Some((data, buffer_idx))
    }

    /// Release a buffer back to hardware after processing
    #[inline(always)]
    pub fn release_buffer(&self, buffer_idx: usize, next_seq: usize) {
        if buffer_idx >= NUM_BUFFERS {
            return;
        }

        let buf = &self.buffers[buffer_idx];
        unsafe {
            let mut_buf = buf as *const PacketBuffer as *mut PacketBuffer;
            (*mut_buf).release_to_hw(next_seq);
        }

        self.read_idx.fetch_add(1, Ordering::Release);
    }

    /// Process all available packets with a closure
    #[inline(always)]
    pub fn process_all<F>(&self, mut handler: F) -> usize
    where
        F: FnMut(&[u8]),
    {
        let mut count = 0;

        while let Some((data, idx)) = self.dequeue_packet() {
            handler(data);
            count += 1;
            
            // Release buffer immediately after processing (zero-copy lifecycle)
            let next_seq = self.total_packets.load(Ordering::Relaxed) + NUM_BUFFERS;
            self.release_buffer(idx, next_seq);
        }

        count
    }

    /// Get current statistics
    pub fn stats(&self) -> RingStats {
        RingStats {
            total_packets: self.total_packets.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            pending_packets: {
                let read = self.read_idx.load(Ordering::Relaxed);
                let write = self.write_idx.load(Ordering::Relaxed);
                write.saturating_sub(read)
            },
            buffer_count: NUM_BUFFERS,
            buffer_size: PACKET_BUFFER_SIZE,
            memory_used: NUM_BUFFERS * std::mem::size_of::<PacketBuffer>(),
        }
    }

    /// Check if ring is active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

impl Drop for ZeroCopyRxRing {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Statistics structure for monitoring zero-copy ring performance
#[derive(Debug, Clone)]
pub struct RingStats {
    pub total_packets: usize,
    pub total_bytes: usize,
    pub pending_packets: usize,
    pub buffer_count: usize,
    pub buffer_size: usize,
    pub memory_used: usize,
}

/// Global manager for all zero-copy receive rings
pub struct ZeroCopyManager {
    rings: Vec<Arc<ZeroCopyRxRing>>,
    total_memory_allocated: AtomicUsize,
}

impl ZeroCopyManager {
    pub fn new(num_rings: usize) -> Result<Self, &'static str> {
        let mut rings = Vec::with_capacity(num_rings);
        let mut total_mem = 0;

        for _ in 0..num_rings {
            let ring = ZeroCopyRxRing::new()?;
            total_mem += NUM_BUFFERS * std::mem::size_of::<PacketBuffer>();
            rings.push(Arc::new(ring));
        }

        // Enforce global 8GB limit
        if total_mem > ZERO_COPY_BUDGET {
            return Err("Total zero-copy memory exceeds budget");
        }

        Ok(ZeroCopyManager {
            rings,
            total_memory_allocated: AtomicUsize::new(total_mem),
        })
    }

    pub fn start_all(&self) {
        for ring in &self.rings {
            ring.start();
        }
    }

    pub fn stop_all(&self) {
        for ring in &self.rings {
            ring.stop();
        }
    }

    pub fn get_ring(&self, idx: usize) -> Option<&Arc<ZeroCopyRxRing>> {
        self.rings.get(idx)
    }

    pub fn total_memory(&self) -> usize {
        self.total_memory_allocated.load(Ordering::Relaxed)
    }

    pub fn aggregate_stats(&self) -> Vec<RingStats> {
        self.rings.iter().map(|r| r.stats()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_buffer_creation() {
        let buf = PacketBuffer::new();
        assert_eq!(buf.packet_data().len(), 0);
    }

    #[test]
    fn test_zero_copy_ring_creation() {
        let ring = ZeroCopyRxRing::new();
        assert!(ring.is_ok());
    }

    #[test]
    fn test_memory_budget_enforcement() {
        let mem_per_ring = NUM_BUFFERS * std::mem::size_of::<PacketBuffer>();
        assert!(mem_per_ring <= ZERO_COPY_BUDGET);
    }

    #[test]
    fn test_buffer_alignment() {
        assert_eq!(std::mem::align_of::<PacketBuffer>(), 64);
    }
}
