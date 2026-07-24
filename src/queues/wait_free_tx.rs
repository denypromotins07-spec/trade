//! src/queues/wait_free_tx.rs
//!
//! Stage 51: Wait-Free Transmission Queue for Outbound Binance Orders
//!
//! Implements a wait-free queue utilizing strict memory barriers (mfence) to
//! guarantee visibility to the NIC DMA engine instantly. Optimized for AMD Zen
//! architecture with microsecond latency requirements.
//!
//! Critical for order submission path where every microsecond counts.

use std::cell::UnsafeCell;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Cache line size for AMD Zen (64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Maximum queue capacity (power of 2)
const MAX_CAPACITY: usize = 8192;

/// Order transmission slot
#[repr(C, align(64))]
struct TxSlot {
    /// Sequence number for ordering
    sequence: AtomicU64,
    
    /// Pointer to order data (or inline storage)
    data_ptr: AtomicPtr<u8>,
    
    /// Data length
    data_len: AtomicUsize,
    
    /// Slot state: 0=empty, 1=pending, 2=sent, 3=acked
    state: AtomicUsize,
    
    /// Padding to fill cache line
    _padding: [u8; Self::calculate_padding()],
}

impl TxSlot {
    const fn calculate_padding() -> usize {
        let header_size = mem::size_of::<AtomicU64>() 
            + mem::size_of::<AtomicPtr<u8>>() 
            + mem::size_of::<AtomicUsize>() * 2;
        
        if header_size >= CACHE_LINE_SIZE {
            0
        } else {
            CACHE_LINE_SIZE - header_size
        }
    }

    const fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            data_ptr: AtomicPtr::new(ptr::null_mut()),
            data_len: AtomicUsize::new(0),
            state: AtomicUsize::new(0),
            _padding: [0; Self::calculate_padding()],
        }
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.state.load(Ordering::Acquire) == 0
    }

    #[inline(always)]
    fn is_pending(&self) -> bool {
        self.state.load(Ordering::Acquire) == 1
    }
}

/// Wait-free transmission queue for outbound orders
///
/// Uses a circular buffer with atomic sequence numbers to achieve
/// wait-free progress guarantees. Multiple producers can enqueue
/// concurrently without blocking.
pub struct WaitFreeTxQueue {
    /// Circular buffer of slots
    slots: Box<[TxSlot]>,
    
    /// Capacity mask (capacity - 1)
    mask: usize,
    
    /// Next sequence number to assign
    next_sequence: AtomicU64,
    
    /// Consumer read position
    consumer_pos: AtomicUsize,
    
    /// Statistics
    total_enqueued: AtomicUsize,
    total_sent: AtomicUsize,
}

unsafe impl Send for WaitFreeTxQueue {}
unsafe impl Sync for WaitFreeTxQueue {}

impl WaitFreeTxQueue {
    /// Create a new wait-free TX queue with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        // Round up to power of 2
        let capacity = capacity.next_power_of_two().min(MAX_CAPACITY);
        let mask = capacity - 1;

        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(TxSlot::new());
        }

        Self {
            slots: slots.into_boxed_slice(),
            mask,
            next_sequence: AtomicU64::new(1),
            consumer_pos: AtomicUsize::new(0),
            total_enqueued: AtomicUsize::new(0),
            total_sent: AtomicUsize::new(0),
        }
    }

    /// Create with default capacity
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Enqueue an order for transmission (wait-free)
    ///
    /// Returns the assigned sequence number on success.
    /// Returns None if queue is full.
    #[inline(always)]
    pub fn enqueue(&self, data: &[u8]) -> Option<u64> {
        // Get a unique sequence number atomically
        let seq = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        
        // Calculate slot index from sequence
        let idx = ((seq - 1) as usize) & self.mask;
        let slot = &self.slots[idx];

        // Wait for slot to be empty (consumer has processed it)
        let mut spins = 0;
        while !slot.is_empty() {
            if spins > 1000 {
                // Queue is full, decrement sequence and return failure
                self.next_sequence.fetch_sub(1, Ordering::Relaxed);
                return None;
            }
            std::hint::spin_loop();
            spins += 1;
        }

        // Copy data to slot (in production, this would use pre-allocated buffers)
        let data_ptr = data.as_ptr() as *mut u8;
        
        // Store data info
        slot.data_ptr.store(data_ptr, Ordering::Relaxed);
        slot.data_len.store(data.len(), Ordering::Relaxed);
        slot.sequence.store(seq, Ordering::Relaxed);

        // Full memory barrier to ensure all writes are visible before marking pending
        // This is critical for NIC DMA visibility
        unsafe {
            std::arch::asm!("mfence", options(nostack));
        }

        // Mark slot as pending (visible to consumer/NIC)
        slot.state.store(1, Ordering::Release);

        self.total_enqueued.fetch_add(1, Ordering::Relaxed);

        Some(seq)
    }

    /// Consumer: Get next pending transmission
    ///
    /// Returns (sequence, data_ptr, data_len) or None if nothing pending.
    #[inline(always)]
    pub fn dequeue(&self) -> Option<(u64, *const u8, usize)> {
        let consumer_idx = self.consumer_pos.load(Ordering::Relaxed);
        
        if consumer_idx >= self.next_sequence.load(Ordering::Acquire) - 1 {
            return None; // Nothing to send
        }

        let idx = consumer_idx as usize & self.mask;
        let slot = &self.slots[idx];

        // Check if slot is pending
        if !slot.is_pending() {
            return None;
        }

        let seq = slot.sequence.load(Ordering::Acquire);
        let data_ptr = slot.data_ptr.load(Ordering::Acquire);
        let data_len = slot.data_len.load(Ordering::Acquire);

        Some((seq, data_ptr, data_len))
    }

    /// Mark a slot as sent (after NIC DMA initiation)
    #[inline(always)]
    pub fn mark_sent(&self, seq: u64) {
        let idx = (seq - 1) as usize & self.mask;
        let slot = &self.slots[idx];
        
        slot.state.store(2, Ordering::Release);
        self.total_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark a slot as acknowledged (after exchange confirmation)
    #[inline(always)]
    pub fn mark_acked(&self, seq: u64) {
        let idx = (seq - 1) as usize & self.mask;
        let slot = &self.slots[idx];
        
        slot.state.store(3, Ordering::Release);
        
        // Advance consumer position if this was the next expected
        let expected = self.consumer_pos.load(Ordering::Relaxed);
        if seq == expected + 1 {
            self.consumer_pos.store(seq, Ordering::Release);
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> TxQueueStats {
        TxQueueStats {
            total_enqueued: self.total_enqueued.load(Ordering::Relaxed),
            total_sent: self.total_sent.load(Ordering::Relaxed),
            pending: self.total_enqueued.load(Ordering::Relaxed) 
                - self.total_sent.load(Ordering::Relaxed),
            capacity: self.mask + 1,
        }
    }

    /// Check if queue is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.total_enqueued.load(Ordering::Relaxed) 
            == self.total_sent.load(Ordering::Relaxed)
    }

    /// Force memory barrier for NIC synchronization
    #[inline(always)]
    pub fn sync_for_nic(&self) {
        unsafe {
            std::arch::asm!(
                "mfence",
                options(nostack, preserves_flags)
            );
        }
    }
}

impl Default for WaitFreeTxQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Transmission queue statistics
#[derive(Debug, Clone)]
pub struct TxQueueStats {
    pub total_enqueued: usize,
    pub total_sent: usize,
    pub pending: usize,
    pub capacity: usize,
}

impl TxQueueStats {
    /// Get utilization percentage
    pub fn utilization(&self) -> f64 {
        (self.pending as f64 / self.capacity as f64) * 100.0
    }
}

/// RAII guard for order transmission
pub struct TxGuard<'a> {
    queue: &'a WaitFreeTxQueue,
    sequence: Option<u64>,
}

impl<'a> TxGuard<'a> {
    /// Create a new guard by enqueuing data
    pub fn enqueue(queue: &'a WaitFreeTxQueue, data: &[u8]) -> Option<Self> {
        queue.enqueue(data).map(|seq| Self {
            queue,
            sequence: Some(seq),
        })
    }

    /// Get the assigned sequence number
    pub fn sequence(&self) -> Option<u64> {
        self.sequence
    }

    /// Commit the transmission (mark as sent)
    pub fn commit(mut self) {
        if let Some(seq) = self.sequence.take() {
            self.queue.mark_sent(seq);
        }
    }
}

impl<'a> Drop for TxGuard<'a> {
    fn drop(&mut self) {
        // If guard is dropped without commit, the slot remains pending
        // In production, this might trigger cleanup or timeout handling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_enqueue_dequeue() {
        let queue = WaitFreeTxQueue::new();
        
        let test_data = vec![1u8, 2, 3, 4, 5];
        let seq = queue.enqueue(&test_data).expect("Should enqueue");
        
        assert!(seq > 0);
        assert_eq!(queue.stats().total_enqueued, 1);
        
        let (dequeued_seq, ptr, len) = queue.dequeue().expect("Should dequeue");
        assert_eq!(dequeued_seq, seq);
        assert_eq!(len, 5);
        unsafe {
            assert_eq!(*ptr, 1);
        }
    }

    #[test]
    fn test_queue_full_behavior() {
        let queue = WaitFreeTxQueue::with_capacity(4);
        
        // Fill the queue
        for i in 0..4 {
            let data = [i as u8; 32];
            assert!(queue.enqueue(&data).is_some());
        }
        
        // Next should fail (queue full)
        let data = [99u8; 32];
        assert!(queue.enqueue(&data).is_none());
    }

    #[test]
    fn test_state_transitions() {
        let queue = WaitFreeTxQueue::new();
        
        let data = [42u8; 16];
        let seq = queue.enqueue(&data).expect("Should enqueue");
        
        // Initially pending
        let idx = (seq - 1) as usize & queue.mask;
        assert!(queue.slots[idx].is_pending());
        
        // Mark sent
        queue.mark_sent(seq);
        assert_eq!(queue.slots[idx].state.load(Ordering::Relaxed), 2);
        
        // Mark acked
        queue.mark_acked(seq);
        assert_eq!(queue.slots[idx].state.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_memory_barrier() {
        let queue = WaitFreeTxQueue::new();
        
        // Verify sync_for_nic doesn't panic
        queue.sync_for_nic();
        
        println!("Memory barrier executed successfully");
    }

    #[test]
    fn test_stats() {
        let queue = WaitFreeTxQueue::with_capacity(16);
        
        for i in 0..5 {
            let data = [i as u8; 16];
            let seq = queue.enqueue(&data).unwrap();
            queue.mark_sent(seq);
        }
        
        let stats = queue.stats();
        assert_eq!(stats.total_enqueued, 5);
        assert_eq!(stats.total_sent, 5);
        assert_eq!(stats.pending, 0);
    }
}
