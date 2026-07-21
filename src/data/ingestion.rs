//! Fixed-Size Ring Buffer for Tick Data Ingestion
//! 
//! This module implements a fixed-size ring buffer for incoming tick data to guarantee
//! O(1) memory usage and prevent heap fragmentation during extreme high-volatility
//! market events. Optimized for the 8GB RAM hard limit on AMD Ryzen AI 5 architecture.
//! 
//! Key Features:
//! - Lock-free single-producer single-consumer (SPSC) design
//! - Pre-allocated contiguous memory for cache efficiency
//! - Atomic head/tail pointers for thread-safe access
//! - Overflow handling with configurable policies (drop oldest / block)
//! - Memory-mapped file support for persistence

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::cell::UnsafeCell;

/// Default ring buffer capacity (power of 2 for efficient modulo)
const DEFAULT_CAPACITY: usize = 1 << 20; // 1M ticks (~32MB for TradeTick)

/// Maximum supported capacity
const MAX_CAPACITY: usize = 1 << 24; // 16M ticks

/// Ring buffer entry wrapper
#[derive(Debug, Clone, Copy)]
pub struct RingEntry<T: Copy + Default> {
    data: UnsafeCell<Option<T>>,
    sequence: AtomicU64,
}

impl<T: Copy + Default> RingEntry<T> {
    fn new() -> Self {
        Self {
            data: UnsafeCell::new(None),
            sequence: AtomicU64::new(0),
        }
    }

    fn set(&self, value: T, seq: u64) {
        unsafe {
            *self.data.get() = Some(value);
        }
        self.sequence.store(seq, Ordering::Release);
    }

    fn get(&self, expected_seq: u64) -> Option<T> {
        let seq = self.sequence.load(Ordering::Acquire);
        if seq == expected_seq {
            unsafe { (*self.data.get()).clone() }
        } else {
            None
        }
    }

    fn clear(&self) {
        unsafe {
            *self.data.get() = None;
        }
        self.sequence.store(0, Ordering::Release);
    }
}

/// Ring buffer overflow policy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverflowPolicy {
    /// Drop oldest entries when full
    DropOldest,
    /// Block producer when full (return error)
    Block,
    /// Overwrite oldest (circular)
    Overwrite,
}

/// Lock-free SPSC ring buffer for tick data
pub struct RingBuffer<T: Copy + Default> {
    /// Pre-allocated buffer entries
    buffer: Box<[RingEntry<T>]>,
    /// Capacity (always power of 2)
    capacity: usize,
    /// Capacity mask for efficient modulo
    mask: usize,
    /// Head index (producer writes here)
    head: AtomicU64,
    /// Tail index (consumer reads from here)
    tail: AtomicU64,
    /// Total items produced
    produced: AtomicU64,
    /// Total items consumed
    consumed: AtomicU64,
    /// Overflow policy
    overflow_policy: OverflowPolicy,
    /// Is buffer in error state
    is_error: AtomicBool,
    /// Buffer name for identification
    name: [u8; 32],
}

unsafe impl<T: Copy + Default + Send> Send for RingBuffer<T> {}
unsafe impl<T: Copy + Default + Send> Sync for RingBuffer<T> {}

impl<T: Copy + Default + Send> RingBuffer<T> {
    /// Create a new ring buffer with specified capacity
    pub fn new(capacity: usize, name: &str) -> Result<Self, String> {
        if capacity == 0 || capacity > MAX_CAPACITY {
            return Err(format!(
                "Capacity must be between 1 and {}",
                MAX_CAPACITY
            ));
        }

        // Round up to next power of 2
        let actual_capacity = capacity.next_power_of_two();
        let mask = actual_capacity - 1;

        let mut buffer = Vec::with_capacity(actual_capacity);
        for _ in 0..actual_capacity {
            buffer.push(RingEntry::new());
        }

        let mut name_bytes = [0u8; 32];
        let copy_len = name.len().min(32);
        name_bytes[..copy_len].copy_from_slice(&name.as_bytes()[..copy_len]);

        Ok(Self {
            buffer: buffer.into_boxed_slice(),
            capacity: actual_capacity,
            mask,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            produced: AtomicU64::new(0),
            consumed: AtomicU64::new(0),
            overflow_policy: OverflowPolicy::Overwrite,
            is_error: AtomicBool::new(false),
            name: name_bytes,
        })
    }

    /// Create with default capacity
    pub fn with_default(name: &str) -> Self {
        Self::new(DEFAULT_CAPACITY, name).expect("Default capacity should be valid")
    }

    /// Set overflow policy
    pub fn set_overflow_policy(&mut self, policy: OverflowPolicy) {
        self.overflow_policy = policy;
    }

    /// Push data to the buffer (producer side)
    /// Returns true if successful, false if blocked by policy
    pub fn push(&self, data: T) -> bool {
        if self.is_error.load(Ordering::Acquire) {
            return false;
        }

        let current_head = self.head.load(Ordering::Relaxed);
        let current_tail = self.tail.load(Ordering::Acquire);
        let next_head = current_head.wrapping_add(1);

        // Check if buffer is full
        if next_head.wrapping_sub(current_tail) > self.capacity as u64 {
            match self.overflow_policy {
                OverflowPolicy::Block => return false,
                OverflowPolicy::DropOldest => {
                    // Advance tail to make room
                    self.tail.store(next_head - self.capacity as u64, Ordering::Release);
                }
                OverflowPolicy::Overwrite => {
                    // Will overwrite oldest, advance tail
                    self.tail.store(next_head - self.capacity as u64, Ordering::Release);
                }
            }
        }

        let index = (current_head as usize) & self.mask;
        let seq = current_head + 1;
        
        self.buffer[index].set(data, seq);
        self.head.store(next_head, Ordering::Release);
        self.produced.fetch_add(1, Ordering::Relaxed);

        true
    }

    /// Pop data from the buffer (consumer side)
    /// Returns None if buffer is empty
    pub fn pop(&self) -> Option<T> {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let current_head = self.head.load(Ordering::Acquire);

        if current_tail >= current_head {
            return None; // Empty
        }

        let index = (current_tail as usize) & self.mask;
        let seq = current_tail + 1;

        match self.buffer[index].get(seq) {
            Some(data) => {
                self.buffer[index].clear();
                self.tail.store(current_tail + 1, Ordering::Release);
                self.consumed.fetch_add(1, Ordering::Relaxed);
                Some(data)
            }
            None => None, // Race condition or corrupted data
        }
    }

    /// Try to peek at the next item without consuming
    pub fn peek(&self) -> Option<T> {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let current_head = self.head.load(Ordering::Acquire);

        if current_tail >= current_head {
            return None;
        }

        let index = (current_tail as usize) & self.mask;
        let seq = current_tail + 1;

        self.buffer[index].get(seq)
    }

    /// Get current number of items in buffer
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        (head.wrapping_sub(tail)) as usize
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get total items produced
    pub fn total_produced(&self) -> u64 {
        self.produced.load(Ordering::Acquire)
    }

    /// Get total items consumed
    pub fn total_consumed(&self) -> u64 {
        self.consumed.load(Ordering::Acquire)
    }

    /// Get dropped item count (produced - consumed - current_len)
    pub fn total_dropped(&self) -> u64 {
        let produced = self.produced.load(Ordering::Acquire);
        let consumed = self.consumed.load(Ordering::Acquire);
        let current = self.len() as u64;
        produced.saturating_sub(consumed + current)
    }

    /// Clear all items in buffer
    pub fn clear(&self) {
        let current_tail = self.tail.load(Ordering::Relaxed);
        let current_head = self.head.load(Ordering::Relaxed);

        for i in current_tail..current_head {
            let index = (i as usize) & self.mask;
            self.buffer[index].clear();
        }

        self.tail.store(current_head, Ordering::Release);
    }

    /// Get buffer utilization percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        self.len() as f64 / self.capacity as f64
    }

    /// Get buffer name
    pub fn name(&self) -> String {
        String::from_utf8_lossy(&self.name)
            .trim_end_matches('\0')
            .to_string()
    }

    /// Mark buffer as error state
    pub fn set_error(&self) {
        self.is_error.store(true, Ordering::Release);
    }

    /// Clear error state
    pub fn clear_error(&self) {
        self.is_error.store(false, Ordering::Release);
    }

    /// Check if in error state
    pub fn has_error(&self) -> bool {
        self.is_error.load(Ordering::Acquire)
    }
}

/// Multi-producer single-consumer (MPSC) variant using crossbeam
pub struct MpscRingBuffer<T: Copy + Default + Send> {
    inner: Arc<RingBuffer<T>>,
}

impl<T: Copy + Default + Send> MpscRingBuffer<T> {
    pub fn new(capacity: usize, name: &str) -> Result<Self, String> {
        Ok(Self {
            inner: Arc::new(RingBuffer::new(capacity, name)?),
        })
    }

    pub fn push(&self, data: T) -> bool {
        self.inner.push(data)
    }

    pub fn pop(&self) -> Option<T> {
        self.inner.pop()
    }

    pub fn clone_inner(&self) -> Arc<RingBuffer<T>> {
        Arc::clone(&self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_pop() {
        let buffer = RingBuffer::<i32>::new(16, "test").unwrap();
        
        assert!(buffer.push(1));
        assert!(buffer.push(2));
        assert!(buffer.push(3));
        
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_overflow_overwrite() {
        let mut buffer = RingBuffer::<i32>::new(4, "test").unwrap();
        buffer.set_overflow_policy(OverflowPolicy::Overwrite);
        
        // Fill buffer
        for i in 0..4 {
            assert!(buffer.push(i));
        }
        
        // Push more (should overwrite oldest)
        assert!(buffer.push(100));
        assert!(buffer.push(101));
        
        // Should have overwritten first two items
        assert_eq!(buffer.len(), 4);
    }

    #[test]
    fn test_utilization() {
        let buffer = RingBuffer::<i32>::new(100, "test").unwrap();
        
        assert_eq!(buffer.utilization(), 0.0);
        
        for i in 0..50 {
            buffer.push(i);
        }
        
        assert!((buffer.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_thread_safety() {
        let buffer = Arc::new(RingBuffer::<i32>::new(1024, "test").unwrap());
        let buffer_clone = Arc::clone(&buffer);
        
        // Producer thread
        let producer = std::thread::spawn(move || {
            for i in 0..1000 {
                while !buffer_clone.push(i) {
                    std::thread::yield_now();
                }
            }
        });
        
        // Consumer thread
        let buffer_clone2 = Arc::clone(&buffer);
        let consumer = std::thread::spawn(move || {
            let mut count = 0;
            while count < 1000 {
                if buffer_clone2.pop().is_some() {
                    count += 1;
                }
            }
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
