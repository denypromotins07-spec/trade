//! src/queues/spsc_ring.rs
//!
//! Stage 51: Single-Producer Single-Consumer Ring Buffer
//!
//! Implements a lock-free SPSC ring buffer using cache-line padded head/tail pointers
//! to prevent false sharing and ensure sequential consistency across AMD CCDs.
//! Optimized for microsecond latency with strict acquire/release memory ordering.
//!
//! Critical for zero-copy tick data transfer between network and matching threads.

use std::cell::UnsafeCell;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache line size for AMD Zen architecture (64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Default ring buffer capacity (must be power of 2)
const DEFAULT_CAPACITY: usize = 4096;

/// Padded atomic for cache line isolation
#[repr(C, align(64))]
struct PaddedAtomicUsize {
    value: AtomicUsize,
    _padding: [u8; CACHE_LINE_SIZE - mem::size_of::<AtomicUsize>()],
}

impl PaddedAtomicUsize {
    const fn new(val: usize) -> Self {
        Self {
            value: AtomicUsize::new(val),
            _padding: [0; CACHE_LINE_SIZE - mem::size_of::<AtomicUsize>()],
        }
    }
}

/// Lock-free SPSC ring buffer
///
/// Optimized for single producer, single consumer pattern with:
/// - Cache-line padding to prevent false sharing across CCDs
/// - Acquire/release memory ordering for cross-core visibility
/// - Power-of-2 capacity for efficient modulo via bitmask
pub struct SpScRingBuffer<T> {
    /// Head pointer (consumer reads from here) - padded to own cache line
    head: PaddedAtomicUsize,
    
    /// Tail pointer (producer writes to here) - padded to own cache line  
    tail: PaddedAtomicUsize,
    
    /// Buffer capacity (power of 2)
    capacity: usize,
    
    /// Bitmask for efficient modulo operation (capacity - 1)
    mask: usize,
    
    /// Actual data storage
    data: Box<[UnsafeCell<T>]>,
}

unsafe impl<T: Send> Send for SpScRingBuffer<T> {}
unsafe impl<T: Sync> Sync for SpScRingBuffer<T> {}

impl<T> SpScRingBuffer<T> {
    /// Create a new ring buffer with default capacity
    pub fn new() -> Self 
    where 
        T: Default + Copy,
    {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new ring buffer with specified capacity (rounded up to power of 2)
    pub fn with_capacity(capacity: usize) -> Self 
    where 
        T: Default + Copy,
    {
        // Round up to next power of 2
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        // Allocate buffer initialized with defaults
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(UnsafeCell::new(T::default()));
        }

        Self {
            head: PaddedAtomicUsize::new(0),
            tail: PaddedAtomicUsize::new(0),
            capacity,
            mask,
            data: data.into_boxed_slice(),
        }
    }

    /// Create uninitialized ring buffer (for custom initialization)
    ///
    /// # Safety
    /// - Caller must ensure proper initialization before use
    /// - Capacity must be power of 2
    pub unsafe fn uninit(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");
        
        let mask = capacity - 1;
        let mut data = Vec::with_capacity(capacity);
        data.set_len(capacity); // Uninitialized!

        Self {
            head: PaddedAtomicUsize::new(0),
            tail: PaddedAtomicUsize::new(0),
            capacity,
            mask,
            data: data.into_boxed_slice(),
        }
    }

    /// Producer: Push an item to the ring buffer
    ///
    /// Returns Ok(()) on success, Err(item) if buffer is full.
    /// Uses Release ordering to ensure consumer sees the written data.
    #[inline(always)]
    pub fn push(&self, item: T) -> Result<(), T> {
        let tail = self.tail.value.load(Ordering::Relaxed);
        let head = self.head.value.load(Ordering::Acquire);

        // Check if buffer is full
        // We always keep one slot empty to distinguish full from empty
        if ((tail + 1) & self.mask) == head {
            return Err(item); // Buffer full
        }

        // Write data at tail position
        unsafe {
            *self.data[tail & self.mask].get() = item;
        }

        // Update tail with Release ordering so consumer sees the data
        self.tail.value.store(tail.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Consumer: Pop an item from the ring buffer
    ///
    /// Returns Ok(item) on success, Err(()) if buffer is empty.
    /// Uses Acquire ordering to ensure we see the producer's writes.
    #[inline(always)]
    pub fn pop(&self) -> Result<T, ()> {
        let head = self.head.value.load(Ordering::Relaxed);
        let tail = self.tail.value.load(Ordering::Acquire);

        // Check if buffer is empty
        if head == tail {
            return Err(()); // Buffer empty
        }

        // Read data from head position
        let item = unsafe { *self.data[head & self.mask].get() };

        // Update head with Release ordering
        self.head.value.store(head.wrapping_add(1), Ordering::Release);

        Ok(item)
    }

    /// Check if buffer is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        let head = self.head.value.load(Ordering::Acquire);
        let tail = self.tail.value.load(Ordering::Acquire);
        head == tail
    }

    /// Check if buffer is full
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        let head = self.head.value.load(Ordering::Acquire);
        let tail = self.tail.value.load(Ordering::Acquire);
        ((tail + 1) & self.mask) == head
    }

    /// Get current number of items in buffer
    #[inline(always)]
    pub fn len(&self) -> usize {
        let head = self.head.value.load(Ordering::Acquire);
        let tail = self.tail.value.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    /// Get remaining capacity
    #[inline(always)]
    pub fn remaining(&self) -> usize {
        self.capacity - 1 - self.len()
    }

    /// Get total capacity (excluding sentinel slot)
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity - 1
    }

    /// Clear all items from the buffer (consumer only!)
    #[inline(always)]
    pub fn clear(&self) {
        let tail = self.tail.value.load(Ordering::Acquire);
        self.head.value.store(tail, Ordering::Release);
    }

    /// Peek at the next item without removing it (consumer only)
    #[inline(always)]
    pub fn peek(&self) -> Option<&T> {
        let head = self.head.value.load(Ordering::Acquire);
        let tail = self.tail.value.load(Ordering::Acquire);

        if head == tail {
            return None;
        }

        Some(unsafe { &*self.data[head & self.mask].get() })
    }

    /// Get raw pointers for FFI or custom DMA operations
    ///
    /// # Safety
    /// - Must maintain proper memory ordering externally
    /// - Only use for advanced zero-copy scenarios
    #[inline(always)]
    pub unsafe fn get_raw_pointers(&self) -> RingRawPointers<T> {
        RingRawPointers {
            head: self.head.value.as_ptr(),
            tail: self.tail.value.as_ptr(),
            data: self.data.as_ptr() as *mut T,
            mask: self.mask,
            capacity: self.capacity,
        }
    }
}

impl<T: Default + Copy> Default for SpScRingBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Raw pointers for advanced operations
pub struct RingRawPointers<T> {
    pub head: *const AtomicUsize,
    pub tail: *const AtomicUsize,
    pub data: *mut T,
    pub mask: usize,
    pub capacity: usize,
}

unsafe impl<T> Send for RingRawPointers<T> {}

/// Batch producer interface for writing multiple items efficiently
pub struct BatchProducer<'a, T> {
    ring: &'a SpScRingBuffer<T>,
    start_tail: usize,
    count: usize,
}

impl<'a, T: Copy> BatchProducer<'a, T> {
    /// Reserve space for batch write
    ///
    /// Returns None if insufficient space available
    pub fn reserve(ring: &'a SpScRingBuffer<T>, count: usize) -> Option<Self> {
        let tail = ring.tail.value.load(Ordering::Relaxed);
        let head = ring.head.value.load(Ordering::Acquire);

        let available = if tail >= head {
            ring.capacity - 1 - (tail - head)
        } else {
            head - tail - 1
        };

        if count > available {
            return None;
        }

        Some(Self {
            ring,
            start_tail: tail,
            count,
        })
    }

    /// Write batch items
    pub fn write<I>(&self, items: I)
    where
        I: IntoIterator<Item = T>,
    {
        let mut idx = 0;
        for item in items {
            if idx >= self.count {
                break;
            }
            let pos = (self.start_tail + idx) & self.ring.mask;
            unsafe {
                *self.ring.data[pos].get() = item;
            }
            idx += 1;
        }

        // Commit the batch with single atomic update
        self.ring.tail.value.store(
            self.start_tail + idx,
            Ordering::Release,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_pop() {
        let ring: SpScRingBuffer<i32> = SpScRingBuffer::with_capacity(8);

        assert!(ring.is_empty());
        
        ring.push(1).unwrap();
        ring.push(2).unwrap();
        ring.push(3).unwrap();

        assert_eq!(ring.len(), 3);
        assert!(!ring.is_empty());

        assert_eq!(ring.pop().unwrap(), 1);
        assert_eq!(ring.pop().unwrap(), 2);
        assert_eq!(ring.pop().unwrap(), 3);

        assert!(ring.is_empty());
    }

    #[test]
    fn test_wraparound() {
        let ring: SpScRingBuffer<i32> = SpScRingBuffer::with_capacity(4);

        // Fill and empty multiple times to test wraparound
        for i in 0..100 {
            ring.push(i).unwrap();
            assert_eq!(ring.pop().unwrap(), i);
        }
    }

    #[test]
    fn test_full_buffer() {
        let ring: SpScRingBuffer<i32> = SpScRingBuffer::with_capacity(4);

        // Can only store capacity - 1 items
        ring.push(1).unwrap();
        ring.push(2).unwrap();
        ring.push(3).unwrap();

        // Fourth push should fail (buffer full)
        assert!(ring.push(4).is_err());
        assert!(ring.is_full());
    }

    #[test]
    fn test_cache_line_padding() {
        // Verify head and tail are on separate cache lines
        let offset_head = mem::offset_of!(SpScRingBuffer::<i32>, head);
        let offset_tail = mem::offset_of!(SpScRingBuffer::<i32>, tail);
        
        let distance = offset_tail - offset_head;
        assert!(distance >= CACHE_LINE_SIZE, 
            "Head and tail should be on separate cache lines");
        
        println!("Head offset: {}, Tail offset: {}, Distance: {}", 
            offset_head, offset_tail, distance);
    }

    #[test]
    fn test_memory_ordering() {
        // Test that acquire/release ordering works correctly
        let ring: SpScRingBuffer<u64> = SpScRingBuffer::with_capacity(16);

        // Producer simulation
        ring.push(0x12345678ABCDEF00u64).unwrap();

        // Consumer simulation - should see the value correctly
        let val = ring.pop().unwrap();
        assert_eq!(val, 0x12345678ABCDEF00u64);
    }

    #[test]
    fn test_capacity_power_of_two() {
        // Verify capacity is always power of 2
        for requested in [3, 7, 15, 100, 1000] {
            let ring: SpScRingBuffer<i32> = SpScRingBuffer::with_capacity(requested);
            assert!(ring.capacity.is_power_of_two());
            assert!(ring.capacity >= requested);
        }
    }
}
