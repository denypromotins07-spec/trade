//! src/queues/batch_mpsc.rs
//!
//! Stage 51: Multi-Producer Single-Consumer Queue with Batch Processing
//!
//! Implements an MPSC queue that batches incoming network packets into contiguous
//! memory blocks, optimizing CPU prefetcher efficiency for the matching engine.
//! Optimized for AMD Zen architecture with microsecond latency targets.
//!
//! Critical for aggregating tick data from multiple network sources efficiently.

use std::cell::UnsafeCell;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Cache line size for AMD Zen (64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Default batch size for network packet aggregation
const DEFAULT_BATCH_SIZE: usize = 256;

/// Maximum number of producers supported
const MAX_PRODUCERS: usize = 16;

/// A single batch block containing contiguous packet data
#[repr(C, align(64))]
struct BatchBlock<T> {
    /// Array of items in this batch
    items: UnsafeCell<[T; DEFAULT_BATCH_SIZE]>,
    
    /// Number of valid items in the batch
    count: AtomicUsize,
    
    /// Next batch in the linked list
    next: AtomicPtr<BatchBlock<T>>,
    
    /// Padding to prevent false sharing
    _padding: [u8; Self::calculate_padding()],
}

impl<T: Default + Copy> BatchBlock<T> {
    const fn calculate_padding() -> usize {
        let header_size = mem::size_of::<UnsafeCell<[T; DEFAULT_BATCH_SIZE]>>() 
            + mem::size_of::<AtomicUsize>() 
            + mem::size_of::<AtomicPtr<BatchBlock<T>>>();
        
        if header_size >= CACHE_LINE_SIZE {
            0
        } else {
            CACHE_LINE_SIZE - (header_size % CACHE_LINE_SIZE)
        }
    }

    fn new() -> Self {
        Self {
            items: UnsafeCell::new([T::default(); DEFAULT_BATCH_SIZE]),
            count: AtomicUsize::new(0),
            next: AtomicPtr::new(ptr::null_mut()),
            _padding: [0; Self::calculate_padding()],
        }
    }

    /// Try to add an item to this batch
    #[inline(always)]
    fn try_push(&self, item: T) -> Result<(), T> {
        let current = self.count.load(Ordering::Relaxed);
        
        if current >= DEFAULT_BATCH_SIZE {
            return Err(item); // Batch is full
        }

        unsafe {
            (*self.items.get())[current] = item;
        }

        self.count.store(current + 1, Ordering::Release);
        Ok(())
    }

    /// Get the count of items
    #[inline(always)]
    fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Check if batch is full
    #[inline(always)]
    fn is_full(&self) -> bool {
        self.len() >= DEFAULT_BATCH_SIZE
    }
}

/// Multi-producer single-consumer batch queue
///
/// Producers append to the current batch until it's full, then atomically
/// link a new batch. Consumer processes entire batches for optimal cache usage.
pub struct BatchMpscQueue<T> {
    /// Head of the batch linked list (consumer reads from here)
    head: AtomicPtr<BatchBlock<T>>,
    
    /// Tail pointer for producers to append
    tail: AtomicPtr<BatchBlock<T>>,
    
    /// Current batch being filled by producers
    current: AtomicPtr<BatchBlock<T>>,
    
    /// Total items enqueued
    total_enqueued: AtomicUsize,
    
    /// Total items dequeued
    total_dequeued: AtomicUsize,
}

unsafe impl<T: Send> Send for BatchMpscQueue<T> {}
unsafe impl<T: Sync> Sync for BatchMpscQueue<T> {}

impl<T: Default + Copy> BatchMpscQueue<T> {
    /// Create a new batch MPSC queue
    pub fn new() -> Self {
        // Allocate initial empty batch
        let initial_batch = Box::new(BatchBlock::new());
        let initial_ptr = Box::into_raw(initial_batch);

        Self {
            head: AtomicPtr::new(initial_ptr),
            tail: AtomicPtr::new(initial_ptr),
            current: AtomicPtr::new(initial_ptr),
            total_enqueued: AtomicUsize::new(0),
            total_dequeued: AtomicUsize::new(0),
        }
    }

    /// Producer: Push an item to the queue
    ///
    /// Uses optimistic locking to find/create space in current batch.
    #[inline(always)]
    pub fn push(&self, item: T) -> Result<(), T> {
        loop {
            let current_ptr = self.current.load(Ordering::Acquire);
            
            if current_ptr.is_null() {
                // Queue is being initialized, retry
                std::hint::spin_loop();
                continue;
            }

            let current = unsafe { &*current_ptr };

            // Try to push to current batch
            match current.try_push(item) {
                Ok(()) => {
                    self.total_enqueued.fetch_add(1, Ordering::Relaxed);
                    return Ok(());
                }
                Err(item) => {
                    // Batch is full, need to create/link a new batch
                    let new_batch = Box::new(BatchBlock::new());
                    let new_ptr = Box::into_raw(new_batch);

                    // Try to link the new batch
                    let expected = current_ptr;
                    if self.current.compare_exchange(
                        expected,
                        new_ptr,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    ).is_ok() {
                        // Successfully linked, update tail
                        self.tail.store(new_ptr, Ordering::Release);
                        
                        // Retry pushing the item to the new batch
                        let new_current = unsafe { &*new_ptr };
                        match new_current.try_push(item) {
                            Ok(()) => {
                                self.total_enqueued.fetch_add(1, Ordering::Relaxed);
                                return Ok(());
                            }
                            Err(item) => {
                                // This shouldn't happen on a fresh batch
                                // Deallocate and return error
                                unsafe {
                                    drop(Box::from_raw(new_ptr));
                                }
                                return Err(item);
                            }
                        }
                    } else {
                        // Another thread linked a batch first, deallocate ours and retry
                        unsafe {
                            drop(Box::from_raw(new_ptr));
                        }
                        // Loop continues to retry
                    }
                }
            }
        }
    }

    /// Consumer: Pop an entire batch for processing
    ///
    /// Returns a slice of items from the oldest batch.
    /// Returns None if no complete batches available.
    #[inline(always)]
    pub fn pop_batch(&self) -> Option<BatchRef<'_, T>> {
        let head_ptr = self.head.load(Ordering::Acquire);
        
        if head_ptr.is_null() {
            return None;
        }

        let head = unsafe { &*head_ptr };
        let count = head.len();

        if count == 0 {
            // Empty batch, check if there's a next batch
            let next_ptr = head.next.load(Ordering::Acquire);
            if next_ptr.is_null() {
                return None; // No more batches
            }

            // Move head to next batch
            if self.head.compare_exchange(
                head_ptr,
                next_ptr,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                // Successfully moved, deallocate old head
                unsafe {
                    drop(Box::from_raw(head_ptr));
                }
                
                // Return reference to new head
                return self.pop_batch();
            } else {
                // CAS failed, retry
                return self.pop_batch();
            }
        }

        Some(BatchRef {
            data: unsafe { &(*head.items.get())[..count] },
            _marker: std::marker::PhantomData,
        })
    }

    /// Consumer: Acknowledge and free a processed batch
    ///
    /// # Safety
    /// - Must be called after processing the batch returned by pop_batch
    #[inline(always)]
    pub unsafe fn ack_batch(&self, batch_ptr: *const BatchBlock<T>) {
        let next_ptr = (*batch_ptr).next.load(Ordering::Acquire);
        
        if !next_ptr.is_null() {
            // Move head forward
            let current_head = self.head.load(Ordering::Relaxed);
            if current_head == batch_ptr as *mut _ {
                self.head.store(next_ptr, Ordering::Release);
                drop(Box::from_raw(batch_ptr as *mut _));
                self.total_dequeued.fetch_add((*batch_ptr).len(), Ordering::Relaxed);
            }
        }
    }

    /// Get total items enqueued
    #[inline(always)]
    pub fn total_enqueued(&self) -> usize {
        self.total_enqueued.load(Ordering::Relaxed)
    }

    /// Get total items dequeued
    #[inline(always)]
    pub fn total_dequeued(&self) -> usize {
        self.total_dequeued.load(Ordering::Relaxed)
    }

    /// Get approximate queue length
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.total_enqueued.load(Ordering::Relaxed) 
            - self.total_dequeued.load(Ordering::Relaxed)
    }

    /// Check if queue is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Default + Copy> Default for BatchMpscQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default + Copy> Drop for BatchMpscQueue<T> {
    fn drop(&mut self) {
        // Free all remaining batches
        let mut current = *self.head.get_mut();
        while !current.is_null() {
            unsafe {
                let next = (*current).next.load(Ordering::Relaxed);
                drop(Box::from_raw(current));
                current = next;
            }
        }
    }
}

/// Reference to a batch of items
pub struct BatchRef<'a, T> {
    data: &'a [T],
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T> BatchRef<'a, T> {
    /// Get slice of items
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Get number of items
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over items
    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for BatchRef<'a, T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_push_pop() {
        let queue: BatchMpscQueue<i32> = BatchMpscQueue::new();

        // Push some items
        for i in 0..10 {
            queue.push(i).unwrap();
        }

        assert_eq!(queue.len(), 10);
        assert!(!queue.is_empty());

        // Pop batch
        let batch = queue.pop_batch().expect("Should have batch");
        assert!(batch.len() > 0);
        println!("Batch size: {}", batch.len());
    }

    #[test]
    fn test_batch_full_behavior() {
        let queue: BatchMpscQueue<i32> = BatchMpscQueue::new();

        // Fill more than one batch
        for i in 0..DEFAULT_BATCH_SIZE + 10 {
            queue.push(i as i32).unwrap();
        }

        assert!(queue.len() > DEFAULT_BATCH_SIZE);

        // First batch should be full
        let batch1 = queue.pop_batch().expect("Should have batch");
        assert_eq!(batch1.len(), DEFAULT_BATCH_SIZE);

        // Second batch should have remainder
        let batch2 = queue.pop_batch().expect("Should have second batch");
        assert_eq!(batch2.len(), 10);
    }

    #[test]
    fn test_concurrent_producers() {
        use std::thread;

        let queue: BatchMpscQueue<usize> = BatchMpscQueue::new();
        let num_producers = 4;
        let items_per_producer = 100;

        let handles: Vec<_> = (0..num_producers)
            .map(|id| {
                thread::spawn(move || {
                    for i in 0..items_per_producer {
                        queue.push(id * 1000 + i).unwrap();
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(queue.len(), num_producers * items_per_producer);
    }

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(mem::align_of::<BatchBlock<i32>>(), CACHE_LINE_SIZE);
        println!("BatchBlock aligned to {} bytes", CACHE_LINE_SIZE);
    }
}
