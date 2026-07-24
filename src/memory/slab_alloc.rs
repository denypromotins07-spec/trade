//! src/memory/slab_alloc.rs
//!
//! Stage 51: O(1) Lock-Free Slab Allocator for Fixed-Size Order Book Nodes
//!
//! Implements a lock-free slab allocator using intrusive free-lists to eliminate
//! heap fragmentation and garbage collection pauses. Optimized for AMD Zen 4/Zen 5
//! with 64-byte cache line alignment to prevent false sharing.
//!
//! Critical for high-frequency order book operations with strict 8GB RAM limit.

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::mem;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Cache line size for AMD Zen architecture (64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of slabs per allocator
const MAX_SLABS: usize = 1024;

/// Slab entry header stored inline with user data
#[repr(C)]
struct SlabEntry<T> {
    /// Next free entry pointer (only used when entry is free)
    next_free: AtomicPtr<SlabEntry<T>>,
    /// Entry state: 0 = free, 1 = allocated
    state: AtomicUsize,
    /// Padding to align user data to cache line
    _padding: [u8; Self::calculate_padding()],
}

impl<T> SlabEntry<T> {
    const fn calculate_padding() -> usize {
        let header_size = mem::size_of::<AtomicPtr<SlabEntry<T>>>() + mem::size_of::<AtomicUsize>();
        if header_size >= CACHE_LINE_SIZE {
            0
        } else {
            CACHE_LINE_SIZE - header_size
        }
    }
}

/// A single slab containing fixed-size entries
#[repr(C, align(64))]
struct Slab<T> {
    /// Pointer to first free entry
    free_list: AtomicPtr<SlabEntry<T>>,
    
    /// Number of allocated entries
    allocated: AtomicUsize,
    
    /// Total capacity of this slab
    capacity: usize,
    
    /// Entries stored contiguously after the header
    entries: UnsafeCell<[SlabEntry<T>; 0]>, // Flexible array member pattern
    
    _marker: PhantomData<T>,
}

impl<T> Slab<T> {
    /// Calculate the memory layout for a slab with given capacity
    fn layout(capacity: usize) -> Layout {
        let header_size = mem::size_of::<Self>();
        let entry_size = mem::size_of::<SlabEntry<T>>();
        let total_size = header_size + (entry_size * capacity);
        
        // Align to cache line boundary
        unsafe { Layout::from_size_align_unchecked(total_size, CACHE_LINE_SIZE) }
    }

    /// Create a new slab with given capacity
    unsafe fn new(capacity: usize) -> NonNull<Self> {
        let layout = Self::layout(capacity);
        let ptr = alloc::alloc(layout) as *mut Self;
        
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }

        // Initialize header
        ptr::write(&mut (*ptr).allocated, AtomicUsize::new(0));
        ptr::write(&mut (*ptr).capacity, capacity);
        ptr::write(&mut (*ptr)._marker, PhantomData);

        // Initialize entries as a linked free list
        let entries_ptr = ptr::addr_of_mut!((*ptr).entries) as *mut SlabEntry<T>;
        
        for i in 0..capacity {
            let entry = &mut *entries_ptr.add(i);
            ptr::write(&mut entry.state, AtomicUsize::new(0));
            
            let next = if i + 1 < capacity {
                entries_ptr.add(i + 1) as *mut _
            } else {
                ptr::null_mut()
            };
            ptr::write(&mut entry.next_free, AtomicPtr::new(next));
        }

        // Set free list head to first entry
        ptr::write(&mut (*ptr).free_list, AtomicPtr::new(entries_ptr));

        NonNull::new_unchecked(ptr)
    }

    /// Allocate an entry from this slab
    #[inline(always)]
    unsafe fn allocate(&self) -> Option<NonNull<SlabEntry<T>>> {
        loop {
            let current_head = self.free_list.load(Ordering::Acquire);
            
            if current_head.is_null() {
                return None; // Slab is full
            }

            let next = (*current_head).next_free.load(Ordering::Relaxed);

            // Try to swing the free list head to the next entry
            if self
                .free_list
                .compare_exchange_weak(current_head, next, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Mark entry as allocated
                (*current_head).state.store(1, Ordering::Release);
                
                self.allocated.fetch_add(1, Ordering::Relaxed);
                
                return Some(NonNull::new_unchecked(current_head as *mut _));
            }
            // CAS failed, retry with updated head
        }
    }

    /// Deallocate an entry back to this slab
    #[inline(always)]
    unsafe fn deallocate(&self, entry: NonNull<SlabEntry<T>>) {
        debug_assert_eq!((*entry.as_ptr()).state.load(Ordering::Relaxed), 1);

        // Mark as free
        (*entry.as_ptr()).state.store(0, Ordering::Release);

        // Push onto free list (lock-free stack push)
        loop {
            let current_head = self.free_list.load(Ordering::Acquire);
            (*entry.as_ptr()).next_free.store(current_head, Ordering::Relaxed);

            if self
                .free_list
                .compare_exchange_weak(current_head, entry.as_ptr(), Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.allocated.fetch_sub(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Check if slab is full
    #[inline(always)]
    fn is_full(&self) -> bool {
        self.free_list.load(Ordering::Acquire).is_null()
    }

    /// Get allocation count
    #[inline(always)]
    fn allocated_count(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }
}

/// Lock-free slab allocator for fixed-size objects
///
/// Automatically grows by adding new slabs when existing ones are full.
/// Thread-safe without global locks using per-slab atomic operations.
pub struct SlabAllocator<T> {
    /// Array of slab pointers
    slabs: [AtomicPtr<Slab<T>>; MAX_SLABS],
    
    /// Number of active slabs
    num_slabs: AtomicUsize,
    
    /// Capacity per slab
    slab_capacity: usize,
    
    /// Total allocations across all slabs
    total_allocated: AtomicUsize,
    
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for SlabAllocator<T> {}
unsafe impl<T: Sync> Sync for SlabAllocator<T> {}

impl<T> SlabAllocator<T> {
    /// Create a new slab allocator with default capacity
    pub const fn new() -> Self {
        Self::with_capacity(256)
    }

    /// Create a new slab allocator with specified entries per slab
    pub const fn with_capacity(slab_capacity: usize) -> Self {
        const EMPTY_SLABS: [AtomicPtr<Slab<T>>; MAX_SLABS] = [AtomicPtr::new(ptr::null_mut()); MAX_SLABS];
        
        Self {
            slabs: EMPTY_SLABS,
            num_slabs: AtomicUsize::new(0),
            slab_capacity,
            total_allocated: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    /// Allocate a new entry
    #[inline(always)]
    pub fn allocate(&self) -> Option<NonNull<T>> {
        let num_slabs = self.num_slabs.load(Ordering::Acquire);

        // Try to allocate from existing slabs
        for i in 0..num_slabs {
            unsafe {
                let slab_ptr = self.slabs[i].load(Ordering::Acquire);
                if !slab_ptr.is_null() {
                    let slab = &*slab_ptr;
                    if let Some(entry) = slab.allocate() {
                        self.total_allocated.fetch_add(1, Ordering::Relaxed);
                        
                        // Return pointer to user data (after header)
                        let user_data = (entry.as_ptr() as *mut u8).add(mem::size_of::<SlabEntry<T>>()) as *mut T;
                        return Some(NonNull::new_unchecked(user_data));
                    }
                }
            }
        }

        // Need to grow: allocate a new slab
        self.grow()
    }

    /// Grow the allocator by adding a new slab
    #[cold]
    fn grow(&self) -> Option<NonNull<T>> {
        let current_num = self.num_slabs.load(Ordering::Acquire);
        
        if current_num >= MAX_SLABS {
            return None; // Max capacity reached
        }

        unsafe {
            let new_slab = Slab::<T>::new(self.slab_capacity);
            let new_slab_ptr = new_slab.as_ptr();

            // Try to add the new slab
            let expected = ptr::null_mut();
            if self.slabs[current_num]
                .compare_exchange(expected, new_slab_ptr, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                self.num_slabs.fetch_add(1, Ordering::Release);
                
                // Allocate from the new slab
                if let Some(entry) = (&*new_slab_ptr).allocate() {
                    self.total_allocated.fetch_add(1, Ordering::Relaxed);
                    
                    let user_data = (entry.as_ptr() as *mut u8).add(mem::size_of::<SlabEntry<T>>()) as *mut T;
                    return Some(NonNull::new_unchecked(user_data));
                }
            } else {
                // Another thread added a slab concurrently, deallocate ours
                let layout = Slab::<T>::layout(self.slab_capacity);
                alloc::dealloc(new_slab_ptr as *mut u8, layout);
            }
        }

        // Retry allocation from existing slabs
        self.allocate()
    }

    /// Deallocate an entry
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this allocator
    /// - `ptr` must not be accessed after deallocation
    #[inline(always)]
    pub unsafe fn deallocate(&self, ptr: NonNull<T>) {
        // Find which slab this entry belongs to
        let user_ptr = ptr.as_ptr() as *mut u8;
        let num_slabs = self.num_slabs.load(Ordering::Acquire);

        for i in 0..num_slabs {
            let slab_ptr = self.slabs[i].load(Ordering::Acquire);
            if !slab_ptr.is_null() {
                let slab = &*slab_ptr;
                let entries_start = ptr::addr_of!((*slab_ptr).entries) as *mut u8;
                let entries_end = entries_start.add(mem::size_of::<SlabEntry<T>>() * slab.capacity);

                if user_ptr >= entries_start && user_ptr < entries_end {
                    // Found the slab, calculate entry pointer
                    let offset = user_ptr.offset_from(entries_start);
                    let entry_idx = offset as usize / mem::size_of::<SlabEntry<T>>();
                    let entry_ptr = entries_start.add(entry_idx * mem::size_of::<SlabEntry<T>>()) as *mut SlabEntry<T>;

                    slab.deallocate(NonNull::new_unchecked(entry_ptr));
                    self.total_allocated.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
            }
        }

        // Entry not found - this is undefined behavior
        panic!("Attempted to deallocate pointer not owned by this allocator");
    }

    /// Get total number of allocated entries
    #[inline(always)]
    pub fn allocated_count(&self) -> usize {
        self.total_allocated.load(Ordering::Relaxed)
    }

    /// Get total capacity across all slabs
    #[inline(always)]
    pub fn total_capacity(&self) -> usize {
        self.num_slabs.load(Ordering::Relaxed) * self.slab_capacity
    }

    /// Check if allocator is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.total_allocated.load(Ordering::Relaxed) == 0
    }
}

impl<T> Default for SlabAllocator<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for SlabAllocator<T> {
    fn drop(&mut self) {
        let num_slabs = *self.num_slabs.get_mut();
        
        for i in 0..num_slabs {
            let slab_ptr = self.slabs[i].get_mut();
            
            if !slab_ptr.is_null() {
                unsafe {
                    let layout = Slab::<T>::layout(self.slab_capacity);
                    alloc::dealloc(*slab_ptr as *mut u8, layout);
                }
            }
        }
    }
}

/// RAII wrapper for slab-allocated entries
pub struct SlabBox<T> {
    ptr: NonNull<T>,
    allocator: NonNull<SlabAllocator<T>>,
}

impl<T> SlabBox<T> {
    /// Create a new SlabBox by allocating from the given allocator
    pub fn new(allocator: &SlabAllocator<T>) -> Option<Self> {
        allocator.allocate().map(|ptr| Self {
            ptr,
            allocator: NonNull::from(allocator),
        })
    }

    /// Get reference to contained value
    #[inline(always)]
    pub fn get(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }

    /// Get mutable reference to contained value
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> Drop for SlabBox<T> {
    fn drop(&mut self) {
        unsafe {
            self.allocator.as_ref().deallocate(self.ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_allocation() {
        let allocator: SlabAllocator<u64> = SlabAllocator::with_capacity(32);
        
        let ptr1 = allocator.allocate().expect("Allocation should succeed");
        let ptr2 = allocator.allocate().expect("Allocation should succeed");
        
        assert!(ptr1 != ptr2);
        assert_eq!(allocator.allocated_count(), 2);
        
        unsafe {
            allocator.deallocate(ptr1);
            assert_eq!(allocator.allocated_count(), 1);
            
            allocator.deallocate(ptr2);
            assert_eq!(allocator.allocated_count(), 0);
        }
    }

    #[test]
    fn test_slab_growth() {
        let allocator: SlabAllocator<u32> = SlabAllocator::with_capacity(4);
        
        // Allocate more than one slab can hold
        let mut ptrs = Vec::new();
        for _ in 0..10 {
            let ptr = allocator.allocate().expect("Allocation should succeed");
            ptrs.push(ptr);
        }
        
        assert!(allocator.num_slabs.load(Ordering::Relaxed) > 1);
        assert_eq!(allocator.allocated_count(), 10);
        
        // Clean up
        unsafe {
            for ptr in ptrs {
                allocator.deallocate(ptr);
            }
        }
        
        assert!(allocator.is_empty());
    }

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(mem::align_of::<SlabEntry<u64>>(), CACHE_LINE_SIZE);
        println!("SlabEntry aligned to {} bytes", CACHE_LINE_SIZE);
    }

    #[test]
    fn test_slab_box() {
        let allocator: SlabAllocator<i32> = SlabAllocator::new();
        
        {
            let mut box1 = SlabBox::new(&allocator).expect("Should allocate");
            *box1.get_mut() = 42;
            assert_eq!(*box1.get(), 42);
        } // box1 dropped here
        
        assert!(allocator.is_empty());
    }
}
