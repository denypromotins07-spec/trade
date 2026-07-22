//! Cache - LRU Slab Allocator
//! 
//! Implements a custom, lock-free LRU cache using a contiguous memory slab
//! and intrusive linked lists to store recent order states without triggering
//! heap allocations or fragmentation. Optimized for AMD Ryzen AI 5 microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::ptr::NonNull;

/// Maximum cache entries (power of 2 for efficient modulo)
const MAX_CACHE_ENTRIES: usize = 16384; // 16K entries

/// Cache entry state flags
const ENTRY_EMPTY: u8 = 0;
const ENTRY_VALID: u8 = 1;
const ENTRY_DIRTY: u8 = 2;
const ENTRY_PINNED: u8 = 4;

/// Cache entry with intrusive linked list pointers
#[repr(C, align(64))]
pub struct CacheEntry<T: Clone + Default> {
    /// Entry key (hash)
    pub key: u64,
    /// Entry value
    pub value: UnsafeCell<T>,
    /// Previous entry index in LRU list
    prev: AtomicUsize,
    /// Next entry index in LRU list
    next: AtomicUsize,
    /// State flags (VALID, DIRTY, PINNED)
    flags: AtomicU8,
    /// Access timestamp (nanoseconds)
    last_access_ns: AtomicU64,
    /// Hash of key for quick comparison
    key_hash: u64,
}

impl<T: Clone + Default> CacheEntry<T> {
    const fn new() -> Self {
        Self {
            key: 0,
            value: UnsafeCell::new(T::default()),
            prev: AtomicUsize::new(usize::MAX),
            next: AtomicUsize::new(usize::MAX),
            flags: AtomicU8::new(ENTRY_EMPTY),
            last_access_ns: AtomicU64::new(0),
            key_hash: 0,
        }
    }
    
    #[inline(always)]
    fn is_valid(&self) -> bool {
        self.flags.load(Ordering::Acquire) & ENTRY_VALID != 0
    }
    
    #[inline(always)]
    fn is_pinned(&self) -> bool {
        self.flags.load(Ordering::Acquire) & ENTRY_PINNED != 0
    }
    
    #[inline(always)]
    fn is_dirty(&self) -> bool {
        self.flags.load(Ordering::Acquire) & ENTRY_DIRTY != 0
    }
    
    #[inline(always)]
    fn set_valid(&self) {
        let mut flags = self.flags.load(Ordering::Acquire);
        loop {
            let new_flags = flags | ENTRY_VALID;
            match self.flags.compare_exchange_weak(
                flags,
                new_flags,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => flags = x,
            }
        }
    }
    
    #[inline(always)]
    fn set_dirty(&self) {
        let mut flags = self.flags.load(Ordering::Acquire);
        loop {
            let new_flags = flags | ENTRY_DIRTY;
            match self.flags.compare_exchange_weak(
                flags,
                new_flags,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => flags = x,
            }
        }
    }
    
    #[inline(always)]
    fn clear_dirty(&self) {
        let mut flags = self.flags.load(Ordering::Acquire);
        loop {
            let new_flags = flags & !ENTRY_DIRTY;
            match self.flags.compare_exchange_weak(
                flags,
                new_flags,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => flags = x,
            }
        }
    }
    
    #[inline(always)]
    fn pin(&self) {
        let mut flags = self.flags.load(Ordering::Acquire);
        loop {
            let new_flags = flags | ENTRY_PINNED;
            match self.flags.compare_exchange_weak(
                flags,
                new_flags,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => flags = x,
            }
        }
    }
    
    #[inline(always)]
    fn unpin(&self) {
        let mut flags = self.flags.load(Ordering::Acquire);
        loop {
            let new_flags = flags & !ENTRY_PINNED;
            match self.flags.compare_exchange_weak(
                flags,
                new_flags,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => flags = x,
            }
        }
    }
    
    #[inline(always)]
    fn clear(&self) {
        self.key.store(0, Ordering::Release);
        self.flags.store(ENTRY_EMPTY, Ordering::Release);
        self.prev.store(usize::MAX, Ordering::Release);
        self.next.store(usize::MAX, Ordering::Release);
    }
}

use std::sync::atomic::AtomicU8;

/// Lock-free LRU cache using slab allocation
#[repr(C, align(64))]
pub struct LruSlabCache<T: Clone + Default> {
    /// Contiguous memory slab for entries
    slab: [CacheEntry<T>; MAX_CACHE_ENTRIES],
    /// Head of LRU list (most recently used)
    lru_head: AtomicUsize,
    /// Tail of LRU list (least recently used, eviction candidate)
    lru_tail: AtomicUsize,
    /// Free list head
    free_head: AtomicUsize,
    /// Number of valid entries
    count: AtomicUsize,
    /// Cache statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    /// Capacity (may be less than MAX_CACHE_ENTRIES)
    capacity: usize,
}

impl<T: Clone + Default> LruSlabCache<T> {
    pub const fn new() -> Self {
        Self {
            slab: [CacheEntry::new(); MAX_CACHE_ENTRIES],
            lru_head: AtomicUsize::new(usize::MAX),
            lru_tail: AtomicUsize::new(usize::MAX),
            free_head: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            capacity: MAX_CACHE_ENTRIES,
        }
    }
    
    /// Initialize the free list
    #[inline(always)]
    pub fn init(&self) {
        // Link all entries in free list
        for i in 0..MAX_CACHE_ENTRIES - 1 {
            self.slab[i].next.store(i + 1, Ordering::Release);
            self.slab[i].prev.store(usize::MAX, Ordering::Release);
        }
        self.slab[MAX_CACHE_ENTRIES - 1].next.store(usize::MAX, Ordering::Release);
    }
    
    /// Get value by key (read-only, updates access time)
    #[inline(always)]
    pub fn get(&self, key: u64) -> Option<T> {
        let key_hash = fxhash::fxhash64(&key);
        
        // Search for key (linear probe through LRU list for simplicity)
        let mut current = self.lru_head.load(Ordering::Acquire);
        while current != usize::MAX {
            let entry = unsafe { self.slab.get_unchecked(current) };
            
            if entry.key_hash == key_hash && entry.is_valid() {
                // Found it! Update access time and move to head
                entry.last_access_ns.store(get_time_ns(), Ordering::Release);
                self.hits.fetch_add(1, Ordering::Relaxed);
                
                // Move to front of LRU (if not already there)
                if current != self.lru_head.load(Ordering::Acquire) {
                    self.move_to_front(current);
                }
                
                // Return cloned value
                return Some(unsafe { (*entry.value.get()).clone() });
            }
            
            current = entry.next.load(Ordering::Acquire);
        }
        
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Insert or update entry
    #[inline(always)]
    pub fn insert(&self, key: u64, value: T) -> bool {
        let key_hash = fxhash::fxhash64(&key);
        let timestamp = get_time_ns();
        
        // First check if key exists
        let mut current = self.lru_head.load(Ordering::Acquire);
        while current != usize::MAX {
            let entry = unsafe { self.slab.get_unchecked(current) };
            
            if entry.key_hash == key_hash && entry.is_valid() {
                // Update existing entry
                unsafe {
                    *entry.value.get() = value;
                }
                entry.last_access_ns.store(timestamp, Ordering::Release);
                entry.set_valid();
                entry.set_dirty();
                
                // Move to front
                self.move_to_front(current);
                return true;
            }
            
            current = entry.next.load(Ordering::Acquire);
        }
        
        // Key doesn't exist - need to allocate new entry
        let new_idx = self.allocate_entry(key, key_hash, value, timestamp);
        if new_idx == usize::MAX {
            return false; // Cache full and couldn't evict
        }
        
        true
    }
    
    /// Allocate a new entry (evicting if necessary)
    #[inline(always)]
    fn allocate_entry(
        &self,
        key: u64,
        key_hash: u64,
        value: T,
        timestamp: u64,
    ) -> usize {
        // Try to get from free list first
        let mut idx = self.free_head.load(Ordering::Acquire);
        
        if idx != usize::MAX {
            // Remove from free list
            let entry = unsafe { self.slab.get_unchecked(idx) };
            let next_free = entry.next.load(Ordering::Acquire);
            
            if self.free_head.compare_exchange_weak(
                idx,
                next_free,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                // Initialize entry
                entry.key.store(key, Ordering::Release);
                entry.key_hash.store(key_hash, Ordering::Release);
                unsafe {
                    *entry.value.get() = value;
                }
                entry.last_access_ns.store(timestamp, Ordering::Release);
                entry.prev.store(usize::MAX, Ordering::Release);
                entry.next.store(self.lru_head.load(Ordering::Acquire), Ordering::Release);
                entry.flags.store(ENTRY_VALID | ENTRY_DIRTY, Ordering::Release);
                
                // Add to LRU head
                self.add_to_lru_head(idx);
                self.count.fetch_add(1, Ordering::Relaxed);
                
                return idx;
            }
        }
        
        // No free entries - evict LRU tail
        if self.evict_lru() {
            // Retry allocation after eviction
            return self.allocate_entry(key, key_hash, value, timestamp);
        }
        
        usize::MAX // Failed to allocate
    }
    
    /// Evict the least recently used entry
    #[inline(always)]
    fn evict_lru(&self) -> bool {
        let tail = self.lru_tail.load(Ordering::Acquire);
        
        if tail == usize::MAX {
            return false; // Empty cache
        }
        
        let entry = unsafe { self.slab.get_unchecked(tail) };
        
        // Can't evict pinned entries
        if entry.is_pinned() {
            // Try to find non-pinned entry
            return self.evict_lru_recursive(tail);
        }
        
        // Remove from LRU list
        self.remove_from_lru(tail);
        
        // Clear entry and add to free list
        let old_key = entry.key.load(Ordering::Acquire);
        entry.clear();
        
        // Add to free list head
        let free_head = self.free_head.load(Ordering::Acquire);
        entry.next.store(free_head, Ordering::Release);
        entry.prev.store(usize::MAX, Ordering::Release);
        
        self.free_head.store(tail, Ordering::Release);
        self.count.fetch_sub(1, Ordering::Relaxed);
        self.evictions.fetch_add(1, Ordering::Relaxed);
        
        true
    }
    
    /// Recursive eviction skipping pinned entries
    #[inline(always)]
    fn evict_lru_recursive(&self, start_idx: usize) -> bool {
        let mut current = start_idx;
        let mut visited = 0;
        
        while visited < self.capacity {
            let entry = unsafe { self.slab.get_unchecked(current) };
            let prev = entry.prev.load(Ordering::Acquire);
            
            if !entry.is_pinned() {
                // Found evictable entry
                self.remove_from_lru(current);
                entry.clear();
                
                let free_head = self.free_head.load(Ordering::Acquire);
                entry.next.store(free_head, Ordering::Release);
                entry.prev.store(usize::MAX, Ordering::Release);
                
                self.free_head.store(current, Ordering::Release);
                self.count.fetch_sub(1, Ordering::Relaxed);
                self.evictions.fetch_add(1, Ordering::Relaxed);
                
                return true;
            }
            
            if prev == usize::MAX {
                break; // Reached end
            }
            
            current = prev;
            visited += 1;
        }
        
        false // All entries pinned
    }
    
    /// Add entry to LRU head
    #[inline(always)]
    fn add_to_lru_head(&self, idx: usize) {
        let entry = unsafe { self.slab.get_unchecked(idx) };
        let old_head = self.lru_head.load(Ordering::Acquire);
        
        entry.next.store(old_head, Ordering::Release);
        entry.prev.store(usize::MAX, Ordering::Release);
        
        if old_head != usize::MAX {
            let old_head_entry = unsafe { self.slab.get_unchecked(old_head) };
            old_head_entry.prev.store(idx, Ordering::Release);
        }
        
        self.lru_head.store(idx, Ordering::Release);
        
        // If this is the only entry, it's also the tail
        if old_head == usize::MAX {
            self.lru_tail.store(idx, Ordering::Release);
        }
    }
    
    /// Remove entry from LRU list
    #[inline(always)]
    fn remove_from_lru(&self, idx: usize) {
        let entry = unsafe { self.slab.get_unchecked(idx) };
        let prev = entry.prev.load(Ordering::Acquire);
        let next = entry.next.load(Ordering::Acquire);
        
        // Update previous entry's next pointer
        if prev != usize::MAX {
            let prev_entry = unsafe { self.slab.get_unchecked(prev) };
            prev_entry.next.store(next, Ordering::Release);
        } else {
            // This was the head
            self.lru_head.store(next, Ordering::Release);
        }
        
        // Update next entry's prev pointer
        if next != usize::MAX {
            let next_entry = unsafe { self.slab.get_unchecked(next) };
            next_entry.prev.store(prev, Ordering::Release);
        } else {
            // This was the tail
            self.lru_tail.store(prev, Ordering::Release);
        }
        
        entry.prev.store(usize::MAX, Ordering::Release);
        entry.next.store(usize::MAX, Ordering::Release);
    }
    
    /// Move entry to front of LRU list
    #[inline(always)]
    fn move_to_front(&self, idx: usize) {
        self.remove_from_lru(idx);
        self.add_to_lru_head(idx);
    }
    
    /// Pin an entry to prevent eviction
    #[inline(always)]
    pub fn pin(&self, key: u64) -> bool {
        let key_hash = fxhash::fxhash64(&key);
        
        let mut current = self.lru_head.load(Ordering::Acquire);
        while current != usize::MAX {
            let entry = unsafe { self.slab.get_unchecked(current) };
            
            if entry.key_hash == key_hash && entry.is_valid() {
                entry.pin();
                return true;
            }
            
            current = entry.next.load(Ordering::Acquire);
        }
        
        false
    }
    
    /// Unpin an entry
    #[inline(always)]
    pub fn unpin(&self, key: u64) -> bool {
        let key_hash = fxhash::fxhash64(&key);
        
        let mut current = self.lru_head.load(Ordering::Acquire);
        while current != usize::MAX {
            let entry = unsafe { self.slab.get_unchecked(current) };
            
            if entry.key_hash == key_hash && entry.is_valid() {
                entry.unpin();
                return true;
            }
            
            current = entry.next.load(Ordering::Acquire);
        }
        
        false
    }
    
    /// Get cache statistics
    #[inline(always)]
    pub fn stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            self.count.load(Ordering::Relaxed) as u64,
            self.capacity as u64,
        )
    }
    
    /// Get hit rate
    #[inline(always)]
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Get current time in nanoseconds
#[inline(always)]
fn get_time_ns() -> u64 {
    use std::time::Instant;
    static START: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insert_get() {
        let cache = LruSlabCache::<u64>::new();
        cache.init();
        
        cache.insert(1, 100);
        cache.insert(2, 200);
        
        assert_eq!(cache.get(1), Some(100));
        assert_eq!(cache.get(2), Some(200));
        assert_eq!(cache.get(3), None);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = LruSlabCache::<u64>::new();
        cache.init();
        
        // Fill cache
        for i in 0..MAX_CACHE_ENTRIES {
            cache.insert(i as u64, i as u64 * 100);
        }
        
        // Access first entry to make it MRU
        cache.get(0);
        
        // Insert new entry - should evict entry 1 (LRU)
        cache.insert(MAX_CACHE_ENTRIES as u64, MAX_CACHE_ENTRIES as u64 * 100);
        
        // Entry 0 should still exist (was accessed)
        assert_eq!(cache.get(0), Some(0));
    }

    #[test]
    fn test_pin_prevents_eviction() {
        let cache = LruSlabCache::<u64>::new();
        cache.init();
        
        cache.insert(1, 100);
        cache.pin(1);
        
        // Fill rest of cache
        for i in 2..MAX_CACHE_ENTRIES + 1 {
            cache.insert(i as u64, i as u64 * 100);
        }
        
        // Pinned entry should still exist
        assert_eq!(cache.get(1), Some(100));
    }

    #[test]
    fn test_hit_rate() {
        let cache = LruSlabCache::<u64>::new();
        cache.init();
        
        cache.insert(1, 100);
        
        cache.get(1); // Hit
        cache.get(1); // Hit
        cache.get(2); // Miss
        
        assert!(cache.hit_rate() > 0.5);
    }
}
