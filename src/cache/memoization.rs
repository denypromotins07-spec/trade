//! Cache - Memoization Layer
//! 
//! Implements a microsecond memoization layer for expensive technical indicators
//! and risk calculations, utilizing atomic read-write locks for concurrent access
//! by UI and execution threads. Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::time::Duration;

/// Maximum memoization entries
const MAX_MEMO_ENTRIES: usize = 8192;

/// Entry validity duration in nanoseconds (default 1 second)
const DEFAULT_TTL_NS: u64 = 1_000_000_000;

/// Memoization entry with atomic state
#[repr(C, align(64))]
pub struct MemoEntry<T: Clone + Default> {
    /// Input hash (for cache key)
    pub input_hash: u64,
    /// Computed result
    pub result: UnsafeCell<T>,
    /// Creation timestamp
    created_ns: AtomicU64,
    /// Time-to-live in nanoseconds
    ttl_ns: u64,
    /// Access count (for LRU approximation)
    access_count: AtomicU64,
    /// Entry is being computed (lock flag)
    computing: AtomicBool,
    /// Entry is valid
    valid: AtomicBool,
}

impl<T: Clone + Default> MemoEntry<T> {
    const fn new() -> Self {
        Self {
            input_hash: 0,
            result: UnsafeCell::new(T::default()),
            created_ns: AtomicU64::new(0),
            ttl_ns: DEFAULT_TTL_NS,
            access_count: AtomicU64::new(0),
            computing: AtomicBool::new(false),
            valid: AtomicBool::new(false),
        }
    }
    
    #[inline(always)]
    fn is_expired(&self, current_ns: u64) -> bool {
        if !self.valid.load(Ordering::Acquire) {
            return true;
        }
        let age = current_ns.wrapping_sub(self.created_ns.load(Ordering::Acquire));
        age > self.ttl_ns
    }
    
    #[inline(always)]
    fn try_acquire_compute_lock(&self) -> bool {
        self.computing
            .compare_exchange_weak(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }
    
    #[inline(always)]
    fn release_compute_lock(&self) {
        self.computing.store(false, Ordering::Release);
    }
    
    #[inline(always)]
    fn store_result(&self, hash: u64, result: T, ttl_ns: u64) {
        self.input_hash.store(hash, Ordering::Release);
        unsafe {
            *self.result.get() = result;
        }
        self.created_ns.store(get_time_ns(), Ordering::Release);
        self.ttl_ns = ttl_ns;
        self.access_count.store(1, Ordering::Release);
        self.valid.store(true, Ordering::Release);
    }
}

/// Atomic read-write lock for concurrent access
#[repr(C, align(64))]
pub struct RwLockStats {
    /// Number of active readers
    readers: AtomicUsize,
    /// Writer is waiting
    writer_waiting: AtomicBool,
    /// Writer is active
    writer_active: AtomicBool,
    /// Read count
    read_count: AtomicU64,
    /// Write count
    write_count: AtomicU64,
    /// Contention count
    contention_count: AtomicU64,
}

impl RwLockStats {
    const fn new() -> Self {
        Self {
            readers: AtomicUsize::new(0),
            writer_waiting: AtomicBool::new(false),
            writer_active: AtomicBool::new(false),
            read_count: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            contention_count: AtomicU64::new(0),
        }
    }
    
    #[inline(always)]
    fn read_lock(&self) {
        // Spin while writer is active or writer is waiting (writer preference)
        while self.writer_active.load(Ordering::Acquire) 
            || self.writer_waiting.load(Ordering::Acquire) 
        {
            self.contention_count.fetch_add(1, Ordering::Relaxed);
            std::hint::spin_loop();
        }
        
        self.readers.fetch_add(1, Ordering::AcqRel);
        self.read_count.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline(always)]
    fn read_unlock(&self) {
        self.readers.fetch_sub(1, Ordering::Release);
    }
    
    #[inline(always)]
    fn write_lock(&self) {
        self.writer_waiting.store(true, Ordering::Release);
        
        // Wait for all readers to finish
        while self.readers.load(Ordering::Acquire) > 0 {
            self.contention_count.fetch_add(1, Ordering::Relaxed);
            std::hint::spin_loop();
        }
        
        // Wait for other writers
        while self.writer_active.load(Ordering::Acquire) {
            self.contention_count.fetch_add(1, Ordering::Relaxed);
            std::hint::spin_loop();
        }
        
        self.writer_waiting.store(false, Ordering::Release);
        self.writer_active.store(true, Ordering::AcqRel);
        self.write_count.fetch_add(1, Ordering::Relaxed);
    }
    
    #[inline(always)]
    fn write_unlock(&self) {
        self.writer_active.store(false, Ordering::Release);
    }
    
    #[inline(always)]
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.read_count.load(Ordering::Relaxed),
            self.write_count.load(Ordering::Relaxed),
            self.contention_count.load(Ordering::Relaxed),
        )
    }
}

/// Memoization cache for expensive computations
#[repr(C, align(64))]
pub struct MemoCache<T: Clone + Default> {
    /// Entries array
    entries: [MemoEntry<T>; MAX_MEMO_ENTRIES],
    /// Global stats lock
    lock: RwLockStats,
    /// Total lookups
    lookups: AtomicU64,
    /// Cache hits
    hits: AtomicU64,
    /// Cache misses
    misses: AtomicU64,
    /// Computations performed
    computations: AtomicU64,
    /// Evictions due to capacity
    evictions: AtomicU64,
    /// Default TTL
    default_ttl_ns: u64,
}

impl<T: Clone + Default> MemoCache<T> {
    pub const fn new() -> Self {
        Self {
            entries: [MemoEntry::new(); MAX_MEMO_ENTRIES],
            lock: RwLockStats::new(),
            lookups: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            computations: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            default_ttl_ns: DEFAULT_TTL_NS,
        }
    }
    
    /// Get cached value or compute it
    #[inline(always)]
    pub fn get_or_compute<F>(&self, input_hash: u64, compute_fn: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.lookups.fetch_add(1, Ordering::Relaxed);
        let current_ns = get_time_ns();
        
        // Try read lock first
        self.lock.read_lock();
        
        // Search for existing valid entry
        let slot_idx = self.find_entry(input_hash);
        
        if let Some(idx) = slot_idx {
            let entry = &self.entries[idx];
            
            if !entry.is_expired(current_ns) {
                // Cache hit!
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                let result = unsafe { (*entry.result.get()).clone() };
                self.lock.read_unlock();
                return result;
            }
        }
        
        self.lock.read_unlock();
        self.misses.fetch_add(1, Ordering::Relaxed);
        
        // Need to compute - acquire write lock
        self.lock.write_lock();
        
        // Double-check after acquiring write lock (another thread may have computed)
        let current_ns = get_time_ns();
        if let Some(idx) = self.find_entry(input_hash) {
            let entry = &self.entries[idx];
            if !entry.is_expired(current_ns) {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                let result = unsafe { (*entry.result.get()).clone() };
                self.lock.write_unlock();
                self.hits.fetch_add(1, Ordering::Relaxed);
                return result;
            }
        }
        
        // Find or allocate slot
        let slot = self.allocate_slot(input_hash);
        
        // Mark as computing
        if self.entries[slot].try_acquire_compute_lock() {
            // Compute the value
            self.computations.fetch_add(1, Ordering::Relaxed);
            let result = compute_fn();
            
            // Store result
            self.entries[slot].store_result(input_hash, result.clone(), self.default_ttl_ns);
            self.entries[slot].release_compute_lock();
            
            self.lock.write_unlock();
            result
        } else {
            // Another thread is computing, wait for it
            self.lock.write_unlock();
            
            // Spin wait for computation to complete
            while self.entries[slot].computing.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            
            // Return the computed value
            unsafe { (*self.entries[slot].result.get()).clone() }
        }
    }
    
    /// Get cached value without computing
    #[inline(always)]
    pub fn get(&self, input_hash: u64) -> Option<T> {
        self.lookups.fetch_add(1, Ordering::Relaxed);
        let current_ns = get_time_ns();
        
        self.lock.read_lock();
        
        if let Some(idx) = self.find_entry(input_hash) {
            let entry = &self.entries[idx];
            
            if !entry.is_expired(current_ns) {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                let result = unsafe { (*entry.result.get()).clone() };
                self.lock.read_unlock();
                return Some(result);
            }
        }
        
        self.lock.read_unlock();
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
    
    /// Insert pre-computed value
    #[inline(always)]
    pub fn insert(&self, input_hash: u64, value: T, ttl_ns: Option<u64>) {
        self.lock.write_lock();
        
        let slot = self.allocate_slot(input_hash);
        let ttl = ttl_ns.unwrap_or(self.default_ttl_ns);
        
        self.entries[slot].store_result(input_hash, value, ttl);
        
        self.lock.write_unlock();
    }
    
    /// Invalidate entry by hash
    #[inline(always)]
    pub fn invalidate(&self, input_hash: u64) -> bool {
        self.lock.write_lock();
        
        if let Some(idx) = self.find_entry(input_hash) {
            self.entries[idx].valid.store(false, Ordering::Release);
            self.lock.write_unlock();
            return true;
        }
        
        self.lock.write_unlock();
        false
    }
    
    /// Clear all entries
    #[inline(always)]
    pub fn clear(&self) {
        self.lock.write_lock();
        
        for entry in &self.entries {
            entry.valid.store(false, Ordering::Release);
        }
        
        self.lock.write_unlock();
    }
    
    /// Find entry by hash
    #[inline(always)]
    fn find_entry(&self, hash: u64) -> Option<usize> {
        // Simple linear probe for now (could use better hashing)
        let start = (hash as usize) % MAX_MEMO_ENTRIES;
        
        for i in 0..MAX_MEMO_ENTRIES {
            let idx = (start + i) % MAX_MEMO_ENTRIES;
            let entry = &self.entries[idx];
            
            if entry.input_hash.load(Ordering::Acquire) == hash && entry.valid.load(Ordering::Acquire) {
                return Some(idx);
            }
            
            // Stop at empty slot
            if !entry.valid.load(Ordering::Acquire) && entry.input_hash.load(Ordering::Acquire) == 0 {
                break;
            }
        }
        
        None
    }
    
    /// Allocate slot for new entry
    #[inline(always)]
    fn allocate_slot(&self, _input_hash: u64) -> usize {
        // Simple strategy: use hash modulo as starting point
        // In production, would use better eviction policy
        
        let mut min_access = u64::MAX;
        let mut candidate = 0;
        
        for i in 0..MAX_MEMO_ENTRIES {
            let entry = &self.entries[i];
            
            // Prefer invalid/expired entries
            if !entry.valid.load(Ordering::Acquire) {
                return i;
            }
            
            // Track least accessed entry
            let access = entry.access_count.load(Ordering::Relaxed);
            if access < min_access {
                min_access = access;
                candidate = i;
            }
        }
        
        // Evict candidate
        self.evictions.fetch_add(1, Ordering::Relaxed);
        candidate
    }
    
    /// Get statistics
    #[inline(always)]
    pub fn stats(&self) -> (u64, u64, u64, u64, u64, u64, u64) {
        let (reads, writes, contentions) = self.lock.stats();
        (
            self.lookups.load(Ordering::Relaxed),
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.computations.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed),
            reads,
            contentions,
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
    
    /// Set default TTL
    #[inline(always)]
    pub fn set_ttl(&mut self, ttl_ns: u64) {
        self.default_ttl_ns = ttl_ns;
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
    fn test_memo_cache_basic() {
        let cache = MemoCache::<i64>::new();
        
        let compute_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let compute_count_clone = compute_count.clone();
        
        // First call - should compute
        let result = cache.get_or_compute(123, || {
            compute_count_clone.fetch_add(1, Ordering::Relaxed);
            42i64
        });
        assert_eq!(result, 42);
        
        // Second call - should hit cache
        let result = cache.get_or_compute(123, || {
            compute_count_clone.fetch_add(1, Ordering::Relaxed);
            99i64
        });
        assert_eq!(result, 42); // Still 42 from cache
        
        // Verify only one computation
        assert_eq!(compute_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = MemoCache::<i64>::new();
        
        cache.insert(456, 100, Some(100_000_000)); // 100ms TTL
        
        assert_eq!(cache.get(456), Some(100));
        
        // Invalidate
        assert!(cache.invalidate(456));
        assert_eq!(cache.get(456), None);
    }

    #[test]
    fn test_hit_rate() {
        let cache = MemoCache::<i64>::new();
        
        cache.insert(1, 10, None);
        
        cache.get(1); // Hit
        cache.get(1); // Hit
        cache.get(2); // Miss
        
        assert!(cache.hit_rate() > 0.5);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let cache = std::sync::Arc::new(MemoCache::<i64>::new());
        let mut handles = vec![];
        
        for i in 0..4 {
            let cache_clone = cache.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let hash = (i * 100 + j) as u64;
                    cache_clone.get_or_compute(hash, || hash as i64 * 10);
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Should have completed without deadlock
        let stats = cache.stats();
        assert!(stats.0 > 0); // Some lookups occurred
    }
}
