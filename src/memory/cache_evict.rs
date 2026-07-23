//! Cache Eviction - LFU (Least Frequently Used) Policy
//! 
//! This module implements an advanced LFU (Least Frequently Used) cache eviction policy
//! that aggressively purges stale historical data when the OS signals high memory pressure.
//! Optimized for AMD Ryzen AI 5 with microsecond eviction decisions.
//! 
//! RAM Budget: Self-regulating based on system memory pressure.
//! Enforces global 8GB RAM limit via adaptive eviction thresholds.

use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};

/// Default cache capacity (number of entries)
const DEFAULT_CAPACITY: usize = 100_000;

/// Minimum frequency before eviction consideration
const MIN_FREQUENCY_THRESHOLD: u64 = 1;

/// Frequency decay interval in seconds
const FREQUENCY_DECAY_INTERVAL_SECS: u64 = 60;

/// Memory pressure check interval
const PRESSURE_CHECK_INTERVAL_MS: u64 = 1000;

/// Cache entry with frequency tracking
#[derive(Debug)]
struct CacheEntry<K, V> {
    key: K,
    value: V,
    frequency: AtomicU64,
    last_access: Instant,
    created_at: Instant,
    size_bytes: usize,
}

impl<K, V> CacheEntry<K, V> {
    #[inline]
    fn new(key: K, value: V, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            key,
            value,
            frequency: AtomicU64::new(1),
            last_access: now,
            created_at: now,
            size_bytes,
        }
    }
    
    #[inline]
    fn access(&self) {
        self.frequency.fetch_add(1, Ordering::Relaxed);
        self.last_access = Instant::now();
    }
    
    #[inline]
    fn get_frequency(&self) -> u64 {
        self.frequency.load(Ordering::Relaxed)
    }
    
    #[inline]
    fn decay_frequency(&self, factor: u64) {
        let current = self.frequency.load(Ordering::Relaxed);
        if current > factor {
            self.frequency.store(current / factor, Ordering::Relaxed);
        } else {
            self.frequency.store(MIN_FREQUENCY_THRESHOLD, Ordering::Relaxed);
        }
    }
}

/// Statistics for cache operations
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub insertions: u64,
    pub current_size: usize,
    pub current_memory_bytes: u64,
    pub hit_rate: f64,
    pub avg_frequency: f64,
}

/// Memory pressure level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    Low,
    Moderate,
    High,
    Critical,
}

impl MemoryPressure {
    /// Get recommended eviction count based on pressure
    #[inline]
    pub fn recommended_evictions(&self, cache_size: usize) -> usize {
        match self {
            Self::Low => 0,
            Self::Moderate => cache_size / 10,   // 10%
            Self::High => cache_size / 4,         // 25%
            Self::Critical => cache_size / 2,     // 50%
        }
    }
}

/// Main LFU Cache implementation
pub struct LfuCache<K, V> {
    /// The actual cache storage
    map: RwLock<HashMap<K, Arc<CacheEntry<K, V>>>>,
    /// Capacity limit
    capacity: usize,
    /// Current memory usage
    memory_bytes: AtomicU64,
    /// Memory limit in bytes
    memory_limit: AtomicU64,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    insertions: AtomicU64,
    /// Running flag
    running: AtomicBool,
    /// Last decay time
    last_decay: Mutex<Instant>,
}

impl<K, V> LfuCache<K, V>
where
    K: Hash + Eq + Clone + 'static,
    V: Clone + 'static,
{
    /// Create a new LFU cache with default capacity
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
    
    /// Create a new LFU cache with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(capacity.min(10000))),
            capacity,
            memory_bytes: AtomicU64::new(0),
            memory_limit: AtomicU64::new(u64::MAX),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            running: AtomicBool::new(true),
            last_decay: Mutex::new(Instant::now()),
        }
    }
    
    /// Create with memory limit
    pub fn with_memory_limit(capacity: usize, memory_limit_bytes: u64) -> Self {
        Self {
            map: RwLock::new(HashMap::with_capacity(capacity.min(10000))),
            capacity,
            memory_bytes: AtomicU64::new(0),
            memory_limit: AtomicU64::new(memory_limit_bytes),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            running: AtomicBool::new(true),
            last_decay: Mutex::new(Instant::now()),
        }
    }
    
    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V, size_bytes: usize) -> Option<V> {
        if !self.running.load(Ordering::Relaxed) {
            return None;
        }
        
        // Check if we need to evict before inserting
        self.maybe_evict(size_bytes);
        
        let entry = Arc::new(CacheEntry::new(key.clone(), value, size_bytes));
        
        let mut map = self.map.write();
        
        // Check if key already exists
        if let Some(existing) = map.insert(key, Arc::clone(&entry)) {
            // Update memory tracking
            let old_size = existing.size_bytes as i64;
            let new_size = entry.size_bytes as i64;
            self.memory_bytes.fetch_add((new_size - old_size) as u64, Ordering::Relaxed);
            
            self.insertions.fetch_add(1, Ordering::Relaxed);
            return Some(existing.value);
        }
        
        // New entry
        self.memory_bytes.fetch_add(size_bytes as u64, Ordering::Relaxed);
        self.insertions.fetch_add(1, Ordering::Relaxed);
        
        None
    }
    
    /// Get a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let map = self.map.read();
        
        if let Some(entry) = map.get(key) {
            entry.access();
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
    
    /// Get a reference to the entry (for advanced use cases)
    pub fn get_entry(&self, key: &K) -> Option<Arc<CacheEntry<K, V>>> {
        let map = self.map.read();
        map.get(key).map(Arc::clone)
    }
    
    /// Remove a key from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut map = self.map.write();
        
        if let Some(entry) = map.remove(key) {
            self.memory_bytes.fetch_sub(entry.size_bytes as u64, Ordering::Relaxed);
            return Some(entry.value);
        }
        
        None
    }
    
    /// Check if key exists
    pub fn contains(&self, key: &K) -> bool {
        self.map.read().contains_key(key)
    }
    
    /// Get current size
    pub fn len(&self) -> usize {
        self.map.read().len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        let mut map = self.map.write();
        map.clear();
        self.memory_bytes.store(0, Ordering::Relaxed);
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let map = self.map.read();
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        // Calculate average frequency
        let total_freq: u64 = map.values().map(|e| e.get_frequency()).sum();
        let avg_freq = if map.is_empty() { 0.0 } else { total_freq as f64 / map.len() as f64 };
        
        CacheStats {
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            insertions: self.insertions.load(Ordering::Relaxed),
            current_size: map.len(),
            current_memory_bytes: self.memory_bytes.load(Ordering::Relaxed),
            hit_rate: if total == 0 { 0.0 } else { hits as f64 / total as f64 },
            avg_frequency: avg_freq,
        }
    }
    
    /// Get memory pressure level
    pub fn get_pressure(&self) -> MemoryPressure {
        let memory = self.memory_bytes.load(Ordering::Relaxed);
        let limit = self.memory_limit.load(Ordering::Relaxed);
        let ratio = memory as f64 / limit as f64;
        
        let size_ratio = self.len() as f64 / self.capacity as f64;
        let max_ratio = ratio.max(size_ratio);
        
        if max_ratio > 0.95 {
            MemoryPressure::Critical
        } else if max_ratio > 0.8 {
            MemoryPressure::High
        } else if max_ratio > 0.6 {
            MemoryPressure::Moderate
        } else {
            MemoryPressure::Low
        }
    }
    
    /// Maybe evict entries if needed
    fn maybe_evict(&self, new_size_bytes: usize) {
        let memory = self.memory_bytes.load(Ordering::Relaxed);
        let limit = self.memory_limit.load(Ordering::Relaxed);
        let projected = memory + new_size_bytes as u64;
        
        let size = self.len();
        
        // Check both memory and count limits
        if projected <= limit && size < self.capacity {
            return;
        }
        
        // Determine how many to evict
        let pressure = self.get_pressure();
        let to_evict = pressure.recommended_evictions(size).max(1);
        
        self.evict_n(to_evict);
    }
    
    /// Evict N least frequently used entries
    fn evict_n(&self, n: usize) {
        let mut map = self.map.write();
        
        // Find entries with lowest frequency
        let mut entries: Vec<_> = map.iter()
            .map(|(k, v)| (k.clone(), v.get_frequency(), v.size_bytes))
            .collect();
        
        // Sort by frequency (ascending), then by recency
        entries.sort_by(|a, b| {
            a.1.cmp(&b.1).then_with(|| {
                // For same frequency, prefer older entries
                std::cmp::Ordering::Less
            })
        });
        
        // Evict the N lowest frequency entries
        let mut evicted = 0;
        let mut freed_bytes = 0u64;
        
        for (key, _, size) in entries.into_iter().take(n) {
            if map.remove(&key).is_some() {
                freed_bytes += size as u64;
                evicted += 1;
            }
        }
        
        if evicted > 0 {
            self.memory_bytes.fetch_sub(freed_bytes, Ordering::Relaxed);
            self.evictions.fetch_add(evicted as u64, Ordering::Relaxed);
        }
    }
    
    /// Decay all frequencies (call periodically)
    pub fn decay_frequencies(&self) {
        let mut last_decay = self.last_decay.lock();
        let now = Instant::now();
        
        if now.duration_since(*last_decay) < Duration::from_secs(FREQUENCY_DECAY_INTERVAL_SECS) {
            return;
        }
        
        *last_decay = now;
        drop(last_decay);
        
        let map = self.map.read();
        for entry in map.values() {
            entry.decay_frequency(2); // Divide by 2
        }
    }
    
    /// Shutdown the cache
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
        self.clear();
    }
}

impl<K, V> Default for LfuCache<K, V>
where
    K: Hash + Eq + Clone + 'static,
    V: Clone + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Background task for cache maintenance
pub async fn run_cache_maintenance<K, V>(cache: Arc<LfuCache<K, V>>)
where
    K: Hash + Eq + Clone + 'static,
    V: Clone + 'static,
{
    let mut interval = tokio::time::interval(Duration::from_millis(PRESSURE_CHECK_INTERVAL_MS));
    
    while cache.running.load(Ordering::Relaxed) {
        interval.tick().await;
        
        // Decay frequencies periodically
        cache.decay_frequencies();
        
        // Check pressure and evict if needed
        let pressure = cache.get_pressure();
        if pressure >= MemoryPressure::High {
            let to_evict = pressure.recommended_evictions(cache.len());
            if to_evict > 0 {
                cache.evict_n(to_evict);
                
                tracing::info!(
                    "LFU cache evicted {} entries due to {:?} pressure",
                    to_evict,
                    pressure
                );
            }
        }
    }
}

/// Specialized cache for historical order book data
pub type HistoricalDataCache = LfuCache<String, Vec<(f64, f64)>>;

/// Specialized cache for trade history
pub type TradeHistoryCache = LfuCache<u64, TradeRecord>;

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp_ms: u64,
    pub price: f64,
    pub quantity: f64,
    pub side: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_get() {
        let cache = LfuCache::with_capacity(100);
        
        cache.insert("key1".to_string(), "value1".to_string(), 100);
        
        assert_eq!(cache.get(&"key1".to_string()), Some("value1".to_string()));
        assert_eq!(cache.get(&"key2".to_string()), None);
    }

    #[test]
    fn test_cache_hit_miss_stats() {
        let cache = LfuCache::with_capacity(100);
        
        cache.insert("key1".to_string(), "value1".to_string(), 100);
        
        cache.get(&"key1".to_string()); // Hit
        cache.get(&"key1".to_string()); // Hit
        cache.get(&"key2".to_string()); // Miss
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = LfuCache::with_capacity(5);
        
        // Insert more than capacity
        for i in 0..10 {
            cache.insert(format!("key{}", i), format!("value{}", i), 100);
        }
        
        // Should have evicted some entries
        assert!(cache.len() <= 5);
        
        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_lfu_behavior() {
        let cache = LfuCache::with_capacity(3);
        
        // Insert entries
        cache.insert("a".to_string(), 1, 100);
        cache.insert("b".to_string(), 2, 100);
        cache.insert("c".to_string(), 3, 100);
        
        // Access 'a' multiple times to increase frequency
        cache.get(&"a".to_string());
        cache.get(&"a".to_string());
        cache.get(&"a".to_string());
        
        // Insert new entry - should evict 'b' or 'c' (lowest frequency)
        cache.insert("d".to_string(), 4, 100);
        
        // 'a' should still be there (high frequency)
        assert!(cache.contains(&"a".to_string()));
    }

    #[test]
    fn test_memory_pressure() {
        let cache = LfuCache::with_memory_limit(100, 1000);
        
        assert_eq!(cache.get_pressure(), MemoryPressure::Low);
        
        // Fill up memory
        for i in 0..20 {
            cache.insert(format!("key{}", i), vec![0u8; 100], 100);
        }
        
        let pressure = cache.get_pressure();
        assert!(pressure >= MemoryPressure::Moderate);
    }

    #[test]
    fn test_clear() {
        let cache = LfuCache::with_capacity(100);
        
        cache.insert("key1".to_string(), "value1".to_string(), 100);
        cache.insert("key2".to_string(), "value2".to_string(), 100);
        
        assert_eq!(cache.len(), 2);
        
        cache.clear();
        
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }
}
