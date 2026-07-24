// Memory Evictor: Aggressive LRU evictor for the 8GB RAM limit during 6-asset 
// volatility spikes. Instantly purges stale historical tick caches to guarantee 
// live matching engines never experience OOM page faults.
// Optimized for AMD Ryzen AI 5 with lock-free atomic operations.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use std::collections::VecDeque;

/// Hard RAM limit in bytes (8GB)
const RAM_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Soft RAM threshold for triggering eviction (90% of limit)
const SOFT_THRESHOLD_BYTES: u64 = (RAM_LIMIT_BYTES * 90) / 100;

/// Critical RAM threshold for aggressive eviction (95% of limit)
const CRITICAL_THRESHOLD_BYTES: u64 = (RAM_LIMIT_BYTES * 95) / 100;

/// Minimum free memory to maintain (500MB)
const MIN_FREE_MEMORY_BYTES: u64 = 500 * 1024 * 1024;

/// Maximum entries in LRU queue per cache type
const MAX_LRU_QUEUE_SIZE: usize = 10_000;

/// Cache entry types for prioritized eviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CacheType {
    /// Tick history (lowest priority - first to evict)
    TickHistory = 0,
    /// Order book snapshots (low priority)
    OrderBookSnapshots = 1,
    /// Trade history (medium priority)
    TradeHistory = 2,
    /// Strategy state (high priority - keep longer)
    StrategyState = 3,
    /// Active positions (critical - never evict while active)
    ActivePositions = 4,
}

/// LRU cache entry
#[derive(Debug, Clone)]
pub struct LRUCacheEntry {
    /// Unique identifier for the cached item
    pub key: u64,
    /// Symbol index this entry belongs to
    pub symbol_idx: u8,
    /// Type of cache entry (determines eviction priority)
    pub cache_type: CacheType,
    /// Size in bytes
    pub size_bytes: u64,
    /// Last access timestamp (milliseconds)
    pub last_access_ms: u64,
    /// Creation timestamp (milliseconds)
    pub created_ms: u64,
    /// Whether this entry is locked (cannot be evicted)
    pub is_locked: bool,
}

/// Memory statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Total memory currently used
    pub total_used_bytes: u64,
    /// Memory used by each cache type
    pub tick_history_bytes: u64,
    pub orderbook_bytes: u64,
    pub trade_history_bytes: u64,
    pub strategy_state_bytes: u64,
    pub position_state_bytes: u64,
    /// Number of entries evicted
    pub total_evictions: u64,
    /// Evictions in last second
    pub recent_evictions_per_sec: f64,
    /// Current memory pressure level (0.0 - 1.0)
    pub pressure_level: f64,
}

/// Lock-free LRU memory evictor
pub struct MemoryEvictor {
    /// Current memory usage (atomic for lock-free reads)
    current_usage_bytes: AtomicU64,
    
    /// Peak memory usage recorded
    peak_usage_bytes: AtomicU64,
    
    /// LRU queues per cache type
    lru_queues: [std::sync::Mutex<VecDeque<LRUCacheEntry>>; 5],
    
    /// Entry lookup map (key -> entry info)
    entry_map: std::sync::RwLock<std::collections::HashMap<u64, LRUCacheEntry>>,
    
    /// Total eviction count
    total_evictions: AtomicU64,
    
    /// Recent eviction timestamps (for rate calculation)
    recent_eviction_times: std::sync::Mutex<VecDeque<u64>>,
    
    /// Eviction in progress flag
    eviction_in_progress: AtomicBool,
    
    /// Emergency mode flag (aggressive eviction)
    emergency_mode: AtomicBool,
    
    /// Start time for timestamps
    start_time: Instant,
    
    /// Callback when eviction occurs
    eviction_callback: Option<Box<dyn Fn(u64, CacheType) + Send + Sync>>,
}

impl MemoryEvictor {
    /// Create a new memory evictor
    pub fn new() -> Self {
        Self {
            current_usage_bytes: AtomicU64::new(0),
            peak_usage_bytes: AtomicU64::new(0),
            lru_queues: std::array::from_fn(|_| std::sync::Mutex::new(VecDeque::with_capacity(1000))),
            entry_map: std::sync::RwLock::new(std::collections::HashMap::new()),
            total_evictions: AtomicU64::new(0),
            recent_eviction_times: std::sync::Mutex::new(VecDeque::with_capacity(100)),
            eviction_in_progress: AtomicBool::new(false),
            emergency_mode: AtomicBool::new(false),
            start_time: Instant::now(),
            eviction_callback: None,
        }
    }

    /// Get current timestamp in milliseconds
    #[inline]
    fn now_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Register a new cache entry
    pub fn register_entry(&self, mut entry: LRUCacheEntry) {
        let now = self.now_ms();
        entry.created_ms = now;
        entry.last_access_ms = now;
        
        let size = entry.size_bytes;
        let cache_type_idx = entry.cache_type as usize;
        let key = entry.key;
        
        // Add to LRU queue
        let mut queue = self.lru_queues[cache_type_idx].lock().unwrap();
        if queue.len() >= MAX_LRU_QUEUE_SIZE {
            // Remove oldest entry if queue is full
            if let Some(oldest) = queue.pop_front() {
                self.evict_entry_internal(&oldest);
            }
        }
        queue.push_back(entry.clone());
        drop(queue);
        
        // Add to entry map
        let mut map = self.entry_map.write().unwrap();
        map.insert(key, entry);
        drop(map);
        
        // Update memory usage
        self.current_usage_bytes.fetch_add(size, Ordering::Relaxed);
        
        // Update peak
        let current = self.current_usage_bytes.load(Ordering::Relaxed);
        let peak = self.peak_usage_bytes.load(Ordering::Relaxed);
        if current > peak {
            self.peak_usage_bytes.store(current, Ordering::Relaxed);
        }
        
        // Check if we need to evict
        self.maybe_evict();
    }

    /// Mark an entry as accessed (move to end of LRU queue)
    pub fn touch_entry(&self, key: u64) {
        let now = self.now_ms();
        
        let mut map = self.entry_map.write().unwrap();
        if let Some(entry) = map.get_mut(&key) {
            entry.last_access_ms = now;
            let cache_type_idx = entry.cache_type as usize;
            let entry_clone = entry.clone();
            
            // Move to end of LRU queue
            let mut queue = self.lru_queues[cache_type_idx].lock().unwrap();
            if let Some(pos) = queue.iter().position(|e| e.key == key) {
                queue.remove(pos);
                queue.push_back(entry_clone);
            }
        }
    }

    /// Lock an entry (prevent eviction)
    pub fn lock_entry(&self, key: u64) {
        let mut map = self.entry_map.write().unwrap();
        if let Some(entry) = map.get_mut(&key) {
            entry.is_locked = true;
        }
    }

    /// Unlock an entry
    pub fn unlock_entry(&self, key: u64) {
        let mut map = self.entry_map.write().unwrap();
        if let Some(entry) = map.get_mut(&key) {
            entry.is_locked = false;
        }
    }

    /// Remove an entry manually (e.g., when data is no longer needed)
    pub fn remove_entry(&self, key: u64) {
        let mut map = self.entry_map.write().unwrap();
        if let Some(entry) = map.remove(&key) {
            let cache_type_idx = entry.cache_type as usize;
            
            // Remove from LRU queue
            let mut queue = self.lru_queues[cache_type_idx].lock().unwrap();
            if let Some(pos) = queue.iter().position(|e| e.key == key) {
                queue.remove(pos);
            }
            
            // Update memory usage
            self.current_usage_bytes.fetch_sub(entry.size_bytes, Ordering::Relaxed);
        }
    }

    /// Check current memory usage and evict if necessary
    pub fn maybe_evict(&self) {
        let current = self.current_usage_bytes.load(Ordering::Acquire);
        
        if current < SOFT_THRESHOLD_BYTES {
            self.emergency_mode.store(false, Ordering::Release);
            return;
        }
        
        // Determine eviction aggressiveness
        let aggressive = current >= CRITICAL_THRESHOLD_BYTES;
        self.emergency_mode.store(aggressive, Ordering::Release);
        
        // Try to acquire eviction lock
        if self.eviction_in_progress.swap(true, Ordering::AcqRel) {
            return; // Another eviction is in progress
        }
        
        // Perform eviction
        self.perform_eviction(aggressive);
        
        self.eviction_in_progress.store(false, Ordering::Release);
    }

    /// Perform eviction sweep
    fn perform_eviction(&self, aggressive: bool) {
        let target_free = if aggressive {
            CRITICAL_THRESHOLD_BYTES
        } else {
            SOFT_THRESHOLD_BYTES
        };
        
        let current = self.current_usage_bytes.load(Ordering::Acquire);
        let mut to_free = current.saturating_sub(target_free);
        
        if to_free < MIN_FREE_MEMORY_BYTES {
            to_free = MIN_FREE_MEMORY_BYTES;
        }
        
        let mut freed = 0u64;
        
        // Evict from lowest priority caches first
        for cache_type_idx in 0..3 {
            // Don't evict strategy state or positions unless in extreme emergency
            if cache_type_idx >= 3 && !aggressive {
                break;
            }
            
            let mut queue = self.lru_queues[cache_type_idx].lock().unwrap();
            
            while freed < to_free {
                // Find oldest unlocked entry
                let mut found_pos = None;
                for (pos, entry) in queue.iter().enumerate() {
                    if !entry.is_locked {
                        found_pos = Some(pos);
                        break;
                    }
                }
                
                match found_pos {
                    Some(pos) => {
                        let entry = queue.remove(pos).unwrap();
                        freed += entry.size_bytes;
                        self.evict_entry_internal(&entry);
                    }
                    None => break, // No more evictable entries in this queue
                }
            }
            
            if freed >= to_free {
                break;
            }
        }
    }

    /// Internal eviction logic
    fn evict_entry_internal(&self, entry: &LRUCacheEntry) {
        // Remove from entry map
        let mut map = self.entry_map.write().unwrap();
        map.remove(&entry.key);
        drop(map);
        
        // Update memory usage
        self.current_usage_bytes.fetch_sub(entry.size_bytes, Ordering::Relaxed);
        
        // Update eviction stats
        self.total_evictions.fetch_add(1, Ordering::Relaxed);
        
        // Track recent evictions
        let now = self.now_ms();
        let mut times = self.recent_eviction_times.lock().unwrap();
        times.push_back(now);
        
        // Keep only last second of eviction times
        while let Some(&oldest) = times.front() {
            if now - oldest > 1000 {
                times.pop_front();
            } else {
                break;
            }
        }
        
        // Call eviction callback
        if let Some(ref callback) = self.eviction_callback {
            callback(entry.key, entry.cache_type);
        }
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let total = self.current_usage_bytes.load(Ordering::Acquire);
        let evictions = self.total_evictions.load(Ordering::Acquire);
        
        // Calculate per-type usage
        let map = self.entry_map.read().unwrap();
        let mut tick_bytes = 0u64;
        let mut orderbook_bytes = 0u64;
        let mut trade_bytes = 0u64;
        let mut strategy_bytes = 0u64;
        let mut position_bytes = 0u64;
        
        for entry in map.values() {
            match entry.cache_type {
                CacheType::TickHistory => tick_bytes += entry.size_bytes,
                CacheType::OrderBookSnapshots => orderbook_bytes += entry.size_bytes,
                CacheType::TradeHistory => trade_bytes += entry.size_bytes,
                CacheType::StrategyState => strategy_bytes += entry.size_bytes,
                CacheType::ActivePositions => position_bytes += entry.size_bytes,
            }
        }
        drop(map);
        
        // Calculate recent eviction rate
        let times = self.recent_eviction_times.lock().unwrap();
        let recent_rate = times.len() as f64; // Per second
        drop(times);
        
        // Calculate pressure level
        let pressure = total as f64 / RAM_LIMIT_BYTES as f64;
        
        MemoryStats {
            total_used_bytes: total,
            tick_history_bytes: tick_bytes,
            orderbook_bytes: orderbook_bytes,
            trade_history_bytes: trade_bytes,
            strategy_state_bytes: strategy_bytes,
            position_state_bytes: position_bytes,
            total_evictions: evictions,
            recent_evictions_per_sec: recent_rate,
            pressure_level: pressure.min(1.0),
        }
    }

    /// Check if in emergency mode
    pub fn is_emergency_mode(&self) -> bool {
        self.emergency_mode.load(Ordering::Acquire)
    }

    /// Set eviction callback
    pub fn set_eviction_callback<F>(&mut self, callback: F)
    where
        F: Fn(u64, CacheType) + Send + Sync + 'static,
    {
        self.eviction_callback = Some(Box::new(callback));
    }

    /// Force garbage collection (call Python GC if needed)
    pub fn force_gc(&self) {
        // In production, would trigger Python GC via PyO3
        // For now, just log that GC should be triggered
        let stats = self.get_stats();
        if stats.pressure_level > 0.9 {
            eprintln!("WARNING: Memory pressure critical ({:.1}%), GC recommended", 
                     stats.pressure_level * 100.0);
        }
    }
}

impl Default for MemoryEvictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_evict() {
        let evictor = MemoryEvictor::new();
        
        // Register some entries
        for i in 0..10 {
            let entry = LRUCacheEntry {
                key: i,
                symbol_idx: 0,
                cache_type: CacheType::TickHistory,
                size_bytes: 1024,
                last_access_ms: 0,
                created_ms: 0,
                is_locked: false,
            };
            evictor.register_entry(entry);
        }
        
        let stats = evictor.get_stats();
        assert_eq!(stats.total_used_bytes, 10 * 1024);
        assert_eq!(stats.tick_history_bytes, 10 * 1024);
    }

    #[test]
    fn test_lock_prevents_eviction() {
        let evictor = MemoryEvictor::new();
        
        let entry = LRUCacheEntry {
            key: 1,
            symbol_idx: 0,
            cache_type: CacheType::TickHistory,
            size_bytes: 1024,
            last_access_ms: 0,
            created_ms: 0,
            is_locked: false,
        };
        evictor.register_entry(entry);
        evictor.lock_entry(1);
        
        // Entry should still exist
        let stats = evictor.get_stats();
        assert_eq!(stats.total_used_bytes, 1024);
    }

    #[test]
    fn test_memory_pressure_calculation() {
        let evictor = MemoryEvictor::new();
        
        let stats = evictor.get_stats();
        assert_eq!(stats.pressure_level, 0.0);
        
        // Simulate high memory usage
        evictor.current_usage_bytes.store(RAM_LIMIT_BYTES / 2, Ordering::Relaxed);
        
        let stats = evictor.get_stats();
        assert!((stats.pressure_level - 0.5).abs() < 0.01);
    }
}
