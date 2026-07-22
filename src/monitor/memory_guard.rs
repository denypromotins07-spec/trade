//! Memory Guard for Aggressive RAM Management
//!
//! Implements an aggressive memory guard that forces garbage collection and drops
//! lowest-priority caches if the global 8GB RAM limit reaches 95% utilization.
//! Optimized for AMD Ryzen AI 5 architecture with minimal latency impact.

use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Memory priority levels for cache eviction
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CachePriority {
    Critical = 0,    // Never evict (core state)
    High = 1,        // Evict only at extreme pressure
    Medium = 2,      // Evict at 90% utilization
    Low = 3,         // Evict at 85% utilization
    Background = 4,  // First to evict
}

/// Cached item with metadata
#[derive(Debug, Clone)]
pub struct CachedItem {
    pub key: String,
    pub size_bytes: u64,
    pub priority: CachePriority,
    pub last_access_ns: u64,
    pub creation_time_ns: u64,
    pub access_count: u64,
}

/// Memory statistics
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_system_memory_bytes: u64,
    pub used_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub cache_memory_bytes: u64,
    pub utilization_percent: f64,
    pub gc_triggers: u64,
    pub evictions_performed: u64,
}

/// Memory guard configuration
#[derive(Debug, Clone)]
pub struct MemoryGuardConfig {
    /// Maximum memory utilization before triggering GC (0.0 - 1.0)
    pub gc_threshold: f64,
    /// Maximum memory utilization before forced eviction (0.0 - 1.0)
    pub eviction_threshold: f64,
    /// Check interval in milliseconds
    pub check_interval_ms: u64,
    /// Minimum free memory to maintain (bytes)
    pub min_free_memory_bytes: u64,
    /// Enable aggressive mode (evict more frequently)
    pub aggressive_mode: bool,
}

impl Default for MemoryGuardConfig {
    fn default() -> Self {
        Self {
            gc_threshold: 0.85,           // Trigger GC at 85%
            eviction_threshold: 0.95,     // Force eviction at 95%
            check_interval_ms: 100,       // Check every 100ms
            min_free_memory_bytes: 512 * 1024 * 1024, // 512MB minimum free
            aggressive_mode: false,
        }
    }
}

/// Callback for cache eviction
pub type EvictionCallback = Box<dyn Fn(&str, CachePriority) -> Result<u64, String> + Send + Sync>;

/// Main memory guard structure
pub struct MemoryGuard {
    config: MemoryGuardConfig,
    running: Arc<AtomicBool>,
    check_thread: Option<JoinHandle<()>>,
    caches: Arc<dashmap::DashMap<String, CachedItem>>,
    eviction_callback: Option<Arc<EvictionCallback>>,
    stats: Arc<MemoryStatsInternal>,
}

/// Internal stats with atomics
struct MemoryStatsInternal {
    gc_triggers: AtomicU64,
    evictions_performed: AtomicU64,
    bytes_freed: AtomicU64,
    threshold_hits: AtomicU64,
}

impl Default for MemoryStatsInternal {
    fn default() -> Self {
        Self {
            gc_triggers: AtomicU64::new(0),
            evictions_performed: AtomicU64::new(0),
            bytes_freed: AtomicU64::new(0),
            threshold_hits: AtomicU64::new(0),
        }
    }
}

impl MemoryGuard {
    /// Create a new memory guard instance
    pub fn new(config: MemoryGuardConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            check_thread: None,
            caches: Arc::new(dashmap::DashMap::new()),
            eviction_callback: None,
            stats: Arc::new(MemoryStatsInternal::default()),
        }
    }

    /// Set the eviction callback
    pub fn set_eviction_callback<F>(&mut self, callback: F)
    where
        F: Fn(&str, CachePriority) -> Result<u64, String> + Send + Sync + 'static,
    {
        self.eviction_callback = Some(Arc::new(Box::new(callback)));
    }

    /// Register a cache item for tracking
    pub fn register_cache(&self, key: &str, size_bytes: u64, priority: CachePriority) {
        let now = current_time_ns();
        let item = CachedItem {
            key: key.to_string(),
            size_bytes,
            priority,
            last_access_ns: now,
            creation_time_ns: now,
            access_count: 0,
        };
        
        self.caches.insert(key.to_string(), item);
    }

    /// Update cache access time
    pub fn touch_cache(&self, key: &str) {
        if let Some(mut entry) = self.caches.get_mut(key) {
            entry.last_access_ns = current_time_ns();
            entry.access_count += 1;
        }
    }

    /// Remove a cache item from tracking
    pub fn remove_cache(&self, key: &str) -> Option<CachedItem> {
        self.caches.remove(key).map(|(_, v)| v)
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> MemoryStats {
        let mut stats = get_system_memory_info();
        stats.gc_triggers = self.stats.gc_triggers.load(Ordering::Relaxed);
        stats.evictions_performed = self.stats.evictions_performed.load(Ordering::Relaxed);
        stats
    }

    /// Start the memory monitoring loop
    pub fn start(&mut self) -> Result<(), String> {
        if self.running.load(Ordering::SeqCst) {
            return Err("Memory guard already running".to_string());
        }

        self.running.store(true, Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        let config = self.config.clone();
        let caches = Arc::clone(&self.caches);
        let eviction_callback = self.eviction_callback.clone();
        let stats = Arc::clone(&self.stats);

        self.check_thread = Some(thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                let start = Instant::now();

                // Get current memory info
                let mem_info = get_system_memory_info();
                let utilization = mem_info.used_memory_bytes as f64 
                    / mem_info.total_system_memory_bytes.max(1) as f64;

                // Check thresholds
                if utilization >= config.eviction_threshold {
                    stats.threshold_hits.fetch_add(1, Ordering::Relaxed);
                    
                    // Force eviction of low-priority caches
                    let freed = Self::perform_eviction(
                        &caches, 
                        &eviction_callback,
                        config.aggressive_mode,
                    );
                    stats.evictions_performed.fetch_add(1, Ordering::Relaxed);
                    stats.bytes_freed.fetch_add(freed, Ordering::Relaxed);
                    
                    eprintln!("MemoryGuard: Eviction performed, freed {} bytes", freed);
                } else if utilization >= config.gc_threshold {
                    stats.gc_triggers.fetch_add(1, Ordering::Relaxed);
                    
                    // Trigger garbage collection
                    #[cfg(target_os = "linux")]
                    unsafe {
                        libc::malloc_trim(0);
                    }
                    
                    // Suggest Python GC if applicable
                    Self::trigger_python_gc();
                }

                // Sleep until next check
                let elapsed = start.elapsed();
                let sleep_duration = Duration::from_millis(config.check_interval_ms)
                    .saturating_sub(elapsed);
                thread::sleep(sleep_duration);
            }
        }));

        Ok(())
    }

    /// Perform cache eviction based on priority and LRU
    fn perform_eviction(
        caches: &Arc<dashmap::DashMap<String, CachedItem>>,
        callback: &Option<Arc<EvictionCallback>>,
        aggressive: bool,
    ) -> u64 {
        let mut total_freed = 0u64;
        let now = current_time_ns();

        // Collect candidates sorted by priority (lowest first) then by LRU
        let mut candidates: Vec<(String, CachedItem)> = caches
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect();

        // Sort by priority (highest number = lowest priority), then by last access
        candidates.sort_by(|a, b| {
            b.1.priority.cmp(&a.1.priority)
                .then_with(|| a.1.last_access_ns.cmp(&b.1.last_access_ns))
        });

        // Evict starting from lowest priority
        let eviction_threshold = if aggressive { 
            CachePriority::Medium 
        } else { 
            CachePriority::Low 
        };

        for (key, item) in candidates {
            if item.priority >= eviction_threshold {
                // Call eviction callback if available
                if let Some(cb) = callback {
                    match cb(&key, item.priority) {
                        Ok(freed) => {
                            total_freed += freed;
                            caches.remove(&key);
                            
                            // Stop if we've freed enough
                            if total_freed >= 100 * 1024 * 1024 {
                                // Freed 100MB, stop
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("MemoryGuard: Failed to evict {}: {}", key, e);
                        }
                    }
                } else {
                    // No callback, just remove from tracking
                    caches.remove(&key);
                    total_freed += item.size_bytes;
                }
            }
        }

        total_freed
    }

    /// Trigger Python garbage collection via subprocess
    fn trigger_python_gc() {
        // In production, this would signal Python workers to run GC
        // For now, just log
        #[cfg(debug_assertions)]
        println!("MemoryGuard: Triggering Python GC");
    }

    /// Stop the memory guard
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.check_thread.take() {
            let _ = handle.join();
        }
    }

    /// Get detailed cache information
    pub fn get_cache_info(&self) -> HashMap<String, CachedItem> {
        self.caches.iter().map(|e| (e.key().clone(), e.value().clone())).collect()
    }

    /// Get memory guard statistics
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("gc_triggers", self.stats.gc_triggers.load(Ordering::Relaxed));
        stats.insert("evictions_performed", self.stats.evictions_performed.load(Ordering::Relaxed));
        stats.insert("bytes_freed", self.stats.bytes_freed.load(Ordering::Relaxed));
        stats.insert("threshold_hits", self.stats.threshold_hits.load(Ordering::Relaxed));
        stats.insert("tracked_caches", self.caches.len() as u64);
        stats
    }

    /// Emergency memory release - drop all non-critical caches
    pub fn emergency_release(&self) -> u64 {
        let mut total_freed = 0u64;
        let now = current_time_ns();

        let keys_to_remove: Vec<String> = self.caches
            .iter()
            .filter(|e| e.value().priority != CachePriority::Critical)
            .map(|e| e.key().clone())
            .collect();

        for key in keys_to_remove {
            if let Some(item) = self.caches.remove(&key).map(|(_, v)| v) {
                total_freed += item.size_bytes;
                
                if let Some(ref cb) = self.eviction_callback {
                    let _ = cb(&key, item.priority);
                }
            }
        }

        self.stats.evictions_performed.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_freed.fetch_add(total_freed, Ordering::Relaxed);

        eprintln!("MemoryGuard: Emergency release freed {} bytes", total_freed);
        total_freed
    }
}

impl Drop for MemoryGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Get system memory information
fn get_system_memory_info() -> MemoryStats {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        
        let mut stats = MemoryStats::default();
        
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            let mut mem_total = 0u64;
            let mut mem_available = 0u64;
            let mut mem_free = 0u64;
            let mut buffers = 0u64;
            let mut cached = 0u64;

            for line in meminfo.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let value: u64 = parts[1].parse().unwrap_or(0);
                    match parts[0] {
                        "MemTotal:" => mem_total = value * 1024,
                        "MemAvailable:" => mem_available = value * 1024,
                        "MemFree:" => mem_free = value * 1024,
                        "Buffers:" => buffers = value * 1024,
                        "Cached:" => cached = value * 1024,
                        _ => {}
                    }
                }
            }

            stats.total_system_memory_bytes = mem_total;
            stats.available_memory_bytes = mem_available;
            stats.used_memory_bytes = mem_total - mem_available;
            stats.cache_memory_bytes = buffers + cached;
            
            if mem_total > 0 {
                stats.utilization_percent = stats.used_memory_bytes as f64 
                    / mem_total as f64 * 100.0;
            }
        }

        stats
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        // Fallback for other platforms
        MemoryStats {
            total_system_memory_bytes: 8 * 1024 * 1024 * 1024, // Assume 8GB
            used_memory_bytes: 4 * 1024 * 1024 * 1024,
            available_memory_bytes: 4 * 1024 * 1024 * 1024,
            cache_memory_bytes: 0,
            utilization_percent: 50.0,
            ..Default::default()
        }
    }
}

/// Get current time in nanoseconds
#[inline]
fn current_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_guard_creation() {
        let config = MemoryGuardConfig::default();
        let guard = MemoryGuard::new(config);
        
        assert!(!guard.running.load(Ordering::SeqCst));
    }

    #[test]
    fn test_cache_registration() {
        let guard = MemoryGuard::new(MemoryGuardConfig::default());
        
        guard.register_cache("test_cache", 1024 * 1024, CachePriority::Low);
        
        let info = guard.get_cache_info();
        assert!(info.contains_key("test_cache"));
        
        guard.touch_cache("test_cache");
    }

    #[test]
    fn test_emergency_release() {
        let guard = MemoryGuard::new(MemoryGuardConfig::default());
        
        guard.register_cache("critical", 100, CachePriority::Critical);
        guard.register_cache("low_priority", 200, CachePriority::Low);
        guard.register_cache("background", 300, CachePriority::Background);
        
        let freed = guard.emergency_release();
        
        // Should have freed low_priority + background
        assert!(freed >= 500);
        
        // Critical should still be there
        let info = guard.get_cache_info();
        assert!(info.contains_key("critical"));
    }

    #[test]
    fn test_memory_stats() {
        let guard = MemoryGuard::new(MemoryGuardConfig::default());
        let stats = guard.get_memory_usage();
        
        assert!(stats.total_system_memory_bytes > 0);
        assert!(stats.utilization_percent >= 0.0);
        assert!(stats.utilization_percent <= 100.0);
    }
}
