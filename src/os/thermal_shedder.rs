// Thermal Shedder: Sheds lowest-Sharpe asset engines and pauses non-essential Python 
// Ray workers if AMD CPU hits thermal limits. Protects microsecond latency of primary 
// BTC/ETH execution threads. Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Instant, Duration};
use std::collections::BinaryHeap;
use std::cmp::Ordering as CmpOrdering;

/// Maximum number of sheddable engines
const MAX_SHEDDABLE_ENGINES: usize = 8;

/// Temperature threshold for shedding (Celsius * 100 for fixed-point)
const TEMP_THRESHOLD_FP: u32 = 85_00; // 85°C

/// Temperature hysteresis (prevent rapid on/off cycling)
const TEMP_HYSTERESIS_FP: u32 = 5_00; // 5°C

/// Sharpe ratio fixed-point scale
const SHARPE_SCALE: i64 = 1000;

/// Engine priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnginePriority {
    Critical = 0,  // BTC, ETH - never shed
    High = 1,      // SOL, BNB - shed only at extreme temps
    Medium = 2,    // Major alts - shed at high temps  
    Low = 3,       // Minor alts - first to shed
}

/// Sheddable engine info with Sharpe ratio
#[derive(Debug, Clone)]
pub struct SheddableEngine {
    pub symbol_idx: u8,
    pub priority: EnginePriority,
    /// Rolling Sharpe ratio (fixed-point)
    pub sharpe_ratio_fp: i64,
    /// Current PnL (micro-USD)
    pub current_pnl_micro: i64,
    /// Last update timestamp
    pub last_update_ms: u64,
    /// Whether currently shed
    pub is_shed: bool,
}

impl PartialOrd for SheddableEngine {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for SheddableEngine {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Lower Sharpe = higher priority to shed
        // Use reverse ordering for min-heap behavior
        other.sharpe_ratio_fp.cmp(&self.sharpe_ratio_fp)
    }
}

/// Thermal shedding state
pub struct ThermalShedder {
    /// Current CPU temperature (fixed-point: Celsius * 100)
    current_temp_fp: AtomicU64,
    /// Peak temperature recorded
    peak_temp_fp: AtomicU64,
    /// Shedding active flag
    shedding_active: AtomicBool,
    /// Temperature at which shedding started (for hysteresis)
    shed_start_temp_fp: AtomicU64,
    
    /// Sheddable engines (ordered by Sharpe ratio)
    engines: [std::sync::RwLock<Option<SheddableEngine>>; MAX_SHEDDABLE_ENGINES],
    
    /// Number of currently shed engines
    shed_count: AtomicU8,
    
    /// Total shed events count
    total_shed_events: AtomicU64,
    
    /// Last shed timestamp
    last_shed_time_ms: AtomicU64,
    
    /// Start time for timestamps
    start_time: Instant,
    
    /// Callback for pausing Ray workers
    ray_worker_pause_callback: Option<Box<dyn Fn(u8) + Send + Sync>>,
}

impl ThermalShedder {
    /// Create a new thermal shedder
    pub fn new() -> Self {
        Self {
            current_temp_fp: AtomicU64::new(0),
            peak_temp_fp: AtomicU64::new(0),
            shedding_active: AtomicBool::new(false),
            shed_start_temp_fp: AtomicU64::new(0),
            engines: std::array::from_fn(|_| std::sync::RwLock::new(None)),
            shed_count: AtomicU8::new(0),
            total_shed_events: AtomicU64::new(0),
            last_shed_time_ms: AtomicU64::new(0),
            start_time: Instant::now(),
            ray_worker_pause_callback: None,
        }
    }

    /// Get current timestamp in milliseconds
    #[inline]
    fn now_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Update CPU temperature reading
    pub fn update_temperature(&self, temp_celsius: f32) {
        let temp_fp = (temp_celsius * 100.0) as u64;
        self.current_temp_fp.store(temp_fp, Ordering::Release);
        
        // Update peak
        let peak = self.peak_temp_fp.load(Ordering::Acquire);
        if temp_fp > peak {
            self.peak_temp_fp.store(temp_fp, Ordering::Release);
        }
        
        // Check if we need to shed or restore
        self.evaluate_shedding();
    }

    /// Register an engine for potential shedding
    pub fn register_engine(&self, engine: SheddableEngine) {
        let idx = engine.symbol_idx as usize;
        if idx >= MAX_SHEDDABLE_ENGINES {
            return;
        }
        
        let mut lock = self.engines[idx].write().unwrap();
        *lock = Some(engine);
    }

    /// Update engine Sharpe ratio
    pub fn update_engine_sharpe(&self, symbol_idx: u8, sharpe_fp: i64, pnl_micro: i64) {
        let idx = symbol_idx as usize;
        if idx >= MAX_SHEDDABLE_ENGINES {
            return;
        }
        
        let mut lock = self.engines[idx].write().unwrap();
        if let Some(ref mut engine) = *lock {
            engine.sharpe_ratio_fp = sharpe_fp;
            engine.current_pnl_micro = pnl_micro;
            engine.last_update_ms = self.now_ms();
        }
    }

    /// Evaluate whether to shed or restore engines
    fn evaluate_shedding(&self) {
        let current_temp = self.current_temp_fp.load(Ordering::Acquire);
        let was_shedding = self.shedding_active.load(Ordering::Acquire);
        
        // Check if we should start shedding
        if !was_shedding && current_temp >= TEMP_THRESHOLD_FP {
            self.start_shedding(current_temp);
        }
        // Check if we can restore (with hysteresis)
        else if was_shedding {
            let restore_threshold = self.shed_start_temp_fp.load(Ordering::Acquire)
                .saturating_sub(TEMP_HYSTERESIS_FP);
            
            if current_temp <= restore_threshold {
                self.restore_all();
            } else {
                // Continue evaluating which engines to shed
                self.maybe_shed_more(current_temp);
            }
        }
    }

    /// Start shedding process
    fn start_shedding(&self, temp_fp: u64) {
        self.shedding_active.store(true, Ordering::Release);
        self.shed_start_temp_fp.store(temp_fp, Ordering::Release);
        
        // Shed lowest priority engines first
        self.shed_lowest_priority();
    }

    /// Shed the lowest priority engine
    fn shed_lowest_priority(&self) {
        // Find lowest priority, lowest Sharpe engine that isn't already shed
        let mut candidates: Vec<(usize, i64, EnginePriority)> = Vec::new();
        
        for i in 0..MAX_SHEDDABLE_ENGINES {
            let lock = self.engines[i].read().unwrap();
            if let Some(ref engine) = *lock {
                if !engine.is_shed && engine.priority != EnginePriority::Critical {
                    candidates.push((i, engine.sharpe_ratio_fp, engine.priority));
                }
            }
        }
        
        if candidates.is_empty() {
            return;
        }
        
        // Sort by priority (low first), then by Sharpe (low first)
        candidates.sort_by(|a, b| {
            a.2.cmp(&b.2).then(a.1.cmp(&b.1))
        });
        
        // Shed the first candidate
        let (idx, _, _) = candidates[0];
        self.shed_engine(idx as u8);
    }

    /// Shed a specific engine
    fn shed_engine(&self, symbol_idx: u8) {
        let idx = symbol_idx as usize;
        if idx >= MAX_SHEDDABLE_ENGINES {
            return;
        }
        
        let mut lock = self.engines[idx].write().unwrap();
        if let Some(ref mut engine) = *lock {
            if engine.is_shed {
                return; // Already shed
            }
            
            engine.is_shed = true;
            
            // Call pause callback if set
            if let Some(ref callback) = self.ray_worker_pause_callback {
                callback(symbol_idx);
            }
        }
        
        drop(lock);
        
        self.shed_count.fetch_add(1, Ordering::Release);
        self.total_shed_events.fetch_add(1, Ordering::Release);
        self.last_shed_time_ms.store(self.now_ms(), Ordering::Release);
    }

    /// Maybe shed more engines based on temperature severity
    fn maybe_shed_more(&self, current_temp: u64) {
        // More aggressive shedding at higher temperatures
        let severity = if current_temp >= TEMP_THRESHOLD_FP + 10_00 {
            3 // Extreme - shed all non-critical
        } else if current_temp >= TEMP_THRESHOLD_FP + 5_00 {
            2 // High - shed low and medium priority
        } else {
            1 // Moderate - shed low priority only
        };
        
        for _ in 0..severity {
            self.shed_lowest_priority();
        }
    }

    /// Restore all shed engines
    fn restore_all(&self) {
        self.shedding_active.store(false, Ordering::Release);
        
        for i in 0..MAX_SHEDDABLE_ENGINES {
            let mut lock = self.engines[i].write().unwrap();
            if let Some(ref mut engine) = *lock {
                engine.is_shed = false;
            }
        }
        
        self.shed_count.store(0, Ordering::Release);
    }

    /// Check if a specific engine is shed
    pub fn is_engine_shed(&self, symbol_idx: u8) -> bool {
        let idx = symbol_idx as usize;
        if idx >= MAX_SHEDDABLE_ENGINES {
            return false;
        }
        
        let lock = self.engines[idx].read().unwrap();
        lock.as_ref().map(|e| e.is_shed).unwrap_or(false)
    }

    /// Get current temperature
    pub fn get_temperature(&self) -> f32 {
        let temp_fp = self.current_temp_fp.load(Ordering::Acquire) as f32;
        temp_fp / 100.0
    }

    /// Get peak temperature
    pub fn get_peak_temperature(&self) -> f32 {
        let temp_fp = self.peak_temp_fp.load(Ordering::Acquire) as f32;
        temp_fp / 100.0
    }

    /// Check if shedding is active
    pub fn is_shedding_active(&self) -> bool {
        self.shedding_active.load(Ordering::Acquire)
    }

    /// Get number of shed engines
    pub fn get_shed_count(&self) -> u8 {
        self.shed_count.load(Ordering::Acquire)
    }

    /// Get total shed events
    pub fn get_total_shed_events(&self) -> u64 {
        self.total_shed_events.load(Ordering::Acquire)
    }

    /// Set callback for pausing Ray workers
    pub fn set_ray_pause_callback<F>(&mut self, callback: F)
    where
        F: Fn(u8) + Send + Sync + 'static,
    {
        self.ray_worker_pause_callback = Some(Box::new(callback));
    }

    /// Emergency shed all non-critical engines immediately
    pub fn emergency_shed_all(&self) {
        for i in 0..MAX_SHEDDABLE_ENGINES {
            let mut lock = self.engines[i].write().unwrap();
            if let Some(ref mut engine) = *lock {
                if engine.priority != EnginePriority::Critical && !engine.is_shed {
                    engine.is_shed = true;
                    self.shed_count.fetch_add(1, Ordering::Release);
                    
                    if let Some(ref callback) = self.ray_worker_pause_callback {
                        callback(i as u8);
                    }
                }
            }
        }
        
        self.shedding_active.store(true, Ordering::Release);
        self.total_shed_events.fetch_add(1, Ordering::Release);
        self.last_shed_time_ms.store(self.now_ms(), Ordering::Release);
    }
}

impl Default for ThermalShedder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_tracking() {
        let shedder = ThermalShedder::new();
        
        shedder.update_temperature(75.0);
        assert!((shedder.get_temperature() - 75.0).abs() < 0.01);
        
        shedder.update_temperature(80.0);
        assert!((shedder.get_peak_temperature() - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_engine_shedding_at_threshold() {
        let shedder = ThermalShedder::new();
        
        // Register a low priority engine
        let engine = SheddableEngine {
            symbol_idx: 5,
            priority: EnginePriority::Low,
            sharpe_ratio_fp: 500, // Low Sharpe
            current_pnl_micro: -1000,
            last_update_ms: 0,
            is_shed: false,
        };
        shedder.register_engine(engine);
        
        // Raise temperature above threshold
        shedder.update_temperature(90.0);
        
        // Engine should be shed
        assert!(shedder.is_engine_shed(5));
        assert!(shedder.is_shedding_active());
    }

    #[test]
    fn test_critical_engine_never_shed() {
        let shedder = ThermalShedder::new();
        
        // Register a critical engine with bad Sharpe
        let engine = SheddableEngine {
            symbol_idx: 0, // BTC
            priority: EnginePriority::Critical,
            sharpe_ratio_fp: -500, // Negative Sharpe
            current_pnl_micro: -10000,
            last_update_ms: 0,
            is_shed: false,
        };
        shedder.register_engine(engine);
        
        // Raise temperature well above threshold
        shedder.update_temperature(95.0);
        
        // Critical engine should NOT be shed
        assert!(!shedder.is_engine_shed(0));
    }

    #[test]
    fn test_emergency_shed() {
        let shedder = ThermalShedder::new();
        
        // Register multiple engines
        for i in 1u8..=5 {
            let engine = SheddableEngine {
                symbol_idx: i,
                priority: EnginePriority::Low,
                sharpe_ratio_fp: 1000,
                current_pnl_micro: 0,
                last_update_ms: 0,
                is_shed: false,
            };
            shedder.register_engine(engine);
        }
        
        shedder.emergency_shed_all();
        
        assert!(shedder.is_shedding_active());
        assert_eq!(shedder.get_shed_count(), 5);
    }
}
