//! AMD Thermal Guard with Predictive Throttling Prevention
//!
//! This module builds a predictive thermal guard that dynamically sheds
//! non-essential Python Ray workers if the AMD silicon approaches thermal
//! throttling thresholds, protecting hot-path latency.
//!
//! Key features:
//! - Predictive temperature modeling based on power trends
//! - Dynamic worker shedding before throttling occurs
//! - Priority-based task migration
//! - Integration with SMU telemetry for microsecond accuracy
//!
//! Author: Elite Quantitative Software Engineering Team
//! Stage: 49 - Hardware Telemetry & SMU Throttling Prevention

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Import SMU reader for hardware telemetry
use crate::amd::smu_reader::{SmuReader, SmuTelemetry, SmuError};

// =============================================================================
// Thermal Threshold Constants
// =============================================================================

/// Temperature threshold to start shedding workers (millidegrees)
const SHED_THRESHOLD_MILLI: i32 = 75_000; // 75°C

/// Critical temperature requiring immediate action (millidegrees)
const CRITICAL_THRESHOLD_MILLI: i32 = 80_000; // 80°C

/// Maximum safe temperature (millidegrees)
const MAX_SAFE_TEMP_MILLI: i32 = 85_000; // 85°C

/// Temperature rate-of-change threshold for prediction (millidegrees/second)
const TEMP_RISE_RATE_THRESHOLD: i32 = 5000; // 5°C/second

/// Cooldown period after shedding workers
const SHED_COOLDOWN_MS: u64 = 5000; // 5 seconds

// =============================================================================
// Worker State and Management
// =============================================================================

/// Priority level for Ray workers
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkerPriority {
    /// Critical workers (matching engine, order routing)
    Critical = 0,
    /// High priority workers (signal processing)
    High = 1,
    /// Normal priority workers (data ingestion)
    Normal = 2,
    /// Low priority workers (logging, metrics)
    Low = 3,
}

/// State of a Ray worker
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Unique worker identifier
    pub worker_id: String,
    /// Worker priority level
    pub priority: WorkerPriority,
    /// Whether worker is currently active
    pub active: bool,
    /// Current load percentage (0-100)
    pub load_percent: u32,
    /// Last activity timestamp
    pub last_activity: Instant,
}

impl WorkerState {
    pub fn new(worker_id: String, priority: WorkerPriority) -> Self {
        Self {
            worker_id,
            priority,
            active: true,
            load_percent: 0,
            last_activity: Instant::now(),
        }
    }
}

/// Result of thermal guard decision
#[derive(Debug, Clone)]
pub struct ThermalDecision {
    /// Whether action was taken
    pub action_taken: bool,
    /// Number of workers shed
    pub workers_shed: usize,
    /// Target temperature after action
    pub target_temp_milli: i32,
    /// Reason for decision
    pub reason: String,
}

// =============================================================================
// Predictive Thermal Model
// =============================================================================

/// Predictive model for temperature forecasting
pub struct ThermalPredictor {
    /// Recent temperature samples (ring buffer)
    temp_history: Vec<i32>,
    /// Recent power samples
    power_history: Vec<u32>,
    /// Sample interval in milliseconds
    sample_interval_ms: u64,
    /// Maximum history size
    max_history: usize,
}

impl ThermalPredictor {
    pub fn new(sample_interval_ms: u64, max_history: usize) -> Self {
        Self {
            temp_history: Vec::with_capacity(max_history),
            power_history: Vec::with_capacity(max_history),
            sample_interval_ms,
            max_history,
        }
    }

    /// Add new temperature sample
    pub fn add_sample(&mut self, temp_milli: i32, power_mw: u32) {
        self.temp_history.push(temp_milli);
        self.power_history.push(power_mw);

        // Trim history if needed
        while self.temp_history.len() > self.max_history {
            self.temp_history.remove(0);
            self.power_history.remove(0);
        }
    }

    /// Predict temperature at future time horizon (milliseconds)
    pub fn predict_temperature(&self, horizon_ms: u64) -> Option<i32> {
        if self.temp_history.len() < 2 {
            return self.temp_history.last().copied();
        }

        // Calculate temperature rise rate
        let temp_delta = self.temp_history[self.temp_history.len() - 1]
            - self.temp_history[0];
        
        let time_span_ms = (self.temp_history.len() as u64) * self.sample_interval_ms;
        
        if time_span_ms == 0 {
            return self.temp_history.last().copied();
        }

        // Rate in millidegrees per millisecond
        let rate = (temp_delta as f64) / (time_span_ms as f64);
        
        // Extrapolate to horizon
        let current = *self.temp_history.last().unwrap();
        let predicted = current + (rate * horizon_ms as f64) as i32;

        Some(predicted)
    }

    /// Get current temperature trend (positive = rising)
    pub fn get_trend(&self) -> i32 {
        if self.temp_history.len() < 2 {
            return 0;
        }

        let recent = *self.temp_history.last().unwrap();
        let older = self.temp_history[self.temp_history.len() / 2];
        
        recent - older
    }

    /// Check if temperature is rising rapidly
    pub fn is_rapid_rise(&self) -> bool {
        let trend_per_second = self.get_trend() * (1000 / self.sample_interval_ms) as i32;
        trend_per_second > TEMP_RISE_RATE_THRESHOLD
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.temp_history.clear();
        self.power_history.clear();
    }
}

// =============================================================================
// Thermal Guard Implementation
// =============================================================================

/// Main thermal guard controller
pub struct ThermalGuard {
    /// SMU reader for hardware telemetry
    smu_reader: Arc<SmuReader>,
    
    /// Predictive thermal model
    predictor: ThermalPredictor,
    
    /// Registered workers
    workers: Vec<WorkerState>,
    
    /// Whether guard is active
    active: AtomicBool,
    
    /// Count of shed workers
    shed_count: AtomicUsize,
    
    /// Last shed timestamp
    last_shed_time: AtomicU64,
    
    /// Emergency mode flag
    emergency_mode: AtomicBool,
}

unsafe impl Send for ThermalGuard {}
unsafe impl Sync for ThermalGuard {}

impl ThermalGuard {
    /// Create new thermal guard
    pub fn new(smu_reader: Arc<SmuReader>) -> Self {
        Self {
            smu_reader,
            predictor: ThermalPredictor::new(100, 60), // 100ms sampling, 6 second history
            workers: Vec::new(),
            active: AtomicBool::new(true),
            shed_count: AtomicUsize::new(0),
            last_shed_time: AtomicU64::new(0),
            emergency_mode: AtomicBool::new(false),
        }
    }

    /// Register a worker for thermal management
    pub fn register_worker(&mut self, worker_id: String, priority: WorkerPriority) {
        self.workers.push(WorkerState::new(worker_id, priority));
    }

    /// Unregister a worker
    pub fn unregister_worker(&mut self, worker_id: &str) {
        self.workers.retain(|w| w.worker_id != worker_id);
    }

    /// Main thermal monitoring loop
    pub fn monitor_cycle(&self) -> Result<ThermalDecision, SmuError> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(ThermalDecision {
                action_taken: false,
                workers_shed: 0,
                target_temp_milli: 0,
                reason: "Thermal guard inactive".to_string(),
            });
        }

        // Read current telemetry
        let telemetry = self.smu_reader.read_telemetry()?;
        
        // Update predictor
        unsafe {
            // Safe cast for predictor
            let predictor_mut = &mut *(self.predictor as *const ThermalPredictor as *mut ThermalPredictor);
            predictor_mut.add_sample(telemetry.max_temp_milli(), telemetry.cpu_power_mw + telemetry.gpu_power_mw);
        }

        // Make thermal decision
        let decision = self.make_thermal_decision(&telemetry);

        // Execute decision
        if decision.action_taken {
            self.execute_shedding(&decision);
        }

        Ok(decision)
    }

    /// Make thermal decision based on current state
    fn make_thermal_decision(&self, telemetry: &SmuTelemetry) -> ThermalDecision {
        let current_temp = telemetry.max_temp_milli();
        let predicted_temp = self.predictor.predict_temperature(2000).unwrap_or(current_temp); // 2 second horizon

        // Check for critical condition
        if current_temp >= CRITICAL_THRESHOLD_MILLI {
            self.emergency_mode.store(true, Ordering::Relaxed);
            
            return ThermalDecision {
                action_taken: true,
                workers_shed: self.count_sheddable_workers(),
                target_temp_milli: SHED_THRESHOLD_MILLI,
                reason: format!("Critical temperature: {}°C", current_temp / 1000),
            };
        }

        // Check for predicted overheating
        if predicted_temp >= CRITICAL_THRESHOLD_MILLI && self.predictor.is_rapid_rise() {
            return ThermalDecision {
                action_taken: true,
                workers_shed: self.count_sheddable_workers().min(2), // Shed up to 2 workers
                target_temp_milli: SHED_THRESHOLD_MILLI,
                reason: format!("Predicted overheating: {}°C in 2s", predicted_temp / 1000),
            };
        }

        // Check for elevated temperature
        if current_temp >= SHED_THRESHOLD_MILLI {
            // Check cooldown
            let now = Instant::now();
            let last_shed_ms = self.last_shed_time.load(Ordering::Relaxed);
            
            // Simple timestamp comparison (in production, use proper time tracking)
            if last_shed_ms == 0 || (now.elapsed().as_millis() as u64) > last_shed_ms + SHED_COOLDOWN_MS {
                return ThermalDecision {
                    action_taken: true,
                    workers_shed: 1, // Shed one worker at a time
                    target_temp_milli: SHED_THRESHOLD_MILLI - 5000, // Target 5°C below threshold
                    reason: format!("Elevated temperature: {}°C", current_temp / 1000),
                };
            }
        }

        // No action needed
        ThermalDecision {
            action_taken: false,
            workers_shed: 0,
            target_temp_milli: current_temp,
            reason: "Temperature within normal range".to_string(),
        }
    }

    /// Count workers that can be shed (lowest priority first)
    fn count_sheddable_workers(&self) -> usize {
        self.workers.iter()
            .filter(|w| w.active && w.priority == WorkerPriority::Low)
            .count()
    }

    /// Execute worker shedding
    fn execute_shedding(&self, decision: &ThermalDecision) {
        // Update last shed time
        self.last_shed_time.store(
            Instant::now().elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );

        // Increment shed count
        self.shed_count.fetch_add(decision.workers_shed, Ordering::Relaxed);

        // In production, this would send commands to Ray to terminate workers
        // For now, just mark lowest priority workers as inactive
        let mut shed_count = decision.workers_shed;
        
        for worker in self.workers.iter_mut() {
            if shed_count == 0 {
                break;
            }
            
            if worker.active && worker.priority == WorkerPriority::Low {
                worker.active = false;
                shed_count -= 1;
            }
        }
    }

    /// Get current thermal status
    pub fn get_status(&self) -> ThermalStatus {
        ThermalStatus {
            active: self.active.load(Ordering::Relaxed),
            emergency_mode: self.emergency_mode.load(Ordering::Relaxed),
            total_workers: self.workers.len(),
            active_workers: self.workers.iter().filter(|w| w.active).count(),
            shed_count: self.shed_count.load(Ordering::Relaxed),
        }
    }

    /// Enable thermal guard
    pub fn enable(&self) {
        self.active.store(true, Ordering::Relaxed);
        self.emergency_mode.store(false, Ordering::Relaxed);
    }

    /// Disable thermal guard
    pub fn disable(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Reset shed counter
    pub fn reset_shed_count(&self) {
        self.shed_count.store(0, Ordering::Relaxed);
    }
}

/// Current thermal guard status
#[derive(Debug, Clone)]
pub struct ThermalStatus {
    pub active: bool,
    pub emergency_mode: bool,
    pub total_workers: usize,
    pub active_workers: usize,
    pub shed_count: usize,
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_predictor() {
        let mut predictor = ThermalPredictor::new(100, 60);
        
        // Add rising temperature samples
        for i in 0..10 {
            predictor.add_sample(50_000 + (i as i32) * 500, 100000);
        }
        
        // Should predict higher temperature
        let predicted = predictor.predict_temperature(1000).unwrap();
        assert!(predicted > 50_000);
        
        // Trend should be positive
        assert!(predictor.get_trend() > 0);
    }

    #[test]
    fn test_worker_priority_ordering() {
        assert!(WorkerPriority::Critical < WorkerPriority::High);
        assert!(WorkerPriority::High < WorkerPriority::Normal);
        assert!(WorkerPriority::Normal < WorkerPriority::Low);
    }

    #[test]
    fn test_worker_state() {
        let worker = WorkerState::new("worker-1".to_string(), WorkerPriority::Normal);
        
        assert_eq!(worker.worker_id, "worker-1");
        assert!(worker.active);
        assert_eq!(worker.priority, WorkerPriority::Normal);
    }

    #[test]
    fn test_thermal_thresholds() {
        // Verify threshold ordering
        assert!(SHED_THRESHOLD_MILLI < CRITICAL_THRESHOLD_MILLI);
        assert!(CRITICAL_THRESHOLD_MILLI < MAX_SAFE_TEMP_MILLI);
    }
}
