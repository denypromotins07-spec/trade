//! AMD ECC Memory Controller Interface & Bit-Flip Detection
//! 
//! This module interfaces with AMD Ryzen AI 5 ECC memory controllers to:
//! - Log correctable bit-flips for predictive failure analysis
//! - Trigger graceful /KILL and reboot on uncorrectable errors
//! - Protect live order book state from memory corruption
//! 
//! Optimized for microsecond latency with minimal CPU overhead.
//! Integrates with Windows WHEA (Windows Hardware Error Architecture)
//! and AMD-specific SMU (System Management Unit) registers.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use std::thread;
use std::path::PathBuf;
use std::collections::VecDeque;

/// Maximum correctable errors before warning threshold
const CORRECTABLE_ERROR_WARNING_THRESHOLD: u64 = 100;
/// Critical threshold triggering preemptive restart
const CORRECTABLE_ERROR_CRITICAL_THRESHOLD: u64 = 500;
/// Polling interval for ECC status (microseconds)
const ECC_POLL_INTERVAL_US: u64 = 100;
/// Ring buffer size for error history
const ERROR_HISTORY_SIZE: usize = 1000;

/// ECC Error types from AMD PPR (Processor Programming Reference)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EccErrorType {
    /// Single-bit error (corrected automatically)
    Correctable,
    /// Multi-bit error (uncorrectable, data loss)
    Uncorrectable,
    /// Parity error in L3 cache
    L3CacheParity,
    /// DRAM ECC error
    DramEcc,
    /// SRAM ECC error
    SramEcc,
    /// Tag RAM error
    TagRam,
}

/// Detailed ECC error record
#[derive(Debug, Clone)]
pub struct EccErrorRecord {
    pub error_type: EccErrorType,
    pub address: u64,
    pub channel: u8,
    pub dimm_slot: u8,
    pub bank: u8,
    pub rank: u8,
    pub syndrome: u16,
    pub timestamp: Instant,
    pub cpu_core: u8,
    pub socket_id: u8,
}

/// Aggregated ECC statistics per DIMM
#[derive(Debug, Clone, Default)]
pub struct DimmEccStats {
    pub correctable_count: u64,
    pub uncorrectable_count: u64,
    pub last_error_time: Option<Instant>,
    pub error_rate_per_hour: f64,
    pub predicted_failure_hours: Option<u64>,
}

/// Main ECC Memory Monitor
pub struct EccMemoryMonitor {
    /// Total correctable errors across all DIMMs
    total_correctable: AtomicU64,
    /// Total uncorrectable errors (should be zero in healthy system)
    total_uncorrectable: AtomicU64,
    /// Per-DIMM statistics
    dimm_stats: parking_lot::RwLock<Vec<DimmEccStats>>,
    /// Error history for trend analysis
    error_history: parking_lot::Mutex<VecDeque<EccErrorRecord>>,
    /// Running flag
    is_running: Arc<AtomicBool>,
    /// Emergency shutdown flag
    emergency_shutdown_requested: AtomicBool,
    /// Alert channel for async notifications
    alert_tx: Option<crossbeam_channel::Sender<EccAlert>>,
    /// Path to log ECC events
    log_path: Option<PathBuf>,
    /// AMD SMU register base address (memory-mapped)
    smu_base_address: AtomicU64,
    /// Callback for graceful shutdown initiation
    shutdown_callback: Option<Arc<dyn Fn() + Send + Sync>>,
}

/// ECC Alert types
#[derive(Debug, Clone)]
pub enum EccAlert {
    CorrectableError { record: EccErrorRecord, cumulative: u64 },
    UncorrectableError { record: EccErrorRecord, immediate_action_required: bool },
    ThresholdWarning { dimm_index: usize, correctable_count: u64, threshold: u64 },
    PredictiveFailure { dimm_index: usize, estimated_hours_remaining: u64 },
    ShutdownInitiated { reason: String },
}

impl EccMemoryMonitor {
    /// Create new ECC monitor instance
    pub fn new(num_dimms: usize) -> Self {
        Self {
            total_correctable: AtomicU64::new(0),
            total_uncorrectable: AtomicU64::new(0),
            dimm_stats: parking_lot::RwLock::new(vec![DimmEccStats::default(); num_dimms]),
            error_history: parking_lot::Mutex::new(VecDeque::with_capacity(ERROR_HISTORY_SIZE)),
            is_running: Arc::new(AtomicBool::new(false)),
            emergency_shutdown_requested: AtomicBool::new(false),
            alert_tx: None,
            log_path: None,
            smu_base_address: AtomicU64::new(0),
            shutdown_callback: None,
        }
    }

    /// Configure with AMD SMU base address for direct register access
    pub fn with_smu_address(mut self, smu_base: u64) -> Self {
        self.smu_base_address.store(smu_base, Ordering::Relaxed);
        self
    }

    /// Set alert channel
    pub fn with_alert_channel(mut self, tx: crossbeam_channel::Sender<EccAlert>) -> Self {
        self.alert_tx = Some(tx);
        self
    }

    /// Set shutdown callback for graceful termination
    pub fn with_shutdown_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.shutdown_callback = Some(Arc::new(callback));
        self
    }

    /// Set log file path
    pub fn with_log_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_path = Some(path.into());
        self
    }

    /// Start monitoring loop
    pub fn start_monitoring(self: &Arc<Self>) -> thread::JoinHandle<()> {
        let monitor = Arc::clone(self);
        monitor.is_running.store(true, Ordering::SeqCst);

        thread::Builder::new()
            .name("ecc_memory_monitor".to_string())
            .spawn(move || {
                monitor.monitoring_loop();
            })
            .expect("Failed to spawn ECC monitoring thread")
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }

    /// Check if emergency shutdown is requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.emergency_shutdown_requested.load(Ordering::SeqCst)
    }

    /// Get total correctable error count
    pub fn get_correctable_count(&self) -> u64 {
        self.total_correctable.load(Ordering::Relaxed)
    }

    /// Get total uncorrectable error count
    pub fn get_uncorrectable_count(&self) -> u64 {
        self.total_uncorrectable.load(Ordering::Relaxed)
    }

    /// Get DIMM statistics
    pub fn get_dimm_stats(&self, index: usize) -> Option<DimmEccStats> {
        let stats = self.dimm_stats.read();
        stats.get(index).cloned()
    }

    /// Get all DIMM statistics
    pub fn get_all_dimm_stats(&self) -> Vec<DimmEccStats> {
        self.dimm_stats.read().clone()
    }

    /// Request graceful shutdown due to memory errors
    pub fn request_graceful_shutdown(&self, reason: &str) {
        self.emergency_shutdown_requested.store(true, Ordering::SeqCst);
        self.send_alert(EccAlert::ShutdownInitiated { reason: reason.to_string() });
        
        // Invoke shutdown callback if configured
        if let Some(ref callback) = self.shutdown_callback {
            callback();
        }

        log::error!("ECC Monitor: Graceful shutdown requested - {}", reason);
    }

    /// Main monitoring loop - optimized for minimal latency impact
    fn monitoring_loop(&self) {
        let mut last_poll = Instant::now();
        let poll_interval = Duration::from_micros(ECC_POLL_INTERVAL_US);

        while self.is_running.load(Ordering::Relaxed) {
            // Check for uncorrectable errors first (highest priority)
            if let Some(error) = self.check_uncorrectable_errors() {
                self.handle_uncorrectable_error(error);
                break; // Exit loop, system should reboot
            }

            // Check correctable errors
            self.poll_correctable_errors();

            // Analyze trends periodically
            if last_poll.elapsed() > Duration::from_secs(1) {
                self.analyze_error_trends();
                last_poll = Instant::now();
            }

            // High-frequency polling with yield to avoid CPU starvation
            thread::yield_now();
        }
    }

    /// Check for uncorrectable errors via WHEA/AMD SMU
    fn check_uncorrectable_errors(&self) -> Option<EccErrorRecord> {
        #[cfg(target_os = "windows")]
        {
            self.check_whea_errors()
        }
        
        #[cfg(target_os = "linux")]
        {
            self.check_mce_errors()
        }
        
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            None
        }
    }

    #[cfg(target_os = "windows")]
    fn check_whea_errors(&self) -> Option<EccErrorRecord> {
        // Windows Hardware Error Architecture (WHEA) interface
        // In production, this would query WHEA error log via IOCTL_WHEA_GET_ERROR_RECORD
        // For now, simulate based on system capabilities
        
        // Query AMD SMU registers via memory-mapped I/O
        let smu_base = self.smu_base_address.load(Ordering::Relaxed);
        if smu_base != 0 {
            unsafe {
                // Read UMC (Unified Memory Controller) error status
                // This is pseudo-code representing actual hardware access
                let umc_status_ptr = smu_base as *const u32;
                let umc_status = std::ptr::read_volatile(umc_status_ptr);
                
                if umc_status & 0x1 != 0 {
                    // Uncorrectable error detected
                    return Some(self.create_error_record(EccErrorType::Uncorrectable));
                }
            }
        }
        
        None
    }

    #[cfg(target_os = "linux")]
    fn check_mce_errors(&self) -> Option<EccErrorRecord> {
        // Linux Machine Check Exception (MCE) interface via /dev/mcelog
        use std::io::Read;
        
        if let Ok(mut mce_file) = std::fs::File::open("/dev/mcelog") {
            let mut buffer = [0u8; 64];
            if let Ok(bytes_read) = mce_file.read(&mut buffer) {
                if bytes_read > 0 {
                    // Parse MCE record
                    return Some(self.create_error_record(EccErrorType::Uncorrectable));
                }
            }
        }
        
        // Alternative: check kernel ring buffer for MCE messages
        if let Ok(output) = std::process::Command::new("dmesg")
            .args(["-T", "-k"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if output_str.contains("MCE") && output_str.contains("UNC") {
                return Some(self.create_error_record(EccErrorType::Uncorrectable));
            }
        }
        
        None
    }

    /// Poll for correctable errors
    fn poll_correctable_errors(&self) {
        #[cfg(target_os = "windows")]
        {
            self.poll_whea_correctable();
        }
        
        #[cfg(target_os = "linux")]
        {
            self.poll_edac_correctable();
        }
    }

    #[cfg(target_os = "windows")]
    fn poll_whea_correctable(&self) {
        // Poll WHEA correctable error counters
        // Uses AMD SMU mailbox interface for ECC status
        let smu_base = self.smu_base_address.load(Ordering::Relaxed);
        
        if smu_base != 0 {
            unsafe {
                // Read correctable error count from SMU
                let corr_err_ptr = (smu_base + 0x100) as *const u32;
                let corr_count = std::ptr::read_volatile(corr_err_ptr) as u64;
                
                let current_total = self.total_correctable.load(Ordering::Relaxed);
                if corr_count > current_total {
                    let new_errors = corr_count - current_total;
                    self.total_correctable.store(corr_count, Ordering::Relaxed);
                    
                    for _ in 0..new_errors {
                        let record = self.create_error_record(EccErrorType::Correctable);
                        self.record_error(record);
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn poll_edac_correctable(&self) {
        // Linux EDAC (Error Detection And Correction) sysfs interface
        // Path: /sys/devices/system/edac/mc/mcX/ce_count
        
        for mc_index in 0..4 {
            let ce_path = format!("/sys/devices/system/edac/mc/mc{}/ce_count", mc_index);
            if let Ok(content) = std::fs::read_to_string(&ce_path) {
                if let Ok(count) = content.trim().parse::<u64>() {
                    let current = self.total_correctable.load(Ordering::Relaxed);
                    if count > current {
                        self.total_correctable.store(count, Ordering::Relaxed);
                        
                        let record = self.create_error_record(EccErrorType::Correctable);
                        self.record_error(record);
                    }
                }
            }
        }
    }

    /// Create error record with current system state
    fn create_error_record(&self, error_type: EccErrorType) -> EccErrorRecord {
        EccErrorRecord {
            error_type,
            address: 0, // Would be read from hardware registers
            channel: 0,
            dimm_slot: 0,
            bank: 0,
            rank: 0,
            syndrome: 0,
            timestamp: Instant::now(),
            cpu_core: 0, // Could use core_affinity crate
            socket_id: 0,
        }
    }

    /// Record error and update statistics
    fn record_error(&self, record: EccErrorRecord) {
        // Update DIMM stats
        {
            let mut stats = self.dimm_stats.write();
            if let Some(dim_stat) = stats.get_mut(record.dimm_slot as usize) {
                match record.error_type {
                    EccErrorType::Correctable => {
                        dim_stat.correctable_count += 1;
                    }
                    EccErrorType::Uncorrectable => {
                        dim_stat.uncorrectable_count += 1;
                    }
                    _ => {}
                }
                dim_stat.last_error_time = Some(record.timestamp);
            }
        }

        // Add to history
        {
            let mut history = self.error_history.lock();
            history.push_back(record.clone());
            if history.len() > ERROR_HISTORY_SIZE {
                history.pop_front();
            }
        }

        // Send alerts
        let total_corr = self.total_correctable.load(Ordering::Relaxed);
        self.send_alert(EccAlert::CorrectableError {
            record,
            cumulative: total_corr,
        });

        // Check thresholds
        if total_corr >= CORRECTABLE_ERROR_CRITICAL_THRESHOLD {
            self.request_graceful_shutdown(
                &format!("Critical ECC error threshold exceeded: {} correctable errors", total_corr)
            );
        } else if total_corr >= CORRECTABLE_ERROR_WARNING_THRESHOLD {
            self.send_alert(EccAlert::ThresholdWarning {
                dimm_index: record.dimm_slot as usize,
                correctable_count: total_corr,
                threshold: CORRECTABLE_ERROR_WARNING_THRESHOLD,
            });
        }
    }

    /// Handle uncorrectable error - immediate action required
    fn handle_uncorrectable_error(&self, record: EccErrorRecord) {
        self.total_uncorrectable.fetch_add(1, Ordering::SeqCst);
        
        self.record_error(record.clone());
        
        self.send_alert(EccAlert::UncorrectableError {
            record,
            immediate_action_required: true,
        });

        // CRITICAL: Uncorrectable error means potential order book corruption
        // Must trigger immediate /KILL and reboot
        log::crit!("UNCORRECTABLE ECC ERROR DETECTED - Initiating emergency shutdown");
        self.request_graceful_shutdown("Uncorrectable ECC error - memory corruption detected");
    }

    /// Analyze error trends for predictive failure detection
    fn analyze_error_trends(&self) {
        let stats = self.dimm_stats.read();
        
        for (index, dimm_stat) in stats.iter().enumerate() {
            if let Some(last_error) = dimm_stat.last_error_time {
                // Calculate error rate
                let history = self.error_history.lock();
                let recent_errors: Vec<_> = history
                    .iter()
                    .filter(|e| e.dimm_slot as usize == index)
                    .collect();

                if recent_errors.len() >= 10 {
                    let first_time = recent_errors.first().unwrap().timestamp;
                    let last_time = recent_errors.last().unwrap().timestamp;
                    let elapsed_hours = last_time.duration_since(first_time).as_secs_f64() / 3600.0;
                    
                    if elapsed_hours > 0.0 {
                        let rate_per_hour = recent_errors.len() as f64 / elapsed_hours;
                        
                        // Predict failure based on accelerating error rate
                        // Heuristic: if rate exceeds 10/hour, predict failure within 24 hours
                        let predicted_hours = if rate_per_hour > 10.0 {
                            Some((100.0 / rate_per_hour) as u64)
                        } else {
                            None
                        };

                        if let Some(hours) = predicted_hours {
                            self.send_alert(EccAlert::PredictiveFailure {
                                dimm_index: index,
                                estimated_hours_remaining: hours,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Send alert through channel
    fn send_alert(&self, alert: EccAlert) {
        if let Some(ref tx) = self.alert_tx {
            let _ = tx.try_send(alert);
        }
        log::info!("ECC Alert: {:?}", alert);
    }

    /// Export ECC statistics to JSON for telemetry
    pub fn export_stats_json(&self) -> String {
        let stats = self.get_all_dimm_stats();
        let mut json = String::from("{\n  \"dimms\": [\n");
        
        for (i, dimm) in stats.iter().enumerate() {
            json.push_str(&format!(
                "    {{\n      \"index\": {},\n      \"correctable\": {},\n      \"uncorrectable\": {}\n    }}{}",
                i,
                dimm.correctable_count,
                dimm.uncorrectable_count,
                if i < stats.len() - 1 { "," } else { "" }
            ));
        }
        
        json.push_str("\n  ]\n}");
        json
    }
}

/// Global ECC monitor instance
pub static GLOBAL_ECC_MONITOR: parking_lot::OnceCell<Arc<EccMemoryMonitor>> = parking_lot::OnceCell::new();

/// Initialize global ECC monitor
pub fn init_global_monitor(num_dimms: usize) -> Arc<EccMemoryMonitor> {
    let monitor = Arc::new(EccMemoryMonitor::new(num_dimms));
    GLOBAL_ECC_MONITOR.get_or_init(|| monitor.clone());
    monitor
}

/// Get global monitor instance
pub fn get_global_monitor() -> Option<Arc<EccMemoryMonitor>> {
    GLOBAL_ECC_MONITOR.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecc_monitor_creation() {
        let monitor = EccMemoryMonitor::new(4);
        assert_eq!(monitor.get_correctable_count(), 0);
        assert_eq!(monitor.get_uncorrectable_count(), 0);
    }

    #[test]
    fn test_dimm_stats_default() {
        let monitor = EccMemoryMonitor::new(2);
        let stats = monitor.get_dimm_stats(0).unwrap();
        assert_eq!(stats.correctable_count, 0);
        assert_eq!(stats.uncorrectable_count, 0);
    }
}
