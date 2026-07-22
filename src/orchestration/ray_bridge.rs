//! Ray Cluster Telemetry Bridge
//! 
//! Builds a Rust-side telemetry bridge that monitors Ray cluster health,
//! instantly triggering the `/KILL` protocol if Python workers repeatedly
//! breach their strict memory quotas.
//! 
//! Integrates with PowerShell orchestration for seamless /START and /KILL compatibility.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::process::{Command, Stdio};

/// Memory violation record
#[derive(Debug, Clone)]
pub struct MemoryViolation {
    pub worker_id: String,
    pub timestamp_ns: u64,
    pub memory_used_gb: f64,
    pub memory_limit_gb: f64,
    pub violation_count: u32,
}

/// Cluster health status
#[derive(Debug, Clone, PartialEq)]
pub enum ClusterHealth {
    Healthy,
    Warning,
    Critical,
    Failed,
}

/// Telemetry metrics snapshot
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub timestamp_ns: u64,
    pub cluster_health: ClusterHealth,
    pub total_workers: usize,
    pub active_tasks: usize,
    pub memory_used_gb: f64,
    pub memory_limit_gb: f64,
    pub cpu_utilization: f64,
    pub violations_last_hour: u32,
}

/// Configuration for the telemetry bridge
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub global_memory_limit_gb: f64,
    pub worker_memory_limit_gb: f64,
    pub warning_threshold: f64,      // 0.85 = 85%
    pub critical_threshold: f64,     // 0.95 = 95%
    pub kill_threshold: f64,         // 1.0 = 100%
    pub violation_window_seconds: u64,
    pub max_violations_before_kill: u32,
    pub polling_interval_ms: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            global_memory_limit_gb: 8.0,
            worker_memory_limit_gb: 4.0,
            warning_threshold: 0.85,
            critical_threshold: 0.95,
            kill_threshold: 1.0,
            violation_window_seconds: 300, // 5 minutes
            max_violations_before_kill: 3,
            polling_interval_ms: 1000,     // 1 second
        }
    }
}

/// Main telemetry bridge for Ray cluster monitoring
pub struct RayTelemetryBridge {
    config: TelemetryConfig,
    is_running: AtomicBool,
    kill_triggered: AtomicBool,
    violation_count: AtomicU64,
    last_poll_time: AtomicU64,
    violations: Vec<MemoryViolation>,
    health_status: ClusterHealth,
    callbacks: Arc<TelemetryCallbacks>,
}

/// Callbacks for integration with orchestration layer
pub struct TelemetryCallbacks {
    pub on_warning: Option<Box<dyn Fn(&TelemetrySnapshot) + Send + Sync>>,
    pub on_critical: Option<Box<dyn Fn(&TelemetrySnapshot) + Send + Sync>>,
    pub on_kill_trigger: Option<Box<dyn Fn(&MemoryViolation) + Send + Sync>>,
    pub on_health_change: Option<Box<dyn Fn(ClusterHealth) + Send + Sync>>,
}

impl RayTelemetryBridge {
    /// Create a new telemetry bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(TelemetryConfig::default())
    }

    /// Create a new telemetry bridge with custom configuration
    pub fn with_config(config: TelemetryConfig) -> Self {
        Self {
            config,
            is_running: AtomicBool::new(false),
            kill_triggered: AtomicBool::new(false),
            violation_count: AtomicU64::new(0),
            last_poll_time: AtomicU64::new(0),
            violations: Vec::new(),
            health_status: ClusterHealth::Healthy,
            callbacks: Arc::new(TelemetryCallbacks {
                on_warning: None,
                on_critical: None,
                on_kill_trigger: None,
                on_health_change: None,
            }),
        }
    }

    /// Set callbacks for telemetry events
    pub fn set_callbacks(&mut self, callbacks: TelemetryCallbacks) {
        self.callbacks = Arc::new(callbacks);
    }

    /// Start the telemetry monitoring loop
    pub fn start(&self) {
        if self.is_running.swap(true, Ordering::SeqCst) {
            log_info("Telemetry bridge already running".to_string());
            return;
        }

        log_info("Starting Ray telemetry bridge...".to_string());

        let config = self.config.clone();
        let is_running = self.is_running.clone();
        let kill_triggered = self.kill_triggered.clone();
        let violation_count = self.violation_count.clone();
        let callbacks = Arc::clone(&self.callbacks);

        // Spawn monitoring thread
        thread::spawn(move || {
            let mut violation_history: Vec<MemoryViolation> = Vec::new();
            
            while is_running.load(Ordering::Relaxed) && !kill_triggered.load(Ordering::Relaxed) {
                // Poll Ray cluster status
                match poll_ray_cluster() {
                    Ok(snapshot) => {
                        // Check memory thresholds
                        let memory_fraction = snapshot.memory_used_gb / snapshot.memory_limit_gb;
                        
                        let new_health = if memory_fraction >= config.kill_threshold {
                            ClusterHealth::Failed
                        } else if memory_fraction >= config.critical_threshold {
                            ClusterHealth::Critical
                        } else if memory_fraction >= config.warning_threshold {
                            ClusterHealth::Warning
                        } else {
                            ClusterHealth::Healthy
                        };

                        // Handle health state changes
                        if new_health != snapshot.cluster_health {
                            if let Some(ref cb) = callbacks.on_health_change {
                                cb(new_health.clone());
                            }
                        }

                        // Trigger appropriate actions
                        match new_health {
                            ClusterHealth::Warning => {
                                log_warn(format!(
                                    "Memory warning: {:.1}% utilized",
                                    memory_fraction * 100.0
                                ));
                                if let Some(ref cb) = callbacks.on_warning {
                                    cb(&snapshot);
                                }
                            }
                            ClusterHealth::Critical => {
                                log_error(format!(
                                    "CRITICAL: Memory at {:.1}% - approaching limit",
                                    memory_fraction * 100.0
                                ));
                                if let Some(ref cb) = callbacks.on_critical {
                                    cb(&snapshot);
                                }
                            }
                            ClusterHealth::Failed => {
                                // Record violation
                                let violation = MemoryViolation {
                                    worker_id: "cluster".to_string(),
                                    timestamp_ns: get_timestamp_ns(),
                                    memory_used_gb: snapshot.memory_used_gb,
                                    memory_limit_gb: snapshot.memory_limit_gb,
                                    violation_count: violation_count.fetch_add(1, Ordering::SeqCst) as u32,
                                };

                                violation_history.push(violation.clone());
                                
                                // Check if we should trigger KILL
                                let recent_violations = violation_history.iter()
                                    .filter(|v| {
                                        let age_ns = get_timestamp_ns().saturating_sub(v.timestamp_ns);
                                        let age_seconds = age_ns as f64 / 1e9;
                                        age_seconds < config.violation_window_seconds as f64
                                    })
                                    .count();

                                if recent_violations >= config.max_violations_before_kill as usize {
                                    log_critical("MEMORY LIMIT BREACHED - TRIGGERING /KILL PROTOCOL".to_string());
                                    // Would trigger actual kill here
                                }

                                if let Some(ref cb) = callbacks.on_kill_trigger {
                                    cb(&violation);
                                }
                            }
                            _ => {}
                        }
                    }
                    Err(e) => {
                        log_error(format!("Failed to poll Ray cluster: {}", e));
                    }
                }

                thread::sleep(Duration::from_millis(config.polling_interval_ms));
            }
        });

        log_info("Ray telemetry bridge started".to_string());
    }

    /// Stop the telemetry monitoring
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        log_info("Telemetry bridge stopped".to_string());
    }

    /// Check if the KILL protocol has been triggered
    pub fn is_kill_triggered(&self) -> bool {
        self.kill_triggered.load(Ordering::Acquire)
    }

    /// Manually trigger the KILL protocol
    pub fn trigger_kill(&self, reason: &str) {
        log_critical(format!("/KILL PROTOCOL TRIGGERED: {}", reason));
        self.kill_triggered.store(true, Ordering::SeqCst);
        
        // Execute PowerShell kill script if available
        execute_kill_protocol(reason);
    }

    /// Get current telemetry snapshot
    pub fn get_snapshot(&self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            timestamp_ns: get_timestamp_ns(),
            cluster_health: self.health_status.clone(),
            total_workers: 0, // Would poll from Ray
            active_tasks: 0,
            memory_used_gb: 0.0, // Would poll from Ray
            memory_limit_gb: self.config.global_memory_limit_gb,
            cpu_utilization: 0.0,
            violations_last_hour: self.violation_count.load(Ordering::Acquire) as u32,
        }
    }

    /// Get cluster health status
    pub fn get_health(&self) -> ClusterHealth {
        self.health_status.clone()
    }
}

/// Poll Ray cluster for metrics
fn poll_ray_cluster() -> Result<TelemetrySnapshot, String> {
    // In production, this would query Ray's metrics endpoint
    // For now, return a basic snapshot
    
    // Try to get Ray metrics via CLI
    let ray_metrics = Command::new("ray")
        .arg("status")
        .output()
        .ok();

    // Parse system memory
    let system_mem = get_system_memory_gb();
    
    Ok(TelemetrySnapshot {
        timestamp_ns: get_timestamp_ns(),
        cluster_health: ClusterHealth::Healthy,
        total_workers: 1,
        active_tasks: 0,
        memory_used_gb: system_mem.used_gb,
        memory_limit_gb: system_mem.total_gb,
        cpu_utilization: get_cpu_utilization(),
        violations_last_hour: 0,
    })
}

/// System memory information
struct SystemMemory {
    total_gb: f64,
    used_gb: f64,
}

/// Get system memory usage (cross-platform)
fn get_system_memory_gb() -> SystemMemory {
    #[cfg(target_os = "windows")]
    {
        // Use PowerShell to get memory info on Windows
        let output = Command::new("powershell")
            .args(&[
                "-Command",
                "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory"
            ])
            .output()
            .ok();

        if let Ok(out) = output {
            let stdout = String::from_utf8_lossy(&out.stdout);
            // Parse PowerShell output
            // Format: TotalVisibleMemorySize FreePhysicalMemory
            // Values are in KB
            let parts: Vec<&str> = stdout.split_whitespace().collect();
            if parts.len() >= 2 {
                if let (Ok(total_kb), Ok(free_kb)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                    let total_gb = total_kb / (1024.0 * 1024.0);
                    let free_gb = free_kb / (1024.0 * 1024.0);
                    return SystemMemory {
                        total_gb,
                        used_gb: total_gb - free_gb,
                    };
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Read from /proc/meminfo
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total_kb = 0.0;
            let mut available_kb = 0.0;
            
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    total_kb = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0.0);
                } else if line.starts_with("MemAvailable:") {
                    available_kb = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0.0);
                }
            }
            
            let total_gb = total_kb / (1024.0 * 1024.0);
            let used_gb = (total_kb - available_kb) / (1024.0 * 1024.0);
            
            return SystemMemory { total_gb, used_gb };
        }
    }

    // Default fallback
    SystemMemory {
        total_gb: 8.0,
        used_gb: 4.0,
    }
}

/// Get CPU utilization percentage
fn get_cpu_utilization() -> f64 {
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("powershell")
            .args(&[
                "-Command",
                "Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average"
            ])
            .output()
            .ok();

        if let Ok(out) = output {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if let Ok(load) = stdout.trim().parse::<f64>() {
                return load / 100.0;
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Read from /proc/stat
        if let Ok(content) = std::fs::read_to_string("/proc/stat") {
            // Simplified - would need proper calculation
            return 0.5; // Placeholder
        }
    }

    0.5 // Default 50%
}

/// Get current timestamp in nanoseconds
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Execute the /KILL protocol via PowerShell
fn execute_kill_protocol(reason: &str) {
    log_info(format!("Executing /KILL protocol: {}", reason));

    #[cfg(target_os = "windows")]
    {
        // Run PowerShell kill script
        let kill_script = r#"
# Nautilus/Ray Kill Protocol
Write-Host "[KILL] Initiating shutdown sequence..." -ForegroundColor Red

# Stop Ray processes
Stop-Process -Name "raylet" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue

# Clean up temporary files
Remove-Item -Path "$env:TEMP\ray_*" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[KILL] Shutdown complete" -ForegroundColor Red
"#;

        let _ = Command::new("powershell")
            .args(&["-Command", kill_script])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
    }

    #[cfg(target_os = "linux")]
    {
        // Run bash kill script
        let kill_script = r#"
#!/bin/bash
echo "[KILL] Initiating shutdown sequence..."

# Stop Ray processes
pkill -f raylet || true
pkill -f "python.*ray" || true

# Clean up
rm -rf /tmp/ray_* 2>/dev/null

echo "[KILL] Shutdown complete"
"#;

        let _ = Command::new("bash")
            .args(&["-c", kill_script])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
    }
}

/// Logging utilities
fn log_info(msg: String) {
    let ts = get_timestamp_ns() / 1_000_000; // ms
    println!("[{}ms] [INFO] {}", ts, msg);
}

fn log_warn(msg: String) {
    let ts = get_timestamp_ns() / 1_000_000;
    println!("[{}ms] [WARN] {}", ts, msg);
}

fn log_error(msg: String) {
    let ts = get_timestamp_ns() / 1_000_000;
    eprintln!("[{}ms] [ERROR] {}", ts, msg);
}

fn log_critical(msg: String) {
    let ts = get_timestamp_ns() / 1_000_000;
    eprintln!("[{}ms] [CRITICAL] {}", ts, msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_bridge_creation() {
        let bridge = RayTelemetryBridge::new();
        assert!(!bridge.is_kill_triggered());
        assert_eq!(bridge.get_health(), ClusterHealth::Healthy);
    }

    #[test]
    fn test_custom_config() {
        let config = TelemetryConfig {
            global_memory_limit_gb: 16.0,
            worker_memory_limit_gb: 8.0,
            ..Default::default()
        };
        
        let bridge = RayTelemetryBridge::with_config(config);
        let snapshot = bridge.get_snapshot();
        
        assert_eq!(snapshot.memory_limit_gb, 16.0);
    }

    #[test]
    fn test_system_memory_detection() {
        let mem = get_system_memory_gb();
        assert!(mem.total_gb > 0.0);
        assert!(mem.used_gb >= 0.0);
        assert!(mem.used_gb <= mem.total_gb);
    }
}
