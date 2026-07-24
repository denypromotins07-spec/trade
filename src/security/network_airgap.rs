// =============================================================================
// Nautilus/Ray Bot - Stage 53: Network Airgap Monitor
// File: src/security/network_airgap.rs
// Purpose: Monitor for unauthorized ARP requests or broadcast storms and
//          trigger /KILL sequence if detected.
// Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
// Constraints: Microsecond Detection, 8GB RAM Limit
// =============================================================================

use std::net::{IpAddr, Ipv4Addr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

/// Maximum allowed ARP requests per second before triggering alarm
const MAX_ARP_RATE_PER_SEC: u64 = 10;

/// Maximum allowed broadcast packets per second
const MAX_BROADCAST_RATE_PER_SEC: u64 = 50;

/// Network airgap monitor state
pub struct NetworkAirgapMonitor {
    /// Flag indicating if monitor is active
    is_active: AtomicBool,
    /// Flag indicating if kill signal has been triggered
    kill_triggered: AtomicBool,
    /// Count of ARP requests in current window
    arp_count: AtomicU64,
    /// Count of broadcast packets in current window
    broadcast_count: AtomicU64,
    /// Start time of current measurement window
    window_start: AtomicU64,
}

impl NetworkAirgapMonitor {
    pub fn new() -> Self {
        Self {
            is_active: AtomicBool::new(false),
            kill_triggered: AtomicBool::new(false),
            arp_count: AtomicU64::new(0),
            broadcast_count: AtomicU64::new(0),
            window_start: AtomicU64::new(0),
        }
    }

    /// Start the monitoring thread
    pub fn start(&self) -> Result<(), String> {
        if self.is_active.load(Ordering::SeqCst) {
            return Err("Monitor already active".to_string());
        }

        log::info!("Starting Network Airgap Monitor...");
        
        self.window_start.store(
            Instant::now().duration_since(Instant::now()).as_secs(), 
            Ordering::Relaxed
        );
        self.is_active.store(true, Ordering::SeqCst);

        // Spawn monitoring thread
        let monitor = Arc::new(self.clone_state());
        thread::spawn(move || {
            Self::monitoring_loop(monitor);
        });

        log::info!("Network Airgap Monitor started.");
        Ok(())
    }

    /// Clone atomic state for thread
    fn clone_state(&self) -> MonitoredState {
        MonitoredState {
            is_active: self.is_active.clone(),
            kill_triggered: self.kill_triggered.clone(),
            arp_count: self.arp_count.clone(),
            broadcast_count: self.broadcast_count.clone(),
            window_start: self.window_start.clone(),
        }
    }

    /// Main monitoring loop (simulated - real impl needs packet capture)
    fn monitoring_loop(state: Arc<MonitoredState>) {
        while state.is_active.load(Ordering::SeqCst) {
            // In production, this would use a packet capture library (e.g., winpcap, libpnet)
            // to inspect incoming frames at the NIC level.
            
            // Simulated check interval
            thread::sleep(Duration::from_millis(100));
            
            // Check for anomalies
            if state.should_trigger_kill() {
                log::error!("NETWORK ANOMALY DETECTED! Triggering /KILL sequence...");
                state.kill_triggered.store(true, Ordering::SeqCst);
                
                // In real impl, call the global kill switch here
                // crate::system::trigger_emergency_kill();
                
                break;
            }
            
            // Reset counters every second
            let elapsed = Instant::now().duration_since(Instant::now()).as_secs();
            if elapsed >= 1 {
                state.arp_count.store(0, Ordering::Relaxed);
                state.broadcast_count.store(0, Ordering::Relaxed);
            }
        }
    }

    /// Inject a received packet for analysis (called by network driver)
    pub fn on_packet_received(&self, is_arp: bool, is_broadcast: bool) {
        if !self.is_active.load(Ordering::SeqCst) {
            return;
        }

        if is_arp {
            let count = self.arp_count.fetch_add(1, Ordering::Relaxed);
            if count > MAX_ARP_RATE_PER_SEC {
                log::warn!("High ARP rate detected: {}", count + 1);
            }
        }

        if is_broadcast {
            let count = self.broadcast_count.fetch_add(1, Ordering::Relaxed);
            if count > MAX_BROADCAST_RATE_PER_SEC {
                log::warn!("High broadcast rate detected: {}", count + 1);
            }
        }
    }

    /// Check if kill should be triggered
    fn should_trigger_kill(&self) -> bool {
        let arp = self.arp_count.load(Ordering::Relaxed);
        let bcast = self.broadcast_count.load(Ordering::Relaxed);

        if arp > MAX_ARP_RATE_PER_SEC {
            log::error!("ARP storm detected: {} > {}", arp, MAX_ARP_RATE_PER_SEC);
            return true;
        }

        if bcast > MAX_BROADCAST_RATE_PER_SEC {
            log::error!("Broadcast storm detected: {} > {}", bcast, MAX_BROADCAST_RATE_PER_SEC);
            return true;
        }

        false
    }

    /// Check if kill has been triggered
    pub fn is_kill_triggered(&self) -> bool {
        self.kill_triggered.load(Ordering::SeqCst)
    }

    /// Stop the monitor
    pub fn stop(&self) {
        log::warn!("Stopping Network Airgap Monitor...");
        self.is_active.store(false, Ordering::SeqCst);
    }
}

/// Shared state for monitoring thread
struct MonitoredState {
    is_active: AtomicBool,
    kill_triggered: AtomicBool,
    arp_count: AtomicU64,
    broadcast_count: AtomicU64,
    window_start: AtomicU64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_creation() {
        let monitor = NetworkAirgapMonitor::new();
        assert!(!monitor.is_active.load(Ordering::SeqCst));
        assert!(!monitor.kill_triggered.load(Ordering::SeqCst));
    }

    #[test]
    fn test_packet_injection() {
        let monitor = NetworkAirgapMonitor::new();
        monitor.on_packet_received(true, false); // ARP
        monitor.on_packet_received(false, true); // Broadcast
        
        assert_eq!(monitor.arp_count.load(Ordering::Relaxed), 1);
        assert_eq!(monitor.broadcast_count.load(Ordering::Relaxed), 1);
    }
}
