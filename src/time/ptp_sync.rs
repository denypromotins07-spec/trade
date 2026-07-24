// =============================================================================
// Nautilus/Ray Crypto Trading Bot - Stage 50
// File: src/time/ptp_sync.rs
// Chapter 3: Precision Time Protocol (PTP) & Hardware Timestamping (Rust)
// 
// Purpose: Implement a lightweight PTP client to synchronize the local
//          AMD Ryzen CPU Time Stamp Counter (TSC) with Binance's server
//          clocks, achieving sub-microsecond global time alignment.
//
// Optimization Targets:
//   - Sub-microsecond time synchronization
//   - Network partition tolerance
//   - AMD Ryzen AI 5 TSC optimization
//   - No blocking on network failures
//
// Constraints:
//   - No LLMs, strict typing, production-grade code
//   - Graceful degradation on network issues
// =============================================================================

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// PTP state enumeration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PtpState {
    /// Initial state, attempting to sync.
    Initializing,
    /// Synchronized with remote clock.
    Synchronized,
    /// Holdover mode (lost sync, using last known offset).
    Holdover,
    /// Error state (network partition detected).
    Error,
}

/// PTP synchronization statistics.
#[derive(Debug, Clone, Copy)]
pub struct PtpStats {
    pub state: PtpState,
    pub offset_ns: i64,
    pub jitter_ns: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub sync_failures: u64,
}

/// Lightweight PTP client for TSC synchronization.
pub struct PtpClient {
    /// Current synchronization state.
    state: AtomicU8,
    /// Estimated offset from remote clock (nanoseconds).
    offset_ns: AtomicI64,
    /// Estimated jitter (nanoseconds).
    jitter_ns: AtomicU64,
    /// Last successful sync timestamp.
    last_sync_time: AtomicU64,
    /// Packet counters.
    packets_sent: AtomicU64,
    packets_received: AtomicU64,
    sync_failures: AtomicU64,
    /// Network partition detection flag.
    partition_detected: AtomicBool,
    /// Holdover timeout (seconds).
    holdover_timeout_secs: u64,
}

// AtomicU8 requires explicit definition.
use std::sync::atomic::AtomicU8;

unsafe impl Send for PtpClient {}
unsafe impl Sync for PtpClient {}

impl PtpClient {
    /// Create a new PTP client.
    /// 
    /// # Arguments
    /// * `holdover_timeout_secs` - Seconds before entering holdover after lost sync
    pub fn new(holdover_timeout_secs: u64) -> Self {
        Self {
            state: AtomicU8::new(PtpState::Initializing as u8),
            offset_ns: AtomicI64::new(0),
            jitter_ns: AtomicU64::new(0),
            last_sync_time: AtomicU64::new(0),
            packets_sent: AtomicU64::new(0),
            packets_received: AtomicU64::new(0),
            sync_failures: AtomicU64::new(0),
            partition_detected: AtomicBool::new(false),
            holdover_timeout_secs,
        }
    }
    
    /// Perform a PTP synchronization exchange.
    /// 
    /// This implements a simplified PTP-like exchange:
    /// 1. Send Sync request with local TSC timestamp (T1)
    /// 2. Receive response with remote timestamp (T2) and reply timestamp (T3)
    /// 3. Record local receive time (T4)
    /// 4. Calculate offset: ((T2 - T1) + (T3 - T4)) / 2
    /// 
    /// # Arguments
    /// * `remote_ts_provider` - Function that returns (T2, T3) remote timestamps
    /// 
    /// # Returns
    /// true if sync successful, false on failure
    pub fn sync<F>(&self, remote_ts_provider: F) -> bool
    where
        F: Fn() -> Option<(u64, u64)>, // Returns (T2, T3) in nanoseconds
    {
        self.packets_sent.fetch_add(1, Ordering::Relaxed);
        
        // T1: Local send timestamp (TSC cycles converted to ns).
        let t1 = get_tsc_nanoseconds();
        
        // Get remote timestamps.
        let (t2, t3) = match remote_ts_provider() {
            Some(ts) => ts,
            None => {
                self.handle_sync_failure();
                return false;
            }
        };
        
        self.packets_received.fetch_add(1, Ordering::Relaxed);
        
        // T4: Local receive timestamp.
        let t4 = get_tsc_nanoseconds();
        
        // Calculate round-trip delay and offset.
        // Delay = (T4 - T1) - (T3 - T2)
        // Offset = ((T2 - T1) + (T3 - T4)) / 2
        let delay_ns = (t4.wrapping_sub(t1) as i64) - (t3.wrapping_sub(t2) as i64);
        let offset_ns = ((t2.wrapping_sub(t1) as i64) + (t3.wrapping_sub(t4) as i64)) / 2;
        
        // Validate timing (detect network anomalies).
        if delay_ns < 0 || delay_ns > 10_000_000 {
            // Negative delay or >10ms indicates bad measurement.
            self.handle_sync_failure();
            return false;
        }
        
        // Update offset with exponential moving average.
        let current_offset = self.offset_ns.load(Ordering::Relaxed);
        let new_offset = (current_offset * 7 + offset_ns) / 8; // EMA with alpha=0.125
        self.offset_ns.store(new_offset, Ordering::Relaxed);
        
        // Update jitter estimate.
        let jitter = (offset_ns - current_offset).unsigned_abs();
        let current_jitter = self.jitter_ns.load(Ordering::Relaxed);
        let new_jitter = (current_jitter * 7 + jitter) / 8;
        self.jitter_ns.store(new_jitter, Ordering::Relaxed);
        
        // Update state.
        self.state.store(PtpState::Synchronized as u8, Ordering::Relaxed);
        self.last_sync_time.store(get_unix_timestamp_nanos(), Ordering::Relaxed);
        self.partition_detected.store(false, Ordering::Relaxed);
        
        true
    }
    
    /// Handle sync failure with backoff and partition detection.
    fn handle_sync_failure(&self) {
        let failures = self.sync_failures.fetch_add(1, Ordering::Relaxed) + 1;
        
        if failures >= 5 {
            self.partition_detected.store(true, Ordering::Relaxed);
            
            let current_state = self.state.load(Ordering::Relaxed);
            if current_state == PtpState::Synchronized as u8 {
                // Check if we should enter holdover.
                let last_sync = self.last_sync_time.load(Ordering::Relaxed);
                let now = get_unix_timestamp_nanos();
                let elapsed_secs = (now - last_sync) / 1_000_000_000;
                
                if elapsed_secs > self.holdover_timeout_secs as u64 {
                    self.state.store(PtpState::Holdover as u8, Ordering::Relaxed);
                } else {
                    self.state.store(PtpState::Error as u8, Ordering::Relaxed);
                }
            }
        }
    }
    
    /// Get current time synchronized to remote clock.
    /// 
    /// # Returns
    /// Synchronized timestamp in nanoseconds, or None if not synced
    pub fn get_synced_time(&self) -> Option<u64> {
        let state = self.state.load(Ordering::Relaxed);
        
        match PtpState::from_u8(state) {
            Some(PtpState::Synchronized) | Some(PtpState::Holdover) => {
                let local_ts = get_tsc_nanoseconds();
                let offset = self.offset_ns.load(Ordering::Relaxed);
                Some(local_ts.wrapping_add(offset as u64))
            }
            _ => None,
        }
    }
    
    /// Convert remote exchange timestamp to local TSC time.
    /// 
    /// # Arguments
    /// * `exchange_ts` - Timestamp from exchange (nanoseconds)
    /// 
    /// # Returns
    /// Estimated local TSC time when exchange timestamp occurred
    pub fn exchange_to_local(&self, exchange_ts: u64) -> u64 {
        let offset = self.offset_ns.load(Ordering::Relaxed);
        // Inverse of local->remote conversion.
        exchange_ts.wrapping_sub(offset as u64)
    }
    
    /// Check if currently synchronized.
    pub fn is_synchronized(&self) -> bool {
        let state = self.state.load(Ordering::Relaxed);
        matches!(
            PtpState::from_u8(state),
            Some(PtpState::Synchronized) | Some(PtpState::Holdover)
        )
    }
    
    /// Get PTP statistics.
    pub fn get_stats(&self) -> PtpStats {
        let state_val = self.state.load(Ordering::Relaxed);
        PtpStats {
            state: PtpState::from_u8(state_val).unwrap_or(PtpState::Error),
            offset_ns: self.offset_ns.load(Ordering::Relaxed),
            jitter_ns: self.jitter_ns.load(Ordering::Relaxed),
            packets_sent: self.packets_sent.load(Ordering::Relaxed),
            packets_received: self.packets_received.load(Ordering::Relaxed),
            sync_failures: self.sync_failures.load(Ordering::Relaxed),
        }
    }
    
    /// Reset sync state (force re-synchronization).
    pub fn reset(&self) {
        self.state.store(PtpState::Initializing as u8, Ordering::Relaxed);
        self.sync_failures.store(0, Ordering::Relaxed);
        self.partition_detected.store(false, Ordering::Relaxed);
    }
}

impl Default for PtpClient {
    fn default() -> Self {
        Self::new(60) // 60 second holdover timeout
    }
}

/// Get current TSC value converted to nanoseconds.
#[inline]
fn get_tsc_nanoseconds() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::_rdtsc;
        unsafe {
            let tsc = _rdtsc();
            // Convert TSC cycles to nanoseconds (approximate, assumes ~4GHz CPU).
            // In production, calibrate this conversion factor.
            tsc * 1_000_000_000 / 4_000_000_000
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Instant::now().elapsed().as_nanos() as u64
    }
}

/// Get Unix timestamp in nanoseconds.
#[inline]
fn get_unix_timestamp_nanos() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// PTP State enum with conversion helpers.
#[repr(u8)]
enum PtpStateInternal {
    Initializing = 0,
    Synchronized = 1,
    Holdover = 2,
    Error = 3,
}

impl PtpState {
    fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(PtpState::Initializing),
            1 => Some(PtpState::Synchronized),
            2 => Some(PtpState::Holdover),
            3 => Some(PtpState::Error),
            _ => None,
        }
    }
}

/// Logging macros.
macro_rules! log_info {
    ($($arg:tt)*) => {
        eprintln!("[INFO] {}", format!($($arg)*));
    };
}

macro_rules! log_warn {
    ($($arg:tt)*) => {
        eprintln!("[WARN] {}", format!($($arg)*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_creation() {
        let client = PtpClient::new(30);
        let stats = client.get_stats();
        assert_eq!(stats.state, PtpState::Initializing);
    }
    
    #[test]
    fn test_sync_success() {
        let client = PtpClient::new(60);
        
        // Simulate successful sync with mock timestamps.
        let result = client.sync(|| {
            Some((1000, 1001)) // T2, T3
        });
        
        assert!(result);
        assert!(client.is_synchronized());
    }
    
    #[test]
    fn test_sync_failure() {
        let client = PtpClient::new(60);
        
        // Simulate repeated failures.
        for _ in 0..5 {
            client.sync(|| None);
        }
        
        let stats = client.get_stats();
        assert_eq!(stats.sync_failures, 5);
        assert!(stats.state == PtpState::Error || stats.state == PtpState::Holdover);
    }
}
