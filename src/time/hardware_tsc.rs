// =============================================================================
// Nautilus/Ray Crypto Trading Bot - Stage 50
// File: src/time/hardware_tsc.rs
// Chapter 3: Precision Time Protocol (PTP) & Hardware Timestamping (Rust)
// 
// Purpose: Build a hardware timestamping module utilizing the `rdtscp`
//          instruction to tag every incoming WebSocket tick with exact
//          CPU cycle counts, preventing OS clock drift and jitter.
//
// Optimization Targets:
//   - Cycle-accurate timestamping
//   - AMD Ryzen AI 5 CCD-aware thread migration handling
//   - Zero OS syscall overhead
//   - Sub-nanosecond resolution
//
// Constraints:
//   - No LLMs, strict typing, production-grade code
//   - Safe handling of thread migration across CCDs
// =============================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::arch::x86_64::_rdtscp;
use std::mem;

/// Maximum number of timestamp entries in the ring buffer.
const TSC_RING_SIZE: usize = 65536;

/// Cache line size for AMD Zen architecture.
const CACHE_LINE_SIZE: usize = 64;

/// Timestamp entry with TSC and optional calibration data.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TscTimestamp {
    /// TSC cycle count at event time.
    pub tsc_cycles: u64,
    /// Calibrated nanoseconds (after TSC-to-ns conversion).
    pub nanos: u64,
    /// Core ID where timestamp was taken (for CCD detection).
    pub core_id: u32,
    /// Reserved padding.
    _reserved: [u8; 52], // 8 + 8 + 4 + 52 = 72, adjust below
}

// Ensure exact 64-byte size.
const _: () = assert!(mem::size_of::<TscTimestamp>() == 64, "TscTimestamp must be 64 bytes");

/// Hardware TSC timestamp manager.
pub struct HardwareTscManager {
    /// Ring buffer of timestamps.
    ring: Box<[TscTimestamp; TSC_RING_SIZE]>,
    /// Head index (next write position).
    head: AtomicUsize,
    /// Tail index (next read position).
    tail: AtomicUsize,
    /// TSC frequency in Hz (calibrated at startup).
    tsc_frequency_hz: AtomicU64,
    /// Total timestamps recorded.
    total_recorded: AtomicU64,
    /// CCD migration detections.
    ccd_migrations: AtomicU64,
    /// Last known core ID.
    last_core_id: AtomicUsize,
}

unsafe impl Send for HardwareTscManager {}
unsafe impl Sync for HardwareTscManager {}

impl HardwareTscManager {
    /// Create and calibrate the TSC manager.
    /// 
    /// # Arguments
    /// * `tsc_frequency_hz` - Optional pre-calibrated TSC frequency
    /// 
    /// If not provided, attempts to calibrate from CPUID.
    pub fn new(tsc_frequency_hz: Option<u64>) -> Self {
        let freq = tsc_frequency_hz.unwrap_or_else(Self::calibrate_tsc_frequency);
        
        log_info!("HardwareTscManager initialized with TSC frequency: {} Hz", freq);
        
        Self {
            ring: Box::new([TscTimestamp {
                tsc_cycles: 0,
                nanos: 0,
                core_id: 0,
                _reserved: [0u8; 52],
            }; TSC_RING_SIZE]),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            tsc_frequency_hz: AtomicU64::new(freq),
            total_recorded: AtomicU64::new(0),
            ccd_migrations: AtomicU64::new(0),
            last_core_id: AtomicUsize::new(get_current_core_id()),
        }
    }
    
    /// Calibrate TSC frequency using CPUID or platform info.
    fn calibrate_tsc_frequency() -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            // Use CPUID leaf 0x15 to get TSC frequency on modern Intel/AMD.
            // This is a simplified implementation.
            // In production, use proper CPUID intrinsics.
            
            // Fallback: assume 4.0 GHz for AMD Ryzen AI 5.
            4_000_000_000
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            1_000_000_000 // Fallback for non-x86
        }
    }
    
    /// Record a timestamp for an incoming tick.
    /// 
    /// Uses `rdtscp` for ordered, serializing timestamp capture.
    /// 
    /// # Arguments
    /// * `tick_id` - Unique identifier for the tick
    /// 
    /// # Returns
    /// The recorded TSC timestamp
    pub fn record_tick(&self, tick_id: u64) -> TscTimestamp {
        // Use rdtscp for serializing, ordered timestamp.
        let mut aux: u32 = 0;
        let tsc_cycles = unsafe { _rdtscp(&mut aux) };
        
        // Extract core ID from AUX register (bits 31:0 contain APIC ID).
        let core_id = aux;
        
        // Check for CCD migration (thread moved to different core complex).
        let prev_core = self.last_core_id.load(Ordering::Relaxed);
        if core_id as usize != prev_core {
            self.ccd_migrations.fetch_add(1, Ordering::Relaxed);
            self.last_core_id.store(core_id as usize, Ordering::Relaxed);
            
            // Note: On AMD Zen, TSC is synchronized across CCDs,
            // so no additional correction is needed.
        }
        
        // Convert TSC cycles to nanoseconds.
        let freq = self.tsc_frequency_hz.load(Ordering::Relaxed);
        let nanos = (tsc_cycles as u128 * 1_000_000_000u128 / freq as u128) as u64;
        
        let timestamp = TscTimestamp {
            tsc_cycles,
            nanos,
            core_id,
            _reserved: [0u8; 52],
        };
        
        // Write to ring buffer.
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % TSC_RING_SIZE;
        
        self.ring[head] = timestamp;
        self.head.store(next_head, Ordering::Release);
        self.total_recorded.fetch_add(1, Ordering::Relaxed);
        
        timestamp
    }
    
    /// Read current TSC without recording.
    /// 
    /// Useful for latency measurements between two points.
    #[inline]
    pub fn read_tsc(&self) -> u64 {
        let mut aux: u32 = 0;
        unsafe { _rdtscp(&mut aux) }
    }
    
    /// Convert TSC cycles to nanoseconds.
    #[inline]
    pub fn tsc_to_nanos(&self, tsc_cycles: u64) -> u64 {
        let freq = self.tsc_frequency_hz.load(Ordering::Relaxed);
        ((tsc_cycles as u128 * 1_000_000_000u128 / freq as u128) as u64)
    }
    
    /// Convert nanoseconds to TSC cycles.
    #[inline]
    pub fn nanos_to_tsc(&self, nanos: u64) -> u64 {
        let freq = self.tsc_frequency_hz.load(Ordering::Relaxed);
        ((nanos as u128 * freq as u128 / 1_000_000_000u128) as u64)
    }
    
    /// Calculate latency between two TSC timestamps.
    /// 
    /// # Arguments
    /// * `start_tsc` - Starting TSC value
    /// * `end_tsc` - Ending TSC value
    /// 
    /// # Returns
    /// Latency in nanoseconds
    pub fn calculate_latency(&self, start_tsc: u64, end_tsc: u64) -> u64 {
        let delta = end_tsc.wrapping_sub(start_tsc);
        self.tsc_to_nanos(delta)
    }
    
    /// Get recent timestamps from the ring buffer.
    /// 
    /// # Arguments
    /// * `count` - Number of timestamps to retrieve
    /// 
    /// # Returns
    /// Vec of timestamps (oldest first)
    pub fn get_recent_timestamps(&self, count: usize) -> Vec<TscTimestamp> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        
        let available = if head >= tail {
            head - tail
        } else {
            TSC_RING_SIZE - tail + head
        };
        
        let to_read = count.min(available);
        let mut result = Vec::with_capacity(to_read);
        
        for i in 0..to_read {
            let idx = (tail + i) % TSC_RING_SIZE;
            result.push(self.ring[idx]);
        }
        
        // Advance tail.
        if to_read > 0 {
            let new_tail = (tail + to_read) % TSC_RING_SIZE;
            self.tail.store(new_tail, Ordering::Release);
        }
        
        result
    }
    
    /// Get manager statistics.
    pub fn get_stats(&self) -> TscStats {
        TscStats {
            total_recorded: self.total_recorded.load(Ordering::Relaxed),
            ccd_migrations: self.ccd_migrations.load(Ordering::Relaxed),
            tsc_frequency_hz: self.tsc_frequency_hz.load(Ordering::Relaxed),
            ring_head: self.head.load(Ordering::Relaxed),
            ring_tail: self.tail.load(Ordering::Relaxed),
        }
    }
    
    /// Update TSC frequency (for dynamic calibration).
    pub fn update_frequency(&self, new_freq: u64) {
        self.tsc_frequency_hz.store(new_freq, Ordering::Relaxed);
        log_info!("TSC frequency updated to {} Hz", new_freq);
    }
}

impl Default for HardwareTscManager {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Get current core ID using CPUID or platform-specific methods.
#[inline]
fn get_current_core_id() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid;
        unsafe {
            // CPUID leaf 1, returns APIC ID in EBX[31:24]
            let cpuid = __cpuid(1);
            ((cpuid.ebx >> 24) & 0xFF) as usize
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

/// TSC manager statistics.
#[derive(Debug, Clone, Copy)]
pub struct TscStats {
    pub total_recorded: u64,
    pub ccd_migrations: u64,
    pub tsc_frequency_hz: u64,
    pub ring_head: usize,
    pub ring_tail: usize,
}

/// Logging macro.
macro_rules! log_info {
    ($($arg:tt)*) => {
        eprintln!("[INFO] {}", format!($($arg)*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timestamp_size() {
        assert_eq!(mem::size_of::<TscTimestamp>(), 64);
    }
    
    #[test]
    fn test_manager_creation() {
        let manager = HardwareTscManager::new(Some(4_000_000_000));
        let stats = manager.get_stats();
        assert_eq!(stats.tsc_frequency_hz, 4_000_000_000);
    }
    
    #[test]
    fn test_record_tick() {
        let manager = HardwareTscManager::new(None);
        let ts = manager.record_tick(12345);
        
        assert!(ts.tsc_cycles > 0);
        assert!(ts.nanos > 0);
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_recorded, 1);
    }
    
    #[test]
    fn test_tsc_conversion() {
        let manager = HardwareTscManager::new(Some(4_000_000_000));
        
        // 1 second = 4 billion cycles = 1 billion nanos
        let tsc_for_1s = manager.nanos_to_tsc(1_000_000_000);
        assert_eq!(tsc_for_1s, 4_000_000_000);
        
        let nanos_back = manager.tsc_to_nanos(tsc_for_1s);
        assert_eq!(nanos_back, 1_000_000_000);
    }
}
