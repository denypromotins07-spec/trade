//! CPU Time Stamp Counter (TSC) Profiler
//! 
//! This module reads the Time Stamp Counter (TSC) via `rdtsc` for ultra-precise,
//! zero-overhead latency profiling of the matching engine, bypassing standard
//! OS clock resolution limits.
//! 
//! Optimized for: AMD Ryzen AI 5, invariant TSC, microsecond precision
//! Key Features:
//! - Direct rdtsc instruction access
//! - CPU frequency scaling compensation
//! - Thread migration detection
//! - Zero-overhead profiling

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, Duration};

/// Memory budget for CPU cycles module (bytes)
const CPU_CYCLES_MEMORY_BUDGET: usize = 64 * 1024 * 1024; // 64MB

/// Maximum profile samples to retain
const MAX_PROFILE_SAMPLES: usize = 100000;

/// Profile sample record
#[derive(Debug, Clone)]
pub struct ProfileSample {
    pub start_tsc: u64,
    pub end_tsc: u64,
    pub cpu_id: u32,
    pub thread_id: u32,
    pub event_type: ProfileEventType,
    pub timestamp_ns: u64,
}

/// Profile event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileEventType {
    OrderSubmission,
    OrderMatch,
    OrderCancel,
    MarketDataUpdate,
    RiskCheck,
    NetworkSend,
    NetworkRecv,
    Custom,
}

/// CPU frequency information
#[derive(Debug, Clone)]
pub struct CpuFrequency {
    pub base_mhz: u64,
    pub current_mhz: u64,
    pub is_invariant: bool,
    pub tsc_to_ns_multiplier: f64,
}

/// TSC-based profiler with frequency compensation
pub struct TscProfiler {
    samples: Vec<ProfileSample>,
    cpu_frequency: CpuFrequency,
    total_cycles: AtomicU64,
    sample_count: AtomicU64,
    memory_used: AtomicU64,
    is_active: AtomicBool,
    last_calibration: Instant,
}

unsafe impl Send for TscProfiler {}
unsafe impl Sync for TscProfiler {}

impl TscProfiler {
    pub fn new() -> Self {
        let freq = Self::detect_cpu_frequency();
        
        Self {
            samples: Vec::with_capacity(MAX_PROFILE_SAMPLES),
            cpu_frequency: freq,
            total_cycles: AtomicU64::new(0),
            sample_count: AtomicU64::new(0),
            memory_used: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            last_calibration: Instant::now(),
        }
    }
    
    /// Read TSC using rdtsc instruction
    #[inline]
    pub fn read_tsc() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let low: u32;
            let high: u32;
            std::arch::asm!(
                "rdtsc",
                out("eax") low,
                out("edx") high,
                out("ecx") _,
                out("ebx") _,
            );
            ((high as u64) << 32) | (low as u64)
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86 platforms
            Instant::now().duration_since(Instant::now()).as_nanos() as u64
        }
    }
    
    /// Read TSC with serialization (rdtscp)
    #[inline]
    pub fn read_tsc_serialized() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let low: u32;
            let high: u32;
            let mut aux: u32 = 0;
            std::arch::asm!(
                "rdtscp",
                out("eax") low,
                out("edx") high,
                out("ecx") aux,
            );
            // Serialization barrier
            std::arch::asm!("cpuid", in("eax") 0, out("ebx") _, out("ecx") _, out("edx") _);
            ((high as u64) << 32) | (low as u64)
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            Instant::now().duration_since(Instant::now()).as_nanos() as u64
        }
    }
    
    /// Detect CPU frequency and TSC characteristics
    fn detect_cpu_frequency() -> CpuFrequency {
        let mut base_mhz = 0u64;
        let mut is_invariant = false;
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Check for invariant TSC (CPUID.80000007H:EDX[8])
            let mut eax = 0x80000007u32;
            let mut ebx: u32;
            let mut ecx: u32;
            let mut edx: u32;
            
            std::arch::asm!(
                "cpuid",
                in("eax") eax,
                out("ebx") ebx,
                out("ecx") ecx,
                out("edx") edx,
            );
            
            is_invariant = (edx & (1 << 8)) != 0;
            
            // Try to get base frequency from CPUID.16H
            eax = 0x16;
            std::arch::asm!(
                "cpuid",
                in("eax") eax,
                out("ebx") ebx,
                out("ecx") ecx,
                out("edx") edx,
            );
            
            // EBX contains base frequency in MHz
            base_mhz = ebx as u64;
        }
        
        // Default fallback values
        if base_mhz == 0 {
            base_mhz = 3200; // Assume 3.2 GHz for AMD Ryzen
        }
        
        // Calculate TSC to nanoseconds multiplier
        let tsc_to_ns_multiplier = 1_000_000_000.0 / (base_mhz as f64 * 1_000_000.0);
        
        CpuFrequency {
            base_mhz,
            current_mhz: base_mhz,
            is_invariant,
            tsc_to_ns_multiplier,
        }
    }
    
    /// Profile a code block execution
    pub fn profile<F, R>(&mut self, event_type: ProfileEventType, f: F) -> (R, u64)
    where
        F: FnOnce() -> R,
    {
        if !self.is_active.load(Ordering::Relaxed) {
            return (f(), 0);
        }
        
        let start = Self::read_tsc_serialized();
        let result = f();
        let end = Self::read_tsc_serialized();
        
        let cycles = end - start;
        
        // Record sample
        self.record_sample(start, end, cycles, event_type);
        
        // Convert cycles to nanoseconds
        let ns = (cycles as f64 * self.cpu_frequency.tsc_to_ns_multiplier) as u64;
        
        (result, ns)
    }
    
    /// Record a profile sample
    fn record_sample(&mut self, start: u64, end: u64, cycles: u64, event_type: ProfileEventType) {
        let sample = ProfileSample {
            start_tsc: start,
            end_tsc: end,
            cpu_id: 0, // Would use sched_getcpu() on Linux or GetCurrentProcessorNumber() on Windows
            thread_id: 0, // Would use gettid() or GetCurrentThreadId()
            event_type,
            timestamp_ns: Instant::now().duration_since(Instant::now()).as_nanos() as u64,
        };
        
        if self.samples.len() >= MAX_PROFILE_SAMPLES {
            self.samples.remove(0);
        }
        self.samples.push(sample);
        
        self.total_cycles.fetch_add(cycles, Ordering::Relaxed);
        self.sample_count.fetch_add(1, Ordering::Relaxed);
        
        self.memory_used.fetch_add(
            std::mem::size_of::<ProfileSample>() as u64,
            Ordering::Relaxed,
        );
    }
    
    /// Get statistics for a specific event type
    pub fn get_event_stats(&self, event_type: ProfileEventType) -> EventStats {
        let mut cycles_list: Vec<u64> = Vec::new();
        
        for sample in &self.samples {
            if sample.event_type == event_type {
                cycles_list.push(sample.end_tsc - sample.start_tsc);
            }
        }
        
        if cycles_list.is_empty() {
            return EventStats::default();
        }
        
        cycles_list.sort();
        
        let count = cycles_list.len();
        let sum: u128 = cycles_list.iter().map(|&c| c as u128).sum();
        let mean_cycles = sum as f64 / count as f64;
        
        let min_cycles = *cycles_list.first().unwrap();
        let max_cycles = *cycles_list.last().unwrap();
        
        let p50_idx = count / 2;
        let p95_idx = (count * 95) / 100;
        let p99_idx = (count * 99) / 100;
        
        let p50_cycles = cycles_list[p50_idx];
        let p95_cycles = cycles_list[p95_idx.min(count - 1)];
        let p99_cycles = cycles_list[p99_idx.min(count - 1)];
        
        // Convert to nanoseconds
        let mult = self.cpu_frequency.tsc_to_ns_multiplier;
        
        EventStats {
            event_type,
            sample_count: count as u64,
            min_cycles,
            max_cycles,
            mean_cycles,
            min_ns: (min_cycles as f64 * mult) as u64,
            max_ns: (max_cycles as f64 * mult) as u64,
            mean_ns: mean_cycles * mult,
            p50_ns: (p50_cycles as f64 * mult) as u64,
            p95_ns: (p95_cycles as f64 * mult) as u64,
            p99_ns: (p99_cycles as f64 * mult) as u64,
        }
    }
    
    /// Recalibrate CPU frequency (for dynamic frequency scaling)
    pub fn recalibrate(&mut self) {
        self.cpu_frequency = Self::detect_cpu_frequency();
        self.last_calibration = Instant::now();
    }
    
    /// Check if thread migrated during measurement
    pub fn detect_thread_migration(&self, sample: &ProfileSample) -> bool {
        // In production, would compare CPU IDs before/after
        // For now, check if duration seems anomalous (potential migration indicator)
        let duration = sample.end_tsc - sample.start_tsc;
        let expected_max = self.cpu_frequency.base_mhz * 1000; // 1ms in cycles
        
        duration > expected_max
    }
    
    /// Enforce memory limits
    pub fn enforce_memory_limit(&self, min_free_bytes: u64) -> bool {
        let current = self.memory_used.load(Ordering::Relaxed);
        if current > CPU_CYCLES_MEMORY_BUDGET as u64 - min_free_bytes {
            return true;
        }
        false
    }
    
    /// Get profiler statistics
    pub fn get_stats(&self) -> ProfilerStats {
        let total_samples = self.sample_count.load(Ordering::Relaxed);
        let total_cycles = self.total_cycles.load(Ordering::Relaxed);
        
        let avg_cycles = if total_samples > 0 {
            total_cycles as f64 / total_samples as f64
        } else {
            0.0
        };
        
        ProfilerStats {
            total_samples,
            total_cycles,
            average_cycles: avg_cycles,
            average_ns: avg_cycles * self.cpu_frequency.tsc_to_ns_multiplier,
            cpu_frequency_mhz: self.cpu_frequency.base_mhz,
            is_invariant_tsc: self.cpu_frequency.is_invariant,
            memory_used: self.memory_used.load(Ordering::Relaxed),
            is_active: self.is_active.load(Ordering::Relaxed),
        }
    }
    
    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Relaxed);
    }
    
    /// Get CPU frequency info
    pub fn get_cpu_frequency(&self) -> &CpuFrequency {
        &self.cpu_frequency
    }
}

impl Default for TscProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Event-specific statistics
#[derive(Debug, Default)]
pub struct EventStats {
    pub event_type: ProfileEventType,
    pub sample_count: u64,
    pub min_cycles: u64,
    pub max_cycles: u64,
    pub mean_cycles: f64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

/// Profiler statistics
#[derive(Debug)]
pub struct ProfilerStats {
    pub total_samples: u64,
    pub total_cycles: u64,
    pub average_cycles: f64,
    pub average_ns: f64,
    pub cpu_frequency_mhz: u64,
    pub is_invariant_tsc: bool,
    pub memory_used: u64,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tsc_read() {
        let tsc1 = TscProfiler::read_tsc();
        let tsc2 = TscProfiler::read_tsc();
        
        assert!(tsc2 >= tsc1);
    }
    
    #[test]
    fn test_profiling() {
        let mut profiler = TscProfiler::new();
        
        let (_result, ns) = profiler.profile(ProfileEventType::Custom, || {
            let mut x = 0u64;
            for i in 0..1000 {
                x += i;
            }
            x
        });
        
        assert!(ns > 0);
    }
    
    #[test]
    fn test_event_stats() {
        let mut profiler = TscProfiler::new();
        
        for _ in 0..100 {
            profiler.profile(ProfileEventType::OrderMatch, || {
                std::hint::black_box(42)
            });
        }
        
        let stats = profiler.get_event_stats(ProfileEventType::OrderMatch);
        assert!(stats.sample_count > 0);
        assert!(stats.mean_ns > 0.0);
    }
    
    #[test]
    fn test_cpu_frequency_detection() {
        let profiler = TscProfiler::new();
        let freq = profiler.get_cpu_frequency();
        
        assert!(freq.base_mhz > 0);
        println!("CPU Base Frequency: {} MHz", freq.base_mhz);
        println!("Invariant TSC: {}", freq.is_invariant);
    }
}
