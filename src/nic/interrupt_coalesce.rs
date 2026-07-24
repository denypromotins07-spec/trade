//! src/nic/interrupt_coalesce.rs
//!
//! Stage 51: Dynamic NIC Interrupt Coalescing Tuning
//!
//! Tunes NIC interrupt coalescing settings dynamically, balancing microsecond
//! tick latency against CPU interrupt storm prevention during high-volatility spikes.
//! Optimized for AMD Ryzen AI 5 architecture with Windows networking stack.
//!
//! Critical for adapting to market conditions while maintaining low latency.

use std::io;
use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Default interrupt moderation settings
const DEFAULT_USECS: u32 = 0; // No moderation for lowest latency
const DEFAULT_FRAMES: u32 = 1; // Process every frame

/// High volatility settings (more aggressive)
const HIGH_VOLATILITY_USECS: u32 = 0;
const HIGH_VOLATILITY_FRAMES: u32 = 1;

/// Normal volatility settings (balanced)
const NORMAL_VOLATILITY_USECS: u32 = 10;
const NORMAL_VOLATILITY_FRAMES: u32 = 4;

/// Low activity settings (power saving)
const LOW_ACTIVITY_USECS: u32 = 50;
const LOW_ACTIVITY_FRAMES: u32 = 16;

/// Threshold for high volatility detection (ticks per second)
const HIGH_VOLATILITY_THRESHOLD: u32 = 10000;

/// Threshold for low activity detection (ticks per second)
const LOW_ACTIVITY_THRESHOLD: u32 = 100;

/// Current coalescing profile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoalesceProfile {
    /// Ultra-low latency for active trading
    LowLatency,
    
    /// Balanced for normal market conditions
    Balanced,
    
    /// Power-saving for low activity periods
    PowerSave,
}

impl CoalesceProfile {
    pub fn usecs(&self) -> u32 {
        match self {
            Self::LowLatency => HIGH_VOLATILITY_USECS,
            Self::Balanced => NORMAL_VOLATILITY_USECS,
            Self::PowerSave => LOW_ACTIVITY_USECS,
        }
    }

    pub fn frames(&self) -> u32 {
        match self {
            Self::LowLatency => HIGH_VOLATILITY_FRAMES,
            Self::Balanced => NORMAL_VOLATILITY_FRAMES,
            Self::PowerSave => LOW_ACTIVITY_FRAMES,
        }
    }
}

/// Dynamic interrupt coalescing manager
pub struct InterruptCoalescer {
    /// Current profile
    current_profile: AtomicU32,
    
    /// Whether adaptive tuning is enabled
    adaptive_enabled: AtomicBool,
    
    /// Last profile change time
    last_change: std::sync::Mutex<Option<Instant>>,
    
    /// Tick rate estimator (exponential moving average)
    tick_rate_ema: std::sync::Mutex<f64>,
    
    /// Minimum interval between profile changes
    min_change_interval: Duration,
}

unsafe impl Send for InterruptCoalescer {}
unsafe impl Sync for InterruptCoalescer {}

impl InterruptCoalescer {
    /// Create a new interrupt coalescer
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            current_profile: AtomicU32::new(CoalesceProfile::LowLatency as u32),
            adaptive_enabled: AtomicBool::new(true),
            last_change: std::sync::Mutex::new(None),
            tick_rate_ema: std::sync::Mutex::new(0.0),
            min_change_interval: Duration::from_millis(100),
        })
    }

    /// Set the coalescing profile directly
    pub fn set_profile(&self, profile: CoalesceProfile) -> io::Result<()> {
        let now = Instant::now();
        
        // Check minimum interval
        {
            let last = self.last_change.lock().unwrap();
            if let Some(last_time) = *last {
                if now.duration_since(last_time) < self.min_change_interval {
                    return Ok(()); // Too soon, skip change
                }
            }
        }

        self.current_profile.store(profile as u32, Ordering::Release);
        
        *self.last_change.lock().unwrap() = Some(now);

        log_info!("Interrupt coalescing profile changed to {:?}", profile);
        log_info!("  Usecs: {}, Frames: {}", profile.usecs(), profile.frames());

        #[cfg(target_os = "windows")]
        {
            // Would execute PowerShell/netsh in production
            // Example: Set-NetAdapterAdvancedProperty -Name "Ethernet" -DisplayName "Interrupt Moderation" -DisplayValue "Enabled"
        }

        Ok(())
    }

    /// Get current profile
    pub fn get_profile(&self) -> CoalesceProfile {
        match self.current_profile.load(Ordering::Acquire) {
            0 => CoalesceProfile::LowLatency,
            1 => CoalesceProfile::Balanced,
            2 => CoalesceProfile::PowerSave,
            _ => CoalesceProfile::LowLatency,
        }
    }

    /// Update based on observed tick rate
    ///
    /// Called periodically to adapt to market conditions.
    pub fn update_tick_rate(&self, ticks_per_second: f64) -> io::Result<()> {
        if !self.adaptive_enabled.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Update EMA
        {
            let mut ema = self.tick_rate_ema.lock().unwrap();
            let alpha = 0.1; // Smoothing factor
            *ema = *ema * (1.0 - alpha) + ticks_per_second * alpha;
        }

        // Determine appropriate profile
        let new_profile = if ticks_per_second >= HIGH_VOLATILITY_THRESHOLD as f64 {
            CoalesceProfile::LowLatency
        } else if ticks_per_second <= LOW_ACTIVITY_THRESHOLD as f64 {
            CoalesceProfile::PowerSave
        } else {
            CoalesceProfile::Balanced
        };

        let current = self.get_profile();
        if new_profile != current {
            self.set_profile(new_profile)?;
        }

        Ok(())
    }

    /// Enable/disable adaptive tuning
    pub fn set_adaptive(&self, enabled: bool) {
        self.adaptive_enabled.store(enabled, Ordering::Relaxed);
        
        if enabled {
            log_info!("Adaptive interrupt coalescing enabled");
        } else {
            log_info!("Adaptive interrupt coalescing disabled");
        }
    }

    /// Force low-latency mode (for /START sequence)
    pub fn force_low_latency(&self) -> io::Result<()> {
        self.set_adaptive(false);
        self.set_profile(CoalesceProfile::LowLatency)
    }

    /// Restore adaptive mode (for normal operation)
    pub fn restore_adaptive(&self) {
        self.set_adaptive(true);
    }

    /// Get current tick rate estimate
    pub fn get_tick_rate_estimate(&self) -> f64 {
        *self.tick_rate_ema.lock().unwrap()
    }
}

impl Default for InterruptCoalescer {
    fn default() -> Self {
        Self::new().expect("Failed to create InterruptCoalescer")
    }
}

/// Logging macro
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[Interrupt Coalesce] {}", format!($($arg)*));
    };
}

/// Query current NIC interrupt moderation settings
pub fn query_current_settings() -> io::Result<(u32, u32)> {
    // In production, would query actual NIC settings
    // On Windows: Get-NetAdapterAdvancedProperty
    // On Linux: ethtool -c <interface>
    
    Ok((DEFAULT_USECS, DEFAULT_FRAMES))
}

/// Apply interrupt coalescing settings
pub fn apply_settings(usecs: u32, frames: u32) -> io::Result<()> {
    log_info!("Applying interrupt coalescing: {} usecs, {} frames", usecs, frames);
    
    #[cfg(target_os = "windows")]
    {
        // Would execute: 
        // Set-NetAdapterAdvancedProperty -Name "Ethernet" -DisplayName "Interrupt Moderation Rate" -DisplayValue "<value>"
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_values() {
        assert_eq!(CoalesceProfile::LowLatency.usecs(), 0);
        assert_eq!(CoalesceProfile::LowLatency.frames(), 1);
        
        assert_eq!(CoalesceProfile::Balanced.usecs(), 10);
        assert_eq!(CoalesceProfile::Balanced.frames(), 4);
        
        assert_eq!(CoalesceProfile::PowerSave.usecs(), 50);
        assert_eq!(CoalesceProfile::PowerSave.frames(), 16);
    }

    #[test]
    fn test_coalescer_creation() {
        let coalescer = InterruptCoalescer::new().unwrap();
        assert_eq!(coalescer.get_profile(), CoalesceProfile::LowLatency);
        assert!(coalescer.adaptive_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_profile_changes() {
        let coalescer = InterruptCoalescer::new().unwrap();
        
        coalescer.set_profile(CoalesceProfile::Balanced).unwrap();
        assert_eq!(coalescer.get_profile(), CoalesceProfile::Balanced);
        
        coalescer.set_profile(CoalesceProfile::PowerSave).unwrap();
        assert_eq!(coalescer.get_profile(), CoalesceProfile::PowerSave);
    }

    #[test]
    fn test_tick_rate_update() {
        let coalescer = InterruptCoalescer::new().unwrap();
        
        // High tick rate should trigger low latency
        coalescer.update_tick_rate(15000.0).unwrap();
        assert_eq!(coalescer.get_profile(), CoalesceProfile::LowLatency);
        
        // Low tick rate should trigger power save
        coalescer.update_tick_rate(50.0).unwrap();
        assert_eq!(coalescer.get_profile(), CoalesceProfile::PowerSave);
    }

    #[test]
    fn test_force_low_latency() {
        let coalescer = InterruptCoalescer::new().unwrap();
        
        coalescer.force_low_latency().unwrap();
        assert_eq!(coalescer.get_profile(), CoalesceProfile::LowLatency);
        assert!(!coalescer.adaptive_enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_query_settings() {
        let (usecs, frames) = query_current_settings().unwrap();
        assert_eq!(usecs, DEFAULT_USECS);
        assert_eq!(frames, DEFAULT_FRAMES);
    }
}
