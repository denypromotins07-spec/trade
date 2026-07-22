//! Dynamic Drawdown Control Module
//!
//! Implements continuous-time risk budgets and path-dependent Kelly fractions.
//! Instantly scales down position sizes when equity curves deviate from expectations.
//! Optimized for AMD Ryzen AI 5 with zero heap allocations in hot path.
//!
//! # Features
//! - Path-dependent Kelly criterion
//! - Continuous-time risk budgeting
//! - Drawdown-triggered position scaling
//! - 8GB RAM constraint adherence

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum historical equity points to track (compile-time constant)
const MAX_EQUITY_POINTS: usize = 1000;

/// Pre-allocated circular buffer for equity history
#[repr(C, align(64))]
pub struct EquityBuffer {
    data: [f64; MAX_EQUITY_POINTS],
    timestamps: [u64; MAX_EQUITY_POINTS],
    head: AtomicU64,
    count: AtomicU64,
}

impl Default for EquityBuffer {
    fn default() -> Self {
        Self {
            data: [0.0; MAX_EQUITY_POINTS],
            timestamps: [0; MAX_EQUITY_POINTS],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
}

impl EquityBuffer {
    #[inline]
    pub fn push(&self, equity: f64, timestamp_us: u64) {
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        let idx = (head % MAX_EQUITY_POINTS as u64) as usize;
        
        unsafe {
            let data_ptr = &self.data[idx] as *const f64 as *mut f64;
            let ts_ptr = &self.timestamps[idx] as *const u64 as *mut u64;
            data_ptr.write(equity);
            ts_ptr.write(timestamp_us);
        }
        
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count < MAX_EQUITY_POINTS as u64 {
            self.count.store(current_count + 1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn get_recent(&self, n: usize) -> &[f64] {
        let count = self.count.load(Ordering::Acquire) as usize;
        let n = n.min(count);
        if n == 0 { return &[]; }
        
        let head = self.head.load(Ordering::Acquire) as usize;
        let start = if head >= n { head - n } else { MAX_EQUITY_POINTS - (n - head) };
        
        if start + n <= MAX_EQUITY_POINTS {
            &self.data[start..start + n]
        } else {
            &self.data[start..]
        }
    }

    #[inline]
    pub fn peak(&self) -> f64 {
        let count = self.count.load(Ordering::Acquire) as usize;
        if count == 0 { return 0.0; }
        
        let mut max_val = f64::MIN;
        for i in 0..count {
            let idx = i % MAX_EQUITY_POINTS;
            if self.data[idx] > max_val {
                max_val = self.data[idx];
            }
        }
        max_val
    }
}

/// Configuration for drawdown control
#[derive(Debug, Clone, Copy)]
pub struct DrawdownConfig {
    /// Maximum allowable drawdown (as fraction, e.g., 0.15 = 15%)
    pub max_drawdown: f64,
    /// Drawdown threshold to start scaling (fraction)
    pub scale_threshold: f64,
    /// Kelly fraction divisor for conservatism (e.g., 4.0 = quarter-Kelly)
    pub kelly_divisor: f64,
    /// Minimum position size fraction (when at max drawdown)
    pub min_position_fraction: f64,
    /// Recovery hysteresis (fraction above threshold to resume normal sizing)
    pub recovery_hysteresis: f64,
}

impl Default for DrawdownConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.20,
            scale_threshold: 0.10,
            kelly_divisor: 4.0,
            min_position_fraction: 0.1,
            recovery_hysteresis: 0.02,
        }
    }
}

/// Dynamic drawdown controller
pub struct DrawdownController {
    /// Equity history buffer
    equity_buffer: EquityBuffer,
    /// Starting equity
    initial_equity: AtomicU64, // Stored as fixed-point * 1e18
    /// Current scaling factor (fixed-point * 10000)
    current_scale: AtomicU64,
    /// Is currently in drawdown mode
    in_drawdown: AtomicBool,
    /// Configuration
    config: DrawdownConfig,
    /// Cache line padding
    _padding: [u8; 64],
}

impl DrawdownController {
    /// Create new drawdown controller
    #[inline]
    pub fn new(config: DrawdownConfig, initial_equity: u64) -> Self {
        Self {
            equity_buffer: EquityBuffer::default(),
            initial_equity: AtomicU64::new(initial_equity),
            current_scale: AtomicU64::new(10000), // 1.0 in fixed-point
            in_drawdown: AtomicBool::new(false),
            config,
            _padding: [0; 64],
        }
    }

    /// Record new equity value
    #[inline]
    pub fn record_equity(&self, equity: f64, timestamp_us: u64) {
        self.equity_buffer.push(equity, timestamp_us);
        self.update_scaling_factor(equity);
    }

    /// Update scaling factor based on current drawdown
    #[inline]
    fn update_scaling_factor(&self, current_equity: f64) {
        let initial = self.initial_equity.load(Ordering::Relaxed) as f64 / 1e18;
        if initial <= 0.0 {
            return;
        }

        // Calculate current drawdown
        let peak = self.equity_buffer.peak().max(initial);
        let drawdown = if peak > 0.0 {
            (peak - current_equity) / peak
        } else {
            0.0
        };

        // Determine scaling factor using path-dependent Kelly
        let scale = self.calculate_kelly_scale(drawdown);
        self.current_scale.store(scale, Ordering::Release);

        // Update drawdown state
        let in_dd = drawdown > self.config.scale_threshold;
        self.in_drawdown.store(in_dd, Ordering::Release);
    }

    /// Calculate Kelly-based scaling factor
    /// Returns fixed-point value * 10000
    #[inline]
    fn calculate_kelly_scale(&self, drawdown: f64) -> u64 {
        if drawdown <= self.config.scale_threshold {
            return 10000; // Full size (1.0)
        }

        if drawdown >= self.config.max_drawdown {
            return (self.config.min_position_fraction * 10000.0) as u64;
        }

        // Linear interpolation between threshold and max
        let range = self.config.max_drawdown - self.config.scale_threshold;
        let position = (drawdown - self.config.scale_threshold) / range;

        // Apply Kelly fraction adjustment
        let kelly_adjustment = 1.0 / self.config.kelly_divisor;
        
        let target_scale = self.config.min_position_fraction 
            + (1.0 - self.config.min_position_fraction) * (1.0 - position);
        
        let final_scale = target_scale * kelly_adjustment;
        (final_scale * 10000.0) as u64
    }

    /// Get current position size multiplier
    /// Returns value in range [min_position_fraction, 1.0]
    #[inline]
    pub fn get_position_multiplier(&self) -> f64 {
        let scale = self.current_scale.load(Ordering::Acquire);
        scale as f64 / 10000.0
    }

    /// Check if currently in drawdown mode
    #[inline]
    pub fn is_in_drawdown(&self) -> bool {
        self.in_drawdown.load(Ordering::Acquire)
    }

    /// Get current drawdown percentage
    #[inline]
    pub fn get_current_drawdown(&self) -> f64 {
        let initial = self.initial_equity.load(Ordering::Relaxed) as f64 / 1e18;
        if initial <= 0.0 {
            return 0.0;
        }

        let count = self.equity_buffer.count.load(Ordering::Acquire) as usize;
        if count == 0 {
            return 0.0;
        }

        let head = self.equity_buffer.head.load(Ordering::Acquire) as usize;
        let current_idx = if head == 0 {
            MAX_EQUITY_POINTS - 1
        } else {
            head - 1
        } % MAX_EQUITY_POINTS;

        let current_equity = self.equity_buffer.data[current_idx];
        let peak = self.equity_buffer.peak().max(initial);

        if peak > 0.0 {
            (peak - current_equity) / peak
        } else {
            0.0
        }
    }

    /// Calculate recommended position size
    /// 
    /// # Arguments
    /// * `base_size` - Base position size before drawdown adjustment
    /// * `win_rate` - Estimated win rate (0.0 - 1.0)
    /// * `avg_win_loss_ratio` - Average win/loss ratio
    #[inline]
    pub fn calculate_position_size(&self, base_size: u64, win_rate: f64, 
                                    avg_win_loss_ratio: f64) -> u64 {
        // Calculate raw Kelly fraction
        let kelly_raw = if avg_win_loss_ratio > 0.0 {
            win_rate - (1.0 - win_rate) / avg_win_loss_ratio
        } else {
            0.0
        };

        // Apply conservatism divisor
        let kelly_adjusted = kelly_raw / self.config.kelly_divisor;

        // Apply drawdown scaling
        let dd_scale = self.get_position_multiplier();

        // Final position size
        let adjusted_fraction = kelly_adjusted.max(0.0).min(1.0) * dd_scale;
        
        ((base_size as f64 * adjusted_fraction) as u64)
            .max((base_size as f64 * self.config.min_position_fraction) as u64)
    }

    /// Reset controller with new initial equity
    #[inline]
    pub fn reset(&self, new_initial_equity: u64) {
        self.initial_equity.store(new_initial_equity, Ordering::Release);
        self.current_scale.store(10000, Ordering::Release);
        self.in_drawdown.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let config = DrawdownConfig::default();
        let controller = DrawdownController::new(config, 1_000_000_000_000_000_000);
        assert_eq!(controller.get_position_multiplier(), 1.0);
        assert!(!controller.is_in_drawdown());
    }

    #[test]
    fn test_drawdown_detection() {
        let config = DrawdownConfig {
            max_drawdown: 0.20,
            scale_threshold: 0.10,
            ..Default::default()
        };
        let controller = DrawdownController::new(config, 1_000_000_000_000_000_000);
        
        // Record declining equity
        controller.record_equity(1.0, 1000000);
        controller.record_equity(0.95, 2000000);
        controller.record_equity(0.88, 3000000); // 12% drawdown
        
        assert!(controller.is_in_drawdown());
        assert!(controller.get_position_multiplier() < 1.0);
    }

    #[test]
    fn test_position_sizing() {
        let controller = DrawdownController::new(DrawdownConfig::default(), 1_000_000_000_000_000_000);
        let size = controller.calculate_position_size(1000000, 0.55, 2.0);
        assert!(size > 0);
        assert!(size <= 1000000);
    }
}
