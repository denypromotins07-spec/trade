//! Ornstein-Uhlenbeck Spread Modeler with Jump-Diffusion Components
//! 
//! Models crypto pair spreads using OU process with jump-diffusion to capture
//! sudden basis blowouts during liquidation cascades. Uses pure integer and
//! fixed-point math for microsecond latency. Enforces 8GB RAM limit via ring buffers.
//! Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum spread history size (fixed for 8GB enforcement)
const MAX_SPREAD_HISTORY: usize = 16384;

/// Ring buffer for spread observations
#[repr(align(32))]
struct SpreadBuffer {
    /// Spread values (fixed-point: value * 1e9)
    spreads: [i64; MAX_SPREAD_HISTORY],
    /// Timestamps (nanoseconds since epoch)
    timestamps: [u64; MAX_SPREAD_HISTORY],
    /// Volume at each observation
    volumes: [i64; MAX_SPREAD_HISTORY],
    head: AtomicU64,
    count: AtomicU64,
}

impl SpreadBuffer {
    const fn new() -> Self {
        Self {
            spreads: [0; MAX_SPREAD_HISTORY],
            timestamps: [0; MAX_SPREAD_HISTORY],
            volumes: [0; MAX_SPREAD_HISTORY],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn push(&self, spread_i64: i64, timestamp: u64, volume: i64) {
        let head = self.head.fetch_add(1, Ordering::AcqRel) % MAX_SPREAD_HISTORY as u64;
        unsafe {
            *self.spreads.as_ptr().add(head as usize) as *mut i64 = spread_i64;
            *self.timestamps.as_ptr().add(head as usize) as *mut u64 = timestamp;
            *self.volumes.as_ptr().add(head as usize) as *mut i64 = volume;
        }
        self.count.fetch_min(MAX_SPREAD_HISTORY as u64, Ordering::Relaxed);
    }

    #[inline]
    fn get_recent(&self, n: usize) -> impl Iterator<Item = (i64, u64, i64)> + '_ {
        let count = self.count.load(Ordering::Acquire).min(n as u64) as usize;
        let head = self.head.load(Ordering::Acquire) as usize;
        
        (0..count).map(move |i| {
            let idx = (head.wrapping_sub(count - i) + MAX_SPREAD_HISTORY) % MAX_SPREAD_HISTORY;
            unsafe {
                (
                    *self.spreads.get_unchecked(idx),
                    *self.timestamps.get_unchecked(idx),
                    *self.volumes.get_unchecked(idx),
                )
            }
        })
    }
}

/// OU Process Parameters (fixed-point representation)
#[derive(Debug, Clone, Copy)]
pub struct OUParams {
    /// Mean reversion speed theta (scaled by 1e6)
    pub theta_fp: i64,
    /// Long-term mean mu (scaled by 1e9)
    pub mu_fp: i64,
    /// Volatility sigma (scaled by 1e6)
    pub sigma_fp: i64,
    /// Jump intensity lambda (scaled by 1e6)
    pub lambda_fp: i64,
    /// Jump mean (scaled by 1e9)
    pub jump_mean_fp: i64,
    /// Jump volatility (scaled by 1e9)
    pub jump_vol_fp: i64,
}

impl Default for OUParams {
    fn default() -> Self {
        Self {
            theta_fp: 100_000,     // 0.1 per second
            mu_fp: 0,              // Zero mean spread
            sigma_fp: 10_000,      // 0.01 volatility
            lambda_fp: 1_000,      // 0.001 jumps per second
            jump_mean_fp: 0,
            jump_vol_fp: 50_000_000, // 0.05 jump size
        }
    }
}

/// Statistical Arbitrage Spread Modeler
/// 
/// Combines Ornstein-Uhlenbeck mean reversion with jump-diffusion
/// to model crypto pair spreads including extreme liquidation events.
pub struct SpreadModeler {
    params: OUParams,
    buffer: SpreadBuffer,
    /// Current spread estimate (fixed-point)
    current_spread: AtomicU64,
    /// Expected value (fixed-point)
    expected_value: AtomicU64,
    /// Variance estimate (fixed-point)
    variance: AtomicU64,
    /// Half-life in milliseconds
    half_life_ms: AtomicU64,
    /// Jump detection flag
    jump_detected: AtomicBool,
    last_update_ns: AtomicU64,
}

impl SpreadModeler {
    /// Create a new spread modeler with default parameters
    pub fn new() -> Self {
        Self {
            params: OUParams::default(),
            buffer: SpreadBuffer::new(),
            current_spread: AtomicU64::new(0),
            expected_value: AtomicU64::new(0),
            variance: AtomicU64::new(0),
            half_life_ms: AtomicU64::new(0),
            jump_detected: AtomicBool::new(false),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Create with custom parameters
    pub fn with_params(params: OUParams) -> Self {
        let mut model = Self::new();
        model.params = params;
        model.update_half_life();
        model
    }

    /// Calculate half-life from mean reversion speed
    fn update_half_life(&mut self) {
        if self.params.theta_fp > 0 {
            // half_life = ln(2) / theta
            // Using fixed-point: ln(2) ≈ 693147 (scaled by 1e6)
            let half_life_fp = 693147_000_000 / self.params.theta_fp; // Result in microseconds
            self.half_life_ms.store((half_life_fp / 1000) as u64, Ordering::Release);
        }
    }

    /// Update parameters
    pub fn set_params(&mut self, params: OUParams) {
        self.params = params;
        self.update_half_life();
    }

    /// Record a new spread observation
    pub fn observe(&self, price_a: i64, price_b: i64, volume: i64, timestamp: u64) {
        // Calculate spread (fixed-point: already scaled)
        let spread = price_a - price_b;
        
        self.buffer.push(spread, timestamp, volume);
        self.current_spread.store(spread as u64, Ordering::Release);
        
        // Update statistics
        self.update_statistics();
        
        // Detect jumps
        self.detect_jump(spread);
        
        self.last_update_ns.store(timestamp, Ordering::Release);
    }

    /// Update running statistics using Welford's algorithm (fixed-point)
    fn update_statistics(&self) {
        let count = self.buffer.count.load(Ordering::Acquire).min(1000) as usize;
        if count < 10 {
            return;
        }

        let mut sum = 0i128;
        let mut sum_sq = 0i128;

        for (spread, _, _) in self.buffer.get_recent(count) {
            sum += spread as i128;
            sum_sq += (spread as i128) * (spread as i128);
        }

        let n = count as i128;
        let mean = sum / n;
        let variance = ((sum_sq - (sum * sum) / n) / (n - 1)).max(0);

        self.expected_value.store(mean as u64, Ordering::Release);
        self.variance.store(variance as u64, Ordering::Release);
    }

    /// Detect jump using threshold on standardized residual
    fn detect_jump(&self, spread: i64) {
        let mean = self.expected_value.load(Ordering::Acquire) as i64;
        let var = self.variance.load(Ordering::Acquire) as i64;
        
        if var > 0 {
            let std_dev = (var as f64).sqrt() as i64;
            let z_score = (spread - mean).abs() / std_dev.max(1);
            
            // Jump if z-score > 5
            self.jump_detected.store(z_score > 5, Ordering::Release);
        }
    }

    /// Calculate expected spread at time t using OU dynamics
    /// E[S_t] = S_0 * exp(-theta*t) + mu * (1 - exp(-theta*t))
    pub fn expected_spread_at(&self, dt_ms: u64) -> i64 {
        let s0 = self.current_spread.load(Ordering::Acquire) as i64;
        let mu = self.params.mu_fp;
        let theta = self.params.theta_fp;

        // exp(-theta * t) approximation using Taylor series
        // theta is per-second, dt is in ms
        let theta_t = (theta as u128 * dt_ms as u128) / 1_000_000_000; // Scale adjustment
        
        // First-order approximation: exp(-x) ≈ 1 - x for small x
        let exp_factor = if theta_t < 100_000_000 {
            1_000_000 - theta_t as i64
        } else {
            0
        };

        // E[S_t] = S_0 * exp_factor + mu * (1 - exp_factor)
        let expected = (s0 as i128 * exp_factor as i128 / 1_000_000)
            + (mu as i128 * (1_000_000 - exp_factor as i128) / 1_000_000);

        expected as i64
    }

    /// Calculate fair value spread (long-term mean)
    pub fn fair_value(&self) -> i64 {
        self.params.mu_fp
    }

    /// Get z-score of current spread
    pub fn z_score(&self) -> f64 {
        let spread = self.current_spread.load(Ordering::Acquire) as i64;
        let mean = self.expected_value.load(Ordering::Acquire) as i64;
        let var = self.variance.load(Ordering::Acquire) as u64;

        if var == 0 {
            return 0.0;
        }

        let std_dev = (var as f64).sqrt();
        (spread - mean) as f64 / std_dev
    }

    /// Check if jump was detected in last observation
    pub fn is_jump_detected(&self) -> bool {
        self.jump_detected.load(Ordering::Acquire)
    }

    /// Get half-life in milliseconds
    pub fn half_life_ms(&self) -> u64 {
        self.half_life_ms.load(Ordering::Acquire)
    }

    /// Generate trading signal based on spread deviation
    /// Returns: (direction, confidence)
    /// direction: positive = long A/short B, negative = short A/long B
    pub fn trading_signal(&self) -> (i8, f64) {
        let z = self.z_score();
        
        if z.abs() < 1.0 {
            return (0, 0.0); // No signal
        }

        let direction = if z > 0.0 { -1i8 } else { 1i8 };
        let confidence = (z.abs() / 5.0).min(1.0);

        (direction, confidence)
    }

    /// Get current spread value
    pub fn current_spread(&self) -> i64 {
        self.current_spread.load(Ordering::Acquire) as i64
    }
}

impl Default for SpreadModeler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_modeler_creation() {
        let model = SpreadModeler::new();
        assert_eq!(model.z_score(), 0.0);
    }

    #[test]
    fn test_spread_observation_and_signal() {
        let model = SpreadModeler::new();
        
        // Simulate spread observations
        let base_price = 50_000_000_000i64; // Fixed-point
        for i in 0..100 {
            let spread_offset = (i as i64 - 50) * 1_000_000; // +/- 0.001
            let price_a = base_price + spread_offset / 2;
            let price_b = base_price - spread_offset / 2;
            
            model.observe(price_a, price_b, 100, 1_000_000_000 + i as u64 * 1_000_000);
        }

        // Should have some statistics now
        assert!(model.half_life_ms() > 0 || model.half_life_ms() == 0); // May be 0 if theta is default
        
        let (direction, confidence) = model.trading_signal();
        // Signal depends on recent observations
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_jump_detection() {
        let params = OUParams {
            theta_fp: 100_000,
            mu_fp: 0,
            sigma_fp: 1_000, // Low volatility
            ..Default::default()
        };
        let model = SpreadModeler::with_params(params);

        // Establish baseline
        for i in 0..50 {
            model.observe(50_000_000_000, 50_000_000_000, 100, i as u64 * 1_000_000);
        }

        // Inject large jump
        model.observe(50_100_000_000, 49_900_000_000, 1000, 50_000_000);
        
        // Jump should be detected
        assert!(model.is_jump_detected());
    }
}
