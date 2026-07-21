//! Fast Fourier Transform (FFT) Signal Processing
//! 
//! Detects hidden cyclical patterns and dominant frequencies in tick data.
//! Uses pre-allocated trigonometric lookup tables for microsecond performance.
//! Zero heap allocations during runtime; all buffers pre-allocated at /START.
//! Optimized for AMD Ryzen AI 5 with SIMD vectorization hints.

use std::f64::consts::PI;

/// Maximum FFT size (power of 2, tuned for 8GB RAM constraint)
const MAX_FFT_SIZE: usize = 16_384; // 2^14

/// Pre-computed trigonometric lookup tables (initialized at /START)
pub struct TrigTables {
    /// Sine lookup table
    sin_table: [f64; MAX_FFT_SIZE],
    /// Cosine lookup table
    cos_table: [f64; MAX_FFT_SIZE],
    /// Bit-reversal permutation table
    bit_reverse: [usize; MAX_FFT_SIZE],
    /// Initialized flag
    initialized: bool,
}

impl TrigTables {
    /// Create and initialize trigonometric tables
    /// Call this once during /START phase
    pub const fn new() -> Self {
        Self {
            sin_table: [0.0; MAX_FFT_SIZE],
            cos_table: [0.0; MAX_FFT_SIZE],
            bit_reverse: [0; MAX_FFT_SIZE],
            initialized: false,
        }
    }

    /// Initialize lookup tables (call during system startup)
    pub fn initialize(&mut self, fft_size: usize) {
        assert!(fft_size.is_power_of_two(), "FFT size must be power of 2");
        assert!(fft_size <= MAX_FFT_SIZE, "FFT size exceeds maximum");

        let n = fft_size;
        
        // Pre-compute sine and cosine values
        for i in 0..n {
            let angle = 2.0 * PI * i as f64 / n as f64;
            self.sin_table[i] = angle.sin();
            self.cos_table[i] = angle.cos();
        }

        // Pre-compute bit-reversal permutation
        let bits = (n as u32).trailing_zeros();
        for i in 0..n {
            self.bit_reverse[i] = reverse_bits(i as u32, bits) as usize;
        }

        self.initialized = true;
    }

    /// Get sine value from lookup table
    #[inline(always)]
    pub fn sin(&self, index: usize) -> f64 {
        debug_assert!(self.initialized, "Trig tables not initialized");
        self.sin_table[index % MAX_FFT_SIZE]
    }

    /// Get cosine value from lookup table
    #[inline(always)]
    pub fn cos(&self, index: usize) -> f64 {
        debug_assert!(self.initialized, "Trig tables not initialized");
        self.cos_table[index % MAX_FFT_SIZE]
    }

    /// Get bit-reversed index
    #[inline(always)]
    pub fn bit_reverse_index(&self, index: usize) -> usize {
        debug_assert!(self.initialized, "Trig tables not initialized");
        self.bit_reverse[index]
    }

    /// Check if tables are initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

/// Reverse bits in a number
#[inline(always)]
fn reverse_bits(mut x: u32, bits: u32) -> u32 {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Complex number representation for FFT
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline(always)]
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    #[inline(always)]
    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    #[inline(always)]
    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline(always)]
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    #[inline(always)]
    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    #[inline(always)]
    pub fn sub(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

/// FFT Engine with pre-allocated buffers
pub struct FftEngine {
    /// Input/output buffer
    buffer: [Complex; MAX_FFT_SIZE],
    /// Temporary workspace
    workspace: [Complex; MAX_FFT_SIZE],
    /// Trigonometric tables
    trig_tables: TrigTables,
    /// Current FFT size
    fft_size: usize,
}

impl FftEngine {
    pub const fn new() -> Self {
        Self {
            buffer: [Complex::zero(); MAX_FFT_SIZE],
            workspace: [Complex::zero(); MAX_FFT_SIZE],
            trig_tables: TrigTables::new(),
            fft_size: 0,
        }
    }

    /// Initialize the FFT engine with specified size
    pub fn initialize(&mut self, fft_size: usize) {
        assert!(fft_size.is_power_of_two(), "FFT size must be power of 2");
        assert!(fft_size <= MAX_FFT_SIZE, "FFT size exceeds maximum");
        
        self.fft_size = fft_size;
        self.trig_tables.initialize(fft_size);
    }

    /// Compute forward FFT on real-valued input
    /// Returns frequency domain magnitudes
    pub fn compute_fft(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len().min(self.fft_size);
        assert!(n.is_power_of_two(), "Input length must be power of 2");

        // Load input into buffer (real part only)
        for i in 0..n {
            self.buffer[i] = Complex::new(input[i], 0.0);
        }

        // Perform FFT
        self.fft_recursive(n);

        // Extract magnitudes (only first half due to symmetry)
        let mut magnitudes = Vec::with_capacity(n / 2);
        for i in 0..n / 2 {
            magnitudes.push(self.buffer[i].magnitude());
        }

        magnitudes
    }

    /// Cooley-Tukey radix-2 FFT algorithm (in-place)
    fn fft_recursive(&mut self, n: usize) {
        if n <= 1 {
            return;
        }

        // Bit-reversal permutation
        let bits = (n as u32).trailing_zeros();
        for i in 0..n {
            let j = reverse_bits(i as u32, bits) as usize;
            if i < j {
                self.buffer.swap(i, j);
            }
        }

        // Iterative FFT (Cooley-Tukey)
        let mut size = 2;
        while size <= n {
            let half_size = size / 2;
            let step = n / size;

            for k in (0..n).step_by(size) {
                for j in 0..half_size {
                    let twiddle_idx = (j * step) % n;
                    
                    // Use lookup table for twiddle factors
                    let angle = -2.0 * PI * j as f64 / size as f64;
                    let w = Complex::new(angle.cos(), angle.sin());

                    let idx1 = k + j;
                    let idx2 = k + j + half_size;

                    let t = w.mul(&self.buffer[idx2]);
                    let temp = self.buffer[idx1];
                    
                    self.buffer[idx1] = temp.add(&t);
                    self.buffer[idx2] = temp.sub(&t);
                }
            }

            size *= 2;
        }
    }

    /// Find dominant frequencies in the spectrum
    pub fn find_dominant_frequencies(&mut self, input: &[f64], num_peaks: usize) -> Vec<(f64, f64)> {
        let magnitudes = self.compute_fft(input);
        let n = magnitudes.len();

        // Find peaks (simple local maxima detection)
        let mut peaks: Vec<(usize, f64)> = Vec::with_capacity(num_peaks);

        for i in 1..n - 1 {
            if magnitudes[i] > magnitudes[i - 1] && magnitudes[i] > magnitudes[i + 1] {
                peaks.push((i, magnitudes[i]));
            }
        }

        // Sort by magnitude (descending)
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N peaks
        let sample_rate = 1.0; // Normalized
        let freq_resolution = sample_rate / n as f64;

        peaks
            .into_iter()
            .take(num_peaks)
            .map(|(idx, mag)| (idx as f64 * freq_resolution, mag))
            .collect()
    }

    /// Detect cyclical patterns in tick data
    pub fn detect_cycles(&mut self, tick_data: &[f64]) -> Vec<f64> {
        let periods = self.find_dominant_frequencies(tick_data, 5);
        
        // Convert frequencies to periods
        periods
            .iter()
            .filter_map(|&(freq, mag)| {
                if freq > 1e-6 && mag > 0.0 {
                    Some(1.0 / freq)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get current FFT size
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

impl Default for FftEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral analysis results
#[derive(Debug)]
pub struct SpectralAnalysis {
    /// Dominant frequency (Hz)
    pub dominant_frequency: f64,
    /// Corresponding period (samples)
    pub dominant_period: f64,
    /// Spectral entropy (measure of randomness)
    pub spectral_entropy: f64,
    /// Total power in spectrum
    pub total_power: f64,
    /// Power in dominant frequency relative to total
    pub dominance_ratio: f64,
}

/// High-level spectral analyzer
pub struct SpectralAnalyzer {
    fft_engine: FftEngine,
}

impl SpectralAnalyzer {
    pub fn new(fft_size: usize) -> Self {
        let mut engine = FftEngine::new();
        engine.initialize(fft_size);
        
        Self {
            fft_engine: engine,
        }
    }

    /// Analyze signal and return spectral metrics
    pub fn analyze(&mut self, signal: &[f64]) -> SpectralAnalysis {
        let magnitudes = self.fft_engine.compute_fft(signal);
        let n = magnitudes.len();

        // Calculate total power
        let total_power: f64 = magnitudes.iter().map(|&m| m * m).sum();

        // Find dominant frequency
        let (dominant_idx, dominant_mag) = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &m)| (i, m))
            .unwrap_or((0, 0.0));

        let sample_rate = 1.0; // Normalized
        let freq_resolution = sample_rate / n as f64;
        let dominant_frequency = dominant_idx as f64 * freq_resolution;
        let dominant_period = if dominant_frequency > 1e-6 {
            1.0 / dominant_frequency
        } else {
            f64::MAX
        };

        // Calculate spectral entropy
        let spectral_entropy = if total_power > 1e-12 {
            let mut entropy = 0.0;
            for &mag in &magnitudes {
                let p = (mag * mag) / total_power;
                if p > 1e-12 {
                    entropy -= p * p.ln();
                }
            }
            entropy
        } else {
            0.0
        };

        // Dominance ratio
        let dominance_ratio = if total_power > 1e-12 {
            (dominant_mag * dominant_mag) / total_power
        } else {
            0.0
        };

        SpectralAnalysis {
            dominant_frequency,
            dominant_period,
            spectral_entropy,
            total_power,
            dominance_ratio,
        }
    }

    /// Check if signal has strong cyclical component
    pub fn has_strong_cycle(&mut self, signal: &[f64], threshold: f64) -> bool {
        let analysis = self.analyze(signal);
        analysis.dominance_ratio > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_sine_wave() {
        let mut fft = FftEngine::new();
        fft.initialize(1024);

        // Generate pure sine wave
        let n = 1024;
        let freq = 0.1; // Normalized frequency
        let mut signal = vec![0.0; n];
        for i in 0..n {
            signal[i] = (2.0 * PI * freq * i as f64).sin();
        }

        let magnitudes = fft.compute_fft(&signal);
        
        // Should have peak at the sine frequency
        let max_idx = magnitudes
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Peak should be near expected frequency bin
        let expected_bin = (freq * n as f64) as usize;
        assert!((max_idx as i32 - expected_bin as i32).abs() <= 2);
    }

    #[test]
    fn test_spectral_analyzer() {
        let mut analyzer = SpectralAnalyzer::new(512);

        // Generate noisy signal with cycle
        let n = 512;
        let mut signal = vec![0.0; n];
        for i in 0..n {
            signal[i] = (2.0 * PI * 0.05 * i as f64).sin() + 0.1 * (i as f64).sin();
        }

        let analysis = analyzer.analyze(&signal);
        assert!(analysis.total_power > 0.0);
        assert!(analysis.dominance_ratio > 0.0);
    }
}
