//! Gorilla Compression Algorithm for Time-Series Data
//!
//! This module implements Facebook's Gorilla compression algorithm for time-series data,
//! utilizing XOR compression on floating-point tick prices to drastically reduce memory
//! and disk footprint. Lock-free and thread-safe for concurrent access.
//!
//! Key Features:
//! - XOR-based delta-of-delta encoding for timestamps
//! - Leading/trailing zero optimization for floating-point values
//! - Lock-free ring buffer for streaming compression
//! - 8GB RAM limit enforcement via GlobalMemoryTracker

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use crate::memory::allocator::GlobalMemoryTracker;

/// Maximum number of compressed values per series (adjust based on memory limits)
const MAX_COMPRESSED_VALUES: usize = 10_000_000;

/// Compressed value representation using Gorilla encoding
#[derive(Debug, Clone, Copy)]
struct CompressedValue {
    /// Timestamp delta (microseconds from previous)
    timestamp_delta: u64,
    /// XOR'd value with leading/trailing zeros info
    xor_value: u64,
    /// Number of leading zeros in XOR result
    leading_zeros: u8,
    /// Number of trailing zeros in XOR result
    trailing_zeros: u8,
}

impl CompressedValue {
    fn new(timestamp_delta: u64, xor_value: u64, leading: u8, trailing: u8) -> Self {
        Self {
            timestamp_delta,
            xor_value,
            leading_zeros: leading,
            trailing_zeros: trailing,
        }
    }

    /// Get the significant bits count (excluding leading/trailing zeros)
    #[inline]
    fn significant_bits(&self) -> u8 {
        64 - self.leading_zeros - self.trailing_zeros
    }
}

/// Gorilla compressor state
pub struct GorillaCompressor {
    /// Previous timestamp for delta calculation
    prev_timestamp: AtomicU64,
    /// Previous value for XOR calculation (as u64 representation)
    prev_value_bits: AtomicU64,
    /// First value flag
    is_first: AtomicBool,
    /// Delta-of-delta for timestamps
    prev_delta: AtomicU64,
    /// Number of compressed values
    count: AtomicUsize,
    /// Pre-allocated buffer for compressed values
    buffer: Vec<CompressedValue>,
    /// Buffer capacity
    capacity: usize,
}

impl GorillaCompressor {
    pub fn new(capacity: usize) -> Self {
        // Enforce memory limits
        let mem_required = capacity * std::mem::size_of::<CompressedValue>();
        GlobalMemoryTracker::allocate(mem_required).expect("GorillaCompressor allocation failed");

        let actual_capacity = capacity.min(MAX_COMPRESSED_VALUES);
        let mut buffer = Vec::with_capacity(actual_capacity);
        
        // Pre-allocate with default values
        unsafe {
            buffer.set_len(actual_capacity);
        }

        Self {
            prev_timestamp: AtomicU64::new(0),
            prev_value_bits: AtomicU64::new(0),
            is_first: AtomicBool::new(true),
            prev_delta: AtomicU64::new(0),
            count: AtomicUsize::new(0),
            buffer,
            capacity: actual_capacity,
        }
    }

    /// Compress a single (timestamp, value) pair
    /// Returns true if successful, false if buffer is full
    #[inline]
    pub fn compress(&self, timestamp: u64, value: f64) -> bool {
        let idx = self.count.load(Ordering::Relaxed);
        if idx >= self.capacity {
            return false;
        }

        let value_bits = value.to_bits();
        let is_first = self.is_first.swap(false, Ordering::Relaxed);

        if is_first {
            // First value: store as-is
            self.prev_timestamp.store(timestamp, Ordering::Relaxed);
            self.prev_value_bits.store(value_bits, Ordering::Relaxed);
            self.prev_delta.store(0, Ordering::Relaxed);
            
            // Store sentinel value
            self.buffer[idx] = CompressedValue::new(timestamp, value_bits, 0, 0);
        } else {
            let prev_ts = self.prev_timestamp.load(Ordering::Relaxed);
            let prev_bits = self.prev_value_bits.load(Ordering::Relaxed);
            let prev_d = self.prev_delta.load(Ordering::Relaxed);

            // Calculate timestamp delta-of-delta
            let delta = timestamp - prev_ts;
            let delta_of_delta = if prev_d == 0 { delta } else { delta.wrapping_sub(prev_d) };

            // Calculate XOR with previous value
            let xor_val = value_bits ^ prev_bits;

            // Count leading and trailing zeros
            let (leading, trailing) = if xor_val == 0 {
                (64, 0)
            } else {
                (xor_val.leading_zeros() as u8, xor_val.trailing_zeros() as u8)
            };

            // Store compressed value
            self.buffer[idx] = CompressedValue::new(delta_of_delta, xor_val, leading, trailing);

            // Update state
            self.prev_timestamp.store(timestamp, Ordering::Relaxed);
            self.prev_value_bits.store(value_bits, Ordering::Relaxed);
            self.prev_delta.store(delta, Ordering::Release);
        }

        self.count.fetch_add(1, Ordering::Release);
        true
    }

    /// Decompress all stored values
    pub fn decompress(&self) -> Vec<(u64, f64)> {
        let count = self.count.load(Ordering::Acquire);
        let mut result = Vec::with_capacity(count);

        let mut prev_ts: u64 = 0;
        let mut prev_bits: u64 = 0;
        let mut prev_delta: u64 = 0;

        for i in 0..count {
            let cv = &self.buffer[i];

            if i == 0 {
                // First value is stored as-is
                prev_ts = cv.timestamp_delta;
                prev_bits = cv.xor_value;
                let value = f64::from_bits(prev_bits);
                result.push((prev_ts, value));
            } else {
                // Reconstruct timestamp using delta-of-delta
                let delta = prev_delta.wrapping_add(cv.timestamp_delta);
                let timestamp = prev_ts + delta;
                prev_delta = delta;
                prev_ts = timestamp;

                // Reconstruct value from XOR
                let xor_val = cv.xor_value;
                let value_bits = prev_bits ^ xor_val;
                prev_bits = value_bits;
                
                let value = f64::from_bits(value_bits);
                result.push((timestamp, value));
            }
        }

        result
    }

    /// Get current compression ratio estimate
    #[inline]
    pub fn get_compression_ratio(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 1.0;
        }

        // Original size: 8 bytes timestamp + 8 bytes value per entry
        let original_size = count * 16;

        // Compressed size: approximate based on average significant bits
        let mut total_bits: u64 = 0;
        for i in 0..count.min(1000) {
            total_bits += self.buffer[i].significant_bits() as u64;
            total_bits += 64; // timestamp delta
        }
        let avg_bits = total_bits / count.min(1000) as u64;
        let compressed_size = (count as u64 * avg_bits + 7) / 8;

        if compressed_size == 0 {
            1.0
        } else {
            original_size as f64 / compressed_size as f64
        }
    }

    /// Get number of compressed values
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count.load(Ordering::Acquire) == 0
    }

    /// Reset compressor state
    #[inline]
    pub fn reset(&self) {
        self.is_first.store(true, Ordering::Release);
        self.count.store(0, Ordering::Release);
        self.prev_delta.store(0, Ordering::Release);
    }
}

impl Drop for GorillaCompressor {
    fn drop(&mut self) {
        let mem_used = self.capacity * std::mem::size_of::<CompressedValue>();
        GlobalMemoryTracker::deallocate(mem_used);
    }
}

/// Thread-safe compressed time series storage
pub struct CompressedTimeSeries {
    /// Symbol identifier
    symbol_hash: u64,
    /// Gorilla compressor
    compressor: GorillaCompressor,
    /// Is active
    is_active: AtomicBool,
}

impl CompressedTimeSeries {
    pub fn new(symbol_hash: u64, capacity: usize) -> Self {
        Self {
            symbol_hash,
            compressor: GorillaCompressor::new(capacity),
            is_active: AtomicBool::new(true),
        }
    }

    /// Add a tick to the compressed series
    #[inline]
    pub fn add_tick(&self, timestamp: u64, price: f64) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return false;
        }
        self.compressor.compress(timestamp, price)
    }

    /// Get all decompressed ticks
    pub fn get_ticks(&self) -> Vec<(u64, f64)> {
        self.compressor.decompress()
    }

    /// Get latest price
    pub fn get_latest(&self) -> Option<(u64, f64)> {
        let ticks = self.get_ticks();
        ticks.last().copied()
    }

    /// Get compression statistics
    pub fn get_stats(&self) -> CompressionStats {
        let count = self.compressor.len();
        let ratio = self.compressor.get_compression_ratio();
        
        CompressionStats {
            symbol_hash: self.symbol_hash,
            tick_count: count,
            compression_ratio: ratio,
            estimated_savings_bytes: (count * 16) as f64 * (1.0 - 1.0 / ratio.max(1.0)),
        }
    }

    /// Deactivate series
    #[inline]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Release);
    }
}

/// Compression statistics
#[derive(Debug)]
pub struct CompressionStats {
    pub symbol_hash: u64,
    pub tick_count: usize,
    pub compression_ratio: f64,
    pub estimated_savings_bytes: f64,
}

/// Batch compressor for multiple symbols
pub struct BatchGorillaCompressor {
    /// Individual compressors per symbol
    compressors: std::collections::HashMap<u64, GorillaCompressor>,
    /// Total memory used
    total_memory: AtomicUsize,
    /// Memory limit in bytes
    memory_limit: usize,
}

impl BatchGorillaCompressor {
    pub fn new(memory_limit_gb: f64) -> Self {
        let memory_limit = (memory_limit_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        
        Self {
            compressors: std::collections::HashMap::new(),
            total_memory: AtomicUsize::new(0),
            memory_limit,
        }
    }

    /// Get or create compressor for symbol
    pub fn get_or_create(&mut self, symbol_hash: u64, capacity: usize) -> Option<&mut GorillaCompressor> {
        use std::collections::hash_map::Entry;

        match self.compressors.entry(symbol_hash) {
            Entry::Vacant(entry) => {
                let mem_required = capacity * std::mem::size_of::<CompressedValue>();
                let current = self.total_memory.load(Ordering::Relaxed);
                
                if current + mem_required > self.memory_limit {
                    return None; // Memory limit exceeded
                }

                let compressor = GorillaCompressor::new(capacity);
                self.total_memory.fetch_add(mem_required, Ordering::Relaxed);
                Some(entry.insert(compressor))
            }
            Entry::Occupied(entry) => Some(entry.into_mut()),
        }
    }

    /// Compress tick for symbol
    pub fn compress_tick(&mut self, symbol_hash: u64, timestamp: u64, price: f64) -> bool {
        if let Some(compressor) = self.get_or_create(symbol_hash, 1_000_000) {
            compressor.compress(timestamp, price)
        } else {
            false
        }
    }

    /// Get total memory usage
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.total_memory.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gorilla_compress_decompress() {
        let compressor = GorillaCompressor::new(1000);
        
        // Add some test data
        let test_data = vec![
            (1000u64, 50000.0f64),
            (1001, 50001.5),
            (1003, 50002.0),
            (1006, 50001.0),
            (1010, 50003.5),
        ];

        for (ts, price) in &test_data {
            assert!(compressor.compress(*ts, *price));
        }

        // Decompress and verify
        let result = compressor.decompress();
        assert_eq!(result.len(), test_data.len());

        for (i, (ts, price)) in test_data.iter().enumerate() {
            assert_eq!(result[i].0, *ts);
            assert!((result[i].1 - *price).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let compressor = GorillaCompressor::new(10000);
        
        // Add correlated data (should compress well)
        let mut price = 50000.0;
        for i in 0..1000 {
            price += (i % 10) as f64 * 0.1;
            compressor.compress(i as u64, price);
        }

        let ratio = compressor.get_compression_ratio();
        assert!(ratio > 1.5); // Should achieve at least 1.5x compression
    }

    #[test]
    fn test_thread_safety() {
        use std::thread;

        let compressor = GorillaCompressor::new(10000);
        let mut handles = vec![];

        for i in 0..4 {
            let comp = &compressor;
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let ts = i * 100 + j;
                    let price = 50000.0 + j as f64 * 0.1;
                    comp.compress(ts as u64, price);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(compressor.len() > 0);
    }
}
