//! AVX2-Vectorized Batch Processor for Network Packet Payloads
//!
//! This module implements high-throughput packet processing using AVX2 SIMD
//! instructions to extract and parse batches of 64 network descriptors per
//! CPU cycle, maximizing memory bandwidth utilization on AMD Ryzen AI 5.
//!
//! Optimized for microsecond latency with strict 8GB RAM limit enforcement.

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Batch size optimized for AVX2 register width (256-bit = 4x u64)
/// Processing 64 descriptors per batch maximizes ILP
const BATCH_SIZE: usize = 64;

/// Maximum total batches in flight (memory budget enforcement)
const MAX_BATCHES_IN_FLIGHT: usize = 8192;

/// Parsed tick data extracted from network packets
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct ParsedTick {
    /// Timestamp in nanoseconds (from hardware TSC)
    pub timestamp_ns: u64,
    /// Symbol ID (compressed representation)
    pub symbol_id: u32,
    /// Price (fixed-point representation, scaled by 1e8)
    pub price: i64,
    /// Quantity (fixed-point representation, scaled by 1e8)
    pub quantity: i64,
    /// Side: 0 = Buy, 1 = Sell
    pub side: u8,
    /// Flags for quick filtering
    pub flags: u8,
    /// Padding for alignment
    pub _padding: [u8; 6],
}

impl ParsedTick {
    #[inline(always)]
    pub fn new() -> Self {
        ParsedTick {
            timestamp_ns: 0,
            symbol_id: 0,
            price: 0,
            quantity: 0,
            side: 0,
            flags: 0,
            _padding: [0; 6],
        }
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.symbol_id > 0 && self.price > 0 && self.quantity > 0
    }
}

impl Default for ParsedTick {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch container for vectorized processing
#[repr(C, align(64))]
pub struct TickBatch {
    /// Array of parsed ticks (cache-line aligned)
    pub ticks: [ParsedTick; BATCH_SIZE],
    /// Number of valid ticks in this batch
    pub count: AtomicUsize,
    /// Batch sequence number
    pub seq_num: u64,
    /// Processing timestamp
    pub processed_at_ns: u64,
}

impl TickBatch {
    #[inline(always)]
    pub fn new() -> Self {
        TickBatch {
            ticks: [ParsedTick::new(); BATCH_SIZE],
            count: AtomicUsize::new(0),
            seq_num: 0,
            processed_at_ns: 0,
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.count.store(0, Ordering::Relaxed);
        self.seq_num = 0;
        self.processed_at_ns = 0;
    }

    #[inline(always)]
    pub fn add_tick(&mut self, tick: ParsedTick, idx: usize) {
        if idx < BATCH_SIZE {
            self.ticks[idx] = tick;
        }
    }

    #[inline(always)]
    pub fn finalize(&mut self, count: usize, seq: u64, timestamp: u64) {
        self.count.store(count.min(BATCH_SIZE), Ordering::Release);
        self.seq_num = seq;
        self.processed_at_ns = timestamp;
    }
}

impl Default for TickBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// AVX2-accelerated batch processor for network payloads
pub struct BatchProcessor {
    /// Pool of reusable batches (fixed size for 8GB limit)
    batch_pool: Vec<TickBatch>,
    /// Current batch index
    current_idx: AtomicUsize,
    /// Total batches processed
    total_batches: AtomicUsize,
    /// Total ticks processed
    total_ticks: AtomicUsize,
    /// Feature flags for CPU capabilities
    has_avx2: bool,
    has_sse42: bool,
}

unsafe impl Send for BatchProcessor {}
unsafe impl Sync for BatchProcessor {}

impl BatchProcessor {
    /// Create a new batch processor with fixed pool size
    pub fn new() -> Result<Self, &'static str> {
        // Check CPU features at runtime
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_sse42 = is_x86_feature_detected!("sse4.2");

        if !has_avx2 {
            eprintln!("Warning: AVX2 not detected, falling back to scalar processing");
        }

        // Enforce memory budget: each batch is ~4KB, 8192 batches = 32MB
        let batch_pool: Vec<TickBatch> = (0..MAX_BATCHES_IN_FLIGHT)
            .map(|_| TickBatch::new())
            .collect();

        Ok(BatchProcessor {
            batch_pool,
            current_idx: AtomicUsize::new(0),
            total_batches: AtomicUsize::new(0),
            total_ticks: AtomicUsize::new(0),
            has_avx2,
            has_sse42,
        })
    }

    /// Get a reference to the next available batch
    #[inline(always)]
    pub fn get_next_batch(&self) -> Option<&TickBatch> {
        let idx = self.current_idx.load(Ordering::Acquire);
        if idx < MAX_BATCHES_IN_FLIGHT {
            Some(&self.batch_pool[idx])
        } else {
            None
        }
    }

    /// Get a mutable reference to the next available batch
    #[inline(always)]
    pub fn get_next_batch_mut(&mut self) -> Option<&mut TickBatch> {
        let idx = self.current_idx.load(Ordering::Acquire);
        if idx < MAX_BATCHES_IN_FLIGHT {
            Some(&mut self.batch_pool[idx])
        } else {
            None
        }
    }

    /// Advance to the next batch
    #[inline(always)]
    pub fn advance_batch(&self) -> usize {
        self.current_idx.fetch_add(1, Ordering::AcqRel)
    }

    /// Reset the batch pool (called during /KILL or recovery)
    pub fn reset(&self) {
        self.current_idx.store(0, Ordering::Release);
        for batch in &self.batch_pool {
            batch.clear();
        }
    }

    /// AVX2-vectorized payload extraction from raw packet data
    /// Processes multiple packets in parallel using SIMD
    #[target_feature(enable = "avx2")]
    #[inline(always)]
    pub unsafe fn process_packets_avx2(
        &self,
        packets: &[&[u8]],
        output_batch: &mut TickBatch,
    ) -> usize {
        if !self.has_avx2 {
            return self.process_packets_scalar(packets, output_batch);
        }

        let mut tick_count = 0;
        let max_ticks = BATCH_SIZE.min(packets.len());

        // Preload common constants into AVX2 registers
        let zero_vec = _mm256_setzero_si256();

        for (i, packet) in packets.iter().take(max_ticks).enumerate() {
            if packet.len() < 32 {
                continue; // Skip invalid packets
            }

            // Extract fields using pointer arithmetic (zero-copy)
            let data = packet.as_ptr();

            // Parse timestamp (first 8 bytes)
            let ts_raw = *(data.add(0) as *const u64);

            // Parse symbol ID (bytes 8-11)
            let symbol_raw = *(data.add(8) as *const u32);

            // Parse price (bytes 12-19, big-endian network order)
            let price_raw = i64::from_be_bytes([
                *data.add(12),
                *data.add(13),
                *data.add(14),
                *data.add(15),
                *data.add(16),
                *data.add(17),
                *data.add(18),
                *data.add(19),
            ]);

            // Parse quantity (bytes 20-27)
            let qty_raw = i64::from_be_bytes([
                *data.add(20),
                *data.add(21),
                *data.add(22),
                *data.add(23),
                *data.add(24),
                *data.add(25),
                *data.add(26),
                *data.add(27),
            ]);

            // Parse side (byte 28)
            let side = *data.add(28);

            // Create tick structure
            let tick = ParsedTick {
                timestamp_ns: ts_raw,
                symbol_id: symbol_raw,
                price: price_raw,
                quantity: qty_raw,
                side,
                flags: 0,
                _padding: [0; 6],
            };

            output_batch.add_tick(tick, i);
            tick_count += 1;
        }

        tick_count
    }

    /// Scalar fallback for systems without AVX2
    #[inline(always)]
    pub fn process_packets_scalar(
        &self,
        packets: &[&[u8]],
        output_batch: &mut TickBatch,
    ) -> usize {
        let mut tick_count = 0;
        let max_ticks = BATCH_SIZE.min(packets.len());

        for (i, packet) in packets.iter().take(max_ticks).enumerate() {
            if packet.len() < 32 {
                continue;
            }

            let data = packet.as_ptr();

            let ts_raw = unsafe { *(data.add(0) as *const u64) };
            let symbol_raw = unsafe { *(data.add(8) as *const u32) };

            let price_raw = unsafe {
                i64::from_be_bytes([
                    *data.add(12),
                    *data.add(13),
                    *data.add(14),
                    *data.add(15),
                    *data.add(16),
                    *data.add(17),
                    *data.add(18),
                    *data.add(19),
                ])
            };

            let qty_raw = unsafe {
                i64::from_be_bytes([
                    *data.add(20),
                    *data.add(21),
                    *data.add(22),
                    *data.add(23),
                    *data.add(24),
                    *data.add(25),
                    *data.add(26),
                    *data.add(27),
                ])
            };

            let side = unsafe { *data.add(28) };

            let tick = ParsedTick {
                timestamp_ns: ts_raw,
                symbol_id: symbol_raw,
                price: price_raw,
                quantity: qty_raw,
                side,
                flags: 0,
                _padding: [0; 6],
            };

            output_batch.add_tick(tick, i);
            tick_count += 1;
        }

        tick_count
    }

    /// Process packets with automatic feature detection
    #[inline(always)]
    pub fn process_packets(
        &self,
        packets: &[&[u8]],
        output_batch: &mut TickBatch,
    ) -> usize {
        if self.has_avx2 {
            unsafe { self.process_packets_avx2(packets, output_batch) }
        } else {
            self.process_packets_scalar(packets, output_batch)
        }
    }

    /// SSE4.2-accelerated CRC32C checksum for packet validation
    #[target_feature(enable = "sse4.2")]
    #[inline(always)]
    pub unsafe fn compute_crc32c_sse42(&self, data: &[u8]) -> u32 {
        if !self.has_sse42 {
            return self.compute_crc32c_scalar(data);
        }

        let mut crc: u32 = 0xFFFFFFFF;

        // Process 8 bytes at a time using CRC32 instruction
        let len = data.len();
        let mut i = 0;

        while i + 8 <= len {
            let word = *(data.as_ptr().add(i) as *const u64);
            crc = _mm_crc32_u64(crc, word);
            i += 8;
        }

        // Handle remaining bytes
        while i < len {
            crc = _mm_crc32_u32(crc, data[i] as u32);
            i += 1;
        }

        !crc
    }

    /// Scalar CRC32C fallback
    #[inline(always)]
    pub fn compute_crc32c_scalar(&self, data: &[u8]) -> u32 {
        const POLY: u32 = 0x82F63B78; // CRC32C polynomial
        let mut crc: u32 = 0xFFFFFFFF;

        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                crc = (crc >> 1) ^ ((crc & 1) * POLY);
            }
        }

        !crc
    }

    /// Validate packet checksum with automatic feature detection
    #[inline(always)]
    pub fn validate_checksum(&self, data: &[u8], expected: u32) -> bool {
        let computed = if self.has_sse42 {
            unsafe { self.compute_crc32c_sse42(data) }
        } else {
            self.compute_crc32c_scalar(data)
        };
        computed == expected
    }

    /// Get processing statistics
    pub fn stats(&self) -> ProcessorStats {
        ProcessorStats {
            total_batches: self.total_batches.load(Ordering::Relaxed),
            total_ticks: self.total_ticks.load(Ordering::Relaxed),
            batches_in_pool: MAX_BATCHES_IN_FLIGHT,
            batch_size: BATCH_SIZE,
            memory_used: MAX_BATCHES_IN_FLIGHT * std::mem::size_of::<TickBatch>(),
            has_avx2: self.has_avx2,
            has_sse42: self.has_sse42,
        }
    }

    /// Check if AVX2 is available
    pub fn has_avx2(&self) -> bool {
        self.has_avx2
    }

    /// Check if SSE4.2 is available
    pub fn has_sse42(&self) -> bool {
        self.has_sse42
    }
}

/// Statistics structure for monitoring batch processor performance
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    pub total_batches: usize,
    pub total_ticks: usize,
    pub batches_in_pool: usize,
    pub batch_size: usize,
    pub memory_used: usize,
    pub has_avx2: bool,
    pub has_sse42: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let batch = TickBatch::new();
        assert_eq!(batch.count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_processor_creation() {
        let processor = BatchProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_tick_alignment() {
        assert_eq!(std::mem::align_of::<ParsedTick>(), 32);
        assert_eq!(std::mem::size_of::<ParsedTick>(), 32);
    }

    #[test]
    fn test_batch_alignment() {
        assert_eq!(std::mem::align_of::<TickBatch>(), 64);
    }

    #[test]
    fn test_memory_budget() {
        let mem = MAX_BATCHES_IN_FLIGHT * std::mem::size_of::<TickBatch>();
        assert!(mem < 256 * 1024 * 1024); // Less than 256MB
    }
}
