//! Network Fuzzing Harness - WebSocket Parser Stress Testing
//!
//! This module builds an automated fuzzing harness that bombards the WebSocket
//! parser with malformed Binance JSON and truncated payloads to guarantee the
//! hot path never panics. Explicitly tests for integer overflows in sequence
//! validator logic.
//!
//! ## Features
//! - Malformed JSON injection
//! - Truncated payload testing
//! - Integer overflow detection
//! - Sequence number fuzzing
//! - Panic-free verification

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Maximum number of fuzzing iterations per test
const MAX_FUZZ_ITERATIONS: usize = 10000;

/// Timeout for individual fuzz tests in milliseconds
const FUZZ_TEST_TIMEOUT_MS: u64 = 5000;

/// Result of a fuzz test iteration
#[derive(Debug, Clone, Copy)]
pub struct FuzzResult {
    pub iteration: usize,
    pub input_size: usize,
    pub panicked: bool,
    pub error_message: Option<&'static str>,
    pub execution_time_ns: u64,
}

/// Statistics from fuzzing run
#[derive(Debug, Clone, Default)]
pub struct FuzzStats {
    pub total_iterations: usize,
    pub successful_parses: usize,
    pub rejected_inputs: usize,
    pub panics_caught: usize,
    pub overflows_detected: usize,
    pub avg_execution_time_ns: u64,
    pub max_execution_time_ns: u64,
    pub min_execution_time_ns: u64,
}

/// Fuzzer for Binance WebSocket messages
pub struct WebSocketFuzzer {
    /// Random seed for deterministic replay
    seed: u64,
    /// Current RNG state (LCG for speed)
    rng_state: AtomicU64,
    /// Statistics
    stats: parking_lot::RwLock<FuzzStats>,
    /// Panic counter
    panics_caught: AtomicUsize,
    /// Overflow counter
    overflows_detected: AtomicUsize,
}

impl WebSocketFuzzer {
    /// Create new fuzzer with given seed
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng_state: AtomicU64::new(seed),
            stats: parking_lot::RwLock::new(FuzzStats::default()),
            panics_caught: AtomicUsize::new(0),
            overflows_detected: AtomicUsize::new(0),
        }
    }

    /// Fast LCG random number generator
    #[inline]
    fn next_random(&self) -> u64 {
        let state = self.rng_state.load(AtomicOrdering::Relaxed);
        let new_state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng_state.store(new_state, AtomicOrdering::Relaxed);
        new_state
    }

    /// Generate random bytes
    #[inline]
    fn random_bytes(&self, len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| (self.next_random() & 0xFF) as u8)
            .collect()
    }

    /// Generate malformed JSON payloads
    pub fn generate_malformed_json(&self, variant: usize) -> String {
        match variant % 20 {
            0 => "{".to_string(),                          // Unclosed brace
            1 => "{\"key\": ".to_string(),                 // Incomplete value
            2 => "{\"key\": }".to_string(),                // Missing value
            3 => "{\"key\": \"value\"".to_string(),        // Missing closing brace
            4 => "{key: \"value\"}".to_string(),           // Unquoted key
            5 => "{\"key\": \"value\",}".to_string(),      // Trailing comma
            6 => "{\"key\": NaN}".to_string(),             // Invalid number
            7 => "{\"key\": Infinity}".to_string(),        // Invalid number
            8 => "{\"key\": undefined}".to_string(),       // JavaScript keyword
            9 => "{\"key\": \"unclosed".to_string(),       // Unclosed string
            10 => "{\"key\": \"value\"\"other\": 1}".to_string(), // Missing comma
            11 => "{\"key\": [1, 2, 3}".to_string(),       // Unclosed array
            12 => "{\"key\": [1, 2, 3]}".to_string(),      // Valid but we'll truncate
            13 => "{\"e\": \"BTCUSDT\", \"p\": ".to_string(), // Partial price
            14 => "{\"lastUpdateId\": ".to_string(),       // Partial update ID
            15 => "{\"bids\": [[\"50000\"".to_string(),    // Incomplete bid
            16 => "{\"asks\": []}}}}}".to_string(),        // Extra closing braces
            17 => "{\"key\": \"\u{0000}\"}".to_string(),   // Null byte in string
            18 => "{\"key\": \"test\\uXXXX\"}".to_string(), // Invalid unicode escape
            19 => "".to_string(),                           // Empty string
            _ => "{\"malformed\": true}".to_string(),
        }
    }

    /// Generate truncated payloads simulating network issues
    pub fn generate_truncated_payload(&self, original_len: usize, truncation_point: usize) -> Vec<u8> {
        let mut payload = self.random_bytes(original_len);
        
        if truncation_point < payload.len() {
            payload.truncate(truncation_point);
        }
        
        payload
    }

    /// Generate sequence numbers for overflow testing
    pub fn generate_overflow_sequences(&self) -> Vec<u64> {
        vec![
            0,
            1,
            u64::MAX,
            u64::MAX - 1,
            u64::MAX / 2,
            u64::MAX / 2 + 1,
            i64::MAX as u64,
            i64::MAX as u64 + 1,
            i64::MIN as u64,
            1000000,
            999999,
            1000001,
        ]
    }

    /// Run fuzzing test on JSON parser
    pub fn fuzz_json_parser<F>(&self, parse_fn: F, iterations: usize) -> FuzzStats
    where
        F: Fn(&str) -> Result<(), String> + Sync + Send,
    {
        let start_time = Instant::now();
        let mut stats = FuzzStats::default();
        stats.total_iterations = iterations.min(MAX_FUZZ_ITERATIONS);
        
        let mut exec_times: Vec<u64> = Vec::with_capacity(stats.total_iterations);
        
        for i in 0..stats.total_iterations {
            let json = self.generate_malformed_json(i);
            let input_size = json.len();
            
            let test_start = Instant::now();
            
            // Use catch_unwind to detect panics
            let result = std::panic::catch_unwind(|| {
                parse_fn(&json)
            });
            
            let exec_time = test_start.elapsed().as_nanos() as u64;
            exec_times.push(exec_time);
            
            match result {
                Ok(Ok(())) => {
                    stats.successful_parses += 1;
                }
                Ok(Err(_)) => {
                    stats.rejected_inputs += 1;
                }
                Err(_) => {
                    stats.panics_caught += 1;
                    self.panics_caught.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
        }
        
        // Calculate timing statistics
        if !exec_times.is_empty() {
            stats.avg_execution_time_ns = exec_times.iter().sum::<u64>() / exec_times.len() as u64;
            stats.max_execution_time_ns = *exec_times.iter().max().unwrap_or(&0);
            stats.min_execution_time_ns = *exec_times.iter().min().unwrap_or(&0);
        }
        
        *self.stats.write() = stats;
        stats
    }

    /// Test sequence validator for integer overflows
    pub fn test_sequence_overflow<V>(&self, validator: &V, get_last_seq: V::GetLastSeqFn) -> FuzzStats
    where
        V: SequenceOverflowTester,
    {
        let mut stats = FuzzStats::default();
        let sequences = self.generate_overflow_sequences();
        
        stats.total_iterations = sequences.len();
        
        for (i, seq) in sequences.iter().enumerate() {
            let test_result = std::panic::catch_unwind(|| {
                validator.test_sequence(*seq)
            });
            
            match test_result {
                Ok(result) => {
                    if result.overflow_detected {
                        stats.overflows_detected += 1;
                        self.overflows_detected.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                    stats.successful_parses += 1;
                }
                Err(_) => {
                    stats.panics_caught += 1;
                    self.panics_caught.fetch_add(1, AtomicOrdering::Relaxed);
                }
            }
        }
        
        stats
    }

    /// Fuzz binary message parser
    pub fn fuzz_binary_parser<F>(&self, parse_fn: F, iterations: usize) -> FuzzStats
    where
        F: Fn(&[u8]) -> Result<(), String> + Sync + Send,
    {
        let mut stats = FuzzStats::default();
        stats.total_iterations = iterations.min(MAX_FUZZ_ITERATIONS);
        
        let mut exec_times: Vec<u64> = Vec::with_capacity(stats.total_iterations);
        
        for i in 0..stats.total_iterations {
            // Generate various payload sizes
            let payload_size = match i % 5 {
                0 => 0,           // Empty
                1 => 1,           // Single byte
                2 => 100,         // Small
                3 => 10000,       // Medium
                _ => 100000,      // Large
            };
            
            let payload = self.random_bytes(payload_size);
            let input_size = payload.len();
            
            // Also test truncation
            let truncation_variants = if i < 100 {
                vec![0, 1, 2, payload.len() / 2, payload.len()]
            } else {
                vec![payload.len()]
            };
            
            for truncated_len in truncation_variants {
                let test_payload = if truncated_len < payload.len() {
                    &payload[..truncated_len]
                } else {
                    &payload[..]
                };
                
                let test_start = Instant::now();
                
                let result = std::panic::catch_unwind(|| {
                    parse_fn(test_payload)
                });
                
                let exec_time = test_start.elapsed().as_nanos() as u64;
                exec_times.push(exec_time);
                
                match result {
                    Ok(Ok(())) => stats.successful_parses += 1,
                    Ok(Err(_)) => stats.rejected_inputs += 1,
                    Err(_) => {
                        stats.panics_caught += 1;
                        self.panics_caught.fetch_add(1, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
        
        if !exec_times.is_empty() {
            stats.avg_execution_time_ns = exec_times.iter().sum::<u64>() / exec_times.len() as u64;
            stats.max_execution_time_ns = *exec_times.iter().max().unwrap_or(&0);
            stats.min_execution_time_ns = *exec_times.iter().min().unwrap_or(&0);
        }
        
        *self.stats.write() = stats;
        stats
    }

    /// Get current statistics
    pub fn get_stats(&self) -> FuzzStats {
        self.stats.read().clone()
    }

    /// Check if any panics were caught
    pub fn has_panics(&self) -> bool {
        self.panics_caught.load(AtomicOrdering::Relaxed) > 0
    }

    /// Get panic count
    pub fn panic_count(&self) -> usize {
        self.panics_caught.load(AtomicOrdering::Relaxed)
    }
}

/// Trait for sequence overflow testing
pub trait SequenceOverflowTester {
    type GetLastSeqFn: Fn() -> u64;
    
    fn test_sequence(&self, seq: u64) -> SequenceTestResult;
}

/// Result of sequence test
#[derive(Debug, Clone)]
pub struct SequenceTestResult {
    pub accepted: bool,
    pub overflow_detected: bool,
    pub error_message: Option<String>,
}

/// Comprehensive fuzzing harness runner
pub struct FuzzHarness {
    fuzzer: Arc<WebSocketFuzzer>,
    results: parking_lot::RwLock<Vec<FuzzResult>>,
}

impl FuzzHarness {
    /// Create new fuzz harness
    pub fn new(seed: u64) -> Self {
        Self {
            fuzzer: Arc::new(WebSocketFuzzer::new(seed)),
            results: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Run comprehensive fuzzing suite
    pub fn run_comprehensive_fuzz<F, B>(&self, json_parser: F, binary_parser: B) -> FuzzStats
    where
        F: Fn(&str) -> Result<(), String> + Sync + Send + 'static,
        B: Fn(&[u8]) -> Result<(), String> + Sync + Send + 'static,
    {
        let mut total_stats = FuzzStats::default();
        
        // JSON fuzzing
        log::info!("Starting JSON parser fuzzing...");
        let json_stats = self.fuzzer.fuzz_json_parser(&json_parser, 5000);
        total_stats.panics_caught += json_stats.panics_caught;
        total_stats.successful_parses += json_stats.successful_parses;
        total_stats.rejected_inputs += json_stats.rejected_inputs;
        
        // Binary fuzzing
        log::info!("Starting binary parser fuzzing...");
        let binary_stats = self.fuzzer.fuzz_binary_parser(&binary_parser, 5000);
        total_stats.panics_caught += binary_stats.panics_caught;
        total_stats.successful_parses += binary_stats.successful_parses;
        total_stats.rejected_inputs += binary_stats.rejected_inputs;
        
        log::info!(
            "Fuzzing complete: {} panics, {} successful, {} rejected",
            total_stats.panics_caught,
            total_stats.successful_parses,
            total_stats.rejected_inputs
        );
        
        total_stats
    }

    /// Verify no panics occurred
    pub fn verify_no_panics(&self) -> bool {
        !self.fuzzer.has_panics()
    }

    /// Get fuzzer reference
    pub fn fuzzer(&self) -> &WebSocketFuzzer {
        &self.fuzzer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malformed_json_generation() {
        let fuzzer = WebSocketFuzzer::new(42);
        
        for i in 0..20 {
            let json = fuzzer.generate_malformed_json(i);
            // Should not panic
            assert!(json.len() <= 1000);
        }
    }

    #[test]
    fn test_overflow_sequences() {
        let fuzzer = WebSocketFuzzer::new(123);
        let sequences = fuzzer.generate_overflow_sequences();
        
        assert!(sequences.contains(&u64::MAX));
        assert!(sequences.contains(&0));
        assert!(sequences.contains(&1));
    }

    #[test]
    fn test_json_parser_fuzz() {
        let fuzzer = WebSocketFuzzer::new(456);
        
        // Simple parser that rejects everything (safe)
        let safe_parser = |_s: &str| -> Result<(), String> { Err("rejected".to_string()) };
        
        let stats = fuzzer.fuzz_json_parser(safe_parser, 100);
        
        assert_eq!(stats.panics_caught, 0);
        assert!(stats.total_iterations > 0);
    }

    #[test]
    fn test_binary_parser_fuzz() {
        let fuzzer = WebSocketFuzzer::new(789);
        
        // Safe binary parser
        let safe_parser = |_b: &[u8]| -> Result<(), String> { Ok(()) };
        
        let stats = fuzzer.fuzz_binary_parser(safe_parser, 100);
        
        // Should not panic
        assert!(stats.total_iterations > 0);
    }

    #[test]
    fn test_harness_verification() {
        let harness = FuzzHarness::new(999);
        
        let safe_json = |_s: &str| Err("safe".to_string());
        let safe_binary = |_b: &[u8]| Ok(());
        
        let _stats = harness.run_comprehensive_fuzz(safe_json, safe_binary);
        
        assert!(harness.verify_no_panics());
    }
}
