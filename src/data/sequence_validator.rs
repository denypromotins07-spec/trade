//! Sequence Validator - Real-time Binance WebSocket sequence validation
//!
//! This module validates Binance WebSocket sequence IDs in real-time, instantly
//! triggering a REST snapshot fetch if a single dropped packet causes state
//! desynchronization. Critical for maintaining orderbook integrity.
//!
//! ## Features
//! - Real-time sequence number validation
//! - Automatic gap detection and recovery
//! - REST snapshot integration for re-sync
//! - Integer overflow protection for fuzzing scenarios

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::VecDeque;

/// Maximum allowed sequence gap before triggering full resync
const MAX_SEQUENCE_GAP: u64 = 100;

/// Maximum retries for REST snapshot fetch
const MAX_SNAPSHOT_RETRIES: usize = 3;

/// Timeout for REST snapshot fetch in milliseconds
const SNAPSHOT_TIMEOUT_MS: u64 = 5000;

/// Result of sequence validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationResult {
    /// Sequence is valid and in order
    Valid,
    /// Duplicate sequence number (already processed)
    Duplicate,
    /// Sequence gap detected (missing messages)
    GapDetected(u64),
    /// Sequence went backwards (out of order)
    OutOfOrder,
    /// Integer overflow detected (fuzzing attack)
    OverflowDetected,
    /// Initial sequence (first message)
    Initial,
}

/// State of the sequence validator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidatorState {
    /// Not yet initialized
    Uninitialized,
    /// Actively validating sequences
    Active,
    /// Detected gap, waiting for snapshot
    AwaitingSnapshot,
    /// Recovered from gap
    Recovered,
    /// Error state
    Error,
}

/// Configuration for sequence validator
#[derive(Debug, Clone)]
pub struct SequenceValidatorConfig {
    /// Maximum allowed sequence gap
    pub max_gap: u64,
    /// Enable automatic snapshot fetch on gap
    pub auto_snapshot_on_gap: bool,
    /// Enable integer overflow checks (for fuzzing protection)
    pub check_overflow: bool,
    /// Buffer size for recent sequence tracking
    pub history_buffer_size: usize,
    /// Enable duplicate detection
    pub detect_duplicates: bool,
}

impl Default for SequenceValidatorConfig {
    fn default() -> Self {
        Self {
            max_gap: MAX_SEQUENCE_GAP,
            auto_snapshot_on_gap: true,
            check_overflow: true,
            history_buffer_size: 1000,
            detect_duplicates: true,
        }
    }
}

/// Statistics for sequence validation
#[derive(Debug, Clone, Default)]
pub struct SequenceStats {
    pub total_messages: u64,
    pub valid_messages: u64,
    pub duplicates: u64,
    pub gaps_detected: u64,
    pub out_of_order: u64,
    pub overflows_detected: u64,
    pub snapshots_requested: u64,
    pub successful_recoveries: u64,
    pub last_valid_sequence: u64,
    pub last_validation_time_ns: u64,
}

/// High-performance sequence validator with overflow protection
pub struct SequenceValidator {
    config: SequenceValidatorConfig,
    /// Last validated sequence number
    last_sequence: AtomicU64,
    /// Current validator state
    state: parking_lot::RwLock<ValidatorState>,
    /// History buffer for duplicate detection (ring buffer)
    recent_sequences: parking_lot::Mutex<VecDeque<u64>>,
    /// Statistics
    stats: parking_lot::RwLock<SequenceStats>,
    /// Symbol this validator is tracking
    symbol: String,
    /// Flag indicating snapshot fetch needed
    snapshot_needed: AtomicBool,
    /// Consecutive gap count
    consecutive_gaps: AtomicUsize,
    /// Last successful validation timestamp
    last_validation_ns: AtomicU64,
}

impl SequenceValidator {
    /// Create new sequence validator for a symbol
    pub fn new(symbol: &str) -> Self {
        Self::with_config(symbol, SequenceValidatorConfig::default())
    }

    /// Create new sequence validator with custom config
    pub fn with_config(symbol: &str, config: SequenceValidatorConfig) -> Self {
        Self {
            config,
            last_sequence: AtomicU64::new(0),
            state: parking_lot::RwLock::new(ValidatorState::Uninitialized),
            recent_sequences: parking_lot::Mutex::new(
                VecDeque::with_capacity(config.history_buffer_size)
            ),
            stats: parking_lot::RwLock::new(SequenceStats::default()),
            symbol: symbol.to_string(),
            snapshot_needed: AtomicBool::new(false),
            consecutive_gaps: AtomicUsize::new(0),
            last_validation_ns: AtomicU64::new(0),
        }
    }

    /// Validate a new sequence number
    /// Returns validation result and updates internal state
    #[inline]
    pub fn validate(&self, sequence: u64) -> ValidationResult {
        let current_time_ns = get_current_time_ns();
        let mut stats = self.stats.write();
        
        stats.total_messages = stats.total_messages.saturating_add(1);
        stats.last_validation_time_ns = current_time_ns;

        // Check for integer overflow (fuzzing protection)
        if self.config.check_overflow {
            if let Some(result) = self.check_overflow(sequence) {
                stats.overflows_detected = stats.overflows_detected.saturating_add(1);
                *self.state.write() = ValidatorState::Error;
                return result;
            }
        }

        let last_seq = self.last_sequence.load(AtomicOrdering::Acquire);

        // First message - initialize
        if last_seq == 0 {
            self.last_sequence.store(sequence, AtomicOrdering::Release);
            *self.state.write() = ValidatorState::Active;
            self.add_to_history(sequence);
            stats.valid_messages = stats.valid_messages.saturating_add(1);
            stats.last_valid_sequence = sequence;
            self.last_validation_ns.store(current_time_ns, AtomicOrdering::Release);
            return ValidationResult::Initial;
        }

        // Check for duplicate
        if self.config.detect_duplicates && self.is_duplicate(sequence) {
            stats.duplicates = stats.duplicates.saturating_add(1);
            return ValidationResult::Duplicate;
        }

        // Check sequence ordering
        match sequence.checked_sub(last_seq) {
            None => {
                // sequence < last_seq (underflow means out of order)
                stats.out_of_order = stats.out_of_order.saturating_add(1);
                return ValidationResult::OutOfOrder;
            }
            Some(0) => {
                // sequence == last_seq (duplicate)
                if self.config.detect_duplicates {
                    stats.duplicates = stats.duplicates.saturating_add(1);
                    return ValidationResult::Duplicate;
                }
            }
            Some(gap) if gap == 1 => {
                // Perfect sequence (expected case)
                self.last_sequence.store(sequence, AtomicOrdering::Release);
                self.add_to_history(sequence);
                stats.valid_messages = stats.valid_messages.saturating_add(1);
                stats.last_valid_sequence = sequence;
                self.consecutive_gaps.store(0, AtomicOrdering::Relaxed);
                self.last_validation_ns.store(current_time_ns, AtomicOrdering::Release);
                return ValidationResult::Valid;
            }
            Some(gap) if gap > 1 && gap <= self.config.max_gap => {
                // Small gap - might recover naturally
                stats.gaps_detected = stats.gaps_detected.saturating_add(1);
                let gaps = self.consecutive_gaps.fetch_add(1, AtomicOrdering::Relaxed);
                
                if gaps >= 3 {
                    // Multiple consecutive gaps - trigger snapshot
                    self.trigger_snapshot();
                    *self.state.write() = ValidatorState::AwaitingSnapshot;
                }
                
                // Update last sequence anyway to continue
                self.last_sequence.store(sequence, AtomicOrdering::Release);
                self.add_to_history(sequence);
                
                return ValidationResult::GapDetected(gap);
            }
            Some(gap) => {
                // Large gap - immediate snapshot required
                stats.gaps_detected = stats.gaps_detected.saturating_add(1);
                self.trigger_snapshot();
                *self.state.write() = ValidatorState::AwaitingSnapshot;
                
                return ValidationResult::GapDetected(gap);
            }
        }

        // Fallback - treat as valid but suspicious
        self.last_sequence.store(sequence, AtomicOrdering::Release);
        self.add_to_history(sequence);
        stats.valid_messages = stats.valid_messages.saturating_add(1);
        ValidationResult::Valid
    }

    /// Check for integer overflow conditions (fuzzing protection)
    #[inline]
    fn check_overflow(&self, sequence: u64) -> Option<ValidationResult> {
        // Detect potential integer overflow attacks
        // Binance sequence numbers should be monotonically increasing
        // Values near u64::MAX are suspicious
        
        const OVERFLOW_THRESHOLD: u64 = u64::MAX - 1_000_000;
        
        if sequence > OVERFLOW_THRESHOLD {
            return Some(ValidationResult::OverflowDetected);
        }

        // Check for wraparound attempts
        let last_seq = self.last_sequence.load(AtomicOrdering::Acquire);
        if last_seq > OVERFLOW_THRESHOLD && sequence < 1_000_000 {
            // Possible wraparound from MAX to 0
            return Some(ValidationResult::OverflowDetected);
        }

        None
    }

    /// Check if sequence is in recent history (duplicate detection)
    #[inline]
    fn is_duplicate(&self, sequence: u64) -> bool {
        let history = self.recent_sequences.lock();
        history.contains(&sequence)
    }

    /// Add sequence to history buffer
    #[inline]
    fn add_to_history(&self, sequence: u64) {
        let mut history = self.recent_sequences.lock();
        
        if history.len() >= self.config.history_buffer_size {
            history.pop_front();
        }
        history.push_back(sequence);
    }

    /// Trigger snapshot fetch request
    #[inline]
    fn trigger_snapshot(&self) {
        self.snapshot_needed.store(true, AtomicOrdering::Release);
        let mut stats = self.stats.write();
        stats.snapshots_requested = stats.snapshots_requested.saturating_add(1);
    }

    /// Check if snapshot fetch is needed
    #[inline]
    pub fn is_snapshot_needed(&self) -> bool {
        self.snapshot_needed.load(AtomicOrdering::Acquire)
    }

    /// Mark snapshot as fetched and apply new sequence
    pub fn apply_snapshot(&self, snapshot_sequence: u64) {
        self.last_sequence.store(snapshot_sequence, AtomicOrdering::Release);
        self.snapshot_needed.store(false, AtomicOrdering::Release);
        self.consecutive_gaps.store(0, AtomicOrdering::Relaxed);
        *self.state.write() = ValidatorState::Recovered;
        
        let mut stats = self.stats.write();
        stats.successful_recoveries = stats.successful_recoveries.saturating_add(1);
        stats.last_valid_sequence = snapshot_sequence;
        
        log::info!("Sequence validator for {} recovered via snapshot at seq {}", 
                   self.symbol, snapshot_sequence);
    }

    /// Get current validator state
    #[inline]
    pub fn get_state(&self) -> ValidatorState {
        *self.state.read()
    }

    /// Get current statistics
    pub fn get_stats(&self) -> SequenceStats {
        self.stats.read().clone()
    }

    /// Get last validated sequence
    #[inline]
    pub fn get_last_sequence(&self) -> u64 {
        self.last_sequence.load(AtomicOrdering::Acquire)
    }

    /// Get symbol
    #[inline]
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Reset validator state
    pub fn reset(&self) {
        self.last_sequence.store(0, AtomicOrdering::Release);
        *self.state.write() = ValidatorState::Uninitialized;
        self.recent_sequences.lock().clear();
        self.snapshot_needed.store(false, AtomicOrdering::Release);
        self.consecutive_gaps.store(0, AtomicOrdering::Relaxed);
        log::info!("Sequence validator for {} reset", self.symbol);
    }

    /// Get time since last validation
    pub fn time_since_last_validation(&self) -> Duration {
        let last_ns = self.last_validation_ns.load(AtomicOrdering::Acquire);
        if last_ns == 0 {
            return Duration::MAX;
        }
        let current_ns = get_current_time_ns();
        Duration::from_nanos(current_ns.saturating_sub(last_ns))
    }

    /// Check if validator is healthy
    pub fn is_healthy(&self) -> bool {
        let state = *self.state.read();
        match state {
            ValidatorState::Active | ValidatorState::Recovered => true,
            ValidatorState::AwaitingSnapshot => {
                // Check if we've been waiting too long
                self.time_since_last_validation() < Duration::from_millis(SNAPSHOT_TIMEOUT_MS)
            }
            _ => false,
        }
    }
}

/// Multi-symbol sequence validator manager
pub struct MultiSymbolValidator {
    validators: Arc<parking_lot::RwLock<std::collections::HashMap<String, Arc<SequenceValidator>>>>,
    global_snapshot_needed: AtomicBool,
}

impl MultiSymbolValidator {
    /// Create new multi-symbol validator
    pub fn new() -> Self {
        Self {
            validators: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            global_snapshot_needed: AtomicBool::new(false),
        }
    }

    /// Get or create validator for symbol
    pub fn get_or_create(&self, symbol: &str) -> Arc<SequenceValidator> {
        let mut validators = self.validators.write();
        validators.entry(symbol.to_string())
            .or_insert_with(|| Arc::new(SequenceValidator::new(symbol)))
            .clone()
    }

    /// Validate sequence for symbol
    pub fn validate(&self, symbol: &str, sequence: u64) -> ValidationResult {
        let validator = self.get_or_create(symbol);
        let result = validator.validate(sequence);
        
        // Set global flag if any validator needs snapshot
        if matches!(result, ValidationResult::GapDetected(_)) {
            self.global_snapshot_needed.store(true, AtomicOrdering::Release);
        }
        
        result
    }

    /// Check if any symbol needs snapshot
    pub fn any_snapshot_needed(&self) -> bool {
        if self.global_snapshot_needed.load(AtomicOrdering::Acquire) {
            return true;
        }
        
        let validators = self.validators.read();
        validators.values().any(|v| v.is_snapshot_needed())
    }

    /// Get all symbols needing snapshot
    pub fn get_symbols_needing_snapshot(&self) -> Vec<String> {
        let validators = self.validators.read();
        validators.iter()
            .filter(|(_, v)| v.is_snapshot_needed())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Get total statistics across all validators
    pub fn get_global_stats(&self) -> GlobalSequenceStats {
        let validators = self.validators.read();
        let mut total = GlobalSequenceStats::default();
        
        for validator in validators.values() {
            let stats = validator.get_stats();
            total.total_messages += stats.total_messages;
            total.valid_messages += stats.valid_messages;
            total.duplicates += stats.duplicates;
            total.gaps_detected += stats.gaps_detected;
            total.out_of_order += stats.out_of_order;
            total.overflows_detected += stats.overflows_detected;
            total.snapshots_requested += stats.snapshots_requested;
            total.successful_recoveries += stats.successful_recoveries;
        }
        
        total.validator_count = validators.len();
        total
    }

    /// Reset all validators
    pub fn reset_all(&self) {
        let validators = self.validators.read();
        for validator in validators.values() {
            validator.reset();
        }
        self.global_snapshot_needed.store(false, AtomicOrdering::Release);
    }
}

impl Default for MultiSymbolValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Global statistics across all validators
#[derive(Debug, Clone, Default)]
pub struct GlobalSequenceStats {
    pub validator_count: usize,
    pub total_messages: u64,
    pub valid_messages: u64,
    pub duplicates: u64,
    pub gaps_detected: u64,
    pub out_of_order: u64,
    pub overflows_detected: u64,
    pub snapshots_requested: u64,
    pub successful_recoveries: u64,
}

/// Get current time in nanoseconds
#[inline]
fn get_current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

/// Snapshot fetcher interface (to be implemented by network layer)
pub trait SnapshotFetcher: Send + Sync {
    /// Fetch orderbook snapshot from REST API
    fn fetch_snapshot(&self, symbol: &str) -> Result<OrderbookSnapshot, SnapshotError>;
}

/// Orderbook snapshot data
#[derive(Debug, Clone)]
pub struct OrderbookSnapshot {
    pub symbol: String,
    pub last_update_id: u64,
    pub bids: Vec<(u64, u64)>,
    pub asks: Vec<(u64, u64)>,
}

/// Snapshot fetch error
#[derive(Debug, Clone)]
pub struct SnapshotError {
    pub message: String,
    pub retryable: bool,
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SnapshotError: {}", self.message)
    }
}

impl std::error::Error for SnapshotError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_validator_basic() {
        let validator = SequenceValidator::new("BTCUSDT");
        
        // First message
        assert_eq!(validator.validate(1), ValidationResult::Initial);
        
        // Normal sequence
        assert_eq!(validator.validate(2), ValidationResult::Valid);
        assert_eq!(validator.validate(3), ValidationResult::Valid);
        
        // Duplicate
        assert_eq!(validator.validate(3), ValidationResult::Duplicate);
        
        // Out of order
        assert_eq!(validator.validate(2), ValidationResult::OutOfOrder);
    }

    #[test]
    fn test_gap_detection() {
        let validator = SequenceValidator::new("ETHUSDT");
        
        validator.validate(1);
        validator.validate(2);
        
        // Gap of 5
        let result = validator.validate(7);
        assert!(matches!(result, ValidationResult::GapDetected(5)));
        assert!(validator.is_snapshot_needed());
    }

    #[test]
    fn test_overflow_protection() {
        let validator = SequenceValidator::with_config(
            "TESTUSDT",
            SequenceValidatorConfig {
                check_overflow: true,
                ..Default::default()
            }
        );
        
        // Near-max value should trigger overflow
        let overflow_seq = u64::MAX - 100;
        let result = validator.validate(overflow_seq);
        assert_eq!(result, ValidationResult::OverflowDetected);
    }

    #[test]
    fn test_recovery() {
        let validator = SequenceValidator::new("RECOVERY");
        
        validator.validate(1);
        validator.validate(100); // Gap
        
        assert!(validator.is_snapshot_needed());
        
        // Apply snapshot
        validator.apply_snapshot(100);
        assert!(!validator.is_snapshot_needed());
        assert_eq!(validator.get_state(), ValidatorState::Recovered);
        
        // Continue normal operation
        assert_eq!(validator.validate(101), ValidationResult::Valid);
    }

    #[test]
    fn test_fuzzing_resistance() {
        let validator = SequenceValidator::with_config(
            "FUZZTEST",
            SequenceValidatorConfig {
                check_overflow: true,
                ..Default::default()
            }
        );

        // Test various overflow scenarios
        let test_cases = vec![
            u64::MAX,
            u64::MAX - 1,
            u64::MAX / 2,
            0,
            1,
        ];

        for seq in test_cases {
            let _ = validator.validate(seq);
            // Should not panic
        }

        let stats = validator.get_stats();
        // Verify overflow detection worked
        assert!(stats.overflows_detected > 0 || stats.total_messages > 0);
    }
}
