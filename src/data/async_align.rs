//! Lock-Free Temporal Alignment Engine
//! 
//! This module builds a lock-free temporal alignment engine that synchronizes asynchronous
//! L2 order book updates and trade tape using strict sequence watermarks to prevent state corruption.
//! 
//! Optimized for:
//! - Microsecond latency via lock-free atomics
//! - 8GB RAM limit enforcement via bounded buffers
//! - AMD Ryzen AI 5 architecture compatibility

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::collections::VecDeque;

/// Lock-free memory counter
static ALIGNMENT_MEMORY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Memory budget for alignment module (300MB)
const ALIGNMENT_MEMORY_BUDGET: u64 = 1024 * 1024 * 300;

/// Maximum buffer size for pending messages
const MAX_PENDING_BUFFER: usize = 50000;

/// Message type enumeration
#[derive(Debug, Clone, Copy)]
pub enum MessageType {
    L2Snapshot,
    L2Update,
    Trade,
    Quote,
}

/// Aligned message structure
#[derive(Debug, Clone)]
pub struct AlignedMessage {
    pub msg_type: MessageType,
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub symbol_id: u32,
    pub payload_hash: u64,
    pub processed: AtomicBool,
}

/// Sequence watermark for tracking progress
pub struct SequenceWatermark {
    /// Last processed sequence number
    last_processed: AtomicU64,
    /// Expected next sequence number
    expected_next: AtomicU64,
    /// Gap detection threshold
    gap_threshold: u64,
    /// Count of detected gaps
    gap_count: AtomicU64,
}

impl SequenceWatermark {
    pub fn new(initial_sequence: u64, gap_threshold: u64) -> Self {
        Self {
            last_processed: AtomicU64::new(initial_sequence),
            expected_next: AtomicU64::new(initial_sequence + 1),
            gap_threshold,
            gap_count: AtomicU64::new(0),
        }
    }
    
    /// Check if sequence is valid (no gap)
    pub fn check_sequence(&self, sequence: u64) -> SequenceStatus {
        let expected = self.expected_next.load(Ordering::Relaxed);
        
        if sequence == expected {
            SequenceStatus::Valid
        } else if sequence < expected {
            SequenceStatus::Duplicate
        } else if sequence > expected + self.gap_threshold {
            SequenceStatus::LargeGap
        } else {
            SequenceStatus::Gap
        }
    }
    
    /// Advance watermark after processing
    pub fn advance(&self, sequence: u64) -> bool {
        let current_expected = self.expected_next.load(Ordering::Relaxed);
        
        if sequence == current_expected {
            self.last_processed.store(sequence, Ordering::Relaxed);
            self.expected_next.store(sequence + 1, Ordering::Relaxed);
            true
        } else {
            if sequence > current_expected {
                self.gap_count.fetch_add(1, Ordering::Relaxed);
            }
            false
        }
    }
    
    /// Get current watermark status
    pub fn get_status(&self) -> WatermarkStatus {
        WatermarkStatus {
            last_processed: self.last_processed.load(Ordering::Relaxed),
            expected_next: self.expected_next.load(Ordering::Relaxed),
            gap_count: self.gap_count.load(Ordering::Relaxed),
        }
    }
}

/// Sequence validation result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SequenceStatus {
    Valid,
    Duplicate,
    Gap,
    LargeGap,
}

/// Watermark status snapshot
#[derive(Debug, Clone)]
pub struct WatermarkStatus {
    pub last_processed: u64,
    pub expected_next: u64,
    pub gap_count: u64,
}

/// Lock-free message buffer with sequence tracking
pub struct AlignedMessageBuffer {
    /// Pending messages waiting for alignment
    pending: VecDeque<AlignedMessage>,
    /// Per-symbol watermarks
    watermarks: Vec<SequenceWatermark>,
    /// Maximum buffer size
    max_size: usize,
    /// Count of dropped messages due to buffer full
    dropped_count: AtomicU64,
}

impl AlignedMessageBuffer {
    /// Create new aligned message buffer
    pub fn new(n_symbols: usize, max_size: usize) -> Result<Self, &'static str> {
        let actual_max = max_size.min(MAX_PENDING_BUFFER);
        let estimated_memory = (actual_max * std::mem::size_of::<AlignedMessage>() 
            + n_symbols * std::mem::size_of::<SequenceWatermark>()) as u64;
        
        let current_usage = ALIGNMENT_MEMORY_COUNTER.load(Ordering::Relaxed);
        if current_usage + estimated_memory > ALIGNMENT_MEMORY_BUDGET {
            return Err("Memory budget exceeded for alignment buffer");
        }
        
        ALIGNMENT_MEMORY_COUNTER.fetch_add(estimated_memory, Ordering::Relaxed);
        
        let watermarks = (0..n_symbols)
            .map(|_| SequenceWatermark::new(0, 100))
            .collect();
        
        Ok(Self {
            pending: VecDeque::with_capacity(actual_max),
            watermarks,
            max_size: actual_max,
            dropped_count: AtomicU64::new(0),
        })
    }
    
    /// Add message to buffer with sequence validation
    pub fn add_message(&mut self, mut msg: AlignedMessage) -> AlignResult {
        // Check symbol validity
        if msg.symbol_id as usize >= self.watermarks.len() {
            return AlignResult::InvalidSymbol;
        }
        
        // Check sequence
        let watermark = &self.watermarks[msg.symbol_id as usize];
        let seq_status = watermark.check_sequence(msg.sequence);
        
        match seq_status {
            SequenceStatus::Duplicate => AlignResult::Duplicate,
            SequenceStatus::LargeGap => {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
                AlignResult::LargeGapDropped
            },
            SequenceStatus::Gap | SequenceStatus::Valid => {
                // Check buffer capacity
                if self.pending.len() >= self.max_size {
                    // Remove oldest message
                    self.pending.pop_front();
                    self.dropped_count.fetch_add(1, Ordering::Relaxed);
                }
                
                msg.processed.store(false, Ordering::Relaxed);
                self.pending.push_back(msg);
                
                if seq_status == SequenceStatus::Valid {
                    AlignResult::AcceptedAndReady
                } else {
                    AlignResult::AcceptedWaiting
                }
            }
        }
    }
    
    /// Get next ready-to-process message
    pub fn get_next_ready(&mut self, symbol_id: u32) -> Option<&AlignedMessage> {
        if symbol_id as usize >= self.watermarks.len() {
            return None;
        }
        
        let watermark = &self.watermarks[symbol_id as usize];
        let expected = watermark.expected_next.load(Ordering::Relaxed);
        
        // Find message with expected sequence
        for msg in &self.pending {
            if msg.symbol_id == symbol_id && msg.sequence == expected {
                return Some(msg);
            }
        }
        
        None
    }
    
    /// Mark message as processed and advance watermark
    pub fn mark_processed(&mut self, symbol_id: u32, sequence: u64) -> bool {
        if symbol_id as usize >= self.watermarks.len() {
            return false;
        }
        
        let watermark = &self.watermarks[symbol_id as usize];
        
        if watermark.advance(sequence) {
            // Remove processed message from buffer
            self.pending.retain(|msg| !(msg.symbol_id == symbol_id && msg.sequence == sequence));
            true
        } else {
            false
        }
    }
    
    /// Process all ready messages for a symbol
    pub fn drain_ready(&mut self, symbol_id: u32) -> Vec<AlignedMessage> {
        let mut result = Vec::new();
        
        if symbol_id as usize >= self.watermarks.len() {
            return result;
        }
        
        let watermark = &self.watermarks[symbol_id as usize];
        let mut expected = watermark.expected_next.load(Ordering::Relaxed);
        
        // Collect consecutive ready messages
        let mut indices_to_remove = Vec::new();
        
        for (idx, msg) in self.pending.iter().enumerate() {
            if msg.symbol_id == symbol_id && msg.sequence == expected {
                indices_to_remove.push(idx);
                result.push(msg.clone());
                expected += 1;
            }
        }
        
        // Remove processed messages (in reverse order to maintain indices)
        for idx in indices_to_remove.into_iter().rev() {
            self.pending.remove(idx);
        }
        
        // Update watermark
        if !result.is_empty() {
            let last_seq = result.last().unwrap().sequence;
            self.watermarks[symbol_id as usize].advance(last_seq);
        }
        
        result
    }
    
    /// Get buffer statistics
    pub fn get_stats(&self) -> BufferStats {
        BufferStats {
            pending_count: self.pending.len(),
            total_dropped: self.dropped_count.load(Ordering::Relaxed),
            watermarks: self.watermarks.iter().map(|w| w.get_status()).collect(),
        }
    }
}

impl Drop for AlignedMessageBuffer {
    fn drop(&mut self) {
        let estimated_memory = (self.pending.capacity() * std::mem::size_of::<AlignedMessage>() 
            + self.watermarks.len() * std::mem::size_of::<SequenceWatermark>()) as u64;
        ALIGNMENT_MEMORY_COUNTER.fetch_sub(estimated_memory, Ordering::Relaxed);
    }
}

/// Alignment result enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlignResult {
    AcceptedAndReady,
    AcceptedWaiting,
    Duplicate,
    InvalidSymbol,
    LargeGapDropped,
}

/// Buffer statistics
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub pending_count: usize,
    pub total_dropped: u64,
    pub watermarks: Vec<WatermarkStatus>,
}

/// Temporal alignment engine combining multiple streams
pub struct TemporalAlignmentEngine {
    message_buffer: AlignedMessageBuffer,
    /// Global time watermark (nanoseconds)
    global_time_watermark: AtomicU64,
    /// Maximum age for messages (nanoseconds)
    max_message_age_ns: u64,
}

impl TemporalAlignmentEngine {
    /// Create new temporal alignment engine
    pub fn new(n_symbols: usize, max_message_age_ms: u64) -> Result<Self, &'static str> {
        let buffer = AlignedMessageBuffer::new(n_symbols, 10000)?;
        
        Ok(Self {
            message_buffer: buffer,
            global_time_watermark: AtomicU64::new(0),
            max_message_age_ns: max_message_age_ms * 1_000_000,
        })
    }
    
    /// Ingest L2 update with alignment
    pub fn ingest_l2_update(
        &mut self,
        timestamp_ns: u64,
        sequence: u64,
        symbol_id: u32,
        payload_hash: u64,
    ) -> AlignResult {
        let msg = AlignedMessage {
            msg_type: MessageType::L2Update,
            timestamp_ns,
            sequence,
            symbol_id,
            payload_hash,
            processed: AtomicBool::new(false),
        };
        
        self.message_buffer.add_message(msg)
    }
    
    /// Ingest trade with alignment
    pub fn ingest_trade(
        &mut self,
        timestamp_ns: u64,
        sequence: u64,
        symbol_id: u32,
        payload_hash: u64,
    ) -> AlignResult {
        let msg = AlignedMessage {
            msg_type: MessageType::Trade,
            timestamp_ns,
            sequence,
            symbol_id,
            payload_hash,
            processed: AtomicBool::new(false),
        };
        
        self.message_buffer.add_message(msg)
    }
    
    /// Get aligned messages ready for processing
    pub fn get_aligned_messages(&mut self, symbol_id: u32) -> Vec<AlignedMessage> {
        self.message_buffer.drain_ready(symbol_id)
    }
    
    /// Purge stale messages older than max age
    pub fn purge_stale(&mut self, current_time_ns: u64) -> usize {
        let cutoff = current_time_ns.saturating_sub(self.max_message_age_ns);
        let initial_len = self.message_buffer.pending.len();
        
        self.message_buffer.pending.retain(|msg| msg.timestamp_ns >= cutoff);
        
        initial_len - self.message_buffer.pending.len()
    }
    
    /// Update global time watermark
    pub fn update_global_watermark(&self, timestamp_ns: u64) {
        self.global_time_watermark.store(timestamp_ns, Ordering::Relaxed);
    }
    
    /// Get engine statistics
    pub fn get_statistics(&self) -> AlignmentStats {
        let buffer_stats = self.message_buffer.get_stats();
        
        AlignmentStats {
            pending_count: buffer_stats.pending_count,
            total_dropped: buffer_stats.total_dropped,
            global_watermark_ns: self.global_time_watermark.load(Ordering::Relaxed),
            max_message_age_ns: self.max_message_age_ns,
        }
    }
}

/// Engine statistics
#[derive(Debug, Clone)]
pub struct AlignmentStats {
    pub pending_count: usize,
    pub total_dropped: u64,
    pub global_watermark_ns: u64,
    pub max_message_age_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sequence_watermark() {
        let wm = SequenceWatermark::new(100, 10);
        
        assert_eq!(wm.check_sequence(101), SequenceStatus::Valid);
        assert_eq!(wm.check_sequence(100), SequenceStatus::Duplicate);
        assert_eq!(wm.check_sequence(105), SequenceStatus::Gap);
        assert_eq!(wm.check_sequence(115), SequenceStatus::LargeGap);
        
        wm.advance(101);
        assert_eq!(wm.check_sequence(102), SequenceStatus::Valid);
    }
    
    #[test]
    fn test_message_buffer() {
        let mut buffer = AlignedMessageBuffer::new(5, 100).unwrap();
        
        let msg1 = AlignedMessage {
            msg_type: MessageType::L2Update,
            timestamp_ns: 1000,
            sequence: 0,
            symbol_id: 0,
            payload_hash: 12345,
            processed: AtomicBool::new(false),
        };
        
        let result = buffer.add_message(msg1);
        assert_eq!(result, AlignResult::AcceptedAndReady);
        
        let ready = buffer.get_next_ready(0);
        assert!(ready.is_some());
        
        buffer.mark_processed(0, 0);
        assert!(buffer.get_next_ready(0).is_none());
    }
}
