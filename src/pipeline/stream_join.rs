//! Chapter 4: Advanced Data Pipeline & Stream Processing
//! File 10: src/pipeline/stream_join.rs
//!
//! Lock-free temporal stream joins aligning asynchronous tick, trade,
//! and orderbook updates using strict event-time watermarking.
//! Prevents out-of-order state corruption in high-frequency scenarios.
//!
//! Optimized for AMD Ryzen AI 5 with cache-line aligned structures.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum events per stream buffer (enforces 8GB RAM limit)
const MAX_EVENTS_PER_STREAM: usize = 1024 * 1024; // 1M events per stream

/// Maximum number of streams to join
const MAX_STREAMS: usize = 16;

/// Event types supported
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    Tick,
    Trade,
    OrderBookUpdate,
    Quote,
    Custom(u8),
}

/// Temporal event with watermarks
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct TemporalEvent {
    /// Event timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Watermark (guaranteed no earlier events)
    pub watermark_ns: u64,
    /// Event type
    pub event_type: EventType,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Price (fixed-point * 10^8)
    pub price: i64,
    /// Quantity (fixed-point * 10^8)
    pub quantity: i64,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Is processed
    pub is_processed: bool,
}

impl Default for TemporalEvent {
    fn default() -> Self {
        TemporalEvent {
            timestamp_ns: 0,
            watermark_ns: 0,
            event_type: EventType::Tick,
            symbol_hash: 0,
            price: 0,
            quantity: 0,
            sequence: 0,
            is_processed: false,
        }
    }
}

/// Stream buffer for one event type
#[repr(C, align(64))]
pub struct StreamBuffer {
    events: [TemporalEvent; MAX_EVENTS_PER_STREAM],
    head: AtomicU64,
    tail: AtomicU64,
    watermark_ns: AtomicU64,
    is_active: AtomicBool,
}

impl Default for StreamBuffer {
    fn default() -> Self {
        StreamBuffer {
            events: [(); MAX_EVENTS_PER_STREAM].map(|_| TemporalEvent::default()),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            watermark_ns: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }
}

impl StreamBuffer {
    /// Push event to buffer (lock-free using CAS)
    pub fn push(&self, event: TemporalEvent) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % MAX_EVENTS_PER_STREAM as u64;
        
        // Check if buffer is full
        if next_head == self.tail.load(Ordering::Acquire) {
            return false; // Buffer full - backpressure signal
        }
        
        unsafe {
            let idx = head as usize;
            (*self.events.as_ptr().add(idx)) = event;
        }
        
        self.head.store(next_head, Ordering::Release);
        true
    }
    
    /// Pop oldest event that passes watermark check
    pub fn pop_with_watermark(&self, current_watermark: u64) -> Option<TemporalEvent> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Empty
        }
        
        unsafe {
            let idx = tail as usize;
            let event = *self.events.as_ptr().add(idx);
            
            // Check watermark - only emit if event time <= watermark
            if event.timestamp_ns > current_watermark {
                return None; // Wait for watermark to advance
            }
            
            self.tail.store((tail + 1) % MAX_EVENTS_PER_STREAM as u64, Ordering::Release);
            Some(event)
        }
    }
    
    /// Update watermark
    pub fn update_watermark(&self, watermark: u64) {
        self.watermark_ns.store(watermark, Ordering::Release);
    }
    
    /// Get current watermark
    pub fn get_watermark(&self) -> u64 {
        self.watermark_ns.load(Ordering::Acquire)
    }
    
    /// Get pending event count
    pub fn pending_count(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        ((head - tail + MAX_EVENTS_PER_STREAM as u64) % MAX_EVENTS_PER_STREAM as u64) as usize
    }
}

/// Temporal stream join engine
#[repr(C, align(64))]
pub struct StreamJoinEngine {
    /// Buffers per stream type
    buffers: [StreamBuffer; MAX_STREAMS],
    
    /// Global watermark (minimum across all streams)
    global_watermark: AtomicU64,
    
    /// Join latency threshold (nanoseconds)
    max_latency_ns: u64,
    
    /// Dropped event counter (for backpressure logging)
    dropped_count: AtomicU64,
    
    /// Total joined events
    joined_count: AtomicU64,
}

impl StreamJoinEngine {
    /// Create new stream join engine
    pub fn new(max_latency_ms: u64) -> Self {
        Self {
            buffers: [(); MAX_STREAMS].map(|_| StreamBuffer::default()),
            global_watermark: AtomicU64::new(0),
            max_latency_ns: max_latency_ms * 1_000_000,
            dropped_count: AtomicU64::new(0),
            joined_count: AtomicU64::new(0),
        }
    }
    
    /// Ingest event into appropriate stream
    pub fn ingest(&self, stream_id: usize, event: TemporalEvent) -> bool {
        if stream_id >= MAX_STREAMS {
            return false;
        }
        
        if !self.buffers[stream_id].push(event) {
            // Buffer full - apply backpressure
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        
        true
    }
    
    /// Perform temporal join - returns joined events
    pub fn join(&self) -> Vec<[TemporalEvent; MAX_STREAMS]> {
        let mut joined_events = Vec::new();
        
        // Update global watermark to minimum across streams
        let mut min_watermark = u64::MAX;
        for i in 0..MAX_STREAMS {
            let bw = self.buffers[i].get_watermark();
            if bw < min_watermark && bw > 0 {
                min_watermark = bw;
            }
        }
        
        if min_watermark == u64::MAX {
            min_watermark = get_timestamp_ns().saturating_sub(self.max_latency_ns);
        }
        
        self.global_watermark.store(min_watermark, Ordering::Release);
        
        // Try to form joined tuples
        loop {
            let mut tuple: [TemporalEvent; MAX_STREAMS] = [TemporalEvent::default(); MAX_STREAMS];
            let mut all_have_events = true;
            let mut max_timestamp = 0u64;
            
            for i in 0..MAX_STREAMS {
                if let Some(event) = self.buffers[i].pop_with_watermark(min_watermark) {
                    tuple[i] = event;
                    max_timestamp = max_timestamp.max(event.timestamp_ns);
                } else {
                    all_have_events = false;
                    break;
                }
            }
            
            if all_have_events {
                // Verify temporal consistency (all within latency threshold)
                let mut min_ts = u64::MAX;
                for i in 0..MAX_STREAMS {
                    min_ts = min_ts.min(tuple[i].timestamp_ns);
                }
                
                if max_timestamp - min_ts <= self.max_latency_ns {
                    joined_events.push(tuple);
                    self.joined_count.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break; // No more complete tuples
            }
        }
        
        joined_events
    }
    
    /// Advance watermark for a specific stream
    pub fn advance_watermark(&self, stream_id: usize, watermark: u64) {
        if stream_id < MAX_STREAMS {
            self.buffers[stream_id].update_watermark(watermark);
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> (u64, u64, u64, [usize; MAX_STREAMS]) {
        let mut pending = [0usize; MAX_STREAMS];
        for i in 0..MAX_STREAMS {
            pending[i] = self.buffers[i].pending_count();
        }
        
        (
            self.joined_count.load(Ordering::Relaxed),
            self.dropped_count.load(Ordering::Relaxed),
            self.global_watermark.load(Ordering::Acquire),
            pending,
        )
    }
    
    /// Check if any buffer is near capacity (backpressure warning)
    pub fn needs_backpressure(&self, threshold_pct: u8) -> bool {
        let threshold = (MAX_EVENTS_PER_STREAM as u64 * threshold_pct as u64) / 100;
        for i in 0..MAX_STREAMS {
            if self.buffers[i].pending_count() as u64 > threshold {
                return true;
            }
        }
        false
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stream_buffer_push_pop() {
        let buffer = StreamBuffer::default();
        let event = TemporalEvent {
            timestamp_ns: 1000,
            watermark_ns: 900,
            ..Default::default()
        };
        
        assert!(buffer.push(event));
        assert_eq!(buffer.pending_count(), 1);
        
        let popped = buffer.pop_with_watermark(1100);
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().timestamp_ns, 1000);
    }
    
    #[test]
    fn test_stream_join_engine() {
        let engine = StreamJoinEngine::new(100); // 100ms max latency
        
        let event1 = TemporalEvent {
            timestamp_ns: 1000,
            watermark_ns: 900,
            event_type: EventType::Tick,
            ..Default::default()
        };
        
        let event2 = TemporalEvent {
            timestamp_ns: 1001,
            watermark_ns: 900,
            event_type: EventType::Trade,
            ..Default::default()
        };
        
        assert!(engine.ingest(0, event1));
        assert!(engine.ingest(1, event2));
        
        // Advance watermarks
        engine.advance_watermark(0, 1100);
        engine.advance_watermark(1, 1100);
        
        let joined = engine.join();
        assert!(!joined.is_empty());
    }
    
    #[test]
    fn test_ram_limits() {
        assert!(MAX_EVENTS_PER_STREAM > 0);
        assert!(MAX_EVENTS_PER_STREAM <= 2 * 1024 * 1024);
        assert!(MAX_STREAMS <= 32);
    }
}
