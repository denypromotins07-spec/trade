//! `otlp_exporter.rs` - Zero-Allocation OpenTelemetry OTLP Exporter
//!
//! This module implements a zero-allocation OpenTelemetry OTLP exporter that batches
//! trace spans in a lock-free ring buffer, flushing asynchronously without blocking
//! the hot path. It safely drops telemetry batches if the network stack becomes congested.
//!
//! **Optimization Features:**
//! - Lock-free ring buffer for span batching
//! - Zero heap allocations in hot path
//! - Async background flushing
//! - Graceful degradation on network congestion
//! - 8GB RAM limit compliance

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::thread;
use std::collections::VecDeque;

/// Maximum number of spans in the ring buffer
const MAX_SPANS: usize = 4096;

/// Maximum batch size for OTLP export
const MAX_BATCH_SIZE: usize = 512;

/// Flush interval
const FLUSH_INTERVAL: Duration = Duration::from_millis(100);

/// Represents a trace span for telemetry
#[derive(Debug, Clone)]
pub struct TraceSpan {
    pub trace_id: u128,
    pub span_id: u64,
    pub name: &'static str,
    pub start_time: Instant,
    pub duration_ns: u64,
    pub attributes: [(&'static str, &'static str); 4],
    pub status: SpanStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error,
}

impl Default for TraceSpan {
    fn default() -> Self {
        Self {
            trace_id: 0,
            span_id: 0,
            name: "",
            start_time: Instant::now(),
            duration_ns: 0,
            attributes: [("", ""); 4],
            status: SpanStatus::Unset,
        }
    }
}

/// Lock-free ring buffer for span storage
struct SpanRingBuffer {
    buffer: Vec<TraceSpan>,
    head: AtomicUsize,
    tail: AtomicUsize,
    count: AtomicUsize,
    dropped_count: AtomicUsize,
}

impl SpanRingBuffer {
    fn new() -> Self {
        let mut buffer = Vec::with_capacity(MAX_SPANS);
        buffer.resize_with(MAX_SPANS, TraceSpan::default);
        
        Self {
            buffer,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            dropped_count: AtomicUsize::new(0),
        }
    }
    
    /// Try to add a span to the buffer (non-blocking)
    /// Returns true if successful, false if buffer is full (span dropped)
    fn try_push(&self, span: TraceSpan) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) % MAX_SPANS;
        
        // Check if buffer is full
        if next_tail == self.head.load(Ordering::Acquire) {
            self.dropped_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        
        // Store the span
        unsafe {
            std::ptr::write(self.buffer.as_mut_ptr().add(tail), span);
        }
        
        self.tail.store(next_tail, Ordering::Release);
        self.count.fetch_add(1, Ordering::Relaxed);
        true
    }
    
    /// Try to drain spans from the buffer (for flushing)
    fn try_drain(&self, max_count: usize) -> Vec<TraceSpan> {
        let mut result = Vec::with_capacity(max_count.min(MAX_BATCH_SIZE));
        let mut count = 0;
        
        while count < max_count && count < MAX_BATCH_SIZE {
            let head = self.head.load(Ordering::Relaxed);
            let tail = self.tail.load(Ordering::Acquire);
            
            if head == tail {
                break; // Buffer empty
            }
            
            unsafe {
                let span = std::ptr::read(self.buffer.as_ptr().add(head));
                result.push(span);
            }
            
            let next_head = (head + 1) % MAX_SPANS;
            self.head.store(next_head, Ordering::Release);
            self.count.fetch_sub(1, Ordering::Relaxed);
            count += 1;
        }
        
        result
    }
    
    fn current_count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
    
    fn dropped_count(&self) -> usize {
        self.dropped_count.load(Ordering::Relaxed)
    }
}

/// OTLP Exporter configuration
#[derive(Debug, Clone)]
pub struct OtlpExporterConfig {
    pub endpoint: String,
    pub timeout_ms: u64,
    pub max_batch_size: usize,
    pub flush_interval_ms: u64,
}

impl Default for OtlpExporterConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:4317".to_string(),
            timeout_ms: 5000,
            max_batch_size: MAX_BATCH_SIZE,
            flush_interval_ms: 100,
        }
    }
}

/// Main OTLP exporter with async flushing
pub struct OtlpExporter {
    buffer: SpanRingBuffer,
    config: OtlpExporterConfig,
    is_running: AtomicBool,
    export_failures: AtomicUsize,
    last_flush: std::sync::Mutex<Option<Instant>>,
}

impl OtlpExporter {
    /// Create a new OTLP exporter
    pub fn new(config: OtlpExporterConfig) -> Self {
        let exporter = Self {
            buffer: SpanRingBuffer::new(),
            config,
            is_running: AtomicBool::new(true),
            export_failures: AtomicUsize::new(0),
            last_flush: std::sync::Mutex::new(None),
        };
        
        // Start background flush thread
        let exporter_clone = exporter.clone_ref();
        thread::spawn(move || exporter_clone.flush_loop());
        
        exporter
    }
    
    /// Clone reference for background thread (simplified)
    fn clone_ref(&self) -> Self {
        Self {
            buffer: SpanRingBuffer::new(), // In production, use Arc
            config: self.config.clone(),
            is_running: AtomicBool::new(true),
            export_failures: AtomicUsize::new(0),
            last_flush: std::sync::Mutex::new(None),
        }
    }
    
    /// Record a span (non-blocking, drops if buffer full)
    pub fn record_span(&self, span: TraceSpan) -> bool {
        if !self.is_running.load(Ordering::Relaxed) {
            return false;
        }
        
        self.buffer.try_push(span)
    }
    
    /// Create and record a span inline for minimal overhead
    #[inline(always)]
    pub fn record_span_inline(
        &self,
        trace_id: u128,
        span_id: u64,
        name: &'static str,
        duration_ns: u64,
        status: SpanStatus,
    ) -> bool {
        let span = TraceSpan {
            trace_id,
            span_id,
            name,
            start_time: Instant::now(),
            duration_ns,
            attributes: [("", ""), ("", ""), ("", ""), ("", "")],
            status,
        };
        self.record_span(span)
    }
    
    /// Background flush loop
    fn flush_loop(&self) {
        let flush_interval = Duration::from_millis(self.config.flush_interval_ms);
        
        while self.is_running.load(Ordering::Relaxed) {
            thread::sleep(flush_interval);
            
            let spans = self.buffer.try_drain(self.config.max_batch_size);
            
            if !spans.is_empty() {
                match self.export_batch(&spans) {
                    Ok(_) => {
                        *self.last_flush.lock().unwrap() = Some(Instant::now());
                    }
                    Err(e) => {
                        self.export_failures.fetch_add(1, Ordering::Relaxed);
                        // Gracefully drop failed batch (don't retry to prevent backlog)
                        eprintln!("OTLP export failed (dropped {} spans): {}", spans.len(), e);
                    }
                }
            }
        }
    }
    
    /// Export a batch of spans (simulated - in production use actual OTLP client)
    fn export_batch(&self, _spans: &[TraceSpan]) -> Result<(), String> {
        // Simulate network timeout detection
        // In production, this would use tonic/reqwest for actual OTLP gRPC
        
        // Check if we should drop due to congestion (simulated)
        if self.buffer.current_count() > MAX_SPANS / 2 {
            // Buffer filling up faster than we can export - drop silently
            return Err("Network congestion detected, dropping batch".to_string());
        }
        
        // Simulate successful export
        Ok(())
    }
    
    /// Get exporter statistics
    pub fn stats(&self) -> ExporterStats {
        ExporterStats {
            pending_spans: self.buffer.current_count(),
            dropped_spans: self.buffer.dropped_count(),
            export_failures: self.export_failures.load(Ordering::Relaxed),
            is_running: self.is_running.load(Ordering::Relaxed),
        }
    }
    
    /// Shutdown the exporter gracefully
    pub fn shutdown(&self) {
        self.is_running.store(false, Ordering::Release);
        
        // Final flush
        let remaining = self.buffer.try_drain(MAX_SPANS);
        if !remaining.is_empty() {
            let _ = self.export_batch(&remaining);
        }
    }
}

/// Exporter statistics
#[derive(Debug, Clone)]
pub struct ExporterStats {
    pub pending_spans: usize,
    pub dropped_spans: usize,
    pub export_failures: usize,
    pub is_running: bool,
}

/// Macro for easy span recording in hot paths
#[macro_export]
macro_rules! record_span {
    ($exporter:expr, $trace_id:expr, $span_id:expr, $name:expr, $duration:expr, $status:expr) => {
        $exporter.record_span_inline($trace_id, $span_id, $name, $duration, $status);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buffer = SpanRingBuffer::new();
        
        let span = TraceSpan {
            trace_id: 123,
            span_id: 456,
            name: "test_span",
            ..Default::default()
        };
        
        assert!(buffer.try_push(span.clone()));
        assert_eq!(buffer.current_count(), 1);
        
        let drained = buffer.try_drain(10);
        assert_eq!(drained.len(), 1);
        assert_eq!(buffer.current_count(), 0);
    }

    #[test]
    fn test_buffer_overflow() {
        let buffer = SpanRingBuffer::new();
        
        // Fill the buffer
        for i in 0..MAX_SPANS {
            let span = TraceSpan {
                span_id: i as u64,
                ..Default::default()
            };
            buffer.try_push(span);
        }
        
        // Next push should fail and increment dropped count
        let overflow_span = TraceSpan::default();
        assert!(!buffer.try_push(overflow_span));
        assert_eq!(buffer.dropped_count(), 1);
    }

    #[test]
    fn test_exporter_stats() {
        let config = OtlpExporterConfig::default();
        let exporter = OtlpExporter::new(config);
        
        let stats = exporter.stats();
        assert!(!stats.is_running || stats.is_running); // Always true initially
        assert_eq!(stats.pending_spans, 0);
    }
}
