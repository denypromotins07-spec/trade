//! Chapter 4: Advanced Data Pipeline & Stream Processing
//! File 12: src/pipeline/backpressure.rs
//!
//! Advanced backpressure mechanism that safely drops lowest-priority telemetry
//! data if the Rust event loop falls behind real-time during extreme volatility.
//! Logs dropped telemetry events directly to SOUL.md for post-mortem analysis.
//!
//! Optimized for AMD Ryzen AI 5 with lock-free ring buffers.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::fs::{OpenOptions, File};
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum telemetry events in buffer (enforces 8GB RAM limit)
const MAX_TELEMETRY_BUFFER: usize = 4 * 1024 * 1024; // 4M events

/// Priority levels for telemetry events
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum TelemetryPriority {
    Critical = 0,  // Never drop - execution fills, cancels
    High = 1,      // Rarely drop - order book updates
    Medium = 2,    // Sometimes drop - signals, features
    Low = 3,       // First to drop - debug info, metrics
}

/// Telemetry event structure
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct TelemetryEvent {
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Event type hash
    pub event_type: u64,
    /// Symbol hash
    pub symbol_hash: u64,
    /// Priority level
    pub priority: TelemetryPriority,
    /// Data payload (first 48 bytes inline)
    pub payload: [u8; 48],
    /// Payload length
    pub payload_len: u16,
    /// Is occupied
    pub is_occupied: bool,
}

impl Default for TelemetryEvent {
    fn default() -> Self {
        TelemetryEvent {
            timestamp_ns: 0,
            event_type: 0,
            symbol_hash: 0,
            priority: TelemetryPriority::Low,
            payload: [0; 48],
            payload_len: 0,
            is_occupied: false,
        }
    }
}

/// Backpressure statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct BackpressureStats {
    pub total_received: u64,
    pub total_dropped: u64,
    pub dropped_by_priority: [u64; 4],
    pub last_drop_timestamp_ns: u64,
    pub max_latency_ns: u64,
    pub current_latency_ns: u64,
}

/// Backpressure-aware telemetry buffer
#[repr(C, align(64))]
pub struct BackpressureBuffer {
    /// Ring buffer for events
    buffer: [TelemetryEvent; MAX_TELEMETRY_BUFFER],
    
    /// Head and tail pointers
    head: AtomicU64,
    tail: AtomicU64,
    
    /// Backpressure thresholds (percentage of buffer full)
    warning_threshold_pct: u8,
    drop_low_threshold_pct: u8,
    drop_medium_threshold_pct: u8,
    drop_high_threshold_pct: u8,
    
    /// Statistics
    stats: std::cell::RefCell<BackpressureStats>,
    
    /// Is backpressure active
    backpressure_active: AtomicBool,
    
    /// Log file handle (SOUL.md)
    log_file: std::cell::RefCell<Option<File>>,
}

impl BackpressureBuffer {
    /// Create new backpressure buffer
    pub fn new(
        warning_pct: u8,
        drop_low_pct: u8,
        drop_med_pct: u8,
        drop_high_pct: u8,
    ) -> Self {
        Self {
            buffer: [(); MAX_TELEMETRY_BUFFER].map(|_| TelemetryEvent::default()),
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            warning_threshold_pct: warning_pct,
            drop_low_threshold_pct: drop_low_pct,
            drop_medium_threshold_pct: drop_med_pct,
            drop_high_threshold_pct: drop_high_pct,
            stats: std::cell::RefCell::new(BackpressureStats::default()),
            backpressure_active: AtomicBool::new(false),
            log_file: std::cell::RefCell::new(None),
        }
    }
    
    /// Initialize logging to SOUL.md
    pub fn init_soul_log(&self, path: &str) -> bool {
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(file) => {
                *self.log_file.borrow_mut() = Some(file);
                true
            }
            Err(_) => false,
        }
    }
    
    /// Push telemetry event with backpressure handling
    pub fn push(&self, mut event: TelemetryEvent) -> Result<bool, &'static str> {
        let mut stats = self.stats.borrow_mut();
        stats.total_received += 1;
        
        // Calculate current latency
        let now = get_timestamp_ns();
        let latency = now.saturating_sub(event.timestamp_ns);
        stats.current_latency_ns = latency;
        stats.max_latency_ns = stats.max_latency_ns.max(latency);
        
        // Check buffer utilization
        let utilization = self.get_utilization_pct();
        
        // Determine if we should drop based on priority and utilization
        let should_drop = match event.priority {
            TelemetryPriority::Critical => false, // Never drop critical
            TelemetryPriority::Low => utilization >= self.drop_low_threshold_pct,
            TelemetryPriority::Medium => utilization >= self.drop_medium_threshold_pct,
            TelemetryPriority::High => utilization >= self.drop_high_threshold_pct,
        };
        
        if should_drop {
            // Drop this event
            stats.total_dropped += 1;
            stats.dropped_by_priority[event.priority as usize] += 1;
            stats.last_drop_timestamp_ns = now;
            
            self.backpressure_active.store(true, Ordering::Relaxed);
            
            // Log to SOUL.md
            self.log_dropped_event(&event);
            
            return Ok(false); // Dropped
        }
        
        // Try to enqueue
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % MAX_TELEMETRY_BUFFER as u64;
        
        if next_head == self.tail.load(Ordering::Acquire) {
            // Buffer full - must drop even if low priority passed check
            stats.total_dropped += 1;
            stats.dropped_by_priority[event.priority as usize] += 1;
            stats.last_drop_timestamp_ns = now;
            
            self.log_dropped_event(&event);
            return Err("BUFFER_FULL");
        }
        
        event.is_occupied = true;
        unsafe {
            let idx = head as usize;
            (*self.buffer.as_mut_ptr().add(idx)) = event;
        }
        
        self.head.store(next_head, Ordering::Release);
        
        // Update backpressure status
        let new_util = self.get_utilization_pct();
        self.backpressure_active.store(
            new_util >= self.warning_threshold_pct,
            Ordering::Relaxed,
        );
        
        Ok(true) // Enqueued successfully
    }
    
    /// Pop oldest event for processing
    pub fn pop(&self) -> Option<TelemetryEvent> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        
        if tail == head {
            return None; // Empty
        }
        
        unsafe {
            let idx = tail as usize;
            let event = *self.buffer.as_ptr().add(idx);
            
            if !event.is_occupied {
                return None;
            }
            
            // Clear slot
            (*self.buffer.as_mut_ptr().add(idx)).is_occupied = false;
        }
        
        self.tail.store((tail + 1) % MAX_TELEMETRY_BUFFER as u64, Ordering::Release);
        Some(event)
    }
    
    /// Get current buffer utilization percentage
    pub fn get_utilization_pct(&self) -> u8 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        let used = if head >= tail {
            head - tail
        } else {
            MAX_TELEMETRY_BUFFER as u64 - tail + head
        };
        
        ((used * 100) / MAX_TELEMETRY_BUFFER as u64) as u8
    }
    
    /// Check if backpressure is active
    pub fn is_backpressure_active(&self) -> bool {
        self.backpressure_active.load(Ordering::Relaxed)
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> BackpressureStats {
        self.stats.borrow().clone()
    }
    
    /// Log dropped event to SOUL.md
    fn log_dropped_event(&self, event: &TelemetryEvent) {
        let mut log_ref = self.log_file.borrow_mut();
        if let Some(ref mut file) = *log_ref {
            let timestamp = event.timestamp_ns;
            let priority = event.priority as u8;
            let _ = writeln!(
                file,
                "[BACKPRESSURE_DROP] ts={} priority={} type={} symbol={}",
                timestamp, priority, event.event_type, event.symbol_hash
            );
            let _ = file.flush();
        }
    }
    
    /// Emergency drain - clear buffer keeping only critical events
    pub fn emergency_drain(&self) -> usize {
        let mut drained = 0;
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        
        let mut current = tail;
        while current != head {
            unsafe {
                let idx = current as usize;
                let event = &mut *self.buffer.as_mut_ptr().add(idx);
                
                if event.is_occupied && event.priority != TelemetryPriority::Critical {
                    event.is_occupied = false;
                    drained += 1;
                    
                    // Log the drop
                    let dropped_event = *event;
                    self.log_dropped_event(&dropped_event);
                }
            }
            current = (current + 1) % MAX_TELEMETRY_BUFFER as u64;
        }
        
        if drained > 0 {
            self.stats.borrow_mut().total_dropped += drained as u64;
        }
        
        drained
    }
}

/// Get timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Convenience function to create telemetry event
pub fn make_telemetry(
    event_type: u64,
    symbol: u64,
    priority: TelemetryPriority,
    payload: &[u8],
) -> TelemetryEvent {
    let mut event = TelemetryEvent::default();
    event.timestamp_ns = get_timestamp_ns();
    event.event_type = event_type;
    event.symbol_hash = symbol;
    event.priority = priority;
    event.payload_len = payload.len().min(48) as u16;
    
    let copy_len = event.payload_len as usize;
    event.payload[..copy_len].copy_from_slice(&payload[..copy_len]);
    event.is_occupied = true;
    
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_push_pop() {
        let buffer = BackpressureBuffer::new(70, 80, 90, 95);
        
        let event = make_telemetry(1, 12345, TelemetryPriority::Medium, b"test data");
        assert!(buffer.push(event).unwrap());
        
        let popped = buffer.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().event_type, 1);
    }
    
    #[test]
    fn test_backpressure_dropping() {
        let buffer = BackpressureBuffer::new(50, 60, 80, 90);
        
        // Fill with low priority events
        for i in 0..MAX_TELEMETRY_BUFFER * 70 / 100 {
            let event = make_telemetry(i as u64, 1, TelemetryPriority::Low, b"low");
            let _ = buffer.push(event);
        }
        
        // Now try medium priority - should be dropped
        let med_event = make_telemetry(999, 1, TelemetryPriority::Medium, b"med");
        let result = buffer.push(med_event);
        assert!(!result.unwrap()); // Should be dropped
        
        // Critical should still succeed
        let crit_event = make_telemetry(999, 1, TelemetryPriority::Critical, b"crit");
        assert!(buffer.push(crit_event).is_ok());
    }
    
    #[test]
    fn test_ram_limits() {
        assert!(MAX_TELEMETRY_BUFFER > 0);
        assert!(MAX_TELEMETRY_BUFFER <= 8 * 1024 * 1024);
    }
}
