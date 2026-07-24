//! `src/execution/fill_propagator.rs`
//!
//! **Lock-Free Fill Propagator**
//! Instantly updates the global portfolio state the microsecond any single asset engine
//! receives a partial or full execution report.
//!
//! **Architecture:**
//! - Uses a lock-free ring buffer to propagate fill events from execution threads to risk threads.
//! - Zero heap allocation in the hot path.
//! - Ensures strict consistency: Risk engine sees fills in the order they occurred (per symbol).

use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};

/// Maximum number of pending fill events in the ring buffer.
/// Sized to handle extreme volatility bursts without overflow.
const FILL_BUFFER_SIZE: usize = 4096;

/// Execution side enum.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillSide {
    Buy = 0,
    Sell = 1,
}

/// Represents a fill event from the exchange.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FillEvent {
    pub symbol_id: u8,
    pub side: u8, // FillSide encoded
    pub fill_price: i64, // Fixed point
    pub fill_quantity: i64, // Fixed point
    pub commission: i64, // Fixed point
    pub trade_id: u64,
    pub timestamp_ns: u64,
    pub is_maker: bool,
}

impl Default for FillEvent {
    fn default() -> Self {
        Self {
            symbol_id: 0,
            side: 0,
            fill_price: 0,
            fill_quantity: 0,
            commission: 0,
            trade_id: 0,
            timestamp_ns: 0,
            is_maker: false,
        }
    }
}

/// The Fill Propagator engine.
/// Lock-free ring buffer for high-throughput fill distribution.
pub struct FillPropagator {
    /// Ring buffer storage.
    buffer: [FillEvent; FILL_BUFFER_SIZE],
    /// Write index (producer).
    write_index: AtomicUsize,
    /// Read index (consumer).
    read_index: AtomicUsize,
    /// Total fills processed (for metrics).
    total_fills: AtomicU64,
}

unsafe impl Send for FillPropagator {}
unsafe impl Sync for FillPropagator {}

impl FillPropagator {
    pub fn new() -> Self {
        Self {
            buffer: [FillEvent::default(); FILL_BUFFER_SIZE],
            write_index: AtomicUsize::new(0),
            read_index: AtomicUsize::new(0),
            total_fills: AtomicU64::new(0),
        }
    }

    /// Pushes a new fill event into the buffer.
    /// Called by execution threads upon receiving an ACK from Binance.
    /// 
    /// Returns `true` if successful, `false` if buffer is full (overflow).
    pub fn push_fill(&self, fill: FillEvent) -> bool {
        let write = self.write_index.load(Ordering::Relaxed);
        let read = self.read_index.load(Ordering::Acquire);
        
        // Check for overflow (buffer full)
        let next_write = (write + 1) % FILL_BUFFER_SIZE;
        if next_write == read {
            // Buffer full - critical situation
            // In production: Trigger backpressure or emergency log
            return false;
        }

        self.buffer[write] = fill;
        self.write_index.store(next_write, Ordering::Release);
        self.total_fills.fetch_add(1, Ordering::Relaxed);

        true
    }

    /// Consumes the next available fill event.
    /// Called by the risk/portfolio update thread.
    /// 
    /// Returns `None` if no fills are pending.
    pub fn pop_fill(&self) -> Option<FillEvent> {
        let read = self.read_index.load(Ordering::Relaxed);
        let write = self.write_index.load(Ordering::Acquire);

        if read == write {
            return None; // Buffer empty
        }

        let fill = self.buffer[read];
        let next_read = (read + 1) % FILL_BUFFER_SIZE;
        self.read_index.store(next_read, Ordering::Release);

        Some(fill)
    }

    /// Drains all pending fills into a vector.
    /// More efficient than repeated `pop_fill` calls for batch processing.
    pub fn drain_fills(&self) -> Vec<FillEvent> {
        let mut fills = Vec::with_capacity(FILL_BUFFER_SIZE);
        
        while let Some(fill) = self.pop_fill() {
            fills.push(fill);
        }
        
        fills
    }

    /// Returns the current number of pending fills.
    pub fn pending_count(&self) -> usize {
        let write = self.write_index.load(Ordering::Relaxed);
        let read = self.read_index.load(Ordering::Relaxed);
        
        if write >= read {
            write - read
        } else {
            FILL_BUFFER_SIZE - read + write
        }
    }

    /// Returns total fills processed since inception.
    pub fn total_processed(&self) -> u64 {
        self.total_fills.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_pop() {
        let propagator = FillPropagator::new();
        
        let fill = FillEvent {
            symbol_id: 1,
            side: FillSide::Buy as u8,
            fill_price: 50000_000_000,
            fill_quantity: 100_000_000,
            commission: 10_000,
            trade_id: 999,
            timestamp_ns: 1234567890,
            is_maker: true,
        };

        assert!(propagator.push_fill(fill));
        assert_eq!(propagator.pending_count(), 1);

        let popped = propagator.pop_fill();
        assert!(popped.is_some());
        let popped = popped.unwrap();
        assert_eq!(popped.trade_id, 999);
        assert_eq!(propagator.pending_count(), 0);
    }

    #[test]
    fn test_drain() {
        let propagator = FillPropagator::new();
        
        for i in 0..10 {
            let fill = FillEvent {
                symbol_id: 1,
                side: 0,
                fill_price: 100,
                fill_quantity: 100,
                commission: 1,
                trade_id: i,
                timestamp_ns: i as u64,
                is_maker: false,
            };
            propagator.push_fill(fill);
        }

        let drains = propagator.drain_fills();
        assert_eq!(drains.len(), 10);
        assert_eq!(propagator.pending_count(), 0);
    }
}
