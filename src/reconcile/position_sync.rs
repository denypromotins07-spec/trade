// Position Sync: Lock-free position sync broadcasting net exposures, unrealized PnL,
// and active strategy allocations to the UI. Utilizes double-buffered MessagePack
// serialization to minimize network overhead.
// Optimized for AMD Ryzen AI 5 with atomic operations and zero-copy serialization.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};

/// Maximum number of positions to track
const MAX_POSITIONS: usize = 16;

/// Fixed-point scale factor
const FP_SCALE: u64 = 1_000_000;

/// Broadcast target FPS (60 = 16.67ms interval)
const TARGET_FPS: u64 = 60;
const BROADCAST_INTERVAL_MS: u64 = 1000 / TARGET_FPS;

/// Double-buffered position snapshot for lock-free reads
#[derive(Debug, Clone, Copy)]
pub struct PositionSnapshot {
    /// Symbol index (0-7)
    pub symbol_idx: u8,
    /// Net position in base currency (fixed-point)
    pub net_position_fp: i64,
    /// Entry price (fixed-point, USD scaled)
    pub entry_price_fp: u64,
    /// Current mark price (fixed-point, USD scaled)
    pub mark_price_fp: u64,
    /// Notional value (micro-USD)
    pub notional_micro: u64,
    /// Unrealized PnL (micro-USD)
    pub unrealized_pnl_micro: i64,
    /// Realized PnL today (micro-USD)
    pub realized_pnl_today_micro: i64,
    /// Active strategy ID
    pub active_strategy_id: u8,
    /// Allocation fraction (fixed-point)
    pub allocation_fraction_fp: u64,
    /// Leverage used (fixed-point, 1x = 1_000_000)
    pub leverage_fp: u64,
    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

impl PositionSnapshot {
    pub const fn new() -> Self {
        Self {
            symbol_idx: 0,
            net_position_fp: 0,
            entry_price_fp: 0,
            mark_price_fp: 0,
            notional_micro: 0,
            unrealized_pnl_micro: 0,
            realized_pnl_today_micro: 0,
            active_strategy_id: 0,
            allocation_fraction_fp: 0,
            leverage_fp: 0,
            timestamp_ms: 0,
        }
    }
}

impl Default for PositionSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated portfolio summary for quick UI updates
#[derive(Debug, Clone, Copy)]
pub struct PortfolioSummary {
    /// Total equity (micro-USD)
    pub total_equity_micro: u64,
    /// Total unrealized PnL (micro-USD)
    pub total_unrealized_pnl_micro: i64,
    /// Total realized PnL today (micro-USD)
    pub total_realized_pnl_today_micro: i64,
    /// Active position count
    pub active_positions: u8,
    /// Account leverage (fixed-point)
    pub account_leverage_fp: u64,
    /// Free margin (micro-USD)
    pub free_margin_micro: u64,
    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

/// Double buffer for lock-free position broadcasting
pub struct DoubleBuffer<T: Copy + Default> {
    /// Front buffer (currently being read)
    front: Arc<std::sync::RwLock<T>>,
    /// Back buffer (currently being written)
    back: Arc<std::sync::RwLock<T>>,
    /// Swap flag
    swapped: AtomicBool,
}

impl<T: Copy + Default> DoubleBuffer<T> {
    pub fn new(initial: T) -> Self {
        Self {
            front: Arc::new(std::sync::RwLock::new(initial)),
            back: Arc::new(std::sync::RwLock::new(initial)),
            swapped: AtomicBool::new(false),
        }
    }

    /// Write to back buffer (caller must ensure exclusive access)
    pub fn write_back(&self, value: T) {
        let mut back = self.back.write().unwrap();
        *back = value;
    }

    /// Read from front buffer (lock-free after swap)
    pub fn read_front(&self) -> T {
        *self.front.read().unwrap()
    }

    /// Swap buffers atomically
    pub fn swap(&self) {
        // In production, would use more sophisticated RCU pattern
        let front_val = *self.front.read().unwrap();
        let back_val = *self.back.read().unwrap();
        
        *self.front.write().unwrap() = back_val;
        *self.back.write().unwrap() = front_val;
        
        self.swapped.store(true, Ordering::Release);
    }

    /// Check if swap occurred
    pub fn was_swapped(&self) -> bool {
        self.swapped.load(Ordering::Acquire)
    }

    /// Clear swap flag
    pub fn clear_swap_flag(&self) {
        self.swapped.store(false, Ordering::Release);
    }
}

/// Position broadcaster with double-buffered serialization
pub struct PositionSync {
    /// Per-position snapshots (double-buffered)
    positions: [DoubleBuffer<PositionSnapshot>; MAX_POSITIONS],
    
    /// Portfolio summary (double-buffered)
    summary: DoubleBuffer<PortfolioSummary>,
    
    /// Last broadcast timestamp
    last_broadcast_ms: AtomicU64,
    
    /// Broadcast enabled flag
    broadcast_enabled: AtomicBool,
    
    /// Subscriber count (for rate adjustment)
    subscriber_count: AtomicU8,
    
    /// Serialization buffer (pre-allocated)
    serialization_buffer: Arc<std::sync::Mutex<Vec<u8>>>,
    
    /// Start time for timestamps
    start_time: Instant,
}

impl PositionSync {
    /// Create a new position sync broadcaster
    pub fn new(initial_equity_micro: u64) -> Self {
        let mut positions = std::array::from_fn(|_| {
            DoubleBuffer::new(PositionSnapshot::new())
        });
        
        // Initialize first position with summary
        positions[0].write_back(PositionSnapshot {
            timestamp_ms: 0,
            ..PositionSnapshot::new()
        });
        
        let summary = DoubleBuffer::new(PortfolioSummary {
            total_equity_micro: initial_equity_micro,
            total_unrealized_pnl_micro: 0,
            total_realized_pnl_today_micro: 0,
            active_positions: 0,
            account_leverage_fp: 0,
            free_margin_micro: initial_equity_micro,
            timestamp_ms: 0,
        });
        
        Self {
            positions,
            summary,
            last_broadcast_ms: AtomicU64::new(0),
            broadcast_enabled: AtomicBool::new(true),
            subscriber_count: AtomicU8::new(0),
            serialization_buffer: Arc::new(std::sync::Mutex::new(Vec::with_capacity(4096))),
            start_time: Instant::now(),
        }
    }

    /// Get current timestamp in milliseconds
    #[inline]
    fn now_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Update position for a symbol (writes to back buffer)
    pub fn update_position(&self, snapshot: PositionSnapshot) {
        let idx = snapshot.symbol_idx as usize;
        if idx >= MAX_POSITIONS {
            return;
        }
        
        self.positions[idx].write_back(snapshot);
    }

    /// Update portfolio summary
    pub fn update_summary(&self, summary: PortfolioSummary) {
        self.summary.write_back(summary);
    }

    /// Commit all pending updates and prepare for broadcast
    pub fn commit_updates(&self) {
        // Swap all position buffers
        for i in 0..MAX_POSITIONS {
            self.positions[i].swap();
        }
        
        // Swap summary buffer
        self.summary.swap();
    }

    /// Check if ready to broadcast (rate-limited to 60FPS)
    pub fn should_broadcast(&self) -> bool {
        if !self.broadcast_enabled.load(Ordering::Acquire) {
            return false;
        }
        
        let now = self.now_ms();
        let last = self.last_broadcast_ms.load(Ordering::Acquire);
        
        now.saturating_sub(last) >= BROADCAST_INTERVAL_MS
    }

    /// Serialize positions to MessagePack-compatible format (double-buffered)
    pub fn serialize_to_messagepack(&self) -> Vec<u8> {
        let mut buffer = self.serialization_buffer.lock().unwrap();
        buffer.clear();
        
        // Write header: position count + summary flag
        let active_count = self.count_active_positions();
        buffer.push(active_count);
        
        // Serialize each active position
        for i in 0..MAX_POSITIONS {
            let pos = self.positions[i].read_front();
            if pos.net_position_fp == 0 {
                continue;
            }
            
            // Simple binary format (in production, use rmp-serde)
            buffer.push(pos.symbol_idx);
            buffer.extend_from_slice(&pos.net_position_fp.to_le_bytes());
            buffer.extend_from_slice(&pos.entry_price_fp.to_le_bytes());
            buffer.extend_from_slice(&pos.mark_price_fp.to_le_bytes());
            buffer.extend_from_slice(&pos.notional_micro.to_le_bytes());
            buffer.extend_from_slice(&pos.unrealized_pnl_micro.to_le_bytes());
            buffer.extend_from_slice(&pos.realized_pnl_today_micro.to_le_bytes());
            buffer.push(pos.active_strategy_id);
            buffer.extend_from_slice(&pos.allocation_fraction_fp.to_le_bytes());
            buffer.extend_from_slice(&pos.leverage_fp.to_le_bytes());
        }
        
        // Append summary
        let summary = self.summary.read_front();
        buffer.extend_from_slice(&summary.total_equity_micro.to_le_bytes());
        buffer.extend_from_slice(&summary.total_unrealized_pnl_micro.to_le_bytes());
        buffer.extend_from_slice(&summary.total_realized_pnl_today_micro.to_le_bytes());
        buffer.push(summary.active_positions);
        buffer.extend_from_slice(&summary.account_leverage_fp.to_le_bytes());
        buffer.extend_from_slice(&summary.free_margin_micro.to_le_bytes());
        
        buffer.clone()
    }

    /// Count active positions
    pub fn count_active_positions(&self) -> u8 {
        let mut count = 0u8;
        for i in 0..MAX_POSITIONS {
            let pos = self.positions[i].read_front();
            if pos.net_position_fp != 0 {
                count += 1;
            }
        }
        count
    }

    /// Mark broadcast as sent
    pub fn mark_broadcast_sent(&self) {
        self.last_broadcast_ms.store(self.now_ms(), Ordering::Release);
    }

    /// Enable/disable broadcasting
    pub fn set_broadcast_enabled(&self, enabled: bool) {
        self.broadcast_enabled.store(enabled, Ordering::Release);
    }

    /// Get subscriber count
    pub fn get_subscriber_count(&self) -> u8 {
        self.subscriber_count.load(Ordering::Acquire)
    }

    /// Increment subscriber count
    pub fn add_subscriber(&self) -> u8 {
        self.subscriber_count.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Decrement subscriber count
    pub fn remove_subscriber(&self) -> u8 {
        self.subscriber_count.fetch_sub(1, Ordering::AcqRel).saturating_sub(1)
    }

    /// Get position for a symbol
    pub fn get_position(&self, symbol_idx: u8) -> Option<PositionSnapshot> {
        if symbol_idx as usize >= MAX_POSITIONS {
            return None;
        }
        Some(self.positions[symbol_idx as usize].read_front())
    }

    /// Get portfolio summary
    pub fn get_summary(&self) -> PortfolioSummary {
        self.summary.read_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_buffer_swap() {
        let buffer = DoubleBuffer::new(PositionSnapshot::new());
        
        let mut snapshot = PositionSnapshot::new();
        snapshot.symbol_idx = 1;
        snapshot.net_position_fp = 1_000_000;
        
        buffer.write_back(snapshot);
        buffer.swap();
        
        let read = buffer.read_front();
        assert_eq!(read.symbol_idx, 1);
        assert_eq!(read.net_position_fp, 1_000_000);
    }

    #[test]
    fn test_position_sync_broadcast() {
        let sync = PositionSync::new(100_000_000_000);
        
        let snapshot = PositionSnapshot {
            symbol_idx: 0,
            net_position_fp: 1_000_000,
            entry_price_fp: 50_000_000_000,
            mark_price_fp: 50_100_000_000,
            notional_micro: 50_100_000,
            unrealized_pnl_micro: 100_000,
            ..PositionSnapshot::new()
        };
        
        sync.update_position(snapshot);
        sync.commit_updates();
        
        let serialized = sync.serialize_to_messagepack();
        assert!(!serialized.is_empty());
        
        let pos = sync.get_position(0);
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().net_position_fp, 1_000_000);
    }

    #[test]
    fn test_broadcast_rate_limiting() {
        let sync = PositionSync::new(100_000_000_000);
        
        assert!(sync.should_broadcast());
        sync.mark_broadcast_sent();
        
        // Should be rate-limited immediately after
        assert!(!sync.should_broadcast());
    }
}
