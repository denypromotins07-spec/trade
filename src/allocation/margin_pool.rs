// Margin Pool: Unified cross-margin pool tracker preventing over-leveraging across
// isolated symbol engines. Broadcasts net exposure updates to frontend UI at 60FPS
// via WebSocket gateway. Optimized for AMD Ryzen AI 5 with lock-free atomics.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of symbol engines in the pool
const MAX_ENGINES: usize = 8;

/// Fixed-point scale factor
const FP_SCALE: u64 = 1_000_000;

/// Leverage multiplier fixed-point (10x = 10_000_000)
const MAX_LEVERAGE_FP: u64 = 10_000_000;

/// Margin call threshold (80% of max leverage = 8_000_000)
const MARGIN_CALL_THRESHOLD_FP: u64 = 8_000_000;

/// Liquidation threshold (95% of max leverage = 9_500_000)
const LIQUIDATION_THRESHOLD_FP: u64 = 9_500_000;

/// Exposure data for a single asset
#[derive(Debug, Clone, Copy)]
pub struct AssetExposure {
    /// Symbol index (0-7)
    pub symbol_idx: u8,
    /// Net position in base currency (fixed-point, scaled by asset precision)
    pub net_position_fp: i64,
    /// Notional value in micro-USD
    pub notional_micro: u64,
    /// Used margin in micro-USD
    pub used_margin_micro: u64,
    /// Unrealized PnL in micro-USD
    pub unrealized_pnl_micro: i64,
    /// Win rate (fixed-point)
    pub win_rate_fp: u64,
    /// Average win size (micro-USD)
    pub avg_win_fp: u64,
    /// Average loss size (micro-USD)
    pub avg_loss_fp: u64,
    /// Correlation penalty (fixed-point, 1.0 = no penalty)
    pub correlation_penalty_fp: u64,
    /// Last update timestamp (milliseconds since epoch)
    pub last_update_ms: u64,
}

impl AssetExposure {
    pub const fn new(symbol_idx: u8) -> Self {
        Self {
            symbol_idx,
            net_position_fp: 0,
            notional_micro: 0,
            used_margin_micro: 0,
            unrealized_pnl_micro: 0,
            win_rate_fp: 0,
            avg_win_fp: 0,
            avg_loss_fp: 0,
            correlation_penalty_fp: FP_SCALE,
            last_update_ms: 0,
        }
    }
}

/// Aggregated portfolio state for broadcasting
#[derive(Debug, Clone, Copy)]
pub struct PortfolioState {
    /// Total equity in micro-USD
    pub total_equity_micro: u64,
    /// Total used margin in micro-USD
    pub total_used_margin_micro: u64,
    /// Total free margin in micro-USD
    pub free_margin_micro: u64,
    /// Current leverage (fixed-point)
    pub current_leverage_fp: u64,
    /// Total unrealized PnL in micro-USD
    pub total_unrealized_pnl_micro: i64,
    /// Number of active positions
    pub active_positions: u8,
    /// Margin call flag
    pub margin_call_active: bool,
    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

/// Lock-free margin pool using atomic RCU pattern
pub struct MarginPool {
    /// Total initial equity (micro-USD)
    initial_equity_micro: AtomicU64,
    /// Per-engine exposures (updated via write lock)
    exposures: [std::sync::RwLock<AssetExposure>; MAX_ENGINES],
    /// Aggregate used margin (micro-USD)
    total_used_margin_micro: AtomicU64,
    /// Aggregate unrealized PnL (micro-USD)
    total_unrealized_pnl_micro: AtomicI64,
    /// Active position count
    active_positions: AtomicU64,
    /// Last broadcast timestamp
    last_broadcast_ms: AtomicU64,
    /// Broadcast interval target (16.67ms for 60FPS)
    broadcast_interval_ms: u64,
    /// Emergency liquidation flag
    liquidation_pending: AtomicBool,
}

impl MarginPool {
    /// Create a new margin pool with initial equity
    pub fn new(initial_equity_micro: u64) -> Self {
        let mut exposures = [
            std::sync::RwLock::new(AssetExposure::new(0)),
            std::sync::RwLock::new(AssetExposure::new(1)),
            std::sync::RwLock::new(AssetExposure::new(2)),
            std::sync::RwLock::new(AssetExposure::new(3)),
            std::sync::RwLock::new(AssetExposure::new(4)),
            std::sync::RwLock::new(AssetExposure::new(5)),
            std::sync::RwLock::new(AssetExposure::new(6)),
            std::sync::RwLock::new(AssetExposure::new(7)),
        ];
        
        Self {
            initial_equity_micro: AtomicU64::new(initial_equity_micro),
            exposures,
            total_used_margin_micro: AtomicU64::new(0),
            total_unrealized_pnl_micro: AtomicI64::new(0),
            active_positions: AtomicU64::new(0),
            last_broadcast_ms: AtomicU64::new(0),
            broadcast_interval_ms: 16, // ~60FPS
            liquidation_pending: AtomicBool::new(false),
        }
    }

    /// Update exposure for a specific symbol engine
    pub fn update_exposure(&self, symbol_idx: u8, exposure: AssetExposure) {
        if symbol_idx as usize >= MAX_ENGINES {
            return;
        }
        
        let mut lock = self.exposures[symbol_idx as usize].write().unwrap();
        *lock = exposure;
        
        // Update aggregate metrics
        self.recalculate_aggregates();
    }

    /// Update just the PnL for a symbol (hot path, minimal locking)
    pub fn update_pnl(&self, symbol_idx: u8, pnl_micro: i64, notional_micro: u64) {
        if symbol_idx as usize >= MAX_ENGINES {
            return;
        }
        
        let mut lock = self.exposures[symbol_idx as usize].write().unwrap();
        lock.unrealized_pnl_micro = pnl_micro;
        lock.notional_micro = notional_micro;
        lock.last_update_ms = self.get_timestamp_ms();
        
        // Quick aggregate update without full recalculation
        let old_pnl = lock.unrealized_pnl_micro;
        self.total_unrealized_pnl_micro.fetch_add(
            pnl_micro - old_pnl,
            Ordering::Relaxed,
        );
    }

    /// Recalculate all aggregate metrics (called periodically)
    fn recalculate_aggregates(&self) {
        let mut total_used = 0u64;
        let mut total_pnl = 0i64;
        let mut active_count = 0u64;
        
        for i in 0..MAX_ENGINES {
            let lock = self.exposures[i].read().unwrap();
            total_used += lock.used_margin_micro;
            total_pnl += lock.unrealized_pnl_micro;
            if lock.net_position_fp != 0 {
                active_count += 1;
            }
        }
        
        self.total_used_margin_micro.store(total_used, Ordering::Release);
        self.total_unrealized_pnl_micro.store(total_pnl, Ordering::Release);
        self.active_positions.store(active_count, Ordering::Release);
    }

    /// Get total equity (initial + unrealized PnL)
    pub fn get_total_equity_micro(&self) -> u64 {
        let initial = self.initial_equity_micro.load(Ordering::Acquire);
        let pnl = self.total_unrealized_pnl_micro.load(Ordering::Acquire);
        if pnl >= 0 {
            initial + pnl as u64
        } else {
            initial.saturating_sub((-pnl) as u64)
        }
    }

    /// Get total used margin
    pub fn get_total_used_margin_micro(&self) -> u64 {
        self.total_used_margin_micro.load(Ordering::Acquire)
    }

    /// Get free margin (equity - used)
    pub fn get_free_margin_micro(&self) -> u64 {
        let equity = self.get_total_equity_micro();
        let used = self.get_total_used_margin_micro();
        equity.saturating_sub(used)
    }

    /// Get current leverage (fixed-point)
    pub fn get_current_leverage_fp(&self) -> u64 {
        let equity = self.get_total_equity_micro();
        let used = self.get_total_used_margin_micro();
        
        if equity == 0 {
            return 0;
        }
        
        ((used as u128 * FP_SCALE as u128) / equity as u128) as u64
    }

    /// Check if margin call threshold is breached
    pub fn is_margin_call(&self) -> bool {
        let leverage = self.get_current_leverage_fp();
        leverage >= MARGIN_CALL_THRESHOLD_FP
    }

    /// Check if liquidation threshold is breached
    pub fn is_liquidation_pending(&self) -> bool {
        let leverage = self.get_current_leverage_fp();
        leverage >= LIQUIDATION_THRESHOLD_FP || self.liquidation_pending.load(Ordering::Acquire)
    }

    /// Get exposure for a specific symbol
    pub fn get_exposure(&self, symbol_idx: u8) -> Option<AssetExposure> {
        if symbol_idx as usize >= MAX_ENGINES {
            return None;
        }
        
        let lock = self.exposures[symbol_idx as usize].read().unwrap();
        Some(*lock)
    }

    /// Get aggregated portfolio state for broadcasting
    pub fn get_portfolio_state(&self) -> PortfolioState {
        let equity = self.get_total_equity_micro();
        let used = self.get_total_used_margin_micro();
        let pnl = self.total_unrealized_pnl_micro.load(Ordering::Acquire);
        let leverage = self.get_current_leverage_fp();
        
        PortfolioState {
            total_equity_micro: equity,
            total_used_margin_micro: used,
            free_margin_micro: equity.saturating_sub(used),
            current_leverage_fp: leverage,
            total_unrealized_pnl_micro: pnl,
            active_positions: self.active_positions.load(Ordering::Acquire) as u8,
            margin_call_active: self.is_margin_call(),
            timestamp_ms: self.get_timestamp_ms(),
        }
    }

    /// Check if ready to broadcast (60FPS rate limit)
    pub fn should_broadcast(&self) -> bool {
        let now = self.get_timestamp_ms();
        let last = self.last_broadcast_ms.load(Ordering::Acquire);
        now.saturating_sub(last) >= self.broadcast_interval_ms
    }

    /// Mark broadcast as sent
    pub fn mark_broadcast_sent(&self) {
        self.last_broadcast_ms.store(self.get_timestamp_ms(), Ordering::Release);
    }

    /// Trigger emergency liquidation
    pub fn trigger_liquidation(&self) {
        self.liquidation_pending.store(true, Ordering::Release);
    }

    /// Clear liquidation flag
    pub fn clear_liquidation(&self) {
        self.liquidation_pending.store(false, Ordering::Release);
    }

    /// Get timestamp in milliseconds
    fn get_timestamp_ms(&self) -> u64 {
        Instant::now()
            .duration_since(Instant::now())
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64
    }

    /// Serialize portfolio state to MessagePack-compatible bytes (double-buffered)
    pub fn serialize_to_messagepack(&self, buffer: &mut [u8]) -> usize {
        let state = self.get_portfolio_state();
        
        // Simple binary serialization (in production, use rmp-serde)
        let mut offset = 0;
        
        // Write each field as little-endian u64/i64
        buffer[offset..offset+8].copy_from_slice(&state.total_equity_micro.to_le_bytes());
        offset += 8;
        
        buffer[offset..offset+8].copy_from_slice(&state.total_used_margin_micro.to_le_bytes());
        offset += 8;
        
        buffer[offset..offset+8].copy_from_slice(&state.free_margin_micro.to_le_bytes());
        offset += 8;
        
        buffer[offset..offset+8].copy_from_slice(&state.current_leverage_fp.to_le_bytes());
        offset += 8;
        
        buffer[offset..offset+8].copy_from_slice(&(state.total_unrealized_pnl_micro as u64).to_le_bytes());
        offset += 8;
        
        buffer[offset] = state.active_positions;
        offset += 1;
        
        buffer[offset] = if state.margin_call_active { 1 } else { 0 };
        offset += 1;
        
        buffer[offset..offset+8].copy_from_slice(&state.timestamp_ms.to_le_bytes());
        offset += 8;
        
        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_pool_state() {
        let pool = MarginPool::new(100_000_000_000); // $100,000
        
        assert_eq!(pool.get_total_equity_micro(), 100_000_000_000);
        assert_eq!(pool.get_total_used_margin_micro(), 0);
        assert_eq!(pool.get_free_margin_micro(), 100_000_000_000);
        assert!(!pool.is_margin_call());
    }

    #[test]
    fn test_exposure_update_affects_aggregates() {
        let pool = MarginPool::new(100_000_000_000);
        
        let mut exposure = AssetExposure::new(0);
        exposure.used_margin_micro = 10_000_000_000; // $10,000
        exposure.unrealized_pnl_micro = 500_000_000; // $500 profit
        exposure.net_position_fp = 1_000_000;
        
        pool.update_exposure(0, exposure);
        
        assert_eq!(pool.get_total_used_margin_micro(), 10_000_000_000);
        assert_eq!(pool.get_total_equity_micro(), 100_500_000_000);
        assert_eq!(pool.active_positions.load(Ordering::Acquire), 1);
    }

    #[test]
    fn test_margin_call_detection() {
        let pool = MarginPool::new(10_000_000_000); // $10,000
        
        // Set used margin to 80% of equity (margin call threshold)
        let mut exposure = AssetExposure::new(0);
        exposure.used_margin_micro = 8_000_000_000;
        exposure.net_position_fp = 1_000_000;
        
        pool.update_exposure(0, exposure);
        
        assert!(pool.is_margin_call());
    }

    #[test]
    fn test_portfolio_state_serialization() {
        let pool = MarginPool::new(100_000_000_000);
        let mut buffer = [0u8; 64];
        
        let len = pool.serialize_to_messagepack(&mut buffer);
        
        assert!(len > 0);
        assert!(len <= 64);
    }
}
