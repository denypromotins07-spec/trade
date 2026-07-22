//! Liquidity Sniping Engine
//!
//! Implements predictive order book decay models to sniper fleeting liquidity.
//! Fires market orders exactly when hidden institutional walls are detected
//! to minimize adverse selection. Uses lock-free data structures for microsecond response.
//!
//! # Features
//! - Hidden liquidity detection via order flow analysis
//! - Predictive decay modeling
//! - Microsecond trigger execution
//! - Zero heap allocation in hot path

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Maximum number of recent trades to track (compile-time constant)
const MAX_RECENT_TRADES: usize = 100;

/// Trade record for decay analysis
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct TradeRecord {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Price in ticks
    pub price: u64,
    /// Size in base currency * 1e8
    pub size: u64,
    /// Is buyer initiated (1 = yes, 0 = no)
    pub is_buyer_maker: u8,
}

impl Default for TradeRecord {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            price: 0,
            size: 0,
            is_buyer_maker: 0,
        }
    }
}

/// Circular buffer for recent trades (lock-free, pre-allocated)
#[repr(C, align(64))]
pub struct TradeBuffer {
    data: [TradeRecord; MAX_RECENT_TRADES],
    head: AtomicU64,
    count: AtomicU64,
}

impl Default for TradeBuffer {
    fn default() -> Self {
        Self {
            data: [TradeRecord::default(); MAX_RECENT_TRADES],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
}

impl TradeBuffer {
    /// Push a new trade record (lock-free, overwrites oldest if full)
    #[inline]
    pub fn push(&self, record: TradeRecord) {
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        let idx = (head % MAX_RECENT_TRADES as u64) as usize;
        
        unsafe {
            // Safe because we have exclusive write access via atomic head
            let ptr = &self.data[idx] as *const TradeRecord as *mut TradeRecord;
            ptr.write(record);
        }
        
        // Update count (capped at MAX)
        let current_count = self.count.load(Ordering::Relaxed);
        if current_count < MAX_RECENT_TRADES as u64 {
            self.count.store(current_count + 1, Ordering::Relaxed);
        }
    }

    /// Get recent trades as slice (for analysis)
    #[inline]
    pub fn get_recent(&self, n: usize) -> &[TradeRecord] {
        let count = self.count.load(Ordering::Acquire) as usize;
        let n = n.min(count);
        if n == 0 {
            return &[];
        }
        
        let head = self.head.load(Ordering::Acquire) as usize;
        let start = if head >= n { head - n } else { MAX_RECENT_TRADES - (n - head) };
        
        // Return contiguous slice (may wrap)
        if start + n <= MAX_RECENT_TRADES {
            &self.data[start..start + n]
        } else {
            // Wrapped case - return what we can
            &self.data[start..]
        }
    }
}

/// Configuration for liquidity sniping
#[derive(Debug, Clone, Copy)]
pub struct SniperConfig {
    /// Minimum hidden wall size to trigger (base currency * 1e8)
    pub min_hidden_size: u64,
    /// Decay threshold (trades per millisecond)
    pub decay_threshold: f64,
    /// Confidence threshold for trigger (0.0 - 1.0, scaled by 10000)
    pub confidence_threshold: u64,
    /// Cooldown between snipes (microseconds)
    pub cooldown_us: u64,
    /// Maximum snipe size (base currency * 1e8)
    pub max_snipe_size: u64,
}

impl Default for SniperConfig {
    fn default() -> Self {
        Self {
            min_hidden_size: 5_000_000,
            decay_threshold: 100.0,
            confidence_threshold: 7500, // 0.75
            cooldown_us: 5000,
            max_snipe_size: 10_000_000,
        }
    }
}

/// Liquidity sniper engine
pub struct LiquiditySniper {
    /// Recent trade buffer
    trade_buffer: TradeBuffer,
    /// Last snipe timestamp
    last_snipe_us: AtomicU64,
    /// Active flag
    active: AtomicBool,
    /// Configuration
    config: SniperConfig,
    /// Cached decay rate (for fast access)
    cached_decay_rate: AtomicU64, // Scaled by 10000
    /// Cache line padding
    _padding: [u8; 64],
}

impl LiquiditySniper {
    /// Create new sniper instance
    #[inline]
    pub const fn new(config: SniperConfig) -> Self {
        Self {
            trade_buffer: TradeBuffer::default(),
            last_snipe_us: AtomicU64::new(0),
            active: AtomicBool::new(false),
            config,
            cached_decay_rate: AtomicU64::new(0),
            _padding: [0; 64],
        }
    }

    /// Activate/deactivate sniper
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Release);
    }

    /// Record a new trade
    #[inline]
    pub fn record_trade(&self, timestamp_us: u64, price: u64, size: u64, is_buyer_maker: bool) {
        let record = TradeRecord {
            timestamp_us,
            price,
            size,
            is_buyer_maker: if is_buyer_maker { 1 } else { 0 },
        };
        self.trade_buffer.push(record);
        
        // Update decay rate estimate
        self.update_decay_rate();
    }

    /// Update cached decay rate from recent trades
    #[inline]
    fn update_decay_rate(&self) {
        let trades = self.trade_buffer.get_recent(20);
        if trades.len() < 2 {
            self.cached_decay_rate.store(0, Ordering::Relaxed);
            return;
        }

        let first = trades[0].timestamp_us;
        let last = trades.last().unwrap().timestamp_us;
        
        if last <= first {
            self.cached_decay_rate.store(0, Ordering::Relaxed);
            return;
        }

        let duration_ms = (last - first) / 1000;
        if duration_ms == 0 {
            self.cached_decay_rate.store(u64::MAX, Ordering::Relaxed);
            return;
        }

        let rate = (trades.len() as u64 * 10000) / duration_ms;
        self.cached_decay_rate.store(rate, Ordering::Relaxed);
    }

    /// Detect hidden institutional walls
    /// Returns estimated hidden size if detected
    #[inline]
    pub fn detect_hidden_wall(&self, visible_size: u64, price: u64, is_bid: bool) -> Option<u64> {
        let trades = self.trade_buffer.get_recent(50);
        if trades.len() < 10 {
            return None;
        }

        // Analyze trade pattern at this price level
        let mut total_size_at_price = 0u64;
        let mut rapid_trades = 0u64;
        
        for trade in trades {
            if trade.price == price {
                total_size_at_price += trade.size;
                rapid_trades += 1;
            }
        }

        // Hidden wall detected if:
        // 1. Total traded size >> visible size
        // 2. Rapid succession of trades at same price
        if total_size_at_price > visible_size * 3 && rapid_trades >= 5 {
            let hidden_estimate = total_size_at_price.saturating_sub(visible_size);
            if hidden_estimate >= self.config.min_hidden_size {
                return Some(hidden_estimate);
            }
        }

        None
    }

    /// Calculate confidence score for snipe opportunity (0-10000 scale)
    #[inline]
    pub fn calculate_confidence(&self, hidden_size: u64, decay_rate: u64) -> u64 {
        let mut confidence = 0u64;

        // Size component (0-5000)
        let size_ratio = (hidden_size * 10000) / self.config.min_hidden_size;
        confidence += size_ratio.min(5000);

        // Decay rate component (0-5000)
        let decay_scaled = (decay_rate * 10000) / (self.config.decency_threshold * 10000.0) as u64;
        confidence += decay_scaled.min(5000);

        confidence.min(10000)
    }

    /// Attempt to snipe liquidity
    /// Returns Some(snipe_size) if triggered, None otherwise
    /// 
    /// # Arguments
    /// * `hidden_size` - Detected hidden liquidity size
    /// * `is_bid` - true to snipe bids (sell), false to snipe asks (buy)
    /// * `current_time_us` - Current timestamp
    #[inline]
    pub fn try_snipe(&self, hidden_size: u64, is_bid: bool, current_time_us: u64) -> Option<u64> {
        if !self.active.load(Ordering::Acquire) {
            return None;
        }

        // Check cooldown
        let last_snipe = self.last_snipe_us.load(Ordering::Relaxed);
        if current_time_us < last_snipe + self.config.cooldown_us {
            return None;
        }

        // Get current decay rate
        let decay_rate = self.cached_decay_rate.load(Ordering::Acquire);

        // Calculate confidence
        let confidence = self.calculate_confidence(hidden_size, decay_rate);
        if confidence < self.config.confidence_threshold {
            return None;
        }

        // Calculate snipe size (proportional to hidden size, capped)
        let snipe_size = (hidden_size / 4).min(self.config.max_snipe_size);

        // Execute snipe (atomically update last_snipe)
        let expected = last_snipe;
        if self.last_snipe_us.compare_exchange(
            expected,
            current_time_us,
            Ordering::SeqCst,
            Ordering::Relaxed
        ).is_ok() {
            Some(snipe_size)
        } else {
            None // Another thread sniped first
        }
    }

    /// Get current decay rate (trades per ms, scaled by 10000)
    #[inline]
    pub fn get_decay_rate(&self) -> u64 {
        self.cached_decay_rate.load(Ordering::Acquire)
    }

    /// Check if sniper is active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    /// Get time since last snipe (microseconds)
    #[inline]
    pub fn time_since_last_snipe(&self, current_time_us: u64) -> u64 {
        let last = self.last_snipe_us.load(Ordering::Relaxed);
        current_time_us.saturating_sub(last)
    }
}

/// Snipe signal for execution engine
#[derive(Debug, Clone, Copy)]
pub struct SnipeSignal {
    /// Direction: true = buy (snipe ask), false = sell (snipe bid)
    pub is_buy: bool,
    /// Size to execute (base currency * 1e8)
    pub size: u64,
    /// Confidence score (0-10000)
    pub confidence: u64,
    /// Trigger price (ticks)
    pub price: u64,
}

impl LiquiditySniper {
    /// Full analysis and signal generation
    #[inline]
    pub fn analyze_and_signal(
        &self,
        visible_bid_size: u64,
        visible_ask_size: u64,
        best_bid: u64,
        best_ask: u64,
        current_time_us: u64,
    ) -> Option<SnipeSignal> {
        // Check both sides for hidden walls
        let bid_hidden = self.detect_hidden_wall(visible_bid_size, best_bid, true);
        let ask_hidden = self.detect_hidden_wall(visible_ask_size, best_ask, false);

        if let Some(hidden) = bid_hidden {
            if let Some(size) = self.try_snipe(hidden, false, current_time_us) {
                let decay = self.cached_decay_rate.load(Ordering::Relaxed);
                let confidence = self.calculate_confidence(hidden, decay);
                return Some(SnipeSignal {
                    is_buy: false,
                    size,
                    confidence,
                    price: best_bid,
                });
            }
        }

        if let Some(hidden) = ask_hidden {
            if let Some(size) = self.try_snipe(hidden, true, current_time_us) {
                let decay = self.cached_decay_rate.load(Ordering::Relaxed);
                let confidence = self.calculate_confidence(hidden, decay);
                return Some(SnipeSignal {
                    is_buy: true,
                    size,
                    confidence,
                    price: best_ask,
                });
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sniper_creation() {
        let config = SniperConfig::default();
        let sniper = LiquiditySniper::new(config);
        assert!(!sniper.is_active());
    }

    #[test]
    fn test_trade_recording() {
        let sniper = LiquiditySniper::new(SniperConfig::default());
        sniper.record_trade(1000000, 50000, 100000, false);
        sniper.record_trade(1001000, 50001, 150000, false);
        assert!(sniper.trade_buffer.count.load(Ordering::Relaxed) >= 2);
    }

    #[test]
    fn test_activation() {
        let sniper = LiquiditySniper::new(SniperConfig::default());
        sniper.set_active(true);
        assert!(sniper.is_active());
        sniper.set_active(false);
        assert!(!sniper.is_active());
    }
}
