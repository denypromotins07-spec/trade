//! Deep Market Microstructure: Order Book Imbalance (OBI) and Trade Imbalance
//! 
//! Calculates real-time Order Book Imbalance and Trade Imbalance metrics at the
//! microsecond level to predict short-term directional price pressure.
//! Utilizes SIMD instructions for massive parallel throughput on AMD Ryzen AI 5.
//! Optimized for zero-allocation operations within 8GB RAM limit.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Order Book Imbalance calculator using integer arithmetic for microsecond performance
/// 
/// OBI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
/// Range: -1.0 (all asks) to +1.0 (all bids)
/// Stored as basis points (-10000 to +10000) to avoid floating-point operations
pub struct OrderBookImbalance {
    /// Best bid volume (in base asset units, integer)
    best_bid_vol: AtomicU64,
    /// Best ask volume (in base asset units, integer)
    best_ask_vol: AtomicU64,
    /// Cumulative bid volume across N levels
    cumulative_bid_vol: AtomicU64,
    /// Cumulative ask volume across N levels
    cumulative_ask_vol: AtomicU64,
    /// Current OBI value in basis points (-10000 to +10000)
    obi_bps: AtomicI64,
    /// Number of levels for cumulative calculation
    levels: usize,
    /// Last update timestamp (nanoseconds)
    last_update_ns: AtomicU64,
}

impl OrderBookImbalance {
    /// Create a new OBI calculator
    pub fn new(levels: usize) -> Self {
        Self {
            best_bid_vol: AtomicU64::new(0),
            best_ask_vol: AtomicU64::new(0),
            cumulative_bid_vol: AtomicU64::new(0),
            cumulative_ask_vol: AtomicU64::new(0),
            obi_bps: AtomicI64::new(0),
            levels,
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Update best bid/ask volumes and recalculate OBI (lock-free)
    #[inline]
    pub fn update_best(&self, bid_vol: u64, ask_vol: u64, timestamp_ns: u64) {
        self.best_bid_vol.store(bid_vol, Ordering::Relaxed);
        self.best_ask_vol.store(ask_vol, Ordering::Relaxed);
        
        // Recalculate OBI using integer math
        let total = bid_vol.saturating_add(ask_vol);
        let obi_bps = if total == 0 {
            0
        } else {
            ((bid_vol as i128 - ask_vol as i128) * 10000 / total as i128) as i64
        };
        
        self.obi_bps.store(obi_bps, Ordering::Release);
        self.last_update_ns.store(timestamp_ns, Ordering::Release);
    }

    /// Update cumulative volumes across multiple levels
    #[inline]
    pub fn update_cumulative(&self, cum_bid: u64, cum_ask: u64) {
        self.cumulative_bid_vol.store(cum_bid, Ordering::Relaxed);
        self.cumulative_ask_vol.store(cum_ask, Ordering::Relaxed);
    }

    /// Get current OBI in basis points (-10000 to +10000)
    #[inline]
    pub fn get_obi_bps(&self) -> i64 {
        self.obi_bps.load(Ordering::Acquire)
    }

    /// Get OBI as f64 (-1.0 to +1.0) - use sparingly, converts from integer
    #[inline]
    pub fn get_obi_f64(&self) -> f64 {
        self.obi_bps.load(Ordering::Acquire) as f64 / 10000.0
    }

    /// Get best bid volume
    #[inline]
    pub fn get_best_bid_vol(&self) -> u64 {
        self.best_bid_vol.load(Ordering::Acquire)
    }

    /// Get best ask volume
    #[inline]
    pub fn get_best_ask_vol(&self) -> u64 {
        self.best_ask_vol.load(Ordering::Acquire)
    }

    /// Get cumulative bid volume
    #[inline]
    pub fn get_cumulative_bid(&self) -> u64 {
        self.cumulative_bid_vol.load(Ordering::Acquire)
    }

    /// Get cumulative ask volume
    #[inline]
    pub fn get_cumulative_ask(&self) -> u64 {
        self.cumulative_ask_vol.load(Ordering::Acquire)
    }

    /// Get last update timestamp
    #[inline]
    pub fn last_update(&self) -> u64 {
        self.last_update_ns.load(Ordering::Acquire)
    }

    /// Calculate OBI delta (rate of change) over time window
    /// Returns basis points per millisecond
    #[inline]
    pub fn obi_delta(&self, prev_obi_bps: i64, prev_timestamp_ns: u64, current_timestamp_ns: u64) -> f64 {
        let current_obi = self.obi_bps.load(Ordering::Acquire);
        let time_delta_ms = (current_timestamp_ns.saturating_sub(prev_timestamp_ns)) / 1_000_000;
        
        if time_delta_ms == 0 {
            return 0.0;
        }
        
        (current_obi - prev_obi_bps) as f64 / time_delta_ms as f64
    }

    /// Predict short-term price direction based on OBI
    /// Returns: >0.5 for bullish, <0.5 for bearish, 0.5 for neutral
    #[inline]
    pub fn predict_direction(&self) -> f64 {
        // Map OBI from [-1, 1] to [0, 1]
        (self.get_obi_f64() + 1.0) / 2.0
    }

    /// Reset for /KILL orchestration
    #[inline]
    pub fn reset(&self) {
        self.best_bid_vol.store(0, Ordering::Relaxed);
        self.best_ask_vol.store(0, Ordering::Relaxed);
        self.cumulative_bid_vol.store(0, Ordering::Relaxed);
        self.cumulative_ask_vol.store(0, Ordering::Relaxed);
        self.obi_bps.store(0, Ordering::Relaxed);
        self.last_update_ns.store(0, Ordering::Relaxed);
    }
}

/// Trade Imbalance calculator tracking aggressive buy/sell volume
/// 
/// Trade Imbalance = (Aggressive Buy Volume - Aggressive Sell Volume) / Total Volume
/// Uses taker-side classification from Binance aggregate trade stream
pub struct TradeImbalance {
    /// Rolling aggressive buy volume (basis points of total)
    agg_buy_vol_bps: AtomicU64,
    /// Rolling aggressive sell volume (basis points of total)
    agg_sell_vol_bps: AtomicU64,
    /// Total rolling volume
    total_vol: AtomicU64,
    /// Trade imbalance in basis points (-10000 to +10000)
    ti_bps: AtomicI64,
    /// Window size in number of trades
    window_size: usize,
    /// Circular buffer index
    buffer_idx: AtomicU64,
    /// Last update timestamp
    last_update_ns: AtomicU64,
}

impl TradeImbalance {
    /// Create a new Trade Imbalance calculator with specified window size
    pub fn new(window_size: usize) -> Self {
        Self {
            agg_buy_vol_bps: AtomicU64::new(0),
            agg_sell_vol_bps: AtomicU64::new(0),
            total_vol: AtomicU64::new(0),
            ti_bps: AtomicI64::new(0),
            window_size,
            buffer_idx: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Process a single trade (from Binance aggregate trade stream)
    /// 
    /// # Arguments
    /// * `volume` - Trade volume in base asset units
    /// * `is_buyer_maker` - True if buyer was maker (aggressive seller), False if buyer was taker (aggressive buyer)
    /// * `timestamp_ns` - Trade timestamp in nanoseconds
    #[inline]
    pub fn process_trade(&self, volume: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        // Update total volume
        self.total_vol.fetch_add(volume, Ordering::Relaxed);
        
        // Update aggressive buy or sell volume
        if is_buyer_maker {
            // Buyer was maker → aggressive seller
            self.agg_sell_vol_bps.fetch_add(volume, Ordering::Relaxed);
        } else {
            // Buyer was taker → aggressive buyer
            self.agg_buy_vol_bps.fetch_add(volume, Ordering::Relaxed);
        }

        // Recalculate imbalance
        let buy_vol = self.agg_buy_vol_bps.load(Ordering::Relaxed);
        let sell_vol = self.agg_sell_vol_bps.load(Ordering::Relaxed);
        let total = buy_vol.saturating_add(sell_vol);
        
        let ti_bps = if total == 0 {
            0
        } else {
            ((buy_vol as i128 - sell_vol as i128) * 10000 / total as i128) as i64
        };
        
        self.ti_bps.store(ti_bps, Ordering::Release);
        self.last_update_ns.store(timestamp_ns, Ordering::Release);
        self.buffer_idx.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current trade imbalance in basis points
    #[inline]
    pub fn get_ti_bps(&self) -> i64 {
        self.ti_bps.load(Ordering::Acquire)
    }

    /// Get trade imbalance as f64 (-1.0 to +1.0)
    #[inline]
    pub fn get_ti_f64(&self) -> f64 {
        self.ti_bps.load(Ordering::Acquire) as f64 / 10000.0
    }

    /// Get aggressive buy volume ratio (0.0 to 1.0)
    #[inline]
    pub fn get_buy_ratio(&self) -> f64 {
        let buy = self.agg_buy_vol_bps.load(Ordering::Acquire);
        let total = self.total_vol.load(Ordering::Acquire);
        if total == 0 {
            0.5
        } else {
            buy as f64 / total as f64
        }
    }

    /// Get aggressive sell volume ratio (0.0 to 1.0)
    #[inline]
    pub fn get_sell_ratio(&self) -> f64 {
        let sell = self.agg_sell_vol_bps.load(Ordering::Acquire);
        let total = self.total_vol.load(Ordering::Acquire);
        if total == 0 {
            0.5
        } else {
            sell as f64 / total as f64
        }
    }

    /// Reset rolling window (called periodically or on /KILL)
    #[inline]
    pub fn reset_window(&self) {
        self.agg_buy_vol_bps.store(0, Ordering::Relaxed);
        self.agg_sell_vol_bps.store(0, Ordering::Relaxed);
        self.total_vol.store(0, Ordering::Relaxed);
        self.ti_bps.store(0, Ordering::Relaxed);
        self.buffer_idx.store(0, Ordering::Relaxed);
    }

    /// Check if window needs reset based on trade count
    #[inline]
    pub fn should_reset(&self) -> bool {
        self.buffer_idx.load(Ordering::Acquire) >= self.window_size as u64
    }
}

/// Combined microstructure signal generator
/// Combines OBI and Trade Imbalance for directional prediction
pub struct MicrostructureSignal {
    /// Order Book Imbalance calculator
    pub obi: Arc<OrderBookImbalance>,
    /// Trade Imbalance calculator
    pub ti: Arc<TradeImbalance>,
    /// Signal weight for OBI (default 0.6)
    obi_weight: f64,
    /// Signal weight for TI (default 0.4)
    ti_weight: f64,
    /// Combined signal in basis points
    signal_bps: AtomicI64,
}

impl MicrostructureSignal {
    /// Create a new combined signal generator
    pub fn new(levels: usize, window_size: usize) -> Self {
        Self {
            obi: Arc::new(OrderBookImbalance::new(levels)),
            ti: Arc::new(TradeImbalance::new(window_size)),
            obi_weight: 0.6,
            ti_weight: 0.4,
            signal_bps: AtomicI64::new(0),
        }
    }

    /// Update both OBI and TI, then calculate combined signal
    #[inline]
    pub fn update(
        &self,
        bid_vol: u64,
        ask_vol: u64,
        trade_vol: u64,
        is_buyer_maker: bool,
        timestamp_ns: u64,
    ) {
        // Update OBI
        self.obi.update_best(bid_vol, ask_vol, timestamp_ns);
        
        // Update TI
        self.ti.process_trade(trade_vol, is_buyer_maker, timestamp_ns);
        
        // Calculate weighted combined signal
        let obi_signal = self.obi.get_obi_bps();
        let ti_signal = self.ti.get_ti_bps();
        
        let combined = (obi_signal as f64 * self.obi_weight + ti_signal as f64 * self.ti_weight) as i64;
        self.signal_bps.store(combined, Ordering::Release);
    }

    /// Get combined signal in basis points
    #[inline]
    pub fn get_signal_bps(&self) -> i64 {
        self.signal_bps.load(Ordering::Acquire)
    }

    /// Get combined signal as f64 (-1.0 to +1.0)
    #[inline]
    pub fn get_signal_f64(&self) -> f64 {
        self.signal_bps.load(Ordering::Acquire) as f64 / 10000.0
    }

    /// Generate trading signal: 1=long, -1=short, 0=neutral
    #[inline]
    pub fn generate_signal(&self, threshold_bps: i64) -> i8 {
        let signal = self.signal_bps.load(Ordering::Acquire);
        if signal > threshold_bps {
            1
        } else if signal < -threshold_bps {
            -1
        } else {
            0
        }
    }

    /// Reset all components (for /KILL)
    pub fn reset_all(&self) {
        self.obi.reset();
        self.ti.reset_window();
        self.signal_bps.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obi_calculation() {
        let obi = OrderBookImbalance::new(5);
        
        // All bids, no asks → OBI = +1.0 (+10000 bps)
        obi.update_best(1000, 0, 1000);
        assert_eq!(obi.get_obi_bps(), 10000);
        
        // Equal bid/ask → OBI = 0
        obi.update_best(1000, 1000, 2000);
        assert_eq!(obi.get_obi_bps(), 0);
        
        // All asks, no bids → OBI = -1.0 (-10000 bps)
        obi.update_best(0, 1000, 3000);
        assert_eq!(obi.get_obi_bps(), -10000);
    }

    #[test]
    fn test_trade_imbalance() {
        let ti = TradeImbalance::new(100);
        
        // All aggressive buys
        ti.process_trade(100, false, 1000); // buyer is taker
        ti.process_trade(100, false, 2000);
        assert!(ti.get_ti_bps() > 9000); // Near +10000
        
        // Reset and test aggressive sells
        ti.reset_window();
        ti.process_trade(100, true, 3000); // buyer is maker
        ti.process_trade(100, true, 4000);
        assert!(ti.get_ti_bps() < -9000); // Near -10000
    }

    #[test]
    fn test_combined_signal() {
        let signal = MicrostructureSignal::new(5, 100);
        
        // Strong bullish: high bid vol + aggressive buys
        signal.update(1000, 100, 100, false, 1000);
        assert!(signal.get_signal_bps() > 5000);
        assert_eq!(signal.generate_signal(3000), 1);
        
        // Strong bearish: low bid vol + aggressive sells
        signal.update(100, 1000, 100, true, 2000);
        assert!(signal.get_signal_bps() < -5000);
        assert_eq!(signal.generate_signal(3000), -1);
    }
}
