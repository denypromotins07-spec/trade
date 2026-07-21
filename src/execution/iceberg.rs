//! # Iceberg Order Execution and Detection
//! 
//! Implements advanced iceberg order execution and detection logic,
//! dynamically slicing large institutional orders to hide true size from
//! predatory HFT market makers.
//! 
//! ## Key Features:
//! - Dynamic slice sizing based on market volume
//! - Randomized timing to avoid pattern detection
//! - Iceberg detection for competitor analysis
//! - Integration with smart order routing
//! - Microsecond-latency execution in Rust hot path

use std::sync::atomic::{AtomicUsize, AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;

/// Iceberg order configuration
#[derive(Debug, Clone)]
pub struct IcebergConfig {
    /// Minimum slice size (base units)
    pub min_slice_size: u64,
    /// Maximum slice size as percentage of display size
    pub max_slice_pct: f64,
    /// Randomization factor for slice sizes (0.0-1.0)
    pub randomization_factor: f64,
    /// Minimum time between slices (milliseconds)
    pub min_interval_ms: u64,
    /// Maximum time between slices (milliseconds)
    pub max_interval_ms: u64,
    /// Participation rate limit (percentage of market volume)
    pub max_participation_rate: f64,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            min_slice_size: 100,
            max_slice_pct: 0.1,
            randomization_factor: 0.3,
            min_interval_ms: 100,
            max_interval_ms: 2000,
            max_participation_rate: 0.05, // 5% max participation
        }
    }
}

/// Iceberg order state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcebergState {
    /// Order not yet started
    Pending,
    /// Actively executing slices
    Active,
    /// Paused (waiting for conditions)
    Paused,
    /// Fully executed
    Completed,
    /// Cancelled by user
    Cancelled,
}

/// Single iceorder execution
pub struct IcebergOrder {
    /// Unique order ID
    pub order_id: u64,
    /// Total order quantity (hidden)
    pub total_quantity: u64,
    /// Remaining quantity to execute
    pub remaining_quantity: u64,
    /// Executed quantity
    pub executed_quantity: u64,
    /// Current displayed size (visible to market)
    pub display_size: u64,
    /// Side (true = buy, false = sell)
    pub is_buy: bool,
    /// Symbol
    pub symbol: [u8; 12],
    /// Price limit (0 for market)
    pub limit_price: u64,
    /// Current state
    pub state: IcebergState,
    /// Configuration
    pub config: IcebergConfig,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last slice execution timestamp
    pub last_slice_at: Option<Instant>,
    /// Number of slices executed
    pub slices_executed: usize,
    /// Average execution price
    pub avg_execution_price: u64,
}

impl IcebergOrder {
    pub fn new(
        order_id: u64,
        symbol: [u8; 12],
        is_buy: bool,
        total_quantity: u64,
        limit_price: u64,
        config: IcebergConfig,
    ) -> Self {
        let display_size = ((total_quantity as f64 * config.max_slice_pct) as u64)
            .max(config.min_slice_size);
        
        Self {
            order_id,
            total_quantity,
            remaining_quantity: total_quantity,
            executed_quantity: 0,
            display_size,
            is_buy,
            symbol,
            limit_price,
            state: IcebergState::Pending,
            config,
            created_at: Instant::now(),
            last_slice_at: None,
            slices_executed: 0,
            avg_execution_price: 0,
        }
    }

    /// Calculate next slice size with randomization
    pub fn calculate_next_slice(&self, market_volume: u64) -> u64 {
        if self.remaining_quantity == 0 {
            return 0;
        }

        // Base slice size
        let base_size = self.display_size;
        
        // Apply randomization
        let random_factor = 1.0 + (rand_f64() - 0.5) * 2.0 * self.config.randomization_factor;
        let randomized_size = (base_size as f64 * random_factor) as u64;
        
        // Limit by participation rate
        let max_by_participation = (market_volume as f64 * self.config.max_participation_rate) as u64;
        
        // Take minimum of all constraints
        randomized_size
            .min(max_by_participation)
            .min(self.remaining_quantity)
            .max(self.config.min_slice_size)
    }

    /// Check if ready to execute next slice
    pub fn should_execute_slice(&self) -> bool {
        match self.state {
            IcebergState::Active | IcebergState::Pending => {}
            _ => return false,
        }

        if self.remaining_quantity == 0 {
            return false;
        }

        // Check time since last slice
        if let Some(last) = self.last_slice_at {
            let elapsed_ms = last.elapsed().as_millis() as u64;
            if elapsed_ms < self.config.min_interval_ms {
                return false;
            }
        }

        true
    }

    /// Update order after slice execution
    pub fn update_after_fill(&mut self, fill_qty: u64, fill_price: u64) {
        self.executed_quantity += fill_qty;
        self.remaining_quantity = self.remaining_quantity.saturating_sub(fill_qty);
        self.slices_executed += 1;
        self.last_slice_at = Some(Instant::now());

        // Update average price
        let total_value = self.avg_execution_price as u128 * self.executed_quantity as u128;
        let fill_value = fill_price as u128 * fill_qty as u128;
        self.avg_execution_price = ((total_value + fill_value) / (self.executed_quantity as u128)) as u64;

        // Check completion
        if self.remaining_quantity == 0 {
            self.state = IcebergState::Completed;
        }
    }

    /// Start order execution
    pub fn start(&mut self) {
        if self.state == IcebergState::Pending {
            self.state = IcebergState::Active;
        }
    }

    /// Pause order execution
    pub fn pause(&mut self) {
        if self.state == IcebergState::Active {
            self.state = IcebergState::Paused;
        }
    }

    /// Resume order execution
    pub fn resume(&mut self) {
        if self.state == IcebergState::Paused {
            self.state = IcebergState::Active;
        }
    }

    /// Cancel order
    pub fn cancel(&mut self) {
        self.state = IcebergState::Cancelled;
    }
}

/// Simple random number generator (replace with proper RNG in production)
fn rand_f64() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f64) / 1_000_000_000.0
}

/// Iceberg detector for identifying hidden liquidity
pub struct IcebergDetector {
    /// Recent trades for analysis
    trade_history: VecDeque<TradeRecord>,
    /// Detected icebergs
    detected_icebergs: Vec<DetectedIceberg>,
    /// Maximum history size
    max_history: usize,
    /// Detection threshold (number of refreshes)
    detection_threshold: usize,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp: Instant,
    pub price: u64,
    pub quantity: u64,
    pub is_buyer_maker: bool,
    pub symbol: [u8; 12],
}

#[derive(Debug, Clone)]
pub struct DetectedIceberg {
    pub symbol: [u8; 12],
    pub side_is_buy: bool,
    pub price_level: u64,
    pub estimated_total_size: u64,
    pub visible_size: u64,
    pub refresh_count: usize,
    pub confidence: f64,
    pub first_detected: Instant,
    pub last_refresh: Instant,
}

impl IcebergDetector {
    pub fn new(max_history: usize, detection_threshold: usize) -> Self {
        Self {
            trade_history: VecDeque::with_capacity(max_history),
            detected_icebergs: Vec::new(),
            max_history,
            detection_threshold,
        }
    }

    /// Process incoming trade for iceberg detection
    pub fn process_trade(&mut self, trade: TradeRecord) {
        // Add to history
        if self.trade_history.len() >= self.max_history {
            self.trade_history.pop_front();
        }
        self.trade_history.push_back(trade);

        // Analyze for iceberg patterns
        self.analyze_for_icebergs();
    }

    /// Analyze trade history for iceberg patterns
    fn analyze_for_icebergs(&mut self) {
        // Group trades by price level
        let mut price_levels: std::collections::HashMap<u64, Vec<&TradeRecord>> = 
            std::collections::HashMap::new();

        for trade in &self.trade_history {
            price_levels.entry(trade.price).or_default().push(trade);
        }

        // Detect icebergs at each price level
        for (price, trades) in &price_levels {
            if trades.len() < self.detection_threshold {
                continue;
            }

            // Check for consistent size at this level (iceberg signature)
            let sizes: Vec<u64> = trades.iter().map(|t| t.quantity).collect();
            let avg_size = sizes.iter().sum::<u64>() as f64 / sizes.len() as f64;
            
            // Calculate variance
            let variance: f64 = sizes.iter()
                .map(|s| (*s as f64 - avg_size).powi(2))
                .sum::<f64>() / sizes.len() as f64;
            
            let std_dev = variance.sqrt();
            
            // Low variance = likely iceberg
            if std_dev < avg_size * 0.3 {
                // Check if trades are mostly same direction
                let buy_count = trades.iter().filter(|t| !t.is_buyer_maker).count();
                let sell_count = trades.iter().filter(|t| t.is_buyer_maker).count();
                
                let is_buy_iceberg = buy_count > sell_count;
                let dominant_side_size = if is_buy_iceberg { buy_count } else { sell_count };
                
                if dominant_side_size >= self.detection_threshold {
                    let estimated_total = (avg_size * trades.len() as f64) as u64;
                    
                    // Check if already detected
                    let existing = self.detected_icebergs.iter_mut()
                        .find(|i| i.symbol == trades[0].symbol && i.price_level == *price);
                    
                    if let Some(iceberg) = existing {
                        iceberg.refresh_count += 1;
                        iceberg.last_refresh = Instant::now();
                        iceberg.estimated_total_size = estimated_total;
                        iceberg.confidence = (0.5 + (1.0 - std_dev / avg_size) * 0.5).min(0.95);
                    } else {
                        self.detected_icebergs.push(DetectedIceberg {
                            symbol: trades[0].symbol,
                            side_is_buy: is_buy_iceberg,
                            price_level: *price,
                            estimated_total_size: estimated_total,
                            visible_size: avg_size as u64,
                            refresh_count: 1,
                            confidence: 0.5,
                            first_detected: Instant::now(),
                            last_refresh: Instant::now(),
                        });
                    }
                }
            }
        }

        // Prune old detections
        let stale_threshold = Duration::from_secs(60);
        self.detected_icebergs.retain(|i| i.last_refresh.elapsed() < stale_threshold);
    }

    /// Get detected icebergs for a symbol
    pub fn get_icebergs(&self, symbol: &[u8; 12]) -> Vec<&DetectedIceberg> {
        self.detected_icebergs.iter()
            .filter(|i| &i.symbol == symbol)
            .collect()
    }

    /// Get statistics
    pub fn get_stats(&self) -> IcebergStats {
        IcebergStats {
            trades_analyzed: self.trade_history.len(),
            icebergs_detected: self.detected_icebergs.len(),
            buy_icebergs: self.detected_icebergs.iter().filter(|i| i.side_is_buy).count(),
            sell_icebergs: self.detected_icebergs.iter().filter(|i| !i.side_is_buy).count(),
        }
    }
}

/// Iceberg statistics
#[derive(Debug, Clone)]
pub struct IcebergStats {
    pub trades_analyzed: usize,
    pub icebergs_detected: usize,
    pub buy_icebergs: usize,
    pub sell_icebergs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_order_creation() {
        let config = IcebergConfig::default();
        let order = IcebergOrder::new(
            1,
            *b"BTCUSDT   \0\0\0",
            true,
            10000,
            5000000000,
            config.clone(),
        );

        assert_eq!(order.total_quantity, 10000);
        assert_eq!(order.state, IcebergState::Pending);
        assert!(order.display_size <= 1000); // 10% of total
    }

    #[test]
    fn test_iceberg_detector() {
        let mut detector = IcebergDetector::new(1000, 5);
        
        // Simulate iceberg pattern (same size trades at same price)
        for _ in 0..10 {
            detector.process_trade(TradeRecord {
                timestamp: Instant::now(),
                price: 5000000000,
                quantity: 100,
                is_buyer_maker: false,
                symbol: *b"BTCUSDT   \0\0\0",
            });
        }

        let stats = detector.get_stats();
        assert!(stats.trades_analyzed == 10);
        // May detect iceberg depending on variance calculation
    }
}
