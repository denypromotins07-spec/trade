//! Advanced Execution & Smart Order Routing - Chapter 4
//! File 12: pov.rs
//! 
//! Implements a Percentage of Volume (POV) execution algorithm that
//! dynamically adjusts the bot's market participation rate to blend
//! seamlessly with natural market volume. Optimized for microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// POV execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct POVConfig {
    /// Target participation rate (percentage, e.g., 10 = 10%)
    pub target_participation_pct: f64,
    /// Minimum participation rate
    pub min_participation_pct: f64,
    /// Maximum participation rate
    pub max_participation_pct: f64,
    /// Aggressiveness factor (adjusts based on urgency)
    pub aggressiveness_factor: f64,
    /// Minimum order quantity per interval
    pub min_order_qty: u64,
    /// Maximum order quantity per interval
    pub max_order_qty: u64,
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Price tolerance from VWAP (ticks)
    pub vwap_tolerance_ticks: i64,
    /// Time horizon for execution (milliseconds)
    pub time_horizon_ms: u64,
}

impl Default for POVConfig {
    fn default() -> Self {
        Self {
            target_participation_pct: 10.0,
            min_participation_pct: 5.0,
            max_participation_pct: 25.0,
            aggressiveness_factor: 1.0,
            min_order_qty: 100,
            max_order_qty: 10000,
            update_interval_ms: 100,
            vwap_tolerance_ticks: 50,
            time_horizon_ms: 3600000, // 1 hour
        }
    }
}

/// POV order state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct POVOrder {
    pub order_id: String,
    pub side: OrderSide,
    pub symbol: String,
    pub total_qty: u64,
    pub filled_qty: u64,
    pub config: POVConfig,
    /// Start timestamp
    pub start_time_ns: u64,
    /// Expected end timestamp
    pub end_time_ns: u64,
    /// Last calculation timestamp
    pub last_calc_ns: u64,
    /// Market volume since start
    pub market_volume: u64,
    /// Our volume since start
    pub our_volume: u64,
    /// Current participation rate
    pub current_participation_pct: f64,
    /// Status
    pub status: OrderStatus,
    /// Average fill price
    pub avg_fill_price: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderStatus {
    Pending,
    Active,
    Paused,
    Completed,
    Cancelled,
    Failed,
}

impl POVOrder {
    /// Create new POV order
    pub fn new(
        order_id: String,
        side: OrderSide,
        symbol: String,
        total_qty: u64,
        config: POVConfig,
        start_time_ns: u64,
    ) -> Self {
        let end_time_ns = start_time_ns + (config.time_horizon_ms * 1_000_000);
        
        Self {
            order_id,
            side,
            symbol,
            total_qty,
            filled_qty: 0,
            config,
            start_time_ns,
            end_time_ns,
            last_calc_ns: start_time_ns,
            market_volume: 0,
            our_volume: 0,
            current_participation_pct: 0.0,
            status: OrderStatus::Pending,
            avg_fill_price: 0,
        }
    }

    /// Get remaining quantity
    #[inline]
    pub fn remaining_qty(&self) -> u64 {
        self.total_qty.saturating_sub(self.filled_qty)
    }

    /// Check if order is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_qty() == 0 || self.status == OrderStatus::Completed
    }

    /// Calculate time elapsed ratio (0.0 to 1.0)
    #[inline]
    pub fn time_elapsed_ratio(&self, current_time_ns: u64) -> f64 {
        let elapsed = current_time_ns.saturating_sub(self.start_time_ns) as f64;
        let total = (self.end_time_ns - self.start_time_ns) as f64;
        (elapsed / total).clamp(0.0, 1.0)
    }

    /// Calculate fill progress ratio (0.0 to 1.0)
    #[inline]
    pub fn fill_progress_ratio(&self) -> f64 {
        if self.total_qty == 0 {
            return 1.0;
        }
        (self.filled_qty as f64 / self.total_qty as f64).clamp(0.0, 1.0)
    }

    /// Calculate dynamic participation rate based on progress and urgency
    pub fn calculate_participation_rate(&mut self, current_time_ns: u64) -> f64 {
        self.last_calc_ns = current_time_ns;
        
        let time_ratio = self.time_elapsed_ratio(current_time_ns);
        let fill_ratio = self.fill_progress_ratio();
        
        // Base participation rate
        let mut participation = self.config.target_participation_pct;
        
        // Adjust based on progress vs time
        // If behind schedule, increase participation
        // If ahead of schedule, decrease participation
        let schedule_diff = fill_ratio - time_ratio;
        
        if schedule_diff < -0.1 {
            // Significantly behind - increase aggression
            participation *= self.config.aggressiveness_factor * 1.5;
        } else if schedule_diff > 0.1 {
            // Ahead of schedule - reduce aggression
            participation *= 0.7;
        }
        
        // Apply min/max bounds
        participation = participation.clamp(
            self.config.min_participation_pct,
            self.config.max_participation_pct,
        );
        
        self.current_participation_pct = participation;
        participation
    }

    /// Calculate order quantity for current interval
    pub fn calculate_interval_qty(
        &self,
        market_volume_interval: u64,
        participation_rate: f64,
    ) -> u64 {
        // Calculate target quantity based on market volume and participation
        let target_qty = (market_volume_interval as f64 * participation_rate / 100.0) as u64;
        
        // Ensure within limits
        let qty = target_qty
            .max(self.config.min_order_qty)
            .min(self.config.max_order_qty)
            .min(self.remaining_qty());
        
        qty
    }

    /// Update order with fill
    pub fn on_fill(&mut self, fill_qty: u64, fill_price: i64, timestamp_ns: u64) {
        self.filled_qty = self.filled_qty.saturating_add(fill_qty);
        self.our_volume = self.our_volume.saturating_add(fill_qty);
        
        // Update average fill price
        if fill_qty > 0 {
            let total_value = (self.avg_fill_price as u128 * self.filled_qty.saturating_sub(fill_qty) as u128)
                + (fill_price as u128 * fill_qty as u128);
            self.avg_fill_price = (total_value / self.filled_qty as u128) as i64;
        }
        
        // Update status
        if self.is_complete() {
            self.status = OrderStatus::Completed;
        } else if self.status == OrderStatus::Pending {
            self.status = OrderStatus::Active;
        }
    }

    /// Update market volume tracking
    pub fn update_market_volume(&mut self, volume: u64) {
        self.market_volume = volume;
    }

    /// Check if order should be paused (e.g., price moved too far from VWAP)
    pub fn should_pause(&self, current_price: i64, vwap: i64) -> bool {
        let deviation = (current_price - vwap).abs();
        deviation > self.config.vwap_tolerance_ticks
    }
}

/// Real-time volume tracker for POV calculations
pub struct VolumeTracker {
    /// Rolling market volume windows
    volume_windows: parking_lot::RwLock<Vec<VolumeWindow>>,
    /// Window size in nanoseconds
    window_size_ns: u64,
    /// Number of windows to track
    num_windows: usize,
    /// Total tracked volume
    total_volume: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct VolumeWindow {
    pub start_ns: u64,
    pub end_ns: u64,
    pub volume: u64,
    pub trade_count: u32,
}

impl VolumeTracker {
    /// Create new volume tracker
    pub fn new(window_size_ms: u64, num_windows: usize) -> Self {
        let window_size_ns = window_size_ms * 1_000_000;
        
        Self {
            volume_windows: parking_lot::RwLock::new(Vec::with_capacity(num_windows)),
            window_size_ns,
            num_windows,
            total_volume: AtomicU64::new(0),
        }
    }

    /// Process a trade
    pub fn process_trade(&self, volume: u64, timestamp_ns: u64) {
        let mut windows = self.volume_windows.write();
        
        // Find or create appropriate window
        let window_idx = ((timestamp_ns / self.window_size_ns) as usize) % self.num_windows;
        
        if window_idx >= windows.len() {
            // Initialize new window
            let start_ns = (timestamp_ns / self.window_size_ns) * self.window_size_ns;
            windows.push(VolumeWindow {
                start_ns,
                end_ns: start_ns + self.window_size_ns,
                volume,
                trade_count: 1,
            });
        } else {
            // Update existing window
            windows[window_idx].volume += volume;
            windows[window_idx].trade_count += 1;
        }
        
        self.total_volume.fetch_add(volume, Ordering::Relaxed);
    }

    /// Get recent market volume over specified period
    pub fn get_recent_volume(&self, period_ms: u64) -> u64 {
        let windows = self.volume_windows.read();
        let period_ns = period_ms * 1_000_000;
        let num_periods = (period_ns / self.window_size_ns) as usize;
        
        windows.iter()
            .rev()
            .take(num_periods.min(windows.len()))
            .map(|w| w.volume)
            .sum()
    }

    /// Get volume-weighted average price estimate
    pub fn get_vwap_estimate(&self, prices: &[i64], volumes: &[u64]) -> f64 {
        if prices.is_empty() || volumes.is_empty() || prices.len() != volumes.len() {
            return 0.0;
        }
        
        let mut total_value = 0u128;
        let mut total_volume = 0u64;
        
        for (price, vol) in prices.iter().zip(volumes.iter()) {
            total_value += (*price as u128) * (*vol as u128);
            total_volume += vol;
        }
        
        if total_volume == 0 {
            return 0.0;
        }
        
        (total_value / total_volume as u128) as f64
    }

    /// Get total tracked volume
    pub fn get_total_volume(&self) -> u64 {
        self.total_volume.load(Ordering::Relaxed)
    }
}

/// POV execution engine
pub struct POVExecutor {
    /// Active POV orders
    active_orders: DashMap<String, POVOrder>,
    /// Volume tracker
    volume_tracker: VolumeTracker,
    /// Order counter
    order_counter: AtomicU64,
    /// Execution enabled
    execution_enabled: AtomicBool,
    /// Recent market prices for VWAP calculation
    recent_prices: parking_lot::RwLock<Vec<(i64, u64)>>, // (price, volume)
}

use dashmap::DashMap;

impl POVExecutor {
    /// Create new POV executor
    pub fn new(volume_window_ms: u64, num_windows: usize) -> Self {
        Self {
            active_orders: DashMap::new(),
            volume_tracker: VolumeTracker::new(volume_window_ms, num_windows),
            order_counter: AtomicU64::new(0),
            execution_enabled: AtomicBool::new(true),
            recent_prices: parking_lot::RwLock::new(Vec::with_capacity(1000)),
        }
    }

    /// Submit new POV order
    pub fn submit_pov_order(
        &self,
        side: OrderSide,
        symbol: String,
        total_qty: u64,
        config: POVConfig,
    ) -> String {
        let order_id = format!("POV-{}", self.order_counter.fetch_add(1, Ordering::Relaxed));
        let start_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let mut order = POVOrder::new(
            order_id.clone(),
            side,
            symbol,
            total_qty,
            config,
            start_time_ns,
        );
        order.status = OrderStatus::Active;
        
        self.active_orders.insert(order_id.clone(), order);
        
        order_id
    }

    /// Process market data and calculate next order
    pub fn process_market_data(
        &self,
        symbol: &str,
        price: i64,
        volume: u64,
        timestamp_ns: u64,
    ) -> Vec<OrderInstruction> {
        // Update volume tracker
        self.volume_tracker.process_trade(volume, timestamp_ns);
        
        // Track recent prices
        {
            let mut prices = self.recent_prices.write();
            prices.push((price, volume));
            if prices.len() > 1000 {
                prices.remove(0);
            }
        }
        
        let mut instructions = Vec::new();
        
        // Process each active order for this symbol
        let order_ids: Vec<_> = self.active_orders.iter()
            .filter(|o| o.value().symbol == symbol && o.value().status == OrderStatus::Active)
            .map(|o| o.key().clone())
            .collect();
        
        for order_id in order_ids {
            if let Some(mut order) = self.active_orders.get_mut(&order_id) {
                // Check if should pause
                let vwap = self.calculate_vwap();
                if order.should_pause(price, vwap as i64) {
                    order.status = OrderStatus::Paused;
                    continue;
                }
                
                // Calculate participation rate
                let participation = order.calculate_participation_rate(timestamp_ns);
                
                // Get recent market volume
                let market_vol = self.volume_tracker.get_recent_volume(order.config.update_interval_ms);
                
                // Calculate order quantity
                let qty = order.calculate_interval_qty(market_vol, participation);
                
                if qty >= order.config.min_order_qty {
                    instructions.push(OrderInstruction {
                        order_id: order_id.clone(),
                        side: order.side,
                        quantity: qty,
                        price: price,
                        urgency: if order.time_elapsed_ratio(timestamp_ns) > 0.8 {
                            Urgency::High
                        } else {
                            Urgency::Normal
                        },
                    });
                }
            }
        }
        
        instructions
    }

    /// Calculate current VWAP from recent prices
    fn calculate_vwap(&self) -> f64 {
        let prices = self.recent_prices.read();
        let price_slice: Vec<i64> = prices.iter().map(|(p, _)| *p).collect();
        let vol_slice: Vec<u64> = prices.iter().map(|(_, v)| *v).collect();
        self.volume_tracker.get_vwap_estimate(&price_slice, &vol_slice)
    }

    /// Process fill for POV order
    pub fn process_order_fill(
        &self,
        order_id: &str,
        fill_qty: u64,
        fill_price: i64,
        timestamp_ns: u64,
    ) -> Option<POVOrder> {
        let mut order = self.active_orders.get_mut(order_id)?;
        order.on_fill(fill_qty, fill_price, timestamp_ns);
        Some(order.clone())
    }

    /// Cancel POV order
    pub fn cancel_order(&self, order_id: &str) -> Option<POVOrder> {
        let mut order = self.active_orders.get_mut(order_id)?;
        order.status = OrderStatus::Cancelled;
        Some(order.clone())
    }

    /// Get order status
    pub fn get_order_status(&self, order_id: &str) -> Option<POVOrder> {
        self.active_orders.get(order_id).map(|o| o.clone())
    }

    /// Enable/disable execution
    pub fn set_execution_enabled(&self, enabled: bool) {
        self.execution_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<POVOrder> {
        self.active_orders.iter()
            .filter(|o| o.value().status == OrderStatus::Active || o.value().status == OrderStatus::Paused)
            .map(|o| o.value().clone())
            .collect()
    }
}

/// Order instruction generated by POV executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderInstruction {
    pub order_id: String,
    pub side: OrderSide,
    pub quantity: u64,
    pub price: i64,
    pub urgency: Urgency,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Urgency {
    Low,
    Normal,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pov_order_basic() {
        let config = POVConfig {
            target_participation_pct: 10.0,
            min_participation_pct: 5.0,
            max_participation_pct: 20.0,
            ..Default::default()
        };

        let start_time = 1000000000u64;
        let mut order = POVOrder::new(
            "test-pov-1".to_string(),
            OrderSide::Buy,
            "BTCUSDT".to_string(),
            10000,
            config,
            start_time,
        );

        assert_eq!(order.remaining_qty(), 10000);
        assert_eq!(order.time_elapsed_ratio(start_time), 0.0);

        // Simulate time passing (halfway through)
        let half_time = start_time + (order.config.time_horizon_ms / 2) * 1_000_000;
        let participation = order.calculate_participation_rate(half_time);
        
        assert!(participation >= order.config.min_participation_pct);
        assert!(participation <= order.config.max_participation_pct);

        // Simulate fills
        order.on_fill(1000, 60000000000, half_time);
        assert_eq!(order.filled_qty, 1000);
        assert_eq!(order.fill_progress_ratio(), 0.1);
    }

    #[test]
    fn test_volume_tracker() {
        let tracker = VolumeTracker::new(1000, 10); // 1 second windows
        
        tracker.process_trade(100, 1000000);
        tracker.process_trade(200, 1500000);
        tracker.process_trade(150, 2500000);
        
        assert!(tracker.get_total_volume() > 0);
        
        let recent_vol = tracker.get_recent_volume(5000); // 5 seconds
        assert!(recent_vol > 0);
    }

    #[test]
    fn test_pov_executor() {
        let executor = POVExecutor::new(100, 10);
        
        let config = POVConfig::default();
        let order_id = executor.submit_pov_order(
            OrderSide::Buy,
            "BTCUSDT".to_string(),
            10000,
            config,
        );
        
        assert!(!order_id.is_empty());
        
        // Process some market data
        let instructions = executor.process_market_data(
            "BTCUSDT",
            60000000000,
            1000,
            1000000000,
        );
        
        // May or may not generate instructions depending on timing
        let _ = instructions;
        
        let status = executor.get_order_status(&order_id);
        assert!(status.is_some());
    }
}
