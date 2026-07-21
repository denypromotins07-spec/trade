//! Advanced Execution & Smart Order Routing - Chapter 4
//! File 10: iceberg.rs
//! 
//! Codes advanced iceberg order execution and detection logic,
//! dynamically slicing large institutional orders to hide true size
//! from predatory HFT market makers. Optimized for microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Iceberg order configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergConfig {
    /// Total order quantity (hidden)
    pub total_quantity: u64,
    /// Visible slice quantity
    pub visible_quantity: u64,
    /// Minimum slice quantity
    pub min_slice_qty: u64,
    /// Maximum slice quantity
    pub max_slice_qty: u64,
    /// Randomize slice sizes (anti-detection)
    pub randomize_slices: bool,
    /// Participation rate limit (percentage of volume)
    pub max_participation_pct: f64,
    /// Time between slices (milliseconds)
    pub slice_interval_ms: u64,
    /// Price tolerance (ticks)
    pub price_tolerance_ticks: i64,
}

/// Iceberg order state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergOrder {
    pub order_id: String,
    pub side: OrderSide,
    pub config: IcebergConfig,
    /// Filled quantity (total)
    pub filled_qty: u64,
    /// Current visible slice remaining
    pub slice_remaining: u64,
    /// Number of slices executed
    pub slices_executed: u32,
    /// Average fill price
    pub avg_fill_price: i64,
    /// Order status
    pub status: OrderStatus,
    /// Creation timestamp
    pub created_ns: u64,
    /// Last update timestamp
    pub last_update_ns: u64,
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
    PartiallyFilled,
    Completed,
    Cancelled,
    Failed,
}

impl IcebergOrder {
    /// Create new iceberg order
    pub fn new(order_id: String, side: OrderSide, config: IcebergConfig) -> Self {
        Self {
            order_id,
            side,
            config,
            filled_qty: 0,
            slice_remaining: config.visible_quantity,
            slices_executed: 0,
            avg_fill_price: 0,
            status: OrderStatus::Pending,
            created_ns: 0,
            last_update_ns: 0,
        }
    }

    /// Get remaining quantity to fill
    #[inline]
    pub fn remaining_qty(&self) -> u64 {
        self.config.total_quantity.saturating_sub(self.filled_qty)
    }

    /// Check if order is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.filled_qty >= self.config.total_quantity || self.status == OrderStatus::Completed
    }

    /// Calculate next slice size (with optional randomization)
    pub fn calculate_next_slice(&self, rng_seed: u64) -> u64 {
        let remaining = self.remaining_qty();
        
        if remaining == 0 {
            return 0;
        }

        let base_slice = self.config.visible_quantity;
        
        if self.config.randomize_slices && remaining > base_slice {
            // Randomize between min and max slice size
            let range = self.config.max_slice_qty - self.config.min_slice_qty;
            let random_offset = (rng_seed % range) as u64;
            let randomized_slice = self.config.min_slice_qty + random_offset;
            
            randomized_slice.min(remaining).min(self.config.max_slice_qty)
        } else {
            base_slice.min(remaining)
        }
    }

    /// Update order with fill
    pub fn on_fill(&mut self, fill_qty: u64, fill_price: i64, timestamp_ns: u64) {
        self.filled_qty = self.filled_qty.saturating_add(fill_qty);
        self.slice_remaining = self.slice_remaining.saturating_sub(fill_qty);
        self.last_update_ns = timestamp_ns;

        // Update average fill price
        if fill_qty > 0 {
            let total_value = (self.avg_fill_price as u128 * self.filled_qty.saturating_sub(fill_qty) as u128)
                + (fill_price as u128 * fill_qty as u128);
            self.avg_fill_price = (total_value / self.filled_qty as u128) as i64;
        }

        // Update status
        if self.is_complete() {
            self.status = OrderStatus::Completed;
        } else if self.filled_qty > 0 {
            self.status = OrderStatus::PartiallyFilled;
        } else {
            self.status = OrderStatus::Active;
        }
    }

    /// Start new slice
    pub fn start_new_slice(&mut self, slice_qty: u64) {
        self.slice_remaining = slice_qty;
        self.slices_executed += 1;
        if self.status == OrderStatus::Pending {
            self.status = OrderStatus::Active;
        }
    }
}

/// Iceberg order detector for identifying hidden institutional orders
pub struct IcebergDetector {
    /// Track fills at each price level
    fills_by_price: DashMap<i64, Vec<FillRecord>>,
    /// Detected icebergs queue
    detected_icebergs: crossbeam_queue::SegQueue<DetectedIceberg>,
    /// Configuration
    min_iceberg_confidence: f64,
    fill_lookback_count: usize,
    /// Statistics
    orders_analyzed: AtomicU64,
    icebergs_detected: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillRecord {
    pub price: i64,
    pub quantity: u64,
    pub is_buyer_maker: bool,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIceberg {
    pub price: i64,
    pub side: OrderSide,
    pub estimated_total_qty: u64,
    pub visible_qty: u64,
    pub refill_count: u32,
    pub confidence: f64,
    pub first_seen_ns: u64,
    pub last_seen_ns: u64,
}

use dashmap::DashMap;

impl IcebergDetector {
    /// Create new iceberg detector
    pub fn new(min_confidence: f64, lookback_count: usize) -> Self {
        Self {
            fills_by_price: DashMap::new(),
            detected_icebergs: crossbeam_queue::SegQueue::new(),
            min_iceberg_confidence: min_confidence,
            fill_lookback_count: lookback_count,
            orders_analyzed: AtomicU64::new(0),
            icebergs_detected: AtomicU64::new(0),
        }
    }

    /// Process a fill/trade
    pub fn process_fill(&self, fill: FillRecord) {
        // Store fill by price
        let fills = self.fills_by_price
            .entry(fill.price)
            .or_insert_with(Vec::new);
        
        fills.push(fill.clone());
        
        // Keep only recent fills
        if fills.len() > self.fill_lookback_count {
            fills.remove(0);
        }

        self.orders_analyzed.fetch_add(1, Ordering::Relaxed);

        // Analyze for iceberg patterns
        self.analyze_for_iceberg(fill.price, fill.is_buyer_maker);
    }

    /// Analyze fills at a price level for iceberg patterns
    fn analyze_for_iceberg(&self, price: i64, is_buyer_maker: bool) {
        if let Some(fills) = self.fills_by_price.get(&price) {
            if fills.len() < 5 {
                return; // Need more data
            }

            // Count refills at this price level
            let mut refill_count = 0;
            let mut total_volume = 0u64;
            let mut visible_sizes: Vec<u64> = Vec::new();
            
            let mut current_slice = 0u64;
            let mut last_qty = 0u64;

            for fill in fills.iter() {
                if fill.is_buyer_maker != is_buyer_maker {
                    continue; // Wrong side
                }

                total_volume += fill.quantity;

                // Detect slice boundaries
                if fill.quantity < last_qty / 2 {
                    // Possible new slice starting
                    if current_slice > 0 {
                        visible_sizes.push(current_slice);
                        refill_count += 1;
                    }
                    current_slice = fill.quantity;
                } else {
                    current_slice += fill.quantity;
                }
                
                last_qty = fill.quantity;
            }

            if current_slice > 0 {
                visible_sizes.push(current_slice);
            }

            // Calculate iceberg confidence
            if refill_count >= 2 && !visible_sizes.is_empty() {
                let avg_visible = visible_sizes.iter().sum::<u64>() as f64 / visible_sizes.len() as f64;
                let variance = visible_sizes.iter()
                    .map(|v| (*v as f64 - avg_visible).powi(2))
                    .sum::<f64>() / visible_sizes.len() as f64;
                
                // Low variance in slice sizes = higher confidence
                let size_consistency = 1.0 - (variance.sqrt() / avg_visible).min(1.0);
                let refill_factor = (refill_count as f64 / 10.0).min(1.0);
                
                let confidence = size_consistency * 0.6 + refill_factor * 0.4;

                if confidence >= self.min_iceberg_confidence {
                    let iceberg = DetectedIceberg {
                        price,
                        side: if is_buyer_maker { OrderSide::Sell } else { OrderSide::Buy },
                        estimated_total_qty: total_volume,
                        visible_qty: avg_visible as u64,
                        refill_count,
                        confidence,
                        first_seen_ns: fills.first().map(|f| f.timestamp_ns).unwrap_or(0),
                        last_seen_ns: fills.last().map(|f| f.timestamp_ns).unwrap_or(0),
                    };

                    self.detected_icebergs.push(iceberg);
                    self.icebergs_detected.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Poll detected icebergs
    pub fn poll_detected_icebergs(&self) -> Vec<DetectedIceberg> {
        let mut icebergs = Vec::new();
        while let Ok(iceberg) = self.detected_icebergs.pop() {
            icebergs.push(iceberg);
        }
        icebergs
    }

    /// Get statistics
    pub fn get_statistics(&self) -> IcebergStats {
        IcebergStats {
            price_levels_tracked: self.fills_by_price.len(),
            orders_analyzed: self.orders_analyzed.load(Ordering::Relaxed),
            icebergs_detected: self.icebergs_detected.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergStats {
    pub price_levels_tracked: usize,
    pub orders_analyzed: u64,
    pub icebergs_detected: u64,
}

/// Iceberg order executor for splitting large orders
pub struct IcebergExecutor {
    /// Active iceberg orders
    active_orders: DashMap<String, IcebergOrder>,
    /// Detector for market icebergs
    detector: IcebergDetector,
    /// Order counter
    order_counter: AtomicU64,
    /// Execution enabled flag
    execution_enabled: AtomicBool,
}

impl IcebergExecutor {
    /// Create new iceberg executor
    pub fn new(detector_config: Option<(f64, usize)>) -> Self {
        let (conf, lookback) = detector_config.unwrap_or((0.7, 50));
        
        Self {
            active_orders: DashMap::new(),
            detector: IcebergDetector::new(conf, lookback),
            order_counter: AtomicU64::new(0),
            execution_enabled: AtomicBool::new(true),
        }
    }

    /// Create and submit new iceberg order
    pub fn submit_iceberg_order(
        &self,
        side: OrderSide,
        total_qty: u64,
        visible_qty: u64,
        price: i64,
    ) -> String {
        let order_id = format!("IBB-{}", self.order_counter.fetch_add(1, Ordering::Relaxed));
        
        let config = IcebergConfig {
            total_quantity: total_qty,
            visible_quantity: visible_qty,
            min_slice_qty: visible_qty / 2,
            max_slice_qty: visible_qty * 2,
            randomize_slices: true,
            max_participation_pct: 10.0,
            slice_interval_ms: 100,
            price_tolerance_ticks: 5,
        };

        let mut order = IcebergOrder::new(order_id.clone(), side, config);
        order.start_new_slice(visible_qty);
        
        self.active_orders.insert(order_id.clone(), order);
        
        order_id
    }

    /// Process fill for an iceberg order
    pub fn process_order_fill(
        &self,
        order_id: &str,
        fill_qty: u64,
        fill_price: i64,
        timestamp_ns: u64,
    ) -> Option<IcebergOrder> {
        let mut order = self.active_orders.get_mut(order_id)?;
        
        order.on_fill(fill_qty, fill_price, timestamp_ns);

        // Check if we need to refresh the slice
        if order.slice_remaining == 0 && !order.is_complete() {
            let next_slice = order.calculate_next_slice(timestamp_ns);
            order.start_new_slice(next_slice);
        }

        Some(order.clone())
    }

    /// Cancel an iceberg order
    pub fn cancel_order(&self, order_id: &str) -> Option<IcebergOrder> {
        let mut order = self.active_orders.get_mut(order_id)?;
        order.status = OrderStatus::Cancelled;
        Some(order.clone())
    }

    /// Get active order status
    pub fn get_order_status(&self, order_id: &str) -> Option<IcebergOrder> {
        self.active_orders.get(order_id).map(|o| o.clone())
    }

    /// Get all active orders
    pub fn get_all_active_orders(&self) -> Vec<IcebergOrder> {
        self.active_orders.iter()
            .filter(|o| o.value().status == OrderStatus::Active || o.value().status == OrderStatus::PartiallyFilled)
            .map(|o| o.value().clone())
            .collect()
    }

    /// Get the detector reference
    pub fn get_detector(&self) -> &IcebergDetector {
        &self.detector
    }

    /// Enable/disable execution
    pub fn set_execution_enabled(&self, enabled: bool) {
        self.execution_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if execution is enabled
    pub fn is_execution_enabled(&self) -> bool {
        self.execution_enabled.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iceberg_order_basic() {
        let config = IcebergConfig {
            total_quantity: 10000,
            visible_quantity: 1000,
            min_slice_qty: 500,
            max_slice_qty: 2000,
            randomize_slices: false,
            max_participation_pct: 10.0,
            slice_interval_ms: 100,
            price_tolerance_ticks: 5,
        };

        let mut order = IcebergOrder::new("test-1".to_string(), OrderSide::Buy, config);
        order.created_ns = 1000000;
        
        assert_eq!(order.remaining_qty(), 10000);
        assert_eq!(order.calculate_next_slice(12345), 1000);

        // Simulate fills
        order.on_fill(500, 5000000000, 2000000);
        assert_eq!(order.filled_qty, 500);
        assert_eq!(order.remaining_qty(), 9500);
        
        order.on_fill(500, 5000000000, 3000000);
        assert_eq!(order.slice_remaining, 0);
        
        // Start new slice
        order.start_new_slice(1000);
        assert_eq!(order.slices_executed, 2);
    }

    #[test]
    fn test_iceberg_detector() {
        let detector = IcebergDetector::new(0.5, 50);
        
        // Simulate fills at same price (iceberg pattern)
        for i in 0..10 {
            let fill = FillRecord {
                price: 5000000000,
                quantity: 100,
                is_buyer_maker: false,
                timestamp_ns: 1000000 + i * 100000,
            };
            detector.process_fill(fill);
        }

        let stats = detector.get_statistics();
        assert_eq!(stats.orders_analyzed, 10);
    }
}
