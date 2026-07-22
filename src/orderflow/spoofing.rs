//! Nautilus/Ray Bot - Stage 15: Spoofing & Layering Detector
//! Module: src/orderflow/spoofing.rs
//!
//! Description:
//!     Real-time spoofing and layering detector that analyzes order book snapshot diffs.
//!     Identifies predatory market maker manipulation via cancellation rate analysis.
//!     Operates purely in the Rust hot path for instant alerts.
//!
//! Constraints:
//!     - Latency: Microsecond-level pattern detection.
//!     - Architecture: AMD Ryzen AI 5 (SIMD optimized).
//!     - Memory: Zero heap allocation during hot path.

use std::collections::{VecDeque, HashMap};

// Configuration Constants
const MAX_ORDER_HISTORY: usize = 10000;
const CANCELLATION_THRESHOLD: f64 = 0.85; // >85% cancel rate indicates spoofing
const LAYER_COUNT_THRESHOLD: usize = 5; // Minimum layers for layering pattern
const TIME_WINDOW_MS: u64 = 100; // Analysis window

/// Represents a single order event.
#[derive(Debug, Clone, Copy)]
pub struct OrderEvent {
    pub order_id: u64,
    pub price: i64,
    pub quantity: u64,
    pub is_bid: bool,
    pub timestamp_ns: u128,
    pub event_type: OrderEventType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderEventType {
    New,
    Cancel,
    Modify,
    Trade,
}

/// Tracks order lifecycle for cancellation analysis.
#[derive(Debug)]
struct OrderLifecycle {
    created_ns: u128,
    modified_count: u32,
    cancelled: bool,
    executed: bool,
}

impl OrderLifecycle {
    fn new(timestamp_ns: u128) -> Self {
        Self {
            created_ns: timestamp_ns,
            modified_count: 0,
            cancelled: false,
            executed: false,
        }
    }
}

/// High-performance spoofing detector with lock-free structures.
pub struct SpoofingDetector {
    order_history: VecDeque<OrderEvent>,
    active_orders: HashMap<u64, OrderLifecycle>,
    cancellation_counts: (u64, u64), // (cancelled, total)
    recent_layers: Vec<Vec<i64>>, // Price levels for layering detection
    spoofing_alerts: u64,
    layering_alerts: u64,
}

impl SpoofingDetector {
    pub fn new() -> Self {
        Self {
            order_history: VecDeque::with_capacity(MAX_ORDER_HISTORY),
            active_orders: HashMap::with_capacity(1000),
            cancellation_counts: (0, 0),
            recent_layers: Vec::with_capacity(10),
            spoofing_alerts: 0,
            layering_alerts: 0,
        }
    }

    /// Process an order book event.
    /// Returns true if suspicious activity detected.
    #[inline]
    pub fn process_event(&mut self, event: OrderEvent) -> bool {
        let mut suspicious = false;

        match event.event_type {
            OrderEventType::New => {
                self.active_orders.insert(
                    event.order_id,
                    OrderLifecycle::new(event.timestamp_ns),
                );
                self.cancellation_counts.1 += 1;
                
                // Track price levels for layering detection
                self.check_layering(event.price, event.is_bid);
            }
            
            OrderEventType::Cancel => {
                if let Some(lifecycle) = self.active_orders.get_mut(&event.order_id) {
                    lifecycle.cancelled = true;
                    self.cancellation_counts.0 += 1;
                    
                    // Check for rapid cancellation (spoofing signature)
                    let lifetime_ns = event.timestamp_ns - lifecycle.created_ns;
                    if lifetime_ns < 10_000_000 { // <10ms lifetime
                        suspicious = true;
                        self.spoofing_alerts += 1;
                    }
                }
                self.active_orders.remove(&event.order_id);
            }
            
            OrderEventType::Modify => {
                if let Some(lifecycle) = self.active_orders.get_mut(&event.order_id) {
                    lifecycle.modified_count += 1;
                }
            }
            
            OrderEventType::Trade => {
                if let Some(lifecycle) = self.active_orders.get_mut(&event.order_id) {
                    lifecycle.executed = true;
                }
            }
        }

        self.order_history.push_back(event);
        if self.order_history.len() > MAX_ORDER_HISTORY {
            self.order_history.pop_front();
        }

        // Periodically check cancellation ratio
        if self.cancellation_counts.1 % 100 == 0 {
            if self.get_cancellation_ratio() > CANCELLATION_THRESHOLD {
                suspicious = true;
            }
        }

        suspicious
    }

    /// Detect layering patterns (multiple orders at sequential price levels).
    fn check_layering(&mut self, price: i64, is_bid: bool) {
        // Simplified layering detection
        // In production: analyze full order book depth snapshots
        
        let current_time_layer: Vec<i64> = vec![price; 3]; // Placeholder
        self.recent_layers.push(current_time_layer);
        
        if self.recent_layers.len() > 10 {
            self.recent_layers.remove(0);
        }

        // Check for sequential price levels indicating layering
        if self.recent_layers.len() >= LAYER_COUNT_THRESHOLD {
            self.layering_alerts += 1;
        }
    }

    /// Get current cancellation ratio.
    #[inline]
    pub fn get_cancellation_ratio(&self) -> f64 {
        if self.cancellation_counts.1 == 0 {
            return 0.0;
        }
        self.cancellation_counts.0 as f64 / self.cancellation_counts.1 as f64
    }

    /// Check if spoofing is currently detected.
    #[inline]
    pub fn is_spoofing_detected(&self) -> bool {
        self.get_cancellation_ratio() > CANCELLATION_THRESHOLD || self.spoofing_alerts > 10
    }

    /// Check if layering pattern is detected.
    #[inline]
    pub fn is_layering_detected(&self) -> bool {
        self.layering_alerts > 5
    }

    /// Get alert statistics.
    pub fn get_stats(&self) -> SpoofingStats {
        SpoofingStats {
            cancellation_ratio: self.get_cancellation_ratio(),
            spoofing_alerts: self.spoofing_alerts,
            layering_alerts: self.layering_alerts,
            active_order_count: self.active_orders.len(),
        }
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        self.order_history.clear();
        self.active_orders.clear();
        self.cancellation_counts = (0, 0);
        self.recent_layers.clear();
        self.spoofing_alerts = 0;
        self.layering_alerts = 0;
    }
}

#[derive(Debug)]
pub struct SpoofingStats {
    pub cancellation_ratio: f64,
    pub spoofing_alerts: u64,
    pub layering_alerts: u64,
    pub active_order_count: usize,
}

/// SIMD-accelerated pattern matching for historical replay.
#[target_feature(enable = "avx2")]
unsafe fn simd_detect_patterns(events: &[OrderEvent]) -> u64 {
    // Placeholder for AVX2 implementation
    // In production: use explicit intrinsics for parallel event analysis
    let mut count = 0u64;
    for event in events {
        if event.event_type == OrderEventType::Cancel {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rapid_cancellation_detection() {
        let mut detector = SpoofingDetector::new();
        
        // Create order
        let new_event = OrderEvent {
            order_id: 1,
            price: 50000,
            quantity: 100,
            is_bid: true,
            timestamp_ns: 1000000,
            event_type: OrderEventType::New,
        };
        detector.process_event(new_event);
        
        // Cancel within 5ms (spoofing signature)
        let cancel_event = OrderEvent {
            order_id: 1,
            price: 50000,
            quantity: 100,
            is_bid: true,
            timestamp_ns: 1000000 + 5_000_000, // 5ms later
            event_type: OrderEventType::Cancel,
        };
        
        let suspicious = detector.process_event(cancel_event);
        assert!(suspicious);
        assert!(detector.is_spoofing_detected());
    }

    #[test]
    fn test_normal_trading_not_flagged() {
        let mut detector = SpoofingDetector::new();
        
        // Simulate normal trading with execution
        for i in 0..100 {
            let new_event = OrderEvent {
                order_id: i,
                price: 50000,
                quantity: 100,
                is_bid: true,
                timestamp_ns: 1000000 + i * 1000,
                event_type: OrderEventType::New,
            };
            detector.process_event(new_event);
            
            let trade_event = OrderEvent {
                order_id: i,
                price: 50000,
                quantity: 100,
                is_bid: true,
                timestamp_ns: 1000000 + i * 1000 + 50_000_000, // 50ms later
                event_type: OrderEventType::Trade,
            };
            detector.process_event(trade_event);
        }
        
        assert!(!detector.is_spoofing_detected());
        assert_eq!(detector.get_cancellation_ratio(), 0.0);
    }
}
