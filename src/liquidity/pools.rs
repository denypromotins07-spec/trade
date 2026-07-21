//! Liquidity Engineering & Market Manipulation Detection - Chapter 2
//! File 4: pools.rs
//! 
//! Constructs a real-time resting liquidity mapper that identifies spoofing,
//! layering, and quote stuffing by tracking order book modifications at the
//! microsecond level. Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use dashmap::DashMap;

/// Represents a single order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: i64,
    pub quantity: u64,
    pub order_count: u32,
    pub last_update_ns: u64,
}

impl OrderBookLevel {
    pub fn new(price: i64) -> Self {
        Self {
            price,
            quantity: 0,
            order_count: 0,
            last_update_ns: 0,
        }
    }
}

/// Detected manipulation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManipulationEvent {
    pub event_type: ManipulationType,
    pub timestamp_ns: u64,
    pub price_levels: Vec<i64>,
    pub confidence: f64,
    pub total_volume: u64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ManipulationType {
    /// Spoofing: Large order placed and quickly cancelled
    Spoofing,
    /// Layering: Multiple orders at sequential price levels
    Layering,
    /// Quote Stuffing: Rapid order submissions/cancellations
    QuoteStuffing,
    /// Wash Trading: Self-matching patterns
    WashTrading,
    /// Momentum Ignition: Orders designed to trigger stop losses
    MomentumIgnition,
}

/// Order modification tracker for detecting patterns
#[derive(Debug, Clone)]
pub struct OrderModification {
    pub price: i64,
    pub quantity: u64,
    pub is_bid: bool,
    pub is_insert: bool, // true = insert/update, false = cancel
    pub timestamp_ns: u64,
}

/// Real-time liquidity pool analyzer
pub struct LiquidityPoolAnalyzer {
    /// Bid side order book snapshot
    bids: DashMap<i64, OrderBookLevel>,
    /// Ask side order book snapshot
    asks: DashMap<i64, OrderBookLevel>,
    /// Recent order modifications (ring buffer)
    modifications: parking_lot::Mutex<Vec<OrderModification>>,
    mod_buffer_size: usize,
    mod_index: AtomicUsize,
    /// Counters for detection
    total_modifications: AtomicU64,
    cancellations_last_ms: AtomicU64,
    /// Detected events queue
    events_queue: crossbeam_queue::SegQueue<ManipulationEvent>,
    /// Configuration
    spoof_threshold_qty: u64,
    spoof_cancel_window_ns: u64,
    layering_min_levels: usize,
    quote_stuffing_threshold: u64,
}

impl LiquidityPoolAnalyzer {
    /// Create new liquidity pool analyzer
    pub fn new(
        spoof_threshold_qty: u64,
        layering_min_levels: usize,
        quote_stuffing_threshold: u64,
    ) -> Self {
        Self {
            bids: DashMap::with_capacity(1024),
            asks: DashMap::with_capacity(1024),
            modifications: parking_lot::Mutex::new(Vec::with_capacity(10000)),
            mod_buffer_size: 10000,
            mod_index: AtomicUsize::new(0),
            total_modifications: AtomicU64::new(0),
            cancellations_last_ms: AtomicU64::new(0),
            events_queue: crossbeam_queue::SegQueue::new(),
            spoof_threshold_qty,
            spoof_cancel_window_ns: 100_000_000, // 100ms
            layering_min_levels,
            quote_stuffing_threshold,
        }
    }

    /// Process order book update (Binance L2 format)
    pub fn process_orderbook_update(
        &self,
        price: i64,
        quantity: u64,
        is_bid: bool,
        timestamp_ns: u64,
    ) {
        let book = if is_bid { &self.bids } else { &self.asks };

        if quantity == 0 {
            // Cancel/remove level
            book.remove(&price);
            self.record_modification(OrderModification {
                price,
                quantity: 0,
                is_bid,
                is_insert: false,
                timestamp_ns,
            });
        } else {
            // Insert/update level
            let mut entry = book.entry(price).or_insert_with(|| OrderBookLevel::new(price));
            entry.quantity = quantity;
            entry.last_update_ns = timestamp_ns;
            
            self.record_modification(OrderModification {
                price,
                quantity,
                is_bid,
                is_insert: true,
                timestamp_ns,
            });
        }

        self.total_modifications.fetch_add(1, Ordering::Relaxed);

        // Run manipulation detection periodically
        if self.total_modifications.load(Ordering::Relaxed) % 10 == 0 {
            self.detect_manipulation(timestamp_ns);
        }
    }

    /// Record order modification for pattern analysis
    fn record_modification(&self, modification: OrderModification) {
        let mut mods = self.modifications.lock();
        let idx = self.mod_index.fetch_add(1, Ordering::Relaxed);
        
        if mods.len() < self.mod_buffer_size {
            mods.push(modification);
        } else {
            let wrap_idx = idx % self.mod_buffer_size;
            mods[wrap_idx] = modification;
        }

        // Track cancellations per millisecond for quote stuffing detection
        if !modification.is_insert {
            let ms = modification.timestamp_ns / 1_000_000;
            self.cancellations_last_ms.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Detect manipulation patterns
    fn detect_manipulation(&self, current_time_ns: u64) {
        let mods = self.modifications.lock();
        if mods.len() < 10 {
            return;
        }

        // Get recent modifications (last 100)
        let start_idx = mods.len().saturating_sub(100);
        let recent: Vec<_> = mods[start_idx..].to_vec();

        // Detect spoofing
        self.detect_spoofing(&recent, current_time_ns);

        // Detect layering
        self.detect_layering(&recent, current_time_ns);

        // Detect quote stuffing
        self.detect_quote_stuffing(&recent, current_time_ns);
    }

    /// Detect spoofing: large orders placed and quickly cancelled
    fn detect_spoofing(&self, mods: &[OrderModification], current_time_ns: u64) {
        // Group by price level
        let mut inserts: std::collections::HashMap<i64, &OrderModification> = std::collections::HashMap::new();
        let mut cancels: std::collections::HashSet<i64> = std::collections::HashSet::new();

        for m in mods.iter().rev() {
            if m.is_insert && m.quantity >= self.spoof_threshold_qty {
                if !inserts.contains_key(&m.price) {
                    inserts.insert(m.price, m);
                }
            }
            if !m.is_insert {
                cancels.insert(m.price);
            }
        }

        // Check for large inserts that were quickly cancelled
        for (price, insert_mod) in &inserts {
            if cancels.contains(price) {
                // Find the cancel time
                let cancel_time = mods.iter()
                    .find(|m| !m.is_insert && m.price == *price)
                    .map(|m| m.timestamp_ns)
                    .unwrap_or(current_time_ns);

                let time_diff = cancel_time - insert_mod.timestamp_ns;
                
                if time_diff < self.spoof_cancel_window_ns {
                    let event = ManipulationEvent {
                        event_type: ManipulationType::Spoofing,
                        timestamp_ns: current_time_ns,
                        price_levels: vec![*price],
                        confidence: (1.0 - (time_diff as f64 / self.spoof_cancel_window_ns as f64)).min(1.0),
                        total_volume: insert_mod.quantity,
                        description: format!(
                            "Spoofing detected: {} quantity at {} cancelled within {}ms",
                            insert_mod.quantity,
                            price,
                            time_diff / 1_000_000
                        ),
                    };
                    self.events_queue.push(event);
                }
            }
        }
    }

    /// Detect layering: multiple orders at sequential price levels
    fn detect_layering(&self, mods: &[OrderModification], current_time_ns: u64) {
        // Group active bid or ask inserts by side
        let mut bid_prices: Vec<i64> = Vec::new();
        let mut ask_prices: Vec<i64> = Vec::new();
        let mut bid_volumes: std::collections::HashMap<i64, u64> = std::collections::HashMap::new();
        let mut ask_volumes: std::collections::HashMap<i64, u64> = std::collections::HashMap::new();

        for m in mods.iter() {
            if m.is_insert {
                if m.is_bid {
                    if !bid_prices.contains(&m.price) {
                        bid_prices.push(m.price);
                    }
                    *bid_volumes.entry(m.price).or_insert(0) += m.quantity;
                } else {
                    if !ask_prices.contains(&m.price) {
                        ask_prices.push(m.price);
                    }
                    *ask_volumes.entry(m.price).or_insert(0) += m.quantity;
                }
            }
        }

        // Check for sequential price levels (layering pattern)
        for (prices, volumes, is_bid) in [
            (&mut bid_prices, &bid_volumes, true),
            (&mut ask_prices, &ask_volumes, false),
        ] {
            if prices.len() < self.layering_min_levels {
                continue;
            }

            prices.sort();

            // Check for sequential levels with similar volumes
            let mut layer_count = 1;
            let mut layer_start = 0;
            
            for i in 1..prices.len() {
                let price_diff = (prices[i] - prices[i - 1]).abs();
                let vol1 = volumes.get(&prices[i - 1]).copied().unwrap_or(0);
                let vol2 = volumes.get(&prices[i]).copied().unwrap_or(0);
                
                // Check if prices are sequential and volumes are similar
                if price_diff <= 50000000 && // Within 0.5%
                   (vol1 as f64 - vol2 as f64).abs() / (vol1.max(vol2) as f64) < 0.3 {
                    layer_count += 1;
                } else {
                    if layer_count >= self.layering_min_levels {
                        let layer_prices: Vec<i64> = prices[layer_start..i].to_vec();
                        let total_vol: u64 = layer_prices.iter()
                            .map(|p| volumes.get(p).copied().unwrap_or(0))
                            .sum();

                        let event = ManipulationEvent {
                            event_type: ManipulationType::Layering,
                            timestamp_ns: current_time_ns,
                            price_levels: layer_prices,
                            confidence: (layer_count as f64 / 10.0).min(1.0),
                            total_volume: total_vol,
                            description: format!(
                                "Layering detected: {} sequential {} levels with {} total volume",
                                layer_count,
                                if is_bid { "bid" } else { "ask" },
                                total_vol
                            ),
                        };
                        self.events_queue.push(event);
                    }
                    layer_count = 1;
                    layer_start = i;
                }
            }

            // Check final segment
            if layer_count >= self.layering_min_levels {
                let layer_prices: Vec<i64> = prices[layer_start..].to_vec();
                let total_vol: u64 = layer_prices.iter()
                    .map(|p| volumes.get(p).copied().unwrap_or(0))
                    .sum();

                let event = ManipulationEvent {
                    event_type: ManipulationType::Layering,
                    timestamp_ns: current_time_ns,
                    price_levels: layer_prices,
                    confidence: (layer_count as f64 / 10.0).min(1.0),
                    total_volume: total_vol,
                    description: format!(
                        "Layering detected: {} sequential {} levels with {} total volume",
                        layer_count,
                        if is_bid { "bid" } else { "ask" },
                        total_vol
                    ),
                };
                self.events_queue.push(event);
            }
        }
    }

    /// Detect quote stuffing: excessive order modifications
    fn detect_quote_stuffing(&self, mods: &[OrderModification], current_time_ns: u64) {
        // Count modifications in last 100ms
        let window_ns = 100_000_000;
        let cutoff = current_time_ns.saturating_sub(window_ns);
        
        let count = mods.iter()
            .filter(|m| m.timestamp_ns >= cutoff)
            .count() as u64;

        if count >= self.quote_stuffing_threshold {
            let cancel_ratio = mods.iter()
                .filter(|m| m.timestamp_ns >= cutoff && !m.is_insert)
                .count() as f64 / count as f64;

            let event = ManipulationEvent {
                event_type: ManipulationType::QuoteStuffing,
                timestamp_ns: current_time_ns,
                price_levels: vec![],
                confidence: (count as f64 / (self.quote_stuffing_threshold as f64 * 2.0)).min(1.0),
                total_volume: 0,
                description: format!(
                    "Quote stuffing detected: {} modifications in 100ms ({}% cancellations)",
                    count,
                    cancel_ratio * 100.0
                ),
            };
            self.events_queue.push(event);
        }
    }

    /// Poll detected manipulation events
    pub fn poll_events(&self) -> Vec<ManipulationEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.events_queue.pop() {
            events.push(event);
        }
        events
    }

    /// Get best bid price
    pub fn get_best_bid(&self) -> Option<i64> {
        self.bids.iter()
            .map(|entry| *entry.key())
            .max()
    }

    /// Get best ask price
    pub fn get_best_ask(&self) -> Option<i64> {
        self.asks.iter()
            .map(|entry| *entry.key())
            .min()
    }

    /// Get mid price
    pub fn get_mid_price(&self) -> Option<f64> {
        match (self.get_best_bid(), self.get_best_ask()) {
            (Some(bid), Some(ask)) => Some((bid as f64 + ask as f64) / 2.0),
            _ => None,
        }
    }

    /// Get spread in ticks
    pub fn get_spread(&self) -> Option<i64> {
        match (self.get_best_bid(), self.get_best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get total resting liquidity on bid side
    pub fn get_bid_liquidity(&self) -> u64 {
        self.bids.iter().map(|e| e.value().quantity).sum()
    }

    /// Get total resting liquidity on ask side
    pub fn get_ask_liquidity(&self) -> u64 {
        self.asks.iter().map(|e| e.value().quantity).sum()
    }

    /// Get order book imbalance
    pub fn get_book_imbalance(&self) -> f64 {
        let bid_liq = self.get_bid_liquidity() as f64;
        let ask_liq = self.get_ask_liquidity() as f64;
        let total = bid_liq + ask_liq;
        
        if total <= 0.0 {
            return 0.0;
        }
        
        (bid_liq - ask_liq) / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_basic() {
        let analyzer = LiquidityPoolAnalyzer::new(1000, 3, 100);
        
        // Add some bids
        analyzer.process_orderbook_update(5000000000, 100, true, 1000000);
        analyzer.process_orderbook_update(4999900000, 200, true, 1000001);
        
        // Add some asks
        analyzer.process_orderbook_update(5000100000, 150, false, 1000002);
        
        assert_eq!(analyzer.get_best_bid(), Some(5000000000));
        assert_eq!(analyzer.get_best_ask(), Some(5000100000));
        assert_eq!(analyzer.get_bid_liquidity(), 300);
        assert_eq!(analyzer.get_ask_liquidity(), 150);
    }

    #[test]
    fn test_book_imbalance() {
        let analyzer = LiquidityPoolAnalyzer::new(1000, 3, 100);
        
        analyzer.process_orderbook_update(5000000000, 300, true, 1000000);
        analyzer.process_orderbook_update(5000100000, 100, false, 1000001);
        
        let imbalance = analyzer.get_book_imbalance();
        assert!(imbalance > 0.0); // More bid liquidity
    }
}
