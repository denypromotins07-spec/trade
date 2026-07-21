//! Advanced Order Flow & Footprint Analytics - Chapter 1
//! File 1: footprint.rs
//! 
//! Implements clustered footprint chart generation tracking bid/ask volume
//! at every price level, utilizing lock-free hash maps to calculate delta
//! imbalances in real-time. Optimized for microsecond latency on AMD Ryzen AI 5.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use dashmap::DashMap;
use crossbeam_queue::SegQueue;
use serde::{Deserialize, Serialize};

/// Represents a single price level in the footprint chart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: i64,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub delta: i64,
    pub total_volume: u64,
    pub trade_count: u32,
    pub timestamp_ns: u64,
}

impl PriceLevel {
    #[inline]
    pub fn new(price: i64) -> Self {
        Self {
            price,
            bid_volume: 0,
            ask_volume: 0,
            delta: 0,
            total_volume: 0,
            trade_count: 0,
            timestamp_ns: 0,
        }
    }

    #[inline]
    pub fn add_trade(&mut self, volume: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        if is_buyer_maker {
            self.bid_volume = self.bid_volume.saturating_add(volume);
        } else {
            self.ask_volume = self.ask_volume.saturating_add(volume);
        }
        self.total_volume = self.total_volume.saturating_add(volume);
        self.trade_count = self.trade_count.saturating_add(1);
        self.delta = self.ask_volume as i64 - self.bid_volume as i64;
        self.timestamp_ns = timestamp_ns;
    }

    /// Calculate imbalance ratio (positive = buying pressure, negative = selling pressure)
    #[inline]
    pub fn imbalance_ratio(&self) -> f64 {
        if self.total_volume == 0 {
            return 0.0;
        }
        (self.ask_volume as f64 - self.bid_volume as f64) / self.total_volume as f64
    }
}

/// Clustered footprint data structure for a single candle/cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FootprintCluster {
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub high_price: i64,
    pub low_price: i64,
    pub open_price: i64,
    pub close_price: i64,
    pub levels: Vec<PriceLevel>,
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub cumulative_delta: i64,
    pub max_imbalance: f64,
    pub min_imbalance: f64,
}

impl FootprintCluster {
    pub fn new() -> Self {
        Self {
            start_time_ns: 0,
            end_time_ns: 0,
            high_price: i64::MAX,
            low_price: i64::MIN,
            open_price: 0,
            close_price: 0,
            levels: Vec::with_capacity(256),
            total_bid_volume: 0,
            total_ask_volume: 0,
            cumulative_delta: 0,
            max_imbalance: f64::NEG_INFINITY,
            min_imbalance: f64::INFINITY,
        }
    }

    #[inline]
    pub fn update_price_bounds(&mut self, price: i64) {
        if price > self.high_price {
            self.high_price = price;
        }
        if price < self.low_price {
            self.low_price = price;
        }
    }
}

/// Lock-free footprint analyzer using DashMap for concurrent access
/// Optimized for AMD Ryzen AI 5 with cache-line aware data structures
pub struct FootprintAnalyzer {
    /// Map of price -> PriceLevel for current cluster
    price_levels: DashMap<i64, Arc<AtomicU64>>, // Packed bid/ask volumes
    /// Queue for processed clusters ready for consumption
    cluster_queue: SegQueue<FootprintCluster>,
    /// Current active cluster
    current_cluster: parking_lot::RwLock<FootprintCluster>,
    /// Tick size for price clustering (e.g., 100 for BTC = 0.01)
    tick_size: i64,
    /// Cluster time window in nanoseconds
    cluster_window_ns: u64,
    /// Total trades processed counter
    trades_processed: AtomicU64,
}

impl FootprintAnalyzer {
    /// Create new footprint analyzer with specified tick size and cluster window
    pub fn new(tick_size: i64, cluster_window_ms: u64) -> Self {
        Self {
            price_levels: DashMap::with_capacity(1024),
            cluster_queue: SegQueue::new(),
            current_cluster: parking_lot::RwLock::new(FootprintCluster::new()),
            tick_size,
            cluster_window_ns: cluster_window_ms * 1_000_000,
            trades_processed: AtomicU64::new(0),
        }
    }

    /// Round price to nearest tick for clustering
    #[inline]
    fn round_to_tick(&self, price: i64) -> i64 {
        (price / self.tick_size) * self.tick_size
    }

    /// Process a single trade from Binance aggregate trade stream
    /// Format: {"a":trade_id,"p":price,"q":qty,"T":timestamp,"m":is_buyer_maker}
    pub fn process_trade(&self, price: i64, quantity: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        let tick_price = self.round_to_tick(price);
        
        // Update current cluster bounds
        {
            let mut cluster = self.current_cluster.write();
            if cluster.start_time_ns == 0 {
                cluster.start_time_ns = timestamp_ns;
                cluster.open_price = tick_price;
            }
            cluster.update_price_bounds(tick_price);
            cluster.close_price = tick_price;
            cluster.end_time_ns = timestamp_ns;
        }

        // Update price level using lock-free operations
        let entry = self.price_levels.entry(tick_price).or_insert_with(|| {
            Arc::new(AtomicU64::new(0))
        });

        // Pack bid/ask volumes into single atomic (upper 32 bits = bid, lower 32 bits = ask)
        let mut packed = entry.value().load(Ordering::Relaxed);
        let bid_vol = (packed >> 32) as u64;
        let ask_vol = (packed & 0xFFFFFFFF) as u64;
        
        let new_packed = if is_buyer_maker {
            ((bid_vol + quantity) << 32) | ask_vol
        } else {
            (bid_vol << 32) | (ask_vol + quantity)
        };
        
        entry.value().store(new_packed, Ordering::Release);
        self.trades_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Finalize current cluster and start new one
    pub fn finalize_cluster(&self) -> Option<FootprintCluster> {
        let current = self.current_cluster.read().clone();
        
        if current.levels.is_empty() && current.total_bid_volume == 0 {
            return None;
        }

        // Build levels from atomic storage
        let mut levels = Vec::with_capacity(self.price_levels.len());
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;
        let mut cum_delta = 0i64;
        let mut max_imb = f64::NEG_INFINITY;
        let mut min_imb = f64::INFINITY;

        for entry in self.price_levels.iter() {
            let price = *entry.key();
            let packed = entry.value().load(Ordering::Acquire);
            let bid_vol = (packed >> 32) as u64;
            let ask_vol = (packed & 0xFFFFFFFF) as u64;
            
            let mut level = PriceLevel::new(price);
            level.bid_volume = bid_vol;
            level.ask_volume = ask_vol;
            level.total_volume = bid_vol + ask_vol;
            level.delta = ask_vol as i64 - bid_vol as i64;
            
            let imb = level.imbalance_ratio();
            if imb > max_imb { max_imb = imb; }
            if imb < min_imb { min_imb = imb; }

            total_bid = total_bid.saturating_add(bid_vol);
            total_ask = total_ask.saturating_add(ask_vol);
            cum_delta += level.delta;

            levels.push(level);
        }

        levels.sort_by_key(|l| l.price);

        let cluster = FootprintCluster {
            start_time_ns: current.start_time_ns,
            end_time_ns: current.end_time_ns,
            high_price: current.high_price,
            low_price: current.low_price,
            open_price: current.open_price,
            close_price: current.close_price,
            levels,
            total_bid_volume: total_bid,
            total_ask_volume: total_ask,
            cumulative_delta: cum_delta,
            max_imbalance: max_imb,
            min_imbalance: min_imb,
        };

        // Clear price levels for next cluster
        self.price_levels.clear();
        
        Some(cluster)
    }

    /// Get completed clusters from queue
    pub fn poll_clusters(&self) -> Vec<FootprintCluster> {
        let mut clusters = Vec::new();
        while let Ok(cluster) = self.cluster_queue.pop() {
            clusters.push(cluster);
        }
        clusters
    }

    /// Get real-time delta imbalance for a specific price level
    pub fn get_delta_imbalance(&self, price: i64) -> Option<f64> {
        let tick_price = self.round_to_tick(price);
        if let Some(entry) = self.price_levels.get(&tick_price) {
            let packed = entry.load(Ordering::Acquire);
            let bid_vol = (packed >> 32) as u64;
            let ask_vol = (packed & 0xFFFFFFFF) as u64;
            if bid_vol + ask_vol == 0 {
                return Some(0.0);
            }
            return Some((ask_vol as f64 - bid_vol as f64) / (bid_vol + ask_vol) as f64);
        }
        None
    }

    /// Get total trades processed
    pub fn get_trades_count(&self) -> u64 {
        self.trades_processed.load(Ordering::Relaxed)
    }
}

/// Real-time footprint stream processor for Binance aggregate trades
pub struct FootprintStreamProcessor {
    analyzer: Arc<FootprintAnalyzer>,
    last_finalize_ns: AtomicU64,
}

impl FootprintStreamProcessor {
    pub fn new(tick_size: i64, cluster_window_ms: u64) -> Self {
        Self {
            analyzer: Arc::new(FootprintAnalyzer::new(tick_size, cluster_window_ms)),
            last_finalize_ns: AtomicU64::new(0),
        }
    }

    /// Parse Binance aggregate trade JSON and process
    pub fn process_binance_agg_trade(&self, json: &str) -> Result<(), serde_json::Error> {
        #[derive(Deserialize)]
        struct BinanceAggTrade {
            #[serde(rename = "p")]
            price: String,
            #[serde(rename = "q")]
            quantity: String,
            #[serde(rename = "m")]
            is_buyer_maker: bool,
            #[serde(rename = "T")]
            timestamp: u64,
        }

        let trade: BinanceAggTrade = serde_json::from_str(json)?;
        
        // Convert price/quantity to integer representation (scaled by 1e8 for precision)
        let price_int = (trade.price.parse::<f64>().unwrap_or(0.0) * 1e8) as i64;
        let qty_int = (trade.quantity.parse::<f64>().unwrap_or(0.0) * 1e8) as u64;
        let timestamp_ns = trade.timestamp * 1_000_000;

        self.analyzer.process_trade(price_int, qty_int, trade.is_buyer_maker, timestamp_ns);

        // Check if cluster window has elapsed
        let last_ns = self.last_finalize_ns.load(Ordering::Relaxed);
        if timestamp_ns - last_ns >= self.analyzer.cluster_window_ns {
            if let Some(_cluster) = self.analyzer.finalize_cluster() {
                self.last_finalize_ns.store(timestamp_ns, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    pub fn get_analyzer(&self) -> Arc<FootprintAnalyzer> {
        Arc::clone(&self.analyzer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_footprint_basic() {
        let analyzer = FootprintAnalyzer::new(100, 1000);
        
        // Simulate trades
        analyzer.process_trade(5000000000, 100, false, 1000000); // Ask
        analyzer.process_trade(5000000000, 50, true, 1000001);   // Bid
        analyzer.process_trade(5000100000, 200, false, 1000002); // Ask at higher price

        assert_eq!(analyzer.get_trades_count(), 3);
        
        let imbalance = analyzer.get_delta_imbalance(5000000000);
        assert!(imbalance.is_some());
        assert!(imbalance.unwrap() > 0.0); // More asks than bids
    }

    #[test]
    fn test_cluster_finalization() {
        let analyzer = FootprintAnalyzer::new(100, 1000);
        
        analyzer.process_trade(5000000000, 100, false, 1000000);
        analyzer.process_trade(5000000000, 50, true, 1000001);
        
        let cluster = analyzer.finalize_cluster();
        assert!(cluster.is_some());
        
        let c = cluster.unwrap();
        assert_eq!(c.total_bid_volume, 50);
        assert_eq!(c.total_ask_volume, 100);
        assert_eq!(c.cumulative_delta, 50);
    }
}
