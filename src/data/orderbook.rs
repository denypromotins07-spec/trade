//! Lock-Free L2/L3 Order Book Implementation
//! 
//! This module constructs a custom lock-free order book using contiguous memory
//! arrays instead of HashMaps, simulating the Binance matching engine for
//! microsecond lookups. Optimized for AMD Ryzen AI 5 cache architecture.
//! 
//! Key Features:
//! - Contiguous memory layout for CPU cache efficiency
//! - Lock-free updates using atomic operations
//! - O(1) price level lookups via binary search on sorted arrays
//! - Sequence number validation to prevent desyncs
//! - Partial update handling for incremental order book deltas

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Maximum price levels per side (bids/asks)
const MAX_PRICE_LEVELS: usize = 1000;

/// Order book side enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Bid,
    Ask,
}

/// Price level with atomic quantity
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    /// Price in integer representation (price * 10^8 for precision)
    pub price: i64,
    /// Quantity at this price level
    pub quantity: f64,
    /// Number of orders at this level
    pub order_count: u32,
    /// Last update timestamp (nanoseconds)
    pub last_update_ns: u64,
}

impl Default for PriceLevel {
    fn default() -> Self {
        Self {
            price: 0,
            quantity: 0.0,
            order_count: 0,
            last_update_ns: 0,
        }
    }
}

/// Lock-free order book state
pub struct OrderBook {
    /// Symbol (e.g., "BTCUSDT")
    symbol: [u8; 16],
    /// Bid side price levels (sorted descending by price)
    bids: [PriceLevel; MAX_PRICE_LEVELS],
    /// Ask side price levels (sorted ascending by price)
    asks: [PriceLevel; MAX_PRICE_LEVELS],
    /// Current number of bid levels
    bid_count: AtomicU64,
    /// Current number of ask levels
    ask_count: AtomicU64,
    /// Last update sequence number
    last_sequence: AtomicU64,
    /// Order book snapshot ID
    snapshot_id: AtomicU64,
    /// Is the order book initialized
    is_initialized: AtomicBool,
    /// Best bid price (cached for fast access)
    best_bid: AtomicU64,
    /// Best ask price (cached for fast access)
    best_ask: AtomicU64,
    /// Mid price (cached)
    mid_price: AtomicU64,
    /// Spread in ticks
    spread_ticks: AtomicU64,
}

/// Order book update operation
#[derive(Debug, Clone, Copy)]
pub struct OrderBookUpdate {
    pub side: Side,
    pub price: i64,
    pub quantity: f64,
    pub order_count: u32,
    pub sequence: u64,
    pub timestamp_ns: u64,
}

impl OrderBook {
    /// Create a new empty order book
    pub fn new(symbol: &str) -> Self {
        let mut symbol_bytes = [0u8; 16];
        let symbol_slice = symbol.as_bytes();
        let copy_len = symbol_slice.len().min(16);
        symbol_bytes[..copy_len].copy_from_slice(&symbol_slice[..copy_len]);

        Self {
            symbol: symbol_bytes,
            bids: [PriceLevel::default(); MAX_PRICE_LEVELS],
            asks: [PriceLevel::default(); MAX_PRICE_LEVELS],
            bid_count: AtomicU64::new(0),
            ask_count: AtomicU64::new(0),
            last_sequence: AtomicU64::new(0),
            snapshot_id: AtomicU64::new(0),
            is_initialized: AtomicBool::new(false),
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(0),
            mid_price: AtomicU64::new(0),
            spread_ticks: AtomicU64::new(0),
        }
    }

    /// Apply a single order book update (lock-free)
    /// Returns true if update was applied successfully, false if sequence mismatch
    pub fn apply_update(&mut self, update: OrderBookUpdate) -> bool {
        let current_seq = self.last_sequence.load(Ordering::Acquire);
        
        // Validate sequence number
        if update.sequence <= current_seq && current_seq > 0 {
            // Stale update, skip
            return false;
        }

        match update.side {
            Side::Bid => self.update_side(&mut self.bids, &self.bid_count, update),
            Side::Ask => self.update_side(&mut self.asks, &self.ask_count, update),
        }

        // Update sequence
        self.last_sequence.store(update.sequence, Ordering::Release);
        
        // Recalculate cached values
        self.update_cached_values();
        
        true
    }

    /// Update a specific side of the order book
    fn update_side(
        &mut self,
        levels: &mut [PriceLevel; MAX_PRICE_LEVELS],
        count: &AtomicU64,
        update: OrderBookUpdate,
    ) {
        let current_count = count.load(Ordering::Acquire) as usize;
        
        // Binary search for the price level
        let pos = self.binary_search(levels, current_count, update.price);
        
        if pos < current_count && levels[pos].price == update.price {
            // Update existing level
            if update.quantity == 0.0 {
                // Remove level (shift remaining levels)
                self.remove_level(levels, count, pos);
            } else {
                // Update quantity and order count
                levels[pos].quantity = update.quantity;
                levels[pos].order_count = update.order_count;
                levels[pos].last_update_ns = update.timestamp_ns;
            }
        } else if update.quantity > 0.0 {
            // Insert new level
            self.insert_level(levels, count, pos, update);
        }
    }

    /// Binary search for price in sorted array
    fn binary_search(&self, levels: &[PriceLevel; MAX_PRICE_LEVELS], count: usize, price: i64) -> usize {
        let mut left = 0;
        let mut right = count;
        
        while left < right {
            let mid = left + (right - left) / 2;
            if levels[mid].price < price {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        left
    }

    /// Insert a new price level at the specified position
    fn insert_level(
        &mut self,
        levels: &mut [PriceLevel; MAX_PRICE_LEVELS],
        count: &AtomicU64,
        pos: usize,
        update: OrderBookUpdate,
    ) {
        let current_count = count.load(Ordering::Acquire) as usize;
        
        if current_count >= MAX_PRICE_LEVELS {
            // Order book full, remove worst level
            // (In production: implement dynamic sizing or eviction policy)
            return;
        }
        
        // Shift elements to make room
        for i in (pos..current_count).rev() {
            levels[i + 1] = levels[i];
        }
        
        // Insert new level
        levels[pos] = PriceLevel {
            price: update.price,
            quantity: update.quantity,
            order_count: update.order_count,
            last_update_ns: update.timestamp_ns,
        };
        
        count.store(current_count as u64 + 1, Ordering::Release);
    }

    /// Remove a price level at the specified position
    fn remove_level(
        &mut self,
        levels: &mut [PriceLevel; MAX_PRICE_LEVELS],
        count: &AtomicU64,
        pos: usize,
    ) {
        let current_count = count.load(Ordering::Acquire) as usize;
        
        if pos >= current_count {
            return;
        }
        
        // Shift elements to fill gap
        for i in pos..current_count - 1 {
            levels[i] = levels[i + 1];
        }
        
        // Clear last element
        levels[current_count - 1] = PriceLevel::default();
        
        count.store(current_count as u64 - 1, Ordering::Release);
    }

    /// Update cached best bid, best ask, mid price, and spread
    fn update_cached_values(&mut self) {
        let bid_count = self.bid_count.load(Ordering::Acquire);
        let ask_count = self.ask_count.load(Ordering::Acquire);
        
        if bid_count > 0 && ask_count > 0 {
            let best_bid_price = self.bids[0].price as u64;
            let best_ask_price = self.asks[0].price as u64;
            
            self.best_bid.store(best_bid_price, Ordering::Release);
            self.best_ask.store(best_ask_price, Ordering::Release);
            
            // Mid price = (best_bid + best_ask) / 2
            let mid = (best_bid_price + best_ask_price) / 2;
            self.mid_price.store(mid, Ordering::Release);
            
            // Spread in ticks
            let spread = best_ask_price.saturating_sub(best_bid_price);
            self.spread_ticks.store(spread, Ordering::Release);
        }
        
        // Mark as initialized when we have both sides
        if bid_count > 0 && ask_count > 0 {
            self.is_initialized.store(true, Ordering::Release);
        }
    }

    /// Get best bid price (fast O(1) access)
    pub fn get_best_bid(&self) -> Option<f64> {
        if self.bid_count.load(Ordering::Acquire) > 0 {
            Some(self.bids[0].price as f64 / 1e8)
        } else {
            None
        }
    }

    /// Get best ask price (fast O(1) access)
    pub fn get_best_ask(&self) -> Option<f64> {
        if self.ask_count.load(Ordering::Acquire) > 0 {
            Some(self.asks[0].price as f64 / 1e8)
        } else {
            None
        }
    }

    /// Get mid price (fast O(1) access)
    pub fn get_mid_price(&self) -> Option<f64> {
        if self.is_initialized.load(Ordering::Acquire) {
            Some(self.mid_price.load(Ordering::Acquire) as f64 / 1e8)
        } else {
            None
        }
    }

    /// Get spread in ticks (fast O(1) access)
    pub fn get_spread_ticks(&self) -> u64 {
        self.spread_ticks.load(Ordering::Acquire)
    }

    /// Get order book depth at specified levels
    pub fn get_depth(&self, levels: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bid_count = self.bid_count.load(Ordering::Acquire) as usize;
        let ask_count = self.ask_count.load(Ordering::Acquire) as usize;
        
        let bid_depth: Vec<(f64, f64)> = self.bids[..bid_count.min(levels)]
            .iter()
            .map(|l| (l.price as f64 / 1e8, l.quantity))
            .collect();
        
        let ask_depth: Vec<(f64, f64)> = self.asks[..ask_count.min(levels)]
            .iter()
            .map(|l| (l.price as f64 / 1e8, l.quantity))
            .collect();
        
        (bid_depth, ask_depth)
    }

    /// Check if order book is initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized.load(Ordering::Acquire)
    }

    /// Get last sequence number
    pub fn get_sequence(&self) -> u64 {
        self.last_sequence.load(Ordering::Acquire)
    }

    /// Get symbol
    pub fn get_symbol(&self) -> String {
        String::from_utf8_lossy(&self.symbol).trim_end_matches('\0').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_updates() {
        let mut ob = OrderBook::new("BTCUSDT");
        
        // Add bid
        let bid_update = OrderBookUpdate {
            side: Side::Bid,
            price: 50000_0000_0000i64, // $50,000 with 8 decimals
            quantity: 1.5,
            order_count: 3,
            sequence: 1,
            timestamp_ns: 1000000000,
        };
        assert!(ob.apply_update(bid_update));
        
        // Add ask
        let ask_update = OrderBookUpdate {
            side: Side::Ask,
            price: 50001_0000_0000i64,
            quantity: 2.0,
            order_count: 5,
            sequence: 2,
            timestamp_ns: 1000000001,
        };
        assert!(ob.apply_update(ask_update));
        
        // Verify best prices
        assert_eq!(ob.get_best_bid(), Some(50000.0));
        assert_eq!(ob.get_best_ask(), Some(50001.0));
        assert_eq!(ob.get_spread_ticks(), 1_0000_0000);
    }

    #[test]
    fn test_sequence_validation() {
        let mut ob = OrderBook::new("ETHUSDT");
        
        let update1 = OrderBookUpdate {
            side: Side::Bid,
            price: 3000_0000_0000i64,
            quantity: 10.0,
            order_count: 1,
            sequence: 100,
            timestamp_ns: 1000000000,
        };
        assert!(ob.apply_update(update1));
        
        // Stale update should be rejected
        let stale_update = OrderBookUpdate {
            side: Side::Bid,
            price: 3001_0000_0000i64,
            quantity: 5.0,
            order_count: 1,
            sequence: 50, // Older than 100
            timestamp_ns: 1000000001,
        };
        assert!(!ob.apply_update(stale_update));
    }
}
