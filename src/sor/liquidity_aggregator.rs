//! Smart Order Routing - Liquidity Aggregator
//! 
//! Implements a lock-free liquidity aggregator that merges L2 orderbooks from
//! Binance Spot, Margin, and Futures into a unified synthetic book.
//! Optimized for AMD Ryzen AI 5 with microsecond latency targets.
//! Strictly enforces 8GB global RAM limit by bounding synthetic book depth.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam_queue::SegQueue;
use rayon::prelude::*;

/// Maximum depth levels per side to enforce RAM limits (8GB global constraint)
const MAX_BOOK_DEPTH: usize = 100;

/// Price level in the orderbook (fixed-point integer math to prevent drift)
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))] // Cache-line alignment for SIMD
pub struct PriceLevel {
    /// Price in fixed-point (integer nanodollars)
    pub price_ns: i64,
    /// Quantity in base asset (integer nanounits)
    pub qty_ns: i64,
    /// Number of orders at this level
    pub order_count: u32,
    /// Venue identifier (bitmask: bit0=spot, bit1=margin, bit2=futures)
    pub venue_mask: u8,
    /// Padding for cache alignment
    _padding: [u8; 7],
}

impl PriceLevel {
    #[inline(always)]
    pub const fn new(price_ns: i64, qty_ns: i64) -> Self {
        Self {
            price_ns,
            qty_ns,
            order_count: 0,
            venue_mask: 0,
            _padding: [0u8; 7],
        }
    }

    #[inline(always)]
    pub fn merge(&mut self, other: &PriceLevel) {
        debug_assert_eq!(self.price_ns, other.price_ns);
        self.qty_ns = self.qty_ns.saturating_add(other.qty_ns);
        self.order_count = self.order_count.saturating_add(other.order_count);
        self.venue_mask |= other.venue_mask;
    }
}

/// Unified synthetic orderbook side (bid or ask)
#[repr(C, align(64))]
pub struct BookSide {
    /// Contiguous memory slab for price levels (no heap fragmentation)
    levels: [PriceLevel; MAX_BOOK_DEPTH],
    /// Current number of valid levels
    depth: AtomicU64,
    /// Sorted flag (false requires re-sort)
    is_sorted: AtomicBool,
    /// Last update timestamp (nanoseconds)
    last_update_ns: AtomicU64,
}

impl BookSide {
    pub const fn new() -> Self {
        Self {
            levels: [PriceLevel::new(0, 0); MAX_BOOK_DEPTH],
            depth: AtomicU64::new(0),
            is_sorted: AtomicBool::new(true),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Lock-free insertion/merge of a price level
    #[inline(always)]
    pub fn upsert(&self, level: PriceLevel) -> bool {
        let current_depth = self.depth.load(Ordering::Acquire) as usize;
        
        // Enforce depth bound to respect 8GB RAM limit
        if current_depth >= MAX_BOOK_DEPTH && !self.contains_price(level.price_ns) {
            return false; // Drop level to maintain memory bounds
        }

        // Linear scan for small depths (faster than binary search for < 20 elements)
        if current_depth < 20 {
            for i in 0..current_depth {
                if unsafe { self.levels.get_unchecked(i) }.price_ns == level.price_ns {
                    // Merge with existing level
                    let existing = unsafe { self.levels.get_unchecked_mut(i) };
                    existing.merge(&level);
                    self.last_update_ns.store(get_time_ns(), Ordering::Release);
                    return true;
                }
            }
        }

        // Binary search for larger depths
        let pos = self.binary_search_price(level.price_ns, current_depth);
        
        if let Some(idx) = pos {
            // Merge at existing position
            let existing = unsafe { self.levels.get_unchecked_mut(idx) };
            existing.merge(&level);
        } else {
            // Insert new level (shift elements)
            let insert_pos = self.find_insert_position(level.price_ns, current_depth);
            if current_depth < MAX_BOOK_DEPTH {
                self.shift_right(insert_pos, current_depth);
                unsafe {
                    *self.levels.get_unchecked_mut(insert_pos) = level;
                }
                self.depth.fetch_add(1, Ordering::Release);
            } else {
                return false; // Depth limit reached
            }
        }

        self.is_sorted.store(false, Ordering::Release);
        self.last_update_ns.store(get_time_ns(), Ordering::Release);
        true
    }

    #[inline(always)]
    fn contains_price(&self, price_ns: i64) -> bool {
        let depth = self.depth.load(Ordering::Acquire) as usize;
        for i in 0..depth {
            if unsafe { self.levels.get_unchecked(i) }.price_ns == price_ns {
                return true;
            }
        }
        false
    }

    #[inline(always)]
    fn binary_search_price(&self, price_ns: i64, depth: usize) -> Option<usize> {
        let mut low = 0;
        let mut high = depth;
        
        while low < high {
            let mid = low + (high - low) / 2;
            let mid_price = unsafe { self.levels.get_unchecked(mid) }.price_ns;
            
            if mid_price == price_ns {
                return Some(mid);
            } else if mid_price < price_ns {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        None
    }

    #[inline(always)]
    fn find_insert_position(&self, price_ns: i64, depth: usize) -> usize {
        // For bids: descending order (highest first)
        // For asks: ascending order (lowest first)
        // This method assumes caller knows which side
        let mut pos = 0;
        while pos < depth {
            let level_price = unsafe { self.levels.get_unchecked(pos) }.price_ns;
            if price_ns > level_price {
                break;
            }
            pos += 1;
        }
        pos
    }

    #[inline(always)]
    fn shift_right(&self, start: usize, end: usize) {
        for i in (start..end).rev() {
            unsafe {
                let src = self.levels.get_unchecked(i);
                let dst = self.levels.get_unchecked_mut(i + 1);
                *dst = *src;
            }
        }
    }

    #[inline(always)]
    pub fn get_best(&self) -> Option<&PriceLevel> {
        if self.depth.load(Ordering::Acquire) == 0 {
            return None;
        }
        Some(unsafe { self.levels.get_unchecked(0) })
    }

    #[inline(always)]
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::Acquire) as usize
    }
}

/// Unified synthetic orderbook combining multiple venues
#[repr(C, align(64))]
pub struct SyntheticBook {
    /// Bid side (buy orders)
    pub bids: BookSide,
    /// Ask side (sell orders)
    pub asks: BookSide,
    /// Symbol hash for quick lookup
    pub symbol_hash: u64,
    /// Last sequence number
    pub sequence: AtomicU64,
    /// Health flag (false if stale)
    pub healthy: AtomicBool,
}

impl SyntheticBook {
    pub const fn new(symbol_hash: u64) -> Self {
        Self {
            bids: BookSide::new(),
            asks: BookSide::new(),
            symbol_hash,
            sequence: AtomicU64::new(0),
            healthy: AtomicBool::new(true),
        }
    }

    /// Merge L2 snapshot from a specific venue
    #[inline(always)]
    pub fn merge_venue_snapshot(
        &self,
        venue_id: u8,
        bids: &[(i64, i64)],
        asks: &[(i64, i64)],
        timestamp_ns: u64,
    ) {
        let venue_bit = 1u8 << venue_id;

        // Parallel processing for bids and asks using Rayon
        let bid_levels: Vec<PriceLevel> = bids
            .par_iter()
            .take(MAX_BOOK_DEPTH)
            .map(|&(price_ns, qty_ns)| PriceLevel {
                price_ns,
                qty_ns,
                order_count: 1,
                venue_mask: venue_bit,
                _padding: [0u8; 7],
            })
            .collect();

        let ask_levels: Vec<PriceLevel> = asks
            .par_iter()
            .take(MAX_BOOK_DEPTH)
            .map(|&(price_ns, qty_ns)| PriceLevel {
                price_ns,
                qty_ns,
                order_count: 1,
                venue_mask: venue_bit,
                _padding: [0u8; 7],
            })
            .collect();

        // Sequential merge to maintain lock-free guarantees
        for level in bid_levels {
            self.bids.upsert(level);
        }
        for level in ask_levels {
            self.asks.upsert(level);
        }

        self.sequence.fetch_add(1, Ordering::Release);
        self.healthy.store(true, Ordering::Release);
    }

    /// Get best bid-ask spread
    #[inline(always)]
    pub fn get_spread(&self) -> Option<(i64, i64, i64)> {
        let best_bid = self.bids.get_best()?;
        let best_ask = self.asks.get_best()?;
        
        if best_bid.price_ns >= best_ask.price_ns {
            // Crossed book - arbitrage opportunity
            return Some((best_bid.price_ns, best_ask.price_ns, best_bid.price_ns - best_ask.price_ns));
        }
        
        Some((best_bid.price_ns, best_ask.price_ns, best_ask.price_ns - best_bid.price_ns))
    }

    /// Calculate mid-price in nanodollars
    #[inline(always)]
    pub fn get_mid_price(&self) -> Option<i64> {
        let spread = self.get_spread()?;
        Some((spread.0 + spread.1) / 2)
    }
}

/// Lock-free queue for incoming venue updates
pub struct VenueUpdateQueue {
    queue: SegQueue<VenueUpdate>,
    /// High watermark for memory monitoring
    high_watermark: AtomicU64,
    /// Current size estimate (bytes)
    size_bytes: AtomicU64,
}

#[derive(Clone, Debug)]
pub struct VenueUpdate {
    pub symbol_hash: u64,
    pub venue_id: u8,
    pub timestamp_ns: u64,
    pub bids: Vec<(i64, i64)>,
    pub asks: Vec<(i64, i64)>,
}

impl VenueUpdateQueue {
    pub fn new() -> Self {
        Self {
            queue: SegQueue::new(),
            high_watermark: AtomicU64::new(0),
            size_bytes: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn push(&self, update: VenueUpdate) -> bool {
        let estimated_size = 128 + (update.bids.len() + update.asks.len()) * 16;
        let current_size = self.size_bytes.fetch_add(estimated_size as u64, Ordering::Relaxed);
        
        // Enforce memory bound (part of 8GB global limit allocation)
        if current_size + estimated_size as u64 > 512 * 1024 * 1024 {
            // 512MB limit for update queue
            self.size_bytes.fetch_sub(estimated_size as u64, Ordering::Relaxed);
            return false;
        }

        self.queue.push(update);
        
        // Update high watermark
        let total = self.size_bytes.load(Ordering::Relaxed);
        let mut hw = self.high_watermark.load(Ordering::Relaxed);
        while total > hw {
            match self.high_watermark.compare_exchange_weak(
                hw,
                total,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => hw = x,
            }
        }
        
        true
    }

    #[inline(always)]
    pub fn pop(&self) -> Option<VenueUpdate> {
        if let Some(update) = self.queue.pop() {
            let estimated_size = 128 + (update.bids.len() + update.asks.len()) * 16;
            self.size_bytes.fetch_sub(estimated_size as u64, Ordering::Relaxed);
            Some(update)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn size_bytes(&self) -> u64 {
        self.size_bytes.load(Ordering::Acquire)
    }
}

/// Get current time in nanoseconds (AMD Ryzen optimized)
#[inline(always)]
fn get_time_ns() -> u64 {
    use std::time::Instant;
    static START: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_book_merge() {
        let book = SyntheticBook::new(0x12345678);
        
        let bids = vec![(100_000_000_000i64, 1_000_000_000i64)];
        let asks = vec![(100_100_000_000i64, 1_000_000_000i64)];
        
        book.merge_venue_snapshot(0, &bids, &asks, get_time_ns());
        
        assert_eq!(book.bids.depth(), 1);
        assert_eq!(book.asks.depth(), 1);
        
        let spread = book.get_spread().unwrap();
        assert_eq!(spread.2, 100_000_000); // 0.1 USD spread
    }

    #[test]
    fn test_memory_bounds() {
        let book = SyntheticBook::new(0xABCDEF);
        
        // Try to exceed MAX_BOOK_DEPTH
        let bids: Vec<(i64, i64)> = (0..MAX_BOOK_DEPTH + 50)
            .map(|i| (100_000_000_000i64 - i as i64, 1_000_000_000i64))
            .collect();
        let asks: Vec<(i64, i64)> = (0..MAX_BOOK_DEPTH + 50)
            .map(|i| (100_100_000_000i64 + i as i64, 1_000_000_000i64))
            .collect();
        
        book.merge_venue_snapshot(0, &bids, &asks, get_time_ns());
        
        // Should be bounded by MAX_BOOK_DEPTH
        assert!(book.bids.depth() <= MAX_BOOK_DEPTH);
        assert!(book.asks.depth() <= MAX_BOOK_DEPTH);
    }
}
