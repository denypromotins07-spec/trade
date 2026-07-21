//! Deep Market Microstructure: Queue Position Tracking
//! 
//! Implements a custom lock-free priority queue to track exact queue positions
//! for limit orders, estimating fill probabilities based on historical cancellation rates.
//! Optimized for microsecond latency on AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use crossbeam_queue::SegQueue;
use crate::memory::allocator::ArenaAllocator;

/// Represents a single order in the queue with metadata for position tracking
#[derive(Debug, Clone)]
pub struct QueueOrder {
    pub order_id: u64,
    pub price_level: i64, // Stored as integer ticks to avoid float drift
    pub quantity: u64,
    pub timestamp_ns: u64,
    pub side: OrderSide,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Bid,
    Ask,
}

/// Lock-free priority queue for tracking order positions at each price level
/// Uses atomic operations for thread-safe access without mutex contention
pub struct LockFreeOrderQueue {
    /// Queue of orders sorted by time priority (FIFO within price levels)
    orders: SegQueue<QueueOrder>,
    /// Total quantity in queue (atomic for lock-free updates)
    total_quantity: AtomicU64,
    /// Count of orders in queue
    order_count: AtomicUsize,
    /// Historical cancellation rate for fill probability estimation (basis points)
    hist_cancel_rate_bps: AtomicU64,
    /// Arena allocator reference for zero-allocation operations
    arena: Arc<ArenaAllocator>,
}

impl LockFreeOrderQueue {
    /// Create a new lock-free order queue with arena allocator
    pub fn new(arena: Arc<ArenaAllocator>) -> Self {
        Self {
            orders: SegQueue::new(),
            total_quantity: AtomicU64::new(0),
            order_count: AtomicUsize::new(0),
            hist_cancel_rate_bps: AtomicU64::new(500), // Default 5% cancellation rate
            arena,
        }
    }

    /// Push an order to the queue (lock-free)
    #[inline]
    pub fn push(&self, order: QueueOrder) {
        self.orders.push(order);
        self.total_quantity.fetch_add(order.quantity, Ordering::Relaxed);
        self.order_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop an order from the queue (lock-free)
    #[inline]
    pub fn pop(&self) -> Option<QueueOrder> {
        if let Some(order) = self.orders.pop() {
            self.total_quantity.fetch_sub(order.quantity, Ordering::Relaxed);
            self.order_count.fetch_sub(1, Ordering::Relaxed);
            Some(order)
        } else {
            None
        }
    }

    /// Get current queue size (lock-free read)
    #[inline]
    pub fn size(&self) -> usize {
        self.order_count.load(Ordering::Acquire)
    }

    /// Get total quantity in queue (lock-free read)
    #[inline]
    pub fn total_qty(&self) -> u64 {
        self.total_quantity.load(Ordering::Acquire)
    }

    /// Update historical cancellation rate (basis points)
    #[inline]
    pub fn update_cancel_rate(&self, rate_bps: u64) {
        self.hist_cancel_rate_bps.store(rate_bps, Ordering::Release);
    }

    /// Estimate fill probability for a new order at given position
    /// 
    /// Formula: P(fill) = (1 - cancel_rate) * (queue_ahead / total_queue)
    /// Returns probability as basis points (0-10000)
    #[inline]
    pub fn estimate_fill_probability(&self, position_ahead: usize) -> u16 {
        let total = self.size();
        if total == 0 {
            return 10000; // 100% if queue is empty
        }

        let cancel_rate = self.hist_cancel_rate_bps.load(Ordering::Acquire);
        let stay_rate = 10000 - cancel_rate.min(10000);

        // Position ahead as ratio of total queue
        let position_ratio = (position_ahead as u64 * 10000) / total as u64;
        
        // Probability decreases linearly with position in queue
        let base_prob = 10000 - position_ratio;
        
        // Adjust for cancellation rate
        ((base_prob as u64 * stay_rate) / 10000) as u16
    }

    /// Calculate exact queue position for a specific order ID
    /// Returns (position, total_ahead) or None if not found
    pub fn find_position(&self, target_order_id: u64) -> Option<(usize, usize)> {
        let mut position = 0;
        let mut found = false;
        
        // Note: This requires iterating the queue, which is O(n)
        // In production, this would be backed by a skip list for O(log n)
        for order in self.orders.iter() {
            if order.order_id == target_order_id {
                found = true;
                break;
            }
            position += 1;
        }

        if found {
            Some((position, self.size() - position))
        } else {
            None
        }
    }

    /// Clear all orders from queue (lock-free reset)
    #[inline]
    pub fn clear(&self) {
        while self.pop().is_some() {}
    }
}

/// Queue position tracker for multiple price levels
pub struct QueuePositionTracker {
    /// Bid side queues indexed by price level (integer ticks)
    bid_queues: dashmap::DashMap<i64, Arc<LockFreeOrderQueue>>,
    /// Ask side queues indexed by price level (integer ticks)
    ask_queues: dashmap::DashMap<i64, Arc<LockFreeOrderQueue>>,
    /// Arena allocator for zero-allocation operations
    arena: Arc<ArenaAllocator>,
    /// Global cancellation rate tracker (basis points)
    global_cancel_rate_bps: AtomicU64,
}

impl QueuePositionTracker {
    /// Create a new queue position tracker
    pub fn new(arena: Arc<ArenaAllocator>) -> Self {
        Self {
            bid_queues: dashmap::DashMap::new(),
            ask_queues: dashmap::DashMap::new(),
            arena,
            global_cancel_rate_bps: AtomicU64::new(500),
        }
    }

    /// Get or create queue for a specific price level and side
    #[inline]
    fn get_or_create_queue(
        &self,
        price_level: i64,
        side: OrderSide,
    ) -> Arc<LockFreeOrderQueue> {
        match side {
            OrderSide::Bid => {
                self.bid_queues
                    .entry(price_level)
                    .or_insert_with(|| Arc::new(LockFreeOrderQueue::new(Arc::clone(&self.arena))))
                    .value()
                    .clone()
            }
            OrderSide::Ask => {
                self.ask_queues
                    .entry(price_level)
                    .or_insert_with(|| Arc::new(LockFreeOrderQueue::new(Arc::clone(&self.arena))))
                    .value()
                    .clone()
            }
        }
    }

    /// Add an order to the appropriate queue
    #[inline]
    pub fn add_order(&self, order: QueueOrder) {
        let queue = self.get_or_create_queue(order.price_level, order.side);
        queue.push(order);
    }

    /// Remove an order from the queue (e.g., on fill or cancel)
    #[inline]
    pub fn remove_order(&self, order_id: u64, price_level: i64, side: OrderSide) -> Option<QueueOrder> {
        let queue = self.get_or_create_queue(price_level, side);
        // In production, this would use a hash index for O(1) removal
        // For now, we iterate and rebuild (acceptable for small queues per level)
        let mut removed = None;
        let mut remaining = Vec::new();

        while let Some(order) = queue.pop() {
            if order.order_id == order_id {
                removed = Some(order);
            } else {
                remaining.push(order);
            }
        }

        // Re-add remaining orders
        for order in remaining {
            queue.push(order);
        }

        removed
    }

    /// Get queue position for a specific order
    #[inline]
    pub fn get_position(&self, order_id: u64, price_level: i64, side: OrderSide) -> Option<(usize, usize)> {
        let queue = self.get_or_create_queue(price_level, side);
        queue.find_position(order_id)
    }

    /// Estimate fill probability for an order at given position
    #[inline]
    pub fn estimate_fill_prob(
        &self,
        order_id: u64,
        price_level: i64,
        side: OrderSide,
    ) -> Option<u16> {
        let queue = self.get_or_create_queue(price_level, side);
        if let Some((position, _)) = queue.find_position(order_id) {
            Some(queue.estimate_fill_probability(position))
        } else {
            None
        }
    }

    /// Update global cancellation rate from historical data
    #[inline]
    pub fn update_global_cancel_rate(&self, rate_bps: u64) {
        self.global_cancel_rate_bps.store(rate_bps.min(10000), Ordering::Release);
        
        // Propagate to all queues
        for entry in self.bid_queues.iter() {
            entry.value().update_cancel_rate(rate_bps.min(10000));
        }
        for entry in self.ask_queues.iter() {
            entry.value().update_cancel_rate(rate_bps.min(10000));
        }
    }

    /// Get total depth at a price level
    #[inline]
    pub fn get_depth_at_level(&self, price_level: i64, side: OrderSide) -> u64 {
        let queue = match side {
            OrderSide::Bid => self.bid_queues.get(&price_level),
            OrderSide::Ask => self.ask_queues.get(&price_level),
        };
        
        queue.map(|q| q.total_qty()).unwrap_or(0)
    }

    /// Clear all queues (for /KILL reset)
    pub fn clear_all(&self) {
        self.bid_queues.clear();
        self.ask_queues.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::ArenaAllocator;

    #[test]
    fn test_lock_free_queue_basic() {
        let arena = Arc::new(ArenaAllocator::new(1024 * 1024));
        let queue = LockFreeOrderQueue::new(arena);

        let order = QueueOrder {
            order_id: 1,
            price_level: 50000,
            quantity: 100,
            timestamp_ns: 1000,
            side: OrderSide::Bid,
        };

        queue.push(order);
        assert_eq!(queue.size(), 1);
        assert_eq!(queue.total_qty(), 100);

        let popped = queue.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().order_id, 1);
        assert_eq!(queue.size(), 0);
    }

    #[test]
    fn test_fill_probability_estimation() {
        let arena = Arc::new(ArenaAllocator::new(1024 * 1024));
        let queue = LockFreeOrderQueue::new(arena);

        // Add 10 orders
        for i in 0..10 {
            queue.push(QueueOrder {
                order_id: i,
                price_level: 50000,
                quantity: 10,
                timestamp_ns: i * 100,
                side: OrderSide::Bid,
            });
        }

        // First order should have high probability
        let prob_first = queue.estimate_fill_probability(0);
        assert!(prob_first > 8000); // >80%

        // Last order should have low probability
        let prob_last = queue.estimate_fill_probability(9);
        assert!(prob_last < 2000); // <20%
    }
}
