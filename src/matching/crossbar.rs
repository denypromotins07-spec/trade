// =============================================================================
// Nautilus/Ray Crypto Trading Bot - Stage 50
// File: src/matching/crossbar.rs
// Chapter 2: FPGA-Style Bitwise Matching Engine (Rust)
// 
// Purpose: Code a hardware-style crossbar switch router that directs
//          incoming market orders to the exact bitwise price level queue
//          without any pointer chasing or heap allocations.
//
// Optimization Targets:
//   - Microsecond latency via direct routing
//   - 8GB RAM limit enforcement via fixed-size arrays
//   - AMD Ryzen AI 5 cache optimization
//   - Zero heap allocation in hot path
//
// Constraints:
//   - No LLMs, strict typing, production-grade code
//   - FPGA crossbar-inspired design
// =============================================================================

use std::mem;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Number of input ports (order sources).
const NUM_INPUTS: usize = 8;

/// Number of output ports (price level queues).
const NUM_OUTPUTS: usize = 64;

/// Maximum pending orders per input port.
const MAX_PENDING_PER_INPUT: usize = 128;

/// Crossbar switch state for routing orders.
#[repr(C, align(64))]
struct CrossbarState {
    /// Routing table: input_idx -> output_idx mapping.
    routing_table: [u8; NUM_INPUTS],
    /// Pending order count per input.
    pending_counts: [AtomicUsize; NUM_INPUTS],
    /// Total routed orders.
    total_routed: AtomicU64,
    /// Padding for cache line alignment.
    _padding: [u8; 32],
}

// Verify size: 8 + (8*8) + 8 + 32 = 112 bytes, need adjustment
const _: () = assert!(mem::size_of::<CrossbarState>() <= 128, "CrossbarState must fit in 2 cache lines");

/// Hardware-style crossbar switch for order routing.
pub struct CrossbarSwitch {
    /// Current switch state.
    state: Box<CrossbarState>,
    /// Output queue pointers (indices into shared memory pool).
    output_queues: Box<[AtomicUsize; NUM_OUTPUTS]>,
    /// Shared memory pool for order storage (pre-allocated).
    /// In production, this would point to actual order data.
    order_pool: Box<[OrderSlot; NUM_OUTPUTS * MAX_PENDING_PER_INPUT]>,
}

/// Pre-allocated order slot (no heap allocation).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OrderSlot {
    pub price: i64,
    pub quantity: i64,
    pub is_buy: bool,
    pub occupied: bool,
}

unsafe impl Send for CrossbarSwitch {}
unsafe impl Sync for CrossbarSwitch {}

impl CrossbarSwitch {
    /// Create a new crossbar switch.
    pub fn new() -> Self {
        let state = Box::new(CrossbarState {
            routing_table: [0u8; NUM_INPUTS],
            pending_counts: Default::default(),
            total_routed: AtomicU64::new(0),
            _padding: [0u8; 32],
        });
        
        let output_queues = Box::new(std::array::from_fn(|_| AtomicUsize::new(0)));
        
        // Pre-allocate order pool.
        let order_pool = Box::new([OrderSlot {
            price: 0,
            quantity: 0,
            is_buy: false,
            occupied: false,
        }; NUM_OUTPUTS * MAX_PENDING_PER_INPUT]);
        
        Self {
            state,
            output_queues,
            order_pool,
        }
    }
    
    /// Configure routing for an input port.
    /// 
    /// # Arguments
    /// * `input_idx` - Input port index (0-7)
    /// * `output_idx` - Output port index (price level, 0-63)
    /// 
    /// # Returns
    /// true if configuration successful, false if invalid indices
    pub fn configure_route(&self, input_idx: usize, output_idx: usize) -> bool {
        if input_idx >= NUM_INPUTS || output_idx >= NUM_OUTPUTS {
            return false;
        }
        
        unsafe {
            let state_ptr = &*self.state as *const CrossbarState as *mut CrossbarState;
            (*state_ptr).routing_table[input_idx] = output_idx as u8;
        }
        
        true
    }
    
    /// Route an order through the crossbar switch.
    /// 
    /// # Arguments
    /// * `input_idx` - Input port where order arrived
    /// * `price` - Order price
    /// * `quantity` - Order quantity
    /// * `is_buy` - true for buy order
    /// 
    /// # Returns
    /// true if order was routed successfully, false if queue full
    pub fn route_order(&self, input_idx: usize, price: i64, quantity: i64, is_buy: bool) -> bool {
        if input_idx >= NUM_INPUTS {
            return false;
        }
        
        // Look up output port from routing table (O(1), no pointer chasing).
        let output_idx = self.state.routing_table[input_idx] as usize;
        
        // Check queue capacity.
        let pending = self.state.pending_counts[input_idx].load(Ordering::Relaxed);
        if pending >= MAX_PENDING_PER_INPUT {
            return false; // Queue full - backpressure
        }
        
        // Find free slot in output queue (using pre-allocated pool).
        let queue_base = output_idx * MAX_PENDING_PER_INPUT;
        let queue_head = self.output_queues[output_idx].load(Ordering::Relaxed);
        
        if queue_head >= MAX_PENDING_PER_INPUT {
            return false;
        }
        
        // Write order to pre-allocated slot (zero heap allocation).
        let slot_idx = queue_base + queue_head;
        let slot = &mut self.order_pool[slot_idx];
        slot.price = price;
        slot.quantity = quantity;
        slot.is_buy = is_buy;
        slot.occupied = true;
        
        // Update counters.
        self.state.pending_counts[input_idx].fetch_add(1, Ordering::Relaxed);
        self.output_queues[output_idx].fetch_add(1, Ordering::Relaxed);
        self.state.total_routed.fetch_add(1, Ordering::Relaxed);
        
        true
    }
    
    /// Drain orders from an output port for processing.
    /// 
    /// # Arguments
    /// * `output_idx` - Output port to drain
    /// * `callback` - Function called for each drained order
    /// 
    /// # Returns
    /// Number of orders drained
    pub fn drain_output<F>(&self, output_idx: usize, mut callback: F) -> usize
    where
        F: FnMut(i64, i64, bool), // price, quantity, is_buy
    {
        if output_idx >= NUM_OUTPUTS {
            return 0;
        }
        
        let queue_base = output_idx * MAX_PENDING_PER_INPUT;
        let queue_len = self.output_queues[output_idx].load(Ordering::Acquire);
        
        let mut drained = 0;
        for i in 0..queue_len {
            let slot_idx = queue_base + i;
            let slot = &self.order_pool[slot_idx];
            
            if slot.occupied {
                callback(slot.price, slot.quantity, slot.is_buy);
                drained += 1;
                
                // Clear slot.
                let slot_mut = &mut self.order_pool[slot_idx];
                slot_mut.occupied = false;
            }
        }
        
        // Reset queue head.
        self.output_queues[output_idx].store(0, Ordering::Release);
        
        drained
    }
    
    /// Get crossbar statistics.
    pub fn get_stats(&self) -> CrossbarStats {
        let mut total_pending = 0;
        for i in 0..NUM_INPUTS {
            total_pending += self.state.pending_counts[i].load(Ordering::Relaxed);
        }
        
        CrossbarStats {
            total_routed: self.state.total_routed.load(Ordering::Relaxed),
            total_pending,
            routing_table: self.state.routing_table.clone(),
        }
    }
    
    /// Reset all routing configuration.
    pub fn reset(&self) {
        unsafe {
            let state_ptr = &*self.state as *const CrossbarState as *mut CrossbarState;
            (*state_ptr).routing_table.fill(0);
            for i in 0..NUM_INPUTS {
                self.state.pending_counts[i].store(0, Ordering::Relaxed);
            }
            for i in 0..NUM_OUTPUTS {
                self.output_queues[i].store(0, Ordering::Relaxed);
            }
            for slot in self.order_pool.iter_mut() {
                slot.occupied = false;
            }
        }
    }
}

impl Default for CrossbarSwitch {
    fn default() -> Self {
        Self::new()
    }
}

/// Crossbar switch statistics.
#[derive(Debug, Clone)]
pub struct CrossbarStats {
    pub total_routed: u64,
    pub total_pending: usize,
    pub routing_table: [u8; NUM_INPUTS],
}

/// Logging macro.
macro_rules! log_info {
    ($($arg:tt)*) => {
        eprintln!("[INFO] {}", format!($($arg)*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crossbar_creation() {
        let crossbar = CrossbarSwitch::new();
        let stats = crossbar.get_stats();
        assert_eq!(stats.total_routed, 0);
    }
    
    #[test]
    fn test_configure_route() {
        let crossbar = CrossbarSwitch::new();
        
        // Configure input 0 -> output 5.
        assert!(crossbar.configure_route(0, 5));
        assert!(!crossbar.configure_route(NUM_INPUTS, 0)); // Invalid
        assert!(!crossbar.configure_route(0, NUM_OUTPUTS)); // Invalid
        
        let stats = crossbar.get_stats();
        assert_eq!(stats.routing_table[0], 5);
    }
    
    #[test]
    fn test_route_order() {
        let crossbar = CrossbarSwitch::new();
        crossbar.configure_route(0, 10);
        
        assert!(crossbar.route_order(0, 50000, 100, true));
        
        let stats = crossbar.get_stats();
        assert_eq!(stats.total_routed, 1);
        assert_eq!(stats.total_pending, 1);
    }
    
    #[test]
    fn test_drain_output() {
        let crossbar = CrossbarSwitch::new();
        crossbar.configure_route(0, 5);
        
        crossbar.route_order(0, 50000, 100, true);
        crossbar.route_order(0, 50001, 200, false);
        
        let mut drained_count = 0;
        let mut total_qty = 0i64;
        
        crossbar.drain_output(5, |_price, qty, _is_buy| {
            drained_count += 1;
            total_qty += qty;
        });
        
        assert_eq!(drained_count, 2);
        assert_eq!(total_qty, 300);
        
        let stats = crossbar.get_stats();
        assert_eq!(stats.total_pending, 0); // Drained
    }
}
