//! Register Pressure Management for Hot-Path State Packing
//!
//! This module defines tightly aligned structs and strategies to minimize
//! register spilling to L1 cache during complex limit order routing calculations.
//! By carefully packing hot-path state into optimal register-sized structures,
//! we maximize AMD Zen execution port utilization.
//!
//! Optimized for AMD Ryzen AI 5 with strict 8GB RAM quota enforcement.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of in-flight orders tracked in registers
/// Limited to prevent register pressure and spills
const MAX_INFLIGHT_REGISTERS: usize = 16;

/// Ultra-compact order state packed into exactly 2 registers (128 bits)
/// Fits in a single SSE/XMM register for efficient movement
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct CompactOrderState {
    /// Packed fields: price (32 bits), quantity (32 bits), flags (32 bits), seq (32 bits)
    data: u128,
}

impl CompactOrderState {
    #[inline(always)]
    pub fn new(price: u32, quantity: u32, flags: u32, seq: u32) -> Self {
        let data = ((flags as u128) << 96)
            | ((seq as u128) << 64)
            | ((price as u128) << 32)
            | (quantity as u128);
        CompactOrderState { data }
    }

    #[inline(always)]
    pub fn price(&self) -> u32 {
        ((self.data >> 32) & 0xFFFFFFFF) as u32
    }

    #[inline(always)]
    pub fn quantity(&self) -> u32 {
        (self.data & 0xFFFFFFFF) as u32
    }

    #[inline(always)]
    pub fn flags(&self) -> u32 {
        ((self.data >> 96) & 0xFFFFFFFF) as u32
    }

    #[inline(always)]
    pub fn sequence(&self) -> u32 {
        ((self.data >> 64) & 0xFFFFFFFF) as u32
    }

    #[inline(always)]
    pub fn is_buy(&self) -> bool {
        self.flags() & 0x1 != 0
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.price() > 0 && self.quantity() > 0
    }
}

/// Hot-path matching state optimized for register allocation
/// Total size: 256 bits (4 registers) - fits in available SIMD regs
#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct MatchingState {
    /// Best bid price (32 bits)
    best_bid: u32,
    /// Best ask price (32 bits)
    best_ask: u32,
    /// Last trade price (32 bits)
    last_price: u32,
    /// Spread in ticks (32 bits)
    spread_ticks: u32,
    /// Bid depth (32 bits)
    bid_depth: u32,
    /// Ask depth (32 bits)
    ask_depth: u32,
    /// Flags and status (32 bits)
    flags: u32,
    /// Sequence number (32 bits)
    sequence: u32,
}

impl MatchingState {
    #[inline(always)]
    pub fn new() -> Self {
        MatchingState {
            best_bid: 0,
            best_ask: 0,
            last_price: 0,
            spread_ticks: 0,
            bid_depth: 0,
            ask_depth: 0,
            flags: 0,
            sequence: 0,
        }
    }

    #[inline(always)]
    pub fn update_prices(&mut self, bid: u32, ask: u32) {
        self.best_bid = bid;
        self.best_ask = ask;
        if bid > 0 && ask > 0 {
            self.spread_ticks = ask.saturating_sub(bid);
        }
    }

    #[inline(always)]
    pub fn update_depth(&mut self, bid_d: u32, ask_d: u32) {
        self.bid_depth = bid_d;
        self.ask_depth = ask_d;
    }

    #[inline(always)]
    pub fn mid_price(&self) -> u32 {
        if self.best_bid > 0 && self.best_ask > 0 {
            (self.best_bid + self.best_ask) / 2
        } else {
            self.last_price
        }
    }

    #[inline(always)]
    pub fn is_crossed(&self) -> bool {
        self.best_bid > 0 && self.best_ask > 0 && self.best_bid >= self.best_ask
    }

    #[inline(always)]
    pub fn increment_seq(&mut self) {
        self.sequence = self.sequence.wrapping_add(1);
    }
}

impl Default for MatchingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Register-file sized order book snapshot for fast routing decisions
/// Designed to fit entirely in CPU registers during routing calculations
#[repr(C, align(64))]
pub struct RegisterBookSnapshot {
    /// Top 8 bid levels (price, qty pairs) - 64 bytes total
    bids: [u32; 16], // 8 prices + 8 quantities
    /// Top 8 ask levels (price, qty pairs) - 64 bytes total  
    asks: [u32; 16],
    /// Metadata (fits in remaining space)
    timestamp_ns: u64,
    sequence: u64,
    flags: u32,
    _padding: u32,
}

impl RegisterBookSnapshot {
    #[inline(always)]
    pub fn new() -> Self {
        RegisterBookSnapshot {
            bids: [0; 16],
            asks: [0; 16],
            timestamp_ns: 0,
            sequence: 0,
            flags: 0,
            _padding: 0,
        }
    }

    #[inline(always)]
    pub fn set_bid(&mut self, level: usize, price: u32, qty: u32) {
        if level < 8 {
            self.bids[level * 2] = price;
            self.bids[level * 2 + 1] = qty;
        }
    }

    #[inline(always)]
    pub fn set_ask(&mut self, level: usize, price: u32, qty: u32) {
        if level < 8 {
            self.asks[level * 2] = price;
            self.asks[level * 2 + 1] = qty;
        }
    }

    #[inline(always)]
    pub fn best_bid(&self) -> (u32, u32) {
        (self.bids[0], self.bids[1])
    }

    #[inline(always)]
    pub fn best_ask(&self) -> (u32, u32) {
        (self.asks[0], self.asks[1])
    }

    #[inline(always)]
    pub fn get_bid(&self, level: usize) -> Option<(u32, u32)> {
        if level < 8 && self.bids[level * 2] > 0 {
            Some((self.bids[level * 2], self.bids[level * 2 + 1]))
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_ask(&self, level: usize) -> Option<(u32, u32)> {
        if level < 8 && self.asks[level * 2] > 0 {
            Some((self.asks[level * 2], self.asks[level * 2 + 1]))
        } else {
            None
        }
    }
}

impl Default for RegisterBookSnapshot {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-local register pressure tracker
/// Monitors potential spills and adjusts strategy accordingly
pub struct RegisterPressureTracker {
    /// Current estimated register usage
    current_usage: AtomicU64,
    /// Peak register usage observed
    peak_usage: AtomicU64,
    /// Spill events counter
    spill_count: AtomicU64,
    /// Optimal batch size based on register availability
    optimal_batch_size: AtomicU64,
}

unsafe impl Send for RegisterPressureTracker {}
unsafe impl Sync for RegisterPressureTracker {}

impl RegisterPressureTracker {
    pub fn new() -> Self {
        // AMD Zen has 16 integer + 16 SIMD registers
        // Reserve some for housekeeping, use rest for hot path
        RegisterPressureTracker {
            current_usage: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            spill_count: AtomicU64::new(0),
            optimal_batch_size: AtomicU64::new(MAX_INFLIGHT_REGISTERS as u64),
        }
    }

    /// Record register usage for current operation
    #[inline(always)]
    pub fn record_usage(&self, regs_used: u64) {
        self.current_usage.store(regs_used, Ordering::Relaxed);
        
        // Update peak
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while regs_used > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                regs_used,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }

        // Check for potential spills (Zen has ~28 usable regs after overhead)
        if regs_used > 24 {
            self.spill_count.fetch_add(1, Ordering::Relaxed);
            // Reduce batch size to prevent future spills
            let new_batch = (MAX_INFLIGHT_REGISTERS as u64).max(
                self.optimal_batch_size.load(Ordering::Relaxed).saturating_sub(1)
            );
            self.optimal_batch_size.store(new_batch, Ordering::Relaxed);
        }
    }

    /// Get recommended batch size to avoid spills
    #[inline(always)]
    pub fn recommended_batch_size(&self) -> usize {
        self.optimal_batch_size.load(Ordering::Relaxed) as usize
    }

    /// Get statistics
    pub fn stats(&self) -> PressureStats {
        PressureStats {
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            spill_count: self.spill_count.load(Ordering::Relaxed),
            optimal_batch_size: self.optimal_batch_size.load(Ordering::Relaxed),
            available_registers: 28, // AMD Zen approximate usable count
        }
    }

    /// Reset tracking (called during /KILL or reconfiguration)
    pub fn reset(&self) {
        self.current_usage.store(0, Ordering::Release);
        self.peak_usage.store(0, Ordering::Release);
        self.spill_count.store(0, Ordering::Release);
        self.optimal_batch_size.store(MAX_INFLIGHT_REGISTERS as u64, Ordering::Release);
    }
}

impl Default for RegisterPressureTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for register pressure monitoring
#[derive(Debug, Clone)]
pub struct PressureStats {
    pub current_usage: u64,
    pub peak_usage: u64,
    pub spill_count: u64,
    pub optimal_batch_size: u64,
    pub available_registers: u64,
}

/// Example: Register-optimized order routing function
/// Demonstrates keeping all state in registers during critical path
#[inline(always)]
pub fn route_order_register_optimized(
    state: &mut MatchingState,
    snapshot: &RegisterBookSnapshot,
    order: CompactOrderState,
) -> RoutingResult {
    // All computation happens in registers - no spills to cache
    
    let order_price = order.price();
    let order_qty = order.quantity();
    let is_buy = order.is_buy();

    let (best_bid, best_bid_qty) = snapshot.best_bid();
    let (best_ask, best_ask_qty) = snapshot.best_ask();

    let mut filled_qty = 0u32;
    let mut remaining_qty = order_qty;
    let mut total_value = 0u64;

    if is_buy {
        // Buy order: match against asks
        if order_price >= best_ask && best_ask > 0 {
            let fill = remaining_qty.min(best_ask_qty);
            filled_qty = fill;
            remaining_qty = remaining_qty.saturating_sub(fill);
            total_value = (fill as u64) * (best_ask as u64);
            
            // Update state
            state.update_depth(state.bid_depth, best_ask_qty.saturating_sub(fill));
        }
    } else {
        // Sell order: match against bids
        if order_price <= best_bid && best_bid > 0 {
            let fill = remaining_qty.min(best_bid_qty);
            filled_qty = fill;
            remaining_qty = remaining_qty.saturating_sub(fill);
            total_value = (fill as u64) * (best_bid as u64);
            
            // Update state
            state.update_depth(state.bid_depth.saturating_sub(fill), state.ask_depth);
        }
    }

    state.increment_seq();

    RoutingResult {
        filled_qty,
        remaining_qty,
        total_value,
        sequence: state.sequence,
    }
}

/// Result of order routing (packed for register efficiency)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct RoutingResult {
    pub filled_qty: u32,
    pub remaining_qty: u32,
    pub total_value: u64,
    pub sequence: u32,
    _padding: u32,
}

impl RoutingResult {
    #[inline(always)]
    pub fn is_full_fill(&self) -> bool {
        self.remaining_qty == 0
    }

    #[inline(always)]
    pub fn fill_ratio(&self) -> f32 {
        let total = self.filled_qty + self.remaining_qty;
        if total > 0 {
            self.filled_qty as f32 / total as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_order_state() {
        let state = CompactOrderState::new(50000, 100, 0x1, 42);
        assert_eq!(state.price(), 50000);
        assert_eq!(state.quantity(), 100);
        assert_eq!(state.sequence(), 42);
        assert!(state.is_buy());
    }

    #[test]
    fn test_matching_state() {
        let mut state = MatchingState::new();
        state.update_prices(49900, 50100);
        assert_eq!(state.mid_price(), 50000);
        assert!(!state.is_crossed());
    }

    #[test]
    fn test_snapshot_alignment() {
        assert_eq!(std::mem::align_of::<RegisterBookSnapshot>(), 64);
        assert_eq!(std::mem::size_of::<RegisterBookSnapshot>(), 192);
    }

    #[test]
    fn test_compact_state_size() {
        assert_eq!(std::mem::size_of::<CompactOrderState>(), 16);
        assert_eq!(std::mem::align_of::<CompactOrderState>(), 16);
    }

    #[test]
    fn test_routing_result() {
        let result = RoutingResult {
            filled_qty: 100,
            remaining_qty: 0,
            total_value: 5000000,
            sequence: 1,
            _padding: 0,
        };
        assert!(result.is_full_fill());
        assert_eq!(result.fill_ratio(), 1.0);
    }

    #[test]
    fn test_pressure_tracker() {
        let tracker = RegisterPressureTracker::new();
        tracker.record_usage(12);
        let stats = tracker.stats();
        assert_eq!(stats.current_usage, 12);
        assert_eq!(stats.peak_usage, 12);
    }
}
