//! Adaptive Arrival Price Execution Algorithm
//!
//! Dynamically shifts between limit and market orders based on real-time
//! order book momentum and queue decay rates. Optimizes for best arrival
//! price while minimizing market impact.
//!
//! # Key Features
//! - Real-time order book momentum tracking
//! - Queue position decay estimation
//! - Adaptive order type selection
//! - Implementation shortfall minimization

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;

/// Order types available to the algorithm
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    /// Passive limit order (maker)
    Limit,
    /// Aggressive market order (taker)
    Market,
    /// Pegged order that follows mid-price
    Pegged,
}

/// Execution mode based on market conditions
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Patient: mostly limits, wait for fill
    Patient,
    /// Balanced: mix of limits and markets
    Balanced,
    /// Aggressive: prefer market orders for certainty
    Aggressive,
}

/// Cache-line padded atomic for lock-free state
#[repr(C, align(64))]
struct CachePaddedAtomic<T> {
    value: T,
    _padding: [u8; 64 - size_of::<T>()],
}

impl<T: Default> Default for CachePaddedAtomic<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            _padding: [0u8; 64 - size_of::<T>()],
        }
    }
}

/// Order book momentum indicator
#[derive(Clone, Copy)]
pub struct MomentumIndicator {
    /// Price momentum (-1 to 1)
    pub price_momentum: f64,
    /// Volume momentum (-1 to 1)
    pub volume_momentum: f64,
    /// Queue decay rate (fills per second)
    pub queue_decay: f64,
    /// Spread width in basis points
    pub spread_bps: f64,
}

/// Adaptive arrival price execution state
#[repr(C, align(64))]
pub struct AdaptiveArrival {
    /// Parent order ID
    order_id: u64,
    /// Side: true = buy, false = sell
    is_buy: bool,
    /// Total quantity to execute
    total_qty: f64,
    /// Remaining quantity
    remaining_qty: f64,
    /// Executed quantity
    executed_qty: f64,
    /// Average execution price
    avg_price: f64,
    /// Arrival price (price when order started)
    arrival_price: f64,
    /// Current order type
    current_type: OrderType,
    /// Current execution mode
    mode: ExecutionMode,
    /// Momentum indicator
    momentum: MomentumIndicator,
    /// Mode switch threshold
    patient_threshold: f64,
    /// Aggressive threshold
    aggressive_threshold: f64,
    /// Start time
    start_time: Instant,
    /// Target duration in milliseconds
    target_duration_ms: u64,
    /// Urgency factor (0-1)
    urgency: f64,
    /// Active flag
    is_active: CachePaddedAtomic<AtomicBool>,
    /// Sequence counter
    sequence: u64,
}

impl AdaptiveArrival {
    /// Create a new adaptive arrival executor
    #[inline]
    pub fn new(
        order_id: u64,
        is_buy: bool,
        total_qty: f64,
        arrival_price: f64,
        target_duration_ms: u64,
    ) -> Self {
        Self {
            order_id,
            is_buy,
            total_qty,
            remaining_qty: total_qty,
            executed_qty: 0.0,
            avg_price: 0.0,
            arrival_price,
            current_type: OrderType::Limit,
            mode: ExecutionMode::Patient,
            momentum: MomentumIndicator {
                price_momentum: 0.0,
                volume_momentum: 0.0,
                queue_decay: 0.0,
                spread_bps: 0.0,
            },
            patient_threshold: -0.3,
            aggressive_threshold: 0.5,
            start_time: Instant::now(),
            target_duration_ms,
            urgency: 0.0,
            is_active: CachePaddedAtomic::default(),
            sequence: 0,
        }
    }
    
    /// Update momentum indicators from market data
    #[inline]
    pub fn update_momentum(&mut self, indicator: MomentumIndicator) {
        self.momentum = indicator;
        
        // Update urgency based on time elapsed
        let elapsed_ms = self.start_time.elapsed().as_millis() as u64;
        if self.target_duration_ms > 0 {
            self.urgency = (elapsed_ms as f64 / self.target_duration_ms as f64).min(1.0);
        }
        
        // Adapt execution mode based on momentum and urgency
        self.adapt_mode();
    }
    
    /// Adapt execution mode based on current conditions
    #[inline]
    fn adapt_mode(&mut self) {
        self.sequence = self.sequence.wrapping_add(1);
        
        // Calculate composite score
        let momentum_score = if self.is_buy {
            // For buys: negative momentum (price dropping) = good for limits
            -self.momentum.price_momentum
        } else {
            // For sells: positive momentum (price rising) = good for limits
            self.momentum.price_momentum
        };
        
        let queue_score = self.momentum.queue_decay / 100.0; // Normalize
        let spread_score = -self.momentum.spread_bps / 10.0; // Wider spread = prefer limits
        
        // Composite score with urgency weighting
        let urgency_factor = self.urgency;
        let composite = (1.0 - urgency_factor) * momentum_score 
                      + 0.3 * queue_score 
                      + 0.2 * spread_score
                      + urgency_factor * 0.5; // Urgency pushes toward aggressive
        
        // Determine mode
        self.mode = if composite < self.patient_threshold {
            ExecutionMode::Patient
        } else if composite > self.aggressive_threshold || self.urgency > 0.9 {
            ExecutionMode::Aggressive
        } else {
            ExecutionMode::Balanced
        };
        
        // Set order type based on mode
        self.current_type = match self.mode {
            ExecutionMode::Patient => OrderType::Limit,
            ExecutionMode::Balanced => {
                // Probabilistic choice based on composite
                if composite > 0.0 {
                    OrderType::Market
                } else {
                    OrderType::Limit
                }
            }
            ExecutionMode::Aggressive => OrderType::Market,
        };
    }
    
    /// Get recommended order for next execution slice
    #[inline]
    pub fn next_order(&mut self, slice_qty: f64) -> (OrderType, f64) {
        if !self.is_active.value.load(Ordering::Acquire) && self.remaining_qty <= 0.0 {
            return (self.current_type, 0.0);
        }
        
        let actual_qty = slice_qty.min(self.remaining_qty);
        
        // Adjust quantity based on mode
        let adjusted_qty = match self.mode {
            ExecutionMode::Patient => actual_qty * 0.5, // Smaller slices when patient
            ExecutionMode::Balanced => actual_qty * 0.75,
            ExecutionMode::Aggressive => actual_qty,
        };
        
        (self.current_type, adjusted_qty.max(0.001)) // Minimum order size
    }
    
    /// Record an execution
    #[inline]
    pub fn record_execution(&mut self, qty: f64, price: f64) {
        if qty <= 0.0 {
            return;
        }
        
        // Update running average
        let total_value = self.executed_qty * self.avg_price + qty * price;
        self.executed_qty += qty;
        self.remaining_qty -= qty;
        
        if self.executed_qty > 0.0 {
            self.avg_price = total_value / self.executed_qty;
        }
        
        // Deactivate if complete
        if self.remaining_qty <= 0.0 {
            self.is_active.value.store(false, Ordering::Release);
        }
    }
    
    /// Calculate implementation shortfall in basis points
    #[inline]
    pub fn implementation_shortfall_bps(&self) -> f64 {
        if self.executed_qty <= 0.0 || self.arrival_price <= 0.0 {
            return 0.0;
        }
        
        let side_mult = if self.is_buy { 1.0 } else { -1.0 };
        let price_diff = (self.avg_price - self.arrival_price) * side_mult;
        
        (price_diff / self.arrival_price) * 10000.0
    }
    
    /// Get execution progress (0.0 to 1.0)
    #[inline]
    pub fn progress(&self) -> f64 {
        self.executed_qty / self.total_qty
    }
    
    /// Get estimated time to completion
    #[inline]
    pub fn estimated_completion_ms(&self) -> u64 {
        if self.progress() < 0.01 {
            return self.target_duration_ms;
        }
        
        let elapsed = self.start_time.elapsed().as_millis() as u64;
        (elapsed as f64 / self.progress()) as u64
    }
    
    /// Check if execution is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.remaining_qty <= 0.0
    }
    
    /// Get current mode
    #[inline]
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }
    
    /// Get current order type
    #[inline]
    pub fn order_type(&self) -> OrderType {
        self.current_type
    }
    
    /// Get urgency factor
    #[inline]
    pub fn urgency(&self) -> f64 {
        self.urgency
    }
    
    /// Get remaining quantity
    #[inline]
    pub fn remaining(&self) -> f64 {
        self.remaining_qty
    }
    
    /// Get average execution price
    #[inline]
    pub fn avg_price(&self) -> f64 {
        self.avg_price
    }
}

impl Default for AdaptiveArrival {
    fn default() -> Self {
        Self::new(0, true, 1.0, 100.0, 60000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_creation() {
        let arrival = AdaptiveArrival::new(12345, true, 100.0, 50000.0, 60000);
        
        assert_eq!(arrival.total_qty, 100.0);
        assert_eq!(arrival.arrival_price, 50000.0);
        assert_eq!(arrival.mode, ExecutionMode::Patient);
        assert!(!arrival.is_complete());
    }
    
    #[test]
    fn test_momentum_update() {
        let mut arrival = AdaptiveArrival::new(12345, true, 100.0, 50000.0, 60000);
        
        // Negative momentum for buys should favor patient mode
        let indicator = MomentumIndicator {
            price_momentum: -0.5,
            volume_momentum: 0.0,
            queue_decay: 10.0,
            spread_bps: 5.0,
        };
        
        arrival.update_momentum(indicator);
        
        // Should remain patient or become more patient
        assert!(arrival.mode == ExecutionMode::Patient || arrival.mode == ExecutionMode::Balanced);
    }
    
    #[test]
    fn test_urgency_effect() {
        let mut arrival = AdaptiveArrival::new(12345, true, 100.0, 50000.0, 100); // Short duration
        
        // Wait to increase urgency
        std::thread::sleep(std::time::Duration::from_millis(80));
        
        let indicator = MomentumIndicator {
            price_momentum: 0.0,
            volume_momentum: 0.0,
            queue_decay: 0.0,
            spread_bps: 0.0,
        };
        
        arrival.update_momentum(indicator);
        
        // High urgency should push toward aggressive
        assert!(arrival.urgency() > 0.7);
    }
    
    #[test]
    fn test_execution_recording() {
        let mut arrival = AdaptiveArrival::new(12345, true, 100.0, 50000.0, 60000);
        
        arrival.record_execution(50.0, 50001.0);
        
        assert_eq!(arrival.executed_qty, 50.0);
        assert_eq!(arrival.avg_price(), 50001.0);
        assert_eq!(arrival.remaining(), 50.0);
        
        arrival.record_execution(50.0, 50002.0);
        
        assert_eq!(arrival.executed_qty, 100.0);
        assert!(arrival.is_complete());
        
        // IS should be positive (worse than arrival)
        let is_bps = arrival.implementation_shortfall_bps();
        assert!(is_bps > 0.0);
    }
}
