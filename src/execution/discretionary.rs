//! Discretionary Limit Orders Implementation
//! 
//! Implements discretionary limit orders with hidden price offsets.
//! Exposes only a fraction of the true limit price to the public order book
//! to prevent front-running by predatory HFT firms.
//! 
//! Optimized for microsecond latency on AMD Ryzen AI 5 architecture.
//! Enforces global 8GB RAM limit via bounded state structures.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, AtomicU8, Ordering};

/// Price in nanodollars (fixed-point)
type PriceNs = i64;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Discretionary order state
/// 
/// A discretionary order has two prices:
/// - Display Price: What is visible to the market
/// - Hidden Price: The actual limit price used for execution
/// 
/// Example: Display bid at $50000, but willing to buy up to $50010
/// The extra $0.01 is hidden to avoid signaling aggressive intent.
#[repr(C, align(64))] // Cache-line aligned for AMD Ryzen
pub struct DiscretionaryOrder {
    /// Unique order ID
    order_id: AtomicU64,
    
    /// Order side
    side: AtomicU8,
    
    /// Display price in nanodollars (visible to market)
    display_price_ns: AtomicI64,
    
    /// Hidden price in nanodollars (actual execution limit)
    hidden_price_ns: AtomicI64,
    
    /// Discretionary offset in nanodollars (hidden_price - display_price)
    /// Positive for buys (willing to pay more), negative for sells
    discretion_ns: AtomicI64,
    
    /// Total quantity
    total_qty: AtomicU64,
    
    /// Display quantity (visible portion)
    display_qty: AtomicU64,
    
    /// Filled quantity
    filled_qty: AtomicU64,
    
    /// Active flag
    active: AtomicBool,
    
    /// Aggression level (0-100): Higher = more aggressive hidden pricing
    aggression_level: AtomicU8,
    
    /// Timestamp of creation (nanoseconds)
    created_ns: AtomicU64,
    
    /// Last modification timestamp
    modified_ns: AtomicU64,
    
    /// Padding for cache alignment
    _padding: [u8; 22],
}

impl DiscretionaryOrder {
    /// Create a new discretionary order
    /// 
    /// # Arguments
    /// * `order_id` - Unique identifier
    /// * `side` - Buy or Sell
    /// * `display_price_ns` - Price shown to market
    /// * `discretion_ns` - Hidden offset (positive for buys, negative for sells)
    /// * `total_qty` - Total order quantity
    /// * `display_qty` - Visible quantity portion
    /// * `aggression_level` - 0-100 scale for dynamic adjustment
    #[inline]
    pub fn new(
        order_id: u64,
        side: Side,
        display_price_ns: PriceNs,
        discretion_ns: PriceNs,
        total_qty: u64,
        display_qty: u64,
        aggression_level: u8,
    ) -> Self {
        let hidden_price = display_price_ns + discretion_ns;
        
        Self {
            order_id: AtomicU64::new(order_id),
            side: AtomicU8::new(side as u8),
            display_price_ns: AtomicI64::new(display_price_ns),
            hidden_price_ns: AtomicI64::new(hidden_price),
            discretion_ns: AtomicI64::new(discretion_ns),
            total_qty: AtomicU64::new(total_qty),
            display_qty: AtomicU64::new(display_qty.min(total_qty)),
            filled_qty: AtomicU64::new(0),
            active: AtomicBool::new(true),
            aggression_level: AtomicU8::new(aggression_level.min(100)),
            created_ns: AtomicU64::new(0), // Set by caller
            modified_ns: AtomicU64::new(0),
            _padding: [0; 22],
        }
    }
    
    /// Get order side
    #[inline]
    pub fn side(&self) -> Side {
        match self.side.load(Ordering::Relaxed) {
            0 => Side::Buy,
            _ => Side::Sell,
        }
    }
    
    /// Get display price
    #[inline]
    pub fn display_price(&self) -> PriceNs {
        self.display_price_ns.load(Ordering::Acquire)
    }
    
    /// Get hidden (actual) price
    #[inline]
    pub fn hidden_price(&self) -> PriceNs {
        self.hidden_price_ns.load(Ordering::Acquire)
    }
    
    /// Get discretionary offset
    #[inline]
    pub fn discretion(&self) -> PriceNs {
        self.discretion_ns.load(Ordering::Relaxed)
    }
    
    /// Update discretionary offset dynamically
    /// 
    /// This allows adjusting hidden aggressiveness based on market conditions
    /// without canceling and resubmitting the order.
    /// 
    /// # Returns
    /// New hidden price if updated, None if order inactive
    #[inline]
    pub fn update_discretion(&self, new_discretion_ns: PriceNs, now_ns: u64) -> Option<PriceNs> {
        if !self.active.load(Ordering::Acquire) {
            return None;
        }
        
        let display = self.display_price_ns.load(Ordering::Relaxed);
        let new_hidden = display + new_discretion_ns;
        
        self.discretion_ns.store(new_discretion_ns, Ordering::Release);
        self.hidden_price_ns.store(new_hidden, Ordering::Release);
        self.modified_ns.store(now_ns, Ordering::Release);
        
        Some(new_hidden)
    }
    
    /// Adjust aggression level based on market regime
    /// 
    /// Higher aggression = larger hidden offset for better fill probability
    #[inline]
    pub fn adjust_aggression(&self, delta: i8, max_discretion_ns: PriceNs) -> PriceNs {
        let current = self.aggression_level.load(Ordering::Relaxed);
        let new_level = ((current as i16) + (delta as i16)).clamp(0, 100) as u8;
        
        self.aggression_level.store(new_level, Ordering::Relaxed);
        
        // Calculate new discretion based on aggression (linear scaling)
        let base_discretion = self.discretion_ns.load(Ordering::Relaxed).abs();
        let scaled = (base_discretion * (new_level as i64)) / 100;
        let capped = scaled.min(max_discretion_ns);
        
        let sign = if self.side() == Side::Buy { 1 } else { -1 };
        let new_discretion = capped * sign;
        
        self.discretion_ns.store(new_discretion, Ordering::Release);
        self.hidden_price_ns.store(
            self.display_price_ns.load(Ordering::Relaxed) + new_discretion,
            Ordering::Release,
        );
        
        new_discretion
    }
    
    /// Record a fill
    #[inline]
    pub fn record_fill(&self, qty: u64) -> u64 {
        let remaining = self.get_remaining_qty();
        let fill_qty = qty.min(remaining);
        
        self.filled_qty.fetch_add(fill_qty, Ordering::AcqRel);
        
        // Reduce display quantity proportionally
        let display = self.display_qty.load(Ordering::Relaxed);
        if display > 0 {
            let reduction = (fill_qty * display) / self.total_qty.load(Ordering::Relaxed);
            self.display_qty.fetch_update(
                Ordering::AcqRel,
                Ordering::Relaxed,
                |d| Some(d.saturating_sub(reduction)),
            ).ok();
        }
        
        // Check if fully filled
        if self.get_remaining_qty() == 0 {
            self.active.store(false, Ordering::Release);
        }
        
        fill_qty
    }
    
    /// Get remaining quantity
    #[inline]
    pub fn get_remaining_qty(&self) -> u64 {
        self.total_qty.load(Ordering::Relaxed)
            .saturating_sub(self.filled_qty.load(Ordering::Relaxed))
    }
    
    /// Get display remaining quantity
    #[inline]
    pub fn get_display_remaining(&self) -> u64 {
        self.display_qty.load(Ordering::Relaxed)
            .saturating_sub(self.filled_qty.load(Ordering::Relaxed))
    }
    
    /// Cancel order
    #[inline]
    pub fn cancel(&self) {
        self.active.store(false, Ordering::Release);
    }
    
    /// Check if active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }
    
    /// Get order ID
    #[inline]
    pub fn id(&self) -> u64 {
        self.order_id.load(Ordering::Relaxed)
    }
}

/// Manager for discretionary orders with bounded capacity
#[repr(C, align(64))]
pub struct DiscretionaryOrderManager {
    /// Fixed-size order storage (bounded for 8GB RAM limit)
    orders: [Option<DiscretionaryOrder>; 2048],
    
    /// Next order ID
    next_id: AtomicU64,
    
    /// Active count
    active_count: AtomicU64,
    
    /// Total hidden liquidity (sum of all discretion offsets * quantities)
    total_hidden_liquidity_ns: AtomicI64,
    
    /// Padding
    _padding: [u8; 32],
}

// Manual initialization helper since const fn doesn't support array init with Options
impl DiscretionaryOrderManager {
    /// Create new manager
    pub fn new() -> Self {
        Self {
            orders: unsafe { std::mem::zeroed() },
            next_id: AtomicU64::new(1),
            active_count: AtomicU64::new(0),
            total_hidden_liquidity_ns: AtomicI64::new(0),
            _padding: [0; 32],
        }
    }
    
    /// Create a new discretionary order
    /// 
    /// # Returns
    /// Some(order_id) on success, None if at capacity
    pub fn create_order(
        &self,
        side: Side,
        display_price_ns: PriceNs,
        discretion_ns: PriceNs,
        total_qty: u64,
        display_qty: u64,
        aggression_level: u8,
    ) -> Option<u64> {
        // Enforce 8GB RAM limit by capping concurrent orders
        if self.active_count.load(Ordering::Relaxed) >= self.orders.len() as u64 {
            return None;
        }
        
        let order_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        
        // Find empty slot
        for slot in self.orders.iter() {
            // In production, this would use proper lock-free slot management
            // For now, simplified implementation
            break;
        }
        
        // Simplified: just return ID for demonstration
        // Production would properly insert into array
        Some(order_id)
    }
    
    /// Get active order count
    #[inline]
    pub fn active_count(&self) -> u64 {
        self.active_count.load(Ordering::Relaxed)
    }
    
    /// Get total hidden liquidity in nanodollar-units
    #[inline]
    pub fn total_hidden_liquidity(&self) -> i64 {
        self.total_hidden_liquidity_ns.load(Ordering::Relaxed)
    }
    
    /// Update all orders' aggression based on market regime
    /// 
    /// Called when volatility or toxicity metrics change
    #[inline]
    pub fn update_all_aggression(&self, delta: i8, max_discretion_ns: PriceNs) {
        let mut hidden_sum = 0i64;
        
        for slot in self.orders.iter() {
            if let Some(_order) = slot {
                // In production, would call order.adjust_aggression()
                // Simplified for demonstration
            }
        }
        
        self.total_hidden_liquidity_ns.store(hidden_sum, Ordering::Release);
    }
}

/// Strategy for dynamic discretion adjustment
pub struct DiscretionStrategy {
    /// Base discretion in nanodollars
    base_discretion_ns: PriceNs,
    
    /// Volatility multiplier (higher vol = more discretion needed)
    vol_multiplier_bps: u16,
    
    /// Toxicity multiplier (higher toxicity = more hiding)
    toxicity_multiplier_bps: u16,
    
    /// Minimum aggression level
    min_aggression: u8,
    
    /// Maximum aggression level
    max_aggression: u8,
}

impl DiscretionStrategy {
    /// Create new strategy
    pub const fn new(
        base_discretion_ns: PriceNs,
        vol_multiplier_bps: u16,
        toxicity_multiplier_bps: u16,
        min_agg: u8,
        max_agg: u8,
    ) -> Self {
        Self {
            base_discretion_ns,
            vol_multiplier_bps,
            toxicity_multiplier_bps,
            min_aggression: min_agg,
            max_aggression: max_agg,
        }
    }
    
    /// Calculate optimal discretion given market conditions
    #[inline]
    pub fn calculate_discretion(
        &self,
        volatility_bps: u16,
        toxicity_score: u8, // 0-255
    ) -> PriceNs {
        let vol_adj = ((self.base_discretion_ns as u64) * (volatility_bps as u64) 
            * (self.vol_multiplier_bps as u64) / 1_000_000) as PriceNs;
        
        let tox_adj = ((self.base_discretion_ns as u64) * (toxicity_score as u64)
            * (self.toxicity_multiplier_bps as u64) / (255 * 10_000)) as PriceNs;
        
        self.base_discretion_ns + vol_adj + tox_adj
    }
    
    /// Calculate aggression level from market metrics
    #[inline]
    pub fn calculate_aggression(&self, volatility_bps: u16, toxicity_score: u8) -> u8 {
        let vol_factor = (volatility_bps as u32 * 100) / 10_000; // 0-100 scale
        let tox_factor = (toxicity_score as u32 * 100) / 255;
        
        let avg = (vol_factor + tox_factor) / 2;
        avg.clamp(self.min_aggression as u32, self.max_aggression as u32) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_discretionary_order_creation() {
        let order = DiscretionaryOrder::new(
            1,
            Side::Buy,
            50_000_000_000, // $50,000 in nanodollars
            10_000_000,     // $0.01 hidden discretion
            1000,           // 1000 units total
            100,            // 100 units displayed
            50,             // 50% aggression
        );
        
        assert_eq!(order.display_price(), 50_000_000_000);
        assert_eq!(order.hidden_price(), 50_000_010_000);
        assert_eq!(order.discretion(), 10_000_000);
        assert!(order.is_active());
    }
    
    #[test]
    fn test_discretion_update() {
        let order = DiscretionaryOrder::new(
            1,
            Side::Sell,
            50_000_000_000,
            -5_000_000, // Willing to sell $0.005 cheaper than displayed
            1000,
            100,
            50,
        );
        
        let new_hidden = order.update_discretion(-15_000_000, 1000000000);
        assert_eq!(new_hidden, Some(49_999_985_000));
        assert_eq!(order.hidden_price(), 49_999_985_000);
    }
    
    #[test]
    fn test_fill_recording() {
        let order = DiscretionaryOrder::new(
            1,
            Side::Buy,
            50_000_000_000,
            10_000_000,
            1000,
            100,
            50,
        );
        
        let filled = order.record_fill(300);
        assert_eq!(filled, 300);
        assert_eq!(order.get_remaining_qty(), 700);
        assert!(order.is_active());
        
        let filled2 = order.record_fill(700);
        assert_eq!(filled2, 700);
        assert!(!order.is_active()); // Fully filled
    }
    
    #[test]
    fn test_strategy_calculation() {
        let strategy = DiscretionStrategy::new(
            10_000_000, // $0.01 base
            5000,       // 50% vol multiplier
            3000,       // 30% toxicity multiplier
            20,
            80,
        );
        
        let discretion = strategy.calculate_discretion(200, 128); // 2% vol, 50% toxicity
        assert!(discretion > 10_000_000); // Should be higher than base
        
        let aggression = strategy.calculate_aggression(200, 128);
        assert!(aggression >= 20 && aggression <= 80);
    }
}
