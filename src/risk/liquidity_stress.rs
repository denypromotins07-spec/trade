//! Liquidity Stress Testing & Time-to-Liquidate Calculator
//! 
//! This module models extreme liquidity evaporation scenarios, calculating
//! the exact time-to-liquidate during a flash crash using lock-free order book
//! depth degradation curves.
//! 
//! Optimized for: AMD Ryzen AI 5, microsecond calculations, 8GB RAM limit
//! Key Features:
//! - Lock-free order book depth modeling
//! - Liquidity evaporation simulation
//! - Time-to-liquidate calculation under stress
//! - Flash crash scenario modeling

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Memory budget for liquidity stress module (bytes)
const LIQUIDITY_MEMORY_BUDGET: usize = 512 * 1024 * 1024; // 512MB

/// Maximum order book depth levels to track
const MAX_DEPTH_LEVELS: usize = 100;

/// Default market impact coefficient
const DEFAULT_IMPACT_COEFFICIENT: f64 = 0.0001;

/// Order book level with atomic updates
#[derive(Debug, Clone)]
pub struct DepthLevel {
    pub price: u64,
    pub bid_size: AtomicU64,
    pub ask_size: AtomicU64,
    pub bid_orders: AtomicU64,
    pub ask_orders: AtomicU64,
}

impl DepthLevel {
    pub fn new(price: u64, bid_size: u64, ask_size: u64) -> Self {
        Self {
            price,
            bid_size: AtomicU64::new(bid_size),
            ask_size: AtomicU64::new(ask_size),
            bid_orders: AtomicU64::new(1),
            ask_orders: AtomicU64::new(1),
        }
    }
    
    /// Get current bid size
    pub fn get_bid_size(&self) -> u64 {
        self.bid_size.load(Ordering::Acquire)
    }
    
    /// Get current ask size
    pub fn get_ask_size(&self) -> u64 {
        self.ask_size.load(Ordering::Acquire)
    }
    
    /// Apply liquidity degradation (flash crash simulation)
    pub fn degrade_liquidity(&self, degradation_factor: f64) {
        let bid_current = self.bid_size.load(Ordering::Acquire);
        let ask_current = self.ask_size.load(Ordering::Acquire);
        
        let bid_new = (bid_current as f64 * (1.0 - degradation_factor)) as u64;
        let ask_new = (ask_current as f64 * (1.0 - degradation_factor)) as u64;
        
        self.bid_size.store(bid_new, Ordering::Release);
        self.ask_size.store(ask_new, Ordering::Release);
    }
}

/// Liquidity stress scenario types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressScenario {
    Normal,
    Elevated,
    Stressed,
    FlashCrash,
    LiquidityCrisis,
}

impl StressScenario {
    /// Get liquidity degradation factor for scenario
    pub fn degradation_factor(&self) -> f64 {
        match self {
            StressScenario::Normal => 0.0,
            StressScenario::Elevated => 0.3,
            StressScenario::Stressed => 0.6,
            StressScenario::FlashCrash => 0.85,
            StressScenario::LiquidityCrisis => 0.95,
        }
    }
    
    /// Get spread widening multiplier
    pub fn spread_multiplier(&self) -> f64 {
        match self {
            StressScenario::Normal => 1.0,
            StressScenario::Elevated => 2.0,
            StressScenario::Stressed => 5.0,
            StressScenario::FlashCrash => 15.0,
            StressScenario::LiquidityCrisis => 50.0,
        }
    }
}

/// Liquidity metrics snapshot
#[derive(Debug, Clone)]
pub struct LiquidityMetrics {
    pub total_bid_liquidity: u64,
    pub total_ask_liquidity: u64,
    pub weighted_spread_bps: f64,
    pub market_depth_score: f64,
    pub liquidity_ratio: f64,
}

/// Time-to-liquidate result
#[derive(Debug, Clone)]
pub struct LiquidationResult {
    pub quantity_to_liquidate: u64,
    pub estimated_time_ms: u64,
    pub expected_slippage_bps: f64,
    pub worst_case_price: u64,
    pub liquidity_consumed_pct: f64,
    pub scenario: StressScenario,
}

/// Lock-free order book depth tracker
pub struct OrderBookDepth {
    levels: Vec<Arc<DepthLevel>>,
    mid_price: AtomicU64,
    base_spread_bps: AtomicU64,
    memory_used: AtomicU64,
    last_update_ns: AtomicU64,
}

unsafe impl Send for OrderBookDepth {}
unsafe impl Sync for OrderBookDepth {}

impl OrderBookDepth {
    pub fn new(initial_price: u64, num_levels: usize, base_spread_bps: u64) -> Self {
        let mut levels = Vec::with_capacity(num_levels.min(MAX_DEPTH_LEVELS));
        let tick_size = 100; // Price increment per level
        
        for i in 0..num_levels.min(MAX_DEPTH_LEVELS) {
            let bid_price = initial_price.saturating_sub((i as u64 + 1) * tick_size);
            let ask_price = initial_price + (i as u64 + 1) * tick_size;
            
            // Decreasing liquidity at deeper levels
            let base_liquidity = 1_000_000 / (i as u64 + 1);
            
            levels.push(Arc::new(DepthLevel::new(
                bid_price,
                base_liquidity,
                base_liquidity,
            )));
        }
        
        Self {
            levels,
            mid_price: AtomicU64::new(initial_price),
            base_spread_bps: AtomicU64::new(base_spread_bps),
            memory_used: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }
    
    /// Get current liquidity metrics
    pub fn get_metrics(&self) -> LiquidityMetrics {
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;
        let mut weighted_spread_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, level) in self.levels.iter().enumerate() {
            let bid = level.get_bid_size();
            let ask = level.get_ask_size();
            
            total_bid += bid;
            total_ask += ask;
            
            // Weight closer levels more heavily
            let weight = 1.0 / (i as f64 + 1.0);
            let spread_at_level = ((level.price as i64 - self.mid_price.load(Ordering::Acquire) as i64).abs() as f64 
                / self.mid_price.load(Ordering::Acquire) as f64) * 10000.0;
            
            weighted_spread_sum += spread_at_level * weight;
            weight_sum += weight;
        }
        
        let weighted_spread = if weight_sum > 0.0 {
            weighted_spread_sum / weight_sum
        } else {
            0.0
        };
        
        let liquidity_ratio = if total_ask > 0 {
            total_bid as f64 / total_ask as f64
        } else {
            1.0
        };
        
        // Market depth score (0-100)
        let depth_score = ((total_bid + total_ask) as f64 / 10_000_000.0).min(100.0);
        
        LiquidityMetrics {
            total_bid_liquidity: total_bid,
            total_ask_liquidity: total_ask,
            weighted_spread_bps: weighted_spread,
            market_depth_score: depth_score,
            liquidity_ratio,
        }
    }
    
    /// Apply stress scenario to order book
    pub fn apply_stress(&self, scenario: StressScenario) {
        let degradation = scenario.degradation_factor();
        let spread_mult = scenario.spread_multiplier();
        
        for level in &self.levels {
            level.degrade_liquidity(degradation);
        }
        
        let base_spread = self.base_spread_bps.load(Ordering::Acquire);
        self.base_spread_bps.store(
            (base_spread as f64 * spread_mult) as u64,
            Ordering::Release,
        );
        
        self.last_update_ns.store(
            Instant::now().duration_since(Instant::now()).as_nanos() as u64,
            Ordering::Relaxed,
        );
    }
    
    /// Calculate time to liquidate a given quantity
    pub fn calculate_liquidation_time(
        &self,
        quantity: u64,
        max_participation_rate: f64,
        scenario: StressScenario,
    ) -> LiquidationResult {
        let metrics = self.get_metrics();
        
        // Adjust for stress scenario
        let effective_liquidity = metrics.total_ask_liquidity as f64 
            * (1.0 - scenario.degradation_factor());
        
        if effective_liquidity <= 0.0 {
            return LiquidationResult {
                quantity_to_liquidate: quantity,
                estimated_time_ms: u64::MAX,
                expected_slippage_bps: 10000.0,
                worst_case_price: 0,
                liquidity_consumed_pct: 100.0,
                scenario,
            };
        }
        
        // Calculate how much of the book we need to consume
        let participation_adjusted_liquidity = effective_liquidity * max_participation_rate;
        
        // Time estimation based on liquidity consumption
        let base_time_ms = if participation_adjusted_liquidity > 0.0 {
            (quantity as f64 / participation_adjusted_liquidity) * 1000.0
        } else {
            u64::MAX as f64
        };
        
        // Apply stress multiplier to time
        let stressed_time_ms = (base_time_ms * scenario.spread_multiplier()) as u64;
        
        // Calculate expected slippage using market impact model
        let impact_coefficient = DEFAULT_IMPACT_COEFFICIENT * scenario.spread_multiplier();
        let slippage_bps = impact_coefficient * (quantity as f64 / effective_liquidity) * 10000.0;
        
        // Worst case price (after full liquidation)
        let mid_price = self.mid_price.load(Ordering::Acquire);
        let worst_case_price = (mid_price as f64 * (1.0 - slippage_bps / 10000.0)) as u64;
        
        // Liquidity consumed percentage
        let liquidity_consumed = (quantity as f64 / effective_liquidity * 100.0).min(100.0);
        
        LiquidationResult {
            quantity_to_liquidate: quantity,
            estimated_time_ms: stressed_time_ms.min(u64::MAX),
            expected_slippage_bps: slippage_bps.min(10000.0),
            worst_case_price,
            liquidity_consumed_pct: liquidity_consumed,
            scenario,
        }
    }
    
    /// Get memory usage
    pub fn memory_used(&self) -> u64 {
        self.memory_used.load(Ordering::Relaxed)
    }
}

/// Liquidity stress testing engine
pub struct LiquidityStressEngine {
    order_books: Vec<Arc<OrderBookDepth>>,
    default_scenario: StressScenario,
    memory_used: AtomicU64,
    is_active: AtomicBool,
}

impl LiquidityStressEngine {
    pub fn new() -> Self {
        Self {
            order_books: Vec::new(),
            default_scenario: StressScenario::Normal,
            memory_used: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }
    
    /// Add an order book to monitor
    pub fn add_order_book(&mut self, symbol: &str, initial_price: u64, num_levels: usize) {
        let ob = Arc::new(OrderBookDepth::new(initial_price, num_levels, 10));
        
        self.memory_used.fetch_add(
            std::mem::size_of::<OrderBookDepth>() as u64 + 
            num_levels as u64 * std::mem::size_of::<DepthLevel>() as u64,
            Ordering::Relaxed,
        );
        
        self.order_books.push(ob);
    }
    
    /// Run stress test on all order books
    pub fn run_stress_test(&self, scenario: StressScenario, quantity: u64) -> Vec<LiquidationResult> {
        self.order_books.iter()
            .map(|ob| ob.calculate_liquidation_time(quantity, 0.1, scenario))
            .collect()
    }
    
    /// Get aggregated liquidity metrics
    pub fn get_aggregate_metrics(&self) -> AggregateLiquidityMetrics {
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;
        let mut avg_spread = 0.0;
        let mut count = 0;
        
        for ob in &self.order_books {
            let metrics = ob.get_metrics();
            total_bid += metrics.total_bid_liquidity;
            total_ask += metrics.total_ask_liquidity;
            avg_spread += metrics.weighted_spread_bps;
            count += 1;
        }
        
        AggregateLiquidityMetrics {
            total_bid_liquidity: total_bid,
            total_ask_liquidity: total_ask,
            average_spread_bps: if count > 0 { avg_spread / count as f64 } else { 0.0 },
            num_order_books: self.order_books.len(),
        }
    }
    
    /// Enforce memory limits
    pub fn enforce_memory_limit(&self, min_free_bytes: u64) -> bool {
        let current = self.memory_used.load(Ordering::Relaxed);
        if current > LIQUIDITY_MEMORY_BUDGET as u64 - min_free_bytes {
            return true;
        }
        false
    }
    
    /// Get engine statistics
    pub fn get_stats(&self) -> LiquidityEngineStats {
        LiquidityEngineStats {
            num_order_books: self.order_books.len(),
            memory_used: self.memory_used.load(Ordering::Relaxed),
            default_scenario: self.default_scenario,
            is_active: self.is_active.load(Ordering::Relaxed),
        }
    }
}

impl Default for LiquidityStressEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregate liquidity metrics across all order books
#[derive(Debug)]
pub struct AggregateLiquidityMetrics {
    pub total_bid_liquidity: u64,
    pub total_ask_liquidity: u64,
    pub average_spread_bps: f64,
    pub num_order_books: usize,
}

/// Engine statistics
#[derive(Debug)]
pub struct LiquidityEngineStats {
    pub num_order_books: usize,
    pub memory_used: u64,
    pub default_scenario: StressScenario,
    pub is_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_book_creation() {
        let ob = OrderBookDepth::new(50000, 10, 10);
        let metrics = ob.get_metrics();
        
        assert!(metrics.total_bid_liquidity > 0);
        assert!(metrics.total_ask_liquidity > 0);
    }
    
    #[test]
    fn test_stress_application() {
        let ob = OrderBookDepth::new(50000, 10, 10);
        let normal_metrics = ob.get_metrics();
        
        ob.apply_stress(StressScenario::FlashCrash);
        let stressed_metrics = ob.get_metrics();
        
        assert!(stressed_metrics.total_bid_liquidity < normal_metrics.total_bid_liquidity);
        assert!(stressed_metrics.total_ask_liquidity < normal_metrics.total_ask_liquidity);
    }
    
    #[test]
    fn test_liquidation_time_calculation() {
        let ob = OrderBookDepth::new(50000, 20, 10);
        
        let normal_result = ob.calculate_liquidation_time(100000, 0.1, StressScenario::Normal);
        let crash_result = ob.calculate_liquidation_time(100000, 0.1, StressScenario::FlashCrash);
        
        assert!(crash_result.estimated_time_ms > normal_result.estimated_time_ms);
        assert!(crash_result.expected_slippage_bps > normal_result.expected_slippage_bps);
    }
    
    #[test]
    fn test_stress_engine() {
        let mut engine = LiquidityStressEngine::new();
        engine.add_order_book("BTC", 50000, 20);
        engine.add_order_book("ETH", 3000, 20);
        
        let results = engine.run_stress_test(StressScenario::Stressed, 500000);
        assert_eq!(results.len(), 2);
    }
}
