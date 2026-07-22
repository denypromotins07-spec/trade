//! Fee Optimizer - Real-Time Fee Calculation and Order Routing
//!
//! This module builds a real-time fee calculator that dynamically routes orders
//! between spot and futures markets based on current BNB balances, funding rates,
//! and maker/taker rebate tiers. Optimized for microsecond latency decisions.
//!
//! ## Features
//! - Multi-market fee comparison
//! - BNB discount calculations
//! - Funding rate arbitrage detection
//! - Maker/taker tier optimization
//! - Net P&L after fees estimation

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::collections::HashMap;

/// Market type for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketType {
    Spot,
    Futures,
    Margin,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type for fee calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    LimitMaker,
}

/// User VIP tier based on 30-day volume
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VipTier {
    Regular,
    Vip1,
    Vip2,
    Vip3,
    Vip4,
    Vip5,
    Vip6,
    Vip7,
    Vip8,
    Vip9,
}

impl VipTier {
    /// Get maker fee rate in basis points (1 bp = 0.01%)
    pub fn maker_fee_bps(self, market: MarketType) -> i64 {
        match (self, market) {
            // Spot maker fees (can be negative = rebate)
            (VipTier::Regular, MarketType::Spot) => 10,  // 0.10%
            (VipTier::Vip1, MarketType::Spot) => 9,
            (VipTier::Vip2, MarketType::Spot) => 8,
            (VipTier::Vip3, MarketType::Spot) => 7,
            (VipTier::Vip4, MarketType::Spot) => 6,
            (VipTier::Vip5, MarketType::Spot) => 5,
            (VipTier::Vip6, MarketType::Spot) => 4,
            (VipTier::Vip7, MarketType::Spot) => 3,
            (VipTier::Vip8, MarketType::Spot) => 2,
            (VipTier::Vip9, MarketType::Spot) => 1,
            
            // Futures maker fees
            (VipTier::Regular, MarketType::Futures) => 2,  // 0.02%
            (VipTier::Vip1, MarketType::Futures) => 2,
            (VipTier::Vip2, MarketType::Futures) => 1,
            (VipTier::Vip3, MarketType::Futures) => 0,
            (VipTier::Vip4, MarketType::Futures) => -1,   // Rebate!
            (VipTier::Vip5, MarketType::Futures) => -2,
            (VipTier::Vip6, MarketType::Futures) => -3,
            (VipTier::Vip7, MarketType::Futures) => -4,
            (VipTier::Vip8, MarketType::Futures) => -5,
            (VipTier::Vip9, MarketType::Futures) => -5,
            
            _ => 10,
        }
    }

    /// Get taker fee rate in basis points
    pub fn taker_fee_bps(self, market: MarketType) -> i64 {
        match (self, market) {
            // Spot taker fees
            (VipTier::Regular, MarketType::Spot) => 10,  // 0.10%
            (VipTier::Vip1, MarketType::Spot) => 9,
            (VipTier::Vip2, MarketType::Spot) => 8,
            (VipTier::Vip3, MarketType::Spot) => 7,
            (VipTier::Vip4, MarketType::Spot) => 6,
            (VipTier::Vip5, MarketType::Spot) => 5,
            (VipTier::Vip6, MarketType::Spot) => 4,
            (VipTier::Vip7, MarketType::Spot) => 3,
            (VipTier::Vip8, MarketType::Spot) => 2,
            (VipTier::Vip9, MarketType::Spot) => 1,
            
            // Futures taker fees
            (VipTier::Regular, MarketType::Futures) => 4,  // 0.04%
            (VipTier::Vip1, MarketType::Futures) => 4,
            (VipTier::Vip2, MarketType::Futures) => 3,
            (VipTier::Vip3, MarketType::Futures) => 3,
            (VipTier::Vip4, MarketType::Futures) => 2,
            (VipTier::Vip5, MarketType::Futures) => 2,
            (VipTier::Vip6, MarketType::Futures) => 1,
            (VipTier::Vip7, MarketType::Futures) => 1,
            (VipTier::Vip8, MarketType::Futures) => 1,
            (VipTier::Vip9, MarketType::Futures) => 1,
            
            _ => 10,
        }
    }
}

/// Fee calculation result
#[derive(Debug, Clone)]
pub struct FeeCalculation {
    pub market: MarketType,
    pub order_type: OrderType,
    pub side: OrderSide,
    pub quantity: u64,
    pub price: u64,
    pub notional: u64,
    pub base_fee_bps: i64,
    pub bnb_discount_bps: i64,
    pub final_fee_bps: i64,
    pub fee_amount: i64,  // Can be negative (rebate)
    pub estimated_slippage: u64,
    pub total_cost: i64,
}

impl FeeCalculation {
    /// Get effective fee rate after all discounts
    pub fn effective_rate_bps(&self) -> f64 {
        self.final_fee_bps as f64
    }

    /// Check if this is a rebate (negative fee)
    pub fn is_rebate(&self) -> bool {
        self.fee_amount < 0
    }
}

/// Routing decision with fee optimization
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub recommended_market: MarketType,
    pub recommended_order_type: OrderType,
    pub reason: String,
    pub expected_fee_bps: i64,
    pub expected_slippage: u64,
    pub total_expected_cost: i64,
    pub alternative_markets: Vec<MarketComparison>,
}

/// Comparison between markets
#[derive(Debug, Clone)]
pub struct MarketComparison {
    pub market: MarketType,
    pub fee_bps: i64,
    slippage_estimate: u64,
    total_cost: i64,
}

/// Current state for fee optimization
#[derive(Debug, Clone)]
pub struct FeeOptimizerState {
    /// Current BNB balance
    pub bnb_balance: u64,
    /// 30-day trading volume in USD (for VIP tier)
    pub volume_30d_usd: u64,
    /// Current funding rates by symbol (futures only)
    pub funding_rates: HashMap<String, i64>,  // In basis points per 8h
    /// Estimated slippage by market and symbol
    pub slippage_estimates: HashMap<(MarketType, String), u64>,
    /// Available liquidity by market
    pub liquidity_by_market: HashMap<MarketType, u64>,
}

impl Default for FeeOptimizerState {
    fn default() -> Self {
        Self {
            bnb_balance: 0,
            volume_30d_usd: 0,
            funding_rates: HashMap::new(),
            slippage_estimates: HashMap::new(),
            liquidity_by_market: HashMap::new(),
        }
    }
}

/// Main fee optimizer engine
pub struct FeeOptimizer {
    /// Current state
    state: parking_lot::RwLock<FeeOptimizerState>,
    /// Current VIP tier
    vip_tier: AtomicUsize,
    /// Enable BNB fee discount
    use_bnb_discount: AtomicBool,
    /// Statistics
    stats: parking_lot::RwLock<OptimizerStats>,
}

/// Optimizer statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizerStats {
    pub total_calculations: usize,
    pub spot_routes: usize,
    pub futures_routes: usize,
    pub total_fees_saved: i64,
    pub rebates_earned: i64,
}

impl FeeOptimizer {
    /// Create new fee optimizer
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(FeeOptimizerState::default()),
            vip_tier: AtomicUsize::new(VipTier::Regular as usize),
            use_bnb_discount: AtomicBool::new(true),
            stats: parking_lot::RwLock::new(OptimizerStats::default()),
        }
    }

    /// Update optimizer state
    pub fn update_state(&self, state: FeeOptimizerState) {
        *self.state.write() = state;
        
        // Update VIP tier based on volume
        let tier = self.calculate_vip_tier(state.volume_30d_usd);
        self.vip_tier.store(tier as usize, AtomicOrdering::Relaxed);
    }

    /// Calculate VIP tier from 30-day volume
    fn calculate_vip_tier(&self, volume_usd: u64) -> VipTier {
        match volume_usd {
            v if v >= 1_000_000_000 => VipTier::Vip9,
            v if v >= 500_000_000 => VipTier::Vip8,
            v if v >= 200_000_000 => VipTier::Vip7,
            v if v >= 100_000_000 => VipTier::Vip6,
            v if v >= 50_000_000 => VipTier::Vip5,
            v if v >= 20_000_000 => VipTier::Vip4,
            v if v >= 10_000_000 => VipTier::Vip3,
            v if v >= 1_000_000 => VipTier::Vip2,
            v if v >= 100_000 => VipTier::Vip1,
            _ => VipTier::Regular,
        }
    }

    /// Get current VIP tier
    pub fn get_vip_tier(&self) -> VipTier {
        unsafe { std::mem::transmute(self.vip_tier.load(AtomicOrdering::Relaxed)) }
    }

    /// Calculate fee for specific order parameters
    pub fn calculate_fee(
        &self,
        market: MarketType,
        order_type: OrderType,
        side: OrderSide,
        quantity: u64,
        price: u64,
        symbol: &str,
    ) -> FeeCalculation {
        let state = self.state.read();
        let tier = self.get_vip_tier();
        
        let notional = (quantity as u128 * price as u128 / 1_0000_0000) as u64; // Adjust for precision
        
        // Base fee based on order type and tier
        let base_fee_bps = match order_type {
            OrderType::Market => tier.taker_fee_bps(market),
            OrderType::Limit => tier.taker_fee_bps(market),
            OrderType::LimitMaker => tier.maker_fee_bps(market),
        };
        
        // BNB discount (25% off when paying with BNB)
        let bnb_discount_bps = if self.use_bnb_discount.load(AtomicOrdering::Relaxed) 
            && state.bnb_balance > 0 
            && market == MarketType::Spot 
        {
            (base_fee_bps.abs() * 25 / 100) as i64
        } else {
            0
        };
        
        // Final fee (bnb discount only applies to positive fees)
        let final_fee_bps = if base_fee_bps >= 0 {
            base_fee_bps - bnb_discount_bps
        } else {
            base_fee_bps  // Rebates not discounted
        };
        
        // Calculate fee amount
        let fee_amount = (notional as i128 * final_fee_bps as i128 / 10000) as i64;
        
        // Estimate slippage
        let slippage_key = (market, symbol.to_string());
        let estimated_slippage = state.slippage_estimates.get(&slippage_key)
            .copied()
            .unwrap_or(notional / 1000);  // Default 0.1%
        
        // Total cost including slippage
        let total_cost = fee_amount + estimated_slippage as i64;
        
        // Update stats
        drop(state);
        self.update_calculation_stats(market, fee_amount);
        
        FeeCalculation {
            market,
            order_type,
            side,
            quantity,
            price,
            notional,
            base_fee_bps,
            bnb_discount_bps,
            final_fee_bps,
            fee_amount,
            estimated_slippage,
            total_cost,
        }
    }

    /// Find optimal routing for order
    pub fn find_optimal_route(
        &self,
        symbol: &str,
        quantity: u64,
        price: u64,
        side: OrderSide,
    ) -> RoutingDecision {
        let state = self.state.read();
        
        // Calculate fees for each market
        let spot_fee = self.calculate_fee(
            MarketType::Spot,
            OrderType::LimitMaker,
            side,
            quantity,
            price,
            symbol,
        );
        
        let futures_fee = self.calculate_fee(
            MarketType::Futures,
            OrderType::LimitMaker,
            side,
            quantity,
            price,
            symbol,
        );
        
        // Compare total costs
        let mut best_market = MarketType::Spot;
        let mut best_cost = spot_fee.total_cost;
        let mut best_fee = spot_fee.final_fee_bps;
        
        if futures_fee.total_cost < spot_fee.total_cost {
            best_market = MarketType::Futures;
            best_cost = futures_fee.total_cost;
            best_fee = futures_fee.final_fee_bps;
        }
        
        // Consider funding rates for futures
        if let Some(funding_rate) = state.funding_rates.get(symbol) {
            // For long positions, funding rate is a cost; for shorts, it's income
            let funding_adjustment = match side {
                OrderSide::Buy => *funding_rate,
                OrderSide::Sell => -*funding_rate,
            };
            
            let adjusted_futures_cost = futures_fee.total_cost + funding_adjustment as i64;
            
            if adjusted_futures_cost < best_cost {
                best_market = MarketType::Futures;
                best_cost = adjusted_futures_cost;
            }
        }
        
        // Build recommendation
        let reason = if best_market == MarketType::Futures && spot_fee.is_rebate() {
            "Futures selected despite spot rebate due to lower total cost".to_string()
        } else if best_fee < 0 {
            format!("Maker rebate of {} bps available", -best_fee)
        } else if spot_fee.bnb_discount_bps > 0 && best_market == MarketType::Spot {
            format!("BNB discount applied: {} bps saved", spot_fee.bnb_discount_bps)
        } else {
            format!("Lowest total cost: {} bps fees + slippage", best_fee)
        };
        
        RoutingDecision {
            recommended_market: best_market,
            recommended_order_type: OrderType::LimitMaker,
            reason,
            expected_fee_bps: best_fee,
            expected_slippage: if best_market == MarketType::Spot {
                spot_fee.estimated_slippage
            } else {
                futures_fee.estimated_slippage
            },
            total_expected_cost: best_cost,
            alternative_markets: vec![
                MarketComparison {
                    market: MarketType::Spot,
                    fee_bps: spot_fee.final_fee_bps,
                    slippage_estimate: spot_fee.estimated_slippage,
                    total_cost: spot_fee.total_cost,
                },
                MarketComparison {
                    market: MarketType::Futures,
                    fee_bps: futures_fee.final_fee_bps,
                    slippage_estimate: futures_fee.estimated_slippage,
                    total_cost: futures_fee.total_cost,
                },
            ],
        }
    }

    /// Update statistics
    fn update_calculation_stats(&self, market: MarketType, fee: i64) {
        let mut stats = self.stats.write();
        stats.total_calculations += 1;
        
        match market {
            MarketType::Spot => stats.spot_routes += 1,
            MarketType::Futures => stats.futures_routes += 1,
            _ => {}
        }
        
        if fee < 0 {
            stats.rebates_earned += fee.abs();
        }
    }

    /// Enable/disable BNB discount
    pub fn set_bnb_discount(&self, enabled: bool) {
        self.use_bnb_discount.store(enabled, AtomicOrdering::Relaxed);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> OptimizerStats {
        self.stats.read().clone()
    }

    /// Calculate net P&L after all fees
    pub fn calculate_net_pnl(
        &self,
        gross_pnl: i64,
        entry_fee: &FeeCalculation,
        exit_fee: &FeeCalculation,
    ) -> i64 {
        gross_pnl - entry_fee.fee_amount - exit_fee.fee_amount
    }

    /// Get break-even price considering fees
    pub fn get_break_even_price(
        &self,
        entry_price: u64,
        side: OrderSide,
        entry_fee_bps: i64,
        exit_fee_bps: i64,
    ) -> u64 {
        let total_fee_bps = (entry_fee_bps + exit_fee_bps).max(0) as u64;
        
        match side {
            OrderSide::Buy => {
                // Need price to rise enough to cover fees
                let fee_adjustment = (entry_price as u128 * total_fee_bps as u128 / 10000) as u64;
                entry_price.saturating_add(fee_adjustment)
            }
            OrderSide::Sell => {
                // Need price to fall enough to cover fees
                let fee_adjustment = (entry_price as u128 * total_fee_bps as u128 / 10000) as u64;
                entry_price.saturating_sub(fee_adjustment)
            }
        }
    }
}

impl Default for FeeOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current time in nanoseconds
fn get_current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vip_tier_fees() {
        let regular = VipTier::Regular;
        assert_eq!(regular.maker_fee_bps(MarketType::Spot), 10);
        assert_eq!(regular.taker_fee_bps(MarketType::Spot), 10);
        assert_eq!(regular.maker_fee_bps(MarketType::Futures), 2);
        assert_eq!(regular.taker_fee_bps(MarketType::Futures), 4);
    }

    #[test]
    fn test_fee_calculation_basic() {
        let optimizer = FeeOptimizer::new();
        
        let fee = optimizer.calculate_fee(
            MarketType::Spot,
            OrderType::LimitMaker,
            OrderSide::Buy,
            1000,  // quantity
            50000, // price
            "BTCUSDT",
        );
        
        assert_eq!(fee.market, MarketType::Spot);
        assert!(fee.fee_amount >= 0);
    }

    #[test]
    fn test_routing_decision() {
        let optimizer = FeeOptimizer::new();
        
        let mut state = FeeOptimizerState::default();
        state.volume_30d_usd = 100_000_000; // VIP6
        optimizer.update_state(state);
        
        let decision = optimizer.find_optimal_route(
            "BTCUSDT",
            1000,
            50000,
            OrderSide::Buy,
        );
        
        assert!(decision.recommended_market == MarketType::Spot 
            || decision.recommended_market == MarketType::Futures);
        assert!(!decision.reason.is_empty());
    }

    #[test]
    fn test_break_even_calculation() {
        let optimizer = FeeOptimizer::new();
        
        // Long position: need price to rise above entry + fees
        let break_even = optimizer.get_break_even_price(50000, OrderSide::Buy, 10, 10);
        assert!(break_even > 50000);
        
        // Short position: need price to fall below entry - fees
        let break_even_short = optimizer.get_break_even_price(50000, OrderSide::Sell, 10, 10);
        assert!(break_even_short < 50000);
    }

    #[test]
    fn test_rebate_detection() {
        let optimizer = FeeOptimizer::new();
        
        let mut state = FeeOptimizerState::default();
        state.volume_30d_usd = 1_000_000_000; // VIP9
        optimizer.update_state(state);
        
        let fee = optimizer.calculate_fee(
            MarketType::Futures,
            OrderType::LimitMaker,
            OrderSide::Buy,
            10000,
            50000,
            "BTCUSDT",
        );
        
        // VIP9 gets maker rebates on futures
        assert!(fee.is_rebate() || fee.final_fee_bps <= 0);
    }
}
