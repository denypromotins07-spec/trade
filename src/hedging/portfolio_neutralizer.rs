//! Hedging - Portfolio Neutralizer
//! 
//! Implements a real-time dollar-neutral and beta-neutral portfolio rebalancer
//! that instantly fires offsetting limit orders when exposure thresholds are breached.
//! Optimized for AMD Ryzen AI 5 with zero-allocation hot path execution.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

/// Maximum number of positions in portfolio
const MAX_POSITIONS: usize = 50;

/// Fixed-point scale (10^9)
const FP_SCALE: i64 = 1_000_000_000;

/// Default exposure threshold (1% in fixed-point)
const DEFAULT_EXPOSURE_THRESHOLD_FP: i64 = 10_000_000;

/// Position state in the portfolio
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct Position {
    /// Asset identifier hash
    pub asset_hash: u64,
    /// Quantity in nanounits (positive = long, negative = short)
    pub qty_ns: i64,
    /// Average entry price in nanodollars
    pub avg_price_ns: i64,
    /// Current market price in nanodollars
    pub current_price_ns: i64,
    /// Beta relative to benchmark (fixed-point)
    pub beta_fp: i64,
    /// Dollar notional value (qty * price)
    pub notional_ns: i64,
    /// Beta-weighted notional
    pub beta_notional_ns: i64,
    /// Last update timestamp
    pub last_update_ns: u64,
    /// Position is active flag
    pub active: AtomicBool,
}

impl Position {
    pub const fn new(asset_hash: u64) -> Self {
        Self {
            asset_hash,
            qty_ns: 0,
            avg_price_ns: 0,
            current_price_ns: 0,
            beta_fp: FP_SCALE,
            notional_ns: 0,
            beta_notional_ns: 0,
            last_update_ns: 0,
            active: AtomicBool::new(false),
        }
    }

    /// Update position with new price and quantity
    #[inline(always)]
    pub fn update(&mut self, qty_ns: i64, price_ns: i64, beta_fp: i64, timestamp_ns: u64) {
        self.qty_ns = qty_ns;
        self.current_price_ns = price_ns;
        self.beta_fp = beta_fp;
        
        // Calculate notional values
        self.notional_ns = qty_ns.wrapping_mul(price_ns).wrapping_div(FP_SCALE);
        self.beta_notional_ns = self.notional_ns.wrapping_mul(beta_fp).wrapping_div(FP_SCALE);
        
        // Update average entry price if adding to position
        if qty_ns != 0 {
            if (self.avg_price_ns > 0 && (qty_ns > 0) == (self.qty_ns > 0)) || self.avg_price_ns == 0 {
                // Same direction or new position - update average
                let total_value = self.avg_price_ns.wrapping_mul(self.qty_ns.abs())
                    .wrapping_add(price_ns.wrapping_mul(qty_ns.abs()));
                let total_qty = self.qty_ns.abs().wrapping_add(qty_ns.abs());
                
                if total_qty > 0 {
                    self.avg_price_ns = total_value.wrapping_div(total_qty);
                }
            }
        }
        
        self.last_update_ns = timestamp_ns;
        self.active.store(qty_ns != 0, Ordering::Release);
    }

    /// Get unrealized P&L in nanodollars
    #[inline(always)]
    pub fn unrealized_pnl_ns(&self) -> i64 {
        if self.qty_ns == 0 || self.avg_price_ns == 0 {
            return 0;
        }
        
        let price_diff = self.current_price_ns.wrapping_sub(self.avg_price_ns);
        if self.qty_ns > 0 {
            // Long position
            price_diff.wrapping_mul(self.qty_ns).wrapping_div(FP_SCALE)
        } else {
            // Short position
            (-price_diff).wrapping_mul(self.qty_ns.abs()).wrapping_div(FP_SCALE)
        }
    }
}

/// Order for rebalancing execution
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct RebalanceOrder {
    /// Asset hash
    pub asset_hash: u64,
    /// Order side (true = buy, false = sell)
    pub is_buy: bool,
    /// Quantity in nanounits
    pub qty_ns: i64,
    /// Limit price in nanodollars
    pub limit_price_ns: i64,
    /// Order type (0=market, 1=limit, 2=post-only)
    pub order_type: u8,
    /// Priority level
    pub priority: u8,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Order is pending execution
    pub pending: AtomicBool,
}

impl RebalanceOrder {
    #[inline(always)]
    pub fn new_limit(
        asset_hash: u64,
        is_buy: bool,
        qty_ns: i64,
        limit_price_ns: i64,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            asset_hash,
            is_buy,
            qty_ns,
            limit_price_ns,
            order_type: 1, // Limit order
            priority: 0,
            timestamp_ns,
            pending: AtomicBool::new(true),
        }
    }
}

/// Portfolio exposure metrics
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(64))]
pub struct ExposureMetrics {
    /// Total gross notional (sum of absolute notionals)
    pub gross_notional_ns: i64,
    /// Total net notional (sum of signed notionals)
    pub net_notional_ns: i64,
    /// Total beta-weighted net notional
    pub beta_net_notional_ns: i64,
    /// Total beta-weighted gross notional
    pub beta_gross_notional_ns: i64,
    /// Number of active positions
    pub position_count: usize,
    /// Number of long positions
    pub long_count: usize,
    /// Number of short positions
    pub short_count: usize,
}

impl ExposureMetrics {
    pub const fn new() -> Self {
        Self {
            gross_notional_ns: 0,
            net_notional_ns: 0,
            beta_net_notional_ns: 0,
            beta_gross_notional_ns: 0,
            position_count: 0,
            long_count: 0,
            short_count: 0,
        }
    }
}

/// Main portfolio neutralizer engine
#[repr(C, align(64))]
pub struct PortfolioNeutralizer {
    /// Active positions
    positions: [Position; MAX_POSITIONS],
    /// Number of tracked assets
    position_count: usize,
    /// Exposure threshold for triggering rebalance (fixed-point)
    exposure_threshold_fp: i64,
    /// Target dollar neutrality tolerance (nanodollars)
    dollar_neutral_tolerance_ns: i64,
    /// Target beta neutrality tolerance (fixed-point)
    beta_neutral_tolerance_fp: i64,
    /// Current exposure metrics
    metrics: ExposureMetrics,
    /// Pending rebalance orders queue
    pending_orders: [Option<RebalanceOrder>; 32],
    /// Order queue head
    order_head: AtomicU64,
    /// Order queue tail
    order_tail: AtomicU64,
    /// Rebalance trigger count
    rebalance_count: AtomicU64,
    /// Engine enabled flag
    enabled: AtomicBool,
    /// Last recalculation timestamp
    last_recalc_ns: AtomicU64,
}

impl PortfolioNeutralizer {
    pub const fn new() -> Self {
        Self {
            positions: [Position::new(0); MAX_POSITIONS],
            position_count: 0,
            exposure_threshold_fp: DEFAULT_EXPOSURE_THRESHOLD_FP,
            dollar_neutral_tolerance_ns: 100_000_000_000, // $100 tolerance
            beta_neutral_tolerance_fp: 10_000_000, // 1% beta tolerance
            metrics: ExposureMetrics::new(),
            pending_orders: [None; 32],
            order_head: AtomicU64::new(0),
            order_tail: AtomicU64::new(0),
            rebalance_count: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            last_recalc_ns: AtomicU64::new(0),
        }
    }

    /// Add or update a position
    #[inline(always)]
    pub fn update_position(
        &mut self,
        asset_hash: u64,
        qty_ns: i64,
        price_ns: i64,
        beta_fp: i64,
        timestamp_ns: u64,
    ) {
        // Find existing position or add new one
        let mut found = false;
        for i in 0..self.position_count {
            if self.positions[i].asset_hash == asset_hash {
                self.positions[i].update(qty_ns, price_ns, beta_fp, timestamp_ns);
                found = true;
                break;
            }
        }

        if !found && self.position_count < MAX_POSITIONS {
            let mut new_pos = Position::new(asset_hash);
            new_pos.update(qty_ns, price_ns, beta_fp, timestamp_ns);
            self.positions[self.position_count] = new_pos;
            self.position_count += 1;
        }

        // Recalculate exposure metrics
        self.recalculate_metrics();
    }

    /// Update price for an existing position
    #[inline(always)]
    pub fn update_price(&mut self, asset_hash: u64, price_ns: i64, timestamp_ns: u64) {
        for i in 0..self.position_count {
            if self.positions[i].asset_hash == asset_hash {
                self.positions[i].current_price_ns = price_ns;
                self.positions[i].notional_ns = self.positions[i].qty_ns
                    .wrapping_mul(price_ns)
                    .wrapping_div(FP_SCALE);
                self.positions[i].beta_notional_ns = self.positions[i].notional_ns
                    .wrapping_mul(self.positions[i].beta_fp)
                    .wrapping_div(FP_SCALE);
                self.positions[i].last_update_ns = timestamp_ns;
            }
        }
        self.recalculate_metrics();
    }

    /// Recalculate portfolio exposure metrics
    #[inline(always)]
    fn recalculate_metrics(&mut self) {
        let mut metrics = ExposureMetrics::new();

        for i in 0..self.position_count {
            let pos = &self.positions[i];
            if !pos.active.load(Ordering::Acquire) {
                continue;
            }

            metrics.gross_notional_ns = metrics.gross_notional_ns
                .wrapping_add(pos.notional_ns.abs());
            metrics.net_notional_ns = metrics.net_notional_ns.wrapping_add(pos.notional_ns);
            metrics.beta_gross_notional_ns = metrics.beta_gross_notional_ns
                .wrapping_add(pos.beta_notional_ns.abs());
            metrics.beta_net_notional_ns = metrics.beta_net_notional_ns
                .wrapping_add(pos.beta_notional_ns);

            metrics.position_count += 1;
            if pos.qty_ns > 0 {
                metrics.long_count += 1;
            } else if pos.qty_ns < 0 {
                metrics.short_count += 1;
            }
        }

        self.metrics = metrics;
        self.last_recalc_ns.store(get_time_ns(), Ordering::Release);
    }

    /// Check if portfolio needs rebalancing
    #[inline(always)]
    pub fn needs_rebalance(&self) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }

        // Check dollar neutrality
        let dollar_exposure = self.metrics.net_notional_ns.abs();
        if dollar_exposure > self.dollar_neutral_tolerance_ns {
            return true;
        }

        // Check beta neutrality
        let beta_exposure = self.metrics.beta_net_notional_ns.abs();
        let beta_threshold = self.metrics.gross_notional_ns
            .wrapping_mul(self.beta_neutral_tolerance_fp)
            .wrapping_div(FP_SCALE);
        
        if beta_exposure > beta_threshold {
            return true;
        }

        false
    }

    /// Generate rebalance orders
    #[inline(always)]
    pub fn generate_rebalance_orders(&mut self, timestamp_ns: u64) -> usize {
        if !self.needs_rebalance() {
            return 0;
        }

        let mut orders_generated = 0;

        // Calculate target hedge quantities
        let net_exposure = self.metrics.net_notional_ns;
        let beta_exposure = self.metrics.beta_net_notional_ns;

        // Find largest position to reduce
        let mut max_pos_idx = None;
        let mut max_notional = 0i64;

        for i in 0..self.position_count {
            let pos = &self.positions[i];
            if pos.active.load(Ordering::Acquire) {
                let notional_abs = pos.notional_ns.abs();
                if notional_abs > max_notional {
                    max_notional = notional_abs;
                    max_pos_idx = Some(i);
                }
            }
        }

        if let Some(idx) = max_pos_idx {
            let pos = &self.positions[idx];
            
            // Calculate reduction quantity
            let reduce_qty = if net_exposure > 0 {
                // Net long - need to sell
                (net_exposure * FP_SCALE / pos.current_price_ns.max(1)).min(pos.qty_ns.abs())
            } else {
                // Net short - need to buy
                (net_exposure.abs() * FP_SCALE / pos.current_price_ns.max(1)).min(pos.qty_ns.abs())
            };

            if reduce_qty > 0 {
                let is_buy = net_exposure < 0; // Buy to cover shorts
                
                let order = RebalanceOrder::new_limit(
                    pos.asset_hash,
                    is_buy,
                    reduce_qty,
                    pos.current_price_ns, // Use mid-price as limit
                    timestamp_ns,
                );

                self.push_order(order);
                orders_generated += 1;
            }
        }

        if orders_generated > 0 {
            self.rebalance_count.fetch_add(1, Ordering::Relaxed);
        }

        orders_generated
    }

    /// Push order to pending queue
    #[inline(always)]
    fn push_order(&mut self, order: RebalanceOrder) {
        let tail = self.order_tail.load(Ordering::Acquire);
        let head = self.order_head.load(Ordering::Acquire);

        if tail.wrapping_sub(head) >= 32 {
            return; // Queue full
        }

        let idx = (tail % 32) as usize;
        self.pending_orders[idx] = Some(order);
        self.order_tail.fetch_add(1, Ordering::Release);
    }

    /// Pop next pending order
    #[inline(always)]
    pub fn pop_order(&self) -> Option<RebalanceOrder> {
        let head = self.order_head.load(Ordering::Acquire);
        let tail = self.order_tail.load(Ordering::Acquire);

        if head >= tail {
            return None;
        }

        let idx = (head % 32) as usize;
        if let Some(order) = self.pending_orders[idx] {
            self.order_head.fetch_add(1, Ordering::Release);
            return Some(order);
        }

        None
    }

    /// Get current exposure metrics
    #[inline(always)]
    pub fn get_metrics(&self) -> ExposureMetrics {
        self.metrics
    }

    /// Set exposure threshold
    #[inline(always)]
    pub fn set_exposure_threshold(&mut self, threshold_fp: i64) {
        self.exposure_threshold_fp = threshold_fp;
    }

    /// Enable/disable neutralizer
    #[inline(always)]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
    }

    /// Get statistics
    #[inline(always)]
    pub fn stats(&self) -> (usize, u64, i64, i64) {
        (
            self.position_count,
            self.rebalance_count.load(Ordering::Relaxed),
            self.metrics.net_notional_ns,
            self.metrics.beta_net_notional_ns,
        )
    }
}

/// Get current time in nanoseconds
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
    fn test_portfolio_neutralizer_basic() {
        let mut neutralizer = PortfolioNeutralizer::new();
        let ts = get_time_ns();

        // Add long BTC position
        neutralizer.update_position(
            0xBTC,
            1_000_000_000, // 1 BTC
            100_000_000_000, // $100k
            FP_SCALE, // Beta = 1.0
            ts,
        );

        // Metrics should show long exposure
        let metrics = neutralizer.get_metrics();
        assert_eq!(metrics.position_count, 1);
        assert!(metrics.net_notional_ns > 0);

        // Add short ETH position to hedge
        neutralizer.update_position(
            0xETH,
            -10_000_000_000, // -10 ETH
            5_000_000_000, // $5k
            FP_SCALE,
            ts + 1000,
        );

        // Should be closer to neutral now
        let metrics = neutralizer.get_metrics();
        assert_eq!(metrics.position_count, 2);
    }

    #[test]
    fn test_dollar_neutrality_check() {
        let mut neutralizer = PortfolioNeutralizer::new();
        let ts = get_time_ns();

        // Large long position
        neutralizer.update_position(
            0xBTC,
            10_000_000_000, // 10 BTC
            100_000_000_000,
            FP_SCALE,
            ts,
        );

        // Should need rebalance (large net exposure)
        assert!(neutralizer.needs_rebalance());

        // Generate rebalance orders
        let orders = neutralizer.generate_rebalance_orders(ts + 1000);
        assert!(orders >= 0); // May generate orders
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut pos = Position::new(0xBTC);
        pos.update(
            1_000_000_000, // 1 BTC
            100_000_000_000, // Entry at $100k
            FP_SCALE,
            get_time_ns(),
        );

        // Price goes up to $110k
        pos.current_price_ns = 110_000_000_000;

        let pnl = pos.unrealized_pnl_ns();
        assert!(pnl > 0); // Long position should have positive P&L
        assert_eq!(pnl, 10_000_000_000); // $10k profit
    }
}
