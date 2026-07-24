// Capital Slicer: Microsecond order sizer ensuring all 6+ parallel execution engines
// never exceed total portfolio equity. Uses contiguous memory arrays to validate
// margin requirements before order submission. Optimized for AMD Ryzen AI 5.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crate::allocation::kelly_fraction::KellyResult;
use crate::allocation::margin_pool::MarginPool;

/// Fixed-point scale factor
const FP_SCALE: u64 = 1_000_000;

/// Maximum number of parallel symbol engines
const MAX_SYMBOL_ENGINES: usize = 8;

/// Minimum order size in quote currency (e.g., $10 USDT)
const MIN_ORDER_SIZE_USD: u64 = 10_000_000; // $10 in micro-units

/// Maximum single order as fraction of portfolio (25% = 250_000)
const MAX_ORDER_FRACTION_FP: u64 = 250_000;

/// Contiguous memory structure for order sizing validation
#[repr(C)]
pub struct OrderSizingContext {
    /// Portfolio equity in micro-USD
    pub portfolio_equity_micro: u64,
    /// Current used margin per symbol (micro-USD)
    pub used_margin_micro: [u64; MAX_SYMBOL_ENGINES],
    /// Available margin per symbol (micro-USD)
    pub available_margin_micro: [u64; MAX_SYMBOL_ENGINES],
    /// Kelly-recommended fractions per symbol (fixed-point)
    pub kelly_fractions_fp: [u64; MAX_SYMBOL_ENGINES],
    /// Active engine count
    pub active_engines: u8,
    /// Padding for alignment
    _padding: [u8; 7],
}

impl OrderSizingContext {
    /// Create a new context with zeroed margins
    pub const fn new() -> Self {
        Self {
            portfolio_equity_micro: 0,
            used_margin_micro: [0; MAX_SYMBOL_ENGINES],
            available_margin_micro: [0; MAX_SYMBOL_ENGINES],
            kelly_fractions_fp: [0; MAX_SYMBOL_ENGINES],
            active_engines: 0,
            _padding: [0; 7],
        }
    }

    /// Update portfolio equity and recalculate available margins
    #[inline]
    pub fn update_equity(&mut self, equity_micro: u64, total_used_margin_micro: u64) {
        self.portfolio_equity_micro = equity_micro;
        
        let total_available = equity_micro.saturating_sub(total_used_margin_micro);
        let per_engine = if self.active_engines > 0 {
            total_available / self.active_engines as u64
        } else {
            total_available
        };
        
        for i in 0..MAX_SYMBOL_ENGINES {
            self.available_margin_micro[i] = per_engine;
        }
    }

    /// Set Kelly fraction for a specific symbol engine
    #[inline]
    pub fn set_kelly_fraction(&mut self, engine_idx: usize, kelly_fp: u64) {
        if engine_idx < MAX_SYMBOL_ENGINES {
            self.kelly_fractions_fp[engine_idx] = kelly_fp.min(MAX_ORDER_FRACTION_FP);
        }
    }

    /// Record margin usage for a symbol engine
    #[inline]
    pub fn record_margin_usage(&mut self, engine_idx: usize, used_micro: u64) {
        if engine_idx < MAX_SYMBOL_ENGINES {
            self.used_margin_micro[engine_idx] = used_micro;
            self.available_margin_micro[engine_idx] = self.portfolio_equity_micro
                .saturating_sub(used_micro);
        }
    }
}

impl Default for OrderSizingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Capital Slicer: Validates and slices orders to fit within portfolio constraints
pub struct CapitalSlicer {
    /// Current sizing context (updated atomically)
    context: std::sync::RwLock<OrderSizingContext>,
    /// Emergency stop flag
    emergency_stop: AtomicBool,
    /// Total allocated capital (micro-USD)
    total_allocated_micro: AtomicU64,
    /// Maximum concurrent orders
    max_concurrent_orders: AtomicU64,
    /// Current concurrent order count
    concurrent_orders: AtomicU64,
}

/// Result of order sizing validation
#[derive(Debug, Clone, Copy)]
pub struct SizedOrder {
    /// Original requested size (micro-USD)
    pub requested_size_micro: u64,
    /// Approved size after slicing (micro-USD)
    pub approved_size_micro: u64,
    /// Symbol engine index
    pub engine_idx: u8,
    /// Whether the order was sliced down
    pub was_sliced: bool,
    /// Rejection reason (0=approved, 1=insufficient_margin, 2=kelly_exceeded, 3=emergency_stop)
    pub rejection_code: u8,
}

impl CapitalSlicer {
    /// Create a new capital slicer
    pub fn new(initial_equity_micro: u64) -> Self {
        let mut ctx = OrderSizingContext::new();
        ctx.portfolio_equity_micro = initial_equity_micro;
        ctx.active_engines = 6; // Default to 6 assets
        
        // Initialize available margins
        let per_engine = initial_equity_micro / 6;
        for i in 0..MAX_SYMBOL_ENGINES {
            ctx.available_margin_micro[i] = per_engine;
        }
        
        Self {
            context: std::sync::RwLock::new(ctx),
            emergency_stop: AtomicBool::new(false),
            total_allocated_micro: AtomicU64::new(0),
            max_concurrent_orders: AtomicU64::new(12), // 2 orders per engine max
            concurrent_orders: AtomicU64::new(0),
        }
    }

    /// Size an order for a specific symbol engine.
    /// Returns SizedOrder with approved size or rejection code.
    /// Zero heap allocations - suitable for microsecond hot path.
    #[inline(always)]
    pub fn size_order(
        &self,
        engine_idx: u8,
        requested_size_micro: u64,
        kelly_fraction_fp: u64,
    ) -> SizedOrder {
        // Check emergency stop first (fast path)
        if self.emergency_stop.load(Ordering::Acquire) {
            return SizedOrder {
                requested_size_micro,
                approved_size_micro: 0,
                engine_idx,
                was_sliced: false,
                rejection_code: 3, // emergency_stop
            };
        }

        // Check concurrent order limit
        let current_orders = self.concurrent_orders.load(Ordering::Acquire);
        let max_orders = self.max_concurrent_orders.load(Ordering::Acquire);
        if current_orders >= max_orders {
            return SizedOrder {
                requested_size_micro,
                approved_size_micro: 0,
                engine_idx,
                was_sliced: false,
                rejection_code: 4, // order_limit_exceeded
            };
        }

        // Read context (shared lock for reads)
        let ctx = self.context.read().unwrap();

        if engine_idx as usize >= MAX_SYMBOL_ENGINES {
            return SizedOrder {
                requested_size_micro,
                approved_size_micro: 0,
                engine_idx,
                was_sliced: false,
                rejection_code: 5, // invalid_engine
            };
        }

        let idx = engine_idx as usize;
        let available_margin = ctx.available_margin_micro[idx];
        let portfolio_equity = ctx.portfolio_equity_micro;

        // Calculate Kelly-maximum size
        let kelly_max_size = (portfolio_equity as u128 * kelly_fraction_fp as u128 / FP_SCALE as u128) as u64;

        // Determine approved size
        let mut approved_size = requested_size_micro;
        let mut was_sliced = false;
        let mut rejection_code = 0u8;

        // Check against available margin
        if approved_size > available_margin {
            approved_size = available_margin;
            was_sliced = true;
            
            if approved_size < MIN_ORDER_SIZE_USD {
                return SizedOrder {
                    requested_size_micro,
                    approved_size_micro: 0,
                    engine_idx,
                    was_sliced: false,
                    rejection_code: 1, // insufficient_margin
                };
            }
        }

        // Check against Kelly maximum
        if approved_size > kelly_max_size {
            approved_size = kelly_max_size;
            was_sliced = true;
            
            if approved_size < MIN_ORDER_SIZE_USD {
                return SizedOrder {
                    requested_size_micro,
                    approved_size_micro: 0,
                    engine_idx,
                    was_sliced: false,
                    rejection_code: 2, // kelly_exceeded
                };
            }
        }

        // Final clamp to max order fraction
        let max_order_size = (portfolio_equity as u128 * MAX_ORDER_FRACTION_FP as u128 / FP_SCALE as u128) as u64;
        if approved_size > max_order_size {
            approved_size = max_order_size;
            was_sliced = true;
        }

        SizedOrder {
            requested_size_micro,
            approved_size_micro: approved_size,
            engine_idx,
            was_sliced,
            rejection_code,
        }
    }

    /// Reserve capacity for an approved order (call before submission)
    #[inline]
    pub fn reserve_capacity(&self, engine_idx: u8, size_micro: u64) -> bool {
        if self.emergency_stop.load(Ordering::Acquire) {
            return false;
        }

        // Increment concurrent order count
        let prev = self.concurrent_orders.fetch_add(1, Ordering::AcqRel);
        if prev >= self.max_concurrent_orders.load(Ordering::Acquire) {
            self.concurrent_orders.fetch_sub(1, Ordering::Release);
            return false;
        }

        // Update allocated capital
        self.total_allocated_micro.fetch_add(size_micro, Ordering::AcqRel);

        // Update context
        let mut ctx = self.context.write().unwrap();
        if engine_idx as usize < MAX_SYMBOL_ENGINES {
            ctx.used_margin_micro[engine_idx as usize] += size_micro;
            ctx.available_margin_micro[engine_idx as usize] -= size_micro;
        }
        
        true
    }

    /// Release capacity after order completion/cancellation
    #[inline]
    pub fn release_capacity(&self, engine_idx: u8, size_micro: u64) {
        // Decrement concurrent order count
        self.concurrent_orders.fetch_sub(1, Ordering::Release);

        // Update allocated capital
        self.total_allocated_micro.fetch_sub(size_micro, Ordering::Release);

        // Update context
        let mut ctx = self.context.write().unwrap();
        if engine_idx as usize < MAX_SYMBOL_ENGINES {
            ctx.used_margin_micro[engine_idx as usize] = 
                ctx.used_margin_micro[engine_idx as usize].saturating_sub(size_micro);
            ctx.available_margin_micro[engine_idx as usize] += size_micro;
        }
    }

    /// Trigger emergency stop (blocks all new orders)
    #[inline]
    pub fn trigger_emergency_stop(&self) {
        self.emergency_stop.store(true, Ordering::Release);
    }

    /// Clear emergency stop
    #[inline]
    pub fn clear_emergency_stop(&self) {
        self.emergency_stop.store(false, Ordering::Release);
    }

    /// Get current utilization statistics
    pub fn get_utilization(&self) -> (u64, u64, u64) {
        let allocated = self.total_allocated_micro.load(Ordering::Acquire);
        let equity = self.context.read().unwrap().portfolio_equity_micro;
        let orders = self.concurrent_orders.load(Ordering::Acquire);
        (allocated, equity, orders)
    }

    /// Update portfolio equity from margin pool
    pub fn sync_with_margin_pool(&self, pool: &MarginPool) {
        let mut ctx = self.context.write().unwrap();
        let equity = pool.get_total_equity_micro();
        let total_used = pool.get_total_used_margin_micro();
        ctx.update_equity(equity, total_used);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_within_limits_approved() {
        let slicer = CapitalSlicer::new(100_000_000_000); // $100,000 in micro-USD
        
        let result = slicer.size_order(0, 1_000_000_000, 100_000); // $1,000 order, 10% Kelly
        
        assert_eq!(result.rejection_code, 0);
        assert_eq!(result.approved_size_micro, 1_000_000_000);
        assert!(!result.was_sliced);
    }

    #[test]
    fn test_order_exceeds_kelly_sliced() {
        let slicer = CapitalSlicer::new(100_000_000_000);
        
        // Request $50,000 but Kelly only allows 10% ($10,000)
        let result = slicer.size_order(0, 50_000_000_000, 100_000);
        
        assert_eq!(result.rejection_code, 0);
        assert!(result.was_sliced);
        assert_eq!(result.approved_size_micro, 10_000_000_000); // 10% of $100k
    }

    #[test]
    fn test_emergency_stop_rejects_all() {
        let slicer = CapitalSlicer::new(100_000_000_000);
        slicer.trigger_emergency_stop();
        
        let result = slicer.size_order(0, 1_000_000_000, 100_000);
        
        assert_eq!(result.rejection_code, 3);
        assert_eq!(result.approved_size_micro, 0);
    }

    #[test]
    fn test_reserve_and_release_capacity() {
        let slicer = CapitalSlicer::new(100_000_000_000);
        
        let size = 1_000_000_000;
        assert!(slicer.reserve_capacity(0, size));
        
        let (allocated, _, orders) = slicer.get_utilization();
        assert_eq!(allocated, size);
        assert_eq!(orders, 1);
        
        slicer.release_capacity(0, size);
        
        let (allocated_after, _, orders_after) = slicer.get_utilization();
        assert_eq!(allocated_after, 0);
        assert_eq!(orders_after, 0);
    }
}
