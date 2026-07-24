//! cross_margin_sync.rs - Cross-Margin Synchronization Engine
//! Stage 54: Nautilus/Ray Crypto Trading Bot
//! Synchronizes cross-margin and portfolio exposure across parallel threads in O(1) time
//! Uses atomic read-copy-update (RCU) pointers for lock-free concurrent access

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam::epoch::{self, Atomic as EpochAtomic, Guard};
use log::{debug, error, info, warn};
use parking_lot::RwLock;

use crate::execution::parallel_router::SymbolId;

/// Maximum number of symbols for cross-margin calculation
const MAX_SYMBOLS: usize = 12;

/// Precision for fixed-point PnL calculations (micro-dollars)
const PNL_PRECISION: i64 = 1_000_000;

/// Margin call threshold (90% of max leverage)
const MARGIN_CALL_THRESHOLD: i64 = 900_000; // 90% in basis points

/// Liquidation threshold (95% of max leverage)
const LIQUIDATION_THRESHOLD: i64 = 950_000; // 95% in basis points

/// Portfolio exposure state (atomically updated via RCU)
#[derive(Debug, Clone)]
pub struct ExposureState {
    /// Total long exposure in micro-dollars
    pub long_exposure: i64,
    /// Total short exposure in micro-dollars
    pub short_exposure: i64,
    /// Net exposure (long - short)
    pub net_exposure: i64,
    /// Gross exposure (long + short)
    pub gross_exposure: i64,
    /// Unrealized PnL in micro-dollars
    pub unrealized_pnl: i64,
    /// Realized PnL today in micro-dollars
    pub realized_pnl_today: i64,
    /// Number of open positions
    pub position_count: u32,
    /// Last update timestamp
    pub last_update: Instant,
    /// Leverage ratio in basis points
    pub leverage_bps: i64,
    /// Available margin in micro-dollars
    pub available_margin: i64,
    /// Used margin in micro-dollars
    pub used_margin: i64,
}

impl ExposureState {
    pub fn new(initial_margin: i64) -> Self {
        Self {
            long_exposure: 0,
            short_exposure: 0,
            net_exposure: 0,
            gross_exposure: 0,
            unrealized_pnl: 0,
            realized_pnl_today: 0,
            position_count: 0,
            last_update: Instant::now(),
            leverage_bps: 0,
            available_margin: initial_margin,
            used_margin: 0,
        }
    }

    /// Calculate current leverage ratio in basis points
    pub fn calculate_leverage(&self, equity: i64) -> i64 {
        if equity <= 0 {
            return 0;
        }
        (self.gross_exposure * 10000) / equity
    }

    /// Check if margin call should be triggered
    pub fn is_margin_call(&self, max_leverage_bps: i64) -> bool {
        if max_leverage_bps <= 0 {
            return false;
        }
        self.leverage_bps >= (max_leverage_bps * MARGIN_CALL_THRESHOLD) / 10000
    }

    /// Check if liquidation is imminent
    pub fn is_liquidation_risk(&self, max_leverage_bps: i64) -> bool {
        if max_leverage_bps <= 0 {
            return false;
        }
        self.leverage_bps >= (max_leverage_bps * LIQUIDATION_THRESHOLD) / 10000
    }
}

impl Default for ExposureState {
    fn default() -> Self {
        Self::new(10_000_000_000) // 10,000 USDT default initial margin
    }
}

/// Per-symbol exposure tracking
#[derive(Debug, Clone)]
pub struct SymbolExposure {
    /// Symbol ID
    pub symbol_id: SymbolId,
    /// Long position size in base units (fixed-point)
    pub long_size: i64,
    /// Short position size in base units (fixed-point)
    pub short_size: i64,
    /// Average entry price for longs (fixed-point)
    pub long_avg_price: i64,
    /// Average entry price for shorts (fixed-point)
    pub short_avg_price: i64,
    /// Current mark price (fixed-point)
    pub mark_price: i64,
    /// Unrealized PnL for this symbol
    pub unrealized_pnl: i64,
    /// Notional value in micro-dollars
    pub notional_value: i64,
}

impl SymbolExposure {
    pub fn new(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            long_size: 0,
            short_size: 0,
            long_avg_price: 0,
            short_avg_price: 0,
            mark_price: 0,
            unrealized_pnl: 0,
            notional_value: 0,
        }
    }

    /// Update mark price and recalculate PnL
    pub fn update_mark_price(&mut self, price: i64) {
        self.mark_price = price;
        
        // Calculate unrealized PnL
        let long_pnl = if self.long_size > 0 && self.long_avg_price > 0 {
            ((price - self.long_avg_price) * self.long_size) / PNL_PRECISION
        } else {
            0
        };
        
        let short_pnl = if self.short_size > 0 && self.short_avg_price > 0 {
            ((self.short_avg_price - price) * self.short_size) / PNL_PRECISION
        } else {
            0
        };
        
        self.unrealized_pnl = long_pnl + short_pnl;
        self.notional_value = (price * (self.long_size + self.short_size)) / PNL_PRECISION;
    }
}

/// Read-Copy-Update (RCU) protected exposure state
struct RcuExposureState {
    /// Current pointer to exposure state (atomic)
    current: EpochAtomic<ExposureState>,
    /// Generation counter for ABA prevention
    generation: AtomicU64,
}

impl RcuExposureState {
    fn new(initial_state: ExposureState) -> Self {
        Self {
            current: EpochAtomic::new(initial_state),
            generation: AtomicU64::new(0),
        }
    }

    /// Read current state (lock-free)
    fn read<'a>(&'a self, guard: &'a Guard) -> &'a ExposureState {
        guard.reference(&self.current)
    }

    /// Update state atomically (O(1) operation)
    fn update<F>(&self, updater: F) -> Result<(), &'static str>
    where
        F: FnOnce(&ExposureState) -> ExposureState,
    {
        let guard = epoch::pin();
        
        // Load current pointer
        let current = self.current.load(Ordering::Acquire, &guard);
        if current.is_null() {
            return Err("Null pointer in RCU state");
        }

        // Create new state
        let old_state = unsafe { current.as_ref() }.unwrap();
        let new_state = updater(old_state);

        // Allocate new state
        let new_ptr = epoch::Owned::new(new_state);
        
        // Compare-and-swap
        match self.current.compare_and_set_current(current, new_ptr, Ordering::AcqRel, &guard) {
            Ok(_) => {
                // Success - defer cleanup of old state
                unsafe {
                    guard.defer_destroy(current);
                }
                self.generation.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                // CAS failed - drop the new allocation
                drop(e.new);
                Err("CAS failed - concurrent modification")
            }
        }
    }
}

/// Cross-margin synchronizer for multi-threaded exposure management
pub struct CrossMarginSync {
    /// Global portfolio exposure (RCU protected)
    global_exposure: Arc<RcuExposureState>,
    
    /// Per-symbol exposures
    symbol_exposures: Arc<RwLock<Vec<SymbolExposure>>>,
    
    /// Initial margin in micro-dollars
    initial_margin: AtomicI64,
    
    /// Max allowed leverage in basis points
    max_leverage_bps: AtomicI64,
    
    /// Emergency stop flag
    emergency_stop: AtomicBool,
    
    /// Margin call active flag
    margin_call_active: AtomicBool,
    
    /// Total updates performed
    update_count: AtomicUsize,
    
    /// Last sync timestamp
    last_sync_time: AtomicU64,
}

impl CrossMarginSync {
    /// Create a new cross-margin synchronizer
    pub fn new() -> Result<Self, String> {
        let initial_margin: i64 = 10_000_000_000; // 10,000 USDT
        let initial_state = ExposureState::new(initial_margin);
        
        // Initialize per-symbol exposures
        let mut symbol_exposures = Vec::with_capacity(MAX_SYMBOLS);
        for i in 0..MAX_SYMBOLS {
            // Create dummy symbol IDs for initialization
            symbol_exposures.push(SymbolExposure::new(
                SymbolId::from_symbol("BTCUSDT").unwrap_or(SymbolId::new(0))
            ));
        }

        Ok(Self {
            global_exposure: Arc::new(RcuExposureState::new(initial_state)),
            symbol_exposures: Arc::new(RwLock::new(symbol_exposures)),
            initial_margin: AtomicI64::new(initial_margin),
            max_leverage_bps: AtomicI64::new(20000), // 20x max leverage
            emergency_stop: AtomicBool::new(false),
            margin_call_active: AtomicBool::new(false),
            update_count: AtomicUsize::new(0),
            last_sync_time: AtomicU64::new(0),
        })
    }

    /// Atomically update exposure for a symbol (O(1) time complexity)
    pub fn update_exposure_atomic(&self, symbol_id: SymbolId, delta: f64) -> Result<(), String> {
        if self.emergency_stop.load(Ordering::Relaxed) {
            return Err("Emergency stop active".to_string());
        }

        let delta_micro = (delta * PNL_PRECISION as f64) as i64;

        self.global_exposure.update(|state| {
            let mut new_state = state.clone();
            
            // Update exposure based on delta sign
            if delta > 0.0 {
                new_state.long_exposure += delta_micro;
            } else {
                new_state.short_exposure += delta_micro.abs();
            }
            
            // Recalculate derived values
            new_state.net_exposure = new_state.long_exposure - new_state.short_exposure;
            new_state.gross_exposure = new_state.long_exposure + new_state.short_exposure;
            
            // Update margin calculations
            let equity = new_state.available_margin + new_state.unrealized_pnl;
            new_state.leverage_bps = new_state.calculate_leverage(equity);
            new_state.used_margin = new_state.gross_exposure / 
                (self.max_leverage_bps.load(Ordering::Relaxed) / 100);
            
            new_state.last_update = Instant::now();
            new_state
        }).map_err(|e| e.to_string())?;

        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        // Store timestamp (nanoseconds since epoch approximation)
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        self.last_sync_time.store(now_ns, Ordering::Release);

        // Check for margin call
        self.check_margin_status();

        Ok(())
    }

    /// Get current global exposure state (lock-free read)
    pub fn get_exposure(&self) -> ExposureState {
        let guard = epoch::pin();
        let state = self.global_exposure.read(&guard);
        state.clone()
    }

    /// Get exposure for a specific symbol
    pub fn get_symbol_exposure(&self, symbol_id: SymbolId) -> Option<SymbolExposure> {
        let exposures = self.symbol_exposures.read();
        exposures.iter().find(|e| e.symbol_id == symbol_id).cloned()
    }

    /// Set mark price for a symbol and update PnL
    pub fn set_mark_price(&self, symbol_id: SymbolId, price: f64) -> Result<(), String> {
        let price_fixed = (price * PNL_PRECISION as f64) as i64;
        
        let mut exposures = self.symbol_exposures.write();
        if let Some(exposure) = exposures.iter_mut().find(|e| e.symbol_id == symbol_id) {
            exposure.update_mark_price(price_fixed);
            
            // Recalculate total unrealized PnL
            let total_unrealized: i64 = exposures.iter().map(|e| e.unrealized_pnl).sum();
            
            // Update global state
            drop(exposures);
            self.global_exposure.update(|state| {
                let mut new_state = state.clone();
                new_state.unrealized_pnl = total_unrealized;
                new_state
            }).map_err(|e| e.to_string())?;
            
            Ok(())
        } else {
            Err(format!("Symbol {:?} not found", symbol_id))
        }
    }

    /// Check and update margin status
    fn check_margin_status(&self) {
        let exposure = self.get_exposure();
        let max_leverage = self.max_leverage_bps.load(Ordering::Relaxed);
        
        if exposure.is_liquidation_risk(max_leverage) {
            if !self.margin_call_active.load(Ordering::Relaxed) {
                error!("LIQUIDATION RISK: Leverage at {} bps", exposure.leverage_bps);
                self.margin_call_active.store(true, Ordering::Release);
                self.emergency_stop.store(true, Ordering::Release);
            }
        } else if exposure.is_margin_call(max_leverage) {
            if !self.margin_call_active.load(Ordering::Relaxed) {
                warn!("MARGIN CALL: Leverage at {} bps", exposure.leverage_bps);
                self.margin_call_active.store(true, Ordering::Release);
            }
        } else {
            // Normal state
            if self.margin_call_active.load(Ordering::Relaxed) {
                info!("Margin status normalized");
                self.margin_call_active.store(false, Ordering::Release);
                self.emergency_stop.store(false, Ordering::Release);
            }
        }
    }

    /// Add a new position to tracking
    pub fn add_position(
        &self,
        symbol_id: SymbolId,
        size: f64,
        price: f64,
        is_long: bool,
    ) -> Result<(), String> {
        let size_fixed = (size * PNL_PRECISION as f64) as i64;
        let price_fixed = (price * PNL_PRECISION as f64) as i64;
        
        let mut exposures = self.symbol_exposures.write();
        let exposure = exposures.iter_mut().find(|e| e.symbol_id == symbol_id)
            .ok_or_else(|| format!("Symbol {:?} not tracked", symbol_id))?;
        
        if is_long {
            // Update average entry price for longs
            let total_size = exposure.long_size + size_fixed;
            if total_size > 0 {
                exposure.long_avg_price = (
                    (exposure.long_avg_price * exposure.long_size) + 
                    (price_fixed * size_fixed)
                ) / total_size;
            }
            exposure.long_size = total_size;
        } else {
            // Update average entry price for shorts
            let total_size = exposure.short_size + size_fixed;
            if total_size > 0 {
                exposure.short_avg_price = (
                    (exposure.short_avg_price * exposure.short_size) + 
                    (price_fixed * size_fixed)
                ) / total_size;
            }
            exposure.short_size = total_size;
        }
        
        // Update position count
        drop(exposures);
        self.global_exposure.update(|state| {
            let mut new_state = state.clone();
            new_state.position_count += 1;
            new_state
        }).map_err(|e| e.to_string())?;
        
        Ok(())
    }

    /// Close a position
    pub fn close_position(
        &self,
        symbol_id: SymbolId,
        size: f64,
        price: f64,
        is_long: bool,
    ) -> Result<i64, String> {
        let size_fixed = (size * PNL_PRECISION as f64) as i64;
        let price_fixed = (price * PNL_PRECISION as f64) as i64;
        
        let mut exposures = self.symbol_exposures.write();
        let exposure = exposures.iter_mut().find(|e| e.symbol_id == symbol_id)
            .ok_or_else(|| format!("Symbol {:?} not tracked", symbol_id))?;
        
        let pnl = if is_long {
            let close_size = size_fixed.min(exposure.long_size);
            let pnl = ((price_fixed - exposure.long_avg_price) * close_size) / PNL_PRECISION;
            exposure.long_size -= close_size;
            pnl
        } else {
            let close_size = size_fixed.min(exposure.short_size);
            let pnl = ((exposure.short_avg_price - price_fixed) * close_size) / PNL_PRECISION;
            exposure.short_size -= close_size;
            pnl
        };
        
        // Update realized PnL
        drop(exposures);
        self.global_exposure.update(|state| {
            let mut new_state = state.clone();
            new_state.realized_pnl_today += pnl;
            new_state.position_count = new_state.position_count.saturating_sub(1);
            new_state
        }).map_err(|e| e.to_string())?;
        
        Ok(pnl)
    }

    /// Get current update count
    pub fn get_update_count(&self) -> usize {
        self.update_count.load(Ordering::Relaxed)
    }

    /// Get last sync timestamp (nanoseconds)
    pub fn get_last_sync_ns(&self) -> u64 {
        self.last_sync_time.load(Ordering::Relaxed)
    }

    /// Enable emergency stop
    pub fn enable_emergency_stop(&self) {
        self.emergency_stop.store(true, Ordering::Release);
        warn!("Cross-margin sync emergency stop enabled");
    }

    /// Disable emergency stop
    pub fn disable_emergency_stop(&self) {
        self.emergency_stop.store(false, Ordering::Release);
        self.margin_call_active.store(false, Ordering::Release);
    }

    /// Set maximum leverage
    pub fn set_max_leverage(&self, leverage_bps: i64) {
        self.max_leverage_bps.store(leverage_bps, Ordering::Release);
        info!("Max leverage set to {} bps ({}x)", leverage_bps, leverage_bps / 100);
    }

    /// Get all symbol exposures
    pub fn get_all_exposures(&self) -> Vec<SymbolExposure> {
        self.symbol_exposures.read().clone()
    }
}

impl Default for CrossMarginSync {
    fn default() -> Self {
        Self::new().expect("Failed to create default CrossMarginSync")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exposure_state_creation() {
        let state = ExposureState::new(10_000_000_000);
        assert_eq!(state.long_exposure, 0);
        assert_eq!(state.short_exposure, 0);
        assert_eq!(state.available_margin, 10_000_000_000);
    }

    #[test]
    fn test_cross_margin_sync_creation() {
        let sync = CrossMarginSync::new();
        assert!(sync.is_ok());
    }

    #[test]
    fn test_exposure_update() {
        let sync = CrossMarginSync::new().unwrap();
        
        let btc_id = SymbolId::from_symbol("BTCUSDT").unwrap();
        let result = sync.update_exposure_atomic(btc_id, 1000.0);
        assert!(result.is_ok());
        
        let exposure = sync.get_exposure();
        assert!(exposure.long_exposure > 0);
    }

    #[test]
    fn test_leverage_calculation() {
        let state = ExposureState::new(10_000_000_000);
        let equity = 10_000_000_000;
        
        // Test with zero exposure
        assert_eq!(state.calculate_leverage(equity), 0);
    }
}
