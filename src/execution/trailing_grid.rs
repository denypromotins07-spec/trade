//! Trailing Grid Trading with Volatility-Adjusted Stops
//!
//! This module combines dynamic grid trading with volatility-adjusted trailing
//! stops, capturing micro-fluctuations in ranging markets while strictly bounding
//! maximum downside exposure. Optimized for AMD Ryzen AI 5 architecture.
//!
//! ## Features
//! - Dynamic grid spacing based on volatility
//! - Trailing stop integration
//! - Maximum drawdown protection
//! - Multi-level grid management
//! - Real-time P&L tracking

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::BTreeMap;

/// Grid order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSide {
    Buy,
    Sell,
}

/// Grid order status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridOrderStatus {
    Pending,
    Active,
    Filled,
    Cancelled,
    Stopped,
}

/// Single grid level
#[derive(Debug, Clone)]
pub struct GridLevel {
    pub level_id: u64,
    pub price: u64,
    pub quantity: u64,
    pub side: GridSide,
    pub status: GridOrderStatus,
    pub fill_price: Option<u64>,
    pub created_at_ns: u64,
    pub filled_at_ns: Option<u64>,
}

/// Trailing stop configuration
#[derive(Debug, Clone)]
pub struct TrailingStopConfig {
    /// Initial trail distance in basis points (100 = 1%)
    pub trail_bps: u64,
    /// Minimum trail distance in ticks
    pub min_trail_ticks: u64,
    /// Activation threshold (profit needed before trailing starts)
    pub activation_threshold_bps: u64,
    /// Maximum allowed loss in basis points
    pub max_loss_bps: u64,
}

impl Default for TrailingStopConfig {
    fn default() -> Self {
        Self {
            trail_bps: 100,      // 1% trail
            min_trail_ticks: 10,
            activation_threshold_bps: 50,  // 0.5% profit before trailing
            max_loss_bps: 200,   // 2% max loss
        }
    }
}

/// Grid configuration
#[derive(Debug, Clone)]
pub struct GridConfig {
    /// Number of grid levels above and below center
    pub num_levels: usize,
    /// Grid spacing in basis points
    pub spacing_bps: u64,
    /// Base quantity per level
    pub base_quantity: u64,
    /// Quantity multiplier per level (for martingale/demartingale)
    pub quantity_multiplier: f64,
    /// Center price for grid
    pub center_price: u64,
    /// Symbol to trade
    pub symbol: String,
    /// Enable trailing stops on filled positions
    pub enable_trailing_stop: bool,
    /// Trailing stop configuration
    pub trailing_stop_config: TrailingStopConfig,
}

impl GridConfig {
    /// Create new grid config with volatility-based spacing
    pub fn with_volatility(
        symbol: &str,
        center_price: u64,
        volatility_bps: u64,
        num_levels: usize,
    ) -> Self {
        // Space grids at 2x volatility for optimal capture
        let spacing_bps = (volatility_bps * 2).max(10);
        
        Self {
            symbol: symbol.to_string(),
            center_price,
            num_levels,
            spacing_bps,
            base_quantity: 100,
            quantity_multiplier: 1.0,
            enable_trailing_stop: true,
            trailing_stop_config: TrailingStopConfig::default(),
        }
    }
}

/// Position with trailing stop
#[derive(Debug, Clone)]
pub struct GridPosition {
    pub symbol: String,
    pub entry_price: u64,
    pub quantity: u64,
    pub side: GridSide,
    pub current_stop_price: u64,
    pub highest_price_since_entry: u64,
    pub lowest_price_since_entry: u64,
    pub is_trailing_active: bool,
    pub unrealized_pnl: i64,
    pub created_at_ns: u64,
}

impl GridPosition {
    /// Create new position
    pub fn new(symbol: &str, entry_price: u64, quantity: u64, side: GridSide) -> Self {
        let now = get_current_time_ns();
        Self {
            symbol: symbol.to_string(),
            entry_price,
            quantity,
            side,
            current_stop_price: entry_price, // Will be set by trailing logic
            highest_price_since_entry: entry_price,
            lowest_price_since_entry: entry_price,
            is_trailing_active: false,
            unrealized_pnl: 0,
            created_at_ns: now,
        }
    }

    /// Update position with new price
    pub fn update_price(&mut self, current_price: u64, config: &TrailingStopConfig) {
        match self.side {
            GridSide::Buy => {
                // Long position - track highest price for trailing stop
                if current_price > self.highest_price_since_entry {
                    self.highest_price_since_entry = current_price;
                    
                    // Calculate new trailing stop
                    let trail_amount = (current_price as u128 * config.trail_bps as u128 / 10000) as u64;
                    let new_stop = current_price.saturating_sub(trail_amount);
                    
                    // Only move stop up
                    if new_stop > self.current_stop_price {
                        self.current_stop_price = new_stop.max(self.entry_price);
                    }
                }
                
                // Check if trailing should activate
                let profit_bps = if self.entry_price > 0 {
                    ((current_price as u128 * 10000 / self.entry_price as u128 - 10000) as i64)
                } else {
                    0
                };
                
                self.is_trailing_active = profit_bps >= config.activation_threshold_bps as i64;
                
                // Calculate P&L
                self.unrealized_pnl = (current_price as i64 - self.entry_price as i64) * self.quantity as i64;
            }
            GridSide::Sell => {
                // Short position - track lowest price for trailing stop
                if current_price < self.lowest_price_since_entry {
                    self.lowest_price_since_entry = current_price;
                    
                    // Calculate new trailing stop
                    let trail_amount = (current_price as u128 * config.trail_bps as u128 / 10000) as u64;
                    let new_stop = current_price.saturating_add(trail_amount);
                    
                    // Only move stop down for shorts
                    if new_stop < self.current_stop_price {
                        self.current_stop_price = new_stop.min(self.entry_price);
                    }
                }
                
                // Check if trailing should activate
                let profit_bps = if self.entry_price > 0 {
                    ((self.entry_price as u128 * 10000 / current_price as u128 - 10000) as i64)
                } else {
                    0
                };
                
                self.is_trailing_active = profit_bps >= config.activation_threshold_bps as i64;
                
                // Calculate P&L
                self.unrealized_pnl = (self.entry_price as i64 - current_price as i64) * self.quantity as i64;
            }
        }
    }

    /// Check if stop loss triggered
    #[inline]
    pub fn is_stop_triggered(&self, current_price: u64) -> bool {
        match self.side {
            GridSide::Buy => current_price <= self.current_stop_price,
            GridSide::Sell => current_price >= self.current_stop_price,
        }
    }

    /// Get max loss in basis points
    pub fn get_max_loss_bps(&self, current_price: u64) -> u64 {
        if self.entry_price == 0 {
            return 0;
        }
        
        let loss = match self.side {
            GridSide::Buy => self.entry_price as i64 - current_price as i64,
            GridSide::Sell => current_price as i64 - self.entry_price as i64,
        };
        
        if loss <= 0 {
            0
        } else {
            ((loss as u128 * 10000 / self.entry_price as u128) as u64)
        }
    }
}

/// Main grid trading engine
pub struct TrailingGridEngine {
    config: GridConfig,
    /// Active grid levels
    grid_levels: parking_lot::RwLock<BTreeMap<u64, GridLevel>>,
    /// Open positions
    positions: parking_lot::RwLock<Vec<GridPosition>>,
    /// Next level ID
    next_level_id: AtomicU64,
    /// Total realized P&L
    realized_pnl: AtomicI64,
    /// Grid active flag
    is_active: AtomicBool,
    /// Statistics
    stats: parking_lot::RwLock<GridStats>,
}

/// Grid statistics
#[derive(Debug, Clone, Default)]
pub struct GridStats {
    pub total_fills: usize,
    pub profitable_fills: usize,
    pub stopped_positions: usize,
    pub total_grid_pnl: i64,
    pub open_positions: usize,
    pub max_drawdown_bps: u64,
}

impl TrailingGridEngine {
    /// Create new grid engine
    pub fn new(config: GridConfig) -> Self {
        let mut engine = Self {
            config,
            grid_levels: parking_lot::RwLock::new(BTreeMap::new()),
            positions: parking_lot::RwLock::new(Vec::new()),
            next_level_id: AtomicU64::new(1),
            realized_pnl: AtomicI64::new(0),
            is_active: AtomicBool::new(false),
            stats: parking_lot::RwLock::new(GridStats::default()),
        };
        
        // Initialize grid levels
        engine.initialize_grid();
        engine
    }

    /// Initialize grid levels around center price
    fn initialize_grid(&mut self) {
        let mut levels = self.grid_levels.write();
        levels.clear();

        let half_levels = self.config.num_levels;
        
        // Create buy levels below center
        for i in 1..=half_levels {
            let discount_bps = i as u64 * self.config.spacing_bps;
            let price = self.config.center_price as u128 
                * (10000 - discount_bps) as u128 
                / 10000;
            let price = price as u64;
            
            let quantity = (self.config.base_quantity as f64 
                * self.config.quantity_multiplier.powi(i as i32)) as u64;
            
            let level = GridLevel {
                level_id: self.next_level_id.fetch_add(1, AtomicOrdering::Relaxed),
                price,
                quantity,
                side: GridSide::Buy,
                status: GridOrderStatus::Pending,
                fill_price: None,
                created_at_ns: get_current_time_ns(),
                filled_at_ns: None,
            };
            
            levels.insert(level.level_id, level);
        }

        // Create sell levels above center
        for i in 1..=half_levels {
            let premium_bps = i as u64 * self.config.spacing_bps;
            let price = self.config.center_price as u128 
                * (10000 + premium_bps) as u128 
                / 10000;
            let price = price as u64;
            
            let quantity = (self.config.base_quantity as f64 
                * self.config.quantity_multiplier.powi(i as i32)) as u64;
            
            let level = GridLevel {
                level_id: self.next_level_id.fetch_add(1, AtomicOrdering::Relaxed),
                price,
                quantity,
                side: GridSide::Sell,
                status: GridOrderStatus::Pending,
                fill_price: None,
                created_at_ns: get_current_time_ns(),
                filled_at_ns: None,
            };
            
            levels.insert(level.level_id, level);
        }
    }

    /// Process price update and check for grid fills
    pub fn process_tick(&self, current_price: u64) -> Vec<GridEvent> {
        let mut events = Vec::new();
        
        // Check grid levels for fills
        {
            let mut levels = self.grid_levels.write();
            
            for (_, level) in levels.iter_mut() {
                if level.status != GridOrderStatus::Active {
                    continue;
                }
                
                let should_fill = match level.side {
                    GridSide::Buy => current_price <= level.price,
                    GridSide::Sell => current_price >= level.price,
                };
                
                if should_fill {
                    level.status = GridOrderStatus::Filled;
                    level.fill_price = Some(current_price);
                    level.filled_at_ns = Some(get_current_time_ns());
                    
                    // Create position with trailing stop
                    if self.config.enable_trailing_stop {
                        let mut position = GridPosition::new(
                            &self.config.symbol,
                            current_price,
                            level.quantity,
                            level.side,
                        );
                        
                        // Set initial stop based on config
                        let trail_amount = (current_price as u128 
                            * self.config.trailing_stop_config.trail_bps as u128 
                            / 10000) as u64;
                        
                        position.current_stop_price = match level.side {
                            GridSide::Buy => current_price.saturating_sub(trail_amount),
                            GridSide::Sell => current_price.saturating_add(trail_amount),
                        };
                        
                        self.positions.write().push(position);
                    }
                    
                    events.push(GridEvent::LevelFilled {
                        level_id: level.level_id,
                        price: current_price,
                        quantity: level.quantity,
                        side: level.side,
                    });
                }
            }
        }
        
        // Update positions and check trailing stops
        {
            let mut positions = self.positions.write();
            let config = self.config.trailing_stop_config.clone();
            
            let mut stopped_indices = Vec::new();
            
            for (i, position) in positions.iter_mut().enumerate() {
                position.update_price(current_price, &config);
                
                if position.is_stop_triggered(current_price) {
                    stopped_indices.push(i);
                    
                    // Record P&L
                    let pnl = position.unrealized_pnl;
                    self.realized_pnl.fetch_add(pnl, AtomicOrdering::Relaxed);
                    
                    events.push(GridEvent::StopTriggered {
                        symbol: position.symbol.clone(),
                        trigger_price: current_price,
                        pnl,
                    });
                }
            }
            
            // Remove stopped positions (reverse order to maintain indices)
            for i in stopped_indices.into_iter().rev() {
                positions.remove(i);
                self.stats.write().stopped_positions += 1;
            }
        }
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.open_positions = self.positions.read().len();
            stats.total_grid_pnl = self.realized_pnl.load(AtomicOrdering::Relaxed);
        }
        
        events
    }

    /// Activate grid (place orders)
    pub fn activate(&self) {
        let mut levels = self.grid_levels.write();
        for (_, level) in levels.iter_mut() {
            if level.status == GridOrderStatus::Pending {
                level.status = GridOrderStatus::Active;
            }
        }
        self.is_active.store(true, AtomicOrdering::Relaxed);
    }

    /// Deactivate grid (cancel orders)
    pub fn deactivate(&self) {
        let mut levels = self.grid_levels.write();
        for (_, level) in levels.iter_mut() {
            if level.status == GridOrderStatus::Active {
                level.status = GridOrderStatus::Cancelled;
            }
        }
        self.is_active.store(false, AtomicOrdering::Relaxed);
    }

    /// Re-center grid on new price
    pub fn recenter(&self, new_center_price: u64) {
        self.config.center_price = new_center_price;
        self.initialize_grid();
    }

    /// Get current grid levels
    pub fn get_levels(&self) -> Vec<GridLevel> {
        self.grid_levels.read().values().cloned().collect()
    }

    /// Get open positions
    pub fn get_positions(&self) -> Vec<GridPosition> {
        self.positions.read().clone()
    }

    /// Get statistics
    pub fn get_stats(&self) -> GridStats {
        self.stats.read().clone()
    }

    /// Get total P&L
    pub fn get_total_pnl(&self) -> i64 {
        self.realized_pnl.load(AtomicOrdering::Relaxed)
    }

    /// Check if any position has hit max loss
    pub fn check_max_violation(&self, current_price: u64) -> bool {
        let positions = self.positions.read();
        let config = &self.config.trailing_stop_config;
        
        for position in positions.iter() {
            if position.get_max_loss_bps(current_price) > config.max_loss_bps {
                return true;
            }
        }
        
        false
    }
}

/// Grid events
#[derive(Debug, Clone)]
pub enum GridEvent {
    LevelFilled {
        level_id: u64,
        price: u64,
        quantity: u64,
        side: GridSide,
    },
    StopTriggered {
        symbol: String,
        trigger_price: u64,
        pnl: i64,
    },
    GridRecentered {
        old_center: u64,
        new_center: u64,
    },
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
    fn test_grid_creation() {
        let config = GridConfig::with_volatility("BTCUSDT", 50000, 50, 5);
        let engine = TrailingGridEngine::new(config);
        
        let levels = engine.get_levels();
        assert_eq!(levels.len(), 10); // 5 buy + 5 sell
        
        let buys: Vec<_> = levels.iter().filter(|l| l.side == GridSide::Buy).collect();
        let sells: Vec<_> = levels.iter().filter(|l| l.side == GridSide::Sell).collect();
        
        assert_eq!(buys.len(), 5);
        assert_eq!(sells.len(), 5);
    }

    #[test]
    fn test_position_trailing_stop() {
        let mut position = GridPosition::new("BTC", 50000, 100, GridSide::Buy);
        let config = TrailingStopConfig::default();
        
        // Price moves up
        position.update_price(52000, &config);
        
        // Stop should have moved up from entry
        assert!(position.current_stop_price > 49000);
        assert!(position.highest_price_since_entry == 52000);
    }

    #[test]
    fn test_stop_trigger_detection() {
        let mut position = GridPosition::new("BTC", 50000, 100, GridSide::Buy);
        position.current_stop_price = 49000;
        
        assert!(!position.is_stop_triggered(49500));
        assert!(position.is_stop_triggered(49000));
        assert!(position.is_stop_triggered(48500));
    }

    #[test]
    fn test_grid_fill_simulation() {
        let config = GridConfig::with_volatility("BTCUSDT", 50000, 50, 3);
        let engine = TrailingGridEngine::new(config);
        engine.activate();
        
        // Simulate price dropping to first buy level
        let events = engine.process_tick(49000);
        
        // Should have at least one fill event
        assert!(!events.is_empty() || engine.get_positions().len() > 0);
    }
}
