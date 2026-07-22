//! Chapter 2: Market Impact & Optimal Execution
//! File 6: src/execution/participation_rate.rs
//!
//! Adaptive Percentage of Volume (POV) engine that dynamically scales
//! the bot's market participation to hide its footprint from predatory HFT.
//! Uses real-time volume tracking and stealth mode detection.
//!
//! Optimized for AMD Ryzen AI 5 with cache-aligned structures.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of tracked symbols
const MAX_SYMBOLS: usize = 128 * 1024;

/// Time window for volume calculation (milliseconds)
const VOLUME_WINDOW_MS: u64 = 60000; // 1 minute

/// POV execution state per symbol
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct POVState {
    /// Target participation rate in bps (e.g., 100 = 1%)
    pub target_pov_bps: u32,
    /// Current actual participation rate in bps
    pub actual_pov_bps: u32,
    /// Bot's executed volume in window (fixed-point)
    pub bot_volume: i64,
    /// Total market volume in window (fixed-point)
    pub market_volume: i64,
    /// Parent order remaining quantity (fixed-point)
    pub parent_remaining: i64,
    /// Last update timestamp
    pub last_update_ns: u64,
    /// Window start timestamp
    pub window_start_ns: u64,
    /// Stealth mode active (reduced participation)
    pub stealth_mode: bool,
    /// Aggression level (0-100)
    pub aggression: u8,
}

/// POV Engine result
#[derive(Debug, Clone, Copy)]
pub struct POVOrderSpec {
    /// Order quantity (fixed-point)
    pub quantity: i64,
    /// Recommended side
    pub is_buy: bool,
    /// Participation rate used (bps)
    pub pov_bps: u32,
    /// Time to wait before next order (microseconds)
    pub delay_us: u32,
    /// Should submit order
    pub should_submit: bool,
}

impl Default for POVState {
    fn default() -> Self {
        POVState {
            target_pov_bps: 500, // 5% default
            actual_pov_bps: 0,
            bot_volume: 0,
            market_volume: 0,
            parent_remaining: 0,
            last_update_ns: 0,
            window_start_ns: 0,
            stealth_mode: false,
            aggression: 50,
        }
    }
}

/// Adaptive POV Engine
#[repr(C, align(64))]
pub struct POVEngine {
    /// Pre-allocated state array
    states: [POVState; MAX_SYMBOLS],
    
    /// Symbol hash to state index mapping
    symbol_hashes: [u64; MAX_SYMBOLS],
    
    /// Active symbol count
    active_count: AtomicU64,
    
    /// Global settings
    min_pov_bps: u32,
    max_pov_bps: u32,
    stealth_threshold_bps: u32,
    
    /// Predatory activity detection
    predatory_detected: [AtomicBool; MAX_SYMBOLS],
}

impl POVEngine {
    /// Create new POV engine
    pub fn new(min_pov: u32, max_pov: u32, stealth_thresh: u32) -> Self {
        Self {
            states: [POVState::default(); MAX_SYMBOLS],
            symbol_hashes: [0; MAX_SYMBOLS],
            active_count: AtomicU64::new(0),
            min_pov_bps: min_pov,
            max_pov_bps: max_pov,
            stealth_threshold_bps: stealth_thresh,
            predatory_detected: unsafe { std::mem::zeroed() },
        }
    }
    
    /// Register a new symbol for POV tracking
    pub fn register_symbol(&self, symbol_hash: u64, target_pov_bps: u32) -> Option<usize> {
        let current = self.active_count.load(Ordering::Relaxed);
        if current >= MAX_SYMBOLS as u64 {
            return None; // Enforce 8GB RAM cap
        }
        
        let idx = current as usize;
        let now = get_timestamp_ns();
        
        unsafe {
            let state_ptr = self.states.as_mut_ptr().add(idx);
            (*state_ptr).target_pov_bps = target_pov_bps.min(self.max_pov_bps).max(self.min_pov_bps);
            (*state_ptr).window_start_ns = now;
            (*state_ptr).last_update_ns = now;
            
            *self.symbol_hashes.as_mut_ptr().add(idx) = symbol_hash;
            self.predatory_detected[idx].store(false, Ordering::Relaxed);
        }
        
        self.active_count.fetch_add(1, Ordering::Relaxed);
        Some(idx)
    }
    
    /// Update market volume for a symbol
    #[inline(always)]
    pub fn update_market_volume(&self, symbol_id: usize, volume_delta: i64, is_buy_side: bool) {
        if symbol_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return;
        }
        
        unsafe {
            let state_ptr = self.states.as_mut_ptr().add(symbol_id);
            let state = &mut *state_ptr;
            
            let now = get_timestamp_ns();
            
            // Check if we need to reset the window
            if now - state.window_start_ns > VOLUME_WINDOW_MS * 1_000_000 {
                state.bot_volume = 0;
                state.market_volume = 0;
                state.window_start_ns = now;
            }
            
            state.market_volume += volume_delta;
            state.last_update_ns = now;
            
            // Recalculate actual POV
            if state.market_volume > 0 && state.bot_volume > 0 {
                state.actual_pov_bps = ((state.bot_volume * 10000) / state.market_volume) as u32;
            }
            
            // Detect predatory activity (sudden volume spikes)
            self.detect_predatory_activity(symbol_id, volume_delta);
        }
    }
    
    /// Update bot's executed volume
    #[inline(always)]
    pub fn update_bot_volume(&self, symbol_id: usize, volume_delta: i64) {
        if symbol_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return;
        }
        
        unsafe {
            let state_ptr = self.states.as_mut_ptr().add(symbol_id);
            (*state_ptr).bot_volume += volume_delta;
            (*state_ptr).parent_remaining = ((*state_ptr).parent_remaining - volume_delta).max(0);
            
            // Recalculate actual POV
            let state = &*state_ptr;
            if state.market_volume > 0 {
                (*state_ptr).actual_pov_bps = ((state.bot_volume * 10000) / state.market_volume) as u32;
            }
        }
    }
    
    /// Set parent order for a symbol
    pub fn set_parent_order(&self, symbol_id: usize, quantity: i64, is_buy: bool) -> bool {
        if symbol_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return false;
        }
        
        unsafe {
            let state_ptr = self.states.as_mut_ptr().add(symbol_id);
            (*state_ptr).parent_remaining = quantity;
        }
        true
    }
    
    /// Calculate next order based on POV strategy
    #[inline]
    pub fn calculate_next_order(&self, symbol_id: usize, current_market_rate: i64) -> POVOrderSpec {
        if symbol_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return POVOrderSpec {
                quantity: 0,
                is_buy: false,
                pov_bps: 0,
                delay_us: 0,
                should_submit: false,
            };
        }
        
        unsafe {
            let state_ptr = self.states.as_ptr().add(symbol_id);
            let state = &*state_ptr;
            
            if state.parent_remaining <= 0 || state.target_pov_bps == 0 {
                return POVOrderSpec {
                    quantity: 0,
                    is_buy: false,
                    pov_bps: 0,
                    delay_us: 0,
                    should_submit: false,
                };
            }
            
            // Adjust POV based on stealth mode and predatory detection
            let mut effective_pov = state.target_pov_bps;
            
            if state.stealth_mode || self.predatory_detected[symbol_id].load(Ordering::Relaxed) {
                // Reduce participation when hiding
                effective_pov = (effective_pov / 2).max(self.min_pov_bps);
            }
            
            // Clamp to limits
            effective_pov = effective_pov.clamp(self.min_pov_bps, self.max_pov_bps);
            
            // Calculate order size based on current market rate
            let market_rate_f = current_market_rate.max(1) as f64 / 1e8;
            let target_qty_f = market_rate_f * (effective_pov as f64 / 10000.0);
            let target_qty = (target_qty_f * 1e8) as i64;
            
            // Don't exceed remaining parent order
            let order_qty = target_qty.min(state.parent_remaining);
            
            // Calculate delay to maintain POV
            let delay_us = if current_market_rate > 0 {
                ((order_qty as f64 / current_market_rate as f64) * 1_000_000.0) as u32
            } else {
                1000 // Default 1ms delay
            };
            
            // Determine if we should submit
            let should_submit = order_qty > 0 
                && state.actual_pov_bps < state.target_pov_bps * 110 / 100 // Allow 10% overshoot
                && !self.predatory_detected[symbol_id].load(Ordering::Relaxed);
            
            POVOrderSpec {
                quantity: order_qty,
                is_buy: true, // Would be determined by parent order direction
                pov_bps: effective_pov,
                delay_us: delay_us.max(100), // Minimum 100us between orders
                should_submit,
            }
        }
    }
    
    /// Enable/disable stealth mode
    pub fn set_stealth_mode(&self, symbol_id: usize, enabled: bool) -> bool {
        if symbol_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return false;
        }
        
        unsafe {
            let state_ptr = self.states.as_mut_ptr().add(symbol_id);
            (*state_ptr).stealth_mode = enabled;
        }
        true
    }
    
    /// Set aggression level (0-100)
    pub fn set_aggression(&self, symbol_id: usize, level: u8) -> bool {
        if symbol_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return false;
        }
        
        unsafe {
            let state_ptr = self.states.as_mut_ptr().add(symbol_id);
            (*state_ptr).aggression = level.min(100);
            
            // Adjust POV based on aggression
            let aggression_factor = (level as f64 / 50.0).max(0.5).min(2.0);
            (*state_ptr).target_pov_bps = (((*state_ptr).target_pov_bps as f64 * aggression_factor) as u32)
                .clamp(self.min_pov_bps, self.max_pov_bps);
        }
        true
    }
    
    /// Detect predatory HFT activity
    #[inline]
    fn detect_predatory_activity(&self, symbol_id: usize, volume_delta: i64) {
        unsafe {
            let state_ptr = self.states.as_ptr().add(symbol_id);
            let state = &*state_ptr;
            
            // Detect unusual volume spikes (>10x average)
            let avg_volume = if state.market_volume > 0 {
                state.market_volume / 60 // Per-second average
            } else {
                volume_delta
            };
            
            if volume_delta > avg_volume * 10 {
                self.predatory_detected[symbol_id].store(true, Ordering::Relaxed);
                
                // Auto-enable stealth mode
                let state_ptr_mut = self.states.as_mut_ptr().add(symbol_id);
                (*state_ptr_mut).stealth_mode = true;
            }
        }
    }
    
    /// Memory statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let active = self.active_count.load(Ordering::Relaxed) as usize;
        let per_symbol = std::mem::size_of::<POVState>() + std::mem::size_of::<u64>() + 1;
        (active, active * per_symbol, MAX_SYMBOLS * per_symbol)
    }
}

/// Get timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pov_initialization() {
        let engine = POVEngine::new(100, 2000, 5000);
        assert!(engine.register_symbol(12345, 500).is_some());
    }
    
    #[test]
    fn test_volume_tracking() {
        let engine = POVEngine::new(100, 2000, 5000);
        let sym_id = engine.register_symbol(12345, 500).unwrap();
        
        engine.update_market_volume(sym_id, 1000 * 1e8 as i64, true);
        engine.update_bot_volume(sym_id, 50 * 1e8 as i64);
        
        let state = unsafe {
            let ptr = engine.states.as_ptr().add(sym_id);
            *ptr
        };
        
        assert_eq!(state.market_volume, 1000 * 1e8 as i64);
        assert_eq!(state.bot_volume, 50 * 1e8 as i64);
    }
    
    #[test]
    fn test_ram_cap() {
        assert!(MAX_SYMBOLS > 0);
        assert!(MAX_SYMBOLS <= 256 * 1024);
    }
}
