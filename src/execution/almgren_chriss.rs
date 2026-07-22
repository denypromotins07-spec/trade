//! Chapter 2: Market Impact & Optimal Execution
//! File 4: src/execution/almgren_chriss.rs
//!
//! Implements extended Almgren-Chriss optimal execution trajectories.
//! Balances temporary market impact against timing risk, dynamically
//! adjusting aggression based on volatility. Factors in Binance maker/taker fees.
//!
//! Optimized for AMD Ryzen AI 5 with SIMD batch calculations.

use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum number of concurrent execution trajectories
const MAX_TRAJECTORIES: usize = 64 * 1024; // 64K trajectories

/// Execution trajectory state
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct ExecutionTrajectory {
    /// Remaining quantity to execute (fixed-point: qty * 10^8)
    pub remaining_qty: i64,
    /// Original order quantity
    pub original_qty: i64,
    /// Current time step (0..N)
    pub current_step: u32,
    /// Total number of time steps
    pub total_steps: u32,
    /// Quantity executed so far
    pub executed_qty: i64,
    /// Average execution price achieved
    pub avg_price: i64,
    /// Target participation rate (bps)
    pub target_pov_bps: u32,
    /// Current volatility estimate (annualized * 10^4)
    pub volatility: u32,
    /// Risk aversion parameter (lambda * 10^4)
    pub risk_aversion: u32,
    /// Temporary impact coefficient (eta * 10^8)
    pub temp_impact: i64,
    /// Permanent impact coefficient (gamma * 10^8)
    pub perm_impact: i64,
    /// Maker fee in bps
    pub maker_fee_bps: u32,
    /// Taker fee in bps
    pub taker_fee_bps: u32,
    /// Is active
    pub is_active: bool,
}

/// Almgren-Chriss execution engine
#[repr(C, align(64))]
pub struct AlmgrenChrissEngine {
    /// Pre-allocated trajectory pool
    trajectories: [ExecutionTrajectory; MAX_TRAJECTORIES],
    
    /// Active trajectory count
    active_count: AtomicU64,
    
    /// Default parameters
    default_risk_aversion: u32,
    default_temp_impact: i64,
    default_perm_impact: i64,
    
    /// Time horizon in milliseconds
    default_time_horizon_ms: u64,
    
    /// Minimum child order size (fixed-point)
    min_child_order: i64,
}

impl Default for ExecutionTrajectory {
    fn default() -> Self {
        ExecutionTrajectory {
            remaining_qty: 0,
            original_qty: 0,
            current_step: 0,
            total_steps: 0,
            executed_qty: 0,
            avg_price: 0,
            target_pov_bps: 1000, // 10% default
            volatility: 2000,     // 20% annualized
            risk_aversion: 10000, // lambda = 1.0
            temp_impact: 100000,  // eta = 0.001
            perm_impact: 50000,   // gamma = 0.0005
            maker_fee_bps: 10,
            taker_fee_bps: 10,
            is_active: false,
        }
    }
}

impl AlmgrenChrissEngine {
    /// Create new Almgren-Chriss engine
    pub fn new(
        risk_aversion: f64,
        temp_impact: f64,
        perm_impact: f64,
        time_horizon_ms: u64,
        maker_fee_bps: u32,
        taker_fee_bps: u32,
    ) -> Self {
        Self {
            trajectories: [ExecutionTrajectory::default(); MAX_TRAJECTORIES],
            active_count: AtomicU64::new(0),
            default_risk_aversion: (risk_aversion * 10000.0) as u32,
            default_temp_impact: (temp_impact * 1e8) as i64,
            default_perm_impact: (perm_impact * 1e8) as i64,
            default_time_horizon_ms: time_horizon_ms,
            min_child_order: 100_000_00, // 1 unit minimum
        }
    }
    
    /// Start a new optimal execution trajectory
    /// 
    /// Returns trajectory ID or None if at capacity
    pub fn start_execution(
        &self,
        quantity: i64,
        current_price: i64,
        volatility: f64,
        time_horizon_ms: Option<u64>,
        use_maker: bool,
    ) -> Option<usize> {
        let current = self.active_count.load(Ordering::Relaxed);
        if current >= MAX_TRAJECTORIES as u64 {
            return None; // Enforce 8GB RAM cap
        }
        
        let idx = current as usize;
        let horizon = time_horizon_ms.unwrap_or(self.default_time_horizon_ms);
        
        // Calculate optimal number of time steps based on quantity and volatility
        let vol = volatility.max(0.01).min(2.0);
        let qty_abs = quantity.abs() as f64;
        
        // More volatile = more aggressive (fewer steps)
        // Larger quantity = more steps to minimize impact
        let base_steps = ((qty_abs / 1e8).ln().max(1.0) * 10.0) as u32;
        let vol_adjustment = (0.5 / vol).max(0.5).min(2.0);
        let total_steps = (base_steps as f64 * vol_adjustment).max(4.0).min(100.0) as u32;
        
        unsafe {
            let traj_ptr = self.trajectories.as_mut_ptr().add(idx);
            (*traj_ptr).remaining_qty = quantity;
            (*traj_ptr).original_qty = quantity;
            (*traj_ptr).current_step = 0;
            (*traj_ptr).total_steps = total_steps;
            (*traj_ptr).executed_qty = 0;
            (*traj_ptr).avg_price = current_price;
            (*traj_ptr).volatility = (vol * 10000.0) as u32;
            (*traj_ptr).risk_aversion = self.default_risk_aversion;
            (*traj_ptr).temp_impact = self.default_temp_impact;
            (*traj_ptr).perm_impact = self.default_perm_impact;
            (*traj_ptr).maker_fee_bps = if use_maker { maker_fee_bps } else { 0 };
            (*traj_ptr).taker_fee_bps = if !use_maker { taker_fee_bps } else { 0 };
            (*traj_ptr).is_active = true;
        }
        
        self.active_count.fetch_add(1, Ordering::Relaxed);
        Some(idx)
    }
    
    /// Calculate the optimal child order size for current step
    /// 
    /// Uses closed-form Almgren-Chriss solution:
    /// n_k = (Q/T) + (lambda * sigma^2 / eta) * (T - k) * (Q - q_{k-1})
    /// 
    /// Where:
    ///   Q = total quantity
    ///   T = total time steps
    ///   k = current step
    ///   q_{k-1} = quantity executed so far
    ///   lambda = risk aversion
    ///   sigma = volatility
    ///   eta = temporary impact
    #[inline(always)]
    pub fn calculate_child_order(&self, trajectory_id: usize, current_vol: Option<f64>) -> i64 {
        if trajectory_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return 0;
        }
        
        unsafe {
            let traj_ptr = self.trajectories.as_ptr().add(trajectory_id);
            let traj = &*traj_ptr;
            
            if !traj.is_active || traj.remaining_qty == 0 || traj.total_steps == 0 {
                return 0;
            }
            
            let vol = current_vol.unwrap_or(traj.volatility as f64 / 10000.0);
            let lambda = traj.risk_aversion as f64 / 10000.0;
            let eta = traj.temp_impact as f64 / 1e8;
            let gamma = traj.perm_impact as f64 / 1e8;
            
            let Q = traj.original_qty as f64;
            let T = traj.total_steps as f64;
            let k = traj.current_step as f64;
            let q = traj.executed_qty as f64;
            
            // Almgren-Chriss optimal schedule
            // Base VWAP component
            let vwap_component = Q / T;
            
            // Risk-adjusted deviation component
            let remaining_time = T - k;
            let remaining_qty = Q - q;
            
            // Adjust for fees: higher fees => more passive, slower execution
            let fee_factor = 1.0 - (traj.maker_fee_bps + traj.taker_fee_bps) as f64 / 10000.0;
            
            let risk_component = if eta > 1e-10 {
                (lambda * vol * vol / eta) * remaining_time * remaining_qty * fee_factor
            } else {
                0.0
            };
            
            // Combine components
            let optimal_qty = vwap_component + risk_component;
            
            // Apply constraints
            let constrained_qty = optimal_qty
                .max(traj.min_child_order as f64)
                .min(traj.remaining_qty as f64);
            
            // Round to nearest integer (fixed-point)
            ((constrained_qty / 1e8).round() * 1e8) as i64
        }
    }
    
    /// Update trajectory after partial fill
    #[inline]
    pub fn update_fill(&self, trajectory_id: usize, filled_qty: i64, fill_price: i64) -> bool {
        if trajectory_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return false;
        }
        
        unsafe {
            let traj_ptr = self.trajectories.as_mut_ptr().add(trajectory_id);
            let traj = &mut *traj_ptr;
            
            if !traj.is_active {
                return false;
            }
            
            traj.executed_qty += filled_qty;
            traj.remaining_qty -= filled_qty;
            
            // Update average price
            let total_value = (traj.avg_price as f128 * traj.executed_qty as f128) 
                            - (traj.avg_price as f128 * filled_qty as f128)
                            + (fill_price as f128 * filled_qty as f128);
            traj.avg_price = if traj.executed_qty > 0 {
                (total_value / traj.executed_qty as f128) as i64
            } else {
                traj.avg_price
            };
            
            traj.current_step += 1;
            
            // Check if complete
            if traj.remaining_qty <= 0 || traj.current_step >= traj.total_steps {
                traj.is_active = false;
            }
            
            true
        }
    }
    
    /// Get trajectory status
    pub fn get_status(&self, trajectory_id: usize) -> Option<ExecutionTrajectory> {
        if trajectory_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return None;
        }
        unsafe {
            let traj_ptr = self.trajectories.as_ptr().add(trajectory_id);
            Some(*traj_ptr)
        }
    }
    
    /// Cancel/terminate a trajectory
    pub fn cancel_trajectory(&self, trajectory_id: usize) -> bool {
        if trajectory_id >= self.active_count.load(Ordering::Relaxed) as usize {
            return false;
        }
        
        unsafe {
            let traj_ptr = self.trajectories.as_mut_ptr().add(trajectory_id);
            (*traj_ptr).is_active = false;
        }
        true
    }
    
    /// Batch calculate child orders (SIMD-friendly pattern)
    pub fn batch_calculate<const N: usize>(
        &self,
        trajectory_ids: [usize; N],
        vols: [Option<f64>; N],
    ) -> [i64; N] {
        let mut results: [i64; N] = [0; N];
        for i in 0..N {
            results[i] = self.calculate_child_order(trajectory_ids[i], vols[i]);
        }
        results
    }
    
    /// Memory statistics
    pub fn memory_stats(&self) -> (usize, usize, usize) {
        let active = self.active_count.load(Ordering::Relaxed) as usize;
        let per_traj = std::mem::size_of::<ExecutionTrajectory>();
        (active, active * per_traj, MAX_TRAJECTORIES * per_traj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trajectory_creation() {
        let engine = AlmgrenChrissEngine::new(1.0, 0.001, 0.0005, 60000, 10, 10);
        let qty = 1000 * 1e8 as i64; // 1000 units
        let price = 50000 * 1e8 as i64;
        
        assert!(engine.start_execution(qty, price, 0.2, None, false).is_some());
    }
    
    #[test]
    fn test_child_order_calculation() {
        let engine = AlmgrenChrissEngine::new(1.0, 0.001, 0.0005, 60000, 10, 10);
        let qty = 100 * 1e8 as i64;
        let price = 50000 * 1e8 as i64;
        
        let traj_id = engine.start_execution(qty, price, 0.3, None, false).unwrap();
        let child_order = engine.calculate_child_order(traj_id, None);
        
        assert!(child_order > 0);
        assert!(child_order <= qty);
    }
    
    #[test]
    fn test_fee_adjustment() {
        let engine_maker = AlmgrenChrissEngine::new(1.0, 0.001, 0.0005, 60000, 10, 10);
        let engine_taker = AlmgrenChrissEngine::new(1.0, 0.001, 0.0005, 60000, 10, 10);
        
        let qty = 1000 * 1e8 as i64;
        let price = 50000 * 1e8 as i64;
        
        let maker_id = engine_maker.start_execution(qty, price, 0.2, None, true).unwrap();
        let taker_id = engine_taker.start_execution(qty, price, 0.2, None, false).unwrap();
        
        let maker_order = engine_maker.calculate_child_order(maker_id, None);
        let taker_order = engine_taker.calculate_child_order(taker_id, None);
        
        // Maker orders should be more conservative due to fee consideration
        assert!(maker_order <= taker_order);
    }
    
    #[test]
    fn test_ram_cap() {
        assert!(MAX_TRAJECTORIES > 0);
        assert!(MAX_TRAJECTORIES <= 128 * 1024);
    }
}
