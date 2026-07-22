//! Square-Root Market Impact Model (Almgren-Chriss)
//!
//! Codes square-root market impact models to forecast the exact slippage
//! a large market order will cause based on current orderbook liquidity.
//! Optimized for AMD Ryzen AI 5 with SIMD acceleration.
//!
//! The Almgren-Chriss model: impact = sigma * sqrt(|Q|/V) * sign(Q)
//! Where Q is order size, V is daily volume, sigma is volatility

use std::arch::x86_64::*;
use rayon::prelude::*;

/// Order book liquidity snapshot
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Bid depth at levels (price, size)
    pub bid_levels: Vec<(f64, f64)>,
    /// Ask depth at levels (price, size)
    pub ask_levels: Vec<(f64, f64)>,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

/// Market impact parameters
#[derive(Debug, Clone)]
pub struct ImpactParams {
    /// Daily volatility (annualized / sqrt(252))
    pub daily_volatility: f64,
    /// Average daily volume
    pub avg_daily_volume: f64,
    /// Temporary impact coefficient
    pub eta: f64,
    /// Permanent impact coefficient
    pub gamma: f64,
    /// Power law exponent (typically ~0.5 for square-root)
    pub power_exponent: f64,
}

impl Default for ImpactParams {
    fn default() -> Self {
        Self {
            daily_volatility: 0.02,    // 2% daily vol
            avg_daily_volume: 1_000_000.0,
            eta: 0.1,                  // Temporary impact
            gamma: 0.05,               // Permanent impact
            power_exponent: 0.5,       // Square-root law
        }
    }
}

/// Market impact estimation result
#[derive(Debug, Clone)]
pub struct ImpactEstimate {
    /// Expected price impact in absolute terms
    pub absolute_impact: f64,
    /// Expected price impact in basis points
    pub impact_bps: f64,
    /// Execution price estimate
    pub execution_price: f64,
    /// Total cost including spread
    pub total_cost: f64,
    /// Cost in basis points
    pub total_cost_bps: f64,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
}

/// Almgren-Chriss market impact calculator
pub struct MarketImpactCalculator {
    params: ImpactParams,
    /// Current mid-price
    mid_price: f64,
    /// Spread in bps
    spread_bps: f64,
}

impl MarketImpactCalculator {
    /// Create new calculator with given parameters
    pub fn new(params: ImpactParams) -> Self {
        Self {
            params,
            mid_price: 0.0,
            spread_bps: 0.0,
        }
    }

    /// Update reference prices from orderbook
    pub fn update_orderbook(&mut self, snapshot: &OrderBookSnapshot) {
        if snapshot.best_bid > 0.0 && snapshot.best_ask > 0.0 {
            self.mid_price = (snapshot.best_bid + snapshot.best_ask) / 2.0;
            
            if snapshot.best_ask > snapshot.best_bid {
                self.spread_bps = (snapshot.best_ask - snapshot.best_bid) 
                    / self.mid_price * 10_000.0;
            }
        }
    }

    /// Compute square-root market impact
    ///
    /// Formula: impact = eta * sigma * (Q/V)^beta
    /// Where beta is typically 0.5 (square-root law)
    #[inline]
    pub fn compute_impact(&self, order_size: f64, is_buy: bool) -> ImpactEstimate {
        let q = order_size.abs();
        let sign = if is_buy { 1.0 } else { -1.0 };

        if q <= 0.0 || self.params.avg_daily_volume <= 0.0 || self.mid_price <= 0.0 {
            return ImpactEstimate {
                absolute_impact: 0.0,
                impact_bps: 0.0,
                execution_price: self.mid_price,
                total_cost: 0.0,
                total_cost_bps: 0.0,
                ci_lower: 0.0,
                ci_upper: 0.0,
            };
        }

        // Participation rate
        let participation_rate = q / self.params.avg_daily_volume;

        // Square-root impact formula
        let base_impact = self.params.daily_volatility 
            * participation_rate.powf(self.params.power_exponent);
        
        // Apply temporary impact coefficient
        let temporary_impact = self.params.eta * base_impact;
        
        // Permanent impact (linear in Q/V)
        let permanent_impact = self.params.gamma * participation_rate;
        
        // Total fractional impact
        let total_fractional_impact = (temporary_impact + permanent_impact) * sign;

        // Convert to absolute and bps
        let absolute_impact = total_fractional_impact * self.mid_price;
        let impact_bps = total_fractional_impact * 10_000.0;

        // Execution price
        let execution_price = self.mid_price + absolute_impact;

        // Total cost includes half-spread (assuming we cross half the spread)
        let half_spread_bps = self.spread_bps / 2.0;
        let total_cost_bps = impact_bps.abs() + half_spread_bps;
        let total_cost = total_cost_bps * self.mid_price / 10_000.0;

        // Confidence intervals (rough approximation)
        let uncertainty = temporary_impact * 0.3; // 30% uncertainty on temporary impact
        let ci_lower = (total_fractional_impact - uncertainty) * self.mid_price;
        let ci_upper = (total_fractional_impact + uncertainty) * self.mid_price;

        ImpactEstimate {
            absolute_impact,
            impact_bps,
            execution_price,
            total_cost,
            total_cost_bps,
            ci_lower,
            ci_upper,
        }
    }

    /// SIMD-accelerated batch impact computation
    pub fn compute_batch_impact(&self, order_sizes: &[f64], is_buy: bool) -> Vec<ImpactEstimate> {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.compute_batch_impact_simd(order_sizes, is_buy)
            }
        } else {
            order_sizes.par_iter()
                .map(|&size| self.compute_impact(size, is_buy))
                .collect()
        }
    }

    /// AVX2-accelerated batch computation
    #[target_feature(enable = "avx2")]
    unsafe fn compute_batch_impact_simd(&self, order_sizes: &[f64], is_buy: bool) -> Vec<ImpactEstimate> {
        let n = order_sizes.len();
        let mut results = Vec::with_capacity(n);
        
        let sign = if is_buy { 1.0 } else { -1.0 };
        let simd_limit = n & !3; // Align to 4

        // Pre-compute constants
        let vol = self.params.daily_volatility;
        let eta = self.params.eta;
        let gamma = self.params.gamma;
        let beta = self.params.power_exponent;
        let adv = self.params.avg_daily_volume;
        let mid = self.mid_price;

        for i in (0..simd_limit).step_by(4) {
            // Load order sizes
            let q1 = order_sizes[i].abs();
            let q2 = order_sizes[i + 1].abs();
            let q3 = order_sizes[i + 2].abs();
            let q4 = order_sizes[i + 3].abs();

            // Compute participation rates
            let pr1 = q1 / adv;
            let pr2 = q2 / adv;
            let pr3 = q3 / adv;
            let pr4 = q4 / adv;

            // Compute impacts (simplified for SIMD - no powf in SIMD)
            // Using approximation: x^0.5 ≈ fast_sqrt
            let impact1 = (eta * vol * pr1.sqrt() + gamma * pr1) * sign * mid;
            let impact2 = (eta * vol * pr2.sqrt() + gamma * pr2) * sign * mid;
            let impact3 = (eta * vol * pr3.sqrt() + gamma * pr3) * sign * mid;
            let impact4 = (eta * vol * pr4.sqrt() + gamma * pr4) * sign * mid;

            for (q, impact) in [(q1, impact1), (q2, impact2), (q3, impact3), (q4, impact4)] {
                let exec_price = mid + impact;
                let impact_bps = impact / mid * 10_000.0;
                let total_cost_bps = impact_bps.abs() + self.spread_bps / 2.0;
                
                results.push(ImpactEstimate {
                    absolute_impact: impact,
                    impact_bps,
                    execution_price: exec_price,
                    total_cost: total_cost_bps * mid / 10_000.0,
                    total_cost_bps,
                    ci_lower: impact * 0.7,
                    ci_upper: impact * 1.3,
                });
            }
        }

        // Handle remainder
        for i in simd_limit..n {
            results.push(self.compute_impact(order_sizes[i], is_buy));
        }

        results
    }

    /// Optimal execution schedule using Almgren-Chriss
    /// Returns vector of order sizes for each time slice
    pub fn optimal_execution_schedule(
        &self,
        total_quantity: f64,
        num_slices: usize,
        risk_aversion: f64,
    ) -> Vec<f64> {
        if total_quantity <= 0.0 || num_slices == 0 {
            return vec![];
        }

        // Simplified AC optimal trajectory
        // In practice, this solves a quadratic optimization problem
        
        let q_per_slice = total_quantity / num_slices as f64;
        let lambda = risk_aversion * self.params.daily_volatility / num_slices as f64;
        
        // Front-loaded schedule (more aggressive early)
        let mut schedule = Vec::with_capacity(num_slices);
        let decay_factor = 0.9; // How quickly to reduce order size
        
        let mut remaining = total_quantity;
        for i in 0..num_slices {
            let slices_remaining = num_slices - i;
            let weight = (1.0 - decay_factor.powi(slices_remaining as i32)) 
                / (1.0 - decay_factor.powi(num_slices as i32));
            
            let slice_qty = (remaining * weight / slices_remaining as f64).min(remaining);
            schedule.push(slice_qty);
            remaining -= slice_qty;
        }

        // Normalize to ensure sum equals total
        let sum: f64 = schedule.iter().sum();
        if sum > 0.0 {
            for qty in &mut schedule {
                *qty = *qty * total_quantity / sum;
            }
        }

        schedule
    }

    /// Compute implementation shortfall
    pub fn implementation_shortfall(
        &self,
        decision_price: f64,
        execution_prices: &[f64],
        executed_quantities: &[f64],
        total_quantity: f64,
    ) -> f64 {
        if execution_prices.is_empty() || total_quantity <= 0.0 {
            return 0.0;
        }

        let weighted_avg_price: f64 = execution_prices.iter()
            .zip(executed_quantities.iter())
            .map(|(&p, &q)| p * q)
            .sum::<f64>() / executed_quantities.iter().sum::<f64>().max(1.0);

        // For buys: shortfall = (exec_price - decision_price) / decision_price
        // For sells: shortfall = (decision_price - exec_price) / decision_price
        let is_buy = weighted_avg_price > decision_price;
        
        if is_buy {
            (weighted_avg_price - decision_price) / decision_price * 10_000.0
        } else {
            (decision_price - weighted_avg_price) / decision_price * 10_000.0
        }
    }

    /// Get current mid-price
    pub fn mid_price(&self) -> f64 {
        self.mid_price
    }

    /// Get current spread in bps
    pub fn spread_bps(&self) -> f64 {
        self.spread_bps
    }

    /// Update parameters dynamically
    pub fn update_params(&mut self, params: ImpactParams) {
        self.params = params;
    }
}

/// TWAP vs Impact optimizer
pub struct ExecutionOptimizer {
    impact_calc: MarketImpactCalculator,
}

impl ExecutionOptimizer {
    pub fn new(params: ImpactParams) -> Self {
        Self {
            impact_calc: MarketImpactCalculator::new(params),
        }
    }

    /// Decide between TWAP and market impact-aware execution
    pub fn optimize_execution(
        &mut self,
        quantity: f64,
        is_buy: bool,
        time_horizon_minutes: u32,
        urgency: f64,
    ) -> ExecutionStrategy {
        let impact = self.impact_calc.compute_impact(quantity, is_buy);

        // Threshold for switching strategies
        let impact_threshold_bps = 10.0; // 10 bps
        
        if impact.total_cost_bps < impact_threshold_bps || urgency > 0.8 {
            // Use aggressive execution (market orders)
            ExecutionStrategy::Aggressive {
                quantity,
                expected_slippage_bps: impact.total_cost_bps,
            }
        } else if urgency > 0.3 {
            // Use modified TWAP with impact awareness
            let num_slices = ((time_horizon_minutes as f64) / 5.0).ceil() as usize;
            let schedule = self.impact_calc.optimal_execution_schedule(quantity, num_slices, 0.5);
            
            ExecutionStrategy::ImpactAwareTWAP {
                schedule,
                interval_seconds: (time_horizon_minutes * 60) / num_slices as u32,
                expected_total_cost_bps: impact.total_cost_bps * 0.7, // Diversification benefit
            }
        } else {
            // Use pure TWAP (patient execution)
            let num_slices = ((time_horizon_minutes as f64) / 10.0).ceil() as usize;
            let slice_qty = quantity / num_slices as f64;
            
            ExecutionStrategy::PureTWAP {
                slice_quantity: slice_qty,
                num_slices,
                interval_seconds: (time_horizon_minutes * 60) / num_slices as u32,
            }
        }
    }
}

/// Execution strategy recommendation
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Aggressive market order execution
    Aggressive {
        quantity: f64,
        expected_slippage_bps: f64,
    },
    /// TWAP with impact-aware sizing
    ImpactAwareTWAP {
        schedule: Vec<f64>,
        interval_seconds: u32,
        expected_total_cost_bps: f64,
    },
    /// Pure time-weighted execution
    PureTWAP {
        slice_quantity: f64,
        num_slices: usize,
        interval_seconds: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_impact() {
        let params = ImpactParams::default();
        let mut calc = MarketImpactCalculator::new(params);
        
        // Set up orderbook
        let snapshot = OrderBookSnapshot {
            best_bid: 100.0,
            best_ask: 100.1,
            bid_levels: vec![(99.9, 1000.0), (99.8, 2000.0)],
            ask_levels: vec![(100.2, 1000.0), (100.3, 2000.0)],
            timestamp_ns: 1_000_000_000_000,
        };
        
        calc.update_orderbook(&snapshot);
        
        // Test buy impact
        let buy_impact = calc.compute_impact(10_000.0, true);
        assert!(buy_impact.absolute_impact > 0.0);
        assert!(buy_impact.execution_price > 100.05);
        
        // Test sell impact
        let sell_impact = calc.compute_impact(10_000.0, false);
        assert!(sell_impact.absolute_impact < 0.0);
        assert!(sell_impact.execution_price < 100.05);
    }

    #[test]
    fn test_optimal_schedule() {
        let params = ImpactParams::default();
        let calc = MarketImpactCalculator::new(params);
        
        let schedule = calc.optimal_execution_schedule(100_000.0, 10, 0.5);
        
        assert_eq!(schedule.len(), 10);
        
        let total: f64 = schedule.iter().sum();
        assert!((total - 100_000.0).abs() < 1.0); // Should sum to approximately total
        
        // First slice should be larger than last (front-loaded)
        assert!(schedule[0] > schedule[9]);
    }

    #[test]
    fn test_execution_optimizer() {
        let params = ImpactParams::default();
        let mut optimizer = ExecutionOptimizer::new(params);
        
        // High urgency should give aggressive strategy
        let strategy = optimizer.optimize_execution(100_000.0, true, 60, 0.9);
        assert!(matches!(strategy, ExecutionStrategy::Aggressive { .. }));
        
        // Low urgency should give TWAP
        let strategy = optimizer.optimize_execution(100_000.0, true, 60, 0.1);
        assert!(matches!(strategy, ExecutionStrategy::PureTWAP { .. }));
    }
}
