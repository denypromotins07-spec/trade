//! Avellaneda-Stoikov Stochastic Control Model for Market Making
//! 
//! This module implements the classic Avellaneda-Stoikov model for optimal
//! market making, deriving reservation prices and optimal spreads using
//! volatility and risk aversion parameters.
//! 
//! The model operates purely in the Rust hot path for instant quote updates.
//! 
//! Mathematical foundation:
//! - Reservation price: r = s - q * γ * σ² * (T - t)
//! - Optimal spread: δ* = 1/γ + γ * σ² * (T - t) / 2
//! 
//! Where:
//! - s = mid price
//! - q = inventory position
//! - γ = risk aversion parameter
//! - σ = volatility
//! - T - t = time horizon remaining
//! 
//! Optimized for:
//! - Microsecond latency execution
//! - AMD Ryzen AI 5 architecture
//! - Zero heap allocation in hot path

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Default time horizon in seconds (e.g., 5 minutes)
const DEFAULT_TIME_HORIZON_SECS: f64 = 300.0;

/// Minimum spread to prevent negative values (basis points)
const MIN_SPREAD_BPS: f64 = 1.0;

/// Maximum spread to prevent excessive widening (basis points)
const MAX_SPREAD_BPS: f64 = 1000.0;

/// Avellaneda-Stoikov model parameters
#[derive(Debug, Clone)]
pub struct ASModelParams {
    /// Risk aversion coefficient (gamma)
    /// Higher values = more conservative quoting
    pub risk_aversion: f64,
    /// Volatility estimate (annualized, as decimal)
    pub volatility: f64,
    /// Time horizon for the model (seconds)
    pub time_horizon_secs: f64,
    /// Order book depth parameter (kappa)
    pub order_book_depth: f64,
    /// Minimum spread (basis points)
    pub min_spread_bps: f64,
    /// Maximum spread (basis points)
    pub max_spread_bps: f64,
}

impl Default for ASModelParams {
    fn default() -> Self {
        Self {
            risk_aversion: 0.1,           // Moderate risk aversion
            volatility: 0.02,             // 2% daily volatility
            time_horizon_secs: DEFAULT_TIME_HORIZON_SECS,
            order_book_depth: 100.0,      // Typical depth parameter
            min_spread_bps: MIN_SPREAD_BPS,
            max_spread_bps: MAX_SPREAD_BPS,
        }
    }
}

impl ASModelParams {
    /// Create new parameters with validation
    pub fn new(
        risk_aversion: f64,
        volatility: f64,
        time_horizon_secs: f64,
    ) -> Result<Self, &'static str> {
        if risk_aversion < 0.0 {
            return Err("Risk aversion must be non-negative");
        }
        if volatility < 0.0 {
            return Err("Volatility must be non-negative");
        }
        if time_horizon_secs <= 0.0 {
            return Err("Time horizon must be positive");
        }

        Ok(Self {
            risk_aversion,
            volatility,
            time_horizon_secs,
            ..Default::default()
        })
    }

    /// Update volatility dynamically
    #[inline]
    pub fn set_volatility(&mut self, vol: f64) {
        self.volatility = vol.max(0.0);
    }

    /// Update risk aversion dynamically
    #[inline]
    pub fn set_risk_aversion(&mut self, gamma: f64) {
        self.risk_aversion = gamma.max(0.0);
    }
}

/// Result of Avellaneda-Stoikov calculation
#[derive(Debug, Clone)]
pub struct ASQuote {
    /// Reservation price (fair value adjusted for inventory)
    pub reservation_price: f64,
    /// Optimal bid price
    pub bid_price: f64,
    /// Optimal ask price
    pub ask_price: f64,
    /// Optimal half-spread (distance from reservation price)
    pub optimal_half_spread: f64,
    /// Full spread in basis points
    pub spread_bps: f64,
    /// Timestamp of calculation (nanoseconds)
    pub timestamp_ns: u64,
}

impl ASQuote {
    /// Check if quotes are valid (non-negative, reasonable)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.bid_price > 0.0
            && self.ask_price > 0.0
            && self.ask_price > self.bid_price
            && self.spread_bps > 0.0
            && self.spread_bps < MAX_SPREAD_BPS * 2.0
    }
}

/// Avellaneda-Stoikov stochastic control model
pub struct AvellanedaStoikovModel {
    /// Model parameters
    params: ASModelParams,
    /// Current inventory position
    inventory: i64,
    /// Start time for time horizon calculation
    start_time_ns: AtomicU64,
    /// Last calculation timestamp
    last_calc_ns: AtomicU64,
    /// Cache for expensive calculations
    cached_gamma_sigma_sq: f64,
}

impl AvellanedaStoikovModel {
    /// Create new model with given parameters
    pub fn new(params: ASModelParams) -> Self {
        let start_time_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Pre-calculate gamma * sigma^2 for efficiency
        let cached_gamma_sigma_sq = params.risk_aversion * params.volatility.powi(2);

        Self {
            params,
            inventory: 0,
            start_time_ns: AtomicU64::new(start_time_ns),
            last_calc_ns: AtomicU64::new(0),
            cached_gamma_sigma_sq,
        }
    }

    /// Update model parameters
    pub fn update_params(&mut self, params: ASModelParams) {
        self.params = params;
        self.cached_gamma_sigma_sq = self.params.risk_aversion * self.params.volatility.powi(2);
    }

    /// Update volatility only (common operation)
    #[inline(always)]
    pub fn update_volatility(&mut self, volatility: f64) {
        self.params.volatility = volatility.max(0.0);
        self.cached_gamma_sigma_sq = self.params.risk_aversion * self.params.volatility.powi(2);
    }

    /// Update risk aversion only
    #[inline(always)]
    pub fn update_risk_aversion(&mut self, risk_aversion: f64) {
        self.params.risk_aversion = risk_aversion.max(0.0);
        self.cached_gamma_sigma_sq = self.params.risk_aversion * self.params.volatility.powi(2);
    }

    /// Set current inventory position
    #[inline]
    pub fn set_inventory(&mut self, inventory: i64) {
        self.inventory = inventory;
    }

    /// Get current inventory
    #[inline]
    pub fn get_inventory(&self) -> i64 {
        self.inventory
    }

    /// Calculate time remaining in horizon (seconds)
    #[inline]
    fn time_remaining(&self) -> f64 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let elapsed_secs = (now_ns - self.start_time_ns.load(Ordering::Relaxed)) as f64 / 1e9;
        (self.params.time_horizon_secs - elapsed_secs).max(0.0)
    }

    /// Reset time horizon (call when model should restart)
    pub fn reset_horizon(&self) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.start_time_ns.store(now_ns, Ordering::Release);
    }

    /// Calculate reservation price using Avellaneda-Stoikov formula
    /// r = s - q * γ * σ² * (T - t)
    #[inline(always)]
    pub fn calculate_reservation_price(&self, mid_price: f64) -> f64 {
        let time_remaining = self.time_remaining();
        
        // Inventory adjustment term
        let inventory_adjustment = self.inventory as f64 
            * self.cached_gamma_sigma_sq 
            * time_remaining;
        
        mid_price - inventory_adjustment
    }

    /// Calculate optimal half-spread using Avellaneda-Stoikov formula
    /// δ* = 1/γ + γ * σ² * (T - t) / 2
    #[inline(always)]
    pub fn calculate_optimal_half_spread(&self) -> f64 {
        let time_remaining = self.time_remaining();
        
        // Base spread from order book dynamics
        let base_spread = 1.0 / self.params.order_book_depth;
        
        // Risk adjustment
        let risk_adjustment = self.cached_gamma_sigma_sq * time_remaining / 2.0;
        
        let half_spread = (base_spread + risk_adjustment) * mid_price_from_spread(base_spread + risk_adjustment);
        
        // Convert to price units (assuming mid_price context)
        half_spread
    }

    /// Calculate full optimal spread in basis points
    #[inline(always)]
    pub fn calculate_optimal_spread_bps(&self) -> f64 {
        let time_remaining = self.time_remaining();
        
        // δ* = 1/γ + γ * σ² * (T - t)
        let gamma_inv = if self.params.risk_aversion > 0.0 {
            1.0 / self.params.risk_aversion
        } else {
            0.0
        };
        
        let risk_term = self.cached_gamma_sigma_sq * time_remaining;
        
        let spread_decimal = gamma_inv + risk_term;
        
        // Convert to basis points and clamp
        let spread_bps = (spread_decimal * 10000.0)
            .clamp(self.params.min_spread_bps, self.params.max_spread_bps);
        
        spread_bps
    }

    /// Generate complete quote using Avellaneda-Stoikov model
    /// This is the main hot-path function optimized for microsecond latency
    #[inline(always)]
    pub fn generate_quote(&self, mid_price: f64) -> ASQuote {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Calculate reservation price
        let reservation_price = self.calculate_reservation_price(mid_price);
        
        // Calculate optimal spread
        let spread_bps = self.calculate_optimal_spread_bps();
        let spread_decimal = spread_bps / 10000.0;
        let half_spread = (mid_price * spread_decimal) / 2.0;
        
        // Derive bid and ask from reservation price
        let bid_price = reservation_price - half_spread;
        let ask_price = reservation_price + half_spread;
        
        // Ensure prices are positive and ordered correctly
        let bid_price = bid_price.max(mid_price * 0.0001); // Prevent zero/negative
        let ask_price = ask_price.max(bid_price * 1.0001); // Ensure ask > bid
        
        self.last_calc_ns.store(now_ns, Ordering::Relaxed);

        ASQuote {
            reservation_price,
            bid_price,
            ask_price,
            optimal_half_spread: half_spread,
            spread_bps,
            timestamp_ns: now_ns,
        }
    }

    /// Generate quote with custom inventory override
    #[inline(always)]
    pub fn generate_quote_with_inventory(
        &self,
        mid_price: f64,
        inventory: i64,
    ) -> ASQuote {
        // Temporarily adjust inventory for calculation
        let original_inventory = self.inventory;
        
        // Inline calculation with custom inventory
        let time_remaining = self.time_remaining();
        
        let inventory_adjustment = inventory as f64 
            * self.cached_gamma_sigma_sq 
            * time_remaining;
        
        let reservation_price = mid_price - inventory_adjustment;
        
        let spread_bps = self.calculate_optimal_spread_bps();
        let spread_decimal = spread_bps / 10000.0;
        let half_spread = (mid_price * spread_decimal) / 2.0;
        
        let bid_price = (reservation_price - half_spread).max(mid_price * 0.0001);
        let ask_price = (reservation_price + half_spread).max(bid_price * 1.0001);
        
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        ASQuote {
            reservation_price,
            bid_price,
            ask_price,
            optimal_half_spread: half_spread,
            spread_bps,
            timestamp_ns: now_ns,
        }
    }

    /// Get model statistics
    pub fn get_stats(&self) -> ASModelStats {
        ASModelStats {
            risk_aversion: self.params.risk_aversion,
            volatility: self.params.volatility,
            time_horizon_secs: self.params.time_horizon_secs,
            time_remaining_secs: self.time_remaining(),
            inventory: self.inventory,
            gamma_sigma_sq: self.cached_gamma_sigma_sq,
        }
    }

    /// Get reference to parameters
    pub fn params(&self) -> &ASModelParams {
        &self.params
    }
}

/// Model statistics snapshot
#[derive(Debug, Clone)]
pub struct ASModelStats {
    pub risk_aversion: f64,
    pub volatility: f64,
    pub time_horizon_secs: f64,
    pub time_remaining_secs: f64,
    pub inventory: i64,
    pub gamma_sigma_sq: f64,
}

/// Helper function for spread calculation
#[inline]
fn mid_price_from_spread(spread_decimal: f64) -> f64 {
    // Placeholder for mid-price context
    1.0
}

/// Extended model with order arrival rate estimation
pub struct ExtendedASModel {
    base_model: AvellanedaStoikovModel,
    /// Estimated order arrival rate (lambda)
    arrival_rate: f64,
    /// Price impact coefficient
    price_impact: f64,
}

impl ExtendedASModel {
    /// Create extended model
    pub fn new(params: ASModelParams, arrival_rate: f64, price_impact: f64) -> Self {
        Self {
            base_model: AvellanedaStoikovModel::new(params),
            arrival_rate: arrival_rate.max(0.0),
            price_impact: price_impact.max(0.0),
        }
    }

    /// Update arrival rate estimate
    pub fn update_arrival_rate(&mut self, rate: f64) {
        self.arrival_rate = rate.max(0.0);
    }

    /// Calculate optimal quote considering order arrival
    pub fn generate_extended_quote(&self, mid_price: f64) -> ASQuote {
        // Start with base AS quote
        let base_quote = self.base_model.generate_quote(mid_price);
        
        // Adjust for arrival rate (higher arrival = can afford wider spread)
        if self.arrival_rate > 0.0 {
            let arrival_adjustment = 1.0 + (1.0 / (1.0 + self.arrival_rate));
            let adjusted_half_spread = base_quote.optimal_half_spread * arrival_adjustment;
            
            let bid_price = (base_quote.reservation_price - adjusted_half_spread)
                .max(mid_price * 0.0001);
            let ask_price = (base_quote.reservation_price + adjusted_half_spread)
                .max(bid_price * 1.0001);
            
            return ASQuote {
                reservation_price: base_quote.reservation_price,
                bid_price,
                ask_price,
                optimal_half_spread: adjusted_half_spread,
                spread_bps: base_quote.spread_bps * arrival_adjustment,
                timestamp_ns: base_quote.timestamp_ns,
            };
        }
        
        base_quote
    }

    /// Get reference to base model
    pub fn base_model(&self) -> &AvellanedaStoikovModel {
        &self.base_model
    }

    /// Get mutable reference to base model
    pub fn base_model_mut(&mut self) -> &mut AvellanedaStoikovModel {
        &mut self.base_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservation_price_with_inventory() {
        let params = ASModelParams {
            risk_aversion: 0.5,
            volatility: 0.02,
            time_horizon_secs: 300.0,
            ..Default::default()
        };
        
        let mut model = AvellanedaStoikovModel::new(params);
        
        // Zero inventory -> reservation price equals mid
        let quote_zero = model.generate_quote(100.0);
        assert!((quote_zero.reservation_price - 100.0).abs() < 0.01);
        
        // Long inventory -> lower reservation price
        model.set_inventory(100);
        let quote_long = model.generate_quote(100.0);
        assert!(quote_long.reservation_price < 100.0);
        
        // Short inventory -> higher reservation price
        model.set_inventory(-100);
        let quote_short = model.generate_quote(100.0);
        assert!(quote_short.reservation_price > 100.0);
    }

    #[test]
    fn test_spread_increases_with_volatility() {
        let mut params = ASModelParams::default();
        params.volatility = 0.01;
        let mut model = AvellanedaStoikovModel::new(params);
        
        let spread_low_vol = model.calculate_optimal_spread_bps();
        
        params.volatility = 0.05;
        model.update_params(params);
        let spread_high_vol = model.calculate_optimal_spread_bps();
        
        assert!(spread_high_vol > spread_low_vol);
    }

    #[test]
    fn test_spread_increases_with_risk_aversion() {
        let mut params = ASModelParams::default();
        params.risk_aversion = 0.1;
        let mut model = AvellanedaStoikovModel::new(params);
        
        let spread_low_ra = model.calculate_optimal_spread_bps();
        
        params.risk_aversion = 0.5;
        model.update_params(params);
        let spread_high_ra = model.calculate_optimal_spread_bps();
        
        assert!(spread_high_ra > spread_low_ra);
    }

    #[test]
    fn test_quote_validity() {
        let model = AvellanedaStoikovModel::new(ASModelParams::default());
        let quote = model.generate_quote(50000.0);
        
        assert!(quote.is_valid());
        assert!(quote.bid_price > 0.0);
        assert!(quote.ask_price > 0.0);
        assert!(quote.ask_price > quote.bid_price);
    }
}
