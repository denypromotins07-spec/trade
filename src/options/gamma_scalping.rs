//! Gamma Scalping Engine
//! 
//! Implements an automated delta-neutral gamma scalping engine that dynamically
//! hedges options portfolios using perpetual futures to capture realized volatility
//! versus implied volatility.
//! 
//! Accounts for Binance-specific funding rates and trading fees.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Option position in the portfolio
#[derive(Debug, Clone)]
pub struct OptionPosition {
    pub symbol: String,
    pub strike: f64,
    pub expiry_days: u32,
    pub quantity: f64,      // Number of contracts (positive = long)
    pub option_type: OptionType,
    pub entry_vol: f64,     // Implied vol at entry
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

/// Perpetual futures hedge position
#[derive(Debug, Clone)]
pub struct HedgePosition {
    pub symbol: String,
    pub quantity: f64,      // Positive = long, negative = short
    pub entry_price: f64,
    pub unrealized_pnl: f64,
}

/// Market data snapshot
#[derive(Debug, Clone)]
pub struct MarketData {
    pub spot_price: f64,
    pub funding_rate: f64,  // 8-hour funding rate (e.g., 0.0001 = 0.01%)
    pub bid_ask_spread: f64,
    pub timestamp_ns: u64,
}

/// Greeks for an option
#[derive(Debug, Clone, Default)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Trading fee structure (Binance-specific)
pub struct FeeStructure {
    pub maker_fee: f64,   // e.g., 0.0002 = 0.02%
    pub taker_fee: f64,   // e.g., 0.0004 = 0.04%
    pub min_fee: f64,     // Minimum fee in quote currency
}

impl Default for FeeStructure {
    fn default() -> Self {
        Self {
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            min_fee: 0.0001,
        }
    }
}

/// Gamma scalping signal
#[derive(Debug, Clone)]
pub struct ScalpSignal {
    pub symbol: String,
    pub action: HedgeAction,
    pub quantity: f64,
    pub target_delta: f64,
    pub current_delta: f64,
    pub estimated_cost: f64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HedgeAction {
    BuyFuture,
    SellFuture,
    CloseHedge,
    Rebalance,
}

/// Main gamma scalping engine
pub struct GammaScalpingEngine {
    positions: Vec<OptionPosition>,
    hedge_positions: Vec<HedgePosition>,
    fee_structure: FeeStructure,
    is_active: AtomicBool,
    last_rebalance_ts: AtomicU64,
    rebalance_threshold_delta: f64,
    total_fees_paid: f64,
    total_funding_paid: f64,
}

impl GammaScalpingEngine {
    pub fn new(fee_structure: FeeStructure) -> Self {
        Self {
            positions: Vec::new(),
            hedge_positions: Vec::new(),
            fee_structure,
            is_active: AtomicBool::new(false),
            last_rebalance_ts: AtomicU64::new(0),
            rebalance_threshold_delta: 0.05, // Rebalance when delta exceeds 5% of notional
            total_fees_paid: 0.0,
            total_funding_paid: 0.0,
        }
    }

    /// Add an option position to the portfolio
    pub fn add_option_position(&mut self, position: OptionPosition) {
        self.positions.push(position);
    }

    /// Remove an option position
    pub fn remove_option_position(&mut self, symbol: &str, strike: f64, option_type: OptionType) {
        self.positions.retain(|p| {
            !(p.symbol == symbol && p.strike == strike && p.option_type == option_type)
        });
    }

    /// Calculate Black-Scholes Greeks for an option
    #[inline(always)]
    pub fn calculate_greeks(
        &self,
        spot: f64,
        strike: f64,
        vol: f64,
        days_to_expiry: u32,
        option_type: OptionType,
        risk_free_rate: f64,
    ) -> Greeks {
        let t = days_to_expiry as f64 / 365.0;
        
        if t <= 0.0 || vol <= 0.0 || spot <= 0.0 || strike <= 0.0 {
            return Greeks::default();
        }

        let d1 = (spot / strike).ln() + (risk_free_rate + 0.5 * vol * vol) * t;
        let d1 = d1 / (vol * t.sqrt());
        let d2 = d1 - vol * t.sqrt();

        // Standard normal PDF
        let npdf = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();

        // Delta
        let delta = match option_type {
            OptionType::Call => self.norm_cdf(d1),
            OptionType::Put => self.norm_cdf(d1) - 1.0,
        };

        // Gamma (same for call and put)
        let gamma = npdf / (spot * vol * t.sqrt());

        // Theta (per day)
        let term1 = -spot * npdf * vol / (2.0 * t.sqrt());
        let term2 = match option_type {
            OptionType::Call => risk_free_rate * strike * (-risk_free_rate * t).exp() * self.norm_cdf(d2),
            OptionType::Put => -risk_free_rate * strike * (-risk_free_rate * t).exp() * self.norm_cdf(-d2),
        };
        let theta = (term1 + term2) / 365.0;

        // Vega (per 1% vol change)
        let vega = spot * npdf * t.sqrt() / 100.0;

        // Rho (per 1% rate change)
        let rho = match option_type {
            OptionType::Call => strike * t * (-risk_free_rate * t).exp() * self.norm_cdf(d2) / 100.0,
            OptionType::Put => -strike * t * (-risk_free_rate * t).exp() * self.norm_cdf(-d2) / 100.0,
        };

        Greeks { delta, gamma, theta, vega, rho }
    }

    /// Calculate portfolio-wide delta
    pub fn get_portfolio_delta(&self, market: &MarketData, risk_free_rate: f64) -> f64 {
        let mut total_delta = 0.0;

        for position in &self.positions {
            let greeks = self.calculate_greeks(
                market.spot_price,
                position.strike,
                position.entry_vol,
                position.expiry_days,
                position.option_type,
                risk_free_rate,
            );

            // Delta contribution = delta * quantity * contract_multiplier
            // Assuming 1 contract = 1 unit of underlying for simplicity
            total_delta += greeks.delta * position.quantity;
        }

        // Add delta from hedge positions (futures have delta = 1)
        for hedge in &self.hedge_positions {
            total_delta += hedge.quantity;
        }

        total_delta
    }

    /// Calculate portfolio gamma
    pub fn get_portfolio_gamma(&self, market: &MarketData, risk_free_rate: f64) -> f64 {
        let mut total_gamma = 0.0;

        for position in &self.positions {
            let greeks = self.calculate_greeks(
                market.spot_price,
                position.strike,
                position.entry_vol,
                position.expiry_days,
                position.option_type,
                risk_free_rate,
            );

            total_gamma += greeks.gamma * position.quantity;
        }

        total_gamma
    }

    /// Generate delta-neutral hedge signal
    pub fn generate_hedge_signal(&self, market: &MarketData, risk_free_rate: f64) -> Option<ScalpSignal> {
        let portfolio_delta = self.get_portfolio_delta(market, risk_free_rate);
        
        if portfolio_delta.abs() < self.rebalance_threshold_delta {
            return None;
        }

        // Determine hedge action
        let (action, quantity) = if portfolio_delta > 0.0 {
            // Long delta: need to short futures to neutralize
            (HedgeAction::SellFuture, -portfolio_delta)
        } else {
            // Short delta: need to long futures to neutralize
            (HedgeAction::BuyFuture, -portfolio_delta)
        };

        // Estimate transaction cost
        let notional = quantity.abs() * market.spot_price;
        let estimated_fee = (notional * self.fee_structure.taker_fee).max(self.fee_structure.min_fee);

        Some(ScalpSignal {
            symbol: "BTCUSDT".to_string(), // Default symbol
            action,
            quantity,
            target_delta: 0.0,
            current_delta: portfolio_delta,
            estimated_cost: estimated_fee,
            timestamp_ns: self.get_timestamp_ns(),
        })
    }

    /// Execute a gamma scalp: buy low, sell high as underlying moves
    pub fn execute_gamma_scalp(
        &mut self,
        market: &MarketData,
        price_move_pct: f64,
        risk_free_rate: f64,
    ) -> Option<f64> {
        let gamma = self.get_portfolio_gamma(market, risk_free_rate);
        
        if gamma <= 0.0 {
            // Need positive gamma to profit from scalping
            return None;
        }

        // Gamma scalping P&L approximation:
        // P&L ≈ 0.5 * gamma * (price_move)^2 - theta * time_decay
        let spot_move = market.spot_price * price_move_pct;
        let gamma_pnl = 0.5 * gamma * spot_move * spot_move;

        // Subtract transaction costs for rebalancing
        let rebalance_cost = self.estimate_rebalance_cost(market, gamma, price_move_pct);
        
        let net_pnl = gamma_pnl - rebalance_cost;

        if net_pnl > 0.0 {
            self.total_fees_paid += rebalance_cost;
            Some(net_pnl)
        } else {
            None
        }
    }

    /// Estimate cost of rebalancing hedge
    fn estimate_rebalance_cost(
        &self,
        market: &MarketData,
        gamma: f64,
        price_move_pct: f64,
    ) -> f64 {
        // Approximate hedge adjustment needed
        let delta_adjustment = gamma * market.spot_price * price_move_pct;
        let notional = delta_adjustment.abs() * market.spot_price;
        
        (notional * self.fee_structure.taker_fee).max(self.fee_structure.min_fee)
    }

    /// Calculate funding cost for perpetual futures hedge
    pub fn calculate_funding_cost(&self, hedge_positions: &[HedgePosition], funding_rate: f64) -> f64 {
        let mut total_funding = 0.0;

        for position in hedge_positions {
            // Funding payment = position_size * price * funding_rate
            // Paid every 8 hours on Binance
            let notional = position.quantity.abs() * position.entry_price;
            let funding_payment = notional * funding_rate;

            // Long positions pay funding when rate is positive
            // Short positions receive funding when rate is positive
            if position.quantity > 0.0 {
                total_funding += funding_payment; // Long pays
            } else {
                total_funding -= funding_payment; // Short receives
            }
        }

        total_funding
    }

    /// Check if gamma scalping is profitable given realized vs implied vol
    pub fn check_scalp_profitability(
        &self,
        realized_vol: f64,
        implied_vol: f64,
        gamma: f64,
        theta: f64,
    ) -> bool {
        // Gamma scalping is profitable when:
        // Realized vol > Implied vol (captured gamma > paid theta)
        // 
        // Simplified condition:
        // 0.5 * gamma * S^2 * (realized_vol^2 - implied_vol^2) > theta
        
        if gamma <= 0.0 {
            return false;
        }

        let vol_diff = realized_vol * realized_vol - implied_vol * implied_vol;
        let gamma_capture = 0.5 * gamma * vol_diff;

        gamma_capture > theta.abs()
    }

    /// Update hedge position after execution
    pub fn update_hedge_position(&mut self, symbol: &str, quantity: f64, entry_price: f64) {
        // Check if position exists
        let existing = self.hedge_positions.iter_mut().find(|p| p.symbol == symbol);

        if let Some(pos) = existing {
            pos.quantity += quantity;
            // Update average entry price
            if pos.quantity != 0.0 {
                pos.entry_price = ((pos.entry_price * (pos.quantity - quantity)).abs() + entry_price * quantity.abs()) / pos.quantity.abs();
            }
        } else {
            self.hedge_positions.push(HedgePosition {
                symbol: symbol.to_string(),
                quantity,
                entry_price,
                unrealized_pnl: 0.0,
            });
        }
    }

    /// Get current timestamp in nanoseconds
    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Approximate standard normal CDF
    #[inline(always)]
    fn norm_cdf(&self, x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        0.5 * (1.0 + sign * y)
    }

    /// Start the gamma scalping engine
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Release);
    }

    /// Stop the gamma scalping engine
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Release);
    }

    /// Check if engine is active
    pub fn is_running(&self) -> bool {
        self.is_active.load(Ordering::Acquire)
    }

    /// Get total fees paid
    pub fn get_total_fees_paid(&self) -> f64 {
        self.total_fees_paid
    }

    /// Get total funding paid
    pub fn get_total_funding_paid(&self) -> f64 {
        self.total_funding_paid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_calculation() {
        let engine = GammaScalpingEngine::new(FeeStructure::default());

        let greeks = engine.calculate_greeks(
            100.0,  // spot
            100.0,  // strike
            0.65,   // vol (65%)
            30,     // days to expiry
            OptionType::Call,
            0.05,   // risk-free rate
        );

        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.theta < 0.0); // Long options decay
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_delta_neutral_hedge() {
        let mut engine = GammaScalpingEngine::new(FeeStructure::default());

        // Add a long call position
        engine.add_option_position(OptionPosition {
            symbol: "BTC".to_string(),
            strike: 100.0,
            expiry_days: 30,
            quantity: 10.0,
            option_type: OptionType::Call,
            entry_vol: 0.65,
        });

        let market = MarketData {
            spot_price: 100.0,
            funding_rate: 0.0001,
            bid_ask_spread: 0.01,
            timestamp_ns: 0,
        };

        let delta = engine.get_portfolio_delta(&market, 0.05);
        
        // Long call should have positive delta
        assert!(delta > 0.0);

        // Generate hedge signal
        let signal = engine.generate_hedge_signal(&market, 0.05);
        
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().action, HedgeAction::SellFuture);
    }

    #[test]
    fn test_funding_cost_calculation() {
        let engine = GammaScalpingEngine::new(FeeStructure::default());

        let hedges = vec![
            HedgePosition {
                symbol: "BTCUSDT".to_string(),
                quantity: 1.0,  // Long 1 BTC
                entry_price: 50000.0,
                unrealized_pnl: 0.0,
            },
            HedgePosition {
                symbol: "ETHUSDT".to_string(),
                quantity: -10.0,  // Short 10 ETH
                entry_price: 3000.0,
                unrealized_pnl: 0.0,
            },
        ];

        let funding_rate = 0.0001; // 0.01% per 8 hours
        let cost = engine.calculate_funding_cost(&hedges, funding_rate);

        // Long BTC pays: 1 * 50000 * 0.0001 = 5
        // Short ETH receives: 10 * 3000 * 0.0001 = 3
        // Net: 5 - 3 = 2
        assert!((cost - 2.0).abs() < 0.01);
    }
}
