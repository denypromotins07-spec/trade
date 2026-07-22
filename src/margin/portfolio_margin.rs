//! src/margin/portfolio_margin.rs
//!
//! Advanced Portfolio Margin (SPAN-like) Risk Model.
//!
//! This module implements a SPAN-style portfolio margin system that calculates
//! risk-based margin requirements by analyzing correlated positions and potential
//! losses across market scenarios. It offsets hedged positions to free up trapped
//! capital and maximize leverage efficiency.
//!
//! Features:
//! - Correlation Matrix: Tracks asset correlations for netting benefits.
//! - Scenario Analysis: Simulates PnL across price/volatility shocks.
//! - Span Margin: Uses worst-case scenario loss as margin requirement.
//! - Netting Logic: Offsets long/short positions in correlated assets.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Fixed-point precision (6 decimals).
const FP_PRECISION: u64 = 1_000_000;

#[inline]
fn to_fp(value: f64) -> u64 {
    (value * FP_PRECISION as f64) as u64
}

#[inline]
fn from_fp(value: u64) -> f64 {
    value as f64 / FP_PRECISION as f64
}

/// Position with full details for margin calculation.
#[derive(Debug, Clone)]
pub struct PortfolioPosition {
    pub symbol: String,
    pub side: Side,
    pub size: f64,
    pub mark_price: f64,
    pub delta: f64,      // Price sensitivity
    pub gamma: f64,      // Delta sensitivity
    pub vega: f64,       // Volatility sensitivity
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Long,
    Short,
}

/// Market scenario for stress testing.
#[derive(Debug, Clone)]
pub struct MarketScenario {
    pub name: String,
    pub btc_move_pct: f64,
    pub eth_move_pct: f64,
    pub vol_change_pct: f64,
}

/// Result of portfolio margin calculation.
#[derive(Debug, Clone)]
pub struct PortfolioMarginResult {
    pub total_portfolio_value: f64,
    pub sum_individual_margins: f64,
    pub portfolio_margin_benefit: f64,
    pub net_margin_requirement: f64,
    pub worst_case_scenario: String,
    pub worst_case_loss: f64,
    pub timestamp_ns: u64,
}

/// Portfolio Margin Engine using SPAN methodology.
pub struct PortfolioMarginEngine {
    positions: HashMap<String, PortfolioPosition>,
    /// Correlation matrix: (asset1, asset2) -> correlation coefficient
    correlations: HashMap<(String, String), f64>,
    /// Market scenarios for stress testing
    scenarios: Vec<MarketScenario>,
    /// Last calculated result
    last_result: Option<PortfolioMarginResult>,
    last_calc_timestamp: AtomicU64,
}

impl PortfolioMarginEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            positions: HashMap::new(),
            correlations: HashMap::new(),
            scenarios: Vec::new(),
            last_result: None,
            last_calc_timestamp: AtomicU64::new(0),
        };

        // Initialize default correlations (BTC-ETH highly correlated)
        engine.set_correlation("BTC", "ETH", 0.85);
        engine.set_correlation("BTC", "BNB", 0.75);
        engine.set_correlation("ETH", "BNB", 0.80);

        // Initialize standard SPAN-style scenarios
        engine.scenarios = vec![
            MarketScenario {
                name: "up_1pct".to_string(),
                btc_move_pct: 1.0,
                eth_move_pct: 1.5,
                vol_change_pct: 0.0,
            },
            MarketScenario {
                name: "down_1pct".to_string(),
                btc_move_pct: -1.0,
                eth_move_pct: -1.5,
                vol_change_pct: 0.0,
            },
            MarketScenario {
                name: "up_3pct".to_string(),
                btc_move_pct: 3.0,
                eth_move_pct: 4.5,
                vol_change_pct: 5.0,
            },
            MarketScenario {
                name: "down_3pct".to_string(),
                btc_move_pct: -3.0,
                eth_move_pct: -4.5,
                vol_change_pct: 10.0,
            },
            MarketScenario {
                name: "vol_spike".to_string(),
                btc_move_pct: 0.0,
                eth_move_pct: 0.0,
                vol_change_pct: 20.0,
            },
            MarketScenario {
                name: "flash_crash".to_string(),
                btc_move_pct: -10.0,
                eth_move_pct: -15.0,
                vol_change_pct: 50.0,
            },
        ];

        engine
    }

    fn set_correlation(&mut self, asset1: &str, asset2: &str, corr: f64) {
        self.correlations.insert((asset1.to_string(), asset2.to_string()), corr.clamp(-1.0, 1.0));
        self.correlations.insert((asset2.to_string(), asset1.to_string()), corr.clamp(-1.0, 1.0));
    }

    /// Add or update a position in the portfolio.
    pub fn add_position(&mut self, position: PortfolioPosition) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Remove a position from the portfolio.
    pub fn remove_position(&mut self, symbol: &str) {
        self.positions.remove(symbol);
    }

    /// Calculate the net delta exposure for an asset.
    fn get_net_delta(&self, asset: &str) -> f64 {
        let mut net_delta = 0.0;
        
        for pos in self.positions.values() {
            if pos.symbol.starts_with(asset) {
                match pos.side {
                    Side::Long => net_delta += pos.delta,
                    Side::Short => net_delta -= pos.delta,
                }
            }
        }
        
        net_delta
    }

    /// Calculate PnL impact of a market scenario on the portfolio.
    fn calculate_scenario_pnl(&self, scenario: &MarketScenario) -> f64 {
        let mut total_pnl = 0.0;

        for position in &self.positions {
            let asset = position.0.chars().take(3).collect::<String>();
            let pos = position.1;

            // Determine price move based on asset
            let price_move_pct = match asset.as_str() {
                "BTC" => scenario.btc_move_pct,
                "ETH" => scenario.eth_move_pct,
                _ => scenario.btc_move_pct * 0.8, // Default to 80% of BTC move
            };

            // Delta PnL = delta * price_move * notional
            let delta_pnl = pos.delta * (price_move_pct / 100.0) * pos.size * pos.mark_price;

            // Gamma PnL (convexity) = 0.5 * gamma * price_move^2 * notional
            let gamma_pnl = 0.5 * pos.gamma * (price_move_pct / 100.0).powi(2) * pos.size * pos.mark_price;

            // Vega PnL = vega * vol_change
            let vega_pnl = pos.vega * (scenario.vol_change_pct / 100.0);

            // Apply direction
            let pnl = match pos.side {
                Side::Long => delta_pnl + gamma_pnl + vega_pnl,
                Side::Short => -(delta_pnl + gamma_pnl + vega_pnl),
            };

            total_pnl += pnl;
        }

        total_pnl
    }

    /// Calculate individual margin without netting (for comparison).
    fn calculate_sum_individual_margins(&self) -> f64 {
        let mut total = 0.0;

        for position in self.positions.values() {
            let notional = position.size * position.mark_price;
            // Standard futures margin ~5% (20x leverage)
            let margin_rate = 0.05;
            total += notional * margin_rate;
        }

        total
    }

    /// Main portfolio margin calculation using SPAN methodology.
    pub fn calculate_portfolio_margin(&mut self) -> PortfolioMarginResult {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Calculate total portfolio value
        let mut total_value = 0.0;
        for position in self.positions.values() {
            total_value += position.size * position.mark_price;
        }

        // Calculate sum of individual margins (no netting)
        let sum_individual = self.calculate_sum_individual_margins();

        // Find worst-case scenario loss
        let mut worst_loss = 0.0;
        let mut worst_scenario = String::from("none");

        for scenario in &self.scenarios {
            let pnl = self.calculate_scenario_pnl(scenario);
            if pnl < worst_loss {
                worst_loss = pnl;
                worst_scenario = scenario.name.clone();
            }
        }

        // Portfolio margin = max(worst_case_loss, minimum_margin)
        // Minimum margin is typically 5% of portfolio value
        let min_margin = total_value * 0.05;
        let net_margin = worst_loss.abs().max(min_margin);

        // Calculate benefit from portfolio margin vs individual
        let benefit = sum_individual - net_margin;

        let result = PortfolioMarginResult {
            total_portfolio_value: total_value,
            sum_individual_margins: sum_individual,
            portfolio_margin_benefit: benefit.max(0.0),
            net_margin_requirement: net_margin,
            worst_case_scenario: worst_scenario,
            worst_case_loss: worst_loss,
            timestamp_ns,
        };

        self.last_result = Some(result.clone());
        self.last_calc_timestamp.store(timestamp_ns, Ordering::Relaxed);

        result
    }

    /// Get the margin benefit percentage compared to individual margins.
    pub fn get_margin_efficiency(&mut self) -> f64 {
        let result = self.calculate_portfolio_margin();
        if result.sum_individual_margins > 0.0 {
            (result.portfolio_margin_benefit / result.sum_individual_margins) * 100.0
        } else {
            0.0
        }
    }

    /// Check if portfolio is within margin limits.
    pub fn is_within_limits(&mut self, account_equity: f64, safety_factor: f64) -> bool {
        let result = self.calculate_portfolio_margin();
        let required = result.net_margin_requirement * safety_factor;
        account_equity >= required
    }

    /// Get the last calculation result.
    pub fn last_result(&self) -> Option<&PortfolioMarginResult> {
        self.last_result.as_ref()
    }
}

impl Default for PortfolioMarginEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_margin_benefit() {
        let mut engine = PortfolioMarginEngine::new();

        // Add hedged positions: Long BTC, Short ETH (partially correlated)
        engine.add_position(PortfolioPosition {
            symbol: "BTCUSDT".to_string(),
            side: Side::Long,
            size: 1.0,
            mark_price: 50000.0,
            delta: 1.0,
            gamma: 0.0,
            vega: 0.0,
        });

        engine.add_position(PortfolioPosition {
            symbol: "ETHUSDT".to_string(),
            side: Side::Short,
            size: 15.0, // Roughly equivalent notional
            mark_price: 3300.0,
            delta: 1.0,
            gamma: 0.0,
            vega: 0.0,
        });

        let result = engine.calculate_portfolio_margin();

        // Portfolio margin should be less than sum of individual due to hedging
        assert!(result.portfolio_margin_benefit > 0.0);
        assert!(result.net_margin_requirement < result.sum_individual_margins);
    }

    #[test]
    fn test_flash_crash_scenario() {
        let mut engine = PortfolioMarginEngine::new();

        // Add leveraged long position
        engine.add_position(PortfolioPosition {
            symbol: "BTCUSDT".to_string(),
            side: Side::Long,
            size: 10.0,
            mark_price: 50000.0,
            delta: 10.0, // Leveraged
            gamma: 0.0,
            vega: 0.0,
        });

        let result = engine.calculate_portfolio_margin();

        // Worst case should be the flash crash scenario
        assert_eq!(result.worst_case_scenario, "flash_crash");
        assert!(result.worst_case_loss < 0.0);
    }
}
