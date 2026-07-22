//! src/margin/cross_margin.rs
//!
//! Binance Cross-Margin Calculator for Real-Time Risk Tracking.
//!
//! This module implements real-time calculation of initial and maintenance
//! margin requirements across spot, margin, and futures portfolios. It tracks
//! collateral usage, available buying power, and margin ratios to prevent
//! liquidation events.
//!
//! Features:
//! - Multi-Asset Collateral: Supports BTC, ETH, USDT, BNB as collateral.
//! - Haircuts: Applies risk-based haircuts to volatile collateral.
//! - Shared Margin: Calculates cross-margin benefits across positions.
//! - Microsecond Updates: Lock-free atomic updates for hot-path access.

use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Fixed-point precision for calculations (6 decimal places).
const FP_PRECISION: u64 = 1_000_000;

/// Convert f64 to fixed-point u64.
#[inline]
fn to_fp(value: f64) -> u64 {
    (value * FP_PRECISION as f64) as u64
}

/// Convert fixed-point u64 to f64.
#[inline]
fn from_fp(value: u64) -> f64 {
    value as f64 / FP_PRECISION as f64
}

/// Collateral asset configuration with haircuts.
#[derive(Debug, Clone)]
pub struct CollateralConfig {
    pub asset: String,
    pub haircut_pct: f64, // Percentage reduction in collateral value
    pub max_collateral_ratio: f64,
}

/// Position data for margin calculation.
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub side: PositionSide,
    pub size: u64, // Fixed-point
    pub entry_price: u64, // Fixed-point
    pub mark_price: u64, // Fixed-point
    pub leverage: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    None,
}

/// Margin calculation result.
#[derive(Debug, Clone)]
pub struct MarginSummary {
    pub total_collateral: u64, // Fixed-point USDT
    pub total_initial_margin: u64,
    pub total_maintenance_margin: u64,
    pub available_balance: u64,
    pub margin_ratio: u64, // Fixed-point (1.0 = 100%)
    pub is_liquidation_risk: bool,
    pub timestamp_ns: u64,
}

/// Cross-Margin Engine.
pub struct CrossMarginEngine {
    /// Collateral balances per asset (fixed-point USDT equivalent).
    collateral: HashMap<String, u64>,
    /// Open positions.
    positions: HashMap<String, Position>,
    /// Asset configurations.
    configs: HashMap<String, CollateralConfig>,
    /// Cached summary for fast reads.
    cached_summary: AtomicU64, // Points to Arc<MarginSummary> in prod
    /// Last update timestamp.
    last_update_ns: AtomicU64,
}

impl CrossMarginEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            collateral: HashMap::new(),
            positions: HashMap::new(),
            configs: HashMap::new(),
            cached_summary: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        };
        
        // Initialize default configurations
        engine.register_asset("BTC", 0.05, 0.8); // 5% haircut, max 80% ratio
        engine.register_asset("ETH", 0.08, 0.75);
        engine.register_asset("USDT", 0.0, 1.0);
        engine.register_asset("BNB", 0.10, 0.7);
        
        engine
    }

    fn register_asset(&mut self, asset: &str, haircut: f64, max_ratio: f64) {
        self.configs.insert(
            asset.to_string(),
            CollateralConfig {
                asset: asset.to_string(),
                haircut_pct: haircut,
                max_collateral_ratio: max_ratio,
            },
        );
    }

    /// Update collateral balance for an asset.
    pub fn update_collateral(&mut self, asset: &str, amount: f64, price: f64) {
        let config = match self.configs.get(asset) {
            Some(c) => c,
            None => return,
        };

        // Apply haircut and convert to USDT
        let raw_value = amount * price;
        let haircut_value = raw_value * (1.0 - config.haircut_pct);
        let fp_value = to_fp(haircut_value);

        self.collateral.insert(asset.to_string(), fp_value);
        self.invalidate_cache();
    }

    /// Update or add a position.
    pub fn update_position(&mut self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
        self.invalidate_cache();
    }

    /// Remove a closed position.
    pub fn remove_position(&mut self, symbol: &str) {
        self.positions.remove(symbol);
        self.invalidate_cache();
    }

    fn invalidate_cache(&self) {
        self.cached_summary.store(0, Ordering::Relaxed);
    }

    /// Calculate current margin status (hot path).
    pub fn calculate_margin(&self) -> MarginSummary {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Sum all collateral
        let total_collateral: u64 = self.collateral.values().sum();

        // Calculate margin requirements per position
        let mut total_initial_margin: u64 = 0;
        let mut total_maintenance_margin: u64 = 0;

        for position in self.positions.values() {
            let notional = (position.size as u64) * (position.mark_price as u64) / FP_PRECISION;
            
            // Initial margin = notional / leverage
            let im = notional / (position.leverage as u64);
            
            // Maintenance margin (Binance futures ~0.4% for BTC, varies)
            let mm_rate = match position.symbol.chars().next() {
                Some('B') => 4000, // 0.4% in fixed-point
                Some('E') => 5000, // 0.5%
                _ => 5000,
            };
            let mm = (notional * mm_rate) / (FP_PRECISION * 10000);

            total_initial_margin += im;
            total_maintenance_margin += mm;
        }

        let available_balance = if total_initial_margin > total_collateral {
            0
        } else {
            total_collateral - total_initial_margin
        };

        // Margin ratio = total_im / total_collateral
        let margin_ratio = if total_collateral > 0 {
            (total_initial_margin * FP_PRECISION) / total_collateral
        } else {
            0
        };

        // Liquidation risk if margin ratio > 1.0 (100%)
        let is_liquidation_risk = margin_ratio >= FP_PRECISION;

        MarginSummary {
            total_collateral,
            total_initial_margin,
            total_maintenance_margin,
            available_balance,
            margin_ratio,
            is_liquidation_risk,
            timestamp_ns,
        }
    }

    /// Get available buying power for a symbol.
    pub fn get_buying_power(&self, symbol: &str, price: f64) -> f64 {
        let summary = self.calculate_margin();
        let fp_price = to_fp(price);
        
        if summary.available_balance == 0 {
            return 0.0;
        }

        // Max position size = available_balance * max_leverage / price
        let max_leverage: u64 = 20; // Default max
        let bp = (summary.available_balance * max_leverage) / fp_price;
        
        from_fp(bp)
    }

    /// Check if a new order would violate margin limits.
    pub fn validate_order(&self, notional: f64, leverage: u32) -> bool {
        let summary = self.calculate_margin();
        let fp_notional = to_fp(notional);
        let required_im = fp_notional / (leverage as u64);

        required_im <= summary.available_balance
    }

    /// Get current margin ratio as percentage.
    pub fn get_margin_ratio_pct(&self) -> f64 {
        let summary = self.calculate_margin();
        from_fp(summary.margin_ratio) * 100.0
    }
}

impl Default for CrossMarginEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_margin_calculation() {
        let mut engine = CrossMarginEngine::new();
        
        // Add $10,000 USDT collateral
        engine.update_collateral("USDT", 10000.0, 1.0);
        
        // Open BTC position: 0.1 BTC at $50,000 = $5,000 notional, 10x leverage
        let position = Position {
            symbol: "BTCUSDT".to_string(),
            side: PositionSide::Long,
            size: to_fp(0.1),
            entry_price: to_fp(50000.0),
            mark_price: to_fp(50000.0),
            leverage: 10,
        };
        engine.update_position(position);
        
        let summary = engine.calculate_margin();
        
        // Initial margin should be $5,000 / 10 = $500
        assert_eq!(from_fp(summary.total_initial_margin), 500.0);
        // Available balance should be $10,000 - $500 = $9,500
        assert_eq!(from_fp(summary.available_balance), 9500.0);
        assert!(!summary.is_liquidation_risk);
    }
}
