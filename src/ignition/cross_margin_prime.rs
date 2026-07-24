//! =============================================================================
//! cross_margin_prime.rs - Unified Cross-Margin Pool Initialization
//! Nautilus/Ray Trading Bot - Stage 60
//! =============================================================================
//! Purpose: Primes the unified cross-margin pool with initial Binance REST snapshots.
//!          Strictly validates account equity before allowing WS matching engines to fire.
//! Constraints: Ensures no trading begins until margin health is verified.
//! =============================================================================

use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Represents the account balance snapshot from Binance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub total_wallet_balance: f64,
    pub available_balance: f64,
    pub total_unrealized_pnl: f64,
    pub margin_ratio: f64,
    pub timestamp: u64,
}

/// Configuration for margin validation thresholds
pub struct MarginConfig {
    /// Minimum required equity in USD to start trading
    pub min_equity_usd: f64,
    /// Maximum allowed margin ratio (e.g., 0.5 for 50%)
    pub max_margin_ratio: f64,
    /// Timeout for REST snapshot fetch
    pub fetch_timeout: Duration,
}

impl Default for MarginConfig {
    fn default() -> Self {
        Self {
            min_equity_usd: 1000.0, // Conservative default
            max_margin_ratio: 0.8,  // 80% margin usage is dangerous
            fetch_timeout: Duration::from_secs(5),
        }
    }
}

/// Result of the margin priming process
#[derive(Debug)]
pub enum PrimeResult {
    Success(AccountSnapshot),
    InsufficientEquity { current: f64, required: f64 },
    MarginRatioExceeded { current: f64, max: f64 },
    FetchTimeout,
    ApiError(String),
}

/// Primes the cross-margin pool by fetching and validating Binance account data
pub async fn prime_cross_margin_pool(config: &MarginConfig) -> PrimeResult {
    log::info!("Priming cross-margin pool...");
    let start = Instant::now();

    // Simulate fetching snapshot from Binance Futures API
    // In production, this uses `reqwest` or a dedicated Binance client
    let snapshot = match fetch_binance_account_snapshot(config.fetch_timeout).await {
        Ok(data) => data,
        Err(e) => return PrimeResult::ApiError(e),
    };

    log::info!(
        "Account Snapshot: Wallet=${}, Available=${}, MarginRatio={:.4}",
        snapshot.total_wallet_balance,
        snapshot.available_balance,
        snapshot.margin_ratio
    );

    // Validate Equity
    if snapshot.total_wallet_balance < config.min_equity_usd {
        log::error!(
            "Insufficient equity: {} < {}",
            snapshot.total_wallet_balance,
            config.min_equity_usd
        );
        return PrimeResult::InsufficientEquity {
            current: snapshot.total_wallet_balance,
            required: config.min_equity_usd,
        };
    }

    // Validate Margin Ratio
    if snapshot.margin_ratio > config.max_margin_ratio {
        log::error!(
            "Margin ratio exceeded: {:.4} > {:.4}",
            snapshot.margin_ratio,
            config.max_margin_ratio
        );
        return PrimeResult::MarginRatioExceeded {
            current: snapshot.margin_ratio,
            max: config.max_margin_ratio,
        };
    }

    let elapsed = start.elapsed();
    log::info!("Cross-margin pool primed successfully in {:?}", elapsed);
    
    PrimeResult::Success(snapshot)
}

/// Mock function to simulate fetching Binance account snapshot
/// In production, replace with actual HTTP client call
async fn fetch_binance_account_snapshot(timeout: Duration) -> Result<AccountSnapshot, String> {
    // Simulate network delay
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Mock data - in production, parse JSON from Binance API
    Ok(AccountSnapshot {
        total_wallet_balance: 50000.0,
        available_balance: 45000.0,
        total_unrealized_pnl: 0.0,
        margin_ratio: 0.1,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prime_success() {
        let config = MarginConfig {
            min_equity_usd: 1000.0,
            max_margin_ratio: 0.5,
            ..Default::default()
        };
        
        // Since our mock returns 50k equity and 0.1 margin ratio, this should succeed
        let result = prime_cross_margin_pool(&config).await;
        assert!(matches!(result, PrimeResult::Success(_)));
    }

    #[tokio::test]
    async fn test_prime_insufficient_equity() {
        let config = MarginConfig {
            min_equity_usd: 100000.0, // Higher than mock data
            ..Default::default()
        };
        
        let result = prime_cross_margin_pool(&config).await;
        assert!(matches!(result, PrimeResult::InsufficientEquity { .. }));
    }
}
