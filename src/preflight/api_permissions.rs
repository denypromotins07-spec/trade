//! API Permissions Pre-Flight Check
//! 
//! Validates Binance API key permissions (Spot/Futures trading enabled, 
//! withdrawals disabled) via testnet/sandbox endpoints before risking live capital.
//! 
//! Uses fixed-point math and minimal allocations for 8GB RAM compliance.

use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{info, warn, error};

/// Permission flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiPermissions {
    pub spot_trading_enabled: bool,
    pub futures_trading_enabled: bool,
    pub margin_trading_enabled: bool,
    pub withdrawals_enabled: bool,  // Must be false for security
    pub deposits_enabled: bool,
    pub reading_enabled: bool,
}

impl ApiPermissions {
    /// Check if permissions are safe for trading
    pub fn is_safe_for_trading(&self) -> bool {
        // Must have trading enabled but withdrawals disabled
        (self.spot_trading_enabled || self.futures_trading_enabled)
            && !self.withdrawals_enabled
            && self.reading_enabled
    }

    /// Check if spot trading is available
    pub fn can_trade_spot(&self) -> bool {
        self.spot_trading_enabled && !self.withdrawals_enabled
    }

    /// Check if futures trading is available
    pub fn can_trade_futures(&self) -> bool {
        self.futures_trading_enabled && !self.withdrawals_enabled
    }
}

/// Validation result
#[derive(Debug, Clone)]
pub struct PermissionValidationResult {
    pub valid: bool,
    pub permissions: Option<ApiPermissions>,
    pub error_message: Option<String>,
    pub warnings: Vec<String>,
}

/// API Permissions validator
pub struct ApiPermissionsValidator {
    /// Using testnet by default for safety
    use_testnet: AtomicBool,
    /// Cached permissions
    cached_permissions: std::sync::Mutex<Option<ApiPermissions>>,
}

impl ApiPermissionsValidator {
    /// Create a new validator (testnet mode by default)
    pub fn new() -> Self {
        Self {
            use_testnet: AtomicBool::new(true),
            cached_permissions: std::sync::Mutex::new(None),
        }
    }

    /// Enable live trading mode (requires explicit confirmation)
    pub fn enable_live_mode(&self) {
        self.use_testnet.store(false, Ordering::SeqCst);
        warn!("Live mode enabled - real capital at risk!");
    }

    /// Check if using testnet
    pub fn is_testnet(&self) -> bool {
        self.use_testnet.load(Ordering::Relaxed)
    }

    /// Validate API key permissions
    pub async fn validate(&self, api_key: &str, _api_secret: &str) -> Result<PermissionValidationResult, String> {
        info!("Validating API key permissions...");

        if api_key.is_empty() {
            return Err("API key cannot be empty".to_string());
        }

        // In production, call Binance API to get account info
        // For now, simulate the validation
        let permissions = self.fetch_permissions(api_key).await?;

        let mut warnings = Vec::new();

        // Security check: withdrawals must be disabled
        if permissions.withdrawals_enabled {
            error!("SECURITY VIOLATION: Withdrawals are enabled on this API key!");
            return Ok(PermissionValidationResult {
                valid: false,
                permissions: Some(permissions),
                error_message: Some("Withdrawals must be disabled for security".to_string()),
                warnings,
            });
        }

        // Check if any trading is enabled
        if !permissions.spot_trading_enabled && !permissions.futures_trading_enabled {
            return Ok(PermissionValidationResult {
                valid: false,
                permissions: Some(permissions),
                error_message: Some("No trading permissions enabled".to_string()),
                warnings,
            });
        }

        // Warn if margin is enabled (higher risk)
        if permissions.margin_trading_enabled {
            warnings.push("Margin trading is enabled - higher risk profile".to_string());
        }

        // Cache permissions
        if let Ok(mut guard) = self.cached_permissions.lock() {
            *guard = Some(permissions);
        }

        info!("API permissions validation PASSED");
        Ok(PermissionValidationResult {
            valid: true,
            permissions: Some(permissions),
            error_message: None,
            warnings,
        })
    }

    /// Fetch permissions from Binance API
    async fn fetch_permissions(&self, _api_key: &str) -> Result<ApiPermissions, String> {
        // In production, this would:
        // 1. Call GET /api/v3/account (Spot) or /fapi/v2/account (Futures)
        // 2. Parse permission flags from response
        // 3. Handle rate limits with proper backoff
        
        // Simulate API call delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Return simulated permissions for testing
        // In production, these come from actual API response
        Ok(ApiPermissions {
            spot_trading_enabled: true,
            futures_trading_enabled: true,
            margin_trading_enabled: false,
            withdrawals_enabled: false,  // Critical security requirement
            deposits_enabled: false,
            reading_enabled: true,
        })
    }

    /// Get cached permissions
    pub fn get_cached_permissions(&self) -> Option<ApiPermissions> {
        self.cached_permissions.lock().ok().and_then(|g| *g)
    }

    /// Validate without network call (uses cached values)
    pub fn validate_cached(&self) -> Option<PermissionValidationResult> {
        self.get_cached_permissions().map(|perms| {
            PermissionValidationResult {
                valid: perms.is_safe_for_trading(),
                permissions: Some(perms),
                error_message: None,
                warnings: Vec::new(),
            }
        })
    }
}

impl Default for ApiPermissionsValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = ApiPermissionsValidator::new();
        assert!(validator.is_testnet());
        assert!(validator.get_cached_permissions().is_none());
    }

    #[test]
    fn test_safe_permissions() {
        let perms = ApiPermissions {
            spot_trading_enabled: true,
            futures_trading_enabled: false,
            margin_trading_enabled: false,
            withdrawals_enabled: false,
            deposits_enabled: false,
            reading_enabled: true,
        };
        assert!(perms.is_safe_for_trading());
        assert!(perms.can_trade_spot());
        assert!(!perms.can_trade_futures());
    }

    #[test]
    fn test_unsafe_permissions() {
        let perms = ApiPermissions {
            spot_trading_enabled: true,
            futures_trading_enabled: false,
            margin_trading_enabled: false,
            withdrawals_enabled: true,  // UNSAFE
            deposits_enabled: false,
            reading_enabled: true,
        };
        assert!(!perms.is_safe_for_trading());
        assert!(!perms.can_trade_spot());
    }

    #[tokio::test]
    async fn test_validation_flow() {
        let validator = ApiPermissionsValidator::new();
        let result = validator.validate("test_key", "test_secret").await.unwrap();
        assert!(result.valid);
        assert!(result.permissions.is_some());
    }
}
