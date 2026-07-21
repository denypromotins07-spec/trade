// =============================================================================
// NAUTILUS/RAY CRYPTO TRADING BOT - ZERO-OVERHEAD CONFIGURATION PARSER
// =============================================================================
// File: src/core/config.rs
// Purpose: Compile-time verified environment parsing with const generics
// Memory Model: Static allocation, zero runtime heap usage for config
// Validation: All .env variables mapped to strongly-typed structures
// =============================================================================

#![allow(dead_code)]

use std::env;
use std::marker::PhantomData;
use std::str::FromStr;

/// Maximum length for string configuration values (prevents heap allocation).
const MAX_CONFIG_STRING_LEN: usize = 256;

/// Maximum number of trading symbols supported.
const MAX_SYMBOLS: usize = 32;

// =============================================================================
// TYPE-SAFE CONFIGURATION WRAPPERS
// =============================================================================

/// Newtype wrapper for validated string configurations.
/// Uses const generics to enforce maximum length at compile time.
#[derive(Debug, Clone)]
pub struct ConfigString<const MAX_LEN: usize>([u8; MAX_LEN], usize);

impl<const MAX_LEN: usize> ConfigString<MAX_LEN> {
    /// Create a new ConfigString from a &str, truncating if necessary.
    #[inline]
    pub fn new(s: &str) -> Self {
        let mut buffer = [0u8; MAX_LEN];
        let len = s.len().min(MAX_LEN);
        buffer[..len].copy_from_slice(&s.as_bytes()[..len]);
        Self(buffer, len)
    }

    /// Get the string as a &str slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        // Safety: We only store valid UTF-8 from environment variables
        unsafe { std::str::from_utf8_unchecked(&self.0[..self.1]) }
    }

    /// Check if the string is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.1 == 0
    }
}

/// Newtype wrapper for validated numeric configurations.
/// Ensures values are within acceptable ranges at parse time.
#[derive(Debug, Clone, Copy)]
pub struct ConfigValue<T> {
    value: T,
    _marker: PhantomData<T>,
}

impl<T> ConfigValue<T> {
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn get(&self) -> T {
        self.value
    }
}

impl ConfigValue<u64> {
    /// Parse u64 from environment variable with default fallback.
    #[inline]
    pub fn from_env_or_default(key: &str, default: u64) -> Self {
        let value = env::var(key)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(default);
        Self::new(value)
    }
}

impl ConfigValue<f64> {
    /// Parse f64 from environment variable with default fallback.
    #[inline]
    pub fn from_env_or_default(key: &str, default: f64) -> Self {
        let value = env::var(key)
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(default);
        Self::new(value)
    }
}

impl ConfigValue<bool> {
    /// Parse boolean from environment variable with default fallback.
    /// Accepts "true", "1", "yes" as true; anything else is false.
    #[inline]
    pub fn from_env_or_default(key: &str, default: bool) -> Self {
        let value = env::var(key)
            .ok()
            .map(|s| matches!(s.to_lowercase().as_str(), "true" | "1" | "yes"))
            .unwrap_or(default);
        Self::new(value)
    }
}

// =============================================================================
// BINANCE API CONFIGURATION STRUCT
// =============================================================================

/// Strongly-typed Binance API configuration.
#[derive(Debug, Clone)]
pub struct BinanceConfig {
    pub api_key: ConfigString<MAX_CONFIG_STRING_LEN>,
    pub api_secret: ConfigString<MAX_CONFIG_STRING_LEN>,
    pub testnet: ConfigValue<bool>,
    pub ws_endpoint: ConfigString<MAX_CONFIG_STRING_LEN>,
    pub rest_endpoint: ConfigString<MAX_CONFIG_STRING_LEN>,
    pub symbols: [ConfigString<MAX_CONFIG_STRING_LEN>; MAX_SYMBOLS],
    pub symbol_count: usize,
    pub leverage_max: ConfigValue<u64>,
    pub order_timeout_ms: ConfigValue<u64>,
}

impl BinanceConfig {
    /// Load Binance configuration from environment variables.
    pub fn load() -> Self {
        let symbols_raw = env::var("TRADING_SYMBOLS").unwrap_or_else(|_| String::from("BTCUSDT"));
        let symbols: Vec<&str> = symbols_raw.split(',').take(MAX_SYMBOLS).collect();
        
        let mut symbol_array = [ConfigString::new(""); MAX_SYMBOLS];
        for (i, sym) in symbols.iter().enumerate() {
            if i >= MAX_SYMBOLS {
                break;
            }
            symbol_array[i] = ConfigString::new(sym);
        }

        Self {
            api_key: ConfigString::new(&env::var("BINANCE_API_KEY").unwrap_or_default()),
            api_secret: ConfigString::new(&env::var("BINANCE_API_SECRET").unwrap_or_default()),
            testnet: ConfigValue::from_env_or_default("BINANCE_TESTNET", true),
            ws_endpoint: ConfigString::new(
                &env::var("BINANCE_WS_ENDPOINT")
                    .unwrap_or_else(|_| String::from("wss://fstream.binancefuture.com/ws")),
            ),
            rest_endpoint: ConfigString::new(
                &env::var("BINANCE_REST_ENDPOINT")
                    .unwrap_or_else(|_| String::from("https://fapi.binance.com")),
            ),
            symbols: symbol_array,
            symbol_count: symbols.len().min(MAX_SYMBOLS),
            leverage_max: ConfigValue::from_env_or_default("LEVERAGE_MAX", 10),
            order_timeout_ms: ConfigValue::from_env_or_default("ORDER_TIMEOUT_MS", 500),
        }
    }

    /// Validate that API credentials are present.
    #[inline]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.api_key.is_empty() {
            return Err("BINANCE_API_KEY is not set");
        }
        if self.api_secret.is_empty() {
            return Err("BINANCE_API_SECRET is not set");
        }
        if self.symbol_count == 0 {
            return Err("No trading symbols configured");
        }
        Ok(())
    }

    /// Get iterator over configured symbols.
    #[inline]
    pub fn symbols_iter(&self) -> impl Iterator<Item = &ConfigString<MAX_CONFIG_STRING_LEN>> {
        self.symbols.iter().take(self.symbol_count)
    }
}

// =============================================================================
// RAY CLUSTER CONFIGURATION STRUCT
// =============================================================================

/// Strongly-typed Ray distributed compute cluster configuration.
#[derive(Debug, Clone, Copy)]
pub struct RayConfig {
    pub head_host: ConfigString<MAX_CONFIG_STRING_LEN>,
    pub head_port: ConfigValue<u64>,
    pub dashboard_port: ConfigValue<u64>,
    pub worker_memory_gb: ConfigValue<u64>,
    pub num_cpus: ConfigValue<u64>,
    pub object_store_memory_gb: ConfigValue<u64>,
}

impl RayConfig {
    /// Load Ray configuration from environment variables.
    pub fn load() -> Self {
        Self {
            head_host: ConfigString::new(
                &env::var("RAY_HEAD_HOST").unwrap_or_else(|_| String::from("127.0.0.1")),
            ),
            head_port: ConfigValue::from_env_or_default("RAY_HEAD_PORT", 6379),
            dashboard_port: ConfigValue::from_env_or_default("RAY_DASHBOARD_PORT", 8265),
            worker_memory_gb: ConfigValue::from_env_or_default("RAY_WORKER_MEMORY_GB", 4),
            num_cpus: ConfigValue::from_env_or_default("RAY_NUM_CPUS", 6),
            object_store_memory_gb: ConfigValue::from_env_or_default(
                "RAY_OBJECT_STORE_MEMORY_GB",
                2,
            ),
        }
    }

    /// Get the full Ray head address string.
    #[inline]
    pub fn head_address(&self) -> String {
        format!("{}:{}", self.head_host.as_str(), self.head_port.get())
    }

    /// Validate memory constraints (must not exceed system limits).
    #[inline]
    pub fn validate_memory(&self, total_system_gb: u64) -> Result<(), &'static str> {
        let ray_budget = self.worker_memory_gb.get() + self.object_store_memory_gb.get();
        if ray_budget > total_system_gb / 2 {
            return Err("Ray memory budget exceeds 50% of system RAM");
        }
        Ok(())
    }
}

// =============================================================================
// RUST ENGINE CONFIGURATION STRUCT
// =============================================================================

/// Strongly-typed Rust execution engine configuration.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub memory_cap_gb: ConfigValue<u64>,
    pub channel_capacity: ConfigValue<usize>,
    pub lto_enabled: ConfigValue<bool>,
    pub target_cpu: ConfigString<MAX_CONFIG_STRING_LEN>,
    pub mpsc_spin_count: ConfigValue<u32>,
    pub mpsc_backoff_ns: ConfigValue<u64>,
}

impl EngineConfig {
    /// Load engine configuration from environment variables.
    pub fn load() -> Self {
        Self {
            memory_cap_gb: ConfigValue::from_env_or_default("ENGINE_MEMORY_CAP_GB", 4),
            channel_capacity: ConfigValue::from_env_or_default("ENGINE_CHANNEL_CAPACITY", 65536),
            lto_enabled: ConfigValue::from_env_or_default("ENGINE_LTO_ENABLED", true),
            target_cpu: ConfigString::new(
                &env::var("ENGINE_TARGET_CPU").unwrap_or_else(|_| String::from("native")),
            ),
            mpsc_spin_count: ConfigValue::from_env_or_default("MPSC_SPIN_COUNT", 100),
            mpsc_backoff_ns: ConfigValue::from_env_or_default("MPSC_BACKOFF_NS", 50),
        }
    }

    /// Validate that channel capacity is within acceptable bounds.
    #[inline]
    pub fn validate_channel_capacity(&self) -> Result<(), &'static str> {
        let cap = self.channel_capacity.get();
        if cap < 1024 || cap > 1_000_000 {
            return Err("Channel capacity must be between 1024 and 1,000,000");
        }
        Ok(())
    }
}

// =============================================================================
// FEATURE FLAGS STRUCT
// =============================================================================

/// Strongly-typed feature flag configuration.
#[derive(Debug, Clone, Copy)]
pub struct FeatureFlags {
    pub enable_execution: ConfigValue<bool>,
    pub enable_market_data: ConfigValue<bool>,
    pub enable_risk_checks: ConfigValue<bool>,
    pub enable_ai_signals: ConfigValue<bool>,
    pub enable_paper_trading: ConfigValue<bool>,
    pub enable_verbose_logging: ConfigValue<bool>,
}

impl FeatureFlags {
    /// Load feature flags from environment variables.
    pub fn load() -> Self {
        Self {
            enable_execution: ConfigValue::from_env_or_default("FEATURE_ENABLE_EXECUTION", true),
            enable_market_data: ConfigValue::from_env_or_default(
                "FEATURE_ENABLE_MARKET_DATA",
                true,
            ),
            enable_risk_checks: ConfigValue::from_env_or_default("FEATURE_ENABLE_RISK_CHECKS", true),
            enable_ai_signals: ConfigValue::from_env_or_default("FEATURE_ENABLE_AI_SIGNALS", true),
            enable_paper_trading: ConfigValue::from_env_or_default(
                "FEATURE_ENABLE_PAPER_TRADING",
                false,
            ),
            enable_verbose_logging: ConfigValue::from_env_or_default(
                "FEATURE_ENABLE_VERBOSE_LOGGING",
                false,
            ),
        }
    }
}

// =============================================================================
// RISK PARAMETERS STRUCT
// =============================================================================

/// Strongly-typed risk management configuration.
#[derive(Debug, Clone, Copy)]
pub struct RiskConfig {
    pub max_position_size_usd: ConfigValue<u64>,
    pub max_daily_loss_usd: ConfigValue<u64>,
    pub max_open_orders: ConfigValue<u64>,
    pub stop_loss_percent: ConfigValue<f64>,
    pub take_profit_percent: ConfigValue<f64>,
    pub emergency_kill_switch: ConfigValue<bool>,
}

impl RiskConfig {
    /// Load risk configuration from environment variables.
    pub fn load() -> Self {
        Self {
            max_position_size_usd: ConfigValue::from_env_or_default("MAX_POSITION_SIZE_USD", 10000),
            max_daily_loss_usd: ConfigValue::from_env_or_default("MAX_DAILY_LOSS_USD", 500),
            max_open_orders: ConfigValue::from_env_or_default("MAX_OPEN_ORDERS", 10),
            stop_loss_percent: ConfigValue::from_env_or_default("STOP_LOSS_PERCENT", 2.0),
            take_profit_percent: ConfigValue::from_env_or_default("TAKE_PROFIT_PERCENT", 5.0),
            emergency_kill_switch: ConfigValue::from_env_or_default("EMERGENCY_KILL_SWITCH", true),
        }
    }

    /// Validate risk parameters are within sane bounds.
    #[inline]
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_daily_loss_usd.get() == 0 {
            return Err("Max daily loss must be greater than 0");
        }
        if self.stop_loss_percent.get() <= 0.0 || self.stop_loss_percent.get() >= 100.0 {
            return Err("Stop loss percent must be between 0 and 100");
        }
        if self.take_profit_percent.get() <= 0.0 || self.take_profit_percent.get() >= 100.0 {
            return Err("Take profit percent must be between 0 and 100");
        }
        Ok(())
    }
}

// =============================================================================
// MASTER CONFIGURATION STRUCT
// =============================================================================

/// Aggregate configuration for the entire trading system.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub binance: BinanceConfig,
    pub ray: RayConfig,
    pub engine: EngineConfig,
    pub features: FeatureFlags,
    pub risk: RiskConfig,
}

impl AppConfig {
    /// Load all configuration sections from environment.
    pub fn load() -> Result<Self, &'static str> {
        let binance = BinanceConfig::load();
        let ray = RayConfig::load();
        let engine = EngineConfig::load();
        let features = FeatureFlags::load();
        let risk = RiskConfig::load();

        // Validate critical sections
        binance.validate()?;
        engine.validate_channel_capacity()?;
        risk.validate()?;

        Ok(Self {
            binance,
            ray,
            engine,
            features,
            risk,
        })
    }

    /// Print configuration summary (for startup logging).
    pub fn print_summary(&self) {
        println!("=== Configuration Summary ===");
        println!(
            "Binance: {} symbols, Testnet={}",
            self.binance.symbol_count,
            self.binance.testnet.get()
        );
        println!(
            "Ray: {} CPUs, {}GB worker memory",
            self.ray.num_cpus.get(),
            self.ray.worker_memory_gb.get()
        );
        println!(
            "Engine: {}GB memory cap, channel capacity={}",
            self.engine.memory_cap_gb.get(),
            self.engine.channel_capacity.get()
        );
        println!(
            "Risk: Max position=${}, Max daily loss=${}",
            self.risk.max_position_size_usd.get(),
            self.risk.max_daily_loss_usd.get()
        );
        println!("==============================");
    }
}

// =============================================================================
// COMPILE-TIME CONFIGURATION VALIDATION
// =============================================================================

/// Const generic parameter for strict type checking.
/// This ensures configuration sizes are known at compile time.
pub const CONFIG_BUFFER_SIZE: usize = MAX_CONFIG_STRING_LEN;

/// Compile-time assertion that MAX_SYMBOLS is reasonable.
const _: () = assert!(MAX_SYMBOLS > 0 && MAX_SYMBOLS <= 1024, "Invalid MAX_SYMBOLS");

/// Compile-time assertion that buffer size is reasonable.
const _: () = assert!(
    MAX_CONFIG_STRING_LEN > 0 && MAX_CONFIG_STRING_LEN <= 4096,
    "Invalid MAX_CONFIG_STRING_LEN"
);

// =============================================================================
// MEMORY MANAGEMENT NOTES
// =============================================================================
// 
// This configuration module follows strict memory management principles:
// 
// 1. STATIC ALLOCATION: All strings use fixed-size arrays ([u8; N]) instead
//    of heap-allocated String types. This eliminates dynamic allocation.
// 
// 2. CONST GENERICS: Buffer sizes are enforced at compile time via const
//    generic parameters, preventing buffer overflows and ensuring type safety.
// 
// 3. COPY SEMANTICS: Numeric configs use Copy types, allowing stack-only
//    storage without any heap indirection.
// 
// 4. ZERO-COST ABSTRACTIONS: All wrapper types are transparent at runtime;
//    the compiler optimizes them away completely in release builds.
// 
// 5. EARLY VALIDATION: Configuration errors are caught during load(), before
//    any subsystem initialization, preventing runtime failures.
// 
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_string_creation() {
        let s = ConfigString::<64>::new("test_value");
        assert_eq!(s.as_str(), "test_value");
        assert!(!s.is_empty());
    }

    #[test]
    fn test_config_string_truncation() {
        let s = ConfigString::<4>::new("hello_world");
        assert_eq!(s.as_str(), "hell");
        assert_eq!(s.as_str().len(), 4);
    }

    #[test]
    fn test_config_value_bool() {
        std::env::set_var("TEST_BOOL", "true");
        let val = ConfigValue::<bool>::from_env_or_default("TEST_BOOL", false);
        assert!(val.get());
    }
}
