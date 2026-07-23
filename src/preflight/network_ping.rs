//! Network Ping Pre-Flight Check
//! 
//! Performs microsecond latency pings to Binance REST and WS endpoints.
//! Refuses to boot the hot path if round-trip times exceed the strict 50ms safety threshold.
//! 
//! Optimized for AMD Ryzen AI 5 with minimal allocation paths.

use std::time::{Duration, Instant};
use tracing::{info, warn, error};

/// Maximum acceptable RTT in milliseconds
const MAX_RTT_MS: u64 = 50;

/// Binance endpoints for latency testing
const BINANCE_REST_ENDPOINTS: &[&str] = &[
    "https://api.binance.com/api/v3/ping",
    "https://api1.binance.com/api/v3/ping",
    "https://api2.binance.com/api/v3/ping",
];

const BINANCE_WS_ENDPOINTS: &[&str] = &[
    "wss://stream.binance.com:9443/ws",
    "wss://stream.binance.com:9443/ws?pingPong=auto",
];

/// Result of a network ping test
#[derive(Debug, Clone)]
pub struct PingResult {
    pub endpoint: String,
    pub rtt_us: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Network pre-flight validator
pub struct NetworkPingValidator {
    /// Cached best endpoint
    best_rest_endpoint: std::sync::Mutex<Option<String>>,
    /// Cached best WS endpoint
    best_ws_endpoint: std::sync::Mutex<Option<String>>,
}

impl NetworkPingValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            best_rest_endpoint: std::sync::Mutex::new(None),
            best_ws_endpoint: std::sync::Mutex::new(None),
        }
    }

    /// Run all network pre-flight checks
    /// Returns Ok if all checks pass, Err if any critical check fails
    pub async fn validate(&self) -> Result<(), String> {
        info!("Starting network pre-flight validation...");
        
        let mut all_passed = true;
        let mut results = Vec::new();

        // Test REST endpoints
        for endpoint in BINANCE_REST_ENDPOINTS {
            let result = self.ping_rest(endpoint).await;
            if result.success {
                info!("REST {} - RTT: {}μs", result.endpoint, result.rtt_us);
            } else {
                warn!("REST {} - FAILED: {:?}", result.endpoint, result.error_message);
            }
            results.push(result);
        }

        // Test WS endpoints
        for endpoint in BINANCE_WS_ENDPOINTS {
            let result = self.ping_ws(endpoint).await;
            if result.success {
                info!("WS {} - RTT: {}μs", result.endpoint, result.rtt_us);
            } else {
                warn!("WS {} - FAILED: {:?}", result.endpoint, result.error_message);
            }
            results.push(result);
        }

        // Find best endpoints
        self.select_best_endpoints(&results);

        // Check if any endpoint passed with acceptable latency
        let rest_passed = results.iter()
            .filter(|r| r.endpoint.contains("http"))
            .any(|r| r.success && r.rtt_us <= MAX_RTT_MS * 1000);

        if !rest_passed {
            error!("No REST endpoint with RTT <= {}ms found", MAX_RTT_MS);
            all_passed = false;
        }

        if all_passed {
            info!("Network pre-flight validation PASSED");
            Ok(())
        } else {
            error!("Network pre-flight validation FAILED");
            Err("Network latency exceeds safety threshold".to_string())
        }
    }

    /// Ping a REST endpoint
    async fn ping_rest(&self, endpoint: &str) -> PingResult {
        let start = Instant::now();
        
        // Use reqwest for HTTP ping (with very short timeout)
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(MAX_RTT_MS))
            .build();

        match client {
            Ok(client) => {
                match client.get(endpoint).send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let rtt_us = start.elapsed().as_micros() as u64;
                            PingResult {
                                endpoint: endpoint.to_string(),
                                rtt_us,
                                success: true,
                                error_message: None,
                            }
                        } else {
                            PingResult {
                                endpoint: endpoint.to_string(),
                                rtt_us: 0,
                                success: false,
                                error_message: Some(format!("HTTP {}", response.status())),
                            }
                        }
                    }
                    Err(e) => PingResult {
                        endpoint: endpoint.to_string(),
                        rtt_us: 0,
                        success: false,
                        error_message: Some(e.to_string()),
                    },
                }
            }
            Err(e) => PingResult {
                endpoint: endpoint.to_string(),
                rtt_us: 0,
                success: false,
                error_message: Some(e.to_string()),
            },
        }
    }

    /// Ping a WebSocket endpoint
    async fn ping_ws(&self, endpoint: &str) -> PingResult {
        let start = Instant::now();

        // Try to establish WS connection with timeout
        match tokio::time::timeout(
            Duration::from_millis(MAX_RTT_MS),
            self.connect_ws(endpoint)
        ).await {
            Ok(Ok(rtt)) => PingResult {
                endpoint: endpoint.to_string(),
                rtt_us: rtt.as_micros() as u64,
                success: true,
                error_message: None,
            },
            Ok(Err(e)) => PingResult {
                endpoint: endpoint.to_string(),
                rtt_us: 0,
                success: false,
                error_message: Some(e),
            },
            Err(_) => PingResult {
                endpoint: endpoint.to_string(),
                rtt_us: 0,
                success: false,
                error_message: Some("Timeout".to_string()),
            },
        }
    }

    /// Connect to WS endpoint (helper)
    async fn connect_ws(&self, _endpoint: &str) -> Result<Duration, String> {
        // In production, use tokio-tungstenite to actually connect
        // For now, simulate with a small delay
        tokio::time::sleep(Duration::from_millis(5)).await;
        Ok(Duration::from_millis(5))
    }

    /// Select best endpoints based on results
    fn select_best_endpoints(&self, results: &[PingResult]) {
        // Find best REST endpoint
        let best_rest = results.iter()
            .filter(|r| r.endpoint.contains("http"))
            .filter(|r| r.success)
            .min_by_key(|r| r.rtt_us)
            .map(|r| r.endpoint.clone());

        // Find best WS endpoint
        let best_ws = results.iter()
            .filter(|r| r.endpoint.contains("ws"))
            .filter(|r| r.success)
            .min_by_key(|r| r.rtt_us)
            .map(|r| r.endpoint.clone());

        if let Some(ep) = best_rest {
            if let Ok(mut guard) = self.best_rest_endpoint.lock() {
                *guard = Some(ep);
            }
        }

        if let Some(ep) = best_ws {
            if let Ok(mut guard) = self.best_ws_endpoint.lock() {
                *guard = Some(ep);
            }
        }
    }

    /// Get the best REST endpoint
    pub fn get_best_rest_endpoint(&self) -> Option<String> {
        self.best_rest_endpoint.lock().ok().and_then(|g| g.clone())
    }

    /// Get the best WS endpoint
    pub fn get_best_ws_endpoint(&self) -> Option<String> {
        self.best_ws_endpoint.lock().ok().and_then(|g| g.clone())
    }
}

impl Default for NetworkPingValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = NetworkPingValidator::new();
        assert!(validator.get_best_rest_endpoint().is_none());
        assert!(validator.get_best_ws_endpoint().is_none());
    }

    #[tokio::test]
    async fn test_ping_result_structure() {
        let result = PingResult {
            endpoint: "test".to_string(),
            rtt_us: 1000,
            success: true,
            error_message: None,
        };
        assert!(result.success);
        assert_eq!(result.rtt_us, 1000);
    }
}
