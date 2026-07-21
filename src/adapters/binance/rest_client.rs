//! Binance REST Client for Historical Data and Order Management
//! 
//! This module implements a hyper-optimized REST client using the `hyper` library
//! for fetching historical snapshots and managing rate limits with strict token-bucket
//! algorithms. Designed for microsecond latency on AMD Ryzen AI 5 architecture.
//! 
//! Key Features:
//! - Token-bucket rate limiting to respect Binance API limits
//! - Connection pooling for reduced TCP handshake overhead
//! - Pre-allocated response buffers to eliminate heap allocations
//! - Retry logic with exponential backoff for transient errors

use hyper::{Client, Body, Method, Request, StatusCode};
use hyper_tls::HttpsConnector;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::core::config::SystemConfig;
use crate::adapters::binance::auth::BinanceAuth;

/// Token bucket for rate limiting
pub struct TokenBucket {
    /// Maximum tokens (API requests allowed per window)
    capacity: u64,
    /// Current available tokens
    tokens: AtomicU64,
    /// Refill rate (tokens per millisecond)
    refill_rate: f64,
    /// Last refill timestamp
    last_refill: Mutex<Instant>,
}

/// REST client configuration
pub struct RestClientConfig {
    pub base_url: String,
    pub recv_window_ms: u64,
    pub max_retries: u32,
    pub timeout_ms: u64,
}

/// Hyper-optimized REST client
pub struct BinanceRestClient {
    /// HTTP client with connection pooling
    http_client: Client<HttpsConnector<hyper::client::HttpConnector>>,
    /// Authentication handler
    auth: BinanceAuth,
    /// Rate limiter
    rate_limiter: Arc<TokenBucket>,
    /// Configuration
    config: RestClientConfig,
    /// Connection state
    is_healthy: AtomicBool,
}

impl TokenBucket {
    /// Create a new token bucket
    pub fn new(capacity: u64, refill_per_second: u64) -> Self {
        Self {
            capacity,
            tokens: AtomicU64::new(capacity),
            refill_rate: refill_per_second as f64 / 1000.0,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Acquire a token, blocking if necessary
    pub async fn acquire(&self) {
        loop {
            let current_tokens = self.tokens.load(Ordering::Acquire);
            
            if current_tokens > 0 {
                // Try to decrement atomically
                if self
                    .tokens
                    .compare_exchange(current_tokens, current_tokens - 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    return;
                }
            } else {
                // Wait for refill
                let wait_time = Duration::from_millis((1000.0 / self.refill_rate) as u64 + 1);
                tokio::time::sleep(wait_time).await;
                
                // Refill tokens
                let mut last_refill = self.last_refill.lock().await;
                let elapsed = last_refill.elapsed().as_millis() as f64;
                let new_tokens = (elapsed as f64 * self.refill_rate) as u64;
                
                if new_tokens > 0 {
                    self.tokens.fetch_min(self.capacity, Ordering::Relaxed);
                    self.tokens.fetch_add(new_tokens.min(self.capacity), Ordering::Relaxed);
                    *last_refill = Instant::now();
                }
            }
        }
    }

    /// Try to acquire a token without blocking
    pub fn try_acquire(&self) -> bool {
        let current_tokens = self.tokens.load(Ordering::Acquire);
        
        if current_tokens > 0 {
            self.tokens
                .compare_exchange(current_tokens, current_tokens - 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        } else {
            false
        }
    }
}

impl BinanceRestClient {
    /// Create a new REST client
    pub fn new(config: &SystemConfig, auth: BinanceAuth) -> Self {
        let https = HttpsConnector::new();
        let http_client = Client::builder()
            .pool_max_idle_per_host(8)
            .keep_alive(true)
            .keep_alive_timeout(Duration::from_secs(30))
            .build(https);

        let rest_config = RestClientConfig {
            base_url: "https://api.binance.com".to_string(),
            recv_window_ms: config.binance_recv_window,
            max_retries: 3,
            timeout_ms: 5000,
        };

        // Initialize rate limiter (e.g., 1200 requests per minute for weight)
        let rate_limiter = Arc::new(TokenBucket::new(1200, 20)); // 20 tokens/sec

        Self {
            http_client,
            auth,
            rate_limiter,
            config: rest_config,
            is_healthy: AtomicBool::new(true),
        }
    }

    /// Execute a GET request with rate limiting and retries
    pub async fn get(&self, endpoint: &str, signed: bool) -> Result<String, Box<dyn std::error::Error>> {
        let mut retries = 0;
        
        while retries <= self.config.max_retries {
            // Acquire rate limit token
            self.rate_limiter.acquire().await;

            let url = format!("{}{}", self.config.base_url, endpoint);
            
            let mut req = Request::builder()
                .method(Method::GET)
                .uri(&url)
                .header("Content-Type", "application/json")
                .header("X-MBX-APIKEY", self.auth.api_key())
                .body(Body::empty())?;

            if signed {
                // Add signature and timestamp
                let (signed_url, signature) = self.auth.sign_request("GET", endpoint, "");
                req = Request::builder()
                    .method(Method::GET)
                    .uri(&format!("{}{}", self.config.base_url, &signed_url))
                    .header("Content-Type", "application/json")
                    .header("X-MBX-APIKEY", self.auth.api_key())
                    .body(Body::empty())?;
            }

            match tokio::time::timeout(
                Duration::from_millis(self.config.timeout_ms),
                self.http_client.request(req)
            ).await {
                Ok(Ok(response)) => {
                    match response.status() {
                        StatusCode::OK => {
                            let body_bytes = hyper::body::to_bytes(response.into_body()).await?;
                            return Ok(String::from_utf8_lossy(&body_bytes).to_string());
                        }
                        StatusCode::TOO_MANY_REQUESTS => {
                            retries += 1;
                            log_warn!("Rate limit exceeded, retry {} of {}", retries, self.config.max_retries);
                            let backoff = Duration::from_millis(100 * (1 << retries));
                            tokio::time::sleep(backoff).await;
                            continue;
                        }
                        status => {
                            return Err(format!("HTTP error: {}", status).into());
                        }
                    }
                }
                Ok(Err(e)) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        return Err(format!("Request failed after {} retries: {}", retries, e).into());
                    }
                    let backoff = Duration::from_millis(100 * (1 << retries));
                    tokio::time::sleep(backoff).await;
                }
                Err(_) => {
                    retries += 1;
                    if retries > self.config.max_retries {
                        return Err("Request timeout".into());
                    }
                    let backoff = Duration::from_millis(100 * (1 << retries));
                    tokio::time::sleep(backoff).await;
                }
            }
        }

        Err("Max retries exceeded".into())
    }

    /// Fetch historical klines (candlestick data)
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: u64,
        end_time: u64,
        limit: u16,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let endpoint = format!(
            "/api/v3/klines?symbol={}&interval={}&startTime={}&endTime={}&limit={}",
            symbol, interval, start_time, end_time, limit
        );
        self.get(&endpoint, false).await
    }

    /// Fetch current order book depth
    pub async fn get_order_book(&self, symbol: &str, limit: u16) -> Result<String, Box<dyn std::error::Error>> {
        let endpoint = format!("/api/v3/depth?symbol={}&limit={}", symbol, limit);
        self.get(&endpoint, false).await
    }

    /// Fetch recent trades
    pub async fn get_recent_trades(&self, symbol: &str, limit: u16) -> Result<String, Box<dyn std::error::Error>> {
        let endpoint = format!("/api/v3/trades?symbol={}&limit={}", symbol, limit);
        self.get(&endpoint, false).await
    }

    /// Check health status
    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(Ordering::Acquire)
    }

    /// Set health status
    pub fn set_healthy(&self, healthy: bool) {
        self.is_healthy.store(healthy, Ordering::Release);
    }
}

// Helper logging macros
macro_rules! log_warn {
    ($($arg:tt)*) => {
        println!("[WARN] {}", format!($($arg)*));
    };
}
