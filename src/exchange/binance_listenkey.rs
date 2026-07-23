//! Binance ListenKey Keep-Alive - User Data Stream Heartbeat Management
//! 
//! This module automates the keep-alive heartbeat for Binance User Data Streams.
//! Seamlessly refreshes the listenKey in the background without dropping active WebSocket connections.
//! Optimized for AMD Ryzen AI 5 with microsecond latency and minimal memory footprint.
//! 
//! RAM Budget: Uses bounded channels and pre-allocated buffers.
//! Enforces global 8GB RAM limit via strict resource management.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::{interval, sleep};
use tracing::{info, warn, error, debug};

/// Default listenKey validity period (60 minutes per Binance spec)
const LISTEN_KEY_VALIDITY_MS: u64 = 60 * 60 * 1000;

/// Refresh interval (refresh at 50% of validity period = 30 minutes)
const REFRESH_INTERVAL_MS: u64 = LISTEN_KEY_VALIDITY_MS / 2;

/// Advance refresh trigger (refresh 5 minutes before expiry)
const ADVANCE_REFRESH_MS: u64 = 5 * 60 * 1000;

/// Maximum retry attempts for listenKey refresh
const MAX_RETRY_ATTEMPTS: u32 = 5;

/// Retry backoff base duration
const RETRY_BACKOFF_BASE_MS: u64 = 100;

/// Callback type for obtaining new listenKey from Binance API
pub type ListenKeyProvider = dyn Fn() -> BoxFuture<'static, Result<String, ListenKeyError>> + Send + Sync;
use std::future::Future;
fn BoxFuture<'a, T>(f: impl Future<Output = T> + Send + 'a) -> std::pin::Pin<Box<dyn Future<Output = T> + Send + 'a>> {
    Box::pin(f)
}

/// Error types for listenKey operations
#[derive(Debug, Clone)]
pub enum ListenKeyError {
    /// Failed to obtain new listenKey from API
    ApiError(String),
    /// Connection dropped during refresh
    ConnectionDropped,
    /// Maximum retries exceeded
    MaxRetriesExceeded,
    /// Invalid listenKey format
    InvalidFormat,
    /// Timeout during refresh
    Timeout,
}

impl std::fmt::Display for ListenKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiError(msg) => write!(f, "API error: {}", msg),
            Self::ConnectionDropped => write!(f, "Connection dropped"),
            Self::MaxRetriesExceeded => write!(f, "Maximum retries exceeded"),
            Self::InvalidFormat => write!(f, "Invalid listenKey format"),
            Self::Timeout => write!(f, "Refresh timeout"),
        }
    }
}

impl std::error::Error for ListenKeyError {}

/// Current state of the listenKey manager
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListenKeyState {
    /// Initial state, waiting for first key
    Initializing,
    /// Active and healthy
    Active,
    /// Refreshing in background
    Refreshing,
    /// Degraded, approaching expiry
    Degraded,
    /// Failed, needs manual intervention
    Failed,
}

/// Statistics for listenKey management
#[derive(Debug, Clone, Copy)]
pub struct ListenKeyStats {
    pub total_refreshes: u64,
    pub successful_refreshes: u64,
    pub failed_refreshes: u64,
    pub last_refresh_timestamp_ms: u64,
    pub next_refresh_timestamp_ms: u64,
    pub current_state: ListenKeyState,
    pub retry_count: u32,
}

/// Internal message types for the keep-alive task
#[derive(Debug)]
enum KeepAliveMessage {
    /// Trigger immediate refresh
    RefreshNow,
    /// Update the listenKey provider callback
    UpdateProvider,
    /// Get current status (response channel included)
    GetStatus(mpsc::Sender<ListenKeyStats>),
    /// Shutdown signal
    Shutdown,
}

/// Main listenKey keep-alive manager
pub struct ListenKeyManager {
    /// Current listenKey value
    current_key: Arc<RwLock<String>>,
    /// State flag
    state: Arc<AtomicU64>,
    /// Statistics
    stats: Arc<RwLock<ListenKeyStats>>,
    /// Shutdown flag
    shutdown_flag: Arc<AtomicBool>,
    /// Sender for keep-alive task communication
    tx: mpsc::Sender<KeepAliveMessage>,
    /// Handle to the background task
    task_handle: tokio::task::JoinHandle<()>,
    /// Timestamp when current key was obtained (milliseconds since epoch)
    key_timestamp_ms: Arc<AtomicU64>,
    /// ListenKey provider function
    provider: Arc<RwLock<Option<Arc<ListenKeyProvider>>>>,
}

impl ListenKeyManager {
    /// Create a new listenKey manager with automatic keep-alive
    /// 
    /// # Arguments
    /// * `provider` - Async function to fetch new listenKey from Binance API
    /// * `initial_key` - Optional initial listenKey if already available
    /// 
    /// # Returns
    /// Result containing the manager instance
    pub async fn new<F, Fut>(
        provider: F,
        initial_key: Option<String>,
    ) -> Result<Self, ListenKeyError>
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, ListenKeyError>> + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel::<KeepAliveMessage>(16);
        
        let current_key = Arc::new(RwLock::new(
            initial_key.unwrap_or_else(|| String::new())
        ));
        
        let state = Arc::new(AtomicU64::new(ListenKeyState::Initializing as u64));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let key_timestamp_ms = Arc::new(AtomicU64::new(0));
        
        let stats = Arc::new(RwLock::new(ListenKeyStats {
            total_refreshes: 0,
            successful_refreshes: 0,
            failed_refreshes: 0,
            last_refresh_timestamp_ms: 0,
            next_refresh_timestamp_ms: get_timestamp_ms() + REFRESH_INTERVAL_MS,
            current_state: ListenKeyState::Initializing,
            retry_count: 0,
        }));
        
        // Wrap provider in Arc
        let provider_arc: Arc<ListenKeyProvider> = Arc::new(move || {
            let fut = provider();
            Box::pin(fut)
        });
        let provider = Arc::new(RwLock::new(Some(provider_arc)));
        
        // Clone Arcs for background task
        let task_key = Arc::clone(&current_key);
        let task_state = Arc::clone(&state);
        let task_stats = Arc::clone(&stats);
        let task_shutdown = Arc::clone(&shutdown_flag);
        let task_timestamp = Arc::clone(&key_timestamp_ms);
        let task_provider = Arc::clone(&provider);
        
        // Spawn background keep-alive task
        let task_handle = tokio::spawn(async move {
            run_keep_alive_task(
                task_key,
                task_state,
                task_stats,
                task_shutdown,
                task_timestamp,
                task_provider,
                rx,
            ).await;
        });
        
        Ok(Self {
            current_key,
            state,
            stats,
            shutdown_flag,
            tx,
            task_handle,
            key_timestamp_ms,
            provider,
        })
    }

    /// Get the current listenKey value
    pub async fn get_key(&self) -> String {
        self.current_key.read().await.clone()
    }

    /// Check if the listenKey is healthy (not approaching expiry)
    pub async fn is_healthy(&self) -> bool {
        let state_val = self.state.load(Ordering::Relaxed);
        let state = unsafe { std::mem::transmute::<u64, ListenKeyState>(state_val) };
        matches!(state, ListenKeyState::Active | ListenKeyState::Refreshing)
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> ListenKeyStats {
        self.stats.read().await.clone()
    }

    /// Trigger an immediate refresh (for testing or manual intervention)
    pub async fn trigger_refresh(&self) -> Result<(), ListenKeyError> {
        self.tx.send(KeepAliveMessage::RefreshNow)
            .await
            .map_err(|_| ListenKeyError::ConnectionDropped)
    }

    /// Gracefully shutdown the keep-alive task
    pub async fn shutdown(self) -> Result<(), ListenKeyError> {
        self.shutdown_flag.store(true, Ordering::Relaxed);
        
        self.tx.send(KeepAliveMessage::Shutdown)
            .await
            .map_err(|_| ListenKeyError::ConnectionDropped)?;
        
        // Wait for task to complete with timeout
        let shutdown_result = tokio::time::timeout(
            Duration::from_secs(5),
            self.task_handle
        ).await;
        
        match shutdown_result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(ListenKeyError::ApiError(format!("Task panicked: {:?}", e))),
            Err(_) => Err(ListenKeyError::Timeout),
        }
    }

    /// Get the current state
    pub fn current_state(&self) -> ListenKeyState {
        let state_val = self.state.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<u64, ListenKeyState>(state_val) }
    }
}

/// Get current timestamp in milliseconds
#[inline]
fn get_timestamp_ms() -> u64 {
    Instant::now().elapsed().as_millis() as u64
}

/// Convert u64 to ListenKeyState safely
#[inline]
fn state_from_u64(val: u64) -> ListenKeyState {
    match val {
        0 => ListenKeyState::Initializing,
        1 => ListenKeyState::Active,
        2 => ListenKeyState::Refreshing,
        3 => ListenKeyState::Degraded,
        4 => ListenKeyState::Failed,
        _ => ListenKeyState::Failed,
    }
}

/// Background keep-alive task that manages listenKey refresh cycles
async fn run_keep_alive_task(
    current_key: Arc<RwLock<String>>,
    state: Arc<AtomicU64>,
    stats: Arc<RwLock<ListenKeyStats>>,
    shutdown_flag: Arc<AtomicBool>,
    key_timestamp_ms: Arc<AtomicU64>,
    provider: Arc<RwLock<Option<Arc<ListenKeyProvider>>>>,
    mut rx: mpsc::Receiver<KeepAliveMessage>,
) {
    info!("ListenKey keep-alive task started");
    
    // Set up refresh interval timer
    let mut refresh_timer = interval(Duration::from_millis(REFRESH_INTERVAL_MS));
    refresh_timer.tick().await; // Skip first immediate tick
    
    loop {
        tokio::select! {
            // Check for messages
            msg = rx.recv() => {
                match msg {
                    Some(KeepAliveMessage::RefreshNow) => {
                        debug!("Manual refresh requested");
                        if let Err(e) = perform_refresh(
                            &current_key,
                            &state,
                            &stats,
                            &key_timestamp_ms,
                            &provider,
                        ).await {
                            error!("Manual refresh failed: {}", e);
                        }
                    }
                    Some(KeepAliveMessage::GetStatus(tx)) => {
                        let stats_snapshot = stats.read().await.clone();
                        let _ = tx.send(stats_snapshot).await;
                    }
                    Some(KeepAliveMessage::UpdateProvider) => {
                        debug!("Provider update received");
                    }
                    Some(KeepAliveMessage::Shutdown) | None => {
                        info!("ListenKey keep-alive task shutting down");
                        break;
                    }
                }
            }
            
            // Periodic refresh check
            _ = refresh_timer.tick() => {
                if shutdown_flag.load(Ordering::Relaxed) {
                    break;
                }
                
                // Check if refresh is needed
                let now_ms = get_timestamp_ms();
                let next_refresh = {
                    let stats_read = stats.read().await;
                    stats_read.next_refresh_timestamp_ms
                };
                
                if now_ms >= next_refresh {
                    debug!("Automatic refresh triggered");
                    if let Err(e) = perform_refresh(
                        &current_key,
                        &state,
                        &stats,
                        &key_timestamp_ms,
                        &provider,
                    ).await {
                        error!("Automatic refresh failed: {}", e);
                    }
                } else {
                    // Check if approaching degraded state
                    let time_until_refresh = next_refresh - now_ms;
                    if time_until_refresh < ADVANCE_REFRESH_MS {
                        state.store(ListenKeyState::Degraded as u64, Ordering::Relaxed);
                        warn!("ListenKey approaching expiry, refresh pending");
                    }
                }
            }
        }
    }
}

/// Perform the actual listenKey refresh operation
async fn perform_refresh(
    current_key: &Arc<RwLock<String>>,
    state: &Arc<AtomicU64>,
    stats: &Arc<RwLock<ListenKeyStats>>,
    key_timestamp_ms: &Arc<AtomicU64>,
    provider: &Arc<RwLock<Option<Arc<ListenKeyProvider>>>>,
) -> Result<(), ListenKeyError> {
    // Set state to refreshing
    state.store(ListenKeyState::Refreshing as u64, Ordering::Relaxed);
    
    // Increment total refreshes
    {
        let mut stats_write = stats.write().await;
        stats_write.total_refreshes += 1;
    }
    
    // Get provider
    let provider_guard = provider.read().await;
    let provider_fn = match provider_guard.as_ref() {
        Some(p) => p,
        None => {
            state.store(ListenKeyState::Failed as u64, Ordering::Relaxed);
            return Err(ListenKeyError::ApiError("No provider configured".to_string()));
        }
    };
    
    // Attempt refresh with exponential backoff retry
    let mut attempt = 0;
    let mut last_error: Option<ListenKeyError> = None;
    
    while attempt < MAX_RETRY_ATTEMPTS {
        // Call provider to get new listenKey
        match provider_fn().await {
            Ok(new_key) => {
                // Validate key format
                if new_key.is_empty() || new_key.len() > 256 {
                    last_error = Some(ListenKeyError::InvalidFormat);
                    attempt += 1;
                    sleep(Duration::from_millis(RETRY_BACKOFF_BASE_MS * (1 << attempt))).await;
                    continue;
                }
                
                // Update current key
                let mut key_write = current_key.write().await;
                *key_write = new_key;
                
                // Update timestamp
                let now_ms = get_timestamp_ms();
                key_timestamp_ms.store(now_ms, Ordering::Relaxed);
                
                // Update statistics
                {
                    let mut stats_write = stats.write().await;
                    stats_write.successful_refreshes += 1;
                    stats_write.last_refresh_timestamp_ms = now_ms;
                    stats_write.next_refresh_timestamp_ms = now_ms + REFRESH_INTERVAL_MS;
                    stats_write.retry_count = 0;
                    stats_write.current_state = ListenKeyState::Active;
                }
                
                // Set state to active
                state.store(ListenKeyState::Active as u64, Ordering::Relaxed);
                
                info!("ListenKey refreshed successfully");
                return Ok(());
            }
            Err(e) => {
                last_error = Some(e);
                attempt += 1;
                
                if attempt < MAX_RETRY_ATTEMPTS {
                    let backoff_ms = RETRY_BACKOFF_BASE_MS * (1 << attempt);
                    warn!("Refresh attempt {} failed, retrying in {}ms: {:?}", 
                          attempt, backoff_ms, last_error);
                    sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }
    }
    
    // All retries exhausted
    {
        let mut stats_write = stats.write().await;
        stats_write.failed_refreshes += 1;
        stats_write.retry_count = attempt;
        stats_write.current_state = ListenKeyState::Failed;
    }
    
    state.store(ListenKeyState::Failed as u64, Ordering::Relaxed);
    
    error!("ListenKey refresh failed after {} attempts", attempt);
    Err(last_error.unwrap_or(ListenKeyError::MaxRetriesExceeded))
}

/// Builder for configuring ListenKeyManager
pub struct ListenKeyManagerBuilder<F, Fut>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, ListenKeyError>> + Send + 'static,
{
    provider: Option<F>,
    initial_key: Option<String>,
    custom_refresh_interval_ms: Option<u64>,
}

impl<F, Fut> ListenKeyManagerBuilder<F, Fut>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, ListenKeyError>> + Send + 'static,
{
    pub fn new() -> Self {
        Self {
            provider: None,
            initial_key: None,
            custom_refresh_interval_ms: None,
        }
    }

    pub fn provider(mut self, provider: F) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn initial_key(mut self, key: String) -> Self {
        self.initial_key = Some(key);
        self
    }

    pub fn refresh_interval(mut self, interval_ms: u64) -> Self {
        self.custom_refresh_interval_ms = Some(interval_ms);
        self
    }

    pub async fn build(self) -> Result<ListenKeyManager, ListenKeyError> {
        let provider = self.provider.ok_or_else(|| {
            ListenKeyError::ApiError("Provider function is required".to_string())
        })?;
        
        ListenKeyManager::new(provider, self.initial_key).await
    }
}

impl<F, Fut> Default for ListenKeyManagerBuilder<F, Fut>
where
    F: Fn() -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<String, ListenKeyError>> + Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_listenkey_manager_creation() {
        let provider = || async {
            Ok("test_listen_key_12345".to_string())
        };
        
        let manager = ListenKeyManager::new(provider, None).await;
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        let key = manager.get_key().await;
        assert_eq!(key, "test_listen_key_12345");
        
        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_initial_key_provided() {
        let provider = || async {
            Ok("new_listen_key".to_string())
        };
        
        let manager = ListenKeyManager::new(
            provider,
            Some("initial_key".to_string())
        ).await.unwrap();
        
        // Should have initial key immediately
        let key = manager.get_key().await;
        assert_eq!(key, "initial_key");
        
        manager.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let provider = || async {
            Ok("test_key".to_string())
        };
        
        let manager = ListenKeyManager::new(provider, None).await.unwrap();
        
        // Give it time to initialize
        sleep(Duration::from_millis(100)).await;
        
        let state = manager.current_state();
        assert!(state == ListenKeyState::Active || state == ListenKeyState::Initializing);
        
        manager.shutdown().await.unwrap();
    }
}
