//! health_ping.rs - Microsecond Health Ping Responder
//! Stage 54: Nautilus/Ray Crypto Trading Bot
//! Ensures frontend instantly displays "Reconnecting" UI if Rust backend restarts
//! Prevents phantom manual override clicks during backend transitions

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::io::{Read, Write};
use std::thread;
use log::{debug, error, info, warn};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Default health check port (can be overridden by port allocator)
const DEFAULT_HEALTH_PORT: u16 = 8081;

/// Ping interval in milliseconds
const PING_INTERVAL_MS: u64 = 100;

/// Timeout for considering backend unhealthy (milliseconds)
const UNHEALTHY_THRESHOLD_MS: u64 = 500;

/// Maximum ping history to retain
const MAX_PING_HISTORY: usize = 100;

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Backend is healthy and responding
    Healthy,
    /// Backend is responding but with high latency
    Degraded,
    /// Backend is not responding
    Unhealthy,
    /// Backend is restarting
    Restarting,
}

/// Ping response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResponse {
    /// Timestamp when ping was received (nanoseconds since epoch)
    pub timestamp_ns: u64,
    /// Server uptime in milliseconds
    pub uptime_ms: u64,
    /// Current health status
    pub status: HealthStatus,
    /// Number of active connections
    pub active_connections: u32,
    /// Memory usage in bytes
    pub memory_used_bytes: u64,
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: f32,
    /// Gateway version string
    pub version: String,
}

impl PingResponse {
    pub fn new(
        uptime_ms: u64,
        status: HealthStatus,
        active_connections: u32,
    ) -> Self {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;

        Self {
            timestamp_ns: now_ns,
            uptime_ms,
            status,
            active_connections,
            memory_used_bytes: 0, // Would be populated by system metrics
            cpu_usage_percent: 0.0,
            version: env!("CARGO_PKG_VERSION").unwrap_or("0.1.0").to_string(),
        }
    }
}

/// Ping history entry for latency tracking
#[derive(Debug, Clone)]
struct PingHistoryEntry {
    timestamp: Instant,
    latency_us: u64,
    status: HealthStatus,
}

/// Health ping server for frontend communication
pub struct HealthPingServer {
    /// Port the server is listening on
    port: AtomicU16,
    /// Server start time
    start_time: Instant,
    /// Running flag
    is_running: Arc<AtomicBool>,
    /// Current health status
    current_status: Arc<RwLock<HealthStatus>>,
    /// Active connection count
    active_connections: AtomicU64,
    /// Ping history for latency analysis
    ping_history: Arc<RwLock<Vec<PingHistoryEntry>>>,
    /// Last successful ping time
    last_ping_time: Arc<RwLock<Option<Instant>>>,
    /// Connection handle for graceful shutdown
    listener: Arc<RwLock<Option<TcpListener>>>,
}

impl HealthPingServer {
    /// Create a new health ping server
    pub fn new(port: u16) -> Self {
        Self {
            port: AtomicU16::new(port),
            start_time: Instant::now(),
            is_running: Arc::new(AtomicBool::new(false)),
            current_status: Arc::new(RwLock::new(HealthStatus::Unhealthy)),
            active_connections: AtomicU64::new(0),
            ping_history: Arc::new(RwLock::new(Vec::with_capacity(MAX_PING_HISTORY))),
            last_ping_time: Arc::new(RwLock::new(None)),
            listener: Arc::new(RwLock::new(None)),
        }
    }

    /// Start the health ping server
    pub fn start(&self) -> Result<(), String> {
        let port = self.port.load(Ordering::Relaxed);
        let addr: SocketAddr = ([127, 0, 0, 1], port).into();

        let listener = TcpListener::bind(addr)
            .map_err(|e| format!("Failed to bind health ping server to {}: {}", addr, e))?;

        listener.set_nonblocking(true)
            .map_err(|e| format!("Failed to set non-blocking: {}", e))?;

        info!("Health ping server started on {}", addr);

        self.listener.write().replace(listener);
        self.is_running.store(true, Ordering::Release);
        *self.current_status.write() = HealthStatus::Healthy;

        // Clone Arcs for server thread
        let is_running = self.is_running.clone();
        let current_status = self.current_status.clone();
        let active_connections = self.active_connections.clone();
        let ping_history = self.ping_history.clone();
        let last_ping_time = self.last_ping_time.clone();
        let listener = self.listener.clone();

        // Spawn server thread
        thread::Builder::new()
            .name("health_ping_server".to_string())
            .spawn(move || {
                Self::server_loop(
                    is_running,
                    current_status,
                    active_connections,
                    ping_history,
                    last_ping_time,
                    listener,
                );
            })
            .map_err(|e| format!("Failed to spawn server thread: {}", e))?;

        Ok(())
    }

    /// Main server loop
    #[allow(clippy::too_many_arguments)]
    fn server_loop(
        is_running: Arc<AtomicBool>,
        current_status: Arc<RwLock<HealthStatus>>,
        active_connections: AtomicU64,
        ping_history: Arc<RwLock<Vec<PingHistoryEntry>>>,
        last_ping_time: Arc<RwLock<Option<Instant>>>,
        listener: Arc<RwLock<Option<TcpListener>>>,
    ) {
        while is_running.load(Ordering::Relaxed) {
            let guard = listener.read();
            if let Some(ref listener) = *guard {
                match listener.accept() {
                    Ok((mut stream, addr)) => {
                        debug!("Health ping connection from {}", addr);
                        active_connections.fetch_add(1, Ordering::Relaxed);

                        // Handle connection in separate thread
                        let status_clone = current_status.clone();
                        let history_clone = ping_history.clone();
                        let last_ping_clone = last_ping_time.clone();
                        let conn_counter = active_connections.clone();

                        thread::Builder::new()
                            .name("health_handler".to_string())
                            .spawn(move || {
                                Self::handle_connection(
                                    &mut stream,
                                    status_clone,
                                    history_clone,
                                    last_ping_clone,
                                );
                                conn_counter.fetch_sub(1, Ordering::Relaxed);
                            })
                            .ok();
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // No pending connections, sleep briefly
                        thread::sleep(Duration::from_millis(PING_INTERVAL_MS));
                    }
                    Err(e) => {
                        debug!("Accept error: {}", e);
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            } else {
                error!("Listener is None");
                break;
            }
        }

        info!("Health ping server stopped");
    }

    /// Handle individual health ping connection
    fn handle_connection(
        stream: &mut TcpStream,
        current_status: Arc<RwLock<HealthStatus>>,
        ping_history: Arc<RwLock<Vec<PingHistoryEntry>>>,
        last_ping_time: Arc<RwLock<Option<Instant>>>,
    ) {
        let handle_start = Instant::now();
        let mut buffer = [0u8; 2048];

        match stream.read(&mut buffer) {
            Ok(n) => {
                let request = String::from_utf8_lossy(&buffer[..n]);
                let response = Self::process_health_request(
                    &request,
                    &current_status,
                    &ping_history,
                    &last_ping_time,
                );

                if let Err(e) = stream.write_all(response.as_bytes()) {
                    debug!("Failed to write response: {}", e);
                }
                let _ = stream.flush();

                // Record ping latency
                let latency_us = handle_start.elapsed().as_micros() as u64;
                let status = *current_status.read();

                let mut history = ping_history.write();
                history.push(PingHistoryEntry {
                    timestamp: handle_start,
                    latency_us,
                    status,
                });

                // Trim history if too long
                if history.len() > MAX_PING_HISTORY {
                    *history = history.split_off(history.len() - MAX_PING_HISTORY);
                }

                // Update last ping time
                *last_ping_time.write() = Some(handle_start);
            }
            Err(e) => {
                debug!("Health connection read error: {}", e);
            }
        }
    }

    /// Process health request and generate response
    fn process_health_request(
        request: &str,
        current_status: &Arc<RwLock<HealthStatus>>,
        ping_history: &Arc<RwLock<Vec<PingHistoryEntry>>>,
        last_ping_time: &Arc<RwLock<Option<Instant>>>,
    ) -> String {
        // Check for specific endpoints
        if request.contains("GET /health") {
            let status = *current_status.read();
            return Self::json_response(status);
        }

        if request.contains("GET /ping") {
            let latency = Self::calculate_avg_latency(ping_history);
            let status = *current_status.read();
            
            // Include latency in response
            return format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{{\"status\":\"{:?}\",\"latency_us\":{}}}",
                status, latency
            );
        }

        if request.contains("GET /status") {
            let history = ping_history.read();
            let last_status = history.last().map(|h| h.status).unwrap_or(HealthStatus::Unhealthy);
            let avg_latency = Self::calculate_avg_latency(ping_history);
            
            return format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{{\"status\":\"{:?}\",\"avg_latency_us\":{},\"ping_count\":{}}}",
                last_status, avg_latency, history.len()
            );
        }

        // Default health response
        let status = *current_status.read();
        Self::json_response(status)
    }

    /// Generate JSON health response
    fn json_response(status: HealthStatus) -> String {
        let uptime_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
        let response = PingResponse::new(uptime_ms, status, 0);
        
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{}",
            serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string())
        )
    }

    /// Calculate average latency from history
    fn calculate_avg_latency(history: &Arc<RwLock<Vec<PingHistoryEntry>>>) -> u64 {
        let entries = history.read();
        if entries.is_empty() {
            return 0;
        }
        
        let sum: u64 = entries.iter().map(|e| e.latency_us).sum();
        sum / entries.len() as u64
    }

    /// Get current health status
    pub fn get_status(&self) -> HealthStatus {
        *self.current_status.read()
    }

    /// Update health status
    pub fn set_status(&self, status: HealthStatus) {
        *self.current_status.write() = status;
        debug!("Health status updated: {:?}", status);
    }

    /// Check if backend appears healthy based on recent pings
    pub fn is_healthy(&self) -> bool {
        let last_ping = self.last_ping_time.read();
        match *last_ping {
            Some(last) => {
                let elapsed = last.elapsed().as_millis() as u64;
                elapsed < UNHEALTHY_THRESHOLD_MS
            }
            None => false,
        }
    }

    /// Get average ping latency in microseconds
    pub fn get_avg_latency_us(&self) -> u64 {
        Self::calculate_avg_latency(&self.ping_history)
    }

    /// Get ping history for analysis
    pub fn get_ping_history(&self) -> Vec<(u64, HealthStatus)> {
        self.ping_history
            .read()
            .iter()
            .map(|e| (e.latency_us, e.status))
            .collect()
    }

    /// Get active connection count
    pub fn get_active_connections(&self) -> u64 {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Get server uptime in milliseconds
    pub fn get_uptime_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Gracefully stop the server
    pub fn stop(&self) {
        info!("Stopping health ping server");
        self.is_running.store(false, Ordering::Release);
        
        // Clear listener to unblock accept
        *self.listener.write() = None;
        
        *self.current_status.write() = HealthStatus::Restarting;
    }

    /// Get the port the server is running on
    pub fn get_port(&self) -> u16 {
        self.port.load(Ordering::Relaxed)
    }
}

impl Default for HealthPingServer {
    fn default() -> Self {
        Self::new(DEFAULT_HEALTH_PORT)
    }
}

impl Drop for HealthPingServer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_server_creation() {
        let server = HealthPingServer::new(19000);
        assert_eq!(server.get_port(), 19000);
        assert_eq!(server.get_status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_health_server_start_stop() {
        let server = HealthPingServer::new(19001);
        
        // Start server
        let result = server.start();
        assert!(result.is_ok());
        
        // Give it time to start
        thread::sleep(Duration::from_millis(100));
        
        // Check status
        assert_eq!(server.get_status(), HealthStatus::Healthy);
        
        // Stop server
        server.stop();
        thread::sleep(Duration::from_millis(100));
    }

    #[test]
    fn test_ping_response_serialization() {
        let response = PingResponse::new(1000, HealthStatus::Healthy, 5);
        let json = serde_json::to_string(&response);
        assert!(json.is_ok());
        
        let json_str = json.unwrap();
        assert!(json_str.contains("\"status\":\"Healthy\""));
        assert!(json_str.contains("\"uptime_ms\":1000"));
    }

    #[test]
    fn test_health_status_enum() {
        assert_eq!(HealthStatus::Healthy as u8, 0);
        assert_eq!(HealthStatus::Degraded as u8, 1);
        assert_eq!(HealthStatus::Unhealthy as u8, 2);
        assert_eq!(HealthStatus::Restarting as u8, 3);
    }
}
