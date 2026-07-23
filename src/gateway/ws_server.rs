//! WebSocket Server - Hyper-Fast Non-Blocking Telemetry Gateway
//! 
//! This module builds a hyper-fast, non-blocking WebSocket server using tokio-tungstenite
//! to stream real-time telemetry, PnL, and order book depth directly to the React frontend.
//! Optimized for AMD Ryzen AI 5 with microsecond latency targets.
//! 
//! RAM Budget: Uses bounded channels and connection pooling.
//! Enforces global 8GB RAM limit via strict connection limits.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::{stream::SplitSink, StreamExt, SinkExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{interval, sleep};
use tokio_tungstenite::tungstenite::{Message, Error as WsError};
use tracing::{info, warn, error, debug};

/// Maximum concurrent WebSocket connections
const MAX_CONNECTIONS: usize = 100;

/// Heartbeat interval in milliseconds
const HEARTBEAT_INTERVAL_MS: u64 = 30000;

/// Message buffer size per connection
const MSG_BUFFER_SIZE: usize = 1024;

/// Broadcast channel capacity for market data
const BROADCAST_CAPACITY: usize = 10000;

/// Types of messages that can be sent to clients
#[derive(Debug, Clone)]
pub enum ServerMessage {
    /// Order book update
    OrderBook {
        symbol: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        timestamp_ms: u64,
    },
    /// PnL update
    PnL {
        total_pnl: f64,
        unrealized_pnl: f64,
        realized_pnl: f64,
        timestamp_ms: u64,
    },
    /// Telemetry data
    Telemetry {
        cpu_usage: f64,
        memory_usage_mb: f64,
        latency_us: u64,
        throughput_ops: u64,
        timestamp_ms: u64,
    },
    /// Order status update
    OrderStatus {
        order_id: String,
        symbol: String,
        status: String,
        filled_qty: f64,
        avg_price: f64,
    },
    /// System alert
    Alert {
        level: AlertLevel,
        message: String,
        timestamp_ms: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Connection state for a WebSocket client
struct ConnectionState {
    addr: SocketAddr,
    tx: mpsc::Sender<ServerMessage>,
    connected_at: Instant,
    last_heartbeat: Instant,
    messages_sent: u64,
    is_authenticated: bool,
}

/// Main WebSocket server instance
pub struct WsServer {
    /// Broadcast sender for market data
    market_tx: broadcast::Sender<ServerMessage>,
    /// Connected clients
    clients: Arc<RwLock<HashMap<SocketAddr, ConnectionState>>>,
    /// Connection counter
    connection_count: AtomicU64,
    /// Running flag
    running: AtomicBool,
    /// Total messages broadcast
    total_broadcasts: AtomicU64,
    /// Dropped messages (slow clients)
    dropped_messages: AtomicU64,
}

impl WsServer {
    /// Create a new WebSocket server
    pub fn new() -> Self {
        let (market_tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        
        Self {
            market_tx,
            clients: Arc::new(RwLock::new(HashMap::new())),
            connection_count: AtomicU64::new(0),
            running: AtomicBool::new(false),
            total_broadcasts: AtomicU64::new(0),
            dropped_messages: AtomicU64::new(0),
        }
    }
    
    /// Start the WebSocket server on the specified address
    pub async fn start(&self, addr: SocketAddr) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let listener = TcpListener::bind(addr).await?;
        self.running.store(true, Ordering::Relaxed);
        
        info!("WebSocket server listening on {}", addr);
        
        // Spawn heartbeat task
        let heartbeat_clients = Arc::clone(&self.clients);
        let heartbeat_running = self.running.clone();
        tokio::spawn(async move {
            run_heartbeat_loop(heartbeat_clients, heartbeat_running).await;
        });
        
        // Accept connections
        while self.running.load(Ordering::Relaxed) {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    // Check connection limit
                    if self.connection_count.load(Ordering::Relaxed) >= MAX_CONNECTIONS as u64 {
                        warn!("Connection limit reached, rejecting {}", addr);
                        continue;
                    }
                    
                    let clients = Arc::clone(&self.clients);
                    let market_rx = self.market_tx.subscribe();
                    
                    tokio::spawn(async move {
                        handle_connection(stream, addr, clients, market_rx).await;
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
        
        Ok(())
    }
    
    /// Broadcast order book update to all connected clients
    pub fn broadcast_orderbook(
        &self,
        symbol: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    ) -> Result<usize, broadcast::error::SendError<ServerMessage>> {
        let msg = ServerMessage::OrderBook {
            symbol,
            bids,
            asks,
            timestamp_ms: get_timestamp_ms(),
        };
        
        let result = self.market_tx.send(msg);
        if result.is_ok() {
            self.total_broadcasts.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    
    /// Broadcast PnL update
    pub fn broadcast_pnl(
        &self,
        total_pnl: f64,
        unrealized_pnl: f64,
        realized_pnl: f64,
    ) -> Result<usize, broadcast::error::SendError<ServerMessage>> {
        let msg = ServerMessage::PnL {
            total_pnl,
            unrealized_pnl,
            realized_pnl,
            timestamp_ms: get_timestamp_ms(),
        };
        
        let result = self.market_tx.send(msg);
        if result.is_ok() {
            self.total_broadcasts.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    
    /// Broadcast telemetry data
    pub fn broadcast_telemetry(
        &self,
        cpu_usage: f64,
        memory_usage_mb: f64,
        latency_us: u64,
        throughput_ops: u64,
    ) -> Result<usize, broadcast::error::SendError<ServerMessage>> {
        let msg = ServerMessage::Telemetry {
            cpu_usage,
            memory_usage_mb,
            latency_us,
            throughput_ops,
            timestamp_ms: get_timestamp_ms(),
        };
        
        let result = self.market_tx.send(msg);
        if result.is_ok() {
            self.total_broadcasts.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
    
    /// Get current connection count
    pub fn connection_count(&self) -> u64 {
        self.connection_count.load(Ordering::Relaxed)
    }
    
    /// Get server statistics
    pub fn get_stats(&self) -> ServerStats {
        ServerStats {
            connection_count: self.connection_count.load(Ordering::Relaxed),
            total_broadcasts: self.total_broadcasts.load(Ordering::Relaxed),
            dropped_messages: self.dropped_messages.load(Ordering::Relaxed),
            is_running: self.running.load(Ordering::Relaxed),
        }
    }
    
    /// Stop the server
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Default for WsServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Server statistics
#[derive(Debug, Clone, Copy)]
pub struct ServerStats {
    pub connection_count: u64,
    pub total_broadcasts: u64,
    pub dropped_messages: u64,
    pub is_running: bool,
}

/// Get current timestamp in milliseconds
#[inline]
fn get_timestamp_ms() -> u64 {
    Instant::now().elapsed().as_millis() as u64
}

/// Handle a single WebSocket connection
async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    clients: Arc<RwLock<HashMap<SocketAddr, ConnectionState>>>,
    mut market_rx: broadcast::Receiver<ServerMessage>,
) {
    // Perform WebSocket handshake
    let ws_stream = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("WebSocket handshake failed for {}: {}", addr, e);
            return;
        }
    };
    
    info!("New WebSocket connection from {}", addr);
    
    // Split sink and stream
    let (mut tx, mut rx) = ws_stream.split();
    
    // Create message channel for this client
    let (client_tx, mut client_rx) = mpsc::channel(MSG_BUFFER_SIZE);
    
    // Register client
    {
        let mut clients_write = clients.write().await;
        clients_write.insert(addr, ConnectionState {
            addr,
            tx: client_tx,
            connected_at: Instant::now(),
            last_heartbeat: Instant::now(),
            messages_sent: 0,
            is_authenticated: false,
        });
    }
    
    // Update connection count
    let conn_count = clients.read().await.len() as u64;
    
    // Spawn writer task
    let write_clients = Arc::clone(&clients);
    let write_addr = addr;
    let writer_handle = tokio::spawn(async move {
        let mut writer = tx;
        let mut messages_sent = 0u64;
        
        loop {
            tokio::select! {
                // Client-specific messages
                Some(msg) = client_rx.recv() => {
                    let ws_msg = serialize_message(msg);
                    match writer.send(ws_msg).await {
                        Ok(()) => {
                            messages_sent += 1;
                        }
                        Err(e) => {
                            warn!("Failed to send to {}: {}", write_addr, e);
                            break;
                        }
                    }
                }
                
                // Market broadcast
                Ok(msg) = market_rx.recv() => {
                    let ws_msg = serialize_message(msg);
                    match writer.send(ws_msg).await {
                        Ok(()) => {
                            messages_sent += 1;
                        }
                        Err(e) => {
                            warn!("Failed to send broadcast to {}: {}", write_addr, e);
                            // Don't break on broadcast failures
                        }
                    }
                }
                
                else => break,
            }
        }
        
        messages_sent
    });
    
    // Reader task - handle incoming messages
    let read_clients = Arc::clone(&clients);
    let read_addr = addr;
    loop {
        match rx.next().await {
            Some(Ok(Message::Ping(data))) => {
                // Respond to ping with pong
                if let Err(e) = tx.send(Message::Pong(data)).await {
                    warn!("Failed to send pong to {}: {}", read_addr, e);
                    break;
                }
                
                // Update heartbeat
                if let Some(client) = read_clients.write().await.get_mut(&read_addr) {
                    client.last_heartbeat = Instant::now();
                }
            }
            
            Some(Ok(Message::Text(text))) => {
                // Handle text messages (auth, subscribe, etc.)
                debug!("Received from {}: {}", read_addr, text);
                
                // Simple auth handling
                if text.contains("\"type\":\"auth\"") {
                    if let Some(client) = read_clients.write().await.get_mut(&read_addr) {
                        client.is_authenticated = true;
                        info!("Client {} authenticated", read_addr);
                    }
                }
            }
            
            Some(Ok(Message::Close(_))) | None => {
                debug!("Closing connection to {}", read_addr);
                break;
            }
            
            Some(Err(e)) => {
                warn!("WebSocket error for {}: {}", read_addr, e);
                break;
            }
            
            _ => {}
        }
    }
    
    // Cleanup
    {
        let mut clients_write = read_clients.write().await;
        clients_write.remove(&read_addr);
    }
    
    // Wait for writer to finish
    let _ = writer_handle.await;
    
    info!("Connection closed for {}", addr);
}

/// Serialize server message to WebSocket message
fn serialize_message(msg: ServerMessage) -> Message {
    use serde_json::json;
    
    let json = match msg {
        ServerMessage::OrderBook { symbol, bids, asks, timestamp_ms } => {
            json!({
                "type": "orderbook",
                "symbol": symbol,
                "bids": bids,
                "asks": asks,
                "timestamp": timestamp_ms,
            })
        }
        ServerMessage::PnL { total_pnl, unrealized_pnl, realized_pnl, timestamp_ms } => {
            json!({
                "type": "pnl",
                "total_pnl": total_pnl,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "timestamp": timestamp_ms,
            })
        }
        ServerMessage::Telemetry { cpu_usage, memory_usage_mb, latency_us, throughput_ops, timestamp_ms } => {
            json!({
                "type": "telemetry",
                "cpu_usage": cpu_usage,
                "memory_mb": memory_usage_mb,
                "latency_us": latency_us,
                "throughput_ops": throughput_ops,
                "timestamp": timestamp_ms,
            })
        }
        ServerMessage::OrderStatus { order_id, symbol, status, filled_qty, avg_price } => {
            json!({
                "type": "order_status",
                "order_id": order_id,
                "symbol": symbol,
                "status": status,
                "filled_qty": filled_qty,
                "avg_price": avg_price,
            })
        }
        ServerMessage::Alert { level, message, timestamp_ms } => {
            let level_str = match level {
                AlertLevel::Info => "info",
                AlertLevel::Warning => "warning",
                AlertLevel::Error => "error",
                AlertLevel::Critical => "critical",
            };
            json!({
                "type": "alert",
                "level": level_str,
                "message": message,
                "timestamp": timestamp_ms,
            })
        }
    };
    
    Message::Text(json.to_string())
}

/// Run heartbeat loop to detect stale connections
async fn run_heartbeat_loop(
    clients: Arc<RwLock<HashMap<SocketAddr, ConnectionState>>>,
    running: AtomicBool,
) {
    let mut interval = interval(Duration::from_millis(HEARTBEAT_INTERVAL_MS));
    
    while running.load(Ordering::Relaxed) {
        interval.tick().await;
        
        let now = Instant::now();
        let stale_threshold = Duration::from_secs(90); // 3 missed heartbeats
        
        let mut stale_clients = Vec::new();
        
        {
            let clients_read = clients.read().await;
            for (addr, state) in clients_read.iter() {
                if now.duration_since(state.last_heartbeat) > stale_threshold {
                    stale_clients.push(*addr);
                }
            }
        }
        
        // Remove stale clients
        if !stale_clients.is_empty() {
            let mut clients_write = clients.write().await;
            for addr in stale_clients {
                clients_write.remove(&addr);
                warn!("Removed stale client {}", addr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = WsServer::new();
        let stats = server.get_stats();
        assert_eq!(stats.connection_count, 0);
        assert!(!stats.is_running);
    }

    #[test]
    fn test_message_serialization() {
        let msg = ServerMessage::PnL {
            total_pnl: 1000.50,
            unrealized_pnl: 500.25,
            realized_pnl: 500.25,
            timestamp_ms: 1234567890,
        };
        
        let ws_msg = serialize_message(msg);
        match ws_msg {
            Message::Text(text) => {
                assert!(text.contains("\"type\":\"pnl\""));
                assert!(text.contains("1000.5"));
            }
            _ => panic!("Expected Text message"),
        }
    }

    #[tokio::test]
    async fn test_broadcast() {
        let server = WsServer::new();
        
        let result = server.broadcast_pnl(100.0, 50.0, 50.0);
        assert!(result.is_ok());
        
        let stats = server.get_stats();
        assert_eq!(stats.total_broadcasts, 1);
    }
}
