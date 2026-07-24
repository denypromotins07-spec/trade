//! port_allocator.rs - Dynamic Port Allocator for Rust Gateway
//! Stage 54: Nautilus/Ray Crypto Trading Bot
//! Finds first available local TCP port, writes to shared memory file,
//! serves port to PowerShell Chrome launcher

use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::Arc;
use std::time::Duration;
use log::{debug, error, info, warn};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Default minimum port for allocation
const DEFAULT_MIN_PORT: u16 = 8000;

/// Default maximum port for allocation
const DEFAULT_MAX_PORT: u16 = 9000;

/// Reserved ports that should not be used
const RESERVED_PORTS: &[u16] = &[
    8000,  // Often used by alternative HTTP
    8080,  // Common HTTP alternate
    8443,  // HTTPS alternate
    9000,  // Often used by dev servers
];

/// Shared memory file name for port communication
const PORT_FILE_NAME: &str = "gateway_port.txt";

/// Port allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortAllocation {
    /// Allocated port number
    pub port: u16,
    /// Timestamp of allocation
    pub allocated_at: u64,
    /// Process ID that owns this allocation
    pub pid: u32,
    /// Whether port is currently in use
    pub is_active: bool,
}

impl PortAllocation {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            allocated_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs(),
            pid: std::process::id(),
            is_active: true,
        }
    }
}

/// Dynamic port allocator with shared memory communication
pub struct PortAllocator {
    /// Minimum port in allocation range
    min_port: u16,
    /// Maximum port in allocation range
    max_port: u16,
    /// Currently allocated port
    allocated_port: Arc<RwLock<Option<PortAllocation>>>,
    /// Path to shared memory directory
    shared_dir: PathBuf,
    /// Flag indicating allocator is running
    is_running: Arc<AtomicBool>,
    /// Health check port
    health_port: AtomicU16,
}

impl PortAllocator {
    /// Create a new port allocator with default range
    pub fn new() -> Result<Self, String> {
        Self::with_range(DEFAULT_MIN_PORT, DEFAULT_MAX_PORT)
    }

    /// Create a port allocator with custom range
    pub fn with_range(min_port: u16, max_port: u16) -> Result<Self, String> {
        if min_port >= max_port {
            return Err("min_port must be less than max_port".to_string());
        }

        // Determine shared directory path
        let shared_dir = Self::find_shared_directory()?;

        Ok(Self {
            min_port,
            max_port,
            allocated_port: Arc::new(RwLock::new(None)),
            shared_dir,
            is_running: Arc::new(AtomicBool::new(false)),
            health_port: AtomicU16::new(0),
        })
    }

    /// Find or create shared memory directory
    fn find_shared_directory() -> Result<PathBuf, String> {
        // Try common locations for shared memory
        let candidates = vec![
            // Relative to current executable
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("shared"),
            // User's home directory
            dirs::home_dir()
                .map(|h| h.join(".nautilus").join("shared")),
            // System temp directory
            std::env::temp_dir().join("nautilus_shared"),
        ];

        for candidate in candidates.into_iter().flatten() {
            // Try to create directory
            if fs::create_dir_all(&candidate).is_ok() {
                // Verify write access
                let test_file = candidate.join(".write_test");
                if File::create(&test_file).is_ok() {
                    let _ = fs::remove_file(test_file);
                    info!("Shared directory: {}", candidate.display());
                    return Ok(candidate);
                }
            }
        }

        Err("Could not find or create writable shared directory".to_string())
    }

    /// Find the first available port in the configured range
    pub fn find_available_port(&self) -> Result<u16, String> {
        debug!(
            "Searching for available port in range {}-{}",
            self.min_port, self.max_port
        );

        for port in self.min_port..=self.max_port {
            // Skip reserved ports
            if RESERVED_PORTS.contains(&port) {
                debug!("Skipping reserved port {}", port);
                continue;
            }

            if Self::is_port_available(port)? {
                debug!("Port {} is available", port);
                return Ok(port);
            }
        }

        Err(format!(
            "No available ports in range {}-{}",
            self.min_port, self.max_port
        ))
    }

    /// Check if a specific port is available
    pub fn is_port_available(port: u16) -> Result<bool, String> {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(_) => Ok(true),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::AddrInUse {
                    Ok(false)
                } else {
                    Err(format!("Error checking port {}: {}", port, e))
                }
            }
        }
    }

    /// Allocate a port and write it to shared memory
    pub fn allocate_and_publish(&mut self) -> Result<PortAllocation, String> {
        // Find available port
        let port = self.find_available_port()?;

        // Create allocation record
        let allocation = PortAllocation::new(port);

        // Write to shared memory file
        self.write_port_file(&allocation)?;

        // Store allocation
        *self.allocated_port.write() = Some(allocation.clone());

        info!("Allocated and published port: {}", port);
        Ok(allocation)
    }

    /// Write port to shared memory file
    fn write_port_file(&self, allocation: &PortAllocation) -> Result<(), String> {
        let port_file = self.shared_dir.join(PORT_FILE_NAME);

        // Ensure directory exists
        if let Some(parent) = port_file.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create shared dir: {}", e))?;
        }

        // Write port number as plain text (for PowerShell compatibility)
        let mut file = File::create(&port_file)
            .map_err(|e| format!("Failed to create port file: {}", e))?;

        file.write_all(allocation.port.to_string().as_bytes())
            .map_err(|e| format!("Failed to write port file: {}", e))?;

        debug!("Written port {} to {}", allocation.port, port_file.display());
        Ok(())
    }

    /// Read port from shared memory file
    pub fn read_port_file(&self) -> Result<Option<u16>, String> {
        let port_file = self.shared_dir.join(PORT_FILE_NAME);

        if !port_file.exists() {
            return Ok(None);
        }

        let mut file = File::open(&port_file)
            .map_err(|e| format!("Failed to open port file: {}", e))?;

        let mut content = String::new();
        file.read_to_string(&mut content)
            .map_err(|e| format!("Failed to read port file: {}", e))?;

        match content.trim().parse::<u16>() {
            Ok(port) => Ok(Some(port)),
            Err(_) => Err("Invalid port number in file".to_string()),
        }
    }

    /// Get the currently allocated port
    pub fn get_allocated_port(&self) -> Option<u16> {
        self.allocated_port.read().as_ref().map(|a| a.port)
    }

    /// Get the shared directory path
    pub fn get_shared_dir(&self) -> &Path {
        &self.shared_dir
    }

    /// Start health check server on allocated port
    pub fn start_health_server(&self) -> Result<(), String> {
        let port = self.get_allocated_port()
            .ok_or("No port allocated")?;

        self.health_port.store(port, Ordering::Release);
        self.is_running.store(true, Ordering::Release);

        let shared_dir = self.shared_dir.clone();
        let is_running = self.is_running.clone();

        std::thread::Builder::new()
            .name("health_server".to_string())
            .spawn(move || {
                info!("Starting health check server on port {}", port);

                let listener = match TcpListener::bind(("127.0.0.1", port)) {
                    Ok(l) => l,
                    Err(e) => {
                        error!("Failed to bind health server: {}", e);
                        return;
                    }
                };

                // Set read timeout
                let _ = listener.set_nonblocking(true);

                while is_running.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            // Handle incoming connection
                            Self::handle_health_connection(&mut stream);
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            // No connections, sleep briefly
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(e) => {
                            debug!("Health server accept error: {}", e);
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    }
                }

                info!("Health check server stopped");
            })
            .map_err(|e| format!("Failed to spawn health server: {}", e))?;

        Ok(())
    }

    /// Handle incoming health check connection
    fn handle_health_connection(stream: &mut TcpStream) {
        let mut buffer = [0u8; 1024];
        
        match stream.read(&mut buffer) {
            Ok(n) => {
                let request = String::from_utf8_lossy(&buffer[..n]);
                
                // Simple HTTP response for health check
                if request.contains("GET /health") || request.contains("GET /api") {
                    let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"healthy\"}";
                    let _ = stream.write_all(response.as_bytes());
                } else if request.contains("GET /port") {
                    // Return port information
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{{\"port\":{},\"websocket_port\":{}}}",
                        stream.local_addr().map(|a| a.port()).unwrap_or(0),
                        stream.local_addr().map(|a| a.port()).unwrap_or(0)
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            }
            Err(e) => {
                debug!("Health connection read error: {}", e);
            }
        }
        
        // Flush and let connection close
        let _ = stream.flush();
    }

    /// Stop the health server
    pub fn stop_health_server(&self) {
        self.is_running.store(false, Ordering::Release);
        info!("Health server stop signal sent");
    }

    /// Clean up port file on shutdown
    pub fn cleanup(&self) {
        self.stop_health_server();
        
        let port_file = self.shared_dir.join(PORT_FILE_NAME);
        if port_file.exists() {
            let _ = fs::remove_file(&port_file);
            debug!("Cleaned up port file");
        }

        *self.allocated_port.write() = None;
    }

    /// Get allocator statistics
    pub fn get_stats(&self) -> PortAllocatorStats {
        PortAllocatorStats {
            min_port: self.min_port,
            max_port: self.max_port,
            allocated_port: self.get_allocated_port(),
            shared_dir: self.shared_dir.clone(),
            is_running: self.is_running.load(Ordering::Relaxed),
        }
    }
}

/// Statistics about the port allocator
#[derive(Debug, Clone)]
pub struct PortAllocatorStats {
    pub min_port: u16,
    pub max_port: u16,
    pub allocated_port: Option<u16>,
    pub shared_dir: PathBuf,
    pub is_running: bool,
}

impl Default for PortAllocator {
    fn default() -> Self {
        Self::new().expect("Failed to create default PortAllocator")
    }
}

impl Drop for PortAllocator {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_port_allocator_creation() {
        let allocator = PortAllocator::new();
        assert!(allocator.is_ok());
    }

    #[test]
    fn test_find_available_port() {
        let allocator = PortAllocator::with_range(10000, 10100).unwrap();
        let port = allocator.find_available_port();
        assert!(port.is_ok());
        
        let port = port.unwrap();
        assert!(port >= 10000 && port <= 10100);
        assert!(!RESERVED_PORTS.contains(&port));
    }

    #[test]
    fn test_is_port_available() {
        // Find a definitely available port
        let port = PortAllocator::find_available_port(20000, 20100).unwrap();
        assert!(PortAllocator::is_port_available(port).unwrap());
    }

    #[test]
    fn test_allocate_and_publish() {
        let mut allocator = PortAllocator::with_range(11000, 11100).unwrap();
        let allocation = allocator.allocate_and_publish();
        
        assert!(allocation.is_ok());
        let alloc = allocation.unwrap();
        assert!(alloc.port >= 11000 && alloc.port <= 11100);
        assert!(alloc.pid > 0);
        
        // Verify file was written
        let read_port = allocator.read_port_file();
        assert!(read_port.is_ok());
        assert_eq!(read_port.unwrap(), Some(alloc.port));
    }

    /// Helper function to find an available port in a range
    fn find_available_port(min: u16, max: u16) -> Result<u16, String> {
        for port in min..=max {
            if PortAllocator::is_port_available(port)? {
                return Ok(port);
            }
        }
        Err("No available ports".to_string())
    }
}
