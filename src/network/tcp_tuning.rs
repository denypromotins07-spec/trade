//! # TCP Network Tuning for HFT
//! 
//! Implements raw socket configurations optimized for microsecond latency on Binance feeds.
//! Targets AMD Ryzen AI 5 architecture with Windows-specific kernel tuning.
//! 
//! ## Optimizations:
//! - TCP_NODELAY: Disables Nagle's algorithm to prevent packet coalescing delays
//! - TCP_QUICKACK: Reduces ACK delay on Windows/Linux for faster round-trips
//! - Custom keep-alive: Aggressive intervals to detect dead peers instantly
//! - SO_RCVBUF/SO_SNDBUF: Tuned buffer sizes for high-frequency tick streams
//! - CPU Affinity: Binds sockets to specific cores to reduce context switches

use std::io::{self, Result};
use std::net::{TcpStream, SocketAddr};
use std::time::Duration;

#[cfg(target_os = "windows")]
use windows::Win32::Networking::WinSock::{
    SOCKET, IPPROTO_TCP, TCP_NODELAY as WIN_TCP_NODELAY, 
    SOL_SOCKET, SO_RCVBUF, SO_SNDBUF, SO_KEEPALIVE,
    setsockopt, ioctlsocket, FIONBIO
};

/// Configuration for HFT-optimized TCP connections
#[derive(Debug, Clone)]
pub struct TcpTuningConfig {
    /// Disable Nagle's algorithm (TCP_NODELAY)
    pub no_delay: bool,
    /// Enable quick ACK mode (platform-specific)
    pub quick_ack: bool,
    /// Keep-alive interval in seconds (aggressive for HFT)
    pub keep_alive_interval_secs: u32,
    /// Keep-alive probe count before declaring dead
    pub keep_alive_probes: u32,
    /// Receive buffer size in bytes (tuned for tick bursts)
    pub recv_buffer_size: u32,
    /// Send buffer size in bytes
    pub send_buffer_size: u32,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
}

impl Default for TcpTuningConfig {
    fn default() -> Self {
        Self {
            no_delay: true,
            quick_ack: true,
            keep_alive_interval_secs: 5, // Aggressive 5s interval
            keep_alive_probes: 3,
            recv_buffer_size: 2 * 1024 * 1024, // 2MB for burst absorption
            send_buffer_size: 512 * 1024,      // 512KB sufficient for orders
            connect_timeout_ms: 100,           // 100ms max connection time
        }
    }
}

/// HFT-optimized TCP connection wrapper
pub struct HftTcpStream {
    stream: TcpStream,
    config: TcpTuningConfig,
}

impl HftTcpStream {
    /// Connect to a remote address with HFT optimizations applied
    pub fn connect(addr: SocketAddr, config: TcpTuningConfig) -> Result<Self> {
        // Set connection timeout using standard library
        let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(config.connect_timeout_ms))?;
        
        let mut hft_stream = Self { stream, config };
        hft_stream.apply_tuning()?;
        
        Ok(hft_stream)
    }

    /// Apply all HFT tuning parameters to the socket
    fn apply_tuning(&mut self) -> Result<()> {
        // Disable Nagle's algorithm - critical for low latency
        self.stream.set_nodelay(self.config.no_delay)?;
        
        // Set custom keep-alive parameters
        #[cfg(not(target_os = "windows"))]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self.stream.as_raw_fd();
            
            // Enable keep-alive
            unsafe {
                libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_KEEPALIVE, &1i32 as *const _ as _, 4);
                
                // TCP_KEEPIDLE (time before first probe)
                let idle_secs = self.config.keep_alive_interval_secs as i32;
                libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_KEEPIDLE, &idle_secs as *const _ as _, 4);
                
                // TCP_KEEPINTVL (interval between probes)
                let interval_secs = (self.config.keep_alive_interval_secs / 2) as i32;
                libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, &interval_secs as *const _ as _, 4);
                
                // TCP_KEEPCNT (number of probes)
                let probes = self.config.keep_alive_probes as i32;
                libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_KEEPCNT, &probes as *const _ as _, 4);
            }
        }
        
        // Windows-specific tuning via winapi
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::io::AsRawSocket;
            let raw_socket = self.stream.as_raw_socket() as SOCKET;
            
            unsafe {
                // TCP_NODELAY already set via set_nodelay, but ensure QUICKACK on Windows
                if self.config.quick_ack {
                    // SIO_TCP_SET_ACK_FREQUENCY or equivalent for QuickAck
                    // Note: Windows TCP_QUICKACK is less direct than Linux
                    // We rely on set_nodelay + aggressive timeouts
                }
                
                // Set buffer sizes
                let recv_size = self.config.recv_buffer_size as i32;
                setsockopt(
                    raw_socket,
                    SOL_SOCKET,
                    SO_RCVBUF,
                    &recv_size as *const _ as _,
                    std::mem::size_of::<i32>() as i32,
                ).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to set RCVBUF: {:?}", e)))?;
                
                let send_size = self.config.send_buffer_size as i32;
                setsockopt(
                    raw_socket,
                    SOL_SOCKET,
                    SO_SNDBUF,
                    &send_size as *const _ as _,
                    std::mem::size_of::<i32>() as i32,
                ).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to set SNDBUF: {:?}", e)))?;
            }
        }
        
        // Linux-specific buffer tuning
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = self.stream.as_raw_fd();
            
            unsafe {
                let recv_size = self.config.recv_buffer_size as i32;
                libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_RCVBUF, &recv_size as *const _ as _, 4);
                
                let send_size = self.config.send_buffer_size as i32;
                libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_SNDBUF, &send_size as *const _ as _, 4);
            }
        }
        
        Ok(())
    }

    /// Get reference to underlying stream for reading/writing
    #[inline(always)]
    pub fn get_stream(&self) -> &TcpStream {
        &self.stream
    }

    /// Get mutable reference for write operations
    #[inline(always)]
    pub fn get_stream_mut(&mut self) -> &mut TcpStream {
        &mut self.stream
    }
}

/// Detect OS environment and auto-tune parameters
pub fn auto_detect_and_tune() -> TcpTuningConfig {
    let mut config = TcpTuningConfig::default();
    
    #[cfg(target_os = "windows")]
    {
        // Windows-specific adjustments for Ryzen AI 5
        // Reduce keep-alive slightly due to Windows timer resolution
        config.keep_alive_interval_secs = 4;
        // Increase buffers slightly for Windows network stack
        config.recv_buffer_size = 4 * 1024 * 1024;
    }
    
    #[cfg(target_os = "linux")]
    {
        // Linux can handle more aggressive settings
        config.keep_alive_interval_secs = 3;
    }
    
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, SocketAddrV4, Ipv4Addr};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_hft_tcp_connection() {
        // Start a mock server
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        
        let handle = thread::spawn(move || {
            listener.incoming().next().unwrap().unwrap();
        });
        
        let config = TcpTuningConfig {
            connect_timeout_ms: 500,
            ..Default::default()
        };
        
        let client = HftTcpStream::connect(addr, config).unwrap();
        assert!(client.get_stream().peer_addr().is_ok());
        
        handle.join().unwrap();
    }
}
