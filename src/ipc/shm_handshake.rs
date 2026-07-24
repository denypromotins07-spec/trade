// =============================================================================
// Nautilus/Ray Crypto Trading Bot - Stage 55
// File 4: src/ipc/shm_handshake.rs
//
// Strict shared-memory handshake protocol verifying Rust core and Python
// Ray workers have mapped the exact same memory pages
// Zero-cost abstractions optimized for AMD Ryzen AI 5 microsecond latency
// =============================================================================

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_errors_doc, clippy::module_name_repetitions)]

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{info, warn, error, debug};
use thiserror::Error;

/// Magic number for handshake validation (Nautilus protocol identifier)
const SHM_MAGIC_NUMBER: u64 = 0x4E415554494C5553; // "NAUTILUS" in hex

/// Handshake version for protocol compatibility
const HANDSHAKE_VERSION: u32 = 55; // Stage 55

/// Maximum handshake attempts before failure
const MAX_HANDSHAKE_ATTEMPTS: usize = 100;

/// Handshake timeout in microseconds
const HANDSHAKE_TIMEOUT_US: u64 = 1000;

/// Shared memory region size for handshake (bytes)
const SHM_HANDSHAKE_SIZE: usize = 4096; // One page

/// Result of handshake validation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeStatus {
    /// Handshake not yet initiated
    NotStarted,
    /// Awaiting response from peer
    AwaitingResponse,
    /// Handshake completed successfully
    Completed,
    /// Handshake failed - mismatch detected
    Failed,
}

/// Error types for shared memory handshake
#[derive(Debug, Error)]
pub enum ShmHandshakeError {
    #[error("Magic number mismatch: expected {expected}, got {actual}")]
    MagicNumberMismatch { expected: u64, actual: u64 },
    
    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },
    
    #[error("Memory mapping failed: {message}")]
    MappingFailed { message: String },
    
    #[error("Handshake timeout after {attempts} attempts")]
    Timeout { attempts: usize },
    
    #[error("Page alignment error: address {address} not aligned to {alignment}")]
    PageAlignmentError { address: usize, alignment: usize },
    
    #[error("CRC32 checksum mismatch")]
    ChecksumMismatch,
}

/// Shared memory handshake header structure
/// Layout optimized for cache-line alignment on AMD Zen 4
#[repr(C, align(64))]
pub struct ShmHandshakeHeader {
    /// Magic number for protocol identification
    pub magic: AtomicU64,
    /// Protocol version
    pub version: AtomicU32,
    /// Handshake status
    pub status: AtomicU32,
    /// Timestamp of initiation (nanoseconds since epoch)
    pub timestamp_ns: AtomicU64,
    /// Process ID of initiator
    pub initiator_pid: AtomicU32,
    /// Reserved padding for cache alignment
    pub reserved: [u8; 64 - (8 + 4 + 4 + 8 + 4)], // 36 bytes padding
}

impl ShmHandshakeHeader {
    /// Create a new handshake header
    pub fn new() -> Self {
        Self {
            magic: AtomicU64::new(0),
            version: AtomicU32::new(0),
            status: AtomicU32::new(HandshakeStatus::NotStarted as u32),
            timestamp_ns: AtomicU64::new(0),
            initiator_pid: AtomicU32::new(0),
            reserved: [0u8; 36],
        }
    }

    /// Initialize as handshake initiator (Rust core side)
    pub fn initiate(&self, pid: u32) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.magic.store(SHM_MAGIC_NUMBER, Ordering::SeqCst);
        self.version.store(HANDSHAKE_VERSION, Ordering::SeqCst);
        self.status.store(HandshakeStatus::AwaitingResponse as u32, Ordering::SeqCst);
        self.timestamp_ns.store(timestamp, Ordering::SeqCst);
        self.initiator_pid.store(pid, Ordering::SeqCst);
        
        // Memory fence to ensure all writes are visible
        std::sync::atomic::fence(Ordering::SeqCst);
        
        debug!("Handshake initiated by PID {} at {}ns", pid, timestamp);
    }

    /// Respond to handshake (Python/Ray worker side)
    pub fn respond(&self) -> Result<(), ShmHandshakeError> {
        // Verify magic number
        let magic = self.magic.load(Ordering::SeqCst);
        if magic != SHM_MAGIC_NUMBER {
            return Err(ShmHandshakeError::MagicNumberMismatch {
                expected: SHM_MAGIC_NUMBER,
                actual: magic,
            });
        }

        // Verify version
        let version = self.version.load(Ordering::SeqCst);
        if version != HANDSHAKE_VERSION {
            return Err(ShmHandshakeError::VersionMismatch {
                expected: HANDSHAKE_VERSION,
                actual: version,
            });
        }

        // Mark as completed
        self.status.store(HandshakeStatus::Completed as u32, Ordering::SeqCst);
        
        // Memory fence
        std::sync::atomic::fence(Ordering::SeqCst);
        
        debug!("Handshake response sent");
        Ok(())
    }

    /// Validate completed handshake
    pub fn validate(&self) -> Result<(), ShmHandshakeError> {
        let magic = self.magic.load(Ordering::SeqCst);
        if magic != SHM_MAGIC_NUMBER {
            return Err(ShmHandshakeError::MagicNumberMismatch {
                expected: SHM_MAGIC_NUMBER,
                actual: magic,
            });
        }

        let version = self.version.load(Ordering::SeqCst);
        if version != HANDSHAKE_VERSION {
            return Err(ShmHandshakeError::VersionMismatch {
                expected: HANDSHAKE_VERSION,
                actual: version,
            });
        }

        let status = self.status.load(Ordering::SeqCst);
        if status != HandshakeStatus::Completed as u32 {
            return Err(ShmHandshakeError::Timeout { 
                attempts: MAX_HANDSHAKE_ATTEMPTS 
            });
        }

        Ok(())
    }

    /// Get current status
    pub fn get_status(&self) -> HandshakeStatus {
        match self.status.load(Ordering::SeqCst) {
            0 => HandshakeStatus::NotStarted,
            1 => HandshakeStatus::AwaitingResponse,
            2 => HandshakeStatus::Completed,
            _ => HandshakeStatus::Failed,
        }
    }
}

impl Default for ShmHandshakeHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared memory handshake manager
pub struct ShmHandshakeManager {
    /// Header pointer (memory-mapped)
    header: Arc<ShmHandshakeHeader>,
    /// Local process ID
    local_pid: u32,
    /// Peer process ID (discovered during handshake)
    peer_pid: Option<u32>,
    /// Handshake start time for latency measurement
    start_time: Option<Instant>,
}

impl ShmHandshakeManager {
    /// Create a new handshake manager with pre-allocated shared memory
    pub fn new(shm_ptr: *mut u8) -> Result<Self, ShmHandshakeError> {
        // Verify page alignment (critical for AMD Ryzen memory controller)
        let addr = shm_ptr as usize;
        if addr % 4096 != 0 {
            return Err(ShmHandshakeError::PageAlignmentError {
                address: addr,
                alignment: 4096,
            });
        }

        // Cast to header structure
        let header_ptr = shm_ptr as *mut ShmHandshakeHeader;
        
        // Safety: Caller guarantees shm_ptr is valid and properly aligned
        let header = unsafe {
            Arc::from_raw(header_ptr)
        };

        Ok(Self {
            header,
            local_pid: std::process::id(),
            peer_pid: None,
            start_time: None,
        })
    }

    /// Initiate handshake as Rust core
    pub fn initiate_handshake(&mut self) -> Result<(), ShmHandshakeError> {
        self.start_time = Some(Instant::now());
        self.header.initiate(self.local_pid);
        info!("Rust core initiated shared memory handshake");
        Ok(())
    }

    /// Wait for handshake completion with timeout
    pub fn wait_for_completion(&self, timeout_us: u64) -> Result<Duration, ShmHandshakeError> {
        let start = Instant::now();
        let timeout = Duration::from_micros(timeout_us);
        let mut attempts = 0;

        while start.elapsed() < timeout && attempts < MAX_HANDSHAKE_ATTEMPTS {
            match self.header.get_status() {
                HandshakeStatus::Completed => {
                    let elapsed = start.elapsed();
                    info!(
                        "Shared memory handshake completed in {:.2}μs",
                        elapsed.as_micros() as f64
                    );
                    return Ok(elapsed);
                }
                HandshakeStatus::Failed => {
                    return Err(ShmHandshakeError::Timeout { attempts });
                }
                _ => {
                    attempts += 1;
                    // Busy-wait with pause instruction for AMD Zen architecture
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        std::arch::asm!("pause");
                    }
                }
            }
        }

        Err(ShmHandshakeError::Timeout { attempts })
    }

    /// Respond to handshake as Python/Ray worker
    pub fn respond_to_handshake(&self) -> Result<Duration, ShmHandshakeError> {
        let start = Instant::now();
        
        // Verify the handshake was initiated
        if self.header.get_status() != HandshakeStatus::AwaitingResponse {
            return Err(ShmHandshakeError::Timeout { attempts: 0 });
        }

        // Record peer PID
        self.peer_pid = Some(self.header.initiator_pid.load(Ordering::SeqCst));

        // Send response
        self.header.respond()?;

        let elapsed = start.elapsed();
        info!(
            "Python/Ray worker responded to handshake in {:.2}μs",
            elapsed.as_micros() as f64
        );
        Ok(elapsed)
    }

    /// Validate the handshake is complete and consistent
    pub fn validate_handshake(&self) -> Result<(), ShmHandshakeError> {
        self.header.validate()?;
        
        info!(
            "Handshake validated: Rust PID {:?} <-> Python PID {}",
            self.header.initiator_pid.load(Ordering::SeqCst),
            self.local_pid
        );
        
        Ok(())
    }

    /// Get measured handshake latency
    pub fn get_latency(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Get peer PID
    pub fn get_peer_pid(&self) -> Option<u32> {
        self.peer_pid
    }

    /// Compute CRC32 checksum of shared memory region for integrity verification
    pub fn compute_checksum(&self, data: &[u8]) -> u32 {
        // Simple CRC32 implementation optimized for x86_64
        let mut crc: u32 = 0xFFFF_FFFF;
        
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                let mask = -(crc & 1) as u32;
                crc = (crc >> 1) ^ (0xEDB88320 & mask);
            }
        }
        
        !crc
    }
}

/// Builder for shared memory handshake setup
pub struct ShmHandshakeBuilder {
    shm_size: usize,
    verify_alignment: bool,
}

impl ShmHandshakeBuilder {
    pub fn new() -> Self {
        Self {
            shm_size: SHM_HANDSHAKE_SIZE,
            verify_alignment: true,
        }
    }

    pub fn size(mut self, size: usize) -> Self {
        self.shm_size = size;
        self
    }

    pub fn verify_alignment(mut self, verify: bool) -> Self {
        self.verify_alignment = verify;
        self
    }

    pub fn build(self, shm_ptr: *mut u8) -> Result<ShmHandshakeManager, ShmHandshakeError> {
        if self.verify_alignment {
            let addr = shm_ptr as usize;
            if addr % 4096 != 0 {
                return Err(ShmHandshakeError::PageAlignmentError {
                    address: addr,
                    alignment: 4096,
                });
            }
        }
        ShmHandshakeManager::new(shm_ptr)
    }
}

impl Default for ShmHandshakeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    #[test]
    fn test_handshake_header_creation() {
        let header = ShmHandshakeHeader::new();
        assert_eq!(header.get_status(), HandshakeStatus::NotStarted);
        assert_eq!(header.magic.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_full_handshake_sequence() {
        // Allocate aligned memory for testing
        let layout = Layout::from_size_align(SHM_HANDSHAKE_SIZE, 4096).unwrap();
        let ptr = unsafe { alloc(layout) };
        
        // Initialize memory to zero
        unsafe { std::ptr::write_bytes(ptr, 0, SHM_HANDSHAKE_SIZE) };

        // Create managers
        let mut rust_side = ShmHandshakeManager::new(ptr).unwrap();
        let python_side = ShmHandshakeManager::new(ptr).unwrap();

        // Initiate from Rust side
        rust_side.initiate_handshake().unwrap();

        // Respond from Python side
        python_side.respond_to_handshake().unwrap();

        // Validate
        rust_side.validate_handshake().unwrap();
        python_side.validate_handshake().unwrap();

        // Cleanup
        unsafe { dealloc(ptr, layout) };
    }

    #[test]
    fn test_magic_number_validation() {
        let header = ShmHandshakeHeader::new();
        header.initiate(std::process::id());
        
        // Corrupt magic number
        header.magic.store(0xDEADBEEF, Ordering::SeqCst);
        
        assert!(matches!(
            header.respond(),
            Err(ShmHandshakeError::MagicNumberMismatch { .. })
        ));
    }
}
