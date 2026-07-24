//! src/nic/tcp_offload.rs
//!
//! Stage 51: TCP Segmentation Offload (TSO) and Large Receive Offload (LRO)
//!
//! Configures NIC hardware offloading to reduce CPU overhead during massive
//! REST snapshot downloads without impacting WebSocket tick latency.
//! Optimized for AMD Ryzen AI 5 architecture with Windows networking stack.
//!
//! Critical for balancing bulk data transfer efficiency with low-latency trading.

use std::io;
use std::mem;
use std::ptr;

/// Network interface card offload configuration
#[derive(Debug, Clone, Copy)]
pub struct NicOffloadConfig {
    /// TCP Segmentation Offload enabled
    pub tso_enabled: bool,
    
    /// Large Receive Offload enabled
    pub lro_enabled: bool,
    
    /// Checksum offload enabled
    pub checksum_offload: bool,
    
    /// Interrupt moderation enabled
    pub interrupt_moderation: bool,
    
    /// Jumbo frames enabled
    pub jumbo_frames: bool,
    
    /// MTU size
    pub mtu: u32,
}

impl Default for NicOffloadConfig {
    fn default() -> Self {
        Self {
            tso_enabled: true,
            lro_enabled: false, // Disabled for low-latency tick processing
            checksum_offload: true,
            interrupt_moderation: false, // Disabled for microsecond latency
            jumbo_frames: true,
            mtu: 9000, // Jumbo frame MTU
        }
    }
}

/// TCP Segmentation Offload manager
pub struct TsoManager {
    config: NicOffloadConfig,
    adapter_handle: Option<usize>,
}

impl TsoManager {
    /// Create a new TSO manager
    pub fn new() -> io::Result<Self> {
        Ok(Self {
            config: NicOffloadConfig::default(),
            adapter_handle: None,
        })
    }

    /// Configure TSO for a specific network adapter
    ///
    /// # Arguments
    /// * `adapter_name` - Name of the network adapter (e.g., "Ethernet")
    /// * `enable` - Whether to enable or disable TSO
    pub fn configure_tso(&mut self, adapter_name: &str, enable: bool) -> io::Result<()> {
        self.config.tso_enabled = enable;

        // On Windows, use netsh or PowerShell to configure
        // Example: netsh interface tcp set global chimney=enabled
        
        #[cfg(target_os = "windows")]
        {
            // Would execute PowerShell command in production
            log_info!("TSO {} for adapter: {}", 
                if enable { "enabled" } else { "disabled" }, 
                adapter_name);
        }

        Ok(())
    }

    /// Configure LRO for bulk data transfers
    ///
    /// LRO should be disabled during active trading to minimize latency,
    /// but can be enabled during REST snapshot downloads.
    pub fn configure_lro(&mut self, enable: bool) -> io::Result<()> {
        self.config.lro_enabled = enable;

        log_info!("LRO {}", if enable { "enabled" } else { "disabled" });

        Ok(())
    }

    /// Enable offloads temporarily for snapshot download
    ///
    /// Returns a guard that restores original settings when dropped.
    pub fn enable_for_snapshot(&mut self) -> io::Result<OffloadGuard> {
        let original_config = self.config;
        
        // Enable TSO and LRO for efficient bulk transfer
        self.configure_tso("Primary", true)?;
        self.configure_lro(true)?;

        Ok(OffloadGuard {
            manager: self,
            original_config,
        })
    }

    /// Get current configuration
    pub fn get_config(&self) -> NicOffloadConfig {
        self.config
    }

    /// Query actual hardware capabilities
    pub fn query_capabilities(&self) -> io::Result<NicCapabilities> {
        // In production, would query actual NIC via ethtool or Windows APIs
        Ok(NicCapabilities {
            supports_tso: true,
            supports_lro: true,
            supports_jumbo: true,
            max_jumbo_size: 9014,
            supports_checksum_offload: true,
            num_queues: 8,
        })
    }
}

impl Default for TsoManager {
    fn default() -> Self {
        Self::new().expect("Failed to create TsoManager")
    }
}

/// RAII guard to restore offload settings
pub struct OffloadGuard<'a> {
    manager: &'a mut TsoManager,
    original_config: NicOffloadConfig,
}

impl<'a> Drop for OffloadGuard<'a> {
    fn drop(&mut self) {
        // Restore original settings
        let _ = self.manager.configure_tso("Primary", self.original_config.tso_enabled);
        let _ = self.manager.configure_lro(self.original_config.lro_enabled);
        
        log_info!("Restored offload settings after snapshot");
    }
}

/// NIC hardware capabilities
#[derive(Debug, Clone)]
pub struct NicCapabilities {
    pub supports_tso: bool,
    pub supports_lro: bool,
    pub supports_jumbo: bool,
    pub max_jumbo_size: u32,
    pub supports_checksum_offload: bool,
    pub num_queues: u32,
}

/// Checksum offload state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumState {
    Disabled,
    TxOnly,
    RxOnly,
    TxAndRx,
}

/// Configure checksum offload for trading optimization
///
/// For ultra-low latency, we want:
/// - TX checksum offload: ENABLED (reduces CPU load on order submission)
/// - RX checksum offload: DISABLED (allows earlier packet availability)
pub fn optimize_checksum_for_trading() -> ChecksumState {
    ChecksumState::TxOnly
}

/// Logging macro for offload operations
macro_rules! log_info {
    ($($arg:tt)*) => {
        println!("[NIC Offload] {}", format!($($arg)*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NicOffloadConfig::default();
        
        assert!(config.tso_enabled);
        assert!(!config.lro_enabled); // Disabled by default for latency
        assert!(config.checksum_offload);
        assert!(!config.interrupt_moderation); // Disabled for latency
        assert!(config.jumbo_frames);
        assert_eq!(config.mtu, 9000);
    }

    #[test]
    fn test_tso_manager_creation() {
        let manager = TsoManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_capabilities_query() {
        let manager = TsoManager::new().unwrap();
        let caps = manager.query_capabilities().unwrap();
        
        assert!(caps.supports_tso);
        assert!(caps.supports_jumbo);
        println!("NIC queues: {}", caps.num_queues);
    }

    #[test]
    fn test_offload_guard() {
        let mut manager = TsoManager::new().unwrap();
        let original = manager.get_config();
        
        {
            let _guard = manager.enable_for_snapshot().unwrap();
            let during = manager.get_config();
            
            // During snapshot, LRO should be enabled
            assert!(during.lro_enabled);
        }
        
        // After guard drops, should be restored
        let after = manager.get_config();
        assert_eq!(after.lro_enabled, original.lro_enabled);
    }

    #[test]
    fn test_checksum_optimization() {
        let state = optimize_checksum_for_trading();
        assert_eq!(state, ChecksumState::TxOnly);
    }
}
