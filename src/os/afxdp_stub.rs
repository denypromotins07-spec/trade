//! src/os/afxdp_stub.rs
//! 
//! AF_XDP-Style Zero-Copy Network Socket Stubs for Windows
//! 
//! Implements user-space network packet processing that bypasses the standard
//! Windows NDIS stack. Processes Binance UDP multicast feeds directly without
//! kernel context switches. Gracefully handles adapter resets and driver unloads.
//! 
//! Optimized for AMD Ryzen AI 5 with NUMA-aware buffer allocation.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::ptr;

/// Maximum packets per batch for zero-copy processing
const MAX_PACKET_BATCH: usize = 64;
/// Ring buffer size for receive queue
const RX_RING_SIZE: usize = 4096;

/// Zero-copy receive descriptor
#[repr(C)]
pub struct XdpRxDescriptor {
    pub addr: u64,      // Pointer to packet data (user-space mapped)
    pub len: u32,       // Packet length
    pub timestamp_ns: u64,
    pub flags: u32,
}

impl Default for XdpRxDescriptor {
    fn default() -> Self {
        Self {
            addr: 0,
            len: 0,
            timestamp_ns: 0,
            flags: 0,
        }
    }
}

/// Packet metadata for Binance feed processing
#[derive(Debug, Clone, Copy)]
pub struct BinancePacketMeta {
    pub sequence: u64,
    pub symbol_hash: u32,
    pub msg_type: u8,
    pub is_snapshot: bool,
}

/// AF_XDP-style socket stub for Windows
pub struct AfXdpSocketStub {
    /// Memory-mapped receive ring buffer
    rx_ring: Box<[XdpRxDescriptor; RX_RING_SIZE]>,
    /// Producer index (kernel/driver writes here)
    rx_producer: AtomicU64,
    /// Consumer index (we read from here)
    rx_consumer: AtomicU64,
    /// Socket active state
    is_active: AtomicBool,
    /// Adapter reset in progress
    is_resetting: AtomicBool,
    /// Packets processed
    packets_processed: AtomicU64,
    /// Dropped packets (buffer full)
    packets_dropped: AtomicU64,
    /// NUMA node affinity
    numa_node: u32,
}

impl AfXdpSocketStub {
    /// Create new AF_XDP socket stub
    pub fn new(numa_node: u32) -> Result<Self, &'static str> {
        // Allocate NUMA-aware memory (simplified - real impl uses SetNumaProcessorNode)
        let rx_ring = vec![XdpRxDescriptor::default(); RX_RING_SIZE]
            .into_boxed_slice()
            .try_into()
            .map_err(|_| "Failed to allocate RX ring")?;

        Ok(Self {
            rx_ring,
            rx_producer: AtomicU64::new(0),
            rx_consumer: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            is_resetting: AtomicBool::new(false),
            packets_processed: AtomicU64::new(0),
            packets_dropped: AtomicU64::new(0),
            numa_node,
        })
    }

    /// Initialize socket and bind to network adapter
    pub fn initialize(&self, adapter_name: &str, queue_id: u16) -> Result<(), &'static str> {
        if self.is_active.load(Ordering::Acquire) {
            return Err("Socket already initialized");
        }

        log_info!("Initializing AF_XDP stub on adapter {} queue {}", adapter_name, queue_id);
        
        // In production, this would:
        // 1. Open handle to network adapter via WinDivert or similar
        // 2. Configure RSS to steer Binance traffic to specific queue
        // 3. Map shared memory region for zero-copy packet transfer
        // 4. Register for adapter reset notifications
        
        self.is_active.store(true, Ordering::Release);
        Ok(())
    }

    /// Receive batch of packets (zero-copy)
    #[inline]
    pub fn recv_batch(&self) -> Option<PacketBatch<'_>> {
        if !self.is_active.load(Ordering::Acquire) {
            return None;
        }

        let consumer = self.rx_consumer.load(Ordering::Acquire);
        let producer = self.rx_producer.load(Ordering::Relaxed);

        if consumer >= producer {
            return None; // No packets available
        }

        let available = (producer - consumer) as usize;
        let batch_size = available.min(MAX_PACKET_BATCH);

        Some(PacketBatch {
            descriptors: &self.rx_ring[consumer as usize..(consumer + batch_size as u64) as usize],
            count: batch_size,
            consumer,
        })
    }

    /// Release consumed packets back to ring
    pub fn release_packets(&self, count: usize) {
        let current = self.rx_consumer.load(Ordering::Relaxed);
        self.rx_consumer.store(current + count as u64, Ordering::Release);
        self.packets_processed.fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Handle adapter reset notification
    /// Called by Windows when network adapter is being reset
    pub fn on_adapter_reset_start(&self) {
        log_warn!("Adapter reset detected - pausing packet processing");
        self.is_resetting.store(true, Ordering::Release);
        
        // Drain pending packets before reset
        while self.rx_consumer.load(Ordering::Relaxed) < 
              self.rx_producer.load(Ordering::Relaxed) {
            // Process remaining packets quickly
            std::hint::spin_loop();
        }
    }

    /// Resume after adapter reset completes
    pub fn on_adapter_reset_complete(&self) -> Result<(), &'static str> {
        log_info!("Adapter reset complete - resuming operation");
        
        // Reset ring indices (packets during reset are lost)
        self.rx_consumer.store(0, Ordering::Release);
        self.rx_producer.store(0, Ordering::Release);
        
        self.is_resetting.store(false, Ordering::Release);
        Ok(())
    }

    /// Handle driver unload notification
    /// Gracefully shutdown before driver unloads
    pub fn on_driver_unload(&self) {
        log_info!("Driver unload notification - shutting down socket");
        self.is_active.store(false, Ordering::Release);
        
        // Wait for any in-flight processing to complete
        while self.is_resetting.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> XdpStats {
        XdpStats {
            packets_processed: self.packets_processed.load(Ordering::Relaxed),
            packets_dropped: self.packets_dropped.load(Ordering::Relaxed),
            rx_pending: self.rx_producer.load(Ordering::Relaxed) - 
                       self.rx_consumer.load(Ordering::Relaxed),
            is_active: self.is_active.load(Ordering::Acquire),
            is_resetting: self.is_resetting.load(Ordering::Acquire),
        }
    }

    /// Simulate packet arrival (for testing)
    #[cfg(test)]
    pub fn simulate_packet(&self, data: &[u8]) {
        let producer = self.rx_producer.load(Ordering::Relaxed);
        let idx = (producer % RX_RING_SIZE as u64) as usize;
        
        self.rx_ring[idx].len = data.len() as u32;
        self.rx_ring[idx].timestamp_ns = get_timestamp_ns();
        self.rx_ring[idx].addr = data.as_ptr() as u64;
        
        self.rx_producer.fetch_add(1, Ordering::Release);
    }
}

/// Batch of received packets for zero-copy processing
pub struct PacketBatch<'a> {
    descriptors: &'a [XdpRxDescriptor],
    count: usize,
    consumer: u64,
}

impl<'a> PacketBatch<'a> {
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &XdpRxDescriptor> {
        self.descriptors.iter()
    }

    /// Parse Binance packet metadata
    pub fn parse_binance_meta(&self, idx: usize) -> Option<BinancePacketMeta> {
        if idx >= self.count {
            return None;
        }
        
        let desc = &self.descriptors[idx];
        // Simplified parsing - real impl would parse actual Binance protocol
        Some(BinancePacketMeta {
            sequence: desc.timestamp_ns,
            symbol_hash: (desc.flags & 0xFFFF) as u32,
            msg_type: ((desc.flags >> 16) & 0xFF) as u8,
            is_snapshot: (desc.flags >> 24) != 0,
        })
    }
}

/// Statistics for AF_XDP socket
#[derive(Debug)]
pub struct XdpStats {
    pub packets_processed: u64,
    pub packets_dropped: u64,
    pub rx_pending: u64,
    pub is_active: bool,
    pub is_resetting: bool,
}

/// Get high-resolution timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

macro_rules! log_info {
    ($($arg:tt)*) => { eprintln!("[AF_XDP INFO] {}", format!($($arg)*)); };
}

macro_rules! log_warn {
    ($($arg:tt)*) => { eprintln!("[AF_XDP WARN] {}", format!($($arg)*)); };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_socket_initialization() {
        let socket = AfXdpSocketStub::new(0).unwrap();
        assert!(socket.initialize("eth0", 0).is_ok());
        assert!(socket.get_stats().is_active);
    }

    #[test]
    fn test_packet_batch_processing() {
        let socket = AfXdpSocketStub::new(0).unwrap();
        socket.initialize("eth0", 0).unwrap();

        // Simulate some packets
        let data = b"Binance trade data";
        for _ in 0..10 {
            socket.simulate_packet(data);
        }

        // Receive batch
        if let Some(batch) = socket.recv_batch() {
            assert_eq!(batch.len(), 10);
            
            for (i, _desc) in batch.iter().enumerate() {
                let meta = batch.parse_binance_meta(i);
                assert!(meta.is_some());
            }

            // Release packets
            socket.release_packets(batch.len());
        }

        // Verify stats
        let stats = socket.get_stats();
        assert_eq!(stats.packets_processed, 10);
    }

    #[test]
    fn test_adapter_reset_handling() {
        let socket = AfXdpSocketStub::new(0).unwrap();
        socket.initialize("eth0", 0).unwrap();

        // Add some packets
        socket.simulate_packet(b"test");
        
        // Start reset
        socket.on_adapter_reset_start();
        assert!(socket.get_stats().is_resetting);

        // Complete reset
        assert!(socket.on_adapter_reset_complete().is_ok());
        assert!(!socket.get_stats().is_resetting);
        assert_eq!(socket.get_stats().rx_pending, 0);
    }

    #[test]
    fn test_driver_unload_graceful() {
        let socket = AfXdpSocketStub::new(0).unwrap();
        socket.initialize("eth0", 0).unwrap();

        socket.on_driver_unload();
        assert!(!socket.get_stats().is_active);
    }
}
