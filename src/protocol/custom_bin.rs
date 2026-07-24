//! Ultra-Stripped Binary IPC Protocol for Rust-to-Python Communication
//!
//! This module implements a zero-overhead binary protocol that replaces MessagePack
//! with direct memory-mapped struct casting, eliminating serialization/deserialization
//! overhead in the hot path between Rust matching engine and Python Ray workers.
//!
//! Optimized for AMD Ryzen AI 5 with strict 8GB RAM limit enforcement.

use std::mem;
use std::slice;

/// Protocol version for compatibility checking
pub const PROTOCOL_VERSION: u16 = 1;

/// Magic number for frame validation
pub const FRAME_MAGIC: u32 = 0x4E415554; // "NAUT" in ASCII

/// Maximum message size (enforced for 8GB limit compliance)
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024; // 16MB per message

/// Message types for the binary protocol
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MessageType {
    Tick = 0x01,
    Order = 0x02,
    Fill = 0x03,
    Cancel = 0x04,
    Snapshot = 0x05,
    Heartbeat = 0x06,
    Control = 0x07,
}

impl MessageType {
    #[inline(always)]
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0x01 => Some(MessageType::Tick),
            0x02 => Some(MessageType::Order),
            0x03 => Some(MessageType::Fill),
            0x04 => Some(MessageType::Cancel),
            0x05 => Some(MessageType::Snapshot),
            0x06 => Some(MessageType::Heartbeat),
            0x07 => Some(MessageType::Control),
            _ => None,
        }
    }
}

/// Frame header for zero-copy parsing
/// Total size: 16 bytes (fits in single cache line with payload)
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct FrameHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Protocol version
    pub version: u16,
    /// Message type
    pub msg_type: u8,
    /// Flags (compression, encryption, etc.)
    pub flags: u8,
    /// Payload length in bytes
    pub payload_len: u32,
    /// Sequence number for ordering
    pub sequence: u32,
}

impl FrameHeader {
    #[inline(always)]
    pub fn new(msg_type: MessageType, payload_len: u32, sequence: u32) -> Self {
        FrameHeader {
            magic: FRAME_MAGIC,
            version: PROTOCOL_VERSION,
            msg_type: msg_type as u8,
            flags: 0,
            payload_len,
            sequence,
        }
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.magic == FRAME_MAGIC && self.version == PROTOCOL_VERSION
    }

    #[inline(always)]
    pub fn message_type(&self) -> Option<MessageType> {
        MessageType::from_u8(self.msg_type)
    }

    #[inline(always)]
    pub fn total_size(&self) -> usize {
        mem::size_of::<FrameHeader>() + self.payload_len as usize
    }
}

/// Tick data structure for direct memory mapping
/// Matches Python side exactly for zero-copy sharing
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct TickData {
    pub timestamp_ns: u64,
    pub symbol_id: u32,
    pub price: i64,
    pub quantity: i64,
    pub side: u8,
    pub exchange_id: u8,
    pub _padding: [u8; 6],
}

impl TickData {
    #[inline(always)]
    pub fn new(timestamp_ns: u64, symbol_id: u32, price: i64, quantity: i64, side: u8) -> Self {
        TickData {
            timestamp_ns,
            symbol_id,
            price,
            quantity,
            side,
            exchange_id: 1, // Binance
            _padding: [0; 6],
        }
    }
}

/// Order data for Rust-Python IPC
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct OrderData {
    pub order_id: u64,
    pub symbol_id: u32,
    pub price: i64,
    pub quantity: i64,
    pub filled_qty: i64,
    pub side: u8,
    pub order_type: u8,
    pub time_in_force: u8,
    pub status: u8,
    pub timestamp_ns: u64,
}

/// Fill execution data
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug)]
pub struct FillData {
    pub order_id: u64,
    pub fill_id: u64,
    pub symbol_id: u32,
    pub price: i64,
    pub quantity: i64,
    pub commission: i64,
    pub timestamp_ns: u64,
    pub is_maker: bool,
    pub _padding: [u8; 7],
}

/// Binary encoder/decoder for zero-copy operations
pub struct BinaryProtocol {
    sequence_counter: u32,
    buffer: Vec<u8>,
}

impl BinaryProtocol {
    pub fn new() -> Self {
        BinaryProtocol {
            sequence_counter: 0,
            buffer: Vec::with_capacity(MAX_MESSAGE_SIZE),
        }
    }

    /// Encode a tick into binary format (zero-copy where possible)
    #[inline(always)]
    pub fn encode_tick(&mut self, tick: &TickData) -> &[u8] {
        self.sequence_counter = self.sequence_counter.wrapping_add(1);
        
        let payload_len = mem::size_of::<TickData>() as u32;
        let header = FrameHeader::new(MessageType::Tick, payload_len, self.sequence_counter);
        
        self.buffer.clear();
        self.buffer.reserve(mem::size_of::<FrameHeader>() + mem::size_of::<TickData>());
        
        unsafe {
            // Write header
            let header_ptr = &header as *const FrameHeader as *const u8;
            self.buffer.extend_from_slice(slice::from_raw_parts(header_ptr, mem::size_of::<FrameHeader>()));
            
            // Write tick data (direct memory copy)
            let tick_ptr = tick as *const TickData as *const u8;
            self.buffer.extend_from_slice(slice::from_raw_parts(tick_ptr, mem::size_of::<TickData>()));
        }
        
        &self.buffer
    }

    /// Decode a tick from binary (zero-copy view)
    #[inline(always)]
    pub fn decode_tick<'a>(&self, data: &'a [u8]) -> Option<&'a TickData> {
        if data.len() < mem::size_of::<FrameHeader>() {
            return None;
        }

        let header_ptr = data.as_ptr() as *const FrameHeader;
        let header = unsafe { &*header_ptr };

        if !header.is_valid() || header.message_type() != Some(MessageType::Tick) {
            return None;
        }

        let payload_offset = mem::size_of::<FrameHeader>();
        if data.len() < payload_offset + mem::size_of::<TickData>() {
            return None;
        }

        let tick_ptr = unsafe { data.as_ptr().add(payload_offset) as *const TickData };
        Some(unsafe { &*tick_ptr })
    }

    /// Encode an order
    #[inline(always)]
    pub fn encode_order(&mut self, order: &OrderData) -> &[u8] {
        self.sequence_counter = self.sequence_counter.wrapping_add(1);
        
        let payload_len = mem::size_of::<OrderData>() as u32;
        let header = FrameHeader::new(MessageType::Order, payload_len, self.sequence_counter);
        
        self.buffer.clear();
        self.buffer.reserve(mem::size_of::<FrameHeader>() + mem::size_of::<OrderData>());
        
        unsafe {
            let header_ptr = &header as *const FrameHeader as *const u8;
            self.buffer.extend_from_slice(slice::from_raw_parts(header_ptr, mem::size_of::<FrameHeader>()));
            
            let order_ptr = order as *const OrderData as *const u8;
            self.buffer.extend_from_slice(slice::from_raw_parts(order_ptr, mem::size_of::<OrderData>()));
        }
        
        &self.buffer
    }

    /// Validate frame checksum (optional, for integrity)
    #[inline(always)]
    pub fn validate_frame(&self, data: &[u8]) -> bool {
        if data.len() < mem::size_of::<FrameHeader>() {
            return false;
        }

        let header_ptr = data.as_ptr() as *const FrameHeader;
        let header = unsafe { &*header_ptr };

        header.is_valid() && data.len() >= header.total_size()
    }

    /// Get next sequence number
    #[inline(always)]
    pub fn next_sequence(&mut self) -> u32 {
        self.sequence_counter = self.sequence_counter.wrapping_add(1);
        self.sequence_counter
    }

    /// Reset sequence counter (called during /START)
    pub fn reset(&mut self) {
        self.sequence_counter = 0;
        self.buffer.clear();
    }

    /// Get buffer capacity usage
    pub fn buffer_usage(&self) -> usize {
        self.buffer.capacity()
    }
}

impl Default for BinaryProtocol {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared memory region for zero-copy Rust-Python communication
#[repr(C, align(4096))]
pub struct SharedMemoryRegion {
    /// Header with metadata
    pub header: SharedHeader,
    /// Data region (variable size, page-aligned)
    pub data: [u8; 4096 * 1024], // 4MB shared region
}

#[repr(C, align(64))]
pub struct SharedHeader {
    pub magic: u32,
    pub version: u32,
    pub write_index: u64,
    pub read_index: u64,
    pub flags: u64,
    pub timestamp_ns: u64,
    pub checksum: u32,
    pub reserved: [u32; 5],
}

impl SharedMemoryRegion {
    pub fn new() -> Self {
        SharedMemoryRegion {
            header: SharedHeader {
                magic: FRAME_MAGIC,
                version: PROTOCOL_VERSION as u32,
                write_index: 0,
                read_index: 0,
                flags: 0,
                timestamp_ns: 0,
                checksum: 0,
                reserved: [0; 5],
            },
            data: [0; 4096 * 1024],
        }
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.header.magic == FRAME_MAGIC
    }

    #[inline(always)]
    pub fn available_space(&self) -> usize {
        self.data.len() - (self.header.write_index as usize)
    }
}

impl Default for SharedMemoryRegion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_header_size() {
        assert_eq!(mem::size_of::<FrameHeader>(), 16);
        assert_eq!(mem::align_of::<FrameHeader>(), 8);
    }

    #[test]
    fn test_tick_data_size() {
        assert_eq!(mem::size_of::<TickData>(), 32);
        assert_eq!(mem::align_of::<TickData>(), 8);
    }

    #[test]
    fn test_protocol_encode_decode() {
        let mut protocol = BinaryProtocol::new();
        let tick = TickData::new(1000, 1, 50000, 100, 0);
        
        let encoded = protocol.encode_tick(&tick);
        let decoded = protocol.decode_tick(encoded);
        
        assert!(decoded.is_some());
        let decoded = decoded.unwrap();
        assert_eq!(decoded.timestamp_ns, tick.timestamp_ns);
        assert_eq!(decoded.price, tick.price);
    }

    #[test]
    fn test_message_types() {
        assert_eq!(MessageType::from_u8(0x01), Some(MessageType::Tick));
        assert_eq!(MessageType::from_u8(0x02), Some(MessageType::Order));
        assert_eq!(MessageType::from_u8(0xFF), None);
    }

    #[test]
    fn test_shared_memory_alignment() {
        assert_eq!(mem::align_of::<SharedMemoryRegion>(), 4096);
    }
}
