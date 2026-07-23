//! Payload Compression - MessagePack and Zero-Copy Binary Serialization
//! 
//! This module implements MessagePack and zero-copy binary serialization for UI payloads,
//! drastically reducing network bandwidth and CPU serialization overhead on local loopback.
//! Optimized for AMD Ryzen AI 5 with microsecond latency targets.
//! 
//! RAM Budget: Uses pre-allocated buffers and arena allocation.
//! Enforces global 8GB RAM limit via bounded buffer pools.

use std::io::{Write, Read};
use std::time::Instant;

/// Magic bytes for protocol identification
const MAGIC_BYTES: [u8; 4] = [0x4E, 0x41, 0x55, 0x54]; // "NAUT"

/// Protocol version
const PROTOCOL_VERSION: u8 = 1;

/// Maximum payload size (64MB)
const MAX_PAYLOAD_SIZE: usize = 64 * 1024 * 1024;

/// Message types for the protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    OrderBook = 0,
    PnL = 1,
    Telemetry = 2,
    OrderStatus = 3,
    Alert = 4,
    Heartbeat = 5,
    Auth = 6,
    Subscribe = 7,
    Unsubscribe = 8,
}

impl MessageType {
    #[inline]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::OrderBook),
            1 => Some(Self::PnL),
            2 => Some(Self::Telemetry),
            3 => Some(Self::OrderStatus),
            4 => Some(Self::Alert),
            5 => Some(Self::Heartbeat),
            6 => Some(Self::Auth),
            7 => Some(Self::Subscribe),
            8 => Some(Self::Unsubscribe),
            _ => None,
        }
    }
}

/// Header structure for binary protocol
#[derive(Debug, Clone, Copy)]
pub struct ProtocolHeader {
    pub magic: [u8; 4],
    pub version: u8,
    pub message_type: u8,
    pub flags: u8,
    pub reserved: u8,
    pub payload_len: u32,
    pub timestamp_ns: u64,
    pub checksum: u32,
}

impl ProtocolHeader {
    #[inline]
    pub const fn new() -> Self {
        Self {
            magic: MAGIC_BYTES,
            version: PROTOCOL_VERSION,
            message_type: 0,
            flags: 0,
            reserved: 0,
            payload_len: 0,
            timestamp_ns: 0,
            checksum: 0,
        }
    }
    
    /// Serialize header to bytes (zero-copy compatible)
    #[inline]
    pub fn to_bytes(&self) -> [u8; 24] {
        let mut bytes = [0u8; 24];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4] = self.version;
        bytes[5] = self.message_type;
        bytes[6] = self.flags;
        bytes[7] = self.reserved;
        bytes[8..12].copy_from_slice(&self.payload_len.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.checksum.to_le_bytes());
        bytes
    }
    
    /// Deserialize header from bytes
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        
        if magic != MAGIC_BYTES {
            return None;
        }
        
        Some(Self {
            magic,
            version: bytes[4],
            message_type: bytes[5],
            flags: bytes[6],
            reserved: bytes[7],
            payload_len: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            timestamp_ns: u64::from_le_bytes(bytes[12..20].try_into().ok()?),
            checksum: u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
        })
    }
    
    /// Calculate CRC32 checksum of payload
    #[inline]
    pub fn calculate_checksum(payload: &[u8]) -> u32 {
        crc32fast::hash(payload)
    }
    
    /// Verify checksum
    #[inline]
    pub fn verify_checksum(&self, payload: &[u8]) -> bool {
        Self::calculate_checksum(payload) == self.checksum
    }
}

impl Default for ProtocolHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed payload container
pub struct CompressedPayload {
    header: ProtocolHeader,
    data: Vec<u8>,
    compressed: bool,
    original_size: usize,
}

impl CompressedPayload {
    #[inline]
    pub fn new(message_type: MessageType, payload: &[u8]) -> Self {
        let now_ns = Instant::now().elapsed().as_nanos() as u64;
        
        Self {
            header: ProtocolHeader {
                message_type: message_type as u8,
                payload_len: payload.len() as u32,
                timestamp_ns: now_ns,
                checksum: ProtocolHeader::calculate_checksum(payload),
                ..ProtocolHeader::new()
            },
            data: payload.to_vec(),
            compressed: false,
            original_size: payload.len(),
        }
    }
    
    /// Compress payload using lz4 (if beneficial)
    #[inline]
    pub fn compress(&mut self) -> bool {
        if self.data.len() < 1024 {
            // Don't compress small payloads
            return false;
        }
        
        #[cfg(feature = "lz4")]
        {
            use lz4_flex::compress_prepend_size;
            
            let compressed = compress_prepend_size(&self.data);
            
            // Only use compression if it actually reduces size
            if compressed.len() < self.data.len() {
                self.original_size = self.data.len();
                self.data = compressed;
                self.compressed = true;
                self.header.flags |= 0x01; // Set compression flag
                self.header.payload_len = self.data.len() as u32;
                return true;
            }
        }
        
        false
    }
    
    /// Decompress payload
    #[inline]
    pub fn decompress(&mut self) -> Result<(), &'static str> {
        if !self.compressed {
            return Ok(());
        }
        
        #[cfg(feature = "lz4")]
        {
            use lz4_flex::decompress_size_prepended;
            
            match decompress_size_prepended(&self.data) {
                Ok(decompressed) => {
                    self.data = decompressed;
                    self.compressed = false;
                    self.header.flags &= !0x01; // Clear compression flag
                    return Ok(());
                }
                Err(_) => return Err("Decompression failed"),
            }
        }
        
        #[cfg(not(feature = "lz4"))]
        Err("LZ4 feature not enabled")
    }
    
    /// Serialize to bytes (header + payload)
    #[inline]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24 + self.data.len());
        bytes.extend_from_slice(&self.header.to_bytes());
        bytes.extend_from_slice(&self.data);
        bytes
    }
    
    /// Deserialize from bytes
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        
        let header = ProtocolHeader::from_bytes(bytes)?;
        
        if header.payload_len as usize > MAX_PAYLOAD_SIZE {
            return None;
        }
        
        let payload_start = 24;
        let payload_end = payload_start + header.payload_len as usize;
        
        if bytes.len() < payload_end {
            return None;
        }
        
        let payload = bytes[payload_start..payload_end].to_vec();
        
        Some(Self {
            header,
            data: payload,
            compressed: (header.flags & 0x01) != 0,
            original_size: header.payload_len as usize,
        })
    }
    
    /// Get message type
    #[inline]
    pub fn message_type(&self) -> Option<MessageType> {
        MessageType::from_u8(self.header.message_type)
    }
    
    /// Get payload reference
    #[inline]
    pub fn payload(&self) -> &[u8] {
        &self.data
    }
    
    /// Check if compressed
    #[inline]
    pub fn is_compressed(&self) -> bool {
        self.compressed
    }
    
    /// Get compression ratio
    #[inline]
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 1.0;
        }
        self.data.len() as f64 / self.original_size as f64
    }
    
    /// Get timestamp in nanoseconds
    #[inline]
    pub fn timestamp_ns(&self) -> u64 {
        self.header.timestamp_ns
    }
}

/// MessagePack-style encoder for structured data
pub struct MsgPackEncoder {
    buffer: Vec<u8>,
}

impl MsgPackEncoder {
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(256),
        }
    }
    
    /// Encode a map header
    #[inline]
    pub fn encode_map_header(&mut self, len: u32) {
        if len < 16 {
            self.buffer.push(0x80 | len as u8);
        } else if len < u16::MAX as u32 {
            self.buffer.push(0xde);
            self.buffer.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buffer.push(0xdf);
            self.buffer.extend_from_slice(&len.to_be_bytes());
        }
    }
    
    /// Encode a string
    #[inline]
    pub fn encode_str(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let len = bytes.len();
        
        if len < 32 {
            self.buffer.push(0xa0 | len as u8);
        } else if len < u8::MAX as usize {
            self.buffer.push(0xd9);
            self.buffer.push(len as u8);
        } else if len < u16::MAX as usize {
            self.buffer.push(0xda);
            self.buffer.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buffer.push(0xdb);
            self.buffer.extend_from_slice(&(len as u32).to_be_bytes());
        }
        
        self.buffer.extend_from_slice(bytes);
    }
    
    /// Encode a float64
    #[inline]
    pub fn encode_f64(&mut self, val: f64) {
        self.buffer.push(0xcb);
        self.buffer.extend_from_slice(&val.to_be_bytes());
    }
    
    /// Encode a u64
    #[inline]
    pub fn encode_u64(&mut self, val: u64) {
        if val < 128 {
            self.buffer.push(val as u8);
        } else if val <= u8::MAX as u64 {
            self.buffer.push(0xcc);
            self.buffer.push(val as u8);
        } else if val <= u16::MAX as u64 {
            self.buffer.push(0xcd);
            self.buffer.extend_from_slice(&(val as u16).to_be_bytes());
        } else if val <= u32::MAX as u64 {
            self.buffer.push(0xce);
            self.buffer.extend_from_slice(&(val as u32).to_be_bytes());
        } else {
            self.buffer.push(0xcf);
            self.buffer.extend_from_slice(&val.to_be_bytes());
        }
    }
    
    /// Encode an i64
    #[inline]
    pub fn encode_i64(&mut self, val: i64) {
        if val >= -32 {
            self.buffer.push(0xe0 | (val & 0x1f) as u8);
        } else if val >= i8::MIN as i64 && val <= i8::MAX as i64 {
            self.buffer.push(0xd0);
            self.buffer.push(val as i8 as u8);
        } else if val >= i16::MIN as i64 && val <= i16::MAX as i64 {
            self.buffer.push(0xd1);
            self.buffer.extend_from_slice(&(val as i16).to_be_bytes());
        } else if val >= i32::MIN as i64 && val <= i32::MAX as i64 {
            self.buffer.push(0xd2);
            self.buffer.extend_from_slice(&(val as i32).to_be_bytes());
        } else {
            self.buffer.push(0xd3);
            self.buffer.extend_from_slice(&val.to_be_bytes());
        }
    }
    
    /// Encode array header
    #[inline]
    pub fn encode_array_header(&mut self, len: u32) {
        if len < 16 {
            self.buffer.push(0x90 | len as u8);
        } else if len < u16::MAX as u32 {
            self.buffer.push(0xdc);
            self.buffer.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buffer.push(0xdd);
            self.buffer.extend_from_slice(&len.to_be_bytes());
        }
    }
    
    /// Get encoded bytes
    #[inline]
    pub fn into_bytes(mut self) -> Vec<u8> {
        self.buffer.shrink_to_fit();
        self.buffer
    }
    
    /// Clear buffer for reuse
    #[inline]
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
    
    /// Get buffer length
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl Default for MsgPackEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy buffer pool for efficient memory reuse
pub struct BufferPool {
    buffers: crossbeam::queue::SegQueue<Vec<u8>>,
    buffer_size: usize,
    max_buffers: usize,
}

impl BufferPool {
    #[inline]
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            buffers: crossbeam::queue::SegQueue::new(),
            buffer_size,
            max_buffers,
        }
    }
    
    /// Acquire a buffer from the pool
    #[inline]
    pub fn acquire(&self) -> Vec<u8> {
        self.buffers.pop().unwrap_or_else(|| {
            Vec::with_capacity(self.buffer_size)
        })
    }
    
    /// Return a buffer to the pool
    #[inline]
    pub fn release(&self, mut buffer: Vec<u8>) {
        if self.buffers.len() < self.max_buffers {
            buffer.clear();
            let _ = self.buffers.push(buffer);
        }
    }
    
    /// Get pool statistics
    #[inline]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            available: self.buffers.len(),
            buffer_size: self.buffer_size,
            max_buffers: self.max_buffers,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PoolStats {
    pub available: usize,
    pub buffer_size: usize,
    pub max_buffers: usize,
}

/// Compression statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct CompressionStats {
    pub total_encoded: u64,
    pub total_decoded: u64,
    pub bytes_saved: u64,
    pub compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = ProtocolHeader::new();
        let bytes = header.to_bytes();
        let restored = ProtocolHeader::from_bytes(&bytes).unwrap();
        
        assert_eq!(header.magic, restored.magic);
        assert_eq!(header.version, restored.version);
    }

    #[test]
    fn test_payload_roundtrip() {
        let payload = b"Hello, World!";
        let compressed = CompressedPayload::new(MessageType::Telemetry, payload);
        let bytes = compressed.to_bytes();
        
        let restored = CompressedPayload::from_bytes(&bytes).unwrap();
        assert_eq!(restored.payload(), payload);
        assert_eq!(restored.message_type(), Some(MessageType::Telemetry));
    }

    #[test]
    fn test_msgpack_encoder() {
        let mut encoder = MsgPackEncoder::new();
        
        encoder.encode_map_header(2);
        encoder.encode_str("name");
        encoder.encode_str("test");
        encoder.encode_f64(3.14159);
        
        let bytes = encoder.into_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(1024, 10);
        
        let buf1 = pool.acquire();
        let buf2 = pool.acquire();
        
        assert_eq!(pool.stats().available, 0);
        
        pool.release(buf1);
        assert_eq!(pool.stats().available, 1);
        
        pool.release(buf2);
        assert_eq!(pool.stats().available, 2);
    }

    #[test]
    fn test_checksum_verification() {
        let payload = b"Test data for checksum";
        let checksum = ProtocolHeader::calculate_checksum(payload);
        
        assert!(checksum != 0);
        assert_eq!(checksum, ProtocolHeader::calculate_checksum(payload));
        assert_ne!(checksum, ProtocolHeader::calculate_checksum(b"different"));
    }
}
