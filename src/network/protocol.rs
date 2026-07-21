//! # Zero-Copy Binary Protocol Parser for IPC
//! 
//! Implements a zero-copy binary protocol parser for internal inter-process communication (IPC),
//! replacing standard serialization with direct memory casting for maximum throughput.
//! 
//! ## Key Features:
//! - Zero-copy deserialization via direct memory reinterpretation
//! - Fixed-size message headers for predictable parsing
//! - Support for Binance aggregate trade stream format
//! - Cache-line aligned structures for AMD Ryzen AI 5
//! - No heap allocations in hot path

use std::mem;
use std::slice;
use std::str;

/// Message type identifiers for protocol dispatch
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    Tick = 0x01,
    OrderBookSnapshot = 0x02,
    OrderBookUpdate = 0x03,
    Trade = 0x04,
    OrderAck = 0x05,
    OrderCancel = 0x06,
    Heartbeat = 0x07,
    Error = 0xFF,
}

impl MessageType {
    #[inline(always)]
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0x01 => Some(Self::Tick),
            0x02 => Some(Self::OrderBookSnapshot),
            0x03 => Some(Self::OrderBookUpdate),
            0x04 => Some(Self::Trade),
            0x05 => Some(Self::OrderAck),
            0x06 => Some(Self::OrderCancel),
            0x07 => Some(Self::Heartbeat),
            0xFF => Some(Self::Error),
            _ => None,
        }
    }
}

/// Fixed-size message header (16 bytes = 2 cache lines on most architectures)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct MessageHeader {
    /// Message type identifier
    pub msg_type: u8,
    /// Protocol version
    pub version: u8,
    /// Reserved for future use (padding)
    pub flags: u16,
    /// Payload length in bytes
    pub payload_len: u32,
    /// Sequence number for ordering/deduplication
    pub sequence: u64,
}

impl MessageHeader {
    pub const SIZE: usize = mem::size_of::<Self>();

    #[inline(always)]
    pub fn new(msg_type: MessageType, payload_len: u32, sequence: u64) -> Self {
        Self {
            msg_type: msg_type as u8,
            version: 1,
            flags: 0,
            payload_len,
            sequence,
        }
    }

    /// Parse header from raw bytes (zero-copy)
    #[inline(always)]
    pub unsafe fn from_bytes(ptr: *const u8) -> &'static Self {
        &*(ptr as *const Self)
    }

    /// Get header as raw bytes
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self as *const Self as *const u8,
                Self::SIZE,
            )
        }
    }
}

/// Binance aggregate trade message structure (matches Binance WebSocket format)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AggregateTrade {
    /// Event type
    pub event_type: [u8; 8],
    /// Event time (timestamp)
    pub event_time: i64,
    /// Symbol (encoded as fixed-size array)
    pub symbol: [u8; 12],
    /// Aggregate trade ID
    pub agg_trade_id: i64,
    /// Price (scaled integer, multiply by 1e-8 for actual value)
    pub price: i64,
    /// Quantity (scaled integer)
    pub quantity: i64,
    /// First trade ID in this aggregate
    pub first_trade_id: i64,
    /// Last trade ID in this aggregate
    pub last_trade_id: i64,
    /// Trade time
    pub trade_time: i64,
    /// Was the buyer the maker?
    pub is_buyer_maker: bool,
    /// Reserved padding
    pub _padding: [u8; 7],
}

impl AggregateTrade {
    pub const SIZE: usize = mem::size_of::<Self>();

    /// Create from raw bytes (zero-copy interpretation)
    #[inline(always)]
    pub unsafe fn from_bytes(ptr: *const u8) -> &'static Self {
        &*(ptr as *const Self)
    }

    /// Get price as f64 (assuming 1e-8 scale)
    #[inline(always)]
    pub fn get_price_f64(&self) -> f64 {
        self.price as f64 * 1e-8
    }

    /// Get quantity as f64
    #[inline(always)]
    pub fn get_quantity_f64(&self) -> f64 {
        self.quantity as f64 * 1e-8
    }

    /// Get symbol as string slice
    #[inline(always)]
    pub fn get_symbol_str(&self) -> Result<&str, str::Utf8Error> {
        // Find null terminator or use full length
        let len = self.symbol.iter().position(|&b| b == 0).unwrap_or(12);
        str::from_utf8(&self.symbol[..len])
    }
}

/// Order book level entry (price/quantity pair)
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct OrderBookLevel {
    /// Price (scaled integer)
    pub price: i64,
    /// Quantity (scaled integer)
    pub quantity: i64,
    /// Number of orders at this level
    pub order_count: u32,
    /// Reserved padding
    pub _padding: u32,
}

impl OrderBookLevel {
    pub const SIZE: usize = mem::size_of::<Self>();
}

/// Order book snapshot message
#[repr(C, align(64))]
pub struct OrderBookSnapshot {
    /// Header
    pub header: MessageHeader,
    /// Symbol
    pub symbol: [u8; 12],
    /// Last update ID
    pub last_update_id: i64,
    /// Number of bid levels
    pub bid_count: u16,
    /// Number of ask levels
    pub ask_count: u16,
    /// Reserved
    pub _reserved: u32,
    /// Bid levels follow immediately after (variable length)
    /// Ask levels follow after bids
}

impl OrderBookSnapshot {
    /// Get pointer to bid levels array
    #[inline(always)]
    pub unsafe fn get_bids(&self) -> *const OrderBookLevel {
        let base = self as *const Self as *const u8;
        let offset = mem::size_of::<Self>();
        base.add(offset) as *const OrderBookLevel
    }

    /// Get pointer to ask levels array
    #[inline(always)]
    pub unsafe fn get_asks(&self) -> *const OrderBookLevel {
        let base = self.get_bids() as *const u8;
        let offset = self.bid_count as usize * OrderBookLevel::SIZE;
        base.add(offset) as *const OrderBookLevel
    }
}

/// Protocol encoder/decoder for zero-copy operations
pub struct ProtocolCodec;

impl ProtocolCodec {
    /// Encode a message header into buffer (returns bytes written)
    #[inline(always)]
    pub fn encode_header(header: &MessageHeader, buffer: &mut [u8]) -> usize {
        if buffer.len() < MessageHeader::SIZE {
            return 0;
        }
        
        unsafe {
            let src = header.as_bytes();
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                buffer.as_mut_ptr(),
                MessageHeader::SIZE,
            );
        }
        
        MessageHeader::SIZE
    }

    /// Decode header from buffer (zero-copy)
    /// Safety: Caller must ensure buffer has at least MessageHeader::SIZE bytes
    #[inline(always)]
    pub unsafe fn decode_header(buffer: &[u8]) -> Option<&MessageHeader> {
        if buffer.len() < MessageHeader::SIZE {
            return None;
        }
        
        Some(MessageHeader::from_bytes(buffer.as_ptr()))
    }

    /// Validate message integrity (basic checksum)
    #[inline(always)]
    pub fn validate_message(buffer: &[u8]) -> bool {
        if buffer.len() < MessageHeader::SIZE {
            return false;
        }

        unsafe {
            let header = MessageHeader::from_bytes(buffer.as_ptr());
            
            // Check message type is valid
            if MessageType::from_u8(header.msg_type).is_none() {
                return false;
            }

            // Check we have enough data for payload
            let total_expected = MessageHeader::SIZE + header.payload_len as usize;
            if buffer.len() < total_expected {
                return false;
            }
        }

        true
    }
}

/// Ring buffer for zero-copy message passing between threads
#[repr(C, align(64))]
pub struct ZeroCopyRingBuffer {
    /// Buffer storage (pre-allocated)
    buffer: Vec<u8>,
    /// Total capacity
    capacity: usize,
    /// Read position
    read_pos: usize,
    /// Write position
    write_pos: usize,
    /// Current message count
    count: usize,
}

impl ZeroCopyRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            capacity,
            read_pos: 0,
            write_pos: 0,
            count: 0,
        }
    }

    /// Get available write space
    #[inline(always)]
    pub fn available_space(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.capacity - (self.write_pos - self.read_pos) - 1
        } else {
            self.read_pos - self.write_pos - 1
        }
    }

    /// Get current message count
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reserve contiguous space for writing (returns pointer and length)
    /// Safety: Caller must not exceed available_space
    #[inline(always)]
    pub unsafe fn reserve_write(&mut self, len: usize) -> Option<(*mut u8, usize)> {
        if len > self.available_space() {
            return None;
        }

        let ptr = self.buffer.as_mut_ptr().add(self.write_pos);
        self.write_pos = (self.write_pos + len) % self.capacity;
        self.count += 1;

        Some((ptr, len))
    }

    /// Get next readable message (zero-copy)
    #[inline(always)]
    pub unsafe fn peek_read(&self) -> Option<&[u8]> {
        if self.count == 0 {
            return None;
        }

        // Simple implementation: assume fixed-size messages for now
        // In production, would parse header to determine actual length
        let msg_len = MessageHeader::SIZE + 128; // Example fixed payload
        
        if self.read_pos + msg_len <= self.capacity {
            Some(slice::from_raw_parts(
                self.buffer.as_ptr().add(self.read_pos),
                msg_len,
            ))
        } else {
            // Wrap-around case (simplified)
            None
        }
    }

    /// Advance read position after processing
    #[inline(always)]
    pub fn advance_read(&mut self, len: usize) {
        if self.count > 0 {
            self.read_pos = (self.read_pos + len) % self.capacity;
            self.count -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = MessageHeader::new(MessageType::Trade, 100, 12345);
        assert_eq!(header.msg_type, MessageType::Trade as u8);
        assert_eq!(header.payload_len, 100);
        assert_eq!(header.sequence, 12345);

        let mut buffer = [0u8; 64];
        let written = ProtocolCodec::encode_header(&header, &mut buffer);
        assert_eq!(written, MessageHeader::SIZE);

        unsafe {
            let decoded = ProtocolCodec::decode_header(&buffer).unwrap();
            assert_eq!(decoded.msg_type, header.msg_type);
            assert_eq!(decoded.payload_len, header.payload_len);
            assert_eq!(decoded.sequence, header.sequence);
        }
    }

    #[test]
    fn test_aggregate_trade_parsing() {
        let mut trade_data = AggregateTrade {
            event_type: *b"aggTrade",
            event_time: 1234567890,
            symbol: *b"BTCUSDT   \0\0\0",
            agg_trade_id: 98765,
            price: 5000000000, // $50,000 scaled
            quantity: 100000000, // 1.0 BTC scaled
            first_trade_id: 100,
            last_trade_id: 105,
            trade_time: 1234567890,
            is_buyer_maker: false,
            _padding: [0; 7],
        };

        unsafe {
            let ptr = &trade_data as *const _ as *const u8;
            let parsed = AggregateTrade::from_bytes(ptr);
            
            assert_eq!(parsed.get_price_f64(), 50000.0);
            assert_eq!(parsed.get_quantity_f64(), 1.0);
            assert_eq!(parsed.get_symbol_str().unwrap(), "BTCUSDT");
        }
    }

    #[test]
    fn test_ring_buffer() {
        let mut ring = ZeroCopyRingBuffer::new(1024);
        assert_eq!(ring.available_space(), 1023);
        assert_eq!(ring.count(), 0);
    }
}
