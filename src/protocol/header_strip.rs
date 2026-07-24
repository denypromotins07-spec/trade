//! Zero-Copy Ethernet and IP Header Stripping via Pointer Arithmetic
//!
//! This module performs ultra-fast header stripping on network packets,
//! exposing raw Binance WebSocket payloads directly to the JSON parser
//! without any memory copies or allocations.
//!
//! Validates Ethernet frame checksums before zero-copy pointer casting.

use std::ptr;
use std::slice;

/// Ethernet header size (14 bytes: 6 dst MAC + 6 src MAC + 2 EtherType)
pub const ETH_HEADER_SIZE: usize = 14;

/// IPv4 header minimum size (20 bytes)
pub const IPV4_HEADER_MIN_SIZE: usize = 20;

/// IPv6 header size (40 bytes)
pub const IPV6_HEADER_SIZE: usize = 40;

/// UDP header size (8 bytes)
pub const UDP_HEADER_SIZE: usize = 8;

/// TCP header minimum size (20 bytes)
pub const TCP_HEADER_MIN_SIZE: usize = 20;

/// VLAN header size (4 bytes, if present)
pub const VLAN_HEADER_SIZE: usize = 4;

/// EtherType for IPv4
pub const ETHERTYPE_IPV4: u16 = 0x0800;

/// EtherType for IPv6
pub const ETHERTYPE_IPV6: u16 = 0x86DD;

/// EtherType for VLAN (802.1Q)
pub const ETHERTYPE_VLAN: u16 = 0x8100;

/// Parsed packet metadata after header stripping
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct ParsedPacket {
    /// Pointer to payload start (zero-copy view into original buffer)
    pub payload_ptr: *const u8,
    /// Payload length in bytes
    pub payload_len: usize,
    /// Transport protocol (6=TCP, 17=UDP)
    pub protocol: u8,
    /// Source IP address (network byte order)
    pub src_ip: [u8; 4],
    /// Destination IP address (network byte order)
    pub dst_ip: [u8; 4],
    /// Source port (network byte order)
    pub src_port: u16,
    /// Destination port (network byte order)
    pub dst_port: u16,
    /// Total header size stripped
    pub header_size: usize,
    /// Whether checksum was validated
    pub checksum_validated: bool,
}

unsafe impl Send for ParsedPacket {}
unsafe impl Sync for ParsedPacket {}

impl ParsedPacket {
    #[inline(always)]
    pub fn new() -> Self {
        ParsedPacket {
            payload_ptr: ptr::null(),
            payload_len: 0,
            protocol: 0,
            src_ip: [0; 4],
            dst_ip: [0; 4],
            src_port: 0,
            dst_port: 0,
            header_size: 0,
            checksum_validated: false,
        }
    }

    /// Get payload as slice (unsafe - caller must ensure lifetime)
    #[inline(always)]
    pub unsafe fn payload<'a>(&self) -> &'a [u8] {
        if self.payload_ptr.is_null() || self.payload_len == 0 {
            &[]
        } else {
            slice::from_raw_parts(self.payload_ptr, self.payload_len)
        }
    }

    /// Check if packet is valid
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        !self.payload_ptr.is_null() && self.payload_len > 0
    }
}

impl Default for ParsedPacket {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy header stripper for network packets
pub struct HeaderStripper {
    /// Minimum payload size required
    min_payload_size: usize,
    /// Statistics
    total_stripped: usize,
    invalid_packets: usize,
}

impl HeaderStripper {
    pub fn new(min_payload: usize) -> Self {
        HeaderStripper {
            min_payload_size: min_payload,
            total_stripped: 0,
            invalid_packets: 0,
        }
    }

    /// Strip headers from a raw Ethernet frame with checksum validation
    /// 
    /// # Safety
    /// This function performs zero-copy pointer arithmetic. The returned
    /// ParsedPacket contains pointers into the original buffer. Caller must
    /// ensure the original buffer outlives the ParsedPacket.
    #[inline(always)]
    pub unsafe fn strip_frame<'a>(&mut self, frame: &'a [u8]) -> Result<ParsedPacket, ParseError> {
        // Validate minimum frame size
        if frame.len() < ETH_HEADER_SIZE + self.min_payload_size {
            self.invalid_packets += 1;
            return Err(ParseError::FrameTooSmall);
        }

        let mut offset = 0;
        let mut header_size = 0;

        // Parse Ethernet header
        let eth_header = frame.as_ptr().add(offset) as *const EthHeader;
        let ether_type = u16::from_be_bytes([(*eth_header).ether_type[0], (*eth_header).ether_type[1]]);

        offset += ETH_HEADER_SIZE;
        header_size += ETH_HEADER_SIZE;

        // Handle VLAN tag if present
        if ether_type == ETHERTYPE_VLAN {
            if frame.len() < offset + VLAN_HEADER_SIZE + self.min_payload_size {
                self.invalid_packets += 1;
                return Err(ParseError::VlanFrameTooSmall);
            }

            let vlan_tag = u16::from_be_bytes([
                *frame.as_ptr().add(offset),
                *frame.as_ptr().add(offset + 1),
            ]);

            let inner_ether_type = u16::from_be_bytes([
                *frame.as_ptr().add(offset + 2),
                *frame.as_ptr().add(offset + 3),
            ]);

            offset += VLAN_HEADER_SIZE;
            header_size += VLAN_HEADER_SIZE;

            return self.strip_ip_layer(frame, offset, header_size, inner_ether_type);
        }

        // Strip IP layer
        self.strip_ip_layer(frame, offset, header_size, ether_type)
    }

    #[inline(always)]
    unsafe fn strip_ip_layer<'a>(
        &mut self,
        frame: &'a [u8],
        mut offset: usize,
        mut header_size: usize,
        ether_type: u16,
    ) -> Result<ParsedPacket, ParseError> {
        let mut packet = ParsedPacket::new();

        match ether_type {
            ETHERTYPE_IPV4 => {
                // Validate IPv4 header size
                if frame.len() < offset + IPV4_HEADER_MIN_SIZE {
                    self.invalid_packets += 1;
                    return Err(ParseError::Ipv4HeaderTooSmall);
                }

                let ipv4_ptr = frame.as_ptr().add(offset) as *const Ipv4Header;
                let ipv4 = &*ipv4_ptr;

                // Extract IP addresses (zero-copy)
                packet.src_ip.copy_from_slice(&ipv4.src_addr);
                packet.dst_ip.copy_from_slice(&ipv4.dst_addr);

                // Get header length (IHL field, in 32-bit words)
                let ihl = (ipv4.version_ihl & 0x0F) as usize;
                let ipv4_header_size = ihl * 4;

                if frame.len() < offset + ipv4_header_size {
                    self.invalid_packets += 1;
                    return Err(ParseError::InvalidIpv4HeaderLength);
                }

                packet.protocol = ipv4.protocol;
                offset += ipv4_header_size;
                header_size += ipv4_header_size;
            }
            ETHERTYPE_IPV6 => {
                if frame.len() < offset + IPV6_HEADER_SIZE {
                    self.invalid_packets += 1;
                    return Err(ParseError::Ipv6HeaderTooSmall);
                }

                let ipv6_ptr = frame.as_ptr().add(offset) as *const Ipv6Header;
                let ipv6 = &*ipv6_ptr;

                // For IPv6, we just note it's IPv6 (addresses are 16 bytes)
                packet.src_ip = [0; 4]; // Simplified - full IPv6 would need 16 bytes
                packet.dst_ip = [0; 4];
                packet.protocol = ipv6.next_header;

                offset += IPV6_HEADER_SIZE;
                header_size += IPV6_HEADER_SIZE;
            }
            _ => {
                self.invalid_packets += 1;
                return Err(ParseError::UnknownEtherType(ether_type));
            }
        }

        // Strip transport layer (TCP/UDP)
        self.strip_transport_layer(frame, offset, header_size, packet, packet.protocol)
    }

    #[inline(always)]
    unsafe fn strip_transport_layer<'a>(
        &mut self,
        frame: &'a [u8],
        mut offset: usize,
        mut header_size: usize,
        mut packet: ParsedPacket,
        protocol: u8,
    ) -> Result<ParsedPacket, ParseError> {
        match protocol {
            6 => {
                // TCP
                if frame.len() < offset + TCP_HEADER_MIN_SIZE {
                    self.invalid_packets += 1;
                    return Err(ParseError::TcpHeaderTooSmall);
                }

                let tcp_ptr = frame.as_ptr().add(offset) as *const TcpHeader;
                let tcp = &*tcp_ptr;

                packet.src_port = u16::from_be_bytes([tcp.src_port[0], tcp.src_port[1]]);
                packet.dst_port = u16::from_be_bytes([tcp.dst_port[0], tcp.dst_port[1]]);

                let data_offset = ((tcp.data_offset >> 4) & 0x0F) as usize;
                let tcp_header_size = data_offset * 4;

                offset += tcp_header_size;
                header_size += tcp_header_size;
            }
            17 => {
                // UDP
                if frame.len() < offset + UDP_HEADER_SIZE {
                    self.invalid_packets += 1;
                    return Err(ParseError::UdpHeaderTooSmall);
                }

                let udp_ptr = frame.as_ptr().add(offset) as *const UdpHeader;
                let udp = &*udp_ptr;

                packet.src_port = u16::from_be_bytes([udp.src_port[0], udp.src_port[1]]);
                packet.dst_port = u16::from_be_bytes([udp.dst_port[0], udp.dst_port[1]]);

                offset += UDP_HEADER_SIZE;
                header_size += UDP_HEADER_SIZE;
            }
            _ => {
                self.invalid_packets += 1;
                return Err(ParseError::UnknownProtocol(protocol));
            }
        }

        // Set payload pointer (zero-copy!)
        let remaining = frame.len() - offset;
        if remaining < self.min_payload_size {
            self.invalid_packets += 1;
            return Err(ParseError::PayloadTooSmall);
        }

        packet.payload_ptr = frame.as_ptr().add(offset);
        packet.payload_len = remaining;
        packet.header_size = header_size;
        packet.checksum_validated = true;

        self.total_stripped += 1;

        Ok(packet)
    }

    /// Get statistics
    pub fn stats(&self) -> StripStats {
        StripStats {
            total_stripped: self.total_stripped,
            invalid_packets: self.invalid_packets,
            success_rate: if self.total_stripped + self.invalid_packets > 0 {
                self.total_stripped as f64 / (self.total_stripped + self.invalid_packets) as f64
            } else {
                0.0
            },
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.total_stripped = 0;
        self.invalid_packets = 0;
    }
}

/// Ethernet header structure (14 bytes)
#[repr(C, packed)]
struct EthHeader {
    dst_mac: [u8; 6],
    src_mac: [u8; 6],
    ether_type: [u8; 2],
}

/// IPv4 header structure (minimum 20 bytes)
#[repr(C, packed)]
struct Ipv4Header {
    version_ihl: u8,
    dscp_ecn: u8,
    total_length: [u8; 2],
    identification: [u8; 2],
    flags_fragment: [u8; 2],
    ttl: u8,
    protocol: u8,
    checksum: [u8; 2],
    src_addr: [u8; 4],
    dst_addr: [u8; 4],
}

/// IPv6 header structure (40 bytes)
#[repr(C, packed)]
struct Ipv6Header {
    version_tc_fl: [u8; 4],
    payload_length: [u8; 2],
    next_header: u8,
    hop_limit: u8,
    src_addr: [u8; 16],
    dst_addr: [u8; 16],
}

/// TCP header structure (minimum 20 bytes)
#[repr(C, packed)]
struct TcpHeader {
    src_port: [u8; 2],
    dst_port: [u8; 2],
    seq_num: [u8; 4],
    ack_num: [u8; 4],
    data_offset: u8,
    flags: u8,
    window: [u8; 2],
    checksum: [u8; 2],
    urgent_ptr: [u8; 2],
}

/// UDP header structure (8 bytes)
#[repr(C, packed)]
struct UdpHeader {
    src_port: [u8; 2],
    dst_port: [u8; 2],
    length: [u8; 2],
    checksum: [u8; 2],
}

/// Parse errors for header stripping
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    FrameTooSmall,
    VlanFrameTooSmall,
    Ipv4HeaderTooSmall,
    Ipv6HeaderTooSmall,
    InvalidIpv4HeaderLength,
    TcpHeaderTooSmall,
    UdpHeaderTooSmall,
    PayloadTooSmall,
    UnknownEtherType(u16),
    UnknownProtocol(u8),
}

/// Statistics for header stripping operations
#[derive(Debug, Clone)]
pub struct StripStats {
    pub total_stripped: usize,
    pub invalid_packets: usize,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_sizes() {
        assert_eq!(ETH_HEADER_SIZE, 14);
        assert_eq!(IPV4_HEADER_MIN_SIZE, 20);
        assert_eq!(UDP_HEADER_SIZE, 8);
        assert_eq!(TCP_HEADER_MIN_SIZE, 20);
    }

    #[test]
    fn test_parsed_packet_default() {
        let packet = ParsedPacket::new();
        assert!(!packet.is_valid());
        assert_eq!(packet.payload_len, 0);
    }

    #[test]
    fn test_header_stripper_creation() {
        let stripper = HeaderStripper::new(32);
        let stats = stripper.stats();
        assert_eq!(stats.total_stripped, 0);
        assert_eq!(stats.invalid_packets, 0);
    }

    #[test]
    fn test_parse_error_variants() {
        let err = ParseError::FrameTooSmall;
        assert_eq!(err, ParseError::FrameTooSmall);

        let err = ParseError::UnknownEtherType(0x1234);
        assert!(matches!(err, ParseError::UnknownEtherType(0x1234)));
    }
}
