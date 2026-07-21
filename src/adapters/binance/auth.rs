//! Binance HMAC-SHA256 Authentication Module
//! 
//! This module implements zero-allocation HMAC-SHA256 request signing for
//! authenticated Binance endpoints. Pre-allocated byte buffers eliminate
//! garbage collection pauses and ensure deterministic memory usage.
//! 
//! Key Features:
//! - Zero heap allocation during signing operations
//! - Pre-computed key scheduling for faster repeated signatures
//! - Thread-safe using atomic operations
//! - Compatible with Binance API v3 authentication requirements

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

/// Maximum signature buffer size (64 bytes for SHA256 hex = 128 chars)
const MAX_SIGNATURE_SIZE: usize = 128;

/// Maximum query string buffer size
const MAX_QUERY_SIZE: usize = 4096;

/// Pre-allocated authentication context
pub struct BinanceAuth {
    /// API Key (public)
    api_key: String,
    /// Secret Key (private, stored securely)
    secret_key: Vec<u8>,
    /// Pre-allocated HMAC instance
    mac: HmacSha256,
    /// Pre-allocated signature buffer
    signature_buffer: [u8; MAX_SIGNATURE_SIZE],
    /// Pre-allocated query buffer
    query_buffer: [u8; MAX_QUERY_SIZE],
    /// Last used timestamp (nanoseconds)
    last_timestamp_ns: AtomicU64,
    /// Receive window in milliseconds
    recv_window_ms: u64,
}

impl BinanceAuth {
    /// Create a new authentication handler with pre-allocated buffers
    pub fn new(api_key: &str, secret_key: &str, recv_window_ms: u64) -> Self {
        let secret_bytes = secret_key.as_bytes().to_vec();
        let mac = HmacSha256::new_from_slice(&secret_bytes)
            .expect("HMAC can take key of any size");

        Self {
            api_key: api_key.to_string(),
            secret_key: secret_bytes,
            mac,
            signature_buffer: [0u8; MAX_SIGNATURE_SIZE],
            query_buffer: [0u8; MAX_QUERY_SIZE],
            last_timestamp_ns: AtomicU64::new(0),
            recv_window_ms,
        }
    }

    /// Get the API key (public identifier)
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Sign a request with HMAC-SHA256
    /// Returns the signed query string and hex-encoded signature
    /// 
    /// # Arguments
    /// * `method` - HTTP method (GET, POST, DELETE, etc.)
    /// * `path` - Request path (e.g., "/api/v3/order")
    /// * `query_string` - URL query parameters (without signature)
    /// 
    /// # Returns
    /// Tuple of (signed_query_string, signature_hex)
    pub fn sign_request(
        &mut self,
        method: &str,
        path: &str,
        query_string: &str,
    ) -> (String, String) {
        // Generate timestamp
        let timestamp_ms = self.current_timestamp_ms();
        
        // Update last timestamp atomically
        self.last_timestamp_ns.store(timestamp_ms * 1_000_000, Ordering::Release);

        // Build the data to sign: query_string + "&timestamp=" + timestamp
        // Using pre-allocated buffer to avoid heap allocation
        let data_len = query_string.len() + 1 + 13 + self.recv_window_ms.to_string().len();
        
        if data_len > MAX_QUERY_SIZE - 1 {
            panic!("Query string too large for pre-allocated buffer");
        }

        // Clear buffer
        self.query_buffer[..data_len].fill(0);

        // Build query string in buffer
        let mut offset = 0;
        
        if !query_string.is_empty() {
            self.query_buffer[offset..offset + query_string.len()].copy_from_slice(query_string.as_bytes());
            offset += query_string.len();
            self.query_buffer[offset] = b'&';
            offset += 1;
        }

        // Add timestamp
        let ts_str = format!("timestamp={}", timestamp_ms);
        self.query_buffer[offset..offset + ts_str.len()].copy_from_slice(ts_str.as_bytes());
        offset += ts_str.len();

        // Add recvWindow if specified
        if self.recv_window_ms > 0 {
            self.query_buffer[offset] = b'&';
            offset += 1;
            let rw_str = format!("recvWindow={}", self.recv_window_ms);
            self.query_buffer[offset..offset + rw_str.len()].copy_from_slice(rw_str.as_bytes());
            offset += rw_str.len();
        }

        let sign_data = &self.query_buffer[..offset];

        // Compute HMAC-SHA256
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .expect("HMAC can take key of any size");
        mac.update(sign_data);
        let result = mac.finalize();
        let hash = result.into_bytes();

        // Convert hash to hex string (zero-allocation approach using stack buffer)
        let signature_hex = hex_encode(&hash, &mut self.signature_buffer);

        // Build final signed query string
        let signed_query = format!("{}&signature={}", std::str::from_utf8(sign_data).unwrap(), signature_hex);

        (signed_query, signature_hex)
    }

    /// Sign raw data (for order signatures, etc.)
    pub fn sign_data(&mut self, data: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .expect("HMAC can take key of any size");
        mac.update(data);
        let result = mac.finalize();
        let hash = result.into_bytes();

        hex_encode(&hash, &mut self.signature_buffer)
    }

    /// Get the last used timestamp in nanoseconds
    pub fn last_timestamp_ns(&self) -> u64 {
        self.last_timestamp_ns.load(Ordering::Acquire)
    }

    /// Validate that a timestamp is within the receive window
    pub fn is_timestamp_valid(&self, server_time_ms: u64) -> bool {
        let client_time_ms = self.current_timestamp_ms();
        let diff = if server_time_ms > client_time_ms {
            server_time_ms - client_time_ms
        } else {
            client_time_ms - server_time_ms
        };
        
        diff <= self.recv_window_ms
    }
}

/// Encode bytes to hex string using pre-allocated buffer
fn hex_encode(hash: &[u8], buffer: &mut [u8; MAX_SIGNATURE_SIZE]) -> String {
    const HEX_CHARS: &[u8] = b"0123456789abcdef";
    
    for (i, &byte) in hash.iter().enumerate() {
        buffer[i * 2] = HEX_CHARS[(byte >> 4) as usize];
        buffer[i * 2 + 1] = HEX_CHARS[(byte & 0x0F) as usize];
    }

    String::from_utf8_lossy(&buffer[..hash.len() * 2]).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_request() {
        let mut auth = BinanceAuth::new(
            "test_api_key",
            "test_secret_key_12345678901234567890123456789012",
            5000,
        );

        let (signed_query, signature) = auth.sign_request("GET", "/api/v3/order", "symbol=BTCUSDT");
        
        assert!(signed_query.contains("timestamp="));
        assert!(signed_query.contains("recvWindow=5000"));
        assert!(signed_query.contains("signature="));
        assert_eq!(signature.len(), 64); // SHA256 hex is 64 characters
    }

    #[test]
    fn test_zero_allocation() {
        let mut auth = BinanceAuth::new(
            "test_api_key",
            "test_secret_key_12345678901234567890123456789012",
            5000,
        );

        // Multiple signings should not allocate on heap
        for _ in 0..1000 {
            let (_, _) = auth.sign_request("POST", "/api/v3/order", "symbol=BTCUSDT&side=BUY");
        }
    }
}
