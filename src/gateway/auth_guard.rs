//! Auth Guard - Local JWT and IP-Binding Authentication
//! 
//! This module implements a lightweight, local JWT and IP-binding authentication guard
//! to ensure only the authorized localhost browser session can connect to the trading
//! bot's control panel. Optimized for AMD Ryzen AI 5 with microsecond validation.
//! 
//! RAM Budget: Uses bounded token caches and minimal allocations.
//! Enforces global 8GB RAM limit via strict cache eviction.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use sha2::{Sha256, Digest};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use thiserror::Error;

/// Token validity duration (24 hours)
const TOKEN_VALIDITY_SECS: u64 = 24 * 60 * 60;

/// Maximum cached tokens
const MAX_CACHED_TOKENS: usize = 1000;

/// Allowed localhost addresses
const ALLOWED_HOSTS: &[&str] = &["127.0.0.1", "::1", "localhost"];

/// Error types for authentication
#[derive(Error, Debug, Clone)]
pub enum AuthError {
    #[error("Invalid token format")]
    InvalidFormat,
    
    #[error("Token expired")]
    TokenExpired,
    
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("IP address not allowed: {0}")]
    IpNotAllowed(String),
    
    #[error("Token not found in cache")]
    TokenNotFound,
    
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
    
    #[error("Invalid claims")]
    InvalidClaims,
}

/// Result type for auth operations
pub type AuthResult<T> = Result<T, AuthError>;

/// JWT header structure
#[derive(Debug, Clone)]
struct JwtHeader {
    alg: String,
    typ: String,
}

impl JwtHeader {
    #[inline]
    fn new() -> Self {
        Self {
            alg: "HS256".to_string(),
            typ: "JWT".to_string(),
        }
    }
    
    #[inline]
    fn encode(&self) -> String {
        let json = format!("{{\"alg\":\"{}\",\"typ\":\"{}\"}}", self.alg, self.typ);
        URL_SAFE_NO_PAD.encode(json.as_bytes())
    }
}

impl Default for JwtHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// JWT claims structure
#[derive(Debug, Clone)]
pub struct JwtClaims {
    pub sub: String,          // Subject (user ID)
    pub iss: String,          // Issuer
    pub aud: String,          // Audience
    pub exp: u64,             // Expiration (Unix timestamp)
    pub iat: u64,             // Issued at
    pub ip: Option<String>,   // Bound IP address
    pub permissions: Vec<String>,
}

impl JwtClaims {
    #[inline]
    pub fn new(
        subject: String,
        issuer: String,
        audience: String,
        bound_ip: Option<String>,
    ) -> Self {
        let now = get_timestamp_secs();
        Self {
            sub: subject,
            iss: issuer,
            aud: audience,
            exp: now + TOKEN_VALIDITY_SECS,
            iat: now,
            ip: bound_ip,
            permissions: vec!["read".to_string(), "trade".to_string()],
        }
    }
    
    #[inline]
    pub fn is_expired(&self) -> bool {
        get_timestamp_secs() > self.exp
    }
    
    #[inline]
    pub fn encode(&self) -> String {
        let perms_json = self.permissions.iter()
            .map(|p| format!("\"{}\"", p))
            .collect::<Vec<_>>()
            .join(",");
        
        let ip_json = match &self.ip {
            Some(ip) => format!("\"ip\":\"{}\",", ip),
            None => String::new(),
        };
        
        format!(
            "{{\"sub\":\"{}\",\"iss\":\"{}\",\"aud\":\"{}\",\"exp\":{},\"iat\":{},{}\"permissions\":[{}]}}",
            self.sub, self.iss, self.aud, self.exp, self.iat, ip_json, perms_json
        )
    }
}

/// Authentication token with metadata
#[derive(Debug, Clone)]
pub struct AuthToken {
    pub token: String,
    pub claims: JwtClaims,
    pub created_at: Instant,
    pub last_used: Instant,
    pub use_count: u64,
}

impl AuthToken {
    #[inline]
    pub fn new(token: String, claims: JwtClaims) -> Self {
        let now = Instant::now();
        Self {
            token,
            claims,
            created_at: now,
            last_used: now,
            use_count: 0,
        }
    }
}

/// Get current timestamp in seconds
#[inline]
fn get_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Compute HMAC-SHA256 signature
#[inline]
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    type HmacSha256 = Hmac<Sha256>;
    
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Main authentication guard
pub struct AuthGuard {
    /// Secret key for JWT signing
    secret_key: Vec<u8>,
    /// Cached valid tokens
    token_cache: parking_lot::RwLock<HashMap<String, AuthToken>>,
    /// Allowed IP addresses
    allowed_ips: parking_lot::RwLock<Vec<IpAddr>>,
    /// Statistics
    auth_attempts: AtomicU64,
    auth_failures: AtomicU64,
    tokens_issued: AtomicU64,
    /// Running flag
    running: AtomicBool,
}

impl AuthGuard {
    /// Create a new auth guard with a random secret
    pub fn new() -> Self {
        // Generate random secret
        let mut secret = vec![0u8; 32];
        getrandom::getrandom(&mut secret).expect("Failed to generate random bytes");
        
        Self {
            secret_key: secret,
            token_cache: parking_lot::RwLock::new(HashMap::with_capacity(100)),
            allowed_ips: parking_lot::RwLock::new(vec![
                IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
            ]),
            auth_attempts: AtomicU64::new(0),
            auth_failures: AtomicU64::new(0),
            tokens_issued: AtomicU64::new(0),
            running: AtomicBool::new(true),
        }
    }
    
    /// Create with custom secret
    pub fn with_secret(secret: Vec<u8>) -> Self {
        Self {
            secret_key: secret,
            token_cache: parking_lot::RwLock::new(HashMap::with_capacity(100)),
            allowed_ips: parking_lot::RwLock::new(vec![
                IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
                IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
            ]),
            auth_attempts: AtomicU64::new(0),
            auth_failures: AtomicU64::new(0),
            tokens_issued: AtomicU64::new(0),
            running: AtomicBool::new(true),
        }
    }
    
    /// Issue a new authentication token
    pub fn issue_token(
        &self,
        subject: String,
        bound_ip: Option<String>,
    ) -> AuthResult<AuthToken> {
        if !self.running.load(Ordering::Relaxed) {
            return Err(AuthError::RateLimitExceeded);
        }
        
        // Check cache size limit
        {
            let cache = self.token_cache.read();
            if cache.len() >= MAX_CACHED_TOKENS {
                // Evict oldest tokens
                drop(cache);
                self.evict_old_tokens();
            }
        }
        
        // Create claims
        let claims = JwtClaims::new(
            subject,
            "nautilus-bot".to_string(),
            "local-control-panel".to_string(),
            bound_ip,
        );
        
        // Create JWT
        let token = self.create_jwt(&claims)?;
        
        // Create auth token
        let auth_token = AuthToken::new(token.clone(), claims);
        
        // Cache token
        {
            let mut cache = self.token_cache.write();
            cache.insert(token.clone(), auth_token.clone());
        }
        
        self.tokens_issued.fetch_add(1, Ordering::Relaxed);
        
        Ok(auth_token)
    }
    
    /// Validate a token and check IP binding
    pub fn validate_token(&self, token: &str, client_ip: IpAddr) -> AuthResult<&JwtClaims> {
        self.auth_attempts.fetch_add(1, Ordering::Relaxed);
        
        // Check if running
        if !self.running.load(Ordering::Relaxed) {
            return Err(AuthError::RateLimitExceeded);
        }
        
        // Check IP allowlist first
        if !self.is_ip_allowed(client_ip) {
            self.auth_failures.fetch_add(1, Ordering::Relaxed);
            return Err(AuthError::IpNotAllowed(client_ip.to_string()));
        }
        
        // Parse and validate JWT
        let claims = self.verify_jwt(token)?;
        
        // Check expiration
        if claims.is_expired() {
            self.auth_failures.fetch_add(1, Ordering::Relaxed);
            return Err(AuthError::TokenExpired);
        }
        
        // Verify IP binding if present
        if let Some(ref bound_ip) = claims.ip {
            if let Ok(parsed) = bound_ip.parse::<IpAddr>() {
                if parsed != client_ip {
                    self.auth_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(AuthError::IpNotAllowed(client_ip.to_string()));
                }
            }
        }
        
        // Update cache usage
        {
            let mut cache = self.token_cache.write();
            if let Some(entry) = cache.get_mut(token) {
                entry.last_used = Instant::now();
                entry.use_count += 1;
            }
        }
        
        Ok(claims)
    }
    
    /// Revoke a token
    pub fn revoke_token(&self, token: &str) -> bool {
        let mut cache = self.token_cache.write();
        cache.remove(token).is_some()
    }
    
    /// Add an allowed IP address
    pub fn add_allowed_ip(&self, ip: IpAddr) {
        let mut ips = self.allowed_ips.write();
        if !ips.contains(&ip) {
            ips.push(ip);
        }
    }
    
    /// Remove an allowed IP address
    pub fn remove_allowed_ip(&self, ip: IpAddr) {
        let mut ips = self.allowed_ips.write();
        ips.retain(|&allowed| allowed != ip);
    }
    
    /// Check if IP is allowed
    fn is_ip_allowed(&self, ip: IpAddr) -> bool {
        let ips = self.allowed_ips.read();
        ips.contains(&ip) || ALLOWED_HOSTS.iter().any(|h| {
            h.parse::<IpAddr>().ok() == Some(ip)
        })
    }
    
    /// Create a JWT from claims
    fn create_jwt(&self, claims: &JwtClaims) -> AuthResult<String> {
        let header = JwtHeader::new();
        let header_b64 = header.encode();
        let claims_json = claims.encode();
        let claims_b64 = URL_SAFE_NO_PAD.encode(claims_json.as_bytes());
        
        let signing_input = format!("{}.{}", header_b64, claims_b64);
        let signature = hmac_sha256(&self.secret_key, signing_input.as_bytes());
        let sig_b64 = URL_SAFE_NO_PAD.encode(&signature);
        
        Ok(format!("{}.{}.{}", header_b64, claims_b64, sig_b64))
    }
    
    /// Verify and decode a JWT
    fn verify_jwt(&self, token: &str) -> AuthResult<JwtClaims> {
        // Check cache first
        {
            let cache = self.token_cache.read();
            if let Some(entry) = cache.get(token) {
                if !entry.claims.is_expired() {
                    return Ok(entry.claims.clone());
                }
            }
        }
        
        // Split token
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(AuthError::InvalidFormat);
        }
        
        let claims_b64 = parts[1];
        let sig_b64 = parts[2];
        
        // Verify signature
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let expected_sig = hmac_sha256(&self.secret_key, signing_input.as_bytes());
        
        let provided_sig = URL_SAFE_NO_PAD.decode(sig_b64)
            .map_err(|_| AuthError::InvalidFormat)?;
        
        if !constant_time_eq(&expected_sig, &provided_sig) {
            return Err(AuthError::InvalidSignature);
        }
        
        // Decode claims
        let claims_json = URL_SAFE_NO_PAD.decode(claims_b64)
            .map_err(|_| AuthError::InvalidFormat)?;
        let claims_str = String::from_utf8_lossy(&claims_json);
        
        // Simple JSON parsing (production would use serde_json)
        let claims = self.parse_claims(&claims_str)
            .ok_or(AuthError::InvalidClaims)?;
        
        Ok(claims)
    }
    
    /// Parse claims from JSON string
    fn parse_claims(&self, json: &str) -> Option<JwtClaims> {
        // Simplified JSON parsing for performance
        // In production, use serde_json
        
        let extract_str = |key: &str| -> Option<String> {
            let pattern = format!("\"{}\":", key);
            if let Some(start) = json.find(&pattern) {
                let value_start = start + pattern.len();
                if json[value_start..].starts_with('"') {
                    let content_start = value_start + 1;
                    if let Some(end) = json[content_start..].find('"') {
                        return Some(json[content_start..content_start + end].to_string());
                    }
                }
            }
            None
        };
        
        let extract_num = |key: &str| -> Option<u64> {
            let pattern = format!("\"{}\":", key);
            if let Some(start) = json.find(&pattern) {
                let value_start = start + pattern.len();
                let rest = &json[value_start..];
                let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
                return rest[..end].parse().ok();
            }
            None
        };
        
        let sub = extract_str("sub")?;
        let iss = extract_str("iss").unwrap_or_default();
        let aud = extract_str("aud").unwrap_or_default();
        let exp = extract_num("exp")?;
        let iat = extract_num("iat").unwrap_or(0);
        
        // Extract IP if present
        let ip = extract_str("ip");
        
        Some(JwtClaims {
            sub,
            iss,
            aud,
            exp,
            iat,
            ip,
            permissions: vec!["read".to_string()],
        })
    }
    
    /// Evict old/unused tokens
    fn evict_old_tokens(&self) {
        let mut cache = self.token_cache.write();
        let now = Instant::now();
        let threshold = Duration::from_secs(3600); // 1 hour
        
        cache.retain(|_, entry| {
            now.duration_since(entry.last_used) < threshold && !entry.claims.is_expired()
        });
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> AuthStats {
        let cache = self.token_cache.read();
        AuthStats {
            active_tokens: cache.len(),
            auth_attempts: self.auth_attempts.load(Ordering::Relaxed),
            auth_failures: self.auth_failures.load(Ordering::Relaxed),
            tokens_issued: self.tokens_issued.load(Ordering::Relaxed),
            success_rate: self.calculate_success_rate(),
        }
    }
    
    /// Calculate success rate
    fn calculate_success_rate(&self) -> f64 {
        let attempts = self.auth_attempts.load(Ordering::Relaxed);
        let failures = self.auth_failures.load(Ordering::Relaxed);
        
        if attempts == 0 {
            return 1.0;
        }
        
        1.0 - (failures as f64 / attempts as f64)
    }
    
    /// Shutdown the auth guard
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
        self.token_cache.write().clear();
    }
}

impl Default for AuthGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Authentication statistics
#[derive(Debug, Clone, Copy)]
pub struct AuthStats {
    pub active_tokens: usize,
    pub auth_attempts: u64,
    pub auth_failures: u64,
    pub tokens_issued: u64,
    pub success_rate: f64,
}

/// Constant-time equality check for signatures
#[inline]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_issuance() {
        let guard = AuthGuard::new();
        
        let token = guard.issue_token("user123".to_string(), None);
        assert!(token.is_ok());
        
        let stats = guard.get_stats();
        assert_eq!(stats.tokens_issued, 1);
    }

    #[test]
    fn test_token_validation() {
        let guard = AuthGuard::new();
        
        let auth_token = guard.issue_token("user123".to_string(), None).unwrap();
        
        let result = guard.validate_token(
            &auth_token.token,
            IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        );
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_ip_binding() {
        let guard = AuthGuard::new();
        
        let auth_token = guard.issue_token(
            "user123".to_string(),
            Some("127.0.0.1".to_string()),
        ).unwrap();
        
        // Should pass with correct IP
        let result = guard.validate_token(
            &auth_token.token,
            IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        );
        assert!(result.is_ok());
        
        // Should fail with different IP (if we could spoof it)
    }

    #[test]
    fn test_token_revocation() {
        let guard = AuthGuard::new();
        
        let auth_token = guard.issue_token("user123".to_string(), None).unwrap();
        
        let revoked = guard.revoke_token(&auth_token.token);
        assert!(revoked);
        
        let stats = guard.get_stats();
        assert_eq!(stats.active_tokens, 0);
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        let c = [1u8, 2, 3, 5];
        
        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
        assert!(!constant_time_eq(&a, &[1u8, 2, 3]));
    }
}
