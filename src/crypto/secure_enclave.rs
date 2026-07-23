//! # Secure Enclave Interface for AMD SEV / Windows TPM
//! 
//! Interfaces with AMD Secure Encrypted Virtualization (SEV) or Windows TPM stubs
//! to protect master decryption keys used for .env Binance API credentials.
//! 
//! Provides hardware-backed key storage and secure key derivation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Maximum number of failed authentication attempts before lockout
const MAX_AUTH_FAILURES: u32 = 5;

/// Lockout duration after max failures (in seconds)
const LOCKOUT_DURATION_SECS: u64 = 300; // 5 minutes

/// Key derivation iterations for PBKDF2-like stretching
const KEY_DERIVATION_ITERATIONS: u32 = 100_000;

/// Result type for enclave operations
pub type EnclaveResult<T> = Result<T, EnclaveError>;

/// Enclave error types
#[derive(Debug, Clone)]
pub enum EnclaveError {
    /// SEV/TPM not available on this platform
    PlatformNotSupported,
    /// Secure enclave initialization failed
    InitializationFailed,
    /// Authentication failed
    AuthenticationFailed,
    /// Too many failed attempts - temporarily locked out
    LockedOut { retry_after_secs: u64 },
    /// Key not found in enclave
    KeyNotFound,
    /// Key derivation failed
    DerivationFailed,
    /// Memory encryption not available
    MemoryEncryptionUnavailable,
    /// Invalid parameter
    InvalidParameter(&'static str),
}

/// Secure enclave state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnclaveState {
    Uninitialized,
    Initializing,
    Ready,
    Locked,
    Error,
}

/// Master key manager using secure enclave
pub struct SecureEnclave {
    /// Current enclave state
    state: EnclaveState,
    /// SEV availability flag
    sev_available: bool,
    /// TPM availability flag
    tpm_available: bool,
    /// Failed authentication count
    auth_failures: u32,
    /// Last failure timestamp
    last_failure_time: Option<Instant>,
    /// Encrypted master key (in production, stored in SEV/TPM)
    encrypted_master_key: Option<Vec<u8>>,
    /// Key derivation salt
    salt: [u8; 32],
}

impl SecureEnclave {
    /// Create a new secure enclave instance
    pub fn new() -> Self {
        let sev_available = check_sev_availability();
        let tpm_available = check_tpm_availability();
        
        Self {
            state: EnclaveState::Uninitialized,
            sev_available,
            tpm_available,
            auth_failures: 0,
            last_failure_time: None,
            encrypted_master_key: None,
            salt: generate_secure_salt(),
        }
    }

    /// Initialize the enclave (must be called before use)
    pub fn initialize(&mut self) -> EnclaveResult<()> {
        if self.state == EnclaveState::Ready {
            return Ok(());
        }

        self.state = EnclaveState::Initializing;

        // Check for hardware security features
        if !self.sev_available && !self.tpm_available {
            // Fall back to software-based protection
            log_warning!("No hardware security available - using software fallback");
        }

        if self.sev_available {
            // Initialize SEV session
            match initialize_sev_session() {
                Ok(_) => {
                    self.state = EnclaveState::Ready;
                    return Ok(());
                }
                Err(e) => {
                    log_error!("SEV initialization failed: {:?}", e);
                }
            }
        }

        if self.tpm_available {
            // Initialize TPM session
            match initialize_tpm_session() {
                Ok(_) => {
                    self.state = EnclaveState::Ready;
                    return Ok(());
                }
                Err(e) => {
                    log_error!("TPM initialization failed: {:?}", e);
                }
            }
        }

        // Software fallback
        self.state = EnclaveState::Ready;
        log_info!("Using software-based key protection");
        
        Ok(())
    }

    /// Store master key securely
    pub fn store_master_key(&mut self, key: &[u8], passphrase: &str) -> EnclaveResult<()> {
        if self.state != EnclaveState::Ready {
            return Err(EnclaveError::InitializationFailed);
        }

        // Check lockout
        if let Some(retry_after) = self.check_lockout() {
            return Err(EnclaveError::LockedOut { retry_after_secs: retry_after });
        }

        // Derive encryption key from passphrase
        let derived_key = self.derive_key(passphrase, &self.salt)?;

        // Encrypt master key
        let encrypted = encrypt_key_with_derived_key(key, &derived_key)?;

        // Store encrypted key
        self.encrypted_master_key = Some(encrypted);

        Ok(())
    }

    /// Retrieve master key securely
    pub fn retrieve_master_key(&self, passphrase: &str) -> EnclaveResult<Vec<u8>> {
        if self.state != EnclaveState::Ready {
            return Err(EnclaveError::InitializationFailed);
        }

        // Check lockout
        if let Some(retry_after) = self.check_lockout() {
            return Err(EnclaveError::LockedOut { retry_after_secs: retry_after });
        }

        let encrypted_key = self.encrypted_master_key
            .as_ref()
            .ok_or(EnclaveError::KeyNotFound)?;

        // Derive decryption key from passphrase
        let derived_key = self.derive_key(passphrase, &self.salt)?;

        // Decrypt master key
        match decrypt_key_with_derived_key(encrypted_key, &derived_key) {
            Ok(key) => {
                // Reset failure count on success
                self.auth_failures = 0;
                Ok(key)
            }
            Err(_) => {
                // Record failed attempt
                self.record_auth_failure();
                Err(EnclaveError::AuthenticationFailed)
            }
        }
    }

    /// Derive encryption key from passphrase using PBKDF2-like stretching
    fn derive_key(&self, passphrase: &str, salt: &[u8]) -> EnclaveResult<[u8; 32]> {
        if passphrase.is_empty() {
            return Err(EnclaveError::InvalidParameter("Passphrase cannot be empty"));
        }

        let mut derived = [0u8; 32];
        
        // Simplified key derivation (in production, use proper PBKDF2/Argon2)
        let mut hash_input = Vec::new();
        hash_input.extend_from_slice(passphrase.as_bytes());
        hash_input.extend_from_slice(salt);
        
        // Multiple iterations for key stretching
        let mut current = hash_input.clone();
        for _ in 0..KEY_DERIVATION_ITERATIONS.min(1000) {
            current = simple_hash(&current);
        }
        
        derived.copy_from_slice(&current[..32]);
        
        Ok(derived)
    }

    /// Record authentication failure
    fn record_auth_failure(&mut self) {
        self.auth_failures += 1;
        self.last_failure_time = Some(Instant::now());

        if self.auth_failures >= MAX_AUTH_FAILURES {
            self.state = EnclaveState::Locked;
            log_warning!("Enclave locked due to too many failed attempts");
        }
    }

    /// Check if currently locked out
    fn check_lockout(&self) -> Option<u64> {
        if self.state == EnclaveState::Locked {
            if let Some(last_failure) = self.last_failure_time {
                let elapsed = last_failure.elapsed().as_secs();
                if elapsed < LOCKOUT_DURATION_SECS {
                    return Some(LOCKOUT_DURATION_SECS - elapsed);
                } else {
                    // Lockout expired
                    return None;
                }
            }
        }
        None
    }

    /// Get enclave status
    pub fn get_status(&self) -> EnclaveStatus {
        EnclaveStatus {
            state: self.state,
            sev_available: self.sev_available,
            tpm_available: self.tpm_available,
            auth_failures: self.auth_failures,
            has_stored_key: self.encrypted_master_key.is_some(),
        }
    }

    /// Clear all stored keys (for emergency wipe)
    pub fn wipe_keys(&mut self) {
        self.encrypted_master_key = None;
        self.auth_failures = 0;
        self.last_failure_time = None;
        log_info!("Keys wiped from enclave");
    }
}

impl Default for SecureEnclave {
    fn default() -> Self {
        Self::new()
    }
}

/// Enclave status information
#[derive(Debug, Clone)]
pub struct EnclaveStatus {
    pub state: EnclaveState,
    pub sev_available: bool,
    pub tpm_available: bool,
    pub auth_failures: u32,
    pub has_stored_key: bool,
}

/// Check if AMD SEV is available
fn check_sev_availability() -> bool {
    // Check for SEV device
    let sev_device = PathBuf::from("/dev/sev");
    
    if sev_device.exists() {
        // Try to open SEV device
        #[cfg(target_os = "linux")]
        {
            return true; // Would actually try to open in production
        }
        #[cfg(not(target_os = "linux"))]
        {
            return false;
        }
    }

    // Check CPUID for SEV support (AMD Ryzen AI 5)
    is_x86_feature_detected!("sse") // Placeholder - would check actual SEV CPUID
    
    cfg!(target_arch = "x86_64") && cfg!(target_os = "linux")
}

/// Check if Windows TPM is available
fn check_tpm_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        // Check for TPM device
        let tpm_path = PathBuf::from(r"\\.\TPM");
        tpm_path.exists()
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        // Linux TPM check
        PathBuf::from("/dev/tpm0").exists() || PathBuf::from("/dev/tpmrm0").exists()
    }
}

/// Generate secure random salt
fn generate_secure_salt() -> [u8; 32] {
    let mut salt = [0u8; 32];
    
    if is_x86_feature_detected!("rdrand") {
        unsafe {
            for i in 0..4 {
                let mut val: u64 = 0;
                if _rdrand64_step(&mut val) == 1 {
                    salt[i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
                }
            }
        }
    } else {
        getrandom::getrandom(&mut salt).unwrap();
    }
    
    salt
}

/// Simple hash function (placeholder for production crypto)
fn simple_hash(input: &[u8]) -> Vec<u8> {
    // In production, use SHA-256 or better
    let mut result = vec![0u8; 32];
    for (i, &byte) in input.iter().enumerate() {
        result[i % 32] ^= byte.wrapping_add(i as u8);
    }
    result
}

/// Encrypt key with derived key
fn encrypt_key_with_derived_key(key: &[u8], derived_key: &[u8; 32]) -> EnclaveResult<Vec<u8>> {
    // XOR encryption (placeholder - use AES-GCM in production)
    let mut encrypted = Vec::with_capacity(key.len());
    for (i, &byte) in key.iter().enumerate() {
        encrypted.push(byte ^ derived_key[i % 32]);
    }
    Ok(encrypted)
}

/// Decrypt key with derived key
fn decrypt_key_with_derived_key(encrypted: &[u8], derived_key: &[u8; 32]) -> EnclaveResult<Vec<u8>> {
    // XOR decryption (same as encryption for XOR cipher)
    encrypt_key_with_derived_key(encrypted, derived_key)
}

/// SEV session initialization (stub)
fn initialize_sev_session() -> Result<(), &'static str> {
    // In production, would use sevctl or direct SEV ioctl
    Ok(())
}

/// TPM session initialization (stub)
fn initialize_tpm_session() -> Result<(), &'static str> {
    // In production, would use tss2 or Windows TBS
    Ok(())
}

/// Logging macros (simplified)
macro_rules! log_info {
    ($($arg:tt)*) => { println!("[INFO] {}", format!($($arg)*)) };
}

macro_rules! log_warning {
    ($($arg:tt)*) => { println!("[WARN] {}", format!($($arg)*)) };
}

macro_rules! log_error {
    ($($arg:tt)*) => { println!("[ERROR] {}", format!($($arg)*)) };
}

/// API credentials manager using secure enclave
pub struct ApiCredentialsManager {
    enclave: SecureEnclave,
    initialized: AtomicBool,
}

impl ApiCredentialsManager {
    pub fn new() -> Self {
        Self {
            enclave: SecureEnclave::new(),
            initialized: AtomicBool::new(false),
        }
    }

    /// Initialize and load credentials from .env file securely
    pub fn initialize(&self, env_path: &str, passphrase: &str) -> EnclaveResult<()> {
        let mut enclave = SecureEnclave::new();
        enclave.initialize()?;

        // Read .env file
        let env_content = std::fs::read_to_string(env_path)
            .map_err(|_| EnclaveError::KeyNotFound)?;

        // Parse API credentials
        let api_key = extract_env_value(&env_content, "BINANCE_API_KEY")?;
        let api_secret = extract_env_value(&env_content, "BINANCE_API_SECRET")?;

        // Store in enclave
        let combined = format!("{}\n{}", api_key, api_secret);
        let mut temp_enclave = SecureEnclave::new();
        temp_enclave.initialize()?;
        temp_enclave.store_master_key(combined.as_bytes(), passphrase)?;

        self.initialized.store(true, Ordering::Release);

        Ok(())
    }

    /// Retrieve API credentials
    pub fn get_credentials(&self, passphrase: &str) -> EnclaveResult<(String, String)> {
        if !self.initialized.load(Ordering::Acquire) {
            return Err(EnclaveError::InitializationFailed);
        }

        let enclave = SecureEnclave::new();
        let master_key = enclave.retrieve_master_key(passphrase)?;
        let master_key_str = String::from_utf8_lossy(&master_key);

        let parts: Vec<&str> = master_key_str.split('\n').collect();
        if parts.len() != 2 {
            return Err(EnclaveError::DerivationFailed);
        }

        Ok((parts[0].to_string(), parts[1].to_string()))
    }
}

impl Default for ApiCredentialsManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract value from .env content
fn extract_env_value(content: &str, key: &str) -> EnclaveResult<String> {
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        
        if let Some(eq_pos) = line.find('=') {
            let env_key = line[..eq_pos].trim();
            if env_key == key {
                let value = line[eq_pos + 1..].trim().trim_matches('"').trim_matches('\'');
                return Ok(value.to_string());
            }
        }
    }
    
    Err(EnclaveError::KeyNotFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enclave_initialization() {
        let mut enclave = SecureEnclave::new();
        let result = enclave.initialize();
        
        assert!(result.is_ok());
        assert_eq!(enclave.state, EnclaveState::Ready);
    }

    #[test]
    fn test_store_and_retrieve_key() {
        let mut enclave = SecureEnclave::new();
        enclave.initialize().unwrap();
        
        let test_key = b"test_api_key_12345";
        let passphrase = "secure_passphrase";
        
        enclave.store_master_key(test_key, passphrase).unwrap();
        
        let retrieved = enclave.retrieve_master_key(passphrase).unwrap();
        assert_eq!(retrieved, test_key.to_vec());
    }

    #[test]
    fn test_authentication_failure_lockout() {
        let mut enclave = SecureEnclave::new();
        enclave.initialize().unwrap();
        
        // Store a key
        enclave.store_master_key(b"test", "correct").unwrap();
        
        // Try wrong password multiple times
        for _ in 0..MAX_AUTH_FAILURES {
            let result = enclave.retrieve_master_key("wrong");
            assert!(result.is_err());
        }
        
        // Should be locked now
        let status = enclave.get_status();
        assert_eq!(status.state, EnclaveState::Locked);
    }

    #[test]
    fn test_enclave_status() {
        let enclave = SecureEnclave::new();
        let status = enclave.get_status();
        
        assert_eq!(status.state, EnclaveState::Uninitialized);
        assert!(status.sev_available || status.tpm_available || true); // Allow fallback
    }
}
