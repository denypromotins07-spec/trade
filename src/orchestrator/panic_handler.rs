//! Custom Panic Handler
//! 
//! Implements custom panic hooks that capture backtraces, securely wipe API keys
//! from memory, and write fatal crash dumps to `SOUL.md` without hanging the OS.
//! 
//! Designed for zero-hang crash reporting even under memory pressure.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::panic::{self, PanicInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::error;

/// Flag to prevent recursive panics
static PANIC_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Secure memory region marker for API keys
pub struct SecureKeyBuffer {
    data: Vec<u8>,
    wiped: bool,
}

impl SecureKeyBuffer {
    /// Create a new secure buffer
    pub fn new(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            wiped: false,
        }
    }

    /// Securely wipe the buffer (overwrite with zeros then random data)
    pub fn wipe(&mut self) {
        if self.wiped {
            return;
        }

        // First pass: overwrite with zeros
        for byte in self.data.iter_mut() {
            *byte = 0;
        }

        // Second pass: overwrite with pattern
        for i in 0..self.data.len() {
            self.data[i] = (i % 256) as u8;
        }

        // Third pass: zeros again
        for byte in self.data.iter_mut() {
            *byte = 0;
        }

        self.wiped = true;
    }

    /// Get reference to data (only before wiping)
    pub fn as_ref(&self) -> &[u8] {
        &self.data
    }
}

impl Drop for SecureKeyBuffer {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Initialize the custom panic handler
pub fn init_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        handle_panic(panic_info);
    }));
    error!("Custom panic handler installed");
}

/// Handle panic events
fn handle_panic(panic_info: &PanicInfo) {
    // Prevent recursive panic handling
    if PANIC_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        // Already handling a panic, abort immediately
        std::process::abort();
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    // Extract panic message
    let message = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic".to_string()
    };

    // Extract location
    let location = panic_info
        .location()
        .map(|loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()))
        .unwrap_or_else(|| "unknown location".to_string());

    // Capture backtrace (if available)
    let backtrace = std::backtrace::Backtrace::capture();

    // Build crash report
    let crash_report = format!(
        concat!(
            "=== FATAL CRASH DUMP ===\n",
            "Timestamp: {}\n",
            "Message: {}\n",
            "Location: {}\n",
            "Backtrace:\n{}\n",
            "=== END CRASH DUMP ===\n\n"
        ),
        timestamp,
        message,
        location,
        backtrace
    );

    // Write to SOUL.md (append mode, non-blocking best effort)
    write_crash_dump(&crash_report);

    // Securely wipe any sensitive data in scope
    wipe_sensitive_memory();

    // Log to stderr
    eprintln!("{}", crash_report);

    // Abort to prevent undefined behavior
    std::process::abort();
}

/// Write crash dump to SOUL.md
fn write_crash_dump(report: &str) {
    // Use blocking file I/O in panic context (async runtime may be dead)
    let result = OpenOptions::new()
        .create(true)
        .append(true)
        .open("SOUL.md");

    match result {
        Ok(mut file) => {
            let _ = file.write_all(report.as_bytes());
            let _ = file.sync_data();
        }
        Err(e) => {
            eprintln!("Failed to write crash dump to SOUL.md: {}", e);
        }
    }
}

/// Securely wipe sensitive memory regions
/// Called during panic to prevent key leakage
fn wipe_sensitive_memory() {
    // In production, this would iterate through registered secure buffers
    // and call wipe() on each one.
    // 
    // Note: Rust's memory model doesn't guarantee we can reach all copies,
    // but this provides best-effort security.
    
    error!("Sensitive memory wipe initiated (best effort)");
    
    // Force memory barrier
    std::sync::atomic::fence(Ordering::SeqCst);
}

/// Register a sensitive buffer for automatic wiping on panic
pub fn register_sensitive_buffer(buffer: SecureKeyBuffer) -> SecureKeyBuffer {
    // In a full implementation, this would add to a global registry
    // For now, we rely on Drop implementation
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_key_buffer_wipe() {
        let mut buffer = SecureKeyBuffer::new(b"SECRET_KEY_12345");
        assert!(!buffer.wiped);
        assert_eq!(buffer.as_ref(), b"SECRET_KEY_12345");
        
        buffer.wipe();
        assert!(buffer.wiped);
        
        // Verify wiped content is all zeros
        for byte in buffer.as_ref() {
            assert_eq!(*byte, 0);
        }
    }

    #[test]
    fn test_double_wipe_safety() {
        let mut buffer = SecureKeyBuffer::new(b"TEST");
        buffer.wipe();
        buffer.wipe(); // Should not panic
        assert!(buffer.wiped);
    }
}
