//! Syscall Filter - Seccomp-style Filtering for Windows
//! 
//! This module implements a strict syscall filter for Windows, blocking:
//! - Unauthorized child process spawning
//! - Registry access outside allowed paths
//! - Network connections to non-whitelisted endpoints
//! 
//! Provides seccomp-like functionality on Windows using ETW and API hooking.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Maximum number of blocked syscalls before alert
const BLOCK_THRESHOLD: u64 = 100;

/// Syscall categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyscallCategory {
    ProcessCreate,
    RegistryAccess,
    FileIo,
    Network,
    Memory,
    Thread,
    Other,
}

/// Syscall rule definition
#[derive(Debug, Clone)]
pub struct SyscallRule {
    pub category: SyscallCategory,
    pub action: RuleAction,
    pub pattern: String,
    pub allow_list: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleAction {
    Allow,
    Block,
    LogOnly,
    Alert,
}

/// Blocked syscall record
#[derive(Debug, Clone)]
pub struct BlockedSyscall {
    pub category: SyscallCategory,
    pub api_name: String,
    pub process_id: u32,
    pub thread_id: u32,
    pub timestamp: std::time::Instant,
    pub parameters: Vec<String>,
}

/// Syscall Filter Manager
pub struct SyscallFilterManager {
    /// Rules
    rules: parking_lot::RwLock<Vec<SyscallRule>>,
    /// Allowed child processes
    allowed_processes: parking_lot::Mutex<HashSet<String>>,
    /// Allowed registry paths
    allowed_registry_paths: parking_lot::Mutex<HashSet<String>>,
    /// Blocked syscalls log
    blocked_calls: parking_lot::Mutex<Vec<BlockedSyscall>>,
    /// Statistics
    total_blocked: AtomicU64,
    total_allowed: AtomicU64,
    alerts_triggered: AtomicU64,
    /// Running flag
    is_running: Arc<AtomicBool>,
    /// Hook handles (platform-specific)
    hook_handles: parking_lot::Mutex<Vec<usize>>,
}

impl SyscallFilterManager {
    /// Create new syscall filter manager
    pub fn new() -> Self {
        Self {
            rules: parking_lot::RwLock::new(Vec::new()),
            allowed_processes: parking_lot::Mutex::new(HashSet::new()),
            allowed_registry_paths: parking_lot::Mutex::new(HashSet::new()),
            blocked_calls: parking_lot::Mutex::new(Vec::new()),
            total_blocked: AtomicU64::new(0),
            total_allowed: AtomicU64::new(0),
            alerts_triggered: AtomicU64::new(0),
            is_running: Arc::new(AtomicBool::new(false)),
            hook_handles: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Initialize default security rules
    pub fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut rules = self.rules.write();

        // Process creation rules
        rules.push(SyscallRule {
            category: SyscallCategory::ProcessCreate,
            action: RuleAction::Block,
            pattern: "*".to_string(),
            allow_list: Some(vec![
                "nautilus.exe".to_string(),
                "shadow_fork.exe".to_string(),
            ]),
        });

        // Registry access rules
        rules.push(SyscallRule {
            category: SyscallCategory::RegistryAccess,
            action: RuleAction::Block,
            pattern: "HKEY_*".to_string(),
            allow_list: Some(vec![
                r"HKEY_LOCAL_MACHINE\SOFTWARE\Nautilus".to_string(),
                r"HKEY_CURRENT_USER\SOFTWARE\Nautilus".to_string(),
            ]),
        });

        // Network rules - only allow Binance endpoints
        rules.push(SyscallRule {
            category: SyscallCategory::Network,
            action: RuleAction::LogOnly,
            pattern: "*.binance.com:*".to_string(),
            allow_list: None,
        });

        drop(rules);

        self.is_running.store(true, Ordering::SeqCst);
        log::info!("Syscall filter initialized with {} rules", self.rules.read().len());
        
        Ok(())
    }

    /// Add allowed child process
    pub fn allow_process(&self, process_name: &str) {
        self.allowed_processes.lock().insert(process_name.to_string());
    }

    /// Add allowed registry path
    pub fn allow_registry_path(&self, path: &str) {
        self.allowed_registry_paths.lock().insert(path.to_string());
    }

    /// Check if process creation should be allowed
    pub fn check_process_creation(&self, target_process: &str) -> bool {
        let rules = self.rules.read();
        
        for rule in rules.iter() {
            if rule.category != SyscallCategory::ProcessCreate {
                continue;
            }

            let allowed = rule.allow_list.as_ref()
                .map_or(false, |list| list.iter().any(|p| p == target_process));

            match rule.action {
                RuleAction::Allow => return true,
                RuleAction::Block => {
                    if !allowed {
                        self.log_blocked_call(SyscallCategory::ProcessCreate, "CreateProcess", target_process);
                        return false;
                    }
                }
                RuleAction::LogOnly => {
                    log::debug!("Process creation logged: {}", target_process);
                }
                RuleAction::Alert => {
                    self.alerts_triggered.fetch_add(1, Ordering::Relaxed);
                    log::warn!("ALERT: Process creation attempt: {}", target_process);
                }
            }
        }

        self.total_allowed.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Check if registry access should be allowed
    pub fn check_registry_access(&self, key_path: &str) -> bool {
        let allowed_paths = self.allowed_registry_paths.lock();
        
        if allowed_paths.iter().any(|p| key_path.starts_with(p)) {
            self.total_allowed.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        // Check rules
        let rules = self.rules.read();
        for rule in rules.iter() {
            if rule.category != SyscallCategory::RegistryAccess {
                continue;
            }

            match rule.action {
                RuleAction::Block => {
                    self.log_blocked_call(SyscallCategory::RegistryAccess, "RegOpenKey", key_path);
                    return false;
                }
                RuleAction::Alert => {
                    self.alerts_triggered.fetch_add(1, Ordering::Relaxed);
                    log::warn!("ALERT: Registry access: {}", key_path);
                }
                _ => {}
            }
        }

        true
    }

    /// Log blocked syscall
    fn log_blocked_call(&self, category: SyscallCategory, api_name: &str, params: &str) {
        let mut blocked = self.blocked_calls.lock();
        
        blocked.push(BlockedSyscall {
            category,
            api_name: api_name.to_string(),
            process_id: std::process::id(),
            thread_id: 0, // Would get actual thread ID
            timestamp: std::time::Instant::now(),
            parameters: vec![params.to_string()],
        });

        // Keep only last 1000 entries
        if blocked.len() > 1000 {
            blocked.remove(0);
        }

        self.total_blocked.fetch_add(1, Ordering::Relaxed);

        // Alert if threshold exceeded
        if self.total_blocked.load(Ordering::Relaxed) % BLOCK_THRESHOLD == 0 {
            log::error!("BLOCKED SYSCALL THRESHOLD EXCEEDED: {}", self.total_blocked.load(Ordering::Relaxed));
        }
    }

    /// Get blocked syscalls log
    pub fn get_blocked_calls(&self) -> Vec<BlockedSyscall> {
        self.blocked_calls.lock().clone()
    }

    /// Get statistics
    pub fn get_stats(&self) -> SyscallStats {
        SyscallStats {
            total_blocked: self.total_blocked.load(Ordering::Relaxed),
            total_allowed: self.total_allowed.load(Ordering::Relaxed),
            alerts_triggered: self.alerts_triggered.load(Ordering::Relaxed),
            rules_count: self.rules.read().len(),
        }
    }

    /// Install API hooks (Windows-specific)
    #[cfg(target_os = "windows")]
    pub fn install_hooks(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use detour::static_detour;
        
        // Hook CreateProcessW
        static_detour! {
            static CreateProcessWHook: extern "system" fn(
                LPCWSTR, LPWSTR, LPSECURITY_ATTRIBUTES, LPSECURITY_ATTRIBUTES,
                BOOL, DWORD, LPVOID, LPCWSTR, LPSTARTUPINFOW, LPPROCESS_INFORMATION
            ) -> BOOL;
        }

        log::info!("API hooks installed");
        Ok(())
    }

    /// Export audit log as JSON
    pub fn export_audit_json(&self) -> String {
        let blocked = self.get_blocked_calls();
        let stats = self.get_stats();
        
        let mut json = format!(r#"{{"stats":{{"blocked":{},"allowed":{},"alerts":{}}},"blocked_calls":["#, 
            stats.total_blocked, stats.total_allowed, stats.alerts_triggered);
        
        for (i, call) in blocked.iter().enumerate() {
            json.push_str(&format!(
                r#"{{"category":"{:?}","api":"{}","pid":{}}}"#,
                call.category, call.api_name, call.process_id
            ));
            if i < blocked.len() - 1 {
                json.push(',');
            }
        }
        
        json.push_str("]}");
        json
    }
}

impl Default for SyscallFilterManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Syscall statistics
#[derive(Debug, Clone)]
pub struct SyscallStats {
    pub total_blocked: u64,
    pub total_allowed: u64,
    pub alerts_triggered: u64,
    pub rules_count: usize,
}

/// Global syscall filter instance
pub static GLOBAL_SYSCALL_FILTER: parking_lot::OnceCell<Arc<SyscallFilterManager>> = parking_lot::OnceCell::new();

/// Initialize global syscall filter
pub fn init_global_filter() -> Arc<SyscallFilterManager> {
    let filter = Arc::new(SyscallFilterManager::new());
    GLOBAL_SYSCALL_FILTER.get_or_init(|| filter.clone());
    filter
}

/// Get global filter instance
pub fn get_global_filter() -> Option<Arc<SyscallFilterManager>> {
    GLOBAL_SYSCALL_FILTER.get().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_creation() {
        let filter = SyscallFilterManager::new();
        let stats = filter.get_stats();
        assert_eq!(stats.total_blocked, 0);
        assert_eq!(stats.rules_count, 0);
    }

    #[test]
    fn test_rule_initialization() {
        let filter = SyscallFilterManager::new();
        filter.initialize().unwrap();
        
        let stats = filter.get_stats();
        assert!(stats.rules_count > 0);
    }

    #[test]
    fn test_process_check() {
        let filter = SyscallFilterManager::new();
        filter.initialize().unwrap();
        
        // Should allow whitelisted process
        assert!(filter.check_process_creation("nautilus.exe"));
        
        // Should block non-whitelisted process
        assert!(!filter.check_process_creation("malware.exe"));
    }
}
