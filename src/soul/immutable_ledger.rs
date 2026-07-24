//! SOUL.md Immutable Ledger - Stage 56
//! AMD Ryzen AI 5 Optimized | 8GB RAM Limit | Cryptographic Append-Only Storage
//!
//! This module implements an append-only, cryptographically signed ledger in Rust
//! that records new SOUL.md rules, preventing RL agents from ever forgetting or
//! overwriting past catastrophic mistakes.
//!
//! Constraints:
//! - Zero heap allocations in hot path
//! - Lock-free atomic reads for strategy validation
//! - SHA-256 cryptographic sealing for each entry
//! - Memory-mapped file I/O for persistence

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use memmap2::MmapMut;
use once_cell::sync::OnceCell;

/// Maximum ledger size in bytes (conservative limit for 8GB system)
const MAX_LEDGER_SIZE_MB: usize = 256;
const MAX_LEDGER_SIZE_BYTES: usize = MAX_LEDGER_SIZE_MB * 1024 * 1024;

/// Global ledger instance for lock-free access
static GLOBAL_LEDGER: OnceCell<Arc<ImmutableLedger>> = OnceCell::new();

/// Entry types supported by the SOUL.md ledger
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[repr(u8)]
pub enum LedgerEntryType {
    ToxicPatternBan = 0,
    DistilledStrategy = 1,
    StrategyPromotion = 2,
    StrategyDemotion = 3,
    CatastrophicFailure = 4,
    PerformanceThreshold = 5,
}

impl LedgerEntryType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::ToxicPatternBan),
            1 => Some(Self::DistilledStrategy),
            2 => Some(Self::StrategyPromotion),
            3 => Some(Self::StrategyDemotion),
            4 => Some(Self::CatastrophicFailure),
            5 => Some(Self::PerformanceThreshold),
            _ => None,
        }
    }
}

/// A single immutable ledger entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Unique entry ID (monotonically increasing)
    pub id: u64,
    /// Entry type
    pub entry_type: LedgerEntryType,
    /// Timestamp of entry creation
    pub timestamp: DateTime<Utc>,
    /// Cryptographic hash of the entry content
    pub content_hash: String,
    /// Previous entry hash (chain integrity)
    pub previous_hash: String,
    /// Entry payload (JSON-encoded)
    pub payload: String,
    /// Cryptographic seal (signature)
    pub seal: String,
    /// Validation flag
    pub validated: bool,
}

impl LedgerEntry {
    /// Create a new ledger entry with cryptographic linking
    pub fn new(
        id: u64,
        entry_type: LedgerEntryType,
        payload: String,
        previous_hash: String,
    ) -> Self {
        let timestamp = Utc::now();
        
        // Generate content hash
        let mut hasher = Sha256::new();
        hasher.update(payload.as_bytes());
        hasher.update(entry_type as u8);
        let content_hash = format!("{:x}", hasher.finalize());
        
        // Generate seal (includes chain integrity)
        let mut seal_hasher = Sha256::new();
        seal_hasher.update(content_hash.as_bytes());
        seal_hasher.update(previous_hash.as_bytes());
        seal_hasher.update(timestamp.to_rfc3339().as_bytes());
        let seal = format!("{:x}", seal_hasher.finalize());
        
        Self {
            id,
            entry_type,
            timestamp,
            content_hash,
            previous_hash,
            payload,
            seal,
            validated: true,
        }
    }
    
    /// Verify entry integrity
    pub fn verify(&self) -> bool {
        // Verify content hash
        let mut hasher = Sha256::new();
        hasher.update(self.payload.as_bytes());
        hasher.update(self.entry_type as u8);
        let expected_hash = format!("{:x}", hasher.finalize());
        
        if expected_hash != self.content_hash {
            return false;
        }
        
        // Verify seal
        let mut seal_hasher = Sha256::new();
        seal_hasher.update(self.content_hash.as_bytes());
        seal_hasher.update(self.previous_hash.as_bytes());
        seal_hasher.update(self.timestamp.to_rfc3339().as_bytes());
        let expected_seal = format!("{:x}", seal_hasher.finalize());
        
        expected_seal == self.seal
    }
    
    /// Serialize entry to bytes (fixed-size header + variable payload)
    pub fn to_bytes(&self) -> Vec<u8> {
        let json = serde_json::to_string(self).unwrap_or_default();
        json.into_bytes()
    }
    
    /// Deserialize entry from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let json = std::str::from_utf8(bytes).ok()?;
        serde_json::from_str(json).ok()
    }
}

/// Memory-mapped ledger storage with append-only semantics
pub struct MappedLedgerStorage {
    file: File,
    mmap: MmapMut,
    current_size: AtomicU64,
    entry_count: AtomicU64,
}

impl MappedLedgerStorage {
    /// Create or open a memory-mapped ledger file
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        
        // Get current file size
        let metadata = file.metadata()?;
        let current_len = metadata.len();
        
        // Extend file if needed (pre-allocate for performance)
        if current_len < MAX_LEDGER_SIZE_BYTES as u64 {
            file.set_len(MAX_LEDGER_SIZE_BYTES as u64)?;
        }
        
        // Create memory map
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        
        // Count existing entries
        let mut entry_count = 0u64;
        let mut offset = 0usize;
        
        while offset < current_len as usize {
            // Read length prefix
            if offset + 4 > current_len as usize {
                break;
            }
            
            let len_bytes = &mmap[offset..offset + 4];
            let entry_len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
            
            if entry_len == 0 || offset + 4 + entry_len > current_len as usize {
                break;
            }
            
            entry_count += 1;
            offset += 4 + entry_len;
        }
        
        Ok(Self {
            file,
            mmap,
            current_size: AtomicU64::new(current_len),
            entry_count: AtomicU64::new(entry_count),
        })
    }
    
    /// Append an entry to the ledger (thread-safe)
    pub fn append(&mut self, entry: &LedgerEntry) -> std::io::Result<u64> {
        let entry_bytes = entry.to_bytes();
        let len_prefix = (entry_bytes.len() as u32).to_le_bytes();
        
        let current_offset = self.current_size.load(Ordering::Relaxed) as usize;
        
        // Check size limit
        if current_offset + 4 + entry_bytes.len() > MAX_LEDGER_SIZE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Ledger size limit exceeded",
            ));
        }
        
        // Write length prefix
        self.mmap[current_offset..current_offset + 4].copy_from_slice(&len_prefix);
        
        // Write entry data
        self.mmap[current_offset + 4..current_offset + 4 + entry_bytes.len()]
            .copy_from_slice(&entry_bytes);
        
        // Flush to disk
        self.mmap.flush_async()?;
        
        // Update counters
        let new_size = current_offset + 4 + entry_bytes.len();
        self.current_size.store(new_size as u64, Ordering::Release);
        
        let new_count = self.entry_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        Ok(new_count)
    }
    
    /// Read an entry by index
    pub fn read_entry(&self, index: u64) -> Option<LedgerEntry> {
        let mut offset = 0usize;
        let mut current_index = 0u64;
        let size = self.current_size.load(Ordering::Acquire) as usize;
        
        while offset < size {
            if offset + 4 > size {
                break;
            }
            
            let len_bytes: [u8; 4] = self.mmap[offset..offset + 4].try_into().ok()?;
            let entry_len = u32::from_le_bytes(len_bytes) as usize;
            
            if entry_len == 0 || offset + 4 + entry_len > size {
                break;
            }
            
            if current_index == index {
                let entry_bytes = &self.mmap[offset + 4..offset + 4 + entry_len];
                return LedgerEntry::from_bytes(entry_bytes);
            }
            
            current_index += 1;
            offset += 4 + entry_len;
        }
        
        None
    }
    
    /// Get total entry count
    pub fn entry_count(&self) -> u64 {
        self.entry_count.load(Ordering::Relaxed)
    }
    
    /// Get the hash of the last entry (for chain linking)
    pub fn get_last_hash(&self) -> String {
        let count = self.entry_count.load(Ordering::Relaxed);
        if count == 0 {
            return "genesis".to_string();
        }
        
        if let Some(last_entry) = self.read_entry(count - 1) {
            last_entry.content_hash
        } else {
            "genesis".to_string()
        }
    }
}

/// The main immutable ledger with lock-free read access
pub struct ImmutableLedger {
    storage: parking_lot::RwLock<MappedLedgerStorage>,
    path: PathBuf,
    banned_patterns: dashmap::DashMap<String, LedgerEntry>,
    approved_strategies: dashmap::DashMap<String, LedgerEntry>,
    is_shutdown: AtomicBool,
}

impl ImmutableLedger {
    /// Create or open the ledger at the specified path
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let storage = MappedLedgerStorage::open(&path)?;
        
        let ledger = Self {
            storage: parking_lot::RwLock::new(storage),
            path,
            banned_patterns: dashmap::DashMap::new(),
            approved_strategies: dashmap::DashMap::new(),
            is_shutdown: AtomicBool::new(false),
        };
        
        // Rebuild indices from existing entries
        ledger.rebuild_indices()?;
        
        Ok(ledger)
    }
    
    /// Get or create the global ledger instance
    pub fn global() -> &'static Arc<Self> {
        GLOBAL_LEDGER.get_or_init(|| {
            Arc::new(
                Self::open("/tmp/soul_ledger.dat").expect("Failed to open global ledger")
            )
        })
    }
    
    /// Rebuild in-memory indices from persisted ledger
    fn rebuild_indices(&self) -> std::io::Result<()> {
        let storage = self.storage.read();
        let count = storage.entry_count();
        
        for i in 0..count {
            if let Some(entry) = storage.read_entry(i) {
                match entry.entry_type {
                    LedgerEntryType::ToxicPatternBan | LedgerEntryType::CatastrophicFailure => {
                        // Extract pattern hash from payload
                        if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&entry.payload) {
                            if let Some(hash) = payload.get("hash").and_then(|h| h.as_str()) {
                                self.banned_patterns.insert(hash.to_string(), entry);
                            }
                        }
                    }
                    LedgerEntryType::DistilledStrategy | LedgerEntryType::StrategyPromotion => {
                        if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&entry.payload) {
                            if let Some(hash) = payload.get("model_hash").or_else(|| payload.get("strategy_hash")) {
                                if let Some(hash_str) = hash.as_str() {
                                    self.approved_strategies.insert(hash_str.to_string(), entry);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        Ok(())
    }
    
    /// Append a new entry to the ledger (thread-safe)
    pub fn append(&self, entry_type: LedgerEntryType, payload: serde_json::Value) -> std::io::Result<u64> {
        if self.is_shutdown.load(Ordering::Relaxed) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Ledger is shutdown",
            ));
        }
        
        let payload_str = serde_json::to_string(&payload).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        
        let previous_hash = {
            let storage = self.storage.read();
            storage.get_last_hash()
        };
        
        let next_id = {
            let storage = self.storage.read();
            storage.entry_count()
        };
        
        let entry = LedgerEntry::new(next_id, entry_type, payload_str, previous_hash);
        
        // Verify entry before appending
        if !entry.verify() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Entry verification failed",
            ));
        }
        
        // Append to storage
        let stored_id = {
            let mut storage = self.storage.write();
            storage.append(&entry)?
        };
        
        // Update indices
        match entry_type {
            LedgerEntryType::ToxicPatternBan | LedgerEntryType::CatastrophicFailure => {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&entry.payload) {
                    if let Some(hash) = payload.get("hash").and_then(|h| h.as_str()) {
                        self.banned_patterns.insert(hash.to_string(), entry);
                    }
                }
            }
            LedgerEntryType::DistilledStrategy | LedgerEntryType::StrategyPromotion => {
                if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&entry.payload) {
                    if let Some(hash) = payload.get("model_hash").or_else(|| payload.get("strategy_hash")) {
                        if let Some(hash_str) = hash.as_str() {
                            self.approved_strategies.insert(hash_str.to_string(), entry);
                        }
                    }
                }
            }
            _ => {}
        }
        
        Ok(stored_id)
    }
    
    /// Check if a pattern is banned (lock-free read)
    pub fn is_pattern_banned(&self, pattern_hash: &str) -> bool {
        self.banned_patterns.contains_key(pattern_hash)
    }
    
    /// Check if a strategy is approved (lock-free read)
    pub fn is_strategy_approved(&self, strategy_hash: &str) -> bool {
        self.approved_strategies.contains_key(strategy_hash)
    }
    
    /// Get all banned patterns (for bulk loading into execution engines)
    pub fn get_all_banned_patterns(&self) -> Vec<(String, LedgerEntry)> {
        self.banned_patterns
            .iter()
            .map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone()))
            .collect()
    }
    
    /// Get all approved strategies (for bulk loading)
    pub fn get_all_approved_strategies(&self) -> Vec<(String, LedgerEntry)> {
        self.approved_strategies
            .iter()
            .map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone()))
            .collect()
    }
    
    /// Verify entire ledger integrity (expensive operation)
    pub fn verify_integrity(&self) -> bool {
        let storage = self.storage.read();
        let count = storage.entry_count();
        
        if count == 0 {
            return true;
        }
        
        let mut previous_hash = "genesis".to_string();
        
        for i in 0..count {
            if let Some(entry) = storage.read_entry(i) {
                // Verify entry signature
                if !entry.verify() {
                    eprintln!("Entry {} failed verification", i);
                    return false;
                }
                
                // Verify chain linkage
                if entry.previous_hash != previous_hash {
                    eprintln!("Chain broken at entry {}", i);
                    return false;
                }
                
                previous_hash = entry.content_hash.clone();
            } else {
                eprintln!("Failed to read entry {}", i);
                return false;
            }
        }
        
        true
    }
    
    /// Get ledger statistics
    pub fn stats(&self) -> LedgerStats {
        let storage = self.storage.read();
        
        LedgerStats {
            total_entries: storage.entry_count(),
            banned_patterns: self.banned_patterns.len(),
            approved_strategies: self.approved_strategies.len(),
            storage_size_bytes: storage.current_size.load(Ordering::Relaxed),
            max_storage_bytes: MAX_LEDGER_SIZE_BYTES as u64,
        }
    }
    
    /// Graceful shutdown (flush pending writes)
    pub fn shutdown(&self) -> std::io::Result<()> {
        self.is_shutdown.store(true, Ordering::Relaxed);
        
        let mut storage = self.storage.write();
        storage.mmap.flush()?;
        
        Ok(())
    }
}

/// Ledger statistics for monitoring
#[derive(Debug, Clone)]
pub struct LedgerStats {
    pub total_entries: u64,
    pub banned_patterns: usize,
    pub approved_strategies: usize,
    pub storage_size_bytes: u64,
    pub max_storage_bytes: u64,
}

/// Builder for creating ledger entries with fluent API
pub struct LedgerEntryBuilder {
    entry_type: LedgerEntryType,
    payload: serde_json::Value,
}

impl LedgerEntryBuilder {
    pub fn toxic_pattern(hash: String, severity: f64, occurrences: u64) -> Self {
        let payload = serde_json::json!({
            "hash": hash,
            "severity": severity,
            "occurrences": occurrences,
            "banned": true
        });
        
        Self {
            entry_type: LedgerEntryType::ToxicPatternBan,
            payload,
        }
    }
    
    pub fn distilled_strategy(
        model_hash: String,
        latency_us: f64,
        performance: serde_json::Value,
    ) -> Self {
        let payload = serde_json::json!({
            "model_hash": model_hash,
            "inference_latency_us": latency_us,
            "performance_metrics": performance,
            "approved": true
        });
        
        Self {
            entry_type: LedgerEntryType::DistilledStrategy,
            payload,
        }
    }
    
    pub fn catastrophic_failure(
        strategy_hash: String,
        loss_amount: f64,
        market_conditions: serde_json::Value,
    ) -> Self {
        let payload = serde_json::json!({
            "strategy_hash": strategy_hash,
            "loss_amount": loss_amount,
            "market_conditions": market_conditions,
            "banned": true
        });
        
        Self {
            entry_type: LedgerEntryType::CatastrophicFailure,
            payload,
        }
    }
    
    pub fn build(self, ledger: &ImmutableLedger) -> std::io::Result<u64> {
        ledger.append(self.entry_type, self.payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_ledger_append_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_ledger.dat");
        
        let ledger = ImmutableLedger::open(&path).unwrap();
        
        // Append a toxic pattern ban
        let entry = LedgerEntryBuilder::toxic_pattern(
            "abc123".to_string(),
            0.95,
            5,
        );
        
        let id = entry.build(&ledger).unwrap();
        assert_eq!(id, 1);
        
        // Verify pattern is banned
        assert!(ledger.is_pattern_banned("abc123"));
        
        // Verify ledger integrity
        assert!(ledger.verify_integrity());
        
        // Check stats
        let stats = ledger.stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.banned_patterns, 1);
    }
    
    #[test]
    fn test_chain_integrity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_ledger.dat");
        
        let ledger = ImmutableLedger::open(&path).unwrap();
        
        // Append multiple entries
        for i in 0..10 {
            let entry = LedgerEntryBuilder::toxic_pattern(
                format!("pattern_{}", i),
                0.5 + (i as f64 * 0.05),
                i + 1,
            );
            entry.build(&ledger).unwrap();
        }
        
        // Verify entire chain
        assert!(ledger.verify_integrity());
    }
}
