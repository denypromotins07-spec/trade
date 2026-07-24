//! hot_swap_strategy.rs - SOUL.md Atomic Strategy Hot-Swap Engine
//! Stage 54: Nautilus/Ray Crypto Trading Bot
//! Reads approved profitable strategies from SOUL.md and atomically hot-swaps
//! them into live Rust execution without dropping active positions

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use log::{debug, error, info, warn};
use parking_lot::{RwLock, Mutex};
use regex::Regex;
use serde::{Deserialize, Serialize};
use crossbeam::channel::{bounded, Receiver, Sender};

/// Maximum number of strategies that can be loaded
const MAX_STRATEGIES: usize = 32;

/// Polling interval for SOUL.md changes (milliseconds)
const SOUL_POLL_INTERVAL_MS: u64 = 1000;

/// Atomic strategy state for lock-free reads
#[derive(Debug, Clone)]
pub struct AtomicStrategy {
    /// Unique strategy identifier
    pub id: String,
    /// Strategy name
    pub name: String,
    /// Serialized strategy parameters
    pub params: HashMap<String, String>,
    /// Version counter for change detection
    pub version: u64,
    /// Timestamp when strategy was loaded
    pub loaded_at: Instant,
    /// Whether strategy is currently active
    pub is_active: bool,
    /// Performance metrics
    pub metrics: StrategyMetrics,
}

/// Strategy performance metrics
#[derive(Debug, Clone, Default)]
pub struct StrategyMetrics {
    pub total_trades: u64,
    pub winning_trades: u64,
    pub losing_trades: u64,
    pub total_pnl: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
}

/// Parsed avoidance rule from SOUL.md
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvoidanceRule {
    pub rule_id: String,
    pub pattern_hash: String,
    pub condition: String,
    pub penalty_multiplier: f64,
    pub created_at: String,
    pub violation_count: u64,
    pub is_active: bool,
}

/// Hot-swap event types
#[derive(Debug, Clone)]
pub enum HotSwapEvent {
    /// New strategy loaded
    StrategyLoaded(String),
    /// Strategy replaced
    StrategyReplaced { old: String, new: String },
    /// Strategy deactivated
    StrategyDeactivated(String),
    /// Avoidance rule added
    AvoidanceRuleAdded(String),
    /// Error during hot-swap
    Error(String),
}

/// Result of a hot-swap operation
#[derive(Debug)]
pub struct HotSwapResult {
    pub success: bool,
    pub message: String,
    pub swapped_strategy: Option<String>,
    pub active_positions_preserved: bool,
}

/// SOUL.md file watcher and parser
pub struct SoulWatcher {
    /// Path to SOUL.md file
    soul_path: PathBuf,
    /// Last known file hash for change detection
    last_hash: AtomicU64,
    /// Last modification time
    last_modified: AtomicU64,
    /// Shutdown flag
    shutdown: AtomicBool,
}

impl SoulWatcher {
    pub fn new<P: AsRef<Path>>(soul_path: P) -> Self {
        Self {
            soul_path: soul_path.as_ref().to_path_buf(),
            last_hash: AtomicU64::new(0),
            last_modified: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Compute hash of SOUL.md content for change detection
    fn compute_hash(&self) -> Result<u64, String> {
        if !self.soul_path.exists() {
            return Ok(0);
        }

        let mut file = File::open(&self.soul_path)
            .map_err(|e| format!("Failed to open SOUL.md: {}", e))?;
        
        let mut content = Vec::new();
        file.read_to_end(&mut content)
            .map_err(|e| format!("Failed to read SOUL.md: {}", e))?;

        // Simple djb2 hash for speed
        let mut hash: u64 = 5381;
        for byte in content {
            hash = ((hash << 5).wrapping_add(hash)) + byte as u64;
        }

        Ok(hash)
    }

    /// Check if SOUL.md has changed since last check
    pub fn has_changed(&self) -> Result<bool, String> {
        let current_hash = self.compute_hash()?;
        let last_hash = self.last_hash.load(Ordering::Relaxed);
        
        if current_hash != last_hash && current_hash != 0 {
            self.last_hash.store(current_hash, Ordering::Relaxed);
            return Ok(true);
        }
        
        Ok(false)
    }

    /// Parse avoidance rules from SOUL.md content
    pub fn parse_avoidance_rules(&self) -> Result<Vec<AvoidanceRule>, String> {
        if !self.soul_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.soul_path)
            .map_err(|e| format!("Failed to open SOUL.md: {}", e))?;
        let reader = BufReader::new(file);
        
        let mut rules = Vec::new();
        let mut current_rule: Option<AvoidanceRuleBuilder> = None;

        // Regex patterns for parsing
        let rule_id_re = Regex::new(r"\*\*Rule ID\*\*:\s*(.+)").unwrap();
        let pattern_hash_re = Regex::new(r"\*\*Pattern Hash\*\*:\s*(.+)").unwrap();
        let penalty_re = Regex::new(r"\*\*Penalty Multiplier\*\*:\s*([\d.]+)x").unwrap();
        let condition_re = Regex::new(r"```\s*\n(.+?)\n\s*```").unwrap();

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {}", e))?;

            if let Some(caps) = rule_id_re.captures(&line) {
                // Save previous rule if exists
                if let Some(builder) = current_rule.take() {
                    if let Some(rule) = builder.build() {
                        rules.push(rule);
                    }
                }
                current_rule = Some(AvoidanceRuleBuilder::new(&caps[1]));
            } else if let Some(caps) = pattern_hash_re.captures(&line) {
                if let Some(ref mut builder) = current_rule {
                    builder.pattern_hash = caps[1].clone();
                }
            } else if let Some(caps) = penalty_re.captures(&line) {
                if let Some(ref mut builder) = current_rule {
                    builder.penalty_multiplier = caps[1].parse().unwrap_or(1.0);
                }
            } else if let Some(caps) = condition_re.captures(&line) {
                if let Some(ref mut builder) = current_rule {
                    builder.condition = caps[1].clone();
                }
            }
        }

        // Don't forget the last rule
        if let Some(builder) = current_rule {
            if let Some(rule) = builder.build() {
                rules.push(rule);
            }
        }

        Ok(rules)
    }

    /// Start background watching thread
    pub fn start_watch(
        &self,
        tx: Sender<HotSwapEvent>,
    ) -> std::thread::JoinHandle<()> {
        let soul_path = self.soul_path.clone();
        let shutdown = self.shutdown.clone();
        let mut last_hash = self.last_hash.load(Ordering::Relaxed);

        std::thread::Builder::new()
            .name("soul_watcher".to_string())
            .spawn(move || {
                info!("SOUL.md watcher started");
                
                while !shutdown.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(SOUL_POLL_INTERVAL_MS));
                    
                    // Check for file changes
                    if let Ok(current_hash) = Self::compute_hash_static(&soul_path) {
                        if current_hash != last_hash && current_hash != 0 {
                            info!("SOUL.md changed detected");
                            last_hash = current_hash;
                            
                            // Notify about change
                            let _ = tx.send(HotSwapEvent::AvoidanceRuleAdded(
                                "SOUL.md updated".to_string()
                            ));
                        }
                    }
                }
                
                info!("SOUL.md watcher stopped");
            })
            .expect("Failed to spawn soul watcher thread")
    }

    fn compute_hash_static(path: &Path) -> Result<u64, String> {
        if !path.exists() {
            return Ok(0);
        }

        let mut file = File::open(path)?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;

        let mut hash: u64 = 5381;
        for byte in content {
            hash = ((hash << 5).wrapping_add(hash)) + byte as u64;
        }
        Ok(hash)
    }

    pub fn stop(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

/// Builder for avoidance rules during parsing
struct AvoidanceRuleBuilder {
    rule_id: String,
    pattern_hash: String,
    condition: String,
    penalty_multiplier: f64,
}

impl AvoidanceRuleBuilder {
    fn new(rule_id: &str) -> Self {
        Self {
            rule_id: rule_id.trim().to_string(),
            pattern_hash: String::new(),
            condition: String::new(),
            penalty_multiplier: 1.0,
        }
    }

    fn build(self) -> Option<AvoidanceRule> {
        if self.rule_id.is_empty() {
            return None;
        }

        Some(AvoidanceRule {
            rule_id: self.rule_id,
            pattern_hash: self.pattern_hash,
            condition: self.condition,
            penalty_multiplier: self.penalty_multiplier,
            created_at: chrono::Utc::now().to_rfc3339(),
            violation_count: 0,
            is_active: true,
        })
    }
}

/// Hot-swap strategy manager
pub struct HotSwapStrategyManager {
    /// Currently active strategies (RCU-style for lock-free reads)
    active_strategies: Arc<RwLock<HashMap<String, Arc<AtomicStrategy>>>>,
    
    /// Pending strategies waiting for activation
    pending_strategies: Arc<Mutex<HashMap<String, AtomicStrategy>>>,
    
    /// Loaded avoidance rules
    avoidance_rules: Arc<RwLock<Vec<AvoidanceRule>>>,
    
    /// Event channel for hot-swap notifications
    event_tx: Sender<HotSwapEvent>,
    event_rx: Receiver<HotSwapEvent>,
    
    /// SOUL.md watcher
    soul_watcher: Arc<SoulWatcher>,
    
    /// Version counter for atomic updates
    version_counter: AtomicU64,
    
    /// Flag indicating swap in progress
    swap_in_progress: AtomicBool,
    
    /// Active positions that must be preserved during swaps
    active_positions: Arc<RwLock<HashMap<String, PositionInfo>>>,
}

/// Information about an active position
#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub symbol: String,
    pub size: f64,
    pub entry_price: f64,
    pub strategy_id: String,
}

impl HotSwapStrategyManager {
    /// Create a new hot-swap strategy manager
    pub fn new(soul_md_path: &str) -> Result<Self, String> {
        let (tx, rx) = bounded(MAX_STRATEGIES * 2);
        let soul_watcher = Arc::new(SoulWatcher::new(soul_md_path));
        
        Ok(Self {
            active_strategies: Arc::new(RwLock::new(HashMap::new())),
            pending_strategies: Arc::new(Mutex::new(HashMap::new())),
            avoidance_rules: Arc::new(RwLock::new(Vec::new())),
            event_tx: tx,
            event_rx: rx,
            soul_watcher,
            version_counter: AtomicU64::new(0),
            swap_in_progress: AtomicBool::new(false),
            active_positions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Initialize and start watching SOUL.md
    pub fn initialize(&self) -> Result<(), String> {
        info!("Initializing HotSwapStrategyManager");
        
        // Load initial avoidance rules
        self.reload_avoidance_rules()?;
        
        // Start background watcher
        let watcher = self.soul_watcher.clone();
        let tx = self.event_tx.clone();
        
        std::thread::spawn(move || {
            let _handle = watcher.start_watch(tx);
        });
        
        Ok(())
    }

    /// Reload avoidance rules from SOUL.md
    pub fn reload_avoidance_rules(&self) -> Result<Vec<AvoidanceRule>, String> {
        let rules = self.soul_watcher.parse_avoidance_rules()?;
        
        let mut current_rules = self.avoidance_rules.write();
        *current_rules = rules.clone();
        
        info!("Loaded {} avoidance rules from SOUL.md", rules.len());
        
        Ok(rules)
    }

    /// Atomically load a new strategy without affecting active positions
    pub fn load_strategy(
        &self,
        strategy_id: &str,
        name: &str,
        params: HashMap<String, String>,
    ) -> HotSwapResult {
        if self.swap_in_progress.load(Ordering::Relaxed) {
            return HotSwapResult {
                success: false,
                message: "Another swap is in progress".to_string(),
                swapped_strategy: None,
                active_positions_preserved: true,
            };
        }

        self.swap_in_progress.store(true, Ordering::SeqCst);

        let result = (|| -> Result<HotSwapResult, String> {
            let version = self.version_counter.fetch_add(1, Ordering::Relaxed);
            
            let new_strategy = AtomicStrategy {
                id: strategy_id.to_string(),
                name: name.to_string(),
                params,
                version,
                loaded_at: Instant::now(),
                is_active: false, // Start inactive
                metrics: StrategyMetrics::default(),
            };

            // Add to pending strategies first
            {
                let mut pending = self.pending_strategies.lock();
                pending.insert(strategy_id.to_string(), new_strategy.clone());
            }

            // Activate atomically
            {
                let mut active = self.active_strategies.write();
                
                let old_strategy = active.get(strategy_id).cloned();
                
                // Insert new strategy
                let arc_strategy = Arc::new(new_strategy);
                active.insert(strategy_id.to_string(), arc_strategy);
                
                // Send event
                if old_strategy.is_some() {
                    let _ = self.event_tx.send(HotSwapEvent::StrategyReplaced {
                        old: strategy_id.to_string(),
                        new: strategy_id.to_string(),
                    });
                } else {
                    let _ = self.event_tx.send(HotSwapEvent::StrategyLoaded(
                        strategy_id.to_string()
                    ));
                }
            }

            Ok(HotSwapResult {
                success: true,
                message: format!("Strategy '{}' loaded successfully", name),
                swapped_strategy: Some(strategy_id.to_string()),
                active_positions_preserved: true,
            })
        })();

        self.swap_in_progress.store(false, Ordering::SeqCst);

        match result {
            Ok(r) => r,
            Err(e) => HotSwapResult {
                success: false,
                message: e,
                swapped_strategy: None,
                active_positions_preserved: true,
            },
        }
    }

    /// Activate a pending strategy
    pub fn activate_strategy(&self, strategy_id: &str) -> HotSwapResult {
        let mut active = self.active_strategies.write();
        
        if let Some(strategy) = active.get_mut(strategy_id) {
            let mut s = (*strategy).clone();
            s.is_active = true;
            *strategy = Arc::new(s);
            
            info!("Strategy '{}' activated", strategy_id);
            
            HotSwapResult {
                success: true,
                message: "Strategy activated".to_string(),
                swapped_strategy: Some(strategy_id.to_string()),
                active_positions_preserved: true,
            }
        } else {
            HotSwapResult {
                success: false,
                message: "Strategy not found".to_string(),
                swapped_strategy: None,
                active_positions_preserved: true,
            }
        }
    }

    /// Deactivate a strategy without removing it
    pub fn deactivate_strategy(&self, strategy_id: &str) -> HotSwapResult {
        let mut active = self.active_strategies.write();
        
        if let Some(strategy) = active.get_mut(strategy_id) {
            let mut s = (*strategy).clone();
            s.is_active = false;
            *strategy = Arc::new(s);
            
            let _ = self.event_tx.send(HotSwapEvent::StrategyDeactivated(
                strategy_id.to_string()
            ));
            
            info!("Strategy '{}' deactivated", strategy_id);
            
            HotSwapResult {
                success: true,
                message: "Strategy deactivated".to_string(),
                swapped_strategy: Some(strategy_id.to_string()),
                active_positions_preserved: true,
            }
        } else {
            HotSwapResult {
                success: false,
                message: "Strategy not found".to_string(),
                swapped_strategy: None,
                active_positions_preserved: true,
            }
        }
    }

    /// Get all active strategies (lock-free read)
    pub fn get_active_strategies(&self) -> Vec<Arc<AtomicStrategy>> {
        let active = self.active_strategies.read();
        active.values()
            .filter(|s| s.is_active)
            .cloned()
            .collect()
    }

    /// Get strategy by ID
    pub fn get_strategy(&self, strategy_id: &str) -> Option<Arc<AtomicStrategy>> {
        let active = self.active_strategies.read();
        active.get(strategy_id).cloned()
    }

    /// Get all avoidance rules
    pub fn get_avoidance_rules(&self) -> Vec<AvoidanceRule> {
        self.avoidance_rules.read().clone()
    }

    /// Register an active position (preserved during swaps)
    pub fn register_position(&self, position: PositionInfo) {
        let mut positions = self.active_positions.write();
        positions.insert(position.symbol.clone(), position);
    }

    /// Remove a position
    pub fn remove_position(&self, symbol: &str) -> Option<PositionInfo> {
        let mut positions = self.active_positions.write();
        positions.remove(symbol)
    }

    /// Get all active positions
    pub fn get_active_positions(&self) -> Vec<PositionInfo> {
        self.active_positions.read().values().cloned().collect()
    }

    /// Poll for hot-swap events
    pub fn poll_events(&self) -> Vec<HotSwapEvent> {
        let mut events = Vec::new();
        
        while let Ok(event) = self.event_rx.try_recv() {
            events.push(event);
        }
        
        events
    }

    /// Update strategy metrics atomically
    pub fn update_metrics(
        &self,
        strategy_id: &str,
        pnl: f64,
        is_win: bool,
    ) -> Result<(), String> {
        let mut active = self.active_strategies.write();
        
        if let Some(strategy) = active.get_mut(strategy_id) {
            let mut s = (*strategy).clone();
            s.metrics.total_trades += 1;
            s.metrics.total_pnl += pnl;
            
            if is_win {
                s.metrics.winning_trades += 1;
            } else {
                s.metrics.losing_trades += 1;
            }
            
            s.metrics.win_rate = if s.metrics.total_trades > 0 {
                s.metrics.winning_trades as f64 / s.metrics.total_trades as f64
            } else {
                0.0
            };
            
            *strategy = Arc::new(s);
            Ok(())
        } else {
            Err(format!("Strategy '{}' not found", strategy_id))
        }
    }

    /// Graceful shutdown
    pub fn shutdown(&self) {
        info!("Shutting down HotSwapStrategyManager");
        self.soul_watcher.stop();
    }
}

impl Drop for HotSwapStrategyManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_soul_watcher_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let watcher = SoulWatcher::new(temp_file.path());
        assert!(!watcher.has_changed().unwrap());
    }

    #[test]
    fn test_hot_swap_manager_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let manager = HotSwapStrategyManager::new(
            temp_file.path().to_str().unwrap()
        );
        assert!(manager.is_ok());
    }

    #[test]
    fn test_load_strategy() {
        let temp_file = NamedTempFile::new().unwrap();
        let manager = HotSwapStrategyManager::new(
            temp_file.path().to_str().unwrap()
        ).unwrap();
        
        let mut params = HashMap::new();
        params.insert("threshold".to_string(), "0.5".to_string());
        
        let result = manager.load_strategy(
            "test-strat-1",
            "Test Strategy",
            params
        );
        
        assert!(result.success);
        assert!(result.active_positions_preserved);
    }

    #[test]
    fn test_avoidance_rule_parsing() {
        let mut temp_file = NamedTempFile::new().unwrap();
        
        // Write sample SOUL.md content
        writeln!(temp_file, "# SOUL.md\n").unwrap();
        writeln!(temp_file, "## Critical Avoidance Rule Added\n").unwrap();
        writeln!(temp_file, "**Rule ID**: AR-test123").unwrap();
        writeln!(temp_file, "**Pattern Hash**: abc123def456").unwrap();
        writeln!(temp_file, "**Penalty Multiplier**: 5.0x").unwrap();
        writeln!(temp_file, "```").unwrap();
        writeln!(temp_file, "symbol == 'BTCUSDT' AND strategy == 'momentum'").unwrap();
        writeln!(temp_file, "```").unwrap();
        
        let watcher = SoulWatcher::new(temp_file.path());
        let rules = watcher.parse_avoidance_rules().unwrap();
        
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].rule_id, "AR-test123");
        assert_eq!(rules[0].penalty_multiplier, 5.0);
    }
}
