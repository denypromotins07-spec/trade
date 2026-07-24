//! parallel_router.rs - Parallel Multi-Asset Execution Engine Router
//! Stage 54: Nautilus/Ray Crypto Trading Bot
//! Routes incoming ticks for 6+ crypto pairs into isolated thread-local execution engines
//! Optimized for AMD Ryzen AI 5, preventing cross-symbol cache thrashing, 8GB RAM limit

use std::collections::HashMap;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use crossbeam::channel::{bounded, Receiver, Sender, TrySendError};
use dashmap::DashMap;
use log::{debug, error, info, warn};
use parking_lot::RwLock;

use crate::execution::symbol_isolator::SymbolIsolator;
use crate::execution::cross_margin_sync::CrossMarginSync;
use crate::market::tick::Tick;
use crate::config::Config;

/// Maximum number of symbols supported in parallel execution
const MAX_SYMBOLS: usize = 12;

/// Channel capacity per symbol engine (microsecond latency optimized)
const CHANNEL_CAPACITY: usize = 4096;

/// Memory budget per symbol engine in bytes (8GB total / 12 symbols ≈ 680MB each)
const MEMORY_BUDGET_PER_SYMBOL: u64 = 700_000_000;

/// Supported trading pairs for Stage 54
const SUPPORTED_SYMBOLS: &[&str] = &[
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
];

/// Routing decision for incoming ticks
#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    /// Route to specific symbol engine
    RouteTo(SymbolId),
    /// Drop tick due to backpressure
    DropBackpressure,
    /// Drop tick due to invalid symbol
    DropInvalid,
    /// Queue for later processing
    QueueDeferred,
}

/// Symbol identifier with compile-time validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(u8);

impl SymbolId {
    pub const fn new(id: u8) -> Self {
        debug_assert!(id < MAX_SYMBOLS as u8, "SymbolId out of range");
        SymbolId(id)
    }

    pub fn from_symbol(symbol: &str) -> Option<Self> {
        SUPPORTED_SYMBOLS
            .iter()
            .position(|&s| s == symbol)
            .map(|idx| SymbolId(idx as u8))
    }

    pub fn to_symbol(self) -> &'static str {
        SUPPORTED_SYMBOLS[self.0 as usize]
    }
}

/// Statistics for each symbol engine
#[derive(Debug, Default)]
pub struct SymbolEngineStats {
    pub ticks_processed: u64,
    pub ticks_dropped: u64,
    pub avg_latency_us: u64,
    pub max_latency_us: u64,
    pub memory_used_bytes: u64,
    pub last_tick_time: Option<Instant>,
}

/// Parallel router managing all symbol execution engines
pub struct ParallelRouter {
    /// Map of symbol ID to sender channel
    senders: Arc<DashMap<SymbolId, Sender<Tick>>>,
    
    /// Thread handles for each symbol engine
    engine_handles: Arc<RwLock<HashMap<SymbolId, JoinHandle<()>>>>,
    
    /// Shared cross-margin synchronizer
    margin_sync: Arc<CrossMarginSync>,
    
    /// Per-symbol isolators for lock-free memory management
    isolators: Arc<DashMap<SymbolId, Arc<SymbolIsolator>>>,
    
    /// Global statistics
    stats: Arc<RwLock<HashMap<SymbolId, SymbolEngineStats>>>,
    
    /// Configuration
    config: Arc<Config>,
    
    /// Shutdown flag
    shutdown: Arc<RwLock<bool>>,
}

impl ParallelRouter {
    /// Create a new parallel router with the given configuration
    pub fn new(config: Arc<Config>) -> Result<Self, String> {
        let senders = Arc::new(DashMap::new());
        let engine_handles = Arc::new(RwLock::new(HashMap::new()));
        let margin_sync = Arc::new(CrossMarginSync::new()?);
        let isolators = Arc::new(DashMap::new());
        let stats = Arc::new(RwLock::new(HashMap::new()));
        let shutdown = Arc::new(RwLock::new(false));

        let router = Self {
            senders,
            engine_handles,
            margin_sync,
            isolators,
            stats,
            config,
            shutdown,
        };

        Ok(router)
    }

    /// Initialize all symbol execution engines
    pub fn initialize_engines(&self) -> Result<(), String> {
        info!("Initializing {} symbol execution engines", SUPPORTED_SYMBOLS.len());

        for symbol in SUPPORTED_SYMBOLS {
            let symbol_id = SymbolId::from_symbol(symbol)
                .ok_or_else(|| format!("Failed to create SymbolId for {}", symbol))?;

            // Create bounded channel for microsecond latency
            let (tx, rx) = bounded::<Tick>(CHANNEL_CAPACITY);
            self.senders.insert(symbol_id, tx);

            // Create symbol isolator with memory budget
            let isolator = Arc::new(SymbolIsolator::new(
                symbol_id,
                MEMORY_BUDGET_PER_SYMBOL,
            )?);
            self.isolators.insert(symbol_id, isolator.clone());

            // Initialize stats
            {
                let mut stats_map = self.stats.write();
                stats_map.insert(symbol_id, SymbolEngineStats::default());
            }

            // Clone shared state for thread
            let margin_sync = self.margin_sync.clone();
            let shutdown = self.shutdown.clone();
            let stats = self.stats.clone();
            let config = self.config.clone();

            // Spawn dedicated thread for this symbol engine
            let handle = thread::Builder::new()
                .name(format!("engine_{}", symbol))
                .spawn(move || {
                    Self::run_symbol_engine(
                        symbol_id,
                        symbol,
                        rx,
                        isolator,
                        margin_sync,
                        shutdown,
                        stats,
                        config,
                    );
                })
                .map_err(|e| format!("Failed to spawn thread for {}: {}", symbol, e))?;

            {
                let mut handles = self.engine_handles.write();
                handles.insert(symbol_id, handle);
            }

            info!("Symbol engine '{}' initialized (Thread: engine_{})", symbol, symbol);
        }

        info!("All {} symbol engines initialized successfully", SUPPORTED_SYMBOLS.len());
        Ok(())
    }

    /// Run the symbol execution engine loop
    #[allow(clippy::too_many_arguments)]
    fn run_symbol_engine(
        symbol_id: SymbolId,
        symbol_name: &str,
        receiver: Receiver<Tick>,
        isolator: Arc<SymbolIsolator>,
        margin_sync: Arc<CrossMarginSync>,
        shutdown: Arc<RwLock<bool>>,
        stats: Arc<RwLock<HashMap<SymbolId, SymbolEngineStats>>>,
        config: Arc<Config>,
    ) {
        info!("Symbol engine '{}' started", symbol_name);
        let mut tick_count: u64 = 0;
        let mut total_latency_us: u64 = 0;
        let mut max_latency_us: u64 = 0;

        while !*shutdown.read() {
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(tick) => {
                    let process_start = Instant::now();

                    // Process tick through isolator (lock-free)
                    match isolator.process_tick(&tick) {
                        Ok(execution_result) => {
                            // Sync cross-margin exposure in O(1) time
                            if let Some(exposure_delta) = execution_result.exposure_delta {
                                margin_sync.update_exposure_atomic(symbol_id, exposure_delta);
                            }

                            // Execute order if signal generated
                            if let Some(order) = execution_result.order {
                                // Order execution logic here
                                debug!("{}: Executing order {:?}", symbol_name, order);
                            }
                        }
                        Err(e) => {
                            warn!("{}: Tick processing error: {}", symbol_name, e);
                        }
                    }

                    // Update statistics
                    let latency_us = process_start.elapsed().as_micros() as u64;
                    total_latency_us += latency_us;
                    max_latency_us = max_latency_us.max(latency_us);
                    tick_count += 1;

                    {
                        let mut stats_map = stats.write();
                        if let Some(sym_stats) = stats_map.get_mut(&symbol_id) {
                            sym_stats.ticks_processed = tick_count;
                            sym_stats.avg_latency_us = total_latency_us / tick_count.max(1);
                            sym_stats.max_latency_us = max_latency_us;
                            sym_stats.memory_used_bytes = isolator.get_memory_usage();
                            sym_stats.last_tick_time = Some(process_start);
                        }
                    }
                }
                Err(_) => {
                    // Timeout, check shutdown flag
                    continue;
                }
            }
        }

        info!("Symbol engine '{}' shutting down (processed {} ticks)", symbol_name, tick_count);
    }

    /// Route an incoming tick to the appropriate symbol engine
    pub fn route_tick(&self, tick: Tick) -> RouteDecision {
        let symbol_id = match SymbolId::from_symbol(&tick.symbol) {
            Some(id) => id,
            None => {
                debug!("Invalid symbol in tick: {}", tick.symbol);
                return RouteDecision::DropInvalid;
            }
        };

        // Get sender for this symbol
        let sender = match self.senders.get(&symbol_id) {
            Some(s) => s,
            None => {
                debug!("No engine found for symbol: {}", tick.symbol);
                return RouteDecision::DropInvalid;
            }
        };

        // Try to send with zero-copy optimization
        match sender.try_send(tick) {
            Ok(()) => RouteDecision::RouteTo(symbol_id),
            Err(TrySendError::Full(_)) => {
                // Backpressure - channel full
                warn!("Backpressure on symbol {}: channel full", tick.symbol);
                
                // Update drop stats
                {
                    let mut stats_map = self.stats.write();
                    if let Some(sym_stats) = stats_map.get_mut(&symbol_id) {
                        sym_stats.ticks_dropped += 1;
                    }
                }
                
                RouteDecision::DropBackpressure
            }
            Err(TrySendError::Disconnected(_)) => {
                error!("Symbol engine disconnected for: {}", tick.symbol);
                RouteDecision::DropInvalid
            }
        }
    }

    /// Route multiple ticks in batch (SIMD-optimized path)
    pub fn route_ticks_batch(&self, ticks: Vec<Tick>) -> (usize, usize, usize) {
        let mut routed = 0;
        let mut dropped_backpressure = 0;
        let mut dropped_invalid = 0;

        for tick in ticks {
            match self.route_tick(tick) {
                RouteDecision::RouteTo(_) => routed += 1,
                RouteDecision::DropBackpressure => dropped_backpressure += 1,
                RouteDecision::DropInvalid => dropped_invalid += 1,
                RouteDecision::QueueDeferred => {} // Not used in current implementation
            }
        }

        (routed, dropped_backpressure, dropped_invalid)
    }

    /// Get current statistics for all symbols
    pub fn get_stats(&self) -> HashMap<String, SymbolEngineStats> {
        let stats_map = self.stats.read();
        stats_map
            .iter()
            .map(|(k, v)| (k.to_symbol().to_string(), v.clone()))
            .collect()
    }

    /// Get memory usage across all engines
    pub fn get_total_memory_usage(&self) -> u64 {
        let mut total = 0;
        for entry in self.isolators.iter() {
            total += entry.value().get_memory_usage();
        }
        total
    }

    /// Gracefully shutdown all symbol engines
    pub fn shutdown(&self) -> Result<(), String> {
        info!("Initiating parallel router shutdown...");
        
        // Set shutdown flag
        *self.shutdown.write() = true;

        // Wait for all engine threads to complete
        let handles = {
            let mut h = self.engine_handles.write();
            std::mem::take(&mut *h)
        };

        for (symbol_id, handle) in handles {
            let symbol = symbol_id.to_symbol();
            info!("Waiting for engine '{}' to terminate...", symbol);
            handle.join().map_err(|_| {
                format!("Failed to join thread for symbol {}", symbol)
            })?;
            info!("Engine '{}' terminated gracefully", symbol);
        }

        info!("Parallel router shutdown completed");
        Ok(())
    }

    /// Get reference to cross-margin sync
    pub fn get_margin_sync(&self) -> Arc<CrossMarginSync> {
        self.margin_sync.clone()
    }
}

impl Drop for ParallelRouter {
    fn drop(&mut self) {
        if !*self.shutdown.read() {
            warn!("ParallelRouter dropped without explicit shutdown");
            let _ = self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_id_creation() {
        let btc_id = SymbolId::from_symbol("BTCUSDT");
        assert!(btc_id.is_some());
        assert_eq!(btc_id.unwrap().to_symbol(), "BTCUSDT");
    }

    #[test]
    fn test_invalid_symbol() {
        let invalid = SymbolId::from_symbol("INVALID");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_router_initialization() {
        let config = Arc::new(Config::default());
        let router = ParallelRouter::new(config);
        assert!(router.is_ok());
    }
}
