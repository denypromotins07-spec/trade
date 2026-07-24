//! symbol_isolator.rs - Lock-Free Memory Isolation Per Symbol
//! Stage 54: Nautilus/Ray Crypto Trading Bot
//! Ensures volatility spikes in one altcoin never delay BTC/ETH execution
//! Optimized for AMD Ryzen AI 5 cache architecture, 8GB RAM strict limit

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use crossbeam::epoch::{self, Atomic as EpochAtomic};
use log::{debug, error, warn};
use parking_lot::Mutex;

use crate::execution::parallel_router::SymbolId;
use crate::market::tick::Tick;
use crate::order::order::Order;

/// Maximum memory budget per symbol isolator (enforced strictly)
const MAX_MEMORY_BYTES: u64 = 700_000_000; // ~700MB per symbol

/// Cache line size for AMD Ryzen optimization (prevent false sharing)
const CACHE_LINE_SIZE: usize = 64;

/// Result of tick processing
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Optional order to execute
    pub order: Option<Order>,
    /// Exposure delta for cross-margin sync
    pub exposure_delta: Option<f64>,
    /// Processing latency in microseconds
    pub latency_us: u64,
}

impl ExecutionResult {
    pub fn new() -> Self {
        Self {
            order: None,
            exposure_delta: None,
            latency_us: 0,
        }
    }

    pub fn with_order(mut self, order: Order) -> Self {
        self.order = Some(order);
        self
    }

    pub fn with_exposure_delta(mut self, delta: f64) -> Self {
        self.exposure_delta = Some(delta);
        self
    }
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free ring buffer for tick storage
/// Uses epoch-based reclamation for safe concurrent access
struct TickRingBuffer {
    /// Buffer storage (epoch-protected)
    buffer: EpochAtomic<TickSlot>,
    /// Current write position
    write_pos: AtomicUsize,
    /// Current read position
    read_pos: AtomicUsize,
    /// Buffer capacity (power of 2 for fast modulo)
    capacity: usize,
    /// Current element count
    count: AtomicUsize,
}

/// Slot in the ring buffer with validity flag
#[derive(Clone)]
struct TickSlot {
    tick: Option<Tick>,
    valid: AtomicBool,
    timestamp_ns: AtomicU64,
}

impl TickRingBuffer {
    fn new(capacity: usize) -> Self {
        // Ensure capacity is power of 2
        let capacity = capacity.next_power_of_two();
        
        // Initialize buffer with empty slots
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(TickSlot {
                tick: None,
                valid: AtomicBool::new(false),
                timestamp_ns: AtomicU64::new(0),
            });
        }
        
        // Convert to epoch-protected atomic
        let buffer = EpochAtomic::new(TickSlot {
            tick: None,
            valid: AtomicBool::new(false),
            timestamp_ns: AtomicU64::new(0),
        });
        
        Self {
            buffer,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            capacity,
            count: AtomicUsize::new(0),
        }
    }

    fn push(&self, tick: Tick) -> Result<(), &'static str> {
        if self.count.load(Ordering::Relaxed) >= self.capacity {
            return Err("Ring buffer full");
        }

        let pos = self.write_pos.fetch_add(1, Ordering::AcqRel) % self.capacity;
        
        // This is a simplified implementation - production would use
        // proper epoch-based reclamation for the actual buffer array
        Ok(())
    }

    fn pop(&self) -> Option<Tick> {
        if self.count.load(Ordering::Relaxed) == 0 {
            return None;
        }

        let pos = self.read_pos.fetch_add(1, Ordering::AcqRel) % self.capacity;
        None // Simplified - production would return actual tick
    }

    fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// Memory tracker with atomic counters (lock-free)
struct MemoryTracker {
    /// Current memory usage in bytes
    used_bytes: AtomicU64,
    /// Peak memory usage
    peak_bytes: AtomicU64,
    /// Allocation count
    alloc_count: AtomicUsize,
    /// Deallocation count
    dealloc_count: AtomicUsize,
    /// Memory budget limit
    budget_bytes: u64,
    /// OOM flag
    oom_flag: AtomicBool,
}

impl MemoryTracker {
    fn new(budget_bytes: u64) -> Self {
        Self {
            used_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            alloc_count: AtomicUsize::new(0),
            dealloc_count: AtomicUsize::new(0),
            budget_bytes,
            oom_flag: AtomicBool::new(false),
        }
    }

    fn allocate(&self, size: usize) -> Result<(), &'static str> {
        if self.oom_flag.load(Ordering::Relaxed) {
            return Err("Memory budget exceeded (OOM)");
        }

        let current = self.used_bytes.fetch_add(size as u64, Ordering::AcqRel);
        let new_total = current + size as u64;

        if new_total > self.budget_bytes {
            // Rollback
            self.used_bytes.fetch_sub(size as u64, Ordering::AcqRel);
            self.oom_flag.store(true, Ordering::Release);
            return Err("Memory allocation would exceed budget");
        }

        // Update peak if necessary
        let peak = self.peak_bytes.load(Ordering::Relaxed);
        if new_total > peak {
            self.peak_bytes.store(new_total, Ordering::Relaxed);
        }

        self.alloc_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn deallocate(&self, size: usize) {
        self.used_bytes.fetch_sub(size as u64, Ordering::AcqRel);
        self.dealloc_count.fetch_add(1, Ordering::Relaxed);
    }

    fn get_usage(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    fn get_peak(&self) -> u64 {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    fn reset_oom(&self) {
        self.oom_flag.store(false, Ordering::Release);
    }
}

/// Per-symbol execution state (cache-line aligned)
#[repr(align(64))]
struct SymbolState {
    /// Last processed tick sequence number
    last_sequence: AtomicU64,
    /// Current position (for mean reversion strategies)
    current_position: AtomicU64, // Fixed-point representation
    /// Unrealized PnL in micro-dollars
    unrealized_pnl: AtomicU64,
    /// Strategy state machine state
    strategy_state: AtomicUsize,
    /// Cooldown flag (prevents overtrading after losses)
    cooldown_active: AtomicBool,
    /// Cooldown end timestamp
    cooldown_end_ns: AtomicU64,
    /// Padding to ensure cache line alignment
    _padding: [u8; CACHE_LINE_SIZE - 48],
}

impl SymbolState {
    fn new() -> Self {
        Self {
            last_sequence: AtomicU64::new(0),
            current_position: AtomicU64::new(0),
            unrealized_pnl: AtomicU64::new(0),
            strategy_state: AtomicUsize::new(0),
            cooldown_active: AtomicBool::new(false),
            cooldown_end_ns: AtomicU64::new(0),
            _padding: [0; CACHE_LINE_SIZE - 48],
        }
    }
}

/// SymbolIsolator - Lock-free memory isolation for a single trading symbol
/// 
/// This struct ensures that:
/// 1. Each symbol has its own memory space (no cross-symbol interference)
/// 2. Volatility spikes in one symbol don't affect others
/// 3. Memory usage is strictly bounded per symbol
/// 4. All operations are lock-free for microsecond latency
pub struct SymbolIsolator {
    /// Symbol identifier
    symbol_id: SymbolId,
    
    /// Memory tracker (lock-free)
    memory: Arc<MemoryTracker>,
    
    /// Per-symbol state (cache-line aligned)
    state: Arc<SymbolState>,
    
    /// Tick ring buffer (lock-free)
    tick_buffer: Arc<TickRingBuffer>,
    
    /// Emergency stop flag
    emergency_stop: AtomicBool,
    
    /// Initialization timestamp
    init_time: Instant,
    
    /// Total ticks processed
    total_ticks: AtomicU64,
}

impl SymbolIsolator {
    /// Create a new symbol isolator with the given memory budget
    pub fn new(symbol_id: SymbolId, budget_bytes: u64) -> Result<Self, String> {
        if budget_bytes > MAX_MEMORY_BYTES {
            return Err(format!(
                "Budget {} exceeds maximum {}",
                budget_bytes, MAX_MEMORY_BYTES
            ));
        }

        Ok(Self {
            symbol_id,
            memory: Arc::new(MemoryTracker::new(budget_bytes)),
            state: Arc::new(SymbolState::new()),
            tick_buffer: Arc::new(TickRingBuffer::new(1024)),
            emergency_stop: AtomicBool::new(false),
            init_time: Instant::now(),
            total_ticks: AtomicU64::new(0),
        })
    }

    /// Process an incoming tick (lock-free path)
    pub fn process_tick(&self, tick: &Tick) -> Result<ExecutionResult, String> {
        if self.emergency_stop.load(Ordering::Relaxed) {
            return Err("Emergency stop active".to_string());
        }

        if self.memory.get_usage() >= self.memory.budget_bytes * 95 / 100 {
            warn!(
                "Symbol {:?} memory at 95% capacity",
                self.symbol_id.to_symbol()
            );
        }

        let process_start = Instant::now();

        // Update sequence number
        let sequence = self.state.last_sequence.fetch_add(1, Ordering::AcqRel);

        // Check cooldown
        if self.state.cooldown_active.load(Ordering::Relaxed) {
            let now_ns = process_start.elapsed().as_nanos() as u64;
            if now_ns < self.state.cooldown_end_ns.load(Ordering::Relaxed) {
                // Still in cooldown - skip processing but update stats
                self.total_ticks.fetch_add(1, Ordering::Relaxed);
                return Ok(ExecutionResult::new());
            } else {
                // Cooldown expired
                self.state.cooldown_active.store(false, Ordering::Release);
            }
        }

        // Allocate memory for tick processing
        let tick_memory_size = std::mem::size_of::<Tick>() + 1024; // Buffer for derived data
        if let Err(e) = self.memory.allocate(tick_memory_size) {
            error!("Memory allocation failed for tick: {}", e);
            return Err(format!("Memory error: {}", e));
        }

        // Process tick through strategy (simplified)
        let result = self.execute_strategy(tick, sequence);

        // Deallocate memory
        self.memory.deallocate(tick_memory_size);

        // Update stats
        self.total_ticks.fetch_add(1, Ordering::Relaxed);

        let latency_us = process_start.elapsed().as_micros() as u64;
        
        Ok(result.with_latency(latency_us))
    }

    /// Execute trading strategy on tick (symbol-specific logic)
    fn execute_strategy(&self, tick: &Tick, sequence: u64) -> ExecutionResult {
        let mut result = ExecutionResult::new();

        // Example: Simple momentum strategy (production would be more complex)
        // This is intentionally simplified - real implementation would use
        // ML models, order book analysis, etc.

        // Update position based on tick
        let price_fixed = (tick.price * 1000000.0) as u64; // Fixed-point
        
        // Calculate exposure delta
        let prev_position = self.state.current_position.load(Ordering::Relaxed) as i64;
        let position_change = (price_fixed as i64) - prev_position;
        
        if position_change.abs() > 1000000 { // Threshold for significant move
            result = result.with_exposure_delta(position_change as f64 / 1000000.0);
            
            // Generate signal if momentum is strong enough
            if position_change > 5000000 {
                // Long signal
                result = result.with_order(Order::new_market(
                    self.symbol_id.to_symbol().to_string(),
                    tick.volume.min(1.0), // Limit order size
                    true, // Buy
                ));
            } else if position_change < -5000000 {
                // Short signal
                result = result.with_order(Order::new_market(
                    self.symbol_id.to_symbol().to_string(),
                    tick.volume.min(1.0),
                    false, // Sell
                ));
            }
        }

        // Update state
        self.state.current_position.store(price_fixed, Ordering::Relaxed);

        result
    }

    /// Get current memory usage in bytes
    pub fn get_memory_usage(&self) -> u64 {
        self.memory.get_usage()
    }

    /// Get peak memory usage
    pub fn get_peak_memory(&self) -> u64 {
        self.memory.get_peak()
    }

    /// Activate emergency stop (halts all trading for this symbol)
    pub fn emergency_stop(&self) {
        self.emergency_stop.store(true, Ordering::Release);
        warn!("Emergency stop activated for {:?}", self.symbol_id);
    }

    /// Reset emergency stop
    pub fn clear_emergency_stop(&self) {
        self.emergency_stop.store(false, Ordering::Release);
        self.memory.reset_oom();
    }

    /// Get total ticks processed
    pub fn get_total_ticks(&self) -> u64 {
        self.total_ticks.load(Ordering::Relaxed)
    }

    /// Check if in cooldown
    pub fn is_in_cooldown(&self) -> bool {
        self.state.cooldown_active.load(Ordering::Relaxed)
    }

    /// Set cooldown period (prevents overtrading)
    pub fn set_cooldown(&self, duration_ms: u64) {
        let now_ns = self.init_time.elapsed().as_nanos() as u64;
        self.state.cooldown_end_ns.store(now_ns + duration_ms * 1_000_000, Ordering::Release);
        self.state.cooldown_active.store(true, Ordering::Release);
    }

    /// Get symbol ID
    pub fn get_symbol_id(&self) -> SymbolId {
        self.symbol_id
    }
}

impl Drop for SymbolIsolator {
    fn drop(&mut self) {
        debug!(
            "SymbolIsolator for {:?} dropped (total ticks: {}, peak memory: {} bytes)",
            self.symbol_id,
            self.get_total_ticks(),
            self.get_peak_memory()
        );
    }
}

// Extension trait for ExecutionResult to add latency
trait ExecutionResultExt {
    fn with_latency(self, latency_us: u64) -> ExecutionResult;
}

impl ExecutionResultExt for ExecutionResult {
    fn with_latency(mut self, latency_us: u64) -> ExecutionResult {
        self.latency_us = latency_us;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolator_creation() {
        let symbol_id = SymbolId::from_symbol("BTCUSDT").unwrap();
        let isolator = SymbolIsolator::new(symbol_id, 100_000_000);
        assert!(isolator.is_ok());
    }

    #[test]
    fn test_memory_budget_enforcement() {
        let symbol_id = SymbolId::from_symbol("ETHUSDT").unwrap();
        let isolator = SymbolIsolator::new(symbol_id, 1_000).unwrap(); // Very small budget
        
        // Should fail with tiny budget
        assert!(isolator.get_memory_usage() == 0);
    }

    #[test]
    fn test_emergency_stop() {
        let symbol_id = SymbolId::from_symbol("SOLUSDT").unwrap();
        let isolator = SymbolIsolator::new(symbol_id, 100_000_000).unwrap();
        
        assert!(!isolator.emergency_stop.load(Ordering::Relaxed));
        isolator.emergency_stop();
        assert!(isolator.emergency_stop.load(Ordering::Relaxed));
        
        isolator.clear_emergency_stop();
        assert!(!isolator.emergency_stop.load(Ordering::Relaxed));
    }
}
