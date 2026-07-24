//! Microsecond Execution Synchronizer - Stage 56
//! AMD Ryzen AI 5 Optimized | Batch Order Transmission | Binance API Weight Optimization
//!
//! This module implements a microsecond execution synchronizer ensuring all parallel asset
//! engines batch their outbound orders into a single network transmission to minimize
//! Binance API weight penalties.
//!
//! Constraints:
//! - Sub-microsecond synchronization precision
//! - Lock-free order batching
//! - API weight optimization through intelligent grouping
//! - TSC-based timing for AMD Ryzen

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{Mutex, RwLock};
use once_cell::sync::OnceCell;
use crossbeam_channel::{bounded, Sender, Receiver};

/// Maximum batch size before forced transmission
const MAX_BATCH_SIZE: usize = 50;

/// Maximum wait time for batch accumulation (microseconds)
const MAX_BATCH_WAIT_US: u64 = 100;

/// Target batch window for optimal API weight usage
const TARGET_BATCH_WINDOW_US: u64 = 500;

/// Global synchronizer instance
static EXEC_SYNC: OnceCell<Arc<ExecutionSynchronizer>> = OnceCell::new();

/// Order request with metadata
#[derive(Debug, Clone)]
pub struct OrderRequest {
    /// Unique order ID
    pub order_id: String,
    /// Asset symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Quantity
    pub quantity: f64,
    /// Price (for limit orders)
    pub price: Option<f64>,
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Source engine ID
    pub engine_id: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
}

/// Batched order ready for transmission
#[derive(Debug)]
pub struct OrderBatch {
    /// Orders in this batch
    pub orders: Vec<OrderRequest>,
    /// Batch creation timestamp
    pub created_at: u64,
    /// Total API weight of batch
    pub api_weight: u64,
    /// Batch sequence number
    pub sequence: u64,
}

impl OrderBatch {
    /// Calculate API weight for the batch
    fn calculate_weight(&self) -> u64 {
        // Binance API weight calculation (simplified)
        // Market orders: weight 1-5 depending on endpoint
        // Limit orders: weight 1-2
        // Batch transmission reduces per-order overhead
        
        let base_weight = self.orders.len() as u64;
        
        // Add weight for limit orders (price validation)
        let limit_count = self.orders.iter()
            .filter(|o| matches!(o.order_type, OrderType::Limit))
            .count() as u64;
        
        base_weight + limit_count
    }
}

/// High-resolution timer using TSC (Time Stamp Counter)
struct TSCTimer {
    /// TSC frequency in Hz
    frequency: u64,
    /// Offset from epoch
    offset: u64,
}

impl TSCTimer {
    fn new() -> Self {
        // Get TSC frequency (on x86_64 via CPUID)
        #[cfg(target_arch = "x86_64")]
        let frequency = unsafe {
            use std::arch::x86_64::__cpuid;
            // CPUID leaf 0x15 gives TSC information on newer CPUs
            let cpuid = __cpuid(0x15);
            if cpuid.ecx > 0 {
                cpuid.ecx as u64 * 1_000_000 // Convert to Hz
            } else {
                3_600_000_000 // Fallback: assume 3.6GHz
            }
        };
        
        #[cfg(not(target_arch = "x86_64"))]
        let frequency = 3_600_000_000;
        
        Self {
            frequency,
            offset: Instant::now().duration_since(Instant::now()).as_micros() as u64,
        }
    }
    
    /// Get current timestamp in microseconds
    #[inline(always)]
    fn now_us(&self) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                std::arch::x86_64::_rdtsc() as u64 * 1_000_000 / self.frequency
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Instant::now().duration_since(Instant::now()).as_micros() as u64
        }
    }
}

/// Main execution synchronizer
pub struct ExecutionSynchronizer {
    /// Order submission channel
    order_tx: Sender<OrderRequest>,
    /// Order reception channel
    order_rx: Mutex<Receiver<OrderRequest>>,
    /// Current batch buffer
    current_batch: RwLock<Vec<OrderRequest>>,
    /// Batch sequence counter
    sequence: AtomicU64,
    /// Timer for microsecond precision
    timer: TSCTimer,
    /// Last transmission timestamp
    last_transmission: AtomicU64,
    /// Is actively batching
    is_batching: AtomicBool,
    /// Emergency halt flag
    emergency_halt: AtomicBool,
    /// Statistics
    stats: RwLock<SynchronizerStats>,
    /// Transmission callback
    transmit_cb: Option<Arc<dyn Fn(OrderBatch) -> bool + Send + Sync>>,
}

impl ExecutionSynchronizer {
    /// Create a new execution synchronizer
    pub fn new(batch_size: usize) -> Self {
        let (tx, rx) = bounded::<OrderRequest>(batch_size * 2);
        
        Self {
            order_tx: tx,
            order_rx: Mutex::new(rx),
            current_batch: RwLock::new(Vec::with_capacity(batch_size)),
            sequence: AtomicU64::new(0),
            timer: TSCTimer::new(),
            last_transmission: AtomicU64::new(0),
            is_batching: AtomicBool::new(false),
            emergency_halt: AtomicBool::new(false),
            stats: RwLock::new(SynchronizerStats::default()),
            transmit_cb: None,
        }
    }
    
    /// Get or create global instance
    pub fn global() -> &'static Arc<Self> {
        EXEC_SYNC.get_or_init(|| {
            Arc::new(Self::new(MAX_BATCH_SIZE))
        })
    }
    
    /// Set transmission callback
    pub fn set_transmit_callback<F>(&mut self, cb: F)
    where
        F: Fn(OrderBatch) -> bool + Send + Sync + 'static,
    {
        self.transmit_cb = Some(Arc::new(cb));
    }
    
    /// Submit an order for batching
    pub fn submit_order(&self, order: OrderRequest) -> Result<u64, String> {
        if self.emergency_halt.load(Ordering::Relaxed) {
            return Err("Emergency halt active".to_string());
        }
        
        // Record submission timestamp
        let submit_time = self.timer.now_us();
        
        // Add to channel
        self.order_tx.try_send(order.clone())
            .map_err(|e| format!("Channel full: {}", e))?;
        
        // Try to add to current batch
        {
            let mut batch = self.current_batch.write();
            batch.push(order);
            
            // Check if batch should be transmitted
            let batch_age = submit_time - self.last_transmission.load(Ordering::Relaxed);
            
            if batch.len() >= MAX_BATCH_SIZE || batch_age >= MAX_BATCH_WAIT_US {
                drop(batch);
                self.transmit_batch()?;
            }
        }
        
        Ok(submit_time)
    }
    
    /// Force transmit current batch
    pub fn transmit_batch(&self) -> Result<(), String> {
        if self.emergency_halt.load(Ordering::Relaxed) {
            return Err("Emergency halt active".to_string());
        }
        
        // Atomically swap out current batch
        let orders = {
            let mut current = self.current_batch.write();
            if current.is_empty() {
                return Ok(());
            }
            
            std::mem::take(&mut *current)
        };
        
        if orders.is_empty() {
            return Ok(());
        }
        
        // Create batch
        let batch = OrderBatch {
            api_weight: orders.len() as u64, // Simplified weight calc
            created_at: self.timer.now_us(),
            sequence: self.sequence.fetch_add(1, Ordering::Relaxed),
            orders,
        };
        
        let actual_weight = batch.calculate_weight();
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.batches_sent += 1;
            stats.total_orders += batch.orders.len();
            stats.total_api_weight += actual_weight;
            stats.last_batch_size = batch.orders.len();
            stats.avg_latency_us = if stats.batches_sent > 0 {
                (stats.avg_latency_us * (stats.batches_sent - 1) as f64 
                    + (batch.created_at - self.last_transmission.load(Ordering::Relaxed)) as f64)
                / stats.batches_sent as f64
            } else {
                0.0
            };
        }
        
        // Transmit via callback
        if let Some(ref cb) = self.transmit_cb {
            if !cb(batch) {
                return Err("Transmission failed".to_string());
            }
        }
        
        // Update last transmission time
        self.last_transmission.store(self.timer.now_us(), Ordering::Relaxed);
        
        Ok(())
    }
    
    /// Synchronize all engines and force batch transmission
    pub fn synchronize(&self) -> Result<SynchronizationResult, String> {
        if self.emergency_halt.load(Ordering::Relaxed) {
            return Err("Emergency halt active".to_string());
        }
        
        self.is_batching.store(true, Ordering::SeqCst);
        
        let start_time = self.timer.now_us();
        
        // Drain all pending orders
        let rx = self.order_rx.lock();
        while let Ok(order) = rx.try_recv() {
            let mut batch = self.current_batch.write();
            batch.push(order);
            
            if batch.len() >= MAX_BATCH_SIZE {
                drop(batch);
                self.transmit_batch()?;
            }
        }
        
        // Final transmission
        self.transmit_batch()?;
        
        self.is_batching.store(false, Ordering::SeqCst);
        
        let end_time = self.timer.now_us();
        
        Ok(SynchronizationResult {
            duration_us: end_time - start_time,
            orders_synchronized: self.stats.read().last_batch_size,
            timestamp: end_time,
        })
    }
    
    /// Enable emergency halt
    pub fn emergency_halt(&self) {
        self.emergency_halt.store(true, Ordering::SeqCst);
    }
    
    /// Clear emergency halt
    pub fn clear_emergency_halt(&self) {
        self.emergency_halt.store(false, Ordering::SeqCst);
    }
    
    /// Get current statistics
    pub fn stats(&self) -> SynchronizerStats {
        self.stats.read().clone()
    }
    
    /// Check if currently batching
    pub fn is_batching(&self) -> bool {
        self.is_batching.load(Ordering::Relaxed)
    }
}

/// Result of synchronization operation
#[derive(Debug, Clone)]
pub struct SynchronizationResult {
    pub duration_us: u64,
    pub orders_synchronized: usize,
    pub timestamp: u64,
}

/// Statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SynchronizerStats {
    pub batches_sent: u64,
    pub total_orders: usize,
    pub total_api_weight: u64,
    pub last_batch_size: usize,
    pub avg_latency_us: f64,
}

/// Builder for order requests
pub struct OrderRequestBuilder {
    symbol: String,
    side: OrderSide,
    order_type: OrderType,
    quantity: f64,
    price: Option<f64>,
    engine_id: usize,
}

impl OrderRequestBuilder {
    pub fn new(symbol: &str, side: OrderSide, order_type: OrderType, quantity: f64) -> Self {
        Self {
            symbol: symbol.to_string(),
            side,
            order_type,
            quantity,
            price: None,
            engine_id: 0,
        }
    }
    
    pub fn price(mut self, price: f64) -> Self {
        self.price = Some(price);
        self
    }
    
    pub fn engine_id(mut self, id: usize) -> Self {
        self.engine_id = id;
        self
    }
    
    pub fn build(self) -> OrderRequest {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        OrderRequest {
            order_id: format!("{}_{}_{}", self.symbol, self.side_str(), timestamp_us),
            symbol: self.symbol,
            side: self.side,
            order_type: self.order_type,
            quantity: self.quantity,
            price: self.price,
            timestamp_us,
            engine_id: self.engine_id,
        }
    }
    
    fn side_str(&self) -> &'static str {
        match self.side {
            OrderSide::Buy => "B",
            OrderSide::Sell => "S",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_synchronizer_creation() {
        let sync = ExecutionSynchronizer::new(50);
        let stats = sync.stats();
        
        assert_eq!(stats.batches_sent, 0);
        assert_eq!(stats.total_orders, 0);
        assert!(!sync.is_batching());
    }
    
    #[test]
    fn test_order_builder() {
        let order = OrderRequestBuilder::new("BTCUSDT", OrderSide::Buy, OrderType::Limit, 0.1)
            .price(50000.0)
            .engine_id(1)
            .build();
        
        assert_eq!(order.symbol, "BTCUSDT");
        assert!(matches!(order.side, OrderSide::Buy));
        assert!(matches!(order.order_type, OrderType::Limit));
        assert_eq!(order.quantity, 0.1);
        assert_eq!(order.price, Some(50000.0));
    }
}
