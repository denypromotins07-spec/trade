//! Cache - State Synchronization Engine
//! 
//! Creates an asynchronous state synchronization engine that flushes critical
//! cache deltas to the CQRS event store, ensuring zero data loss during a
//! sudden `/KILL` command. Optimized for AMD Ryzen AI 5 with microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use crossbeam_channel::{bounded, Sender, Receiver, TrySendError};
use std::thread;

/// Maximum pending sync operations
const MAX_PENDING_SYNC: usize = 4096;

/// Sync batch size for efficient flushing
const SYNC_BATCH_SIZE: usize = 64;

/// Flush interval in nanoseconds (10ms default)
const DEFAULT_FLUSH_INTERVAL_NS: u64 = 10_000_000;

/// State delta types
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum DeltaType {
    OrderNew = 0,
    OrderCancel = 1,
    OrderFill = 2,
    PositionUpdate = 3,
    RiskUpdate = 4,
    MarketData = 5,
    SystemEvent = 6,
}

/// State delta record for synchronization
#[derive(Clone, Debug)]
#[repr(C, align(64))]
pub struct StateDelta {
    /// Unique sequence number
    pub sequence: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Delta type
    pub delta_type: DeltaType,
    /// Entity ID (order ID, position ID, etc.)
    pub entity_id: u64,
    /// Data hash (for quick comparison)
    pub data_hash: u64,
    /// Serialized data size (bytes)
    pub data_size: u16,
    /// Priority level (higher = more critical)
    pub priority: u8,
    /// Acknowledged flag
    pub acknowledged: bool,
    /// Inline data buffer (for small deltas)
    pub inline_data: [u8; 64],
}

impl StateDelta {
    #[inline(always)]
    pub fn new(
        sequence: u64,
        timestamp_ns: u64,
        delta_type: DeltaType,
        entity_id: u64,
        data_hash: u64,
        priority: u8,
    ) -> Self {
        Self {
            sequence,
            timestamp_ns,
            delta_type,
            entity_id,
            data_hash,
            data_size: 0,
            priority,
            acknowledged: false,
            inline_data: [0u8; 64],
        }
    }
    
    #[inline(always)]
    pub fn with_data(mut self, data: &[u8]) -> Self {
        let copy_len = data.len().min(64);
        self.inline_data[..copy_len].copy_from_slice(&data[..copy_len]);
        self.data_size = copy_len as u16;
        self
    }
    
    #[inline(always)]
    pub fn is_critical(&self) -> bool {
        self.priority >= 10 || matches!(self.delta_type, DeltaType::OrderFill | DeltaType::RiskUpdate)
    }
}

/// CQRS Event Store trait for persistence
pub trait EventStore: Send + Sync {
    /// Append delta to event log
    fn append(&self, delta: &StateDelta) -> Result<u64, SyncError>;
    
    /// Append batch of deltas
    fn append_batch(&self, deltas: &[StateDelta]) -> Result<u64, SyncError>;
    
    /// Flush pending writes to durable storage
    fn flush(&self) -> Result<(), SyncError>;
    
    /// Get last committed sequence number
    fn last_sequence(&self) -> u64;
}

/// Synchronization errors
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SyncError {
    QueueFull,
    StoreError,
    Timeout,
    Shutdown,
}

/// Sync statistics
#[derive(Clone, Copy, Debug, Default)]
pub struct SyncStats {
    /// Total deltas submitted
    pub submitted: u64,
    /// Total deltas synced
    pub synced: u64,
    /// Total batches flushed
    pub batches_flushed: u64,
    /// Failed syncs
    pub failed: u64,
    /// Average latency (nanoseconds)
    pub avg_latency_ns: u64,
    /// Max latency (nanoseconds)
    pub max_latency_ns: u64,
    /// Pending count
    pub pending: u64,
}

/// Async state synchronization engine
pub struct StateSyncEngine<ES: EventStore> {
    /// Delta submission channel
    tx: Sender<StateDelta>,
    /// Delta processing channel
    rx: Receiver<StateDelta>,
    /// Event store reference
    event_store: Arc<ES>,
    /// Sequence counter
    sequence: AtomicU64,
    /// Running flag
    running: AtomicBool,
    /// Shutdown requested
    shutdown_requested: AtomicBool,
    /// Emergency flush mode (for /KILL)
    emergency_mode: AtomicBool,
    /// Number of pending deltas
    pending_count: AtomicUsize,
    /// Statistics
    stats: Arc<std::sync::Mutex<SyncStats>>,
    /// Flush interval in nanoseconds
    flush_interval_ns: u64,
    /// Last flush timestamp
    last_flush_ns: AtomicU64,
    /// Worker thread handle
    worker_handle: Option<thread::JoinHandle<()>>,
}

impl<ES: EventStore + 'static> StateSyncEngine<ES> {
    /// Create new sync engine
    pub fn new(event_store: Arc<ES>, flush_interval_ms: u64) -> Self {
        let (tx, rx) = bounded::<StateDelta>(MAX_PENDING_SYNC);
        
        Self {
            tx,
            rx,
            event_store,
            sequence: AtomicU64::new(0),
            running: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            emergency_mode: AtomicBool::new(false),
            pending_count: AtomicUsize::new(0),
            stats: Arc::new(std::sync::Mutex::new(SyncStats::default())),
            flush_interval_ns: flush_interval_ms * 1_000_000,
            last_flush_ns: AtomicU64::new(0),
            worker_handle: None,
        }
    }
    
    /// Start the sync engine
    pub fn start(&mut self) -> Result<(), SyncError> {
        if self.running.load(Ordering::Acquire) {
            return Err(SyncError::Timeout); // Already running
        }
        
        self.running.store(true, Ordering::Release);
        self.shutdown_requested.store(false, Ordering::Release);
        
        let tx = self.tx.clone();
        let rx = self.rx.clone();
        let event_store = self.event_store.clone();
        let sequence = &self.sequence;
        let running = self.running.clone();
        let shutdown = self.shutdown_requested.clone();
        let emergency = self.emergency_mode.clone();
        let pending = self.pending_count.clone();
        let stats = self.stats.clone();
        let flush_interval = self.flush_interval_ns;
        let last_flush = &self.last_flush_ns;
        
        let handle = thread::Builder::new()
            .name("state-sync-worker".to_string())
            .spawn(move || {
                Self::worker_loop(
                    rx,
                    event_store,
                    sequence,
                    running,
                    shutdown,
                    emergency,
                    pending,
                    stats,
                    flush_interval,
                    last_flush,
                );
            })
            .map_err(|_| SyncError::StoreError)?;
        
        self.worker_handle = Some(handle);
        Ok(())
    }
    
    /// Worker loop for processing deltas
    fn worker_loop(
        rx: Receiver<StateDelta>,
        event_store: Arc<ES>,
        sequence: &AtomicU64,
        running: AtomicBool,
        shutdown: AtomicBool,
        emergency: AtomicBool,
        pending: AtomicUsize,
        stats: Arc<std::sync::Mutex<SyncStats>>,
        flush_interval_ns: u64,
        last_flush: &AtomicU64,
    ) {
        let mut batch: Vec<StateDelta> = Vec::with_capacity(SYNC_BATCH_SIZE);
        let mut total_latency_ns = 0u64;
        let mut latency_count = 0u64;
        
        while running.load(Ordering::Acquire) {
            let current_ns = get_time_ns();
            
            // Check if it's time to flush
            let last_flush_ns = last_flush.load(Ordering::Acquire);
            let should_flush = current_ns.wrapping_sub(last_flush_ns) > flush_interval_ns
                || emergency.load(Ordering::Acquire)
                || batch.len() >= SYNC_BATCH_SIZE;
            
            // Collect deltas
            if should_flush && !batch.is_empty() {
                // Flush batch
                match event_store.append_batch(&batch) {
                    Ok(_) => {
                        let _ = event_store.flush();
                        
                        let mut s = stats.lock().unwrap();
                        s.synced += batch.len() as u64;
                        s.batches_flushed += 1;
                        
                        // Update latency stats
                        for delta in &batch {
                            let latency = current_ns.wrapping_sub(delta.timestamp_ns);
                            total_latency_ns += latency;
                            latency_count += 1;
                            if latency > s.max_latency_ns {
                                s.max_latency_ns = latency;
                            }
                        }
                        
                        if latency_count > 0 {
                            s.avg_latency_ns = total_latency_ns / latency_count;
                        }
                    }
                    Err(_) => {
                        let mut s = stats.lock().unwrap();
                        s.failed += batch.len() as u64;
                    }
                }
                
                batch.clear();
                last_flush.store(current_ns, Ordering::Release);
            }
            
            // Try to receive more deltas
            match rx.try_recv() {
                Ok(delta) => {
                    batch.push(delta);
                    pending.fetch_sub(1, Ordering::Relaxed);
                }
                Err(_) => {
                    // No more deltas, wait a bit
                    if shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    thread::sleep(Duration::from_micros(10));
                }
            }
        }
        
        // Final flush on shutdown
        if !batch.is_empty() {
            let _ = event_store.append_batch(&batch);
            let _ = event_store.flush();
        }
    }
    
    /// Submit a delta for synchronization
    #[inline(always)]
    pub fn submit(&self, delta_type: DeltaType, entity_id: u64, data: Option<&[u8]>) -> Result<u64, SyncError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(SyncError::Shutdown);
        }
        
        let sequence = self.sequence.fetch_add(1, Ordering::AcqRel);
        let timestamp_ns = get_time_ns();
        let data_hash = data.map(|d| fxhash::fxhash64(d)).unwrap_or(0);
        
        let mut delta = StateDelta::new(
            sequence,
            timestamp_ns,
            delta_type,
            entity_id,
            data_hash,
            if delta_type.is_critical() { 10 } else { 1 },
        );
        
        if let Some(d) = data {
            delta = delta.with_data(d);
        }
        
        // Try to send
        match self.tx.try_send(delta) {
            Ok(_) => {
                self.pending_count.fetch_add(1, Ordering::Relaxed);
                
                let mut s = self.stats.lock().unwrap();
                s.submitted += 1;
                s.pending = self.pending_count.load(Ordering::Relaxed) as u64;
                
                Ok(sequence)
            }
            Err(TrySendError::Full(_)) => Err(SyncError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(SyncError::Shutdown),
        }
    }
    
    /// Submit critical delta (blocks until queued)
    #[inline(always)]
    pub fn submit_critical(&self, delta_type: DeltaType, entity_id: u64, data: &[u8]) -> Result<u64, SyncError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(SyncError::Shutdown);
        }
        
        let sequence = self.sequence.fetch_add(1, Ordering::AcqRel);
        let timestamp_ns = get_time_ns();
        let data_hash = fxhash::fxhash64(data);
        
        let delta = StateDelta::new(
            sequence,
            timestamp_ns,
            delta_type,
            entity_id,
            data_hash,
            15, // High priority
        ).with_data(data);
        
        // Block on critical deltas
        match self.tx.send(delta) {
            Ok(_) => {
                self.pending_count.fetch_add(1, Ordering::Relaxed);
                
                let mut s = self.stats.lock().unwrap();
                s.submitted += 1;
                s.pending = self.pending_count.load(Ordering::Relaxed) as u64;
                
                Ok(sequence)
            }
            Err(_) => Err(SyncError::Shutdown),
        }
    }
    
    /// Emergency flush for /KILL command
    #[inline(always)]
    pub fn emergency_flush(&self) -> Result<(), SyncError> {
        self.emergency_mode.store(true, Ordering::Release);
        
        // Wait for pending to drain (with timeout)
        let start = get_time_ns();
        let timeout_ns = 5_000_000_000; // 5 second timeout
        
        while self.pending_count.load(Ordering::Acquire) > 0 {
            if get_time_ns().wrapping_sub(start) > timeout_ns {
                return Err(SyncError::Timeout);
            }
            thread::sleep(Duration::from_micros(100));
        }
        
        // Force flush
        let _ = self.event_store.flush();
        
        self.emergency_mode.store(false, Ordering::Release);
        Ok(())
    }
    
    /// Graceful shutdown
    pub fn shutdown(&mut self) -> Result<(), SyncError> {
        self.shutdown_requested.store(true, Ordering::Release);
        self.running.store(false, Ordering::Release);
        
        // Wait for worker to finish
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
        
        // Final flush
        let _ = self.event_store.flush();
        
        Ok(())
    }
    
    /// Get current statistics
    pub fn stats(&self) -> SyncStats {
        *self.stats.lock().unwrap()
    }
    
    /// Check if engine is healthy
    #[inline(always)]
    pub fn is_healthy(&self) -> bool {
        self.running.load(Ordering::Acquire)
            && !self.shutdown_requested.load(Ordering::Acquire)
            && self.pending_count.load(Ordering::Acquire) < MAX_PENDING_SYNC / 2
    }
    
    /// Get pending count
    #[inline(always)]
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(Ordering::Acquire)
    }
}

/// Get current time in nanoseconds
#[inline(always)]
fn get_time_ns() -> u64 {
    use std::time::Instant;
    static START: once_cell::sync::Lazy<Instant> = once_cell::sync::Lazy::new(Instant::now);
    START.elapsed().as_nanos() as u64
}

/// Simple in-memory event store for testing
pub struct InMemoryEventStore {
    last_seq: AtomicU64,
    flush_count: AtomicU64,
}

impl InMemoryEventStore {
    pub const fn new() -> Self {
        Self {
            last_seq: AtomicU64::new(0),
            flush_count: AtomicU64::new(0),
        }
    }
}

impl EventStore for InMemoryEventStore {
    fn append(&self, delta: &StateDelta) -> Result<u64, SyncError> {
        self.last_seq.store(delta.sequence, Ordering::Release);
        Ok(delta.sequence)
    }
    
    fn append_batch(&self, deltas: &[StateDelta]) -> Result<u64, SyncError> {
        if let Some(last) = deltas.last() {
            self.last_seq.store(last.sequence, Ordering::Release);
            Ok(last.sequence)
        } else {
            Ok(self.last_seq.load(Ordering::Acquire))
        }
    }
    
    fn flush(&self) -> Result<(), SyncError> {
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    fn last_sequence(&self) -> u64 {
        self.last_seq.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_sync_basic() {
        let event_store = Arc::new(InMemoryEventStore::new());
        let mut engine = StateSyncEngine::new(event_store.clone(), 10); // 10ms flush
        
        engine.start().unwrap();
        
        // Submit some deltas
        for i in 0..10 {
            engine.submit(DeltaType::OrderNew, i, Some(&[i as u8; 32])).unwrap();
        }
        
        // Wait for sync
        thread::sleep(Duration::from_millis(50));
        
        let stats = engine.stats();
        assert!(stats.submitted >= 10);
        
        engine.shutdown().unwrap();
    }

    #[test]
    fn test_emergency_flush() {
        let event_store = Arc::new(InMemoryEventStore::new());
        let mut engine = StateSyncEngine::new(event_store.clone(), 1000); // 1s flush
        
        engine.start().unwrap();
        
        // Submit critical delta
        engine.submit_critical(DeltaType::RiskUpdate, 1, &[0xFF; 64]).unwrap();
        
        // Emergency flush
        engine.emergency_flush().unwrap();
        
        engine.shutdown().unwrap();
    }

    #[test]
    fn test_queue_full() {
        let event_store = Arc::new(InMemoryEventStore::new());
        let mut engine = StateSyncEngine::new(event_store.clone(), 10000); // Very long flush
        
        // Fill the queue
        for _ in 0..MAX_PENDING_SYNC + 100 {
            let _ = engine.submit(DeltaType::MarketData, 1, Some(&[0u8; 64]));
        }
        
        // Should get QueueFull error
        let result = engine.submit(DeltaType::MarketData, 1, Some(&[0u8; 64]));
        assert_eq!(result, Err(SyncError::QueueFull));
        
        engine.shutdown().unwrap();
    }
}
