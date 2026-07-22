//! State Replay Engine - Deterministic CQRS Event Log Replay
//!
//! This module creates a deterministic replay engine that feeds historical CQRS
//! event logs back into the system to verify that state transitions are perfectly
//! reproducible across restarts. Essential for debugging and audit compliance.
//!
//! ## Features
//! - Event log serialization/deserialization
//! - Deterministic state reconstruction
//! - Checkpoint and restore functionality
//! - State hash verification
//! - Multi-symbol state tracking

use std::collections::{HashMap, BTreeMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Event types in the CQRS log
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    OrderSubmitted,
    OrderFilled,
    OrderCancelled,
    OrderModified,
    TradeExecuted,
    BalanceUpdated,
    PositionChanged,
    SignalGenerated,
    RiskLimitChanged,
    SystemStateChange,
}

impl EventType {
    /// Serialize event type to u8 for compact storage
    pub fn to_u8(&self) -> u8 {
        match self {
            EventType::OrderSubmitted => 0,
            EventType::OrderFilled => 1,
            EventType::OrderCancelled => 2,
            EventType::OrderModified => 3,
            EventType::TradeExecuted => 4,
            EventType::BalanceUpdated => 5,
            EventType::PositionChanged => 6,
            EventType::SignalGenerated => 7,
            EventType::RiskLimitChanged => 8,
            EventType::SystemStateChange => 9,
        }
    }

    /// Deserialize event type from u8
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(EventType::OrderSubmitted),
            1 => Some(EventType::OrderFilled),
            2 => Some(EventType::OrderCancelled),
            3 => Some(EventType::OrderModified),
            4 => Some(EventType::TradeExecuted),
            5 => Some(EventType::BalanceUpdated),
            6 => Some(EventType::PositionChanged),
            7 => Some(EventType::SignalGenerated),
            8 => Some(EventType::RiskLimitChanged),
            9 => Some(EventType::SystemStateChange),
            _ => None,
        }
    }
}

/// CQRS Event record
#[derive(Debug, Clone)]
pub struct CQRS_EVENT {
    pub event_id: u64,
    pub event_type: EventType,
    pub timestamp_ns: u64,
    pub symbol: String,
    pub payload: Vec<u8>,
    pub checksum: u32,
    pub sequence_number: u64,
}

impl CQRS_EVENT {
    /// Create new event with checksum
    pub fn new(
        event_id: u64,
        event_type: EventType,
        symbol: &str,
        payload: Vec<u8>,
        sequence_number: u64,
    ) -> Self {
        let timestamp_ns = get_current_time_ns();
        let checksum = calculate_checksum(&payload);

        Self {
            event_id,
            event_type,
            timestamp_ns,
            symbol: symbol.to_string(),
            payload,
            checksum,
            sequence_number,
        }
    }

    /// Verify event integrity
    pub fn verify_checksum(&self) -> bool {
        calculate_checksum(&self.payload) == self.checksum
    }

    /// Serialize event to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        // Event ID (8 bytes)
        data.extend_from_slice(&self.event_id.to_le_bytes());
        
        // Event type (1 byte)
        data.push(self.event_type.to_u8());
        
        // Timestamp (8 bytes)
        data.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        
        // Symbol length + symbol
        let symbol_bytes = self.symbol.as_bytes();
        data.push(symbol_bytes.len() as u8);
        data.extend_from_slice(symbol_bytes);
        
        // Payload length + payload
        data.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        data.extend_from_slice(&self.payload);
        
        // Checksum (4 bytes)
        data.extend_from_slice(&self.checksum.to_le_bytes());
        
        // Sequence number (8 bytes)
        data.extend_from_slice(&self.sequence_number.to_le_bytes());
        
        data
    }

    /// Deserialize event from bytes
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 34 {
            return None; // Minimum size check
        }

        let mut offset = 0;

        // Event ID
        let event_id = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        offset += 8;

        // Event type
        let event_type = EventType::from_u8(data[offset])?;
        offset += 1;

        // Timestamp
        let timestamp_ns = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
        offset += 8;

        // Symbol
        let symbol_len = data[offset] as usize;
        offset += 1;
        let symbol = String::from_utf8(data[offset..offset + symbol_len].to_vec()).ok()?;
        offset += symbol_len;

        // Payload
        let payload_len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
        offset += 4;
        let payload = data[offset..offset + payload_len].to_vec();
        offset += payload_len;

        // Checksum
        let checksum = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
        offset += 4;

        // Sequence number
        let sequence_number = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);

        Some(Self {
            event_id,
            event_type,
            timestamp_ns,
            symbol,
            payload,
            checksum,
            sequence_number,
        })
    }
}

/// Calculate CRC32 checksum
fn calculate_checksum(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (if crc & 1 == 1 { 0xEDB88320 } else { 0 });
        }
    }
    !crc
}

/// Get current time in nanoseconds
fn get_current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

/// Trading system state snapshot
#[derive(Debug, Clone, Default)]
pub struct SystemState {
    /// Account balances by asset
    pub balances: HashMap<String, i64>,
    /// Open positions by symbol
    pub positions: HashMap<String, Position>,
    /// Pending orders by ID
    pub orders: HashMap<u64, Order>,
    /// Last processed event sequence
    pub last_sequence: u64,
    /// State hash for verification
    pub state_hash: u64,
}

/// Position state
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,
    pub avg_entry_price: u64,
    pub unrealized_pnl: i64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            symbol: String::new(),
            quantity: 0,
            avg_entry_price: 0,
            unrealized_pnl: 0,
        }
    }
}

/// Order state
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    pub symbol: String,
    pub side: u8, // 0=buy, 1=sell
    pub price: u64,
    pub quantity: u64,
    pub filled_quantity: u64,
    pub status: u8, // 0=pending, 1=partial, 2=filled, 3=cancelled
}

/// Event log storage trait
pub trait EventLogStorage: Send + Sync {
    /// Append event to log
    fn append(&self, event: &CQRS_EVENT) -> Result<(), String>;
    
    /// Read events from sequence number
    fn read_from(&self, from_sequence: u64, limit: usize) -> Result<Vec<CQRS_EVENT>, String>;
    
    /// Get latest sequence number
    fn get_latest_sequence(&self) -> u64;
    
    /// Truncate log to sequence number
    fn truncate_to(&self, sequence: u64) -> Result<(), String>;
}

/// In-memory event log storage (for testing)
pub struct InMemoryEventLog {
    events: parking_lot::RwLock<BTreeMap<u64, CQRS_EVENT>>,
    next_id: AtomicU64,
}

impl InMemoryEventLog {
    pub fn new() -> Self {
        Self {
            events: parking_lot::RwLock::new(BTreeMap::new()),
            next_id: AtomicU64::new(1),
        }
    }
}

impl Default for InMemoryEventLog {
    fn default() -> Self {
        Self::new()
    }
}

impl EventLogStorage for InMemoryEventLog {
    fn append(&self, event: &CQRS_EVENT) -> Result<(), String> {
        let mut events = self.events.write();
        events.insert(event.sequence_number, event.clone());
        Ok(())
    }

    fn read_from(&self, from_sequence: u64, limit: usize) -> Result<Vec<CQRS_EVENT>, String> {
        let events = self.events.read();
        Ok(events
            .range(from_sequence..)
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect())
    }

    fn get_latest_sequence(&self) -> u64 {
        let events = self.events.read();
        events.last_key_value().map(|(k, _)| *k).unwrap_or(0)
    }

    fn truncate_to(&self, sequence: u64) -> Result<(), String> {
        let mut events = self.events.write();
        events.retain(|&k, _| k <= sequence);
        Ok(())
    }
}

/// State replay engine for deterministic replay
pub struct StateReplayEngine {
    /// Event log storage
    storage: Arc<dyn EventLogStorage>,
    /// Current system state
    current_state: parking_lot::RwLock<SystemState>,
    /// Event counter
    events_processed: AtomicUsize,
    /// Replay statistics
    replay_stats: parking_lot::RwLock<ReplayStats>,
    /// Checkpoint history
    checkpoints: parking_lot::RwLock<Vec<Checkpoint>>,
}

/// Replay statistics
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    pub total_events_replayed: usize,
    pub successful_replays: usize,
    pub failed_replays: usize,
    pub checksum_failures: usize,
    pub state_mismatches: usize,
    pub replay_duration_ms: u64,
    pub events_per_second: f64,
}

/// State checkpoint for quick restoration
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub sequence_number: u64,
    pub timestamp_ns: u64,
    pub state_hash: u64,
    pub state_snapshot: SystemState,
}

impl StateReplayEngine {
    /// Create new replay engine with storage
    pub fn new(storage: Arc<dyn EventLogStorage>) -> Self {
        Self {
            storage,
            current_state: parking_lot::RwLock::new(SystemState::default()),
            events_processed: AtomicUsize::new(0),
            replay_stats: parking_lot::RwLock::new(ReplayStats::default()),
            checkpoints: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Record an event to the log
    pub fn record_event(
        &self,
        event_type: EventType,
        symbol: &str,
        payload: Vec<u8>,
    ) -> Result<u64, String> {
        let sequence = self.storage.get_latest_sequence() + 1;
        let event_id = self.events_processed.load(AtomicOrdering::Relaxed) as u64 + 1;

        let event = CQRS_EVENT::new(event_id, event_type, symbol, payload, sequence);

        // Verify before storing
        if !event.verify_checksum() {
            return Err("Checksum verification failed".to_string());
        }

        self.storage.append(&event)?;
        self.events_processed.fetch_add(1, AtomicOrdering::Relaxed);

        Ok(sequence)
    }

    /// Replay all events from beginning
    pub fn replay_all(&self) -> Result<ReplayStats, String> {
        self.replay_from(0)
    }

    /// Replay events from specific sequence number
    pub fn replay_from(&self, from_sequence: u64) -> Result<ReplayStats, String> {
        let start_time = std::time::Instant::now();
        let mut stats = ReplayStats::default();

        // Reset state before replay
        *self.current_state.write() = SystemState::default();

        // Read events in batches
        let batch_size = 1000;
        let mut current_seq = from_sequence;
        let mut state = self.current_state.write();

        loop {
            let events = self.storage.read_from(current_seq, batch_size)?;
            
            if events.is_empty() {
                break;
            }

            for event in events {
                stats.total_events_replayed += 1;

                // Verify checksum
                if !event.verify_checksum() {
                    stats.checksum_failures += 1;
                    stats.failed_replays += 1;
                    continue;
                }

                // Apply event to state
                if let Err(e) = self.apply_event(&mut state, &event) {
                    stats.failed_replays += 1;
                    log::error!("Failed to apply event {}: {}", event.event_id, e);
                    continue;
                }

                stats.successful_replays += 1;
                current_seq = event.sequence_number + 1;
            }
        }

        // Update final state
        state.last_sequence = current_seq.saturating_sub(1);
        state.state_hash = self.calculate_state_hash(&state);

        drop(state);

        // Calculate duration
        stats.replay_duration_ms = start_time.elapsed().as_millis() as u64;
        if stats.replay_duration_ms > 0 {
            stats.events_per_second = 
                (stats.total_events_replayed as f64 * 1000.0) / stats.replay_duration_ms as f64;
        }

        *self.replay_stats.write() = stats.clone();
        Ok(stats)
    }

    /// Apply single event to state
    fn apply_event(&self, state: &mut SystemState, event: &CQRS_EVENT) -> Result<(), String> {
        match event.event_type {
            EventType::OrderSubmitted => {
                // Parse order from payload and add to state
                if event.payload.len() >= 24 {
                    let order_id = u64::from_le_bytes(event.payload[0..8].try_into().unwrap());
                    let order = Order {
                        order_id,
                        symbol: event.symbol.clone(),
                        side: event.payload.get(8).copied().unwrap_or(0),
                        price: u64::from_le_bytes(event.payload[9..17].try_into().unwrap_or([0; 8])),
                        quantity: u64::from_le_bytes(event.payload[17..25].try_into().unwrap_or([0; 8])),
                        filled_quantity: 0,
                        status: 0,
                    };
                    state.orders.insert(order_id, order);
                }
            }
            EventType::OrderFilled => {
                // Update order status
                if event.payload.len() >= 16 {
                    let order_id = u64::from_le_bytes(event.payload[0..8].try_into().unwrap());
                    let filled_qty = u64::from_le_bytes(event.payload[8..16].try_into().unwrap_or([0; 8]));
                    
                    if let Some(order) = state.orders.get_mut(&order_id) {
                        order.filled_quantity = filled_qty;
                        order.status = 2; // filled
                    }
                }
            }
            EventType::OrderCancelled => {
                // Remove or mark order as cancelled
                if event.payload.len() >= 8 {
                    let order_id = u64::from_le_bytes(event.payload[0..8].try_into().unwrap());
                    if let Some(order) = state.orders.get_mut(&order_id) {
                        order.status = 3; // cancelled
                    }
                }
            }
            EventType::BalanceUpdated => {
                // Update balance
                if event.payload.len() >= 12 {
                    let asset_len = event.payload[0] as usize;
                    if event.payload.len() > asset_len + 8 {
                        let asset = String::from_utf8_lossy(&event.payload[1..asset_len + 1]).to_string();
                        let balance = i64::from_le_bytes(
                            event.payload[asset_len + 1..asset_len + 9].try_into().unwrap_or([0; 8])
                        );
                        state.balances.insert(asset, balance);
                    }
                }
            }
            _ => {
                // Other event types - just track them
            }
        }

        Ok(())
    }

    /// Calculate state hash for verification
    fn calculate_state_hash(&self, state: &SystemState) -> u64 {
        let mut hash: u64 = 0;
        
        // Hash balances
        for (asset, balance) in &state.balances {
            hash = hash.wrapping_add(asset.bytes().fold(0u64, |h, b| h.wrapping_add(b as u64)));
            hash = hash.wrapping_add(*balance as u64);
        }

        // Hash orders
        for (&order_id, order) in &state.orders {
            hash = hash.wrapping_add(order_id);
            hash = hash.wrapping_add(order.quantity);
        }

        hash
    }

    /// Create checkpoint at current state
    pub fn create_checkpoint(&self) -> Result<Checkpoint, String> {
        let state = self.current_state.read();
        let sequence = state.last_sequence;

        let checkpoint = Checkpoint {
            sequence_number: sequence,
            timestamp_ns: get_current_time_ns(),
            state_hash: state.state_hash,
            state_snapshot: (*state).clone(),
        };

        self.checkpoints.write().push(checkpoint.clone());
        Ok(checkpoint)
    }

    /// Restore from checkpoint
    pub fn restore_from_checkpoint(&self, checkpoint: &Checkpoint) -> Result<(), String> {
        *self.current_state.write() = checkpoint.state_snapshot.clone();
        Ok(())
    }

    /// Verify state consistency
    pub fn verify_state(&self) -> bool {
        let state = self.current_state.read();
        let calculated_hash = self.calculate_state_hash(&state);
        calculated_hash == state.state_hash
    }

    /// Get current state
    pub fn get_state(&self) -> SystemState {
        self.current_state.read().clone()
    }

    /// Get replay statistics
    pub fn get_stats(&self) -> ReplayStats {
        self.replay_stats.read().clone()
    }

    /// Get checkpoint count
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = CQRS_EVENT::new(
            1,
            EventType::OrderSubmitted,
            "BTCUSDT",
            vec![1, 2, 3, 4],
            1,
        );

        let serialized = event.serialize();
        let deserialized = CQRS_EVENT::deserialize(&serialized).unwrap();

        assert_eq!(event.event_id, deserialized.event_id);
        assert_eq!(event.event_type, deserialized.event_type);
        assert_eq!(event.symbol, deserialized.symbol);
        assert!(deserialized.verify_checksum());
    }

    #[test]
    fn test_in_memory_storage() {
        let storage = Arc::new(InMemoryEventLog::new());
        
        let event = CQRS_EVENT::new(
            1,
            EventType::OrderSubmitted,
            "BTCUSDT",
            vec![1, 2, 3],
            1,
        );

        assert!(storage.append(&event).is_ok());
        assert_eq!(storage.get_latest_sequence(), 1);

        let events = storage.read_from(0, 10).unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_replay_engine_basic() {
        let storage = Arc::new(InMemoryEventLog::new());
        let engine = StateReplayEngine::new(storage.clone());

        // Record some events
        engine.record_event(EventType::OrderSubmitted, "BTCUSDT", vec![1, 2, 3]).unwrap();
        engine.record_event(EventType::BalanceUpdated, "USDT", vec![4, 5, 6]).unwrap();

        // Replay
        let stats = engine.replay_all().unwrap();

        assert_eq!(stats.total_events_replayed, 2);
        assert_eq!(stats.successful_replays, 2);
        assert_eq!(stats.checksum_failures, 0);
    }

    #[test]
    fn test_checkpoint_restore() {
        let storage = Arc::new(InMemoryEventLog::new());
        let engine = StateReplayEngine::new(storage);

        // Create checkpoint
        let checkpoint = engine.create_checkpoint().unwrap();

        // Modify state
        engine.record_event(EventType::BalanceUpdated, "USDT", vec![1, 2, 3]).unwrap();
        engine.replay_all().unwrap();

        // Restore
        assert!(engine.restore_from_checkpoint(&checkpoint).is_ok());

        // Verify
        assert!(engine.verify_state());
    }

    #[test]
    fn test_checksum_verification() {
        let mut event = CQRS_EVENT::new(
            1,
            EventType::OrderSubmitted,
            "BTCUSDT",
            vec![1, 2, 3],
            1,
        );

        assert!(event.verify_checksum());

        // Corrupt payload
        event.payload[0] = 255;
        assert!(!event.verify_checksum());
    }
}
