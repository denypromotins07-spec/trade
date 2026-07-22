//! Tick Simulator - Synthesizes trade ticks from orderbook crosses
//!
//! This module generates high-fidelity execution data by simulating trade ticks
//! directly from orderbook state changes. Essential for backtesting when exchange
//! REST APIs are rate-limited or when historical tick data is incomplete.
//!
//! ## Features
//! - Realistic trade generation from bid-ask crosses
//! - Volume-weighted price simulation
//! - Microstructure noise modeling
//! - Latency injection for realistic backtesting

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::data::book_builder::{OrderbookBuilder, Side, OrderEntry};

/// Represents a synthesized trade tick
#[derive(Debug, Clone)]
pub struct TradeTick {
    pub trade_id: u64,
    pub symbol: String,
    pub price: u64,
    pub quantity: u64,
    pub timestamp_ns: u64,
    pub side: TradeSide, // Aggressor side
    pub is_buyer_maker: bool,
    pub simulated: bool,
}

/// Trade side (aggressor)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,  // Market buy hitting the ask
    Sell, // Market sell hitting the bid
}

/// Configuration for tick simulation
#[derive(Debug, Clone)]
pub struct TickSimulatorConfig {
    /// Minimum trade size in base units
    pub min_trade_size: u64,
    /// Maximum trade size in base units
    pub max_trade_size: u64,
    /// Probability of trade occurring on orderbook update (0.0-1.0)
    pub trade_probability: f64,
    /// Add microstructure noise to prices
    pub add_noise: bool,
    /// Noise magnitude in ticks
    pub noise_ticks: u64,
    /// Simulate network latency
    pub simulate_latency: bool,
    /// Base latency in microseconds
    pub base_latency_us: u64,
    /// Latency variance in microseconds
    pub latency_variance_us: u64,
}

impl Default for TickSimulatorConfig {
    fn default() -> Self {
        Self {
            min_trade_size: 1,
            max_trade_size: 10000,
            trade_probability: 0.3,
            add_noise: true,
            noise_ticks: 1,
            simulate_latency: true,
            base_latency_us: 100,
            latency_variance_us: 50,
        }
    }
}

/// High-performance tick simulator using ring buffer for zero-allocation streaming
pub struct TickSimulator {
    config: TickSimulatorConfig,
    /// Ring buffer for generated ticks (pre-allocated)
    tick_buffer: VecDeque<TradeTick>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Trade ID counter
    trade_id_counter: AtomicU64,
    /// Total ticks generated
    total_ticks_generated: AtomicUsize,
    /// Last trade timestamp per symbol
    last_trade_ts: Arc<parking_lot::RwLock<std::collections::HashMap<String, u64>>>,
    /// Random seed for deterministic replay
    random_seed: u64,
    /// Current RNG state (simple LCG for speed)
    rng_state: AtomicU64,
}

impl TickSimulator {
    /// Create new tick simulator with default config
    pub fn new() -> Self {
        Self::with_config(TickSimulatorConfig::default())
    }

    /// Create new tick simulator with custom config
    pub fn with_config(config: TickSimulatorConfig) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_nanos() as u64;
        
        Self {
            config,
            tick_buffer: VecDeque::with_capacity(10000),
            max_buffer_size: 10000,
            trade_id_counter: AtomicU64::new(0),
            total_ticks_generated: AtomicUsize::new(0),
            last_trade_ts: Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            random_seed: seed,
            rng_state: AtomicU64::new(seed),
        }
    }

    /// Set random seed for deterministic replay
    pub fn set_seed(&mut self, seed: u64) {
        self.random_seed = seed;
        self.rng_state.store(seed, AtomicOrdering::Relaxed);
    }

    /// Fast linear congruential generator for random numbers
    #[inline]
    fn next_random(&self) -> u64 {
        let state = self.rng_state.load(AtomicOrdering::Relaxed);
        // LCG parameters (Numerical Recipes)
        let new_state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.rng_state.store(new_state, AtomicOrdering::Relaxed);
        new_state
    }

    /// Generate random number in range [0, max)
    #[inline]
    fn random_range(&self, max: u64) -> u64 {
        if max == 0 { return 0; }
        self.next_random() % max
    }

    /// Generate random float in [0.0, 1.0)
    #[inline]
    fn random_float(&self) -> f64 {
        (self.next_random() & 0x000FFFFFFFFFFFFF) as f64 / 0x0010000000000000u64 as f64
    }

    /// Synthesize trade ticks from an orderbook cross event
    /// Returns vector of generated trade ticks
    #[inline]
    pub fn simulate_cross(&mut self, book: &OrderbookBuilder, aggressor_side: TradeSide, 
                          cross_price: u64, available_quantity: u64, timestamp_ns: u64) 
                          -> Vec<TradeTick> {
        let mut ticks = Vec::new();
        
        // Check if we should generate a trade based on probability
        if self.random_float() > self.config.trade_probability {
            return ticks;
        }

        // Determine trade quantity
        let trade_qty = self.determine_trade_quantity(available_quantity);
        if trade_qty < self.config.min_trade_size {
            return ticks;
        }

        // Apply microstructure noise if configured
        let final_price = if self.config.add_noise {
            let noise_offset = (self.random_range(self.config.noise_ticks * 2 + 1) as i64) 
                - (self.config.noise_ticks as i64);
            (cross_price as i64 + noise_offset).max(1) as u64
        } else {
            cross_price
        };

        // Apply latency simulation
        let final_timestamp = if self.config.simulate_latency {
            let latency_ns = self.simulate_latency_ns();
            timestamp_ns.saturating_add(latency_ns)
        } else {
            timestamp_ns
        };

        // Generate trade tick
        let tick = TradeTick {
            trade_id: self.trade_id_counter.fetch_add(1, AtomicOrdering::Relaxed),
            symbol: book.symbol().to_string(),
            price: final_price,
            quantity: trade_qty.min(available_quantity),
            timestamp_ns: final_timestamp,
            side: aggressor_side,
            is_buyer_maker: aggressor_side == TradeSide::Sell,
            simulated: true,
        };

        // Update last trade timestamp
        {
            let mut last_ts = self.last_trade_ts.write();
            last_ts.insert(book.symbol().to_string(), final_timestamp);
        }

        self.total_ticks_generated.fetch_add(1, AtomicOrdering::Relaxed);
        
        // Add to buffer
        if self.tick_buffer.len() >= self.max_buffer_size {
            self.tick_buffer.pop_front();
        }
        self.tick_buffer.push_back(tick.clone());
        
        ticks.push(tick);
        ticks
    }

    /// Determine trade quantity based on distribution
    #[inline]
    fn determine_trade_quantity(&self, available: u64) -> u64 {
        // Use a mixture distribution: small trades more common
        let r = self.random_float();
        
        let qty = if r < 0.7 {
            // 70% small trades (1-10% of available)
            let pct = 0.01 + self.random_float() * 0.09;
            (available as f64 * pct) as u64
        } else if r < 0.9 {
            // 20% medium trades (10-50% of available)
            let pct = 0.1 + self.random_float() * 0.4;
            (available as f64 * pct) as u64
        } else {
            // 10% large trades (50-100% of available)
            let pct = 0.5 + self.random_float() * 0.5;
            (available as f64 * pct) as u64
        };

        qty.max(self.config.min_trade_size).min(self.config.max_trade_size).min(available)
    }

    /// Simulate network latency in nanoseconds
    #[inline]
    fn simulate_latency_ns(&self) -> u64 {
        let base_ns = self.config.base_latency_us * 1000;
        let variance_ns = self.config.latency_variance_us * 1000;
        let random_offset = (self.random_range(variance_ns * 2) as i64) - (variance_ns as i64);
        (base_ns as i64 + random_offset).max(0) as u64
    }

    /// Generate ticks from orderbook snapshot (for initial state)
    pub fn generate_initial_ticks(&mut self, book: &OrderbookBuilder, 
                                   count: usize) -> Vec<TradeTick> {
        let mut ticks = Vec::with_capacity(count);
        let current_ns = get_current_time_ns();
        
        for _ in 0..count {
            let (price, qty, side) = match self.random_float() {
                r if r < 0.5 => {
                    // Simulate buyer-initiated trade at ask
                    if let Some((ask_price, ask_qty)) = book.best_ask() {
                        (ask_price, ask_qty, TradeSide::Buy)
                    } else {
                        continue;
                    }
                }
                _ => {
                    // Simulate seller-initiated trade at bid
                    if let Some((bid_price, bid_qty)) = book.best_bid() {
                        (bid_price, bid_qty, TradeSide::Sell)
                    } else {
                        continue;
                    }
                }
            };

            let trade_qty = self.determine_trade_quantity(qty);
            if trade_qty < self.config.min_trade_size {
                continue;
            }

            let tick = TradeTick {
                trade_id: self.trade_id_counter.fetch_add(1, AtomicOrdering::Relaxed),
                symbol: book.symbol().to_string(),
                price,
                quantity: trade_qty,
                timestamp_ns: current_ns.saturating_sub(self.random_range(1_000_000_000)), // Within last second
                side,
                is_buyer_maker: side == TradeSide::Sell,
                simulated: true,
            };

            self.tick_buffer.push_back(tick.clone());
            ticks.push(tick);
        }

        self.total_ticks_generated.fetch_add(ticks.len(), AtomicOrdering::Relaxed);
        ticks
    }

    /// Get recent ticks from buffer
    pub fn get_recent_ticks(&self, count: usize) -> Vec<TradeTick> {
        self.tick_buffer.iter().rev().take(count).cloned().collect()
    }

    /// Clear tick buffer
    pub fn clear_buffer(&mut self) {
        self.tick_buffer.clear();
    }

    /// Get statistics
    pub fn get_stats(&self) -> TickSimulatorStats {
        TickSimulatorStats {
            total_ticks: self.total_ticks_generated.load(AtomicOrdering::Relaxed),
            buffer_size: self.tick_buffer.len(),
            max_buffer_size: self.max_buffer_size,
        }
    }

    /// Process orderbook update and potentially generate ticks
    pub fn process_orderbook_update(&mut self, book: &OrderbookBuilder,
                                     prev_best_bid: Option<u64>,
                                     prev_best_ask: Option<u64>,
                                     timestamp_ns: u64) -> Vec<TradeTick> {
        let mut ticks = Vec::new();
        
        let current_bid = book.best_bid().map(|(p, _)| p);
        let current_ask = book.best_ask().map(|(p, _)| p);

        // Detect cross (bid >= ask indicates potential trade)
        if let (Some(bid), Some(ask)) = (current_bid, current_ask) {
            if bid >= ask {
                // Orderbook is crossed - generate trade
                let cross_price = bid.wrapping_add(ask) / 2;
                let available_qty = book.best_bid().map(|(_, q)| q).unwrap_or(0)
                    .min(book.best_ask().map(|(_, q)| q).unwrap_or(0));
                
                let cross_ticks = self.simulate_cross(
                    book,
                    TradeSide::Buy,
                    cross_price,
                    available_qty,
                    timestamp_ns,
                );
                ticks.extend(cross_ticks);
            }
        }

        // Detect price improvement trades
        if let (Some(prev_bid), Some(curr_bid)) = (prev_best_bid, current_bid) {
            if curr_bid > prev_bid {
                // Bid moved up - potential buyer aggression
                if let Some((ask_price, ask_qty)) = book.best_ask() {
                    if curr_bid >= ask_price {
                        let cross_ticks = self.simulate_cross(
                            book,
                            TradeSide::Buy,
                            ask_price,
                            ask_qty,
                            timestamp_ns,
                        );
                        ticks.extend(cross_ticks);
                    }
                }
            }
        }

        if let (Some(prev_ask), Some(curr_ask)) = (prev_best_ask, current_ask) {
            if curr_ask < prev_ask {
                // Ask moved down - potential seller aggression
                if let Some((bid_price, bid_qty)) = book.best_bid() {
                    if curr_ask <= bid_price {
                        let cross_ticks = self.simulate_cross(
                            book,
                            TradeSide::Sell,
                            bid_price,
                            bid_qty,
                            timestamp_ns,
                        );
                        ticks.extend(cross_ticks);
                    }
                }
            }
        }

        ticks
    }
}

impl Default for TickSimulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for tick simulator
#[derive(Debug, Clone)]
pub struct TickSimulatorStats {
    pub total_ticks: usize,
    pub buffer_size: usize,
    pub max_buffer_size: usize,
}

/// Get current time in nanoseconds since epoch
#[inline]
fn get_current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos() as u64
}

/// Batch tick processor for high-throughput scenarios
pub struct BatchTickProcessor {
    simulator: TickSimulator,
    batch_size: usize,
    pending_ticks: Vec<TradeTick>,
}

impl BatchTickProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            simulator: TickSimulator::new(),
            batch_size,
            pending_ticks: Vec::with_capacity(batch_size),
        }
    }

    /// Process batch of orderbook updates
    pub fn process_batch(&mut self, updates: &[(u64, u64, Side, u64)]) -> Vec<TradeTick> {
        // updates: (price, quantity, side, timestamp_ns)
        let mut all_ticks = Vec::new();
        
        for (price, qty, side, ts) in updates {
            // Simulate effect on orderbook and generate ticks
            // This is simplified - real implementation would maintain full book state
            let random_qty = self.simulator.determine_trade_quantity(*qty);
            
            if random_qty > 0 && self.simulator.random_float() < self.simulator.config.trade_probability {
                let tick = TradeTick {
                    trade_id: self.simulator.trade_id_counter.fetch_add(1, AtomicOrdering::Relaxed),
                    symbol: "BATCH".to_string(),
                    price: *price,
                    quantity: random_qty,
                    timestamp_ns: *ts,
                    side: match side {
                        Side::Bid => TradeSide::Buy,
                        Side::Ask => TradeSide::Sell,
                    },
                    is_buyer_maker: *side == Side::Ask,
                    simulated: true,
                };
                
                self.pending_ticks.push(tick.clone());
                
                if self.pending_ticks.len() >= self.batch_size {
                    all_ticks.extend(self.pending_ticks.drain(..));
                }
            }
        }
        
        all_ticks
    }

    /// Flush remaining ticks
    pub fn flush(&mut self) -> Vec<TradeTick> {
        self.pending_ticks.drain(..).collect()
    }

    /// Get reference to simulator
    pub fn simulator(&self) -> &TickSimulator {
        &self.simulator
    }

    /// Get mutable reference to simulator
    pub fn simulator_mut(&mut self) -> &mut TickSimulator {
        &mut self.simulator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_simulator_basic() {
        let mut simulator = TickSimulator::with_config(TickSimulatorConfig {
            trade_probability: 1.0, // Always generate trades for testing
            add_noise: false,
            simulate_latency: false,
            ..Default::default()
        });
        
        simulator.set_seed(42); // Deterministic
        
        let mut book = OrderbookBuilder::new("BTCUSDT");
        book.apply_delta(50000, 100, Side::Bid, 1000);
        book.apply_delta(50100, 50, Side::Ask, 1001);
        
        let ticks = simulator.simulate_cross(&book, TradeSide::Buy, 50100, 50, 1002);
        
        assert!(!ticks.is_empty());
        assert!(ticks[0].simulated);
        assert_eq!(ticks[0].symbol, "BTCUSDT");
    }

    #[test]
    fn test_deterministic_replay() {
        let mut sim1 = TickSimulator::new();
        let mut sim2 = TickSimulator::new();
        
        sim1.set_seed(12345);
        sim2.set_seed(12345);
        
        // Should produce same sequence
        let rand1a = sim1.random_float();
        let rand2a = sim2.random_float();
        
        assert_eq!(rand1a, rand2a);
    }

    #[test]
    fn test_batch_processor() {
        let mut processor = BatchTickProcessor::new(5);
        
        let updates = vec![
            (50000u64, 100u64, Side::Bid, 1000u64),
            (50100u64, 50u64, Side::Ask, 1001u64),
            (50000u64, 100u64, Side::Bid, 1002u64),
            (50100u64, 50u64, Side::Ask, 1003u64),
            (50000u64, 100u64, Side::Bid, 1004u64),
            (50100u64, 50u64, Side::Ask, 1005u64),
        ];
        
        let ticks = processor.process_batch(&updates);
        // May have fewer ticks due to probability filtering
        assert!(ticks.len() <= updates.len());
    }
}
