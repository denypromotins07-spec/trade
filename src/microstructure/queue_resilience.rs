//! Queue Resilience Metric for Limit Order Book Analysis
//! 
//! Measures order book resilience using limit order replenishment rates,
//! dynamically adjusting execution aggression based on depth vanishing probability.
//! 
//! Key metrics:
//! - Replenishment Rate: Speed at which cancelled orders are replaced
//! - Depth Vanishing Probability: Likelihood of liquidity disappearing
//! - Resilience Score: Composite metric (0.0 to 1.0) indicating book health
//! 
//! Memory: Bounded circular buffers enforce 8GB RAM limit
//! Latency: Microsecond updates with incremental calculations

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Maximum events to track per side (bounded memory)
const MAX_EVENTS_PER_SIDE: usize = 500_000;

/// Maximum price levels to track
const MAX_PRICE_LEVELS: usize = 100;

/// Time window for resilience calculation in nanoseconds
const RESILIENCE_WINDOW_NS: u64 = 5_000_000_000; // 5 seconds

/// Order book event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderBookEvent {
    /// New limit order added
    Add,
    /// Existing order cancelled
    Cancel,
    /// Order executed (trade)
    Trade,
    /// Order modified (price/volume change)
    Modify,
}

/// Single order book event record
#[derive(Debug, Clone)]
pub struct LOBEvent {
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Event type
    pub event_type: OrderBookEvent,
    /// Price level (normalized)
    pub price_level: u32,
    /// Volume at this level
    pub volume: f64,
    /// Side: true = bid, false = ask
    pub is_bid: bool,
}

/// Price level statistics
#[derive(Debug, Clone, Default)]
pub struct PriceLevelStats {
    /// Total add volume
    pub add_volume: f64,
    /// Total cancel volume
    pub cancel_volume: f64,
    /// Total trade volume
    pub trade_volume: f64,
    /// Number of add events
    pub add_count: u64,
    /// Number of cancel events
    pub cancel_count: u64,
    /// Last update timestamp
    pub last_update_ns: u64,
}

/// Queue resilience calculator
pub struct QueueResilience {
    /// Event buffer for bid side
    bid_events: VecDeque<LOBEvent>,
    /// Event buffer for ask side
    ask_events: VecDeque<LOBEvent>,
    /// Price level statistics for bids
    bid_levels: Vec<PriceLevelStats>,
    /// Price level statistics for asks
    ask_levels: Vec<PriceLevelStats>,
    /// Current resilience score (bid)
    bid_resilience: f64,
    /// Current resilience score (ask)
    ask_resilience: f64,
    /// Depth vanishing probability (bid)
    bid_vanish_prob: f64,
    /// Depth vanishing probability (ask)
    ask_vanish_prob: f64,
    /// Last calculation timestamp
    last_calc_ns: AtomicU64,
    /// Execution aggression multiplier (derived from resilience)
    aggression_multiplier: f64,
    /// Market making flag
    is_market_making: AtomicBool,
}

impl QueueResilience {
    /// Create new queue resilience calculator
    pub fn new(num_price_levels: usize) -> Self {
        let num_levels = num_price_levels.min(MAX_PRICE_LEVELS);
        
        Self {
            bid_events: VecDeque::with_capacity(MAX_EVENTS_PER_SIDE),
            ask_events: VecDeque::with_capacity(MAX_EVENTS_PER_SIDE),
            bid_levels: vec![PriceLevelStats::default(); num_levels],
            ask_levels: vec![PriceLevelStats::default(); num_levels],
            bid_resilience: 0.5,
            ask_resilience: 0.5,
            bid_vanish_prob: 0.5,
            ask_vanish_prob: 0.5,
            last_calc_ns: AtomicU64::new(0),
            aggression_multiplier: 1.0,
            is_market_making: AtomicBool::new(false),
        }
    }
    
    /// Record order book event
    pub fn record_event(&mut self, event: LOBEvent) {
        let buffer = if event.is_bid {
            &mut self.bid_events
        } else {
            &mut self.ask_events
        };
        
        // Enforce bounded memory
        if buffer.len() >= MAX_EVENTS_PER_SIDE {
            buffer.pop_front();
        }
        buffer.push_back(event.clone());
        
        // Update price level statistics
        let level_idx = event.price_level as usize;
        if event.is_bid && level_idx < self.bid_levels.len() {
            self.update_level_stats(&mut self.bid_levels[level_idx], &event);
        } else if !event.is_bid && level_idx < self.ask_levels.len() {
            self.update_level_stats(&mut self.ask_levels[level_idx], &event);
        }
        
        // Incremental resilience update every 100 events
        if buffer.len() % 100 == 0 {
            self.update_resilience(event.timestamp_ns);
        }
    }
    
    /// Update price level statistics
    fn update_level_stats(&self, stats: &mut PriceLevelStats, event: &LOBEvent) {
        match event.event_type {
            OrderBookEvent::Add => {
                stats.add_volume += event.volume;
                stats.add_count += 1;
            }
            OrderBookEvent::Cancel => {
                stats.cancel_volume += event.volume;
                stats.cancel_count += 1;
            }
            OrderBookEvent::Trade => {
                stats.trade_volume += event.volume;
            }
            OrderBookEvent::Modify => {
                // Treat modify as partial cancel + add
                stats.cancel_volume += event.volume * 0.5;
                stats.add_volume += event.volume * 0.5;
            }
        }
        stats.last_update_ns = event.timestamp_ns;
    }
    
    /// Update resilience scores
    pub fn update_resilience(&mut self, current_time_ns: u64) {
        self.bid_resilience = self.calculate_resilience(&self.bid_events, current_time_ns);
        self.ask_resilience = self.calculate_resilience(&self.ask_events, current_time_ns);
        
        self.bid_vanish_prob = self.calculate_vanish_probability(&self.bid_events, current_time_ns);
        self.ask_vanish_prob = self.calculate_vanish_probability(&self.ask_events, current_time_ns);
        
        // Update aggression multiplier based on average resilience
        let avg_resilience = (self.bid_resilience + self.ask_resilience) / 2.0;
        self.aggression_multiplier = self.resilience_to_aggression(avg_resilience);
        
        self.last_calc_ns.store(current_time_ns, Ordering::Relaxed);
    }
    
    /// Calculate resilience score for one side
    fn calculate_resilience(&self, events: &VecDeque<LOBEvent>, current_time_ns: u64) -> f64 {
        if events.is_empty() {
            return 0.5; // Neutral default
        }
        
        let window_start = current_time_ns.saturating_sub(RESILIENCE_WINDOW_NS);
        
        let mut add_volume = 0.0;
        let mut cancel_volume = 0.0;
        let mut trade_volume = 0.0;
        let mut weighted_add_time = 0.0;
        let mut add_weight_sum = 0.0;
        
        for event in events.iter().rev() {
            if event.timestamp_ns < window_start {
                break;
            }
            
            // Time-weighted: recent events matter more
            let time_weight = 1.0 - (current_time_ns - event.timestamp_ns) as f64 / RESILIENCE_WINDOW_NS as f64;
            
            match event.event_type {
                OrderBookEvent::Add => {
                    add_volume += event.volume * time_weight;
                    weighted_add_time += event.timestamp_ns as f64 * time_weight;
                    add_weight_sum += time_weight;
                }
                OrderBookEvent::Cancel => {
                    cancel_volume += event.volume * time_weight;
                }
                OrderBookEvent::Trade => {
                    trade_volume += event.volume * time_weight;
                }
                _ => {}
            }
        }
        
        if add_volume + cancel_volume + trade_volume == 0.0 {
            return 0.5;
        }
        
        // Replenishment ratio: how much adds compensate for cancels + trades
        let consumption = cancel_volume + trade_volume;
        let replenishment_ratio = if consumption > 0.0 {
            (add_volume / consumption).min(3.0) // Cap at 3x
        } else {
            1.0
        };
        
        // Average replenishment time (lower is better)
        let avg_replenish_time = if add_weight_sum > 0.0 {
            let weighted_avg_time = weighted_add_time / add_weight_sum;
            let time_since_avg = (current_time_ns as f64 - weighted_avg_time) / 1_000_000.0; // ms
            time_since_avg.max(0.0).min(1000.0) // Cap at 1 second
        } else {
            500.0 // Default 500ms
        };
        
        // Time factor: faster replenishment = higher resilience
        let time_factor = (1000.0 - avg_replenish_time) / 1000.0;
        
        // Combine factors
        let ratio_component = (replenishment_ratio / 3.0).min(1.0) * 0.6;
        let time_component = time_factor * 0.4;
        
        (ratio_component + time_component).max(0.0).min(1.0)
    }
    
    /// Calculate probability of depth vanishing
    fn calculate_vanish_probability(&self, events: &VecDeque<LOBEvent>, current_time_ns: u64) -> f64 {
        if events.is_empty() {
            return 0.5;
        }
        
        let window_start = current_time_ns.saturating_sub(RESILIENCE_WINDOW_NS);
        
        // Look for sequences of cancels without replacement
        let mut consecutive_cancels = 0;
        let mut max_consecutive_cancels = 0;
        let mut total_cancel_volume = 0.0;
        let mut cancel_bursts = 0;
        
        for event in events.iter().rev() {
            if event.timestamp_ns < window_start {
                break;
            }
            
            match event.event_type {
                OrderBookEvent::Cancel => {
                    consecutive_cancels += 1;
                    total_cancel_volume += event.volume;
                    max_consecutive_cancels = max_consecutive_cancels.max(consecutive_cancels);
                }
                OrderBookEvent::Add => {
                    if consecutive_cancels >= 3 {
                        cancel_bursts += 1;
                    }
                    consecutive_cancels = 0;
                }
                OrderBookEvent::Trade => {
                    consecutive_cancels = consecutive_cancels / 2; // Trades partially reset
                }
                _ => {}
            }
        }
        
        // Base probability from cancel burst frequency
        let burst_prob = (cancel_bursts as f64 / 10.0).min(1.0);
        
        // Volume component
        let volume_prob = (total_cancel_volume / 1000.0).min(1.0); // Normalize by expected volume
        
        // Consecutive cancel component
        let consecutive_prob = (max_consecutive_cancels as f64 / 10.0).min(1.0);
        
        // Weighted combination
        let vanish_prob = burst_prob * 0.4 + volume_prob * 0.3 + consecutive_prob * 0.3;
        
        vanish_prob.max(0.0).min(1.0)
    }
    
    /// Convert resilience score to execution aggression multiplier
    fn resilience_to_aggression(&self, resilience: f64) -> f64 {
        // High resilience -> aggressive execution (can take liquidity, it will replenish)
        // Low resilience -> passive execution (need to provide liquidity)
        
        // Sigmoid-like mapping
        let aggression = 1.0 / (1.0 + (-10.0 * (resilience - 0.5)).exp());
        
        // Map to [0.5, 2.0] range
        0.5 + aggression * 1.5
    }
    
    /// Get recommended execution strategy
    pub fn get_execution_recommendation(&self, is_buy: bool) -> ExecutionRecommendation {
        let (resilience, vanish_prob) = if is_buy {
            (self.ask_resilience, self.ask_vanish_prob) // Buying hits asks
        } else {
            (self.bid_resilience, self.bid_vanish_prob) // Selling hits bids
        };
        
        let aggression = if vanish_prob > 0.7 {
            ExecutionAggression::Passive // High vanish risk, be patient
        } else if resilience > 0.7 {
            ExecutionAggression::Aggressive // High resilience, can be aggressive
        } else if resilience > 0.4 {
            ExecutionAggression::Neutral
        } else {
            ExecutionAggression::VeryPassive // Low resilience, very careful
        };
        
        ExecutionRecommendation {
            aggression,
            max_participation_rate: self.calculate_max_participation(vanish_prob),
            recommended_spread: self.calculate_recommended_spread(resilience, vanish_prob),
            urgency_score: (1.0 - resilience) * vanish_prob,
        }
    }
    
    /// Calculate maximum participation rate based on vanish probability
    fn calculate_max_participation(&self, vanish_prob: f64) -> f64 {
        // Higher vanish probability -> lower participation to avoid moving market
        (1.0 - vanish_prob * 0.8).max(0.05).min(0.5) // 5% to 50%
    }
    
    /// Calculate recommended spread adjustment
    fn calculate_recommended_spread(&self, resilience: f64, vanish_prob: f64) -> f64 {
        // Base spread in basis points
        let base_spread = 5.0;
        
        // Adjust based on resilience and vanish probability
        let resilience_adj = (1.0 - resilience) * 10.0; // Low resilience -> wider spread
        let vanish_adj = vanish_prob * 15.0; // High vanish prob -> wider spread
        
        (base_spread + resilience_adj + vanish_adj).min(50.0) // Cap at 50 bps
    }
    
    /// Get current resilience statistics
    pub fn get_statistics(&self) -> ResilienceStatistics {
        ResilienceStatistics {
            bid_resilience: self.bid_resilience,
            ask_resilience: self.ask_resilience,
            bid_vanish_prob: self.bid_vanish_prob,
            ask_vanish_prob: self.ask_vanish_prob,
            aggression_multiplier: self.aggression_multiplier,
            avg_resilience: (self.bid_resilience + self.ask_resilience) / 2.0,
            min_resilience: self.bid_resilience.min(self.ask_resilience),
            resilience_imbalance: (self.bid_resilience - self.ask_resilience).abs(),
        }
    }
    
    /// Check if market is in fragile state
    pub fn is_fragile(&self) -> bool {
        let stats = self.get_statistics();
        stats.avg_resilience < 0.3 || stats.bid_vanish_prob > 0.8 || stats.ask_vanish_prob > 0.8
    }
    
    /// Reset all statistics
    pub fn reset(&mut self) {
        self.bid_events.clear();
        self.ask_events.clear();
        for level in &mut self.bid_levels {
            *level = PriceLevelStats::default();
        }
        for level in &mut self.ask_levels {
            *level = PriceLevelStats::default();
        }
        self.bid_resilience = 0.5;
        self.ask_resilience = 0.5;
        self.bid_vanish_prob = 0.5;
        self.ask_vanish_prob = 0.5;
        self.aggression_multiplier = 1.0;
    }
}

/// Resilience statistics snapshot
#[derive(Debug, Clone)]
pub struct ResilienceStatistics {
    pub bid_resilience: f64,
    pub ask_resilience: f64,
    pub bid_vanish_prob: f64,
    pub ask_vanish_prob: f64,
    pub aggression_multiplier: f64,
    pub avg_resilience: f64,
    pub min_resilience: f64,
    pub resilience_imbalance: f64,
}

/// Execution aggression level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAggression {
    VeryPassive,
    Passive,
    Neutral,
    Aggressive,
    VeryAggressive,
}

/// Execution recommendation
#[derive(Debug, Clone)]
pub struct ExecutionRecommendation {
    pub aggression: ExecutionAggression,
    pub max_participation_rate: f64,
    pub recommended_spread: f64, // in basis points
    pub urgency_score: f64,
}

impl ExecutionRecommendation {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self.aggression {
            ExecutionAggression::VeryPassive => "Extremely cautious - liquidity very fragile",
            ExecutionAggression::Passive => "Cautious - elevated vanish risk",
            ExecutionAggression::Neutral => "Normal execution conditions",
            ExecutionAggression::Aggressive => "Can execute aggressively - resilient book",
            ExecutionAggression::VeryAggressive => "Maximum aggression - highly resilient",
        }
    }
}

/// Multi-symbol resilience tracker
pub struct MultiSymbolResilience {
    /// Per-symbol resilience calculators
    symbols: std::collections::HashMap<String, QueueResilience>,
    /// Global fragility flag
    global_fragile: AtomicBool,
}

impl MultiSymbolResilience {
    pub fn new() -> Self {
        Self {
            symbols: std::collections::HashMap::new(),
            global_fragile: AtomicBool::new(false),
        }
    }
    
    /// Get or create resilience calculator for symbol
    pub fn get_or_create(&mut self, symbol: &str, num_levels: usize) -> &mut QueueResilience {
        self.symbols.entry(symbol.to_string())
            .or_insert_with(|| QueueResilience::new(num_levels))
    }
    
    /// Check if any symbol is fragile
    pub fn update_global_fragility(&self) -> bool {
        let any_fragile = self.symbols.values().any(|qr| qr.is_fragile());
        self.global_fragile.store(any_fragile, Ordering::Relaxed);
        any_fragile
    }
    
    /// Get aggregate resilience across all symbols
    pub fn get_aggregate_stats(&self) -> AggregateResilienceStats {
        if self.symbols.is_empty() {
            return AggregateResilienceStats::default();
        }
        
        let mut total_bid_res = 0.0;
        let mut total_ask_res = 0.0;
        let mut min_resilience = 1.0;
        let mut fragile_count = 0;
        
        for qr in self.symbols.values() {
            let stats = qr.get_statistics();
            total_bid_res += stats.bid_resilience;
            total_ask_res += stats.ask_resilience;
            min_resilience = min_resilience.min(stats.min_resilience);
            if qr.is_fragile() {
                fragile_count += 1;
            }
        }
        
        let count = self.symbols.len() as f64;
        AggregateResilienceStats {
            avg_bid_resilience: total_bid_res / count,
            avg_ask_resilience: total_ask_res / count,
            min_resilience,
            fragile_symbols: fragile_count,
            total_symbols: self.symbols.len(),
            global_fragile: self.global_fragile.load(Ordering::Relaxed),
        }
    }
}

/// Aggregate resilience statistics
#[derive(Debug, Clone, Default)]
pub struct AggregateResilienceStats {
    pub avg_bid_resilience: f64,
    pub avg_ask_resilience: f64,
    pub min_resilience: f64,
    pub fragile_symbols: usize,
    pub total_symbols: usize,
    pub global_fragile: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resilience_calculation() {
        let mut qr = QueueResilience::new(10);
        let base_time = 1_000_000_000_000u64;
        
        // Add healthy order flow
        for i in 0..100 {
            qr.record_event(LOBEvent {
                timestamp_ns: base_time + i as u64 * 10_000_000,
                event_type: OrderBookEvent::Add,
                price_level: (i % 10) as u32,
                volume: 1.0,
                is_bid: true,
            });
            
            qr.record_event(LOBEvent {
                timestamp_ns: base_time + i as u64 * 10_000_000 + 5_000_000,
                event_type: OrderBookEvent::Trade,
                price_level: (i % 10) as u32,
                volume: 0.5,
                is_bid: true,
            });
        }
        
        qr.update_resilience(base_time + 1_000_000_000);
        let stats = qr.get_statistics();
        
        // Should have decent resilience from balanced add/trade flow
        assert!(stats.bid_resilience > 0.3);
    }
    
    #[test]
    fn test_fragile_market_detection() {
        let mut qr = QueueResilience::new(10);
        let base_time = 1_000_000_000_000u64;
        
        // Simulate flash crash scenario - many cancels, no adds
        for i in 0..50 {
            qr.record_event(LOBEvent {
                timestamp_ns: base_time + i as u64 * 1_000_000,
                event_type: OrderBookEvent::Cancel,
                price_level: (i % 10) as u32,
                volume: 10.0,
                is_bid: true,
            });
        }
        
        qr.update_resilience(base_time + 100_000_000);
        
        // Should detect fragile state
        assert!(qr.is_fragile() || qr.get_statistics().bid_vanish_prob > 0.5);
    }
    
    #[test]
    fn test_memory_bounded_buffers() {
        let mut qr = QueueResilience::new(10);
        let base_time = 1_000_000_000_000u64;
        
        // Add more events than buffer capacity
        for i in 0..MAX_EVENTS_PER_SIDE + 1000 {
            qr.record_event(LOBEvent {
                timestamp_ns: base_time + i as u64 * 1_000_000,
                event_type: OrderBookEvent::Add,
                price_level: 0,
                volume: 1.0,
                is_bid: true,
            });
        }
        
        // Buffers should not exceed capacity
        assert!(qr.bid_events.len() <= MAX_EVENTS_PER_SIDE);
        assert!(qr.ask_events.len() <= MAX_EVENTS_PER_SIDE);
    }
    
    #[test]
    fn test_execution_recommendation() {
        let mut qr = QueueResilience::new(10);
        let base_time = 1_000_000_000_000u64;
        
        // Create resilient market
        for i in 0..100 {
            qr.record_event(LOBEvent {
                timestamp_ns: base_time + i as u64 * 10_000_000,
                event_type: OrderBookEvent::Add,
                price_level: (i % 10) as u32,
                volume: 5.0,
                is_bid: true,
            });
        }
        
        qr.update_resilience(base_time + 1_000_000_000);
        
        let rec = qr.get_execution_recommendation(false); // Selling
        
        // Should recommend more aggressive execution in resilient market
        assert!(rec.max_participation_rate > 0.1);
        assert!(rec.recommended_spread < 20.0);
    }
}
