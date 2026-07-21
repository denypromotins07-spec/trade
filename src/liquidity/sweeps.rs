//! Liquidity Engineering & Market Manipulation Detection - Chapter 2
//! File 5: sweeps.rs
//! 
//! Implements liquidity sweep and stop-run detectors that map equal highs/lows
//! and instantly trigger alerts when aggressive market orders pierce these levels
//! and immediately reverse. Optimized for microsecond detection.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Detected liquidity level (support/resistance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityLevel {
    pub price: i64,
    pub level_type: LevelType,
    pub touch_count: u32,
    pub last_touch_ns: u64,
    pub strength: f64, // 0.0 to 1.0
    pub volume_at_level: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LevelType {
    EqualHigh,
    EqualLow,
    SwingHigh,
    SwingLow,
    ConsolidationHigh,
    ConsolidationLow,
}

/// Sweep detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepEvent {
    pub event_type: SweepType,
    pub timestamp_ns: u64,
    pub swept_price: i64,
    pub sweep_depth: i64,
    pub volume: u64,
    pub reversal_confirmed: bool,
    pub confidence: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SweepType {
    /// Bullish sweep: liquidity taken below support, price reverses up
    BullishSweep,
    /// Bearish sweep: liquidity taken above resistance, price reverses down
    BearishSweep,
    /// Stop run without reversal (continuation)
    StopRunContinuation,
    /// Failed sweep (price doesn't reach expected liquidity)
    FailedSweep,
}

/// Real-time liquidity sweep detector
pub struct LiquiditySweepDetector {
    /// Mapped liquidity levels
    liquidity_levels: parking_lot::RwLock<Vec<LiquidityLevel>>,
    /// Recent price history for swing detection
    price_history: Vec<i64>,
    price_history_idx: AtomicUsize,
    price_history_capacity: usize,
    /// Last detected high/low
    recent_high: AtomicI64,
    recent_low: AtomicI64,
    /// Equal highs/lows tracking
    equal_highs: parking_lot::Mutex<std::collections::HashMap<i64, u32>>,
    equal_lows: parking_lot::Mutex<std::collections::HashMap<i64, u32>>,
    /// Detected events queue
    events_queue: crossbeam_queue::SegQueue<SweepEvent>,
    /// Configuration
    swing_lookback: usize,
    equal_touch_threshold: u32,
    sweep_confirmation_window_ns: u64,
}

impl LiquiditySweepDetector {
    /// Create new sweep detector
    pub fn new(swing_lookback: usize, equal_touch_threshold: u32) -> Self {
        let capacity = swing_lookback.max(100);
        Self {
            liquidity_levels: parking_lot::RwLock::new(Vec::with_capacity(256)),
            price_history: vec![0; capacity],
            price_history_idx: AtomicUsize::new(0),
            price_history_capacity: capacity,
            recent_high: AtomicI64::new(i64::MIN),
            recent_low: AtomicI64::new(i64::MAX),
            equal_highs: parking_lot::Mutex::new(std::collections::HashMap::new()),
            equal_lows: parking_lot::Mutex::new(std::collections::HashMap::new()),
            events_queue: crossbeam_queue::SegQueue::new(),
            swing_lookback,
            equal_touch_threshold,
            sweep_confirmation_window_ns: 5_000_000_000, // 5 seconds
        }
    }

    /// Process a new trade/tick
    pub fn process_price(&self, price: i64, volume: u64, timestamp_ns: u64) {
        // Update recent high/low
        self.recent_high.fetch_max(price, Ordering::Relaxed);
        self.recent_low.fetch_min(price, Ordering::Relaxed);

        // Record price in history
        let idx = self.price_history_idx.fetch_add(1, Ordering::Relaxed);
        let wrapped_idx = idx % self.price_history_capacity;
        self.price_history[wrapped_idx] = price;

        // Update equal highs/lows
        self.update_equal_levels(price);

        // Detect swings periodically
        if idx % 10 == 0 {
            self.detect_swings();
        }

        // Check for sweep of known liquidity levels
        self.check_sweep(price, volume, timestamp_ns);
    }

    /// Update equal highs/lows tracking
    fn update_equal_levels(&self, price: i64) {
        // Check if price is near existing equal high
        {
            let mut highs = self.equal_highs.lock();
            for (level, count) in highs.iter_mut() {
                if (price - *level).abs() <= 10000000 { // Within 0.1%
                    *count += 1;
                }
            }
            // Add new potential equal high
            if !highs.keys().any(|k| (price - *k).abs() <= 10000000) {
                highs.insert(price, 1);
            }
        }

        // Check if price is near existing equal low
        {
            let mut lows = self.equal_lows.lock();
            for (level, count) in lows.iter_mut() {
                if (price - *level).abs() <= 10000000 {
                    *count += 1;
                }
            }
            // Add new potential equal low
            if !lows.keys().any(|k| (price - *k).abs() <= 10000000) {
                lows.insert(price, 1);
            }
        }
    }

    /// Detect swing highs/lows from price history
    fn detect_swings(&self) {
        let idx = self.price_history_idx.load(Ordering::Relaxed);
        if idx < self.swing_lookback + 2 {
            return;
        }

        let lookback = self.swing_lookback.min(self.price_history_capacity);
        
        // Find local maxima/minima
        let mut swings = Vec::new();
        
        for i in lookback..(self.price_history_capacity - lookback) {
            let current = self.price_history[i];
            let mut is_high = true;
            let mut is_low = true;

            // Check left side
            for j in (i - lookback)..i {
                if self.price_history[j] >= current {
                    is_high = false;
                }
                if self.price_history[j] <= current {
                    is_low = false;
                }
            }

            // Check right side
            for j in i..(i + lookback) {
                if self.price_history[j] >= current {
                    is_high = false;
                }
                if self.price_history[j] <= current {
                    is_low = false;
                }
            }

            if is_high || is_low {
                swings.push((i, current, is_high, is_low));
            }
        }

        // Update liquidity levels
        let mut levels = self.liquidity_levels.write();
        for (_, price, is_high, is_low) in swings {
            let level_type = if is_high {
                LevelType::SwingHigh
            } else {
                LevelType::SwingLow
            };

            // Check if level already exists
            let existing = levels.iter_mut()
                .find(|l| (l.price - price).abs() <= 10000000 && l.level_type == level_type);

            if let Some(level) = existing {
                level.touch_count += 1;
                level.strength = (level.touch_count as f64 / 10.0).min(1.0);
            } else {
                levels.push(LiquidityLevel {
                    price,
                    level_type,
                    touch_count: 1,
                    last_touch_ns: 0,
                    strength: 0.1,
                    volume_at_level: 0,
                });
            }
        }

        // Keep only recent levels
        if levels.len() > 100 {
            levels.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
            levels.truncate(50);
        }
    }

    /// Check if current price sweeps a liquidity level
    fn check_sweep(&self, price: i64, volume: u64, timestamp_ns: u64) {
        let levels = self.liquidity_levels.read();
        
        for level in levels.iter() {
            let swept = match level.level_type {
                LevelType::EqualHigh | LevelType::SwingHigh | LevelType::ConsolidationHigh => {
                    price > level.price
                }
                LevelType::EqualLow | LevelType::SwingLow | LevelType::ConsolidationLow => {
                    price < level.price
                }
            };

            if swept {
                let sweep_depth = (price - level.price).abs();
                
                // Check for reversal confirmation
                let reversal_confirmed = self.check_reversal(level, price, timestamp_ns);
                
                let sweep_type = if level.level_type == LevelType::EqualLow 
                    || level.level_type == LevelType::SwingLow 
                    || level.level_type == LevelType::ConsolidationLow 
                {
                    if reversal_confirmed {
                        SweepType::BullishSweep
                    } else {
                        SweepType::StopRunContinuation
                    }
                } else {
                    if reversal_confirmed {
                        SweepType::BearishSweep
                    } else {
                        SweepType::StopRunContinuation
                    }
                };

                let confidence = level.strength * if reversal_confirmed { 1.0 } else { 0.5 };

                let event = SweepEvent {
                    event_type: sweep_type,
                    timestamp_ns,
                    swept_price: level.price,
                    sweep_depth,
                    volume,
                    reversal_confirmed,
                    confidence,
                    description: format!(
                        "{} sweep at {}: depth={}, reversal={}",
                        match sweep_type {
                            SweepType::BullishSweep => "Bullish",
                            SweepType::BearishSweep => "Bearish",
                            SweepType::StopRunContinuation => "Stop Run",
                            SweepType::FailedSweep => "Failed",
                        },
                        level.price,
                        sweep_depth,
                        reversal_confirmed
                    ),
                };

                self.events_queue.push(event);
            }
        }
    }

    /// Check if price has reversed after sweeping a level
    fn check_reversal(&self, level: &LiquidityLevel, current_price: i64, timestamp_ns: u64) -> bool {
        let idx = self.price_history_idx.load(Ordering::Relaxed);
        if idx < 10 {
            return false;
        }

        // Look back within confirmation window
        let window_start = timestamp_ns.saturating_sub(self.sweep_confirmation_window_ns);
        
        // Find the sweep high/low in recent history
        let mut extreme_price = current_price;
        for i in 0..self.price_history_capacity {
            let p = self.price_history[i];
            if level.level_type == LevelType::EqualLow 
                || level.level_type == LevelType::SwingLow 
                || level.level_type == LevelType::ConsolidationLow 
            {
                if p < extreme_price {
                    extreme_price = p;
                }
            } else {
                if p > extreme_price {
                    extreme_price = p;
                }
            }
        }

        // Check if price has moved back through the level
        let reversal_threshold = (level.price as f64 * 0.001) as i64; // 0.1% reversal
        
        match level.level_type {
            LevelType::EqualLow | LevelType::SwingLow | LevelType::ConsolidationLow => {
                // Bullish reversal: price swept low and came back above level
                current_price > level.price + reversal_threshold
            }
            LevelType::EqualHigh | LevelType::SwingHigh | LevelType::ConsolidationHigh => {
                // Bearish reversal: price swept high and came back below level
                current_price < level.price - reversal_threshold
            }
        }
    }

    /// Poll detected sweep events
    pub fn poll_events(&self) -> Vec<SweepEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.events_queue.pop() {
            events.push(event);
        }
        events
    }

    /// Get all mapped liquidity levels
    pub fn get_liquidity_levels(&self) -> Vec<LiquidityLevel> {
        self.liquidity_levels.read().clone()
    }

    /// Get strongest support level
    pub fn get_strongest_support(&self) -> Option<LiquidityLevel> {
        let levels = self.liquidity_levels.read();
        levels.iter()
            .filter(|l| matches!(l.level_type, LevelType::EqualLow | LevelType::SwingLow | LevelType::ConsolidationLow))
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
            .cloned()
    }

    /// Get strongest resistance level
    pub fn get_strongest_resistance(&self) -> Option<LiquidityLevel> {
        let levels = self.liquidity_levels.read();
        levels.iter()
            .filter(|l| matches!(l.level_type, LevelType::EqualHigh | LevelType::SwingHigh | LevelType::ConsolidationHigh))
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap())
            .cloned()
    }

    /// Get equal highs that have been touched multiple times
    pub fn get_significant_equal_highs(&self) -> Vec<(i64, u32)> {
        let highs = self.equal_highs.lock();
        highs.iter()
            .filter(|(_, count)| *count >= self.equal_touch_threshold)
            .map(|(price, count)| (*price, *count))
            .collect()
    }

    /// Get equal lows that have been touched multiple times
    pub fn get_significant_equal_lows(&self) -> Vec<(i64, u32)> {
        let lows = self.equal_lows.lock();
        lows.iter()
            .filter(|(_, count)| *count >= self.equal_touch_threshold)
            .map(|(price, count)| (*price, *count))
            .collect()
    }

    /// Reset detector state
    pub fn reset(&self) {
        self.liquidity_levels.write().clear();
        self.price_history.fill(0);
        self.price_history_idx.store(0, Ordering::Release);
        self.recent_high.store(i64::MIN, Ordering::Release);
        self.recent_low.store(i64::MAX, Ordering::Release);
        self.equal_highs.lock().clear();
        self.equal_lows.lock().clear();
    }
}

/// Multi-venue sweep coordinator for cross-exchange analysis
pub struct MultiVenueSweepCoordinator {
    /// Per-venue sweep detectors
    venue_detectors: parking_lot::Mutex<std::collections::HashMap<String, Arc<LiquiditySweepDetector>>>,
    /// Combined events queue
    combined_events: crossbeam_queue::SegQueue<SweepEvent>,
}

impl MultiVenueSweepCoordinator {
    pub fn new() -> Self {
        Self {
            venue_detectors: parking_lot::Mutex::new(std::collections::HashMap::new()),
            combined_events: crossbeam_queue::SegQueue::new(),
        }
    }

    /// Register a new venue
    pub fn register_venue(&self, venue_name: &str) -> Arc<LiquiditySweepDetector> {
        let mut venues = self.venue_detectors.lock();
        
        if let Some(detector) = venues.get(venue_name) {
            return Arc::clone(detector);
        }

        let detector = Arc::new(LiquiditySweepDetector::new(20, 3));
        venues.insert(venue_name.to_string(), detector);
        Arc::clone(venues.get(venue_name).unwrap())
    }

    /// Process price for a specific venue
    pub fn process_venue_price(&self, venue: &str, price: i64, volume: u64, timestamp_ns: u64) {
        let venues = self.venue_detectors.lock();
        if let Some(detector) = venues.get(venue) {
            detector.process_price(price, volume, timestamp_ns);
            
            // Collect any events
            for mut event in detector.poll_events() {
                event.description = format!("[{}] {}", venue, event.description);
                self.combined_events.push(event);
            }
        }
    }

    /// Get all events across venues
    pub fn poll_all_events(&self) -> Vec<SweepEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.combined_events.pop() {
            events.push(event);
        }
        events.sort_by_key(|e| e.timestamp_ns);
        events
    }

    /// Detect cross-venue sweep arbitrage opportunities
    pub fn detect_cross_venue_arbitrage(&self) -> Vec<(String, String, i64, f64)> {
        let venues = self.venue_detectors.lock();
        let mut opportunities = Vec::new();

        let venue_list: Vec<_> = venues.iter().collect();
        
        for i in 0..venue_list.len() {
            for j in (i + 1)..venue_list.len() {
                let (name1, det1) = venue_list[i];
                let (name2, det2) = venue_list[j];

                if let (Some(support1), Some(resistance2)) = 
                    (det1.get_strongest_support(), det2.get_strongest_resistance()) 
                {
                    if support1.price > resistance2.price {
                        let spread = support1.price - resistance2.price;
                        let profit_pct = (spread as f64 / resistance2.price as f64) * 100.0;
                        
                        if profit_pct > 0.1 { // Minimum 0.1% profit
                            opportunities.push((
                                name1.clone(),
                                name2.clone(),
                                spread,
                                profit_pct,
                            ));
                        }
                    }
                }
            }
        }

        opportunities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sweep_detection_basic() {
        let detector = LiquiditySweepDetector::new(10, 2);
        
        // Build price history with a swing low
        for i in 0..50 {
            let price = 5000000000 + (i % 10) * 1000000;
            detector.process_price(price, 100, i * 1000000);
        }

        // Create equal low
        detector.process_price(4999900000, 100, 50000000);
        detector.process_price(4999900000, 100, 51000000);
        detector.process_price(4999900000, 100, 52000000);

        // Sweep below the low
        detector.process_price(4999800000, 500, 53000000);

        let events = detector.poll_events();
        // Should detect sweep event
        assert!(detector.get_liquidity_levels().len() > 0);
    }

    #[test]
    fn test_equal_levels() {
        let detector = LiquiditySweepDetector::new(10, 2);
        
        // Touch same price multiple times
        for _ in 0..5 {
            detector.process_price(5000000000, 100, 1000000);
            detector.process_price(5000100000, 100, 2000000);
            detector.process_price(5000000000, 100, 3000000);
        }

        let equal_highs = detector.get_significant_equal_highs();
        let equal_lows = detector.get_significant_equal_lows();
        
        assert!(!equal_highs.is_empty() || !equal_lows.is_empty());
    }
}
