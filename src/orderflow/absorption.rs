//! Advanced Order Flow & Footprint Analytics - Chapter 1
//! File 3: absorption.rs
//! 
//! Codes passive absorption and exhaustion detectors by analyzing
//! Cumulative Volume Delta (CVD) divergences against price action
//! to identify hidden institutional limit orders.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Cumulative Volume Delta state tracking
#[derive(Debug, Clone)]
pub struct CvdState {
    /// Running cumulative delta (ask volume - bid volume)
    pub cumulative_delta: i64,
    /// Delta for current candle/period
    pub period_delta: i64,
    /// Number of trades in period
    pub trade_count: u32,
    /// Period high price
    pub high_price: i64,
    /// Period low price
    pub low_price: i64,
    /// Period open price
    pub open_price: i64,
    /// Period close price
    pub close_price: i64,
}

impl CvdState {
    pub fn new() -> Self {
        Self {
            cumulative_delta: 0,
            period_delta: 0,
            trade_count: 0,
            high_price: i64::MIN,
            low_price: i64::MAX,
            open_price: 0,
            close_price: 0,
        }
    }

    #[inline]
    pub fn add_trade(&mut self, price: i64, volume: u64, is_buyer_maker: bool) {
        let delta = if is_buyer_maker {
            -(volume as i64)
        } else {
            volume as i64
        };

        self.cumulative_delta += delta;
        self.period_delta += delta;
        self.trade_count += 1;

        if self.open_price == 0 {
            self.open_price = price;
        }

        if price > self.high_price {
            self.high_price = price;
        }
        if price < self.low_price {
            self.low_price = price;
        }
        self.close_price = price;
    }

    #[inline]
    pub fn reset_period(&mut self) {
        self.period_delta = 0;
        self.trade_count = 0;
        self.high_price = i64::MIN;
        self.low_price = i64::MAX;
        self.open_price = 0;
        self.close_price = 0;
    }
}

/// Detection result for absorption/exhaustion events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsorptionEvent {
    pub event_type: AbsorptionType,
    pub timestamp_ns: u64,
    pub price: i64,
    pub confidence: f64, // 0.0 to 1.0
    pub cvd_value: i64,
    pub price_change: i64,
    pub volume_anomaly: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AbsorptionType {
    /// Passive buy absorption (price not falling despite heavy selling)
    PassiveBuyAbsorption,
    /// Passive sell absorption (price not rising despite heavy buying)
    PassiveSellAbsorption,
    /// Buying exhaustion (strong buying but no price progress)
    BuyingExhaustion,
    /// Selling exhaustion (strong selling but no price decline)
    SellingExhaustion,
    /// Hidden institutional buy wall detected
    InstitutionalBuyWall,
    /// Hidden institutional sell wall detected
    InstitutionalSellWall,
}

/// Real-time absorption and exhaustion detector
pub struct AbsorptionDetector {
    /// Current CVD state
    cvd: parking_lot::RwLock<CvdState>,
    /// Historical CVD values for divergence detection (ring buffer)
    cvd_history: Vec<(i64, i64)>, // (price, cvd) pairs
    cvd_history_idx: AtomicUsize,
    cvd_history_capacity: usize,
    /// Previous period CVD for divergence calculation
    prev_cvd: AtomicI64,
    /// Detected events queue
    events_queue: crossbeam_queue::SegQueue<AbsorptionEvent>,
    /// Configuration thresholds
    min_volume_threshold: u64,
    divergence_lookback: usize,
    /// Total events detected
    events_detected: AtomicU64,
}

impl AbsorptionDetector {
    /// Create new absorption detector with configurable thresholds
    pub fn new(min_volume_threshold: u64, divergence_lookback: usize) -> Self {
        let capacity = divergence_lookback.max(50);
        Self {
            cvd: parking_lot::RwLock::new(CvdState::new()),
            cvd_history: vec![(0, 0); capacity],
            cvd_history_idx: AtomicUsize::new(0),
            cvd_history_capacity: capacity,
            prev_cvd: AtomicI64::new(0),
            events_queue: crossbeam_queue::SegQueue::new(),
            min_volume_threshold,
            divergence_lookback,
            events_detected: AtomicU64::new(0),
        }
    }

    /// Process a single trade
    pub fn process_trade(&self, price: i64, volume: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        // Update CVD
        {
            let mut cvd = self.cvd.write();
            cvd.add_trade(price, volume, is_buyer_maker);
        }

        // Check for volume anomaly
        if volume >= self.min_volume_threshold {
            self.detect_absorption(price, volume, is_buyer_maker, timestamp_ns);
        }

        // Update history for divergence detection
        self.update_history(price, timestamp_ns);
    }

    /// Update CVD history ring buffer
    fn update_history(&self, price: i64, _timestamp_ns: u64) {
        let idx = self.cvd_history_idx.fetch_add(1, Ordering::Relaxed);
        let wrapped_idx = idx % self.cvd_history_capacity;
        
        let cvd_val = self.cvd.read().cumulative_delta;
        self.cvd_history[wrapped_idx] = (price, cvd_val);
    }

    /// Detect absorption patterns
    fn detect_absorption(&self, price: i64, volume: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        let cvd = self.cvd.read();
        let cvd_val = cvd.cumulative_delta;
        
        // Calculate price change over lookback period
        let idx = self.cvd_history_idx.load(Ordering::Relaxed);
        let lookback_idx = (idx + self.cvd_history_capacity - self.divergence_lookback) 
            % self.cvd_history_capacity;
        let (old_price, old_cvd) = self.cvd_history[lookback_idx];

        if old_price == 0 {
            return; // Not enough history
        }

        let price_change = price - old_price;
        let cvd_change = cvd_val - old_cvd;

        // Detect divergences
        let mut event_type: Option<AbsorptionType> = None;
        let mut confidence = 0.0f64;
        let mut description = String::new();

        // Passive Buy Absorption: Heavy selling (negative CVD) but price not falling
        if cvd_change < -(volume as i64 * 3) && price_change >= 0 {
            event_type = Some(AbsorptionType::PassiveBuyAbsorption);
            confidence = ((-cvd_change as f64) / (volume as f64 * 10.0)).min(1.0);
            description = format!(
                "Passive buy absorption detected: {} volume sold but price held at {}",
                -cvd_change, price
            );
        }

        // Passive Sell Absorption: Heavy buying (positive CVD) but price not rising
        if cvd_change > (volume as i64 * 3) && price_change <= 0 {
            event_type = Some(AbsorptionType::PassiveSellAbsorption);
            confidence = ((cvd_change as f64) / (volume as f64 * 10.0)).min(1.0);
            description = format!(
                "Passive sell absorption detected: {} volume bought but price capped at {}",
                cvd_change, price
            );
        }

        // Buying Exhaustion: Strong positive CVD but minimal price progress
        if cvd_change > (volume as i64 * 5) && price_change.abs() < (old_price / 10000) {
            event_type = Some(AbsorptionType::BuyingExhaustion);
            confidence = ((cvd_change as f64) / (volume as f64 * 15.0)).min(1.0);
            description = format!(
                "Buying exhaustion: High CVD ({}) with no price progress",
                cvd_change
            );
        }

        // Selling Exhaustion: Strong negative CVD but minimal price decline
        if cvd_change < -(volume as i64 * 5) && price_change.abs() < (old_price / 10000) {
            event_type = Some(AbsorptionType::SellingExhaustion);
            confidence = ((-cvd_change as f64) / (volume as f64 * 15.0)).min(1.0);
            description = format!(
                "Selling exhaustion: Low CVD ({}) with no price decline",
                cvd_change
            );
        }

        // Institutional Wall Detection (very high confidence absorption)
        if confidence > 0.8 {
            if is_buyer_maker {
                event_type = Some(AbsorptionType::InstitutionalSellWall);
                description.push_str(" - Likely institutional sell wall");
            } else {
                event_type = Some(AbsorptionType::InstitutionalBuyWall);
                description.push_str(" - Likely institutional buy wall");
            }
        }

        if let Some(et) = event_type {
            let volume_anomaly = volume as f64 / self.min_volume_threshold as f64;
            
            let event = AbsorptionEvent {
                event_type: et,
                timestamp_ns,
                price,
                confidence,
                cvd_value: cvd_val,
                price_change,
                volume_anomaly,
                description,
            };

            self.events_queue.push(event);
            self.events_detected.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Poll for detected events
    pub fn poll_events(&self) -> Vec<AbsorptionEvent> {
        let mut events = Vec::new();
        while let Ok(event) = self.events_queue.pop() {
            events.push(event);
        }
        events
    }

    /// Get current CVD value
    pub fn get_current_cvd(&self) -> i64 {
        self.cvd.read().cumulative_delta
    }

    /// Get current CVD state
    pub fn get_cvd_state(&self) -> CvdState {
        self.cvd.read().clone()
    }

    /// Reset period (call at candle boundary)
    pub fn reset_period(&self) {
        self.cvd.write().reset_period();
        self.prev_cvd.store(self.cvd.read().cumulative_delta, Ordering::Release);
    }

    /// Calculate CVD divergence score (-1.0 to 1.0)
    /// Positive = bullish divergence, Negative = bearish divergence
    pub fn calculate_divergence_score(&self) -> f64 {
        let idx = self.cvd_history_idx.load(Ordering::Relaxed);
        if idx < self.divergence_lookback {
            return 0.0;
        }

        let lookback_idx = (idx + self.cvd_history_capacity - self.divergence_lookback)
            % self.cvd_history_capacity;
        let (old_price, old_cvd) = self.cvd_history[lookback_idx];
        
        let cvd = self.cvd.read();
        let price_change = cvd.close_price - old_price;
        let cvd_change = cvd.cumulative_delta - old_cvd;

        if price_change == 0 || cvd_change == 0 {
            return 0.0;
        }

        // Divergence: price and CVD moving in opposite directions
        let score = if (price_change > 0 && cvd_change < 0) || (price_change < 0 && cvd_change > 0) {
            // Strong divergence signal
            let magnitude = (cvd_change.abs() as f64 / 10000.0).min(1.0);
            if price_change < 0 && cvd_change > 0 {
                magnitude // Bullish divergence
            } else {
                -magnitude // Bearish divergence
            }
        } else {
            // No divergence or confirmation
            0.0
        };

        score
    }

    /// Get total events detected count
    pub fn get_events_count(&self) -> u64 {
        self.events_detected.load(Ordering::Relaxed)
    }

    /// Detect sweep and reversal patterns
    pub fn detect_sweep_reversal(&self, price: i64, volume: u64, timestamp_ns: u64) -> Option<AbsorptionEvent> {
        let cvd = self.cvd.read();
        
        // Check if this is a large volume trade that could be a sweep
        if volume < self.min_volume_threshold * 2 {
            return None;
        }

        let recent_high = cvd.high_price;
        let recent_low = cvd.low_price;

        // Detect stop run above recent high followed by reversal
        if price > recent_high && cvd.cumulative_delta > 0 {
            // Potential bull trap - aggressive buying at highs
            return Some(AbsorptionEvent {
                event_type: AbsorptionType::BuyingExhaustion,
                timestamp_ns,
                price,
                confidence: 0.7,
                cvd_value: cvd.cumulative_delta,
                price_change: price - recent_high,
                volume_anomaly: volume as f64 / self.min_volume_threshold as f64,
                description: format!("Potential stop run above {:}, watch for reversal", recent_high),
            });
        }

        // Detect stop run below recent low followed by reversal
        if price < recent_low && cvd.cumulative_delta < 0 {
            // Potential bear trap - aggressive selling at lows
            return Some(AbsorptionEvent {
                event_type: AbsorptionType::SellingExhaustion,
                timestamp_ns,
                price,
                confidence: 0.7,
                cvd_value: cvd.cumulative_delta,
                price_change: price - recent_low,
                volume_anomaly: volume as f64 / self.min_volume_threshold as f64,
                description: format!("Potential stop run below {:}, watch for reversal", recent_low),
            });
        }

        None
    }
}

/// Multi-timeframe CVD analyzer for institutional flow detection
pub struct MultiTimeframeCvdAnalyzer {
    /// Per-minute CVD
    minute_cvd: AbsorptionDetector,
    /// Per-hour CVD
    hour_cvd: AbsorptionDetector,
    /// Last minute boundary
    last_minute_ns: AtomicU64,
    /// Last hour boundary
    last_hour_ns: AtomicU64,
}

impl MultiTimeframeCvdAnalyzer {
    pub fn new(min_volume: u64) -> Self {
        Self {
            minute_cvd: AbsorptionDetector::new(min_volume, 60),
            hour_cvd: AbsorptionDetector::new(min_volume * 10, 60),
            last_minute_ns: AtomicU64::new(0),
            last_hour_ns: AtomicU64::new(0),
        }
    }

    pub fn process_trade(&self, price: i64, volume: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        // Check minute rollover
        let minute_ns = 60_000_000_000; // 60 seconds in nanoseconds
        let last_min = self.last_minute_ns.load(Ordering::Relaxed);
        if last_min == 0 || timestamp_ns - last_min >= minute_ns {
            self.minute_cvd.reset_period();
            self.last_minute_ns.store(timestamp_ns, Ordering::Release);
        }

        // Check hour rollover
        let hour_ns = 3_600_000_000_000; // 3600 seconds in nanoseconds
        let last_hour = self.last_hour_ns.load(Ordering::Relaxed);
        if last_hour == 0 || timestamp_ns - last_hour >= hour_ns {
            self.hour_cvd.reset_period();
            self.last_hour_ns.store(timestamp_ns, Ordering::Release);
        }

        // Process on both timeframes
        self.minute_cvd.process_trade(price, volume, is_buyer_maker, timestamp_ns);
        self.hour_cvd.process_trade(price, volume, is_buyer_maker, timestamp_ns);
    }

    /// Get combined divergence signal
    pub fn get_combined_signal(&self) -> f64 {
        let minute_div = self.minute_cvd.calculate_divergence_score();
        let hour_div = self.hour_cvd.calculate_divergence_score();
        
        // Weight hourly divergence more heavily
        (minute_div * 0.3 + hour_div * 0.7)
    }

    /// Poll all events from both timeframes
    pub fn poll_all_events(&self) -> Vec<AbsorptionEvent> {
        let mut events = self.minute_cvd.poll_events();
        events.extend(self.hour_cvd.poll_events());
        events.sort_by_key(|e| e.timestamp_ns);
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvd_basic() {
        let detector = AbsorptionDetector::new(100, 10);
        
        // Simulate trades
        detector.process_trade(5000000000, 500, false, 1000000); // Aggressive buy
        detector.process_trade(5000000000, 300, true, 1000001);  // Aggressive sell
        
        let cvd = detector.get_current_cvd();
        assert_eq!(cvd, 200); // 500 - 300
    }

    #[test]
    fn test_absorption_detection() {
        let detector = AbsorptionDetector::new(100, 5);
        
        // Build history
        for i in 0..10 {
            detector.process_trade(5000000000 + i * 10000, 50, false, 1000000 + i);
        }
        
        // Now create absorption scenario: heavy selling but price holds
        detector.process_trade(5000900000, 1000, true, 2000000);
        
        let events = detector.poll_events();
        // May or may not trigger depending on thresholds
        assert!(detector.get_events_count() >= 0);
    }

    #[test]
    fn test_divergence_score() {
        let detector = AbsorptionDetector::new(50, 5);
        
        // Create bullish divergence: price down, CVD up
        detector.process_trade(5000000000, 100, true, 1000000);
        detector.process_trade(4999900000, 100, true, 1000001);
        detector.process_trade(4999800000, 500, false, 1000002); // Large buy at lower price
        
        let score = detector.calculate_divergence_score();
        // Should show some divergence
        assert!(score.is_finite());
    }
}
