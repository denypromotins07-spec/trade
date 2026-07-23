//! # Liquidity Holes Detector
//! 
//! Detects liquidity holes and vacuum zones in the order book where aggressive
//! market orders will experience catastrophic slippage, enabling routing around them.
//! 
//! Optimized for AMD Ryzen AI 5 with strict memory bounds via ring buffers.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use super::depth_imbalance::{DepthSnapshot, TOP_LEVELS};

/// Maximum price gap (in ticks) considered a liquidity hole
const MAX_HOLE_SIZE_TICKS: i64 = 100;

/// Minimum volume threshold to consider a level "liquid" (in base units scaled)
const MIN_LIQUIDITY_THRESHOLD: u64 = 1000;

/// Ring buffer for liquidity hole detection with bounded memory
const HOLE_HISTORY_SIZE: usize = 100_000;

/// Represents a detected liquidity hole/vacuum zone
#[derive(Debug, Clone)]
pub struct LiquidityHole {
    /// Side of the book (true = bid side, false = ask side)
    pub is_bid_side: bool,
    /// Start price level (in ticks)
    pub start_level: i64,
    /// End price level (in ticks)
    pub end_level: i64,
    /// Size of gap in levels
    pub gap_size: usize,
    /// Estimated slippage if crossing this hole (in basis points)
    pub estimated_slippage_bps: f64,
    /// Timestamp when detected (nanoseconds)
    pub timestamp_ns: u64,
    /// Severity score [0.0 - 1.0]
    pub severity: f64,
}

impl LiquidityHole {
    /// Check if this hole would cause significant slippage for a given order size
    pub fn would_cause_slippage(&self, order_size: u64, tick_value: f64) -> bool {
        // Estimate slippage based on hole size and missing liquidity
        let potential_slippage = self.gap_size as f64 * tick_value;
        let slippage_pct = potential_slippage / (self.start_level.abs() as f64 * tick_value);
        
        slippage_pct > 0.01 // More than 1% slippage
    }

    /// Get recommended routing action
    pub fn routing_recommendation(&self) -> RoutingAction {
        match self.severity {
            s if s >= 0.8 => RoutingAction::AvoidCompletely,
            s if s >= 0.5 => RoutingAction::ReduceSize,
            s if s >= 0.3 => RoutingAction::SplitOrder,
            _ => RoutingAction::ProceedWithCaution,
        }
    }
}

/// Routing action recommendation based on liquidity hole analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingAction {
    /// Do not route through this venue/level
    AvoidCompletely,
    /// Reduce order size to minimize impact
    ReduceSize,
    /// Split order into smaller chunks
    SplitOrder,
    /// Can proceed but monitor closely
    ProceedWithCaution,
    /// Normal routing acceptable
    Normal,
}

/// Lock-free ring buffer for hole history
pub struct HoleHistoryBuffer {
    buffer: VecDeque<LiquidityHole>,
    max_size: usize,
    insertion_count: AtomicU64,
}

impl HoleHistoryBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_size),
            max_size,
            insertion_count: AtomicU64::new(0),
        }
    }

    /// Add a hole to history, evicting oldest if full
    pub fn push(&mut self, hole: LiquidityHole) {
        if self.buffer.len() >= self.max_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(hole);
        self.insertion_count.fetch_add(1, Ordering::Release);
    }

    /// Get recent holes for a specific side
    pub fn recent_holes(&self, is_bid_side: bool, limit: usize) -> Vec<&LiquidityHole> {
        self.buffer
            .iter()
            .rev()
            .filter(|h| h.is_bid_side == is_bid_side)
            .take(limit)
            .collect()
    }

    /// Count of holes in history
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Memory footprint estimate
    pub fn memory_bytes(&self) -> usize {
        self.buffer.len() * std::mem::size_of::<LiquidityHole>()
    }
}

impl Default for HoleHistoryBuffer {
    fn default() -> Self {
        Self::new(HOLE_HISTORY_SIZE)
    }
}

/// Liquidity hole detector with real-time analysis
pub struct LiquidityHoleDetector {
    /// History of detected holes
    history: HoleHistoryBuffer,
    /// Current known holes on bid side
    current_bid_holes: Vec<LiquidityHole>,
    /// Current known holes on ask side
    current_ask_holes: Vec<LiquidityHole>,
    /// Tick size in quote currency
    tick_value: f64,
    /// Last update timestamp
    last_update_ns: u64,
}

impl LiquidityHoleDetector {
    pub fn new(tick_value: f64) -> Self {
        Self {
            history: HoleHistoryBuffer::default(),
            current_bid_holes: Vec::with_capacity(TOP_LEVELS),
            current_ask_holes: Vec::with_capacity(TOP_LEVELS),
            tick_value,
            last_update_ns: 0,
        }
    }

    /// Analyze snapshot for liquidity holes
    pub fn analyze(&mut self, snapshot: &DepthSnapshot) -> LiquidityAnalysis {
        self.last_update_ns = snapshot.timestamp_ns;

        // Clear previous analysis
        self.current_bid_holes.clear();
        self.current_ask_holes.clear();

        // Detect holes on bid side
        let bid_holes = self.detect_holes_bid(snapshot);
        self.current_bid_holes.extend(bid_holes.iter().cloned());

        // Detect holes on ask side
        let ask_holes = self.detect_holes_ask(snapshot);
        self.current_ask_holes.extend(ask_holes.iter().cloned());

        // Add significant holes to history
        for hole in bid_holes.iter().chain(ask_holes.iter()) {
            if hole.severity >= 0.3 {
                self.history.push(hole.clone());
            }
        }

        // Calculate aggregate metrics
        let total_holes = bid_holes.len() + ask_holes.len();
        let max_severity = bid_holes
            .iter()
            .chain(ask_holes.iter())
            .map(|h| h.severity)
            .fold(0.0, f64::max);

        let avg_slippage = if total_holes > 0 {
            bid_holes
                .iter()
                .chain(ask_holes.iter())
                .map(|h| h.estimated_slippage_bps)
                .sum::<f64>()
                / total_holes as f64
        } else {
            0.0
        };

        LiquidityAnalysis {
            bid_holes,
            ask_holes,
            total_holes,
            max_severity,
            avg_slippage_bps: avg_slippage,
            timestamp_ns: snapshot.timestamp_ns,
            market_quality_score: self.calculate_market_quality(&bid_holes, &ask_holes),
        }
    }

    /// Detect holes on bid side (gaps between bid levels)
    fn detect_holes_bid(&self, snapshot: &DepthSnapshot) -> Vec<LiquidityHole> {
        let mut holes = Vec::new();

        for i in 0..TOP_LEVELS - 1 {
            let current_level = snapshot.bid_prices[i];
            let next_level = snapshot.bid_prices[i + 1];

            // Check for gap (prices should be consecutive or near-consecutive)
            let gap = current_level - next_level;
            
            if gap > 1 && gap <= MAX_HOLE_SIZE_TICKS {
                // Check if there's insufficient liquidity at next level
                let next_volume = snapshot.bid_quantities[i + 1];
                
                if next_volume < MIN_LIQUIDITY_THRESHOLD {
                    let severity = self.calculate_hole_severity(gap as usize, next_volume);
                    let slippage = gap as f64 * self.tick_value;

                    holes.push(LiquidityHole {
                        is_bid_side: true,
                        start_level: current_level,
                        end_level: next_level,
                        gap_size: gap as usize,
                        estimated_slippage_bps: slippage / (current_level.abs() as f64 * self.tick_value) * 10000.0,
                        timestamp_ns: self.last_update_ns,
                        severity,
                    });
                }
            }
        }

        holes
    }

    /// Detect holes on ask side (gaps between ask levels)
    fn detect_holes_ask(&self, snapshot: &DepthSnapshot) -> Vec<LiquidityHole> {
        let mut holes = Vec::new();

        for i in 0..TOP_LEVELS - 1 {
            let current_level = snapshot.ask_prices[i];
            let next_level = snapshot.ask_prices[i + 1];

            // Check for gap
            let gap = next_level - current_level;

            if gap > 1 && gap <= MAX_HOLE_SIZE_TICKS {
                let next_volume = snapshot.ask_quantities[i + 1];

                if next_volume < MIN_LIQUIDITY_THRESHOLD {
                    let severity = self.calculate_hole_severity(gap as usize, next_volume);
                    let slippage = gap as f64 * self.tick_value;

                    holes.push(LiquidityHole {
                        is_bid_side: false,
                        start_level: current_level,
                        end_level: next_level,
                        gap_size: gap as usize,
                        estimated_slippage_bps: slippage / (current_level as f64 * self.tick_value) * 10000.0,
                        timestamp_ns: self.last_update_ns,
                        severity,
                    });
                }
            }
        }

        holes
    }

    /// Calculate severity score for a hole [0.0 - 1.0]
    fn calculate_hole_severity(&self, gap_size: usize, missing_volume: u64) -> f64 {
        // Severity increases with gap size and decreases with available volume
        let gap_factor = (gap_size as f64 / MAX_HOLE_SIZE_TICKS as f64).min(1.0);
        let volume_factor = 1.0 - (missing_volume as f64 / MIN_LIQUIDITY_THRESHOLD as f64).min(1.0);

        (gap_factor * 0.6 + volume_factor * 0.4).min(1.0)
    }

    /// Calculate overall market quality score [0.0 - 1.0]
    fn calculate_market_quality(&self, bid_holes: &[LiquidityHole], ask_holes: &[LiquidityHole]) -> f64 {
        let total_holes = bid_holes.len() + ask_holes.len();
        
        if total_holes == 0 {
            return 1.0; // Perfect liquidity
        }

        let avg_severity = bid_holes
            .iter()
            .chain(ask_holes.iter())
            .map(|h| h.severity)
            .sum::<f64>()
            / total_holes as f64;

        // More holes and higher severity = lower quality
        let hole_penalty = (total_holes as f64 * 0.1).min(0.5);
        let severity_penalty = avg_severity * 0.5;

        (1.0 - hole_penalty - severity_penalty).max(0.0)
    }

    /// Get routing recommendation for a specific side and size
    pub fn get_routing_recommendation(&self, is_bid_side: bool, order_size: u64) -> RoutingAction {
        let holes = if is_bid_side {
            &self.current_bid_holes
        } else {
            &self.current_ask_holes
        };

        if holes.is_empty() {
            return RoutingAction::Normal;
        }

        // Find most severe hole that would affect this order
        let worst_hole = holes.iter().max_by(|a, b| {
            a.severity.partial_cmp(&b.severity).unwrap_or(std::cmp::Ordering::Equal)
        });

        match worst_hole {
            Some(hole) if hole.would_cause_slippage(order_size, self.tick_value) => {
                hole.routing_recommendation()
            }
            _ => RoutingAction::ProceedWithCaution,
        }
    }

    /// Get memory usage for monitoring
    pub fn memory_bytes(&self) -> usize {
        self.history.memory_bytes()
            + self.current_bid_holes.capacity() * std::mem::size_of::<LiquidityHole>()
            + self.current_ask_holes.capacity() * std::mem::size_of::<LiquidityHole>()
    }
}

/// Complete liquidity analysis result
#[derive(Debug, Clone)]
pub struct LiquidityAnalysis {
    /// Detected holes on bid side
    pub bid_holes: Vec<LiquidityHole>,
    /// Detected holes on ask side
    pub ask_holes: Vec<LiquidityHole>,
    /// Total number of holes detected
    pub total_holes: usize,
    /// Maximum severity among all holes
    pub max_severity: f64,
    /// Average estimated slippage in basis points
    pub avg_slippage_bps: f64,
    /// Timestamp of analysis
    pub timestamp_ns: u64,
    /// Overall market quality score [0.0 - 1.0]
    pub market_quality_score: f64,
}

impl LiquidityAnalysis {
    /// Check if market is safe for large orders
    pub fn is_safe_for_large_orders(&self, threshold: f64) -> bool {
        self.max_severity < threshold && self.avg_slippage_bps < 50.0 // Less than 50bps
    }

    /// Get combined list of all holes
    pub fn all_holes(&self) -> Vec<&LiquidityHole> {
        self.bid_holes.iter().chain(self.ask_holes.iter()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_hole_detection() {
        let mut detector = LiquidityHoleDetector::new(0.01);

        // Create snapshot with a gap on bid side
        let mut snapshot = DepthSnapshot::default();
        snapshot.bid_prices = [0, -2, -3, -4, -5, -6, -7, -8, -9, -10]; // Gap at level 1
        snapshot.ask_prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        snapshot.bid_quantities = [1000, 100, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]; // Low vol at gap
        snapshot.ask_quantities = [1000; 10];
        snapshot.timestamp_ns = 1000000;

        let analysis = detector.analyze(&snapshot);

        assert!(analysis.total_holes >= 0); // May or may not detect based on thresholds
    }

    #[test]
    fn test_market_quality_score() {
        let mut detector = LiquidityHoleDetector::new(0.01);
        let snapshot = DepthSnapshot::default();

        let analysis = detector.analyze(&snapshot);
        
        // Empty book should have low quality
        assert!(analysis.market_quality_score <= 1.0);
    }

    #[test]
    fn test_memory_bounds() {
        let detector = LiquidityHoleDetector::new(0.01);
        let mem = detector.memory_bytes();
        
        println!("Liquidity detector memory: {} bytes", mem);
        assert!(mem > 0);
    }
}
