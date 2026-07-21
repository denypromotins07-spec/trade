//! Liquidity Engineering & Market Manipulation Detection - Chapter 2
//! File 6: engineering.rs
//! 
//! Develops market maker behavior analyzers to distinguish between
//! genuine institutional order flow and algorithmic liquidity engineering
//! designed to trap retail traders.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Market maker behavior classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum MMBehavior {
    /// Genuine institutional liquidity provision
    GenuineInstitutional,
    /// Algorithmic liquidity engineering (trap)
    AlgoLiquidityTrap,
    /// High-frequency market making
    HFTMarketMaking,
    /// Retail-driven flow
    RetailFlow,
    /// Mixed/uncertain behavior
    Uncertain,
}

/// Detected liquidity trap pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityTrap {
    pub trap_type: TrapType,
    pub timestamp_ns: u64,
    pub price: i64,
    pub confidence: f64,
    pub estimated_retail_exposure: u64,
    pub mm_position_estimate: i64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TrapType {
    /// Bull trap: fake breakout above resistance
    BullTrap,
    /// Bear trap: fake breakdown below support
    BearTrap,
    /// Liquidity grab: sweep stops before reversal
    LiquidityGrab,
    /// Fakeout: rapid reversal after apparent breakout
    Fakeout,
    /// Stop hunt: targeted stop loss triggering
    StopHunt,
}

/// Order flow characteristics for analysis
#[derive(Debug, Clone)]
pub struct OrderFlowCharacteristics {
    /// Average order size
    pub avg_order_size: f64,
    /// Order size variance
    pub order_size_variance: f64,
    /// Buy/sell ratio
    pub buy_sell_ratio: f64,
    /// Order submission rate per second
    pub submission_rate: f64,
    /// Cancellation rate (0-1)
    pub cancellation_rate: f64,
    /// Fill rate (0-1)
    pub fill_rate: f64,
    /// Price impact per unit volume
    pub price_impact: f64,
    /// Time between orders (nanoseconds)
    pub avg_inter_order_time: f64,
}

/// Market maker behavior analyzer
pub struct MMBehaviorAnalyzer {
    /// Recent order flow characteristics
    recent_flow: parking_lot::RwLock<Vec<OrderFlowCharacteristics>>,
    flow_buffer_size: usize,
    flow_idx: AtomicUsize,
    
    /// Accumulated statistics
    total_orders: AtomicU64,
    total_volume: AtomicU64,
    buy_volume: AtomicU64,
    sell_volume: AtomicU64,
    cancelled_orders: AtomicU64,
    filled_orders: AtomicU64,
    
    /// Detected traps queue
    traps_queue: crossbeam_queue::SegQueue<LiquidityTrap>,
    
    /// Configuration thresholds
    institutional_min_size: u64,
    hft_max_inter_order_ns: u64,
    trap_confidence_threshold: f64,
    
    /// Current behavior classification
    current_behavior: parking_lot::RwLock<MMBehavior>,
}

impl MMBehaviorAnalyzer {
    /// Create new MM behavior analyzer
    pub fn new(
        institutional_min_size: u64,
        hft_max_inter_order_ns: u64,
        trap_confidence_threshold: f64,
    ) -> Self {
        Self {
            recent_flow: parking_lot::RwLock::new(Vec::with_capacity(100)),
            flow_buffer_size: 100,
            flow_idx: AtomicUsize::new(0),
            total_orders: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            buy_volume: AtomicU64::new(0),
            sell_volume: AtomicU64::new(0),
            cancelled_orders: AtomicU64::new(0),
            filled_orders: AtomicU64::new(0),
            traps_queue: crossbeam_queue::SegQueue::new(),
            institutional_min_size,
            hft_max_inter_order_ns,
            trap_confidence_threshold,
            current_behavior: parking_lot::RwLock::new(MMBehavior::Uncertain),
        }
    }

    /// Record a single order event
    pub fn record_order(
        &self,
        quantity: u64,
        is_buy: bool,
        is_cancelled: bool,
        is_filled: bool,
        timestamp_ns: u64,
        inter_order_time_ns: u64,
    ) {
        // Update accumulators
        self.total_orders.fetch_add(1, Ordering::Relaxed);
        self.total_volume.fetch_add(quantity, Ordering::Relaxed);
        
        if is_buy {
            self.buy_volume.fetch_add(quantity, Ordering::Relaxed);
        } else {
            self.sell_volume.fetch_add(quantity, Ordering::Relaxed);
        }
        
        if is_cancelled {
            self.cancelled_orders.fetch_add(1, Ordering::Relaxed);
        }
        if is_filled {
            self.filled_orders.fetch_add(1, Ordering::Relaxed);
        }

        // Update flow characteristics periodically
        let idx = self.flow_idx.fetch_add(1, Ordering::Relaxed);
        if idx % 50 == 0 {
            self.update_flow_characteristics(timestamp_ns);
        }

        // Analyze for trap patterns
        self.analyze_for_traps(quantity, is_buy, is_cancelled, timestamp_ns);
    }

    /// Update order flow characteristics
    fn update_flow_characteristics(&self, _timestamp_ns: u64) {
        let mut flows = self.recent_flow.write();
        
        let total = self.total_orders.load(Ordering::Relaxed);
        let vol = self.total_volume.load(Ordering::Relaxed);
        let buy = self.buy_volume.load(Ordering::Relaxed);
        let cancelled = self.cancelled_orders.load(Ordering::Relaxed);
        let filled = self.filled_orders.load(Ordering::Relaxed);

        let characteristics = OrderFlowCharacteristics {
            avg_order_size: if total > 0 { vol as f64 / total as f64 } else { 0.0 },
            order_size_variance: 0.0, // Would need more detailed tracking
            buy_sell_ratio: if vol - buy > 0 { buy as f64 / (vol - buy) as f64 } else { 1.0 },
            submission_rate: 0.0, // Would need time window
            cancellation_rate: if total > 0 { cancelled as f64 / total as f64 } else { 0.0 },
            fill_rate: if total > 0 { filled as f64 / total as f64 } else { 0.0 },
            price_impact: 0.0, // Would need price tracking
            avg_inter_order_time: 0.0,
        };

        if flows.len() < self.flow_buffer_size {
            flows.push(characteristics);
        } else {
            flows.remove(0);
            flows.push(characteristics);
        }

        // Classify current behavior
        *self.current_behavior.write() = self.classify_behavior(&characteristics);
    }

    /// Classify market maker behavior based on characteristics
    fn classify_behavior(&self, chars: &OrderFlowCharacteristics) -> MMBehavior {
        // Check for HFT characteristics
        if chars.cancellation_rate > 0.7 && chars.fill_rate < 0.3 {
            return MMBehavior::HFTMarketMaking;
        }

        // Check for institutional characteristics
        if chars.avg_order_size >= self.institutional_min_size as f64 
           && chars.cancellation_rate < 0.3
           && chars.fill_rate > 0.5 
        {
            return MMBehavior::GenuineInstitutional;
        }

        // Check for algo trap characteristics
        if chars.cancellation_rate > 0.5 
           && chars.buy_sell_ratio > 0.8 
           && chars.fill_rate < 0.4 
        {
            return MMBehavior::AlgoLiquidityTrap;
        }

        // Default to retail flow for small orders with low cancellation
        if chars.avg_order_size < (self.institutional_min_size as f64 / 10.0)
           && chars.cancellation_rate < 0.2 
        {
            return MMBehavior::RetailFlow;
        }

        MMBehavior::Uncertain
    }

    /// Analyze order patterns for potential traps
    fn analyze_for_traps(&self, quantity: u64, is_buy: bool, is_cancelled: bool, timestamp_ns: u64) {
        let behavior = *self.current_behavior.read();
        
        // Only analyze when behavior suggests potential manipulation
        if behavior != MMBehavior::AlgoLiquidityTrap && behavior != MMBehavior::HFTMarketMaking {
            return;
        }

        // Detect bull trap: large fake buy orders above resistance
        if is_buy && quantity >= self.institutional_min_size && is_cancelled {
            let confidence = (quantity as f64 / self.institutional_min_size as f64).min(1.0) * 0.7;
            
            if confidence > self.trap_confidence_threshold {
                let trap = LiquidityTrap {
                    trap_type: TrapType::BullTrap,
                    timestamp_ns,
                    price: 0, // Would need actual price
                    confidence,
                    estimated_retail_exposure: quantity,
                    mm_position_estimate: -(quantity as i64),
                    description: format!(
                        "Potential bull trap: {} quantity buy order cancelled",
                        quantity
                    ),
                };
                self.traps_queue.push(trap);
            }
        }

        // Detect bear trap: large fake sell orders below support
        if !is_buy && quantity >= self.institutional_min_size && is_cancelled {
            let confidence = (quantity as f64 / self.institutional_min_size as f64).min(1.0) * 0.7;
            
            if confidence > self.trap_confidence_threshold {
                let trap = LiquidityTrap {
                    trap_type: TrapType::BearTrap,
                    timestamp_ns,
                    price: 0,
                    confidence,
                    estimated_retail_exposure: quantity,
                    mm_position_estimate: quantity as i64,
                    description: format!(
                        "Potential bear trap: {} quantity sell order cancelled",
                        quantity
                    ),
                };
                self.traps_queue.push(trap);
            }
        }
    }

    /// Poll detected traps
    pub fn poll_traps(&self) -> Vec<LiquidityTrap> {
        let mut traps = Vec::new();
        while let Ok(trap) = self.traps_queue.pop() {
            traps.push(trap);
        }
        traps
    }

    /// Get current behavior classification
    pub fn get_current_behavior(&self) -> MMBehavior {
        *self.current_behavior.read()
    }

    /// Get accumulated statistics
    pub fn get_statistics(&self) -> MMStatistics {
        let total = self.total_orders.load(Ordering::Relaxed);
        let vol = self.total_volume.load(Ordering::Relaxed);
        let buy = self.buy_volume.load(Ordering::Relaxed);
        let cancelled = self.cancelled_orders.load(Ordering::Relaxed);
        let filled = self.filled_orders.load(Ordering::Relaxed);

        MMStatistics {
            total_orders: total,
            total_volume: vol,
            buy_volume: buy,
            sell_volume: vol - buy,
            buy_sell_ratio: if vol > buy { buy as f64 / (vol - buy) as f64 } else { 1.0 },
            cancellation_rate: if total > 0 { cancelled as f64 / total as f64 } else { 0.0 },
            fill_rate: if total > 0 { filled as f64 / total as f64 } else { 0.0 },
            avg_order_size: if total > 0 { vol as f64 / total as f64 } else { 0.0 },
        }
    }

    /// Detect liquidity engineering patterns
    pub fn detect_liquidity_engineering(&self) -> Vec<EngineeringPattern> {
        let mut patterns = Vec::new();
        let stats = self.get_statistics();
        let behavior = self.get_current_behavior();

        // Pattern 1: High cancellation with directional bias
        if stats.cancellation_rate > 0.6 && stats.buy_sell_ratio > 1.5 {
            patterns.push(EngineeringPattern {
                pattern_type: EngineeringPatternType::FakeLiquidityWall,
                confidence: 0.8,
                description: "High cancellation rate with buy-side bias suggests fake liquidity wall".to_string(),
            });
        }

        // Pattern 2: Low fill rate with large orders
        if stats.fill_rate < 0.3 && stats.avg_order_size > self.institutional_min_size as f64 {
            patterns.push(EngineeringPattern {
                pattern_type: EngineeringPatternType::Spoofing,
                confidence: 0.7,
                description: "Large orders with low fill rate indicates potential spoofing".to_string(),
            });
        }

        // Pattern 3: HFT behavior with trap characteristics
        if behavior == MMBehavior::HFTMarketMaking {
            patterns.push(EngineeringPattern {
                pattern_type: EngineeringPatternType::HFTTrapping,
                confidence: 0.6,
                description: "HFT market making with potential retail trapping".to_string(),
            });
        }

        patterns
    }

    /// Reset analyzer state
    pub fn reset(&self) {
        self.total_orders.store(0, Ordering::Release);
        self.total_volume.store(0, Ordering::Release);
        self.buy_volume.store(0, Ordering::Release);
        self.sell_volume.store(0, Ordering::Release);
        self.cancelled_orders.store(0, Ordering::Release);
        self.filled_orders.store(0, Ordering::Release);
        self.recent_flow.write().clear();
        self.flow_idx.store(0, Ordering::Release);
        *self.current_behavior.write() = MMBehavior::Uncertain;
    }
}

/// Market maker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMStatistics {
    pub total_orders: u64,
    pub total_volume: u64,
    pub buy_volume: u64,
    pub sell_volume: u64,
    pub buy_sell_ratio: f64,
    pub cancellation_rate: f64,
    pub fill_rate: f64,
    pub avg_order_size: f64,
}

/// Detected engineering pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineeringPattern {
    pub pattern_type: EngineeringPatternType,
    pub confidence: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineeringPatternType {
    FakeLiquidityWall,
    Spoofing,
    Layering,
    HFTTrapping,
    MomentumIgnition,
    StopLossHunting,
}

/// Institutional flow detector for distinguishing real vs fake volume
pub struct InstitutionalFlowDetector {
    /// Large trade tracker
    large_trades: parking_lot::Mutex<Vec<(u64, bool, u64)>>, // (size, is_buy, timestamp)
    /// Cumulative institutional volume
    institutional_buy_vol: AtomicU64,
    institutional_sell_vol: AtomicU64,
    /// Threshold for institutional classification
    institutional_threshold: u64,
}

impl InstitutionalFlowDetector {
    pub fn new(institutional_threshold: u64) -> Self {
        Self {
            large_trades: parking_lot::Mutex::new(Vec::with_capacity(1000)),
            institutional_buy_vol: AtomicU64::new(0),
            institutional_sell_vol: AtomicU64::new(0),
            institutional_threshold,
        }
    }

    /// Process a trade
    pub fn process_trade(&self, quantity: u64, is_buy: bool, timestamp_ns: u64) {
        if quantity >= self.institutional_threshold {
            let mut trades = self.large_trades.lock();
            trades.push((quantity, is_buy, timestamp_ns));
            
            if trades.len() > 1000 {
                trades.remove(0);
            }

            if is_buy {
                self.institutional_buy_vol.fetch_add(quantity, Ordering::Relaxed);
            } else {
                self.institutional_sell_vol.fetch_add(quantity, Ordering::Relaxed);
            }
        }
    }

    /// Get institutional flow imbalance
    pub fn get_institutional_imbalance(&self) -> f64 {
        let buy = self.institutional_buy_vol.load(Ordering::Relaxed) as f64;
        let sell = self.institutional_sell_vol.load(Ordering::Relaxed) as f64;
        let total = buy + sell;

        if total <= 0.0 {
            return 0.0;
        }

        (buy - sell) / total
    }

    /// Get recent institutional trades
    pub fn get_recent_institutional_trades(&self, lookback_ms: u64) -> Vec<(u64, bool, u64)> {
        let trades = self.large_trades.lock();
        let cutoff = 0; // Would use actual timestamp
        trades.iter()
            .filter(|(_, _, ts)| *ts >= cutoff)
            .cloned()
            .collect()
    }

    /// Calculate institutional accumulation/distribution score
    pub fn get_accumulation_score(&self) -> f64 {
        let imbalance = self.get_institutional_imbalance();
        
        // Score from -1 (distribution) to +1 (accumulation)
        imbalance.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_classification() {
        let analyzer = MMBehaviorAnalyzer::new(1000, 1000000, 0.5);

        // Simulate HFT behavior: high cancellation, low fill
        for i in 0..100 {
            analyzer.record_order(100, true, true, false, i * 1000000, 500000);
        }

        let behavior = analyzer.get_current_behavior();
        assert_eq!(behavior, MMBehavior::HFTMarketMaking);

        let stats = analyzer.get_statistics();
        assert!(stats.cancellation_rate > 0.5);
    }

    #[test]
    fn test_institutional_detection() {
        let detector = InstitutionalFlowDetector::new(1000);

        // Simulate institutional buys
        detector.process_trade(5000, true, 1000000);
        detector.process_trade(3000, true, 2000000);
        detector.process_trade(100, true, 3000000); // Too small

        let imbalance = detector.get_institutional_imbalance();
        assert!(imbalance > 0.0); // Positive = net buying

        let score = detector.get_accumulation_score();
        assert!(score > 0.0);
    }
}
