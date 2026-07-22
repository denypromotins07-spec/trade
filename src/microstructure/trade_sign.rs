//! Lee-Ready Algorithm and Trade Sign Classification
//!
//! Implements the Lee-Ready algorithm and advanced trade sign classification
//! to accurately label aggressive buyers vs sellers directly from raw exchange feeds.
//! Optimized for microsecond latency on AMD Ryzen AI 5 architecture.

use std::cmp::Ordering;

/// Trade direction classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSign {
    /// Aggressive buyer (took from ask side)
    Buy,
    /// Aggressive seller (hit bid side)
    Sell,
    /// Unable to classify
    Unknown,
}

/// Tick rule classification for bulk volume analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TickRule {
    Uptick,
    ZeroUptick,
    Downtick,
    ZeroDowntick,
}

/// Result of trade sign classification
#[derive(Debug, Clone)]
pub struct ClassifiedTrade {
    pub timestamp_ns: u64,
    pub price: f64,
    pub quantity: f64,
    pub sign: TradeSign,
    pub confidence: f64,
    pub method: &'static str,
}

/// Lee-Ready classifier state
pub struct LeeReadyClassifier {
    /// Previous tick price
    prev_price: Option<f64>,
    /// Previous tick direction
    prev_direction: Option<TickRule>,
    /// Rolling mid-price estimate
    mid_price: f64,
    /// Mid-price update count
    mid_update_count: usize,
}

impl LeeReadyClassifier {
    /// Create a new Lee-Ready classifier
    pub fn new() -> Self {
        Self {
            prev_price: None,
            prev_direction: None,
            mid_price: 0.0,
            mid_update_count: 0,
        }
    }

    /// Update the reference mid-price
    #[inline]
    pub fn update_mid_price(&mut self, bid: f64, ask: f64) {
        if bid > 0.0 && ask > 0.0 && ask > bid {
            let new_mid = (bid + ask) / 2.0;
            // Exponential moving average for mid-price
            let alpha = 0.1;
            self.mid_price = alpha * new_mid + (1.0 - alpha) * self.mid_price;
            self.mid_update_count += 1;
        }
    }

    /// Classify a single trade using Lee-Ready rules
    ///
    /// Lee-Ready Algorithm:
    /// 1. Compare trade price to current bid-ask midpoint
    /// 2. If price > midpoint + epsilon -> Buy
    /// 3. If price < midpoint - epsilon -> Sell
    /// 4. If near midpoint, use tick test (price change from previous trade)
    pub fn classify(&mut self, price: f64, _quantity: f64) -> TradeSign {
        // Need valid mid-price
        if self.mid_update_count == 0 || self.mid_price <= 0.0 {
            return TradeSign::Unknown;
        }

        // Price comparison with epsilon for spread crossing
        let epsilon = self.mid_price * 0.0001; // 1 basis point tolerance

        match price.partial_cmp(&self.mid_price) {
            Some(Ordering::Greater) if price > self.mid_price + epsilon => {
                self.prev_price = Some(price);
                TradeSign::Buy
            }
            Some(Ordering::Less) if price < self.mid_price - epsilon => {
                self.prev_price = Some(price);
                TradeSign::Sell
            }
            _ => {
                // Near midpoint - use tick test
                self.classify_tick_test(price)
            }
        }
    }

    /// Tick test for trades at the midpoint
    fn classify_tick_test(&mut self, price: f64) -> TradeSign {
        let direction = match self.prev_price {
            Some(prev) => {
                match price.partial_cmp(&prev) {
                    Some(Ordering::Greater) => TickRule::Uptick,
                    Some(Ordering::Less) => TickRule::Downtick,
                    Some(Ordering::Equal) => {
                        // Zero tick - inherit previous direction or use previous price comparison
                        match self.prev_direction {
                            Some(TickRule::Uptick) | Some(TickRule::ZeroUptick) => TickRule::ZeroUptick,
                            Some(TickRule::Downtick) | Some(TickRule::ZeroDowntick) => TickRule::ZeroDowntick,
                            None => TickRule::Uptick, // Default assumption
                        }
                    }
                    None => TickRule::Uptick,
                }
            }
            None => TickRule::Uptick,
        };

        self.prev_price = Some(price);
        self.prev_direction = Some(direction);

        match direction {
            TickRule::Uptick | TickRule::ZeroUptick => TradeSign::Buy,
            TickRule::Downtick | TickRule::ZeroDowntick => TradeSign::Sell,
        }
    }

    /// Classify with confidence score
    pub fn classify_with_confidence(
        &mut self,
        price: f64,
        quantity: f64,
        bid: f64,
        ask: f64,
    ) -> ClassifiedTrade {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Update mid-price first
        self.update_mid_price(bid, ask);

        let sign = self.classify(price, quantity);

        // Compute confidence based on price position within spread
        let confidence = if self.mid_price > 0.0 && ask > bid {
            let spread = ask - bid;
            let distance_from_mid = (price - self.mid_price).abs();
            let max_distance = spread / 2.0;

            if max_distance > 0.0 {
                // Higher confidence when price is far from midpoint (clearly buy or sell)
                (distance_from_mid / max_distance).min(1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        let method = if confidence > 0.8 {
            "lee_ready_quote"
        } else if confidence > 0.5 {
            "lee_ready_tick"
        } else {
            "lee_ready_unknown"
        };

        ClassifiedTrade {
            timestamp_ns,
            price,
            quantity,
            sign,
            confidence,
            method,
        }
    }

    /// Reset classifier state
    pub fn reset(&mut self) {
        self.prev_price = None;
        self.prev_direction = None;
        self.mid_price = 0.0;
        self.mid_update_count = 0;
    }
}

impl Default for LeeReadyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Bulk Volume Classification (BVC) for aggregated data
pub struct BulkVolumeClassifier {
    /// Average daily volume for scaling
    avg_daily_volume: f64,
    /// Sigma parameter for normal CDF
    sigma: f64,
}

impl BulkVolumeClassifier {
    /// Create new BVC classifier
    pub fn new(avg_daily_volume: f64, sigma: f64) -> Self {
        Self {
            avg_daily_volume: avg_daily_volume.max(1.0),
            sigma: sigma.max(0.001),
        }
    }

    /// Classify net signed volume using bulk volume method
    ///
    /// BVC assumes that the proportion of buyer-initiated trades
    /// follows a normal distribution centered at the price change.
    pub fn classify_bar(&self, price_change: f64, volume: f64) -> (f64, f64) {
        // Z-score of price change
        let z = price_change / self.sigma;

        // Normal CDF approximation (buyer proportion)
        let buyer_proportion = self.normal_cdf(z);
        let seller_proportion = 1.0 - buyer_proportion;

        // Signed volume
        let signed_volume = volume * (buyer_proportion - seller_proportion);

        (buyer_proportion, signed_volume)
    }

    /// Standard normal CDF approximation
    #[inline]
    fn normal_cdf(&self, x: f64) -> f64 {
        // Approximation using error function
        // Φ(x) ≈ 0.5 * (1 + erf(x / √2))
        const SQRT_2: f64 = 1.4142135623730951;
        
        // Abramowitz and Stegun approximation
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989422804014327; // 1/√(2π)
        
        let poly = t * (0.319381530 + t * (-0.356563782 + t * 
                   (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        
        let result = 0.5 + d * x.exp() * poly * if x >= 0.0 { 1.0 } else { -1.0 };
        result.max(0.0).min(1.0)
    }

    /// Update average daily volume
    pub fn update_avg_volume(&mut self, new_avg: f64) {
        self.avg_daily_volume = new_avg.max(1.0);
    }
}

/// Combined classifier using multiple methods
pub struct EnsembleTradeClassifier {
    lee_ready: LeeReadyClassifier,
    bvc: BulkVolumeClassifier,
    /// Weight for Lee-Ready vs BVC
    lee_ready_weight: f64,
}

impl EnsembleTradeClassifier {
    /// Create ensemble classifier
    pub fn new(avg_daily_volume: f64, sigma: f64, lee_ready_weight: f64) -> Self {
        Self {
            lee_ready: LeeReadyClassifier::new(),
            bvc: BulkVolumeClassifier::new(avg_daily_volume, sigma),
            lee_ready_weight: lee_ready_weight.clamp(0.0, 1.0),
        }
    }

    /// Classify trade using ensemble of methods
    pub fn classify(
        &mut self,
        price: f64,
        quantity: f64,
        bid: f64,
        ask: f64,
        prev_close: f64,
    ) -> ClassifiedTrade {
        // Lee-Ready classification
        let lr_result = self.lee_ready.classify_with_confidence(price, quantity, bid, ask);

        // BVC classification
        let price_change = if prev_close > 0.0 {
            price - prev_close
        } else {
            0.0
        };
        let (_, bvc_signed_volume) = self.bvc.classify_bar(price_change, quantity);
        
        let bvc_sign = if bvc_signed_volume > 0.0 {
            TradeSign::Buy
        } else if bvc_signed_volume < 0.0 {
            TradeSign::Sell
        } else {
            TradeSign::Unknown
        };

        // Ensemble decision
        let final_sign = if lr_result.confidence > 0.7 || self.lee_ready_weight > 0.5 {
            lr_result.sign
        } else {
            bvc_sign
        };

        // Adjust confidence based on agreement
        let agreement_bonus = if lr_result.sign == bvc_sign { 0.1 } else { -0.1 };
        let final_confidence = (lr_result.confidence + agreement_bonus).clamp(0.0, 1.0);

        ClassifiedTrade {
            timestamp_ns: lr_result.timestamp_ns,
            price,
            quantity,
            sign: final_sign,
            confidence: final_confidence,
            method: if lr_result.confidence > 0.7 {
                "ensemble_lee_ready_dominant"
            } else if bvc_sign != TradeSign::Unknown {
                "ensemble_combined"
            } else {
                "ensemble_lee_ready_only"
            },
        }
    }

    /// Get cumulative order flow (net signed volume)
    pub fn cumulative_order_flow(&self, trades: &[ClassifiedTrade]) -> f64 {
        trades.iter()
            .map(|t| {
                let sign = match t.sign {
                    TradeSign::Buy => 1.0,
                    TradeSign::Sell => -1.0,
                    TradeSign::Unknown => 0.0,
                };
                sign * t.quantity * t.confidence
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lee_ready_classifier() {
        let mut classifier = LeeReadyClassifier::new();
        
        // Set up bid-ask
        classifier.update_mid_price(100.0, 100.1);
        
        // Trade above mid should be buy
        let sign = classifier.classify(100.08, 1.0);
        assert_eq!(sign, TradeSign::Buy);
        
        // Trade below mid should be sell
        let sign = classifier.classify(99.95, 1.0);
        assert_eq!(sign, TradeSign::Sell);
    }

    #[test]
    fn test_tick_test() {
        let mut classifier = LeeReadyClassifier::new();
        classifier.update_mid_price(100.0, 100.0); // Zero spread
        
        // Series of upticks
        let sign1 = classifier.classify(100.0, 1.0);
        let sign2 = classifier.classify(100.1, 1.0);
        let sign3 = classifier.classify(100.2, 1.0);
        
        assert_eq!(sign2, TradeSign::Buy);
        assert_eq!(sign3, TradeSign::Buy);
    }

    #[test]
    fn test_bulk_volume_classifier() {
        let bvc = BulkVolumeClassifier::new(1_000_000.0, 0.02);
        
        // Large positive price change -> mostly buys
        let (buyer_prop, signed_vol) = bvc.classify_bar(0.05, 1000.0);
        assert!(buyer_prop > 0.5);
        assert!(signed_vol > 0.0);
        
        // Large negative price change -> mostly sells
        let (buyer_prop, signed_vol) = bvc.classify_bar(-0.05, 1000.0);
        assert!(buyer_prop < 0.5);
        assert!(signed_vol < 0.0);
    }

    #[test]
    fn test_ensemble_classifier() {
        let mut classifier = EnsembleTradeClassifier::new(1_000_000.0, 0.02, 0.7);
        
        let trade = classifier.classify(
            100.5,   // price
            10.0,    // quantity
            100.4,   // bid
            100.6,   // ask
            100.0,   // prev close
        );
        
        assert!(trade.confidence > 0.0);
        assert!(trade.confidence <= 1.0);
    }
}
