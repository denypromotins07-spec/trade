//! `src/market/funding.rs`
//! 
//! **Perpetual Swap Funding Rate Engine**
//! 
//! Implements funding rate calculators and open interest trackers for perpetual swaps.
//! Helps the bot dynamically adjust leverage and avoid holding positions during expensive funding intervals.
//! 
//! **Features:**
//! - Funding rate calculation and prediction
//! - Open interest tracking
//! - Annualized funding cost estimation
//! - Optimal position timing suggestions

use std::collections::VecDeque;

/// Funding rate data point
#[derive(Debug, Clone)]
pub struct FundingRate {
    pub timestamp: u64,
    pub rate: f64,      // Actual funding rate (e.g., 0.0001 = 0.01%)
    pub mark_price: f64,
    pub index_price: f64,
}

/// Open interest snapshot
#[derive(Debug, Clone)]
pub struct OpenInterest {
    pub timestamp: u64,
    pub open_interest: f64,  // In base currency units
    pub volume_24h: f64,
}

/// Funding rate statistics
#[derive(Debug, Clone)]
pub struct FundingStats {
    pub current_rate: f64,
    pub avg_rate_8h: f64,
    pub avg_rate_24h: f64,
    pub annualized_rate: f64,
    pub predicted_next_rate: f64,
}

/// Position side for funding calculations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Funding rate tracker and analyzer
pub struct FundingTracker {
    funding_history: VecDeque<FundingRate>,
    oi_history: VecDeque<OpenInterest>,
    max_history_size: usize,
    funding_interval_hours: u64,
}

impl FundingTracker {
    pub fn new(max_history_size: usize, funding_interval_hours: u64) -> Self {
        Self {
            funding_history: VecDeque::with_capacity(max_history_size),
            oi_history: VecDeque::with_capacity(max_history_size),
            max_history_size,
            funding_interval_hours,
        }
    }

    /// Adds a new funding rate observation
    #[inline]
    pub fn add_funding_rate(&mut self, rate: FundingRate) {
        self.funding_history.push_back(rate);
        while self.funding_history.len() > self.max_history_size {
            self.funding_history.pop_front();
        }
    }

    /// Adds an open interest observation
    #[inline]
    pub fn add_open_interest(&mut self, oi: OpenInterest) {
        self.oi_history.push_back(oi);
        while self.oi_history.len() > self.max_history_size {
            self.oi_history.pop_front();
        }
    }

    /// Calculates current funding statistics
    #[inline]
    pub fn get_stats(&self) -> Option<FundingStats> {
        if self.funding_history.is_empty() {
            return None;
        }

        let current_rate = self.funding_history.back()?.rate;

        // Calculate averages
        let rates: Vec<f64> = self.funding_history.iter().map(|f| f.rate).collect();
        
        let avg_8h = calculate_average(&rates, 8 / self.funding_interval_hours as usize);
        let avg_24h = calculate_average(&rates, 24 / self.funding_interval_hours as usize);
        
        // Annualized: rate * (365 * 24 / funding_interval)
        let periods_per_year = (365 * 24) / self.funding_interval_hours;
        let annualized_rate = current_rate * periods_per_year as f64;

        // Simple prediction using linear regression on recent rates
        let predicted_next = predict_next_funding_rate(&rates);

        Some(FundingStats {
            current_rate,
            avg_rate_8h: avg_8h,
            avg_rate_24h: avg_24h,
            annualized_rate,
            predicted_next_rate: predicted_next,
        })
    }

    /// Calculates the expected funding payment for a position
    #[inline]
    pub fn calculate_funding_payment(
        &self,
        position_side: PositionSide,
        notional_value: f64,
    ) -> Option<f64> {
        let stats = self.get_stats()?;
        
        // Payment = notional * funding_rate
        // Long pays when rate is positive, Short receives
        // Short pays when rate is negative, Long receives
        let payment = match position_side {
            PositionSide::Long => notional_value * stats.current_rate,
            PositionSide::Short => -notional_value * stats.current_rate,
        };

        Some(payment)
    }

    /// Returns whether it's a good time to hold a long position based on funding
    #[inline]
    pub fn is_favorable_for_long(&self, threshold: f64) -> bool {
        if let Some(stats) = self.get_stats() {
            stats.annualized_rate < threshold
        } else {
            true
        }
    }

    /// Returns whether it's a good time to hold a short position based on funding
    #[inline]
    pub fn is_favorable_for_short(&self, threshold: f64) -> bool {
        if let Some(stats) = self.get_stats() {
            stats.annualized_rate > -threshold
        } else {
            true
        }
    }

    /// Gets the current open interest trend
    #[inline]
    pub fn get_oi_trend(&self) -> OiTrend {
        if self.oi_history.len() < 2 {
            return OiTrend::Unknown;
        }

        let recent: Vec<f64> = self.oi_history.iter()
            .take(5)
            .map(|o| o.open_interest)
            .collect();
        
        if recent.len() < 2 {
            return OiTrend::Unknown;
        }

        let first_avg = recent[..recent.len()/2].iter().sum::<f64>() / (recent.len()/2) as f64;
        let second_avg = recent[recent.len()/2..].iter().sum::<f64>() / (recent.len() - recent.len()/2) as f64;

        let change_pct = (second_avg - first_avg) / first_avg;

        if change_pct > 0.05 {
            OiTrend::IncreasingStrong
        } else if change_pct > 0.01 {
            OiTrend::IncreasingMild
        } else if change_pct < -0.05 {
            OiTrend::DecreasingStrong
        } else if change_pct < -0.01 {
            OiTrend::DecreasingMild
        } else {
            OiTrend::Stable
        }
    }

    /// Returns time until next funding payment (in hours)
    #[inline]
    pub fn hours_until_next_funding(&self) -> u64 {
        // Simplified: assumes funding every 8 hours at 00:00, 08:00, 16:00 UTC
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let hours_since_epoch = now / 3600;
        let next_funding_hour = ((hours_since_epoch / self.funding_interval_hours) + 1) * self.funding_interval_hours;
        
        (next_funding_hour - hours_since_epoch) as u64
    }
}

/// Open interest trend classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OiTrend {
    IncreasingStrong,
    IncreasingMild,
    Stable,
    DecreasingMild,
    DecreasingStrong,
    Unknown,
}

#[inline]
fn calculate_average(rates: &[f64], num_periods: usize) -> f64 {
    if rates.is_empty() || num_periods == 0 {
        return 0.0;
    }
    
    let take_count = num_periods.min(rates.len());
    let sum: f64 = rates[rates.len() - take_count..].iter().sum();
    sum / take_count as f64
}

#[inline]
fn predict_next_funding_rate(rates: &[f64]) -> f64 {
    if rates.len() < 2 {
        return rates.last().copied().unwrap_or(0.0);
    }

    // Simple momentum-based prediction
    let recent = rates[rates.len().saturating_sub(3)..].to_vec();
    if recent.len() < 2 {
        return *rates.last().unwrap();
    }

    // Linear extrapolation
    let first_half: f64 = recent[..recent.len()/2].iter().sum::<f64>() / (recent.len()/2) as f64;
    let second_half: f64 = recent[recent.len()/2..].iter().sum::<f64>() / (recent.len() - recent.len()/2) as f64;
    
    let momentum = second_half - first_half;
    second_half + momentum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_funding_tracker() {
        let mut tracker = FundingTracker::new(100, 8);
        
        // Add some sample funding rates
        for i in 0..10 {
            tracker.add_funding_rate(FundingRate {
                timestamp: i * 8 * 3600,
                rate: 0.0001 + (i as f64 * 0.00001),
                mark_price: 50000.0,
                index_price: 49995.0,
            });
        }

        let stats = tracker.get_stats().unwrap();
        assert!(stats.current_rate > 0.0);
        assert!(stats.annualized_rate > 0.0);
    }
}
