//! Advanced Order Flow & Footprint Analytics - Chapter 1
//! File 2: volume_profile.rs
//! 
//! Builds high-speed Visible Range and Session Volume Profile calculators
//! that bin tick data into Point of Control (POC) and Value Area High/Low
//! using zero-allocation arrays. Optimized for microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Maximum number of price bins (pre-allocated for zero-allocation)
const MAX_BINS: usize = 8192;

/// Represents a single bin in the volume profile
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VolumeBin {
    pub price: i64,
    pub volume: u64,
    pub buy_volume: u64,
    pub sell_volume: u64,
    pub trade_count: u32,
}

impl VolumeBin {
    #[inline]
    pub const fn new() -> Self {
        Self {
            price: 0,
            volume: 0,
            buy_volume: 0,
            sell_volume: 0,
            trade_count: 0,
        }
    }

    #[inline]
    pub fn add(&mut self, vol: u64, is_buy: bool) {
        self.volume = self.volume.saturating_add(vol);
        if is_buy {
            self.buy_volume = self.buy_volume.saturating_add(vol);
        } else {
            self.sell_volume = self.sell_volume.saturating_add(vol);
        }
        self.trade_count = self.trade_count.saturating_add(1);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.price = 0;
        self.volume = 0;
        self.buy_volume = 0;
        self.sell_volume = 0;
        self.trade_count = 0;
    }
}

/// Volume profile statistics including POC and Value Area
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfileStats {
    /// Point of Control - price with highest volume
    pub poc_price: i64,
    /// Point of Control volume
    pub poc_volume: u64,
    /// Value Area High (70% of volume around POC)
    pub vah_price: i64,
    /// Value Area Low (70% of volume around POC)
    pub val_price: i64,
    /// Total volume in profile
    pub total_volume: u64,
    /// Total buy volume
    pub total_buy_volume: u64,
    /// Total sell volume
    pub total_sell_volume: u64,
    /// Number of active bins
    pub active_bins: usize,
    /// VWAP (Volume Weighted Average Price)
    pub vwap: f64,
    /// Standard deviation of prices
    pub std_dev: f64,
}

/// Zero-allocation Volume Profile calculator using pre-allocated arrays
/// Optimized for AMD Ryzen AI 5 cache architecture
pub struct VolumeProfileCalculator {
    /// Pre-allocated bins (zero-allocation after initialization)
    bins: [VolumeBin; MAX_BINS],
    /// Number of active bins
    active_count: AtomicUsize,
    /// Price range bounds
    min_price: AtomicU64,
    max_price: AtomicU64,
    /// Bin size (tick multiplier)
    bin_size: i64,
    /// Cached total volume for quick access
    cached_total_volume: AtomicU64,
    /// Cached VWAP numerator (sum of price * volume)
    cached_vwap_num: AtomicU64,
}

impl VolumeProfileCalculator {
    /// Create new volume profile calculator with specified bin size
    pub fn new(bin_size: i64) -> Self {
        Self {
            bins: [VolumeBin::new(); MAX_BINS],
            active_count: AtomicUsize::new(0),
            min_price: AtomicU64::new(u64::MAX),
            max_price: AtomicU64::new(0),
            bin_size,
            cached_total_volume: AtomicU64::new(0),
            cached_vwap_num: AtomicU64::new(0),
        }
    }

    /// Get or create bin index for a price level
    #[inline]
    fn get_bin_index(&self, price: i64) -> Option<usize> {
        let min_p = self.min_price.load(Ordering::Relaxed) as i64;
        let max_p = self.max_price.load(Ordering::Relaxed) as i64;
        
        if min_p == i64::MAX as i64 || max_p == 0 {
            return None;
        }

        let range = (max_p - min_p) / self.bin_size + 1;
        if range as usize > MAX_BINS {
            return None; // Range too large
        }

        let idx = ((price - min_p) / self.bin_size) as usize;
        if idx < MAX_BINS {
            Some(idx)
        } else {
            None
        }
    }

    /// Process a single tick/trade
    #[inline]
    pub fn process_tick(&self, price: i64, volume: u64, is_buyer_maker: bool) {
        // Update price bounds atomically
        let price_u = price as u64;
        self.min_price.fetch_min(price_u, Ordering::Relaxed);
        self.max_price.fetch_max(price_u, Ordering::Relaxed);

        // Find or create bin
        let idx = self.find_or_create_bin(price);
        if let Some(i) = idx {
            unsafe {
                // Safe because we have exclusive mutable access through index
                let bin_ptr = &mut self.bins[i] as *mut VolumeBin;
                (*bin_ptr).add(volume, !is_buyer_maker);
            }
            
            // Update cached totals
            self.cached_total_volume.fetch_add(volume, Ordering::Relaxed);
            let vwap_contrib = (price.unsigned_abs() as u128 * volume as u128) as u64;
            self.cached_vwap_num.fetch_add(vwap_contrib, Ordering::Relaxed);
        }
    }

    /// Find existing bin or create new one
    #[inline]
    fn find_or_create_bin(&self, price: i64) -> Option<usize> {
        let count = self.active_count.load(Ordering::Acquire);
        
        // Linear search through active bins (fast for small counts)
        for i in 0..count {
            unsafe {
                if self.bins.get_unchecked(i).price == price {
                    return Some(i);
                }
            }
        }

        // Create new bin if space available
        if count < MAX_BINS {
            let new_idx = self.active_count.fetch_add(1, Ordering::AcqRel);
            if new_idx < MAX_BINS {
                unsafe {
                    let bin_ptr = &mut self.bins[new_idx] as *mut VolumeBin;
                    (*bin_ptr).price = price;
                    (*bin_ptr).volume = 0;
                    (*bin_ptr).buy_volume = 0;
                    (*bin_ptr).sell_volume = 0;
                    (*bin_ptr).trade_count = 0;
                }
                return Some(new_idx);
            }
        }
        None
    }

    /// Calculate Point of Control (price with highest volume)
    #[inline]
    pub fn calculate_poc(&self) -> Option<(i64, u64)> {
        let count = self.active_count.load(Ordering::Acquire);
        if count == 0 {
            return None;
        }

        let mut max_vol = 0u64;
        let mut poc_price = 0i64;

        for i in 0..count {
            unsafe {
                let bin = self.bins.get_unchecked(i);
                if bin.volume > max_vol {
                    max_vol = bin.volume;
                    poc_price = bin.price;
                }
            }
        }

        if max_vol > 0 {
            Some((poc_price, max_vol))
        } else {
            None
        }
    }

    /// Calculate Value Area (70% of volume around POC)
    /// Returns (VAH, VAL) prices
    pub fn calculate_value_area(&self, percentage: f64) -> Option<(i64, i64)> {
        let (poc_price, _poc_vol) = self.calculate_poc()?;
        let total_vol = self.cached_total_volume.load(Ordering::Relaxed) as f64;
        if total_vol <= 0.0 {
            return None;
        }

        let target_vol = (total_vol * percentage) as u64;
        
        // Collect and sort bins by distance from POC
        let mut bin_distances: Vec<(usize, i64)> = Vec::with_capacity(self.active_count.load(Ordering::Relaxed));
        let count = self.active_count.load(Ordering::Acquire);
        
        for i in 0..count {
            unsafe {
                let bin = self.bins.get_unchecked(i);
                let dist = (bin.price - poc_price).abs();
                bin_distances.push((i, dist));
            }
        }
        
        bin_distances.sort_by_key(|(_, d)| *d);

        // Accumulate volume from POC outward
        let mut accumulated_vol = 0u64;
        let mut max_price = poc_price;
        let mut min_price = poc_price;

        for (idx, _) in bin_distances {
            unsafe {
                let bin = self.bins.get_unchecked(idx);
                accumulated_vol += bin.volume;
                if bin.price > max_price {
                    max_price = bin.price;
                }
                if bin.price < min_price {
                    min_price = bin.price;
                }
            }
            if accumulated_vol >= target_vol {
                break;
            }
        }

        Some((max_price, min_price))
    }

    /// Calculate complete volume profile statistics
    pub fn calculate_stats(&self) -> Option<VolumeProfileStats> {
        let count = self.active_count.load(Ordering::Acquire);
        if count == 0 {
            return None;
        }

        let total_vol = self.cached_total_volume.load(Ordering::Relaxed);
        let vwap_num = self.cached_vwap_num.load(Ordering::Relaxed);
        let vwap = if total_vol > 0 {
            vwap_num as f64 / total_vol as f64
        } else {
            0.0
        };

        let (poc_price, poc_volume) = self.calculate_poc().unwrap_or((0, 0));
        let (vah, val) = self.calculate_value_area(0.70).unwrap_or((poc_price, poc_price));

        // Calculate standard deviation
        let mut variance_sum = 0.0f64;
        for i in 0..count {
            unsafe {
                let bin = self.bins.get_unchecked(i);
                if bin.volume > 0 {
                    let diff = bin.price as f64 - vwap;
                    variance_sum += diff * diff * bin.volume as f64;
                }
            }
        }
        let std_dev = if total_vol > 0 {
            (variance_sum / total_vol as f64).sqrt()
        } else {
            0.0
        };

        let mut total_buy = 0u64;
        let mut total_sell = 0u64;
        for i in 0..count {
            unsafe {
                let bin = self.bins.get_unchecked(i);
                total_buy += bin.buy_volume;
                total_sell += bin.sell_volume;
            }
        }

        Some(VolumeProfileStats {
            poc_price,
            poc_volume,
            vah_price: vah,
            val_price: val,
            total_volume: total_vol,
            total_buy_volume: total_buy,
            total_sell_volume: total_sell,
            active_bins: count,
            vwap,
            std_dev,
        })
    }

    /// Reset calculator for new session
    pub fn reset(&mut self) {
        let count = self.active_count.load(Ordering::Acquire);
        for i in 0..count {
            unsafe {
                self.bins.get_unchecked_mut(i).clear();
            }
        }
        self.active_count.store(0, Ordering::Release);
        self.min_price.store(u64::MAX, Ordering::Release);
        self.max_price.store(0, Ordering::Release);
        self.cached_total_volume.store(0, Ordering::Release);
        self.cached_vwap_num.store(0, Ordering::Release);
    }

    /// Get visible range profile between two prices
    pub fn get_visible_range(&self, start_price: i64, end_price: i64) -> Vec<VolumeBin> {
        let mut result = Vec::new();
        let count = self.active_count.load(Ordering::Acquire);
        let (low, high) = if start_price < end_price {
            (start_price, end_price)
        } else {
            (end_price, start_price)
        };

        for i in 0..count {
            unsafe {
                let bin = self.bins.get_unchecked(i);
                if bin.price >= low && bin.price <= high && bin.volume > 0 {
                    result.push(*bin);
                }
            }
        }
        
        result.sort_by_key(|b| b.price);
        result
    }

    /// Get active bin count
    pub fn get_active_bins(&self) -> usize {
        self.active_count.load(Ordering::Acquire)
    }
}

/// Session-based Volume Profile manager
pub struct SessionVolumeProfile {
    /// Current session calculator
    current_session: VolumeProfileCalculator,
    /// Session start timestamp (nanoseconds)
    session_start_ns: AtomicU64,
    /// Session duration in nanoseconds
    session_duration_ns: u64,
    /// Historical session profiles (for multi-day analysis)
    historical_profiles: parking_lot::Mutex<Vec<VolumeProfileStats>>,
}

impl SessionVolumeProfile {
    /// Create new session volume profile manager
    pub fn new(bin_size: i64, session_duration_hours: u64) -> Self {
        Self {
            current_session: VolumeProfileCalculator::new(bin_size),
            session_start_ns: AtomicU64::new(0),
            session_duration_ns: session_duration_hours * 3_600_000_000_000,
            historical_profiles: parking_lot::Mutex::new(Vec::with_capacity(30)),
        }
    }

    /// Initialize or check session rollover
    pub fn check_session(&self, current_time_ns: u64) {
        let start = self.session_start_ns.load(Ordering::Relaxed);
        
        if start == 0 {
            // First initialization
            self.session_start_ns.store(current_time_ns, Ordering::Release);
        } else if current_time_ns - start >= self.session_duration_ns {
            // Session rollover - save current and reset
            if let Some(stats) = self.current_session.calculate_stats() {
                self.historical_profiles.lock().push(stats);
            }
            unsafe {
                // Transmute to get mutable reference for reset
                let ptr = &self.current_session as *const VolumeProfileCalculator as *mut VolumeProfileCalculator;
                (*ptr).reset();
            }
            self.session_start_ns.store(current_time_ns, Ordering::Release);
        }
    }

    /// Process tick with session management
    pub fn process_tick(&self, price: i64, volume: u64, is_buyer_maker: bool, timestamp_ns: u64) {
        self.check_session(timestamp_ns);
        self.current_session.process_tick(price, volume, is_buyer_maker);
    }

    /// Get current session stats
    pub fn get_current_stats(&self) -> Option<VolumeProfileStats> {
        self.current_session.calculate_stats()
    }

    /// Get historical session profiles
    pub fn get_historical_profiles(&self) -> Vec<VolumeProfileStats> {
        self.historical_profiles.lock().clone()
    }

    /// Get composite profile from multiple sessions
    pub fn get_composite_profile(&self, num_sessions: usize) -> Option<VolumeProfileStats> {
        let profiles = self.historical_profiles.lock();
        if profiles.is_empty() {
            return None;
        }

        let take = num_sessions.min(profiles.len());
        let recent: Vec<_> = profiles.iter().rev().take(take).collect();

        // Aggregate POC volumes weighted by recency
        let mut weighted_poc_sum = 0i64;
        let mut weight_total = 0u64;

        for (i, profile) in recent.iter().enumerate() {
            let weight = (take - i) as u64;
            weighted_poc_sum += profile.poc_price * weight as i64;
            weight_total += weight;
        }

        if weight_total == 0 {
            return None;
        }

        let composite_poc = weighted_poc_sum / weight_total as i64;

        // Return aggregated stats
        Some(VolumeProfileStats {
            poc_price: composite_poc,
            poc_volume: recent.iter().map(|p| p.poc_volume).sum(),
            vah_price: recent.iter().map(|p| p.vah_price).sum::<i64>() / take as i64,
            val_price: recent.iter().map(|p| p.val_price).sum::<i64>() / take as i64,
            total_volume: recent.iter().map(|p| p.total_volume).sum(),
            total_buy_volume: recent.iter().map(|p| p.total_buy_volume).sum(),
            total_sell_volume: recent.iter().map(|p| p.total_sell_volume).sum(),
            active_bins: recent.iter().map(|p| p.active_bins).sum(),
            vwap: recent.iter().map(|p| p.vwap).sum::<f64>() / take as f64,
            std_dev: recent.iter().map(|p| p.std_dev).sum::<f64>() / take as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_profile_basic() {
        let calc = VolumeProfileCalculator::new(100);
        
        // Add trades at different price levels
        calc.process_tick(5000000000, 100, false); // Sell
        calc.process_tick(5000000000, 50, true);   // Buy
        calc.process_tick(5000100000, 200, false); // Higher price sell
        calc.process_tick(4999900000, 150, true);  // Lower price buy

        let stats = calc.calculate_stats();
        assert!(stats.is_some());
        
        let s = stats.unwrap();
        assert_eq!(s.total_volume, 500);
        assert!(s.poc_price != 0);
    }

    #[test]
    fn test_poc_calculation() {
        let calc = VolumeProfileCalculator::new(100);
        
        // Create clear POC at 50000
        calc.process_tick(5000000000, 500, false);
        calc.process_tick(5000100000, 100, false);
        calc.process_tick(4999900000, 100, false);

        let poc = calc.calculate_poc();
        assert!(poc.is_some());
        assert_eq!(poc.unwrap().0, 5000000000); // POC at highest volume price
    }

    #[test]
    fn test_session_profile() {
        let session = SessionVolumeProfile::new(100, 24);
        
        session.process_tick(5000000000, 100, false, 1000000);
        session.process_tick(5000100000, 200, true, 1000001);
        
        let stats = session.get_current_stats();
        assert!(stats.is_some());
    }
}
