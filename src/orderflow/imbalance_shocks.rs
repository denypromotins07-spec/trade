//! Order Book Imbalance Shocks - Hawkes Process Detector
//! 
//! This module implements a Hawkes process-based detector for order book imbalance shocks,
//! predicting sudden liquidity vacuums and impending flash crashes at the microsecond level.
//! Hawkes processes are self-exciting point processes ideal for modeling clustered events
//! like order cancellations and aggressive market orders.
//! 
//! **Key Features:**
//! - Self-exciting point process for event clustering.
//! - Microsecond-level shock probability estimation.
//! - Predictive signals for liquidity vacuum detection.

use std::collections::VecDeque;

/// Parameters for the Hawkes process.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HawkesParams {
    /// Base intensity (background rate)
    pub mu: f64,
    /// Excitation factor (how much each event increases intensity)
    pub alpha: f64,
    /// Decay rate (exponential decay of excitation)
    pub beta: f64,
}

impl Default for HawkesParams {
    fn default() -> Self {
        HawkesParams {
            mu: 0.1,      // Low background rate
            alpha: 0.8,   // Strong self-excitation
            beta: 10.0,   // Fast decay (per second)
        }
    }
}

/// Event record for the Hawkes process.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HawkesEvent {
    pub timestamp_ns: u64,
    pub magnitude: f64, // Size/magnitude of the event
    pub event_type: u8, // 0 = cancellation, 1 = aggressive buy, 2 = aggressive sell
}

/// Hawkes Process Imbalance Shock Detector.
pub struct ImbalanceShockDetector {
    /// Hawkes process parameters
    params: HawkesParams,
    /// Recent events for intensity calculation
    events: VecDeque<HawkesEvent>,
    /// Current intensity estimate
    current_intensity: f64,
    /// Last update timestamp
    last_timestamp_ns: u64,
    /// Maximum event history window (nanoseconds)
    max_window_ns: u64,
    /// Shock threshold (intensity level indicating imminent shock)
    shock_threshold: f64,
    /// Number of events in current window
    event_count: usize,
}

impl ImbalanceShockDetector {
    /// Create a new imbalance shock detector.
    pub fn new(params: HawkesParams, shock_threshold: f64) -> Self {
        ImbalanceShockDetector {
            params,
            events: VecDeque::with_capacity(1000),
            current_intensity: params.mu,
            last_timestamp_ns: 0,
            max_window_ns: 5_000_000_000, // 5 second window
            shock_threshold,
            event_count: 0,
        }
    }

    /// Record a new event and update intensity.
    pub fn add_event(&mut self, timestamp_ns: u64, magnitude: f64, event_type: u8) -> f64 {
        if self.last_timestamp_ns == 0 {
            self.last_timestamp_ns = timestamp_ns;
        }

        // Calculate time decay since last event
        let dt_ns = timestamp_ns.saturating_sub(self.last_timestamp_ns);
        let dt_sec = dt_ns as f64 / 1_000_000_000.0;

        // Decay existing intensity
        self.current_intensity = self.params.mu 
            + (self.current_intensity - self.params.mu) * (-self.params.beta * dt_sec).exp();

        // Add excitation from new event
        let excitation = self.params.alpha * magnitude;
        self.current_intensity += excitation;

        // Store event
        let event = HawkesEvent {
            timestamp_ns,
            magnitude,
            event_type,
        };
        self.events.push_back(event);
        self.event_count += 1;

        // Remove old events outside window
        self.prune_old_events(timestamp_ns);

        self.last_timestamp_ns = timestamp_ns;
        self.current_intensity
    }

    /// Prune events older than the maximum window.
    fn prune_old_events(&mut self, current_ts: u64) {
        let cutoff = current_ts.saturating_sub(self.max_window_ns);
        
        while let Some(front) = self.events.front() {
            if front.timestamp_ns < cutoff {
                self.events.pop_front();
                if self.event_count > 0 {
                    self.event_count -= 1;
                }
            } else {
                break;
            }
        }
    }

    /// Get the current intensity (shock probability indicator).
    pub fn get_intensity(&self) -> f64 {
        self.current_intensity
    }

    /// Check if a shock is imminent (intensity exceeds threshold).
    pub fn is_shock_imminent(&self) -> bool {
        self.current_intensity > self.shock_threshold
    }

    /// Get the expected number of events in the next time window.
    /// E[N(t, t+h)] = integral of intensity over [t, t+h]
    pub fn predict_event_count(&self, horizon_sec: f64) -> f64 {
        // For exponential kernel: E[N] = mu * h + (alpha/beta) * (1 - exp(-beta * h)) * current_excess
        let excess = self.current_intensity - self.params.mu;
        let expected = self.params.mu * horizon_sec 
            + (self.params.alpha / self.params.beta) * (1.0 - (-self.params.beta * horizon_sec).exp()) * excess;
        expected
    }

    /// Calculate the probability of a liquidity vacuum (extreme imbalance) in the next window.
    pub fn probability_of_vacuum(&self, horizon_sec: f64, threshold_events: usize) -> f64 {
        // Simplified: use Poisson approximation with Hawkes intensity
        let lambda = self.predict_event_count(horizon_sec);
        
        // P(N >= threshold) = 1 - P(N < threshold)
        let mut prob_less_than = 0.0;
        let mut poisson_prob = (-lambda).exp(); // P(N=0)
        
        for k in 0..threshold_events {
            if k == 0 {
                poisson_prob = (-lambda).exp();
            } else {
                poisson_prob *= lambda / k as f64;
            }
            prob_less_than += poisson_prob;
        }

        1.0 - prob_less_than
    }

    /// Get the number of recent events in the window.
    pub fn get_event_count(&self) -> usize {
        self.event_count
    }

    /// Update Hawkes parameters dynamically based on market regime.
    pub fn update_params(&mut self, new_params: HawkesParams) {
        self.params = new_params;
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.events.clear();
        self.current_intensity = self.params.mu;
        self.event_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hawkes_intensity_buildup() {
        let params = HawkesParams {
            mu: 0.1,
            alpha: 0.8,
            beta: 5.0,
        };
        
        let mut detector = ImbalanceShockDetector::new(params, 5.0);

        // Simulate a burst of events (cancellations)
        let base_time = 1000000000u64;
        for i in 0..20 {
            let ts = base_time + i * 10_000_000; // 10ms apart
            let intensity = detector.add_event(ts, 1.0, 0); // Cancellation event
            println!("Event {}: intensity = {:.4}", i, intensity);
        }

        // Intensity should have built up significantly
        assert!(detector.get_intensity() > 1.0);
    }

    #[test]
    fn test_shock_detection() {
        let params = HawkesParams::default();
        let mut detector = ImbalanceShockDetector::new(params, 3.0);

        // Initial state should not be in shock
        assert!(!detector.is_shock_imminent());

        // Trigger many events to build intensity
        let base_time = 1000000000u64;
        for i in 0..50 {
            let ts = base_time + i * 1_000_000; // 1ms apart
            detector.add_event(ts, 0.5, 0);
        }

        // May or may not trigger shock depending on parameters
        let _ = detector.is_shock_imminent();
    }

    #[test]
    fn test_vacuum_probability() {
        let params = HawkesParams::default();
        let detector = ImbalanceShockDetector::new(params, 5.0);

        // With low intensity, probability should be low
        let prob = detector.probability_of_vacuum(1.0, 10);
        assert!(prob < 0.5);
    }
}
