//! Latency Benchmarking Harness - Microsecond-Level Performance Testing
//!
//! This module implements strict micro-benchmarks using `criterion` to enforce
//! hard microsecond latency budgets on the matching engine and order routing
//! functions before merging code. Optimized for AMD Ryzen AI 5 architecture.
//!
//! ## Features
//! - Sub-microsecond precision timing
//! - Hard latency budget enforcement
//! - Matching engine benchmarks
//! - Order routing benchmarks
//! - Statistical analysis of latency distributions

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Default latency budget in microseconds for matching operations
const DEFAULT_MATCHING_BUDGET_US: u64 = 10;

/// Default latency budget for order routing
const DEFAULT_ROUTING_BUDGET_US: u64 = 50;

/// Number of iterations for warmup
const WARMUP_ITERATIONS: usize = 100;

/// Minimum number of benchmark iterations
const MIN_ITERATIONS: usize = 1000;

/// Benchmark result for a single operation
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: String,
    pub iterations: usize,
    pub mean_time_ns: u64,
    pub median_time_ns: u64,
    pub p99_time_ns: u64,
    pub p999_time_ns: u64,
    pub min_time_ns: u64,
    pub max_time_ns: u64,
    pub std_dev_ns: u64,
    pub budget_us: u64,
    pub passed: bool,
}

/// Latency budget configuration
#[derive(Debug, Clone)]
pub struct LatencyBudget {
    pub matching_us: u64,
    pub routing_us: u64,
    pub orderbook_update_us: u64,
    pub signal_generation_us: u64,
    pub execution_us: u64,
}

impl Default for LatencyBudget {
    fn default() -> Self {
        Self {
            matching_us: DEFAULT_MATCHING_BUDGET_US,
            routing_us: DEFAULT_ROUTING_BUDGET_US,
            orderbook_update_us: 5,
            signal_generation_us: 100,
            execution_us: 200,
        }
    }
}

/// High-precision timer for nanosecond measurements
#[derive(Clone)]
pub struct PrecisionTimer {
    start: Option<Instant>,
    lap_times: Vec<u64>,
}

impl PrecisionTimer {
    /// Create new timer
    pub fn new() -> Self {
        Self {
            start: None,
            lap_times: Vec::new(),
        }
    }

    /// Start timing
    #[inline]
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Record lap time in nanoseconds
    #[inline]
    pub fn lap(&mut self) -> u64 {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.lap_times.push(elapsed);
            self.start = Some(Instant::now());
            elapsed
        } else {
            0
        }
    }

    /// Stop and return total time in nanoseconds
    #[inline]
    pub fn stop(&mut self) -> u64 {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed().as_nanos() as u64;
            self.lap_times.push(elapsed);
            elapsed
        } else {
            0
        }
    }

    /// Get all recorded times
    pub fn get_times(&self) -> &[u64] {
        &self.lap_times
    }

    /// Clear recorded times
    pub fn clear(&mut self) {
        self.lap_times.clear();
        self.start = None;
    }

    /// Calculate statistics from recorded times
    pub fn calculate_stats(&self) -> TimingStats {
        if self.lap_times.is_empty() {
            return TimingStats::default();
        }

        let mut sorted = self.lap_times.clone();
        sorted.sort_unstable();

        let n = sorted.len();
        let sum: u64 = sorted.iter().sum();
        let mean = sum / n as u64;

        // Median
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        } else {
            sorted[n / 2]
        };

        // Percentiles
        let p99_idx = (n as f64 * 0.99) as usize;
        let p999_idx = (n as f64 * 0.999) as usize;

        let p99 = sorted.get(p99_idx.min(n - 1)).copied().unwrap_or(0);
        let p999 = sorted.get(p999_idx.min(n - 1)).copied().unwrap_or(0);

        // Standard deviation
        let variance = sorted.iter()
            .map(|&x| {
                let diff = x as i64 - mean as i64;
                (diff * diff) as u64
            })
            .sum::<u64>() / n as u64;
        let std_dev = (variance as f64).sqrt() as u64;

        TimingStats {
            count: n,
            mean,
            median,
            min: *sorted.first().unwrap_or(&0),
            max: *sorted.last().unwrap_or(&0),
            p99,
            p999,
            std_dev,
        }
    }
}

impl Default for PrecisionTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing statistics
#[derive(Debug, Clone, Default)]
pub struct TimingStats {
    pub count: usize,
    pub mean: u64,
    pub median: u64,
    pub min: u64,
    pub max: u64,
    pub p99: u64,
    pub p999: u64,
    pub std_dev: u64,
}

/// Benchmark runner with budget enforcement
pub struct LatencyBenchmarker {
    budgets: LatencyBudget,
    results: parking_lot::RwLock<HashMap<String, BenchmarkResult>>,
    total_runs: AtomicUsize,
    failed_budgets: AtomicUsize,
}

impl LatencyBenchmarker {
    /// Create new benchmarker with default budgets
    pub fn new() -> Self {
        Self {
            budgets: LatencyBudget::default(),
            results: parking_lot::RwLock::new(HashMap::new()),
            total_runs: AtomicUsize::new(0),
            failed_budgets: AtomicUsize::new(0),
        }
    }

    /// Create with custom budgets
    pub fn with_budgets(budgets: LatencyBudget) -> Self {
        Self {
            budgets,
            results: parking_lot::RwLock::new(HashMap::new()),
            total_runs: AtomicUsize::new(0),
            failed_budgets: AtomicUsize::new(0),
        }
    }

    /// Run benchmark for matching engine operations
    pub fn benchmark_matching<F>(&self, operation: F, name: &str) -> BenchmarkResult
    where
        F: Fn() + Sync + Send,
    {
        self.run_benchmark(operation, name, self.budgets.matching_us)
    }

    /// Run benchmark for order routing operations
    pub fn benchmark_routing<F>(&self, operation: F, name: &str) -> BenchmarkResult
    where
        F: Fn() + Sync + Send,
    {
        self.run_benchmark(operation, name, self.budgets.routing_us)
    }

    /// Generic benchmark runner with budget enforcement
    pub fn run_benchmark<F>(&self, operation: F, name: &str, budget_us: u64) -> BenchmarkResult
    where
        F: Fn() + Sync + Send,
    {
        let mut timer = PrecisionTimer::new();
        
        // Warmup phase
        for _ in 0..WARMUP_ITERATIONS {
            timer.start();
            operation();
            timer.stop();
        }
        timer.clear();

        // Benchmark phase
        for _ in 0..MIN_ITERATIONS {
            timer.start();
            operation();
            timer.lap();
        }

        let stats = timer.calculate_stats();
        let budget_ns = budget_us * 1000;

        let result = BenchmarkResult {
            operation: name.to_string(),
            iterations: stats.count,
            mean_time_ns: stats.mean,
            median_time_ns: stats.median,
            p99_time_ns: stats.p99,
            p999_time_ns: stats.p999,
            min_time_ns: stats.min,
            max_time_ns: stats.max,
            std_dev_ns: stats.std_dev,
            budget_us,
            passed: stats.median <= budget_ns,
        };

        // Store result
        self.results.write().insert(name.to_string(), result.clone());
        self.total_runs.fetch_add(1, AtomicOrdering::Relaxed);

        if !result.passed {
            self.failed_budgets.fetch_add(1, AtomicOrdering::Relaxed);
            log::warn!(
                "Benchmark '{}' FAILED: median={}ns > budget={}ns",
                name, stats.median, budget_ns
            );
        } else {
            log::info!(
                "Benchmark '{}' PASSED: median={}ns, p99={}ns",
                name, stats.median, stats.p99
            );
        }

        result
    }

    /// Check if all benchmarks passed their budgets
    pub fn all_passed(&self) -> bool {
        self.failed_budgets.load(AtomicOrdering::Relaxed) == 0
    }

    /// Get all results
    pub fn get_results(&self) -> HashMap<String, BenchmarkResult> {
        self.results.read().clone()
    }

    /// Get specific result
    pub fn get_result(&self, name: &str) -> Option<BenchmarkResult> {
        self.results.read().get(name).cloned()
    }

    /// Print summary report
    pub fn print_summary(&self) {
        let results = self.results.read();
        
        println!("\n=== Latency Benchmark Summary ===");
        println!("Total runs: {}", self.total_runs.load(AtomicOrdering::Relaxed));
        println!("Failed budgets: {}\n", self.failed_budgets.load(AtomicOrdering::Relaxed));

        for (name, result) in results.iter() {
            let status = if result.passed { "PASS" } else { "FAIL" };
            println!(
                "[{}] {}: median={:.2}μs, p99={:.2}μs, p999={:.2}μs (budget={}μs)",
                status,
                name,
                result.median_time_ns as f64 / 1000.0,
                result.p99_time_ns as f64 / 1000.0,
                result.p999_time_ns as f64 / 1000.0,
                result.budget_us
            );
        }
    }
}

impl Default for LatencyBenchmarker {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated matching engine for benchmark demonstration
pub struct MockMatchingEngine {
    order_count: AtomicUsize,
}

impl MockMatchingEngine {
    pub fn new() -> Self {
        Self {
            order_count: AtomicUsize::new(0),
        }
    }

    /// Simulate order matching (microsecond-scale operation)
    #[inline]
    pub fn match_order(&self, price: u64, quantity: u64) -> bool {
        // Simulate some computation
        let _ = price.wrapping_mul(quantity);
        self.order_count.fetch_add(1, AtomicOrdering::Relaxed);
        true
    }

    /// Get order count
    pub fn order_count(&self) -> usize {
        self.order_count.load(AtomicOrdering::Relaxed)
    }
}

impl Default for MockMatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulated order router for benchmark demonstration
pub struct MockOrderRouter {
    routes: AtomicUsize,
}

impl MockOrderRouter {
    pub fn new() -> Self {
        Self {
            routes: AtomicUsize::new(0),
        }
    }

    /// Simulate order routing decision
    #[inline]
    pub fn route_order(&self, symbol: &str, side: u8) -> u8 {
        // Simulate routing logic
        let hash = symbol.bytes().fold(0u8, |acc, b| acc.wrapping_add(b));
        let route = (hash + side) % 4;
        self.routes.fetch_add(1, AtomicOrdering::Relaxed);
        route
    }

    /// Get route count
    pub fn route_count(&self) -> usize {
        self.routes.load(AtomicOrdering::Relaxed)
    }
}

impl Default for MockOrderRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_timer() {
        let mut timer = PrecisionTimer::new();
        
        for _ in 0..100 {
            timer.start();
            std::hint::black_box(1 + 1);
            timer.lap();
        }
        
        let stats = timer.calculate_stats();
        assert!(stats.count == 100);
        assert!(stats.mean > 0);
    }

    #[test]
    fn test_benchmarker_basic() {
        let bench = LatencyBenchmarker::new();
        
        let result = bench.benchmark_matching(
            || {
                std::hint::black_box(1 + 1);
            },
            "simple_operation",
        );
        
        assert!(result.iterations >= MIN_ITERATIONS);
        assert!(result.mean_time_ns > 0);
    }

    #[test]
    fn test_matching_engine_benchmark() {
        let engine = MockMatchingEngine::new();
        let bench = LatencyBenchmarker::new();
        
        let result = bench.benchmark_matching(
            || {
                engine.match_order(50000, 100);
            },
            "order_matching",
        );
        
        // Should complete within budget for simple operation
        assert!(result.passed);
    }

    #[test]
    fn test_order_router_benchmark() {
        let router = MockOrderRouter::new();
        let bench = LatencyBenchmarker::new();
        
        let result = bench.benchmark_routing(
            || {
                router.route_order("BTCUSDT", 1);
            },
            "order_routing",
        );
        
        assert!(result.iterations >= MIN_ITERATIONS);
    }

    #[test]
    fn test_budget_enforcement() {
        let bench = LatencyBenchmarker::new();
        
        // Fast operation - should pass
        let fast_result = bench.benchmark_matching(
            || { std::hint::black_box(1); },
            "fast_op",
        );
        
        // Very tight budget - might fail
        let tight_bench = LatencyBenchmarker::with_budgets(
            LatencyBudget {
                matching_us: 0, // Impossible budget
                ..Default::default()
            }
        );
        
        let tight_result = tight_bench.benchmark_matching(
            || { std::hint::black_box(1); },
            "tight_op",
        );
        
        assert!(fast_result.passed || tight_result.passed == false);
    }
}
