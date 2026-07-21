//! Flamegraph Profiling: Ultra-Low-Overhead Sampling
//! 
//! Generates continuous, ultra-low-overhead sampling for live flamegraphs
//! of the Rust hot path, identifying micro-bottlenecks without impacting
//! execution latency. Uses statistical sampling to minimize overhead.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Stack frame representation for profiling
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// Source file
    pub file: String,
    /// Line number
    pub line: u32,
}

impl StackFrame {
    /// Create a new stack frame
    pub fn new(function: &str, file: &str, line: u32) -> Self {
        Self {
            function: function.to_string(),
            file: file.to_string(),
            line,
        }
    }
}

/// Sampled stack trace
#[derive(Debug, Clone)]
pub struct StackSample {
    /// Stack frames (bottom to top)
    pub frames: Vec<StackFrame>,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Thread ID
    pub thread_id: u64,
}

/// Aggregated flamegraph node
#[derive(Debug, Clone)]
pub struct FlameGraphNode {
    /// Function identifier
    pub function: String,
    /// Total samples in this node
    pub total_samples: u64,
    /// Self samples (excluding children)
    pub self_samples: u64,
    /// Child nodes
    pub children: HashMap<String, FlameGraphNode>,
}

impl FlameGraphNode {
    /// Create a new root node
    pub fn new(function: &str) -> Self {
        Self {
            function: function.to_string(),
            total_samples: 0,
            self_samples: 0,
            children: HashMap::new(),
        }
    }

    /// Add a sample to the tree
    pub fn add_sample(&mut self, frames: &[StackFrame], idx: usize) {
        self.total_samples += 1;
        
        if idx >= frames.len() {
            self.self_samples += 1;
            return;
        }
        
        let frame = &frames[idx];
        let child_key = format!("{}:{}:{}", frame.function, frame.file, frame.line);
        
        let child = self.children.entry(child_key).or_insert_with(|| {
            FlameGraphNode::new(&frame.function)
        });
        
        child.add_sample(frames, idx + 1);
    }

    /// Get total sample count including all descendants
    pub fn get_total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Get percentage of total samples
    pub fn percentage(&self, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            (self.total_samples as f64 / total as f64) * 100.0
        }
    }
}

/// Low-overhead profiler using statistical sampling
pub struct FlamegraphProfiler {
    /// Enable/disable profiling
    enabled: AtomicBool,
    /// Sample counter
    sample_count: AtomicU64,
    /// Dropped samples (when queue full)
    dropped_samples: AtomicU64,
    /// Root of the flamegraph tree
    root: Arc<std::sync::RwLock<FlameGraphNode>>,
    /// Sampling interval in microseconds
    sampling_interval_us: AtomicU64,
    /// Last sample timestamp
    last_sample_ns: AtomicU64,
    /// Start time
    start_time: AtomicU64,
}

impl FlamegraphProfiler {
    /// Create a new profiler with specified sampling interval
    pub fn new(sampling_interval_us: u64) -> Self {
        Self {
            enabled: AtomicBool::new(false),
            sample_count: AtomicU64::new(0),
            dropped_samples: AtomicU64::new(0),
            root: Arc::new(std::sync::RwLock::new(FlameGraphNode::new("root"))),
            sampling_interval_us: AtomicU64::new(sampling_interval_us),
            last_sample_ns: AtomicU64::new(0),
            start_time: AtomicU64::new(0),
        }
    }

    /// Start profiling
    #[inline]
    pub fn start(&self) {
        self.enabled.store(true, Ordering::Release);
        self.start_time.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            Ordering::Release,
        );
    }

    /// Stop profiling
    #[inline]
    pub fn stop(&self) {
        self.enabled.store(false, Ordering::Release);
    }

    /// Check if profiling is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Record a sample (called by sampling thread)
    /// This should be called from a separate low-priority thread
    pub fn record_sample(&self, sample: StackSample) {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }

        // Rate limiting check
        let interval_ns = self.sampling_interval_us.load(Ordering::Acquire) * 1000;
        let now = sample.timestamp_ns;
        let last = self.last_sample_ns.load(Ordering::Relaxed);

        if now - last < interval_ns {
            self.dropped_samples.fetch_add(1, Ordering::Relaxed);
            return;
        }

        self.last_sample_ns.store(now, Ordering::Relaxed);

        // Add to flamegraph tree
        if let Ok(mut root) = self.root.write() {
            root.add_sample(&sample.frames, 0);
        }

        self.sample_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Simulate recording a sample with given frames
    pub fn record_sample_simulated(&self, frames: Vec<StackFrame>) {
        let sample = StackSample {
            frames,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            thread_id: std::thread::current().id().as_u64(),
        };
        self.record_sample(sample);
    }

    /// Get current sample count
    #[inline]
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Acquire)
    }

    /// Get dropped sample count
    #[inline]
    pub fn dropped_count(&self) -> u64 {
        self.dropped_samples.load(Ordering::Acquire)
    }

    /// Generate flamegraph output in collapsed format
    /// Format: "func1;func2;func3 count"
    pub fn generate_collapsed(&self) -> String {
        let mut output = String::new();
        let root = self.root.read().unwrap();
        
        self.generate_collapsed_recursive(&root, Vec::new(), &mut output);
        
        output
    }

    fn generate_collapsed_recursive(
        &self,
        node: &FlameGraphNode,
        path: Vec<&str>,
        output: &mut String,
    ) {
        if node.total_samples == 0 {
            return;
        }

        let mut current_path = path;
        current_path.push(&node.function);

        if node.children.is_empty() || node.self_samples > 0 {
            // Leaf or has self time
            let path_str = current_path.join(";");
            output.push_str(&path_str);
            output.push(' ');
            output.push_str(&node.self_samples.to_string());
            output.push('\n');
        }

        for child in node.children.values() {
            self.generate_collapsed_recursive(child, current_path.clone(), output);
        }
    }

    /// Export flamegraph data as JSON for visualization
    pub fn export_json(&self) -> serde_json::Value {
        let root = self.root.read().unwrap();
        self.node_to_json(&root)
    }

    fn node_to_json(&self, node: &FlameGraphNode) -> serde_json::Value {
        use serde_json::json;
        
        let children: Vec<_> = node.children.values()
            .map(|c| self.node_to_json(c))
            .collect();
        
        json!({
            "name": node.function,
            "value": node.total_samples,
            "self_value": node.self_samples,
            "children": children
        })
    }

    /// Reset profiler state
    pub fn reset(&self) {
        *self.root.write().unwrap() = FlameGraphNode::new("root");
        self.sample_count.store(0, Ordering::Relaxed);
        self.dropped_samples.store(0, Ordering::Relaxed);
        self.last_sample_ns.store(0, Ordering::Relaxed);
        self.start_time.store(0, Ordering::Relaxed);
    }

    /// Get profiling duration in seconds
    pub fn duration_secs(&self) -> f64 {
        let start = self.start_time.load(Ordering::Acquire);
        if start == 0 {
            return 0.0;
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        (now - start) as f64 / 1_000_000_000.0
    }

    /// Get samples per second rate
    pub fn samples_per_second(&self) -> f64 {
        let duration = self.duration_secs();
        let count = self.sample_count.load(Ordering::Acquire);
        
        if duration == 0.0 {
            return 0.0;
        }
        
        count as f64 / duration
    }
}

/// RAII guard for scoped profiling (manual instrumentation)
pub struct ProfileScope<'a> {
    profiler: &'a FlamegraphProfiler,
    function: &'static str,
    file: &'static str,
    line: u32,
    start: Instant,
}

impl<'a> ProfileScope<'a> {
    /// Create a new profile scope
    pub fn new(
        profiler: &'a FlamegraphProfiler,
        function: &'static str,
        file: &'static str,
        line: u32,
    ) -> Self {
        Self {
            profiler,
            function,
            file,
            line,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ProfileScope<'a> {
    fn drop(&mut self) {
        // Record sample on scope exit
        let frame = StackFrame::new(self.function, self.file, self.line);
        self.profiler.record_sample_simulated(vec![frame]);
    }
}

/// Macro for easy scoped profiling
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr) => {
        let _scope = $crate::telemetry::flamegraph::ProfileScope::new(
            $profiler,
            concat!(module_path!, "::", stringify!(fn)),
            file!(),
            line!(),
        );
    };
}

/// Background sampler thread that periodically captures stack traces
pub fn start_background_sampler(profiler: Arc<FlamegraphProfiler>, interval_us: u64) {
    std::thread::spawn(move || {
        profiler.start();
        
        loop {
            if profiler.is_enabled() {
                // In production, this would use backtrace crate or perf_events
                // For now, simulate with a dummy sample
                #[cfg(debug_assertions)]
                {
                    let sample = StackSample {
                        frames: vec![
                            StackFrame::new("background_sampler", "flamegraph.rs", 100),
                            StackFrame::new("main_loop", "main.rs", 50),
                        ],
                        timestamp_ns: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos() as u64,
                        thread_id: std::thread::current().id().as_u64(),
                    };
                    profiler.record_sample(sample);
                }
            }
            
            std::thread::sleep(Duration::from_micros(interval_us));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let profiler = FlamegraphProfiler::new(1000); // 1ms sampling
        
        assert!(!profiler.is_enabled());
        
        profiler.start();
        assert!(profiler.is_enabled());
        
        // Record some samples
        profiler.record_sample_simulated(vec![
            StackFrame::new("test_func", "test.rs", 10),
        ]);
        
        assert_eq!(profiler.sample_count(), 1);
        
        profiler.stop();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_flamegraph_generation() {
        let profiler = FlamegraphProfiler::new(1000);
        profiler.start();
        
        // Record samples with common prefix
        for _ in 0..10 {
            profiler.record_sample_simulated(vec![
                StackFrame::new("main", "main.rs", 1),
                StackFrame::new("process", "process.rs", 10),
                StackFrame::new("compute", "compute.rs", 20),
            ]);
        }
        
        for _ in 0..5 {
            profiler.record_sample_simulated(vec![
                StackFrame::new("main", "main.rs", 1),
                StackFrame::new("process", "process.rs", 10),
                StackFrame::new("io", "io.rs", 30),
            ]);
        }
        
        let collapsed = profiler.generate_collapsed();
        assert!(collapsed.contains("main"));
        assert!(collapsed.contains("process"));
        
        profiler.stop();
    }

    #[test]
    fn test_profile_scope_raii() {
        let profiler = FlamegraphProfiler::new(1000);
        profiler.start();
        
        {
            let _scope = ProfileScope::new(&profiler, "test_function", "test.rs", 42);
            // Do some work
            std::thread::sleep(Duration::from_millis(1));
        }
        
        // Sample should be recorded when scope drops
        assert!(profiler.sample_count() >= 1);
        
        profiler.stop();
    }
}
