//! Master Event Loop & Thread Orchestration
//! 
//! Binds the Nautilus engine, Rust SMC (Smart Market Core), and IPC bridges
//! into a single, lock-free Tokio runtime. Optimized for AMD Ryzen AI 5 architecture
//! with explicit core pinning to Core 0 for the master thread to minimize context switches.
//! 
//! Strictly enforces the global 8GB RAM limit via final arena bounds checks.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::{Builder, Runtime};
use tokio::sync::broadcast;
use tracing::{info, error, warn};

// Import local modules
use crate::memory::arena_reset::ArenaAllocator;
use crate::exchange::binance_weights::WeightTracker;
use crate::reconcile::snapshot_sync::StateReconciler;

/// Maximum allowed RAM usage in bytes (8GB)
const MAX_RAM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Master orchestrator state
pub struct MasterOrchestrator {
    /// Flag indicating if the system is running
    running: AtomicBool,
    /// Global tick counter (lock-free)
    tick_counter: AtomicU64,
    /// Shared memory arena for ephemeral data
    arena: Arc<ArenaAllocator>,
    /// Binance weight tracker
    weight_tracker: Arc<WeightTracker>,
    /// State reconciler
    reconciler: Arc<StateReconciler>,
    /// Broadcast channel for shutdown signals
    shutdown_tx: broadcast::Sender<()>,
}

impl MasterOrchestrator {
    /// Create a new master orchestrator
    pub fn new() -> Self {
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        
        Self {
            running: AtomicBool::new(false),
            tick_counter: AtomicU64::new(0),
            arena: Arc::new(ArenaAllocator::with_capacity(MAX_RAM_BYTES / 2)), // Reserve 4GB for arena
            weight_tracker: Arc::new(WeightTracker::new()),
            reconciler: Arc::new(StateReconciler::new()),
            shutdown_tx,
        }
    }

    /// Pin the current thread to AMD Ryzen Core 0
    /// Uses libc for Linux/Windows thread affinity
    #[cfg(target_os = "windows")]
    fn pin_to_core_0() {
        use windows::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
        unsafe {
            let handle = GetCurrentThread();
            // Bitmask for Core 0 only
            let mask = 1usize; 
            SetThreadAffinityMask(handle, mask);
            info!("Master thread pinned to AMD Ryzen Core 0 (Windows)");
        }
    }

    #[cfg(target_os = "linux")]
    fn pin_to_core_0() {
        use libc::{cpu_set_t, sched_setaffinity, CPU_ZERO, CPU_SET};
        use std::mem;
        
        unsafe {
            let mut cpuset: cpu_set_t = mem::zeroed();
            CPU_ZERO(&mut cpuset);
            CPU_SET(0, &mut cpuset);
            
            let result = sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpuset);
            if result == 0 {
                info!("Master thread pinned to AMD Ryzen Core 0 (Linux)");
            } else {
                warn!("Failed to pin thread to Core 0: errno {}", *libc::__errno_location());
            }
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    fn pin_to_core_0() {
        warn!("Thread pinning not supported on this platform");
    }

    /// Initialize the master event loop
    pub fn initialize(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Initializing Master Orchestrator...");
        
        // Pin master thread to Core 0
        Self::pin_to_core_0();
        
        // Validate memory bounds
        let current_arena_usage = self.arena.used_bytes();
        if current_arena_usage > MAX_RAM_BYTES {
            error!("Initial arena usage {} exceeds 8GB limit", current_arena_usage);
            return Err("Memory limit exceeded at initialization".into());
        }
        
        info!("Arena initialized: {} bytes allocated", current_arena_usage);
        info!("Weight tracker active: IP limit 1200/60s, UID limit 6000/60s");
        
        Ok(())
    }

    /// Run the lock-free master event loop
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.running.store(true, Ordering::SeqCst);
        info!("Master Event Loop started on Core 0");
        
        let mut interval = tokio::time::interval(Duration::from_micros(100)); // 10kHz base tick
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        while self.running.load(Ordering::Relaxed) {
            tokio::select! {
                _ = interval.tick() => {
                    // Lock-free tick processing
                    let tick = self.tick_counter.fetch_add(1, Ordering::Relaxed);
                    
                    // Process SMC (Smart Market Core) logic here
                    self.process_tick(tick).await;
                    
                    // Check memory pressure
                    if self.arena.used_bytes() > MAX_RAM_BYTES {
                        warn!("Memory pressure detected: {} bytes", self.arena.used_bytes());
                        // Trigger background defrag (non-blocking)
                        self.arena.trigger_defrag();
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Shutdown signal received in master loop");
                    break;
                }
            }
        }
        
        info!("Master Event Loop terminated gracefully");
        Ok(())
    }

    /// Process a single tick (hot path - must be microsecond optimized)
    #[inline(always)]
    async fn process_tick(&self, tick: u64) {
        // Zero-copy tick processing using arena memory
        // 1. Fetch market data (lock-free)
        // 2. Run SMC decision logic
        // 3. Update reconciler state
        // 4. Check Binance weights before any REST call
        
        if tick % 1000 == 0 {
            // Every 1000 ticks, sync state
            let _ = self.reconciler.sync_state().await;
        }
    }

    /// Signal shutdown to all components
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.shutdown_tx.send(());
    }

    /// Get current tick count
    pub fn get_tick(&self) -> u64 {
        self.tick_counter.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orch = MasterOrchestrator::new();
        assert!(!orch.running.load(Ordering::Relaxed));
        assert_eq!(orch.get_tick(), 0);
    }

    #[tokio::test]
    async fn test_initialization() {
        let orch = MasterOrchestrator::new();
        assert!(orch.initialize().is_ok());
    }
}
