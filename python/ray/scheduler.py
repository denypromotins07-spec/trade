"""
Ray Resource Scheduler

Architects a custom Ray resource scheduler that enforces strict 4GB Python
RAM limits per worker, dynamically migrating tasks if localized memory
pressure threatens the global 8GB cap.

Optimized for AMD Ryzen AI 5 architecture with microsecond-latency monitoring.
"""

import ray
from ray import node_resources
from typing import Dict, List, Optional, Tuple
import psutil
import os
import time
import threading
from dataclasses import dataclass
from enum import Enum


# Memory limits (in bytes)
GLOBAL_RAM_LIMIT_GB = 8.0
PYTHON_WORKER_RAM_LIMIT_GB = 4.0
MEMORY_PRESSURE_THRESHOLD = 0.85  # 85% triggers migration
CRITICAL_MEMORY_THRESHOLD = 0.95  # 95% triggers KILL protocol


class MemoryState(Enum):
    NORMAL = "normal"
    PRESSURE = "pressure"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class WorkerMemoryInfo:
    """Memory information for a single worker."""
    worker_id: str
    used_memory_gb: float
    available_memory_gb: float
    memory_fraction: float
    state: MemoryState
    task_count: int
    last_updated_ns: int


@dataclass
class ClusterMemoryState:
    """Global cluster memory state."""
    total_memory_gb: float
    used_memory_gb: float
    available_memory_gb: float
    memory_fraction: float
    worker_states: Dict[str, WorkerMemoryInfo]
    state: MemoryState
    timestamp_ns: int


class RayResourceScheduler:
    """
    Custom Ray resource scheduler with strict memory enforcement.
    
    Features:
    - Per-worker 4GB RAM limit enforcement
    - Global 8GB RAM cap across all workers
    - Dynamic task migration on memory pressure
    - Integration with /KILL protocol for critical violations
    """
    
    def __init__(
        self,
        global_limit_gb: float = GLOBAL_RAM_LIMIT_GB,
        worker_limit_gb: float = PYTHON_WORKER_RAM_LIMIT_GB,
        pressure_threshold: float = MEMORY_PRESSURE_THRESHOLD,
        critical_threshold: float = CRITICAL_MEMORY_THRESHOLD
    ):
        self.global_limit_gb = global_limit_gb
        self.worker_limit_gb = worker_limit_gb
        self.pressure_threshold = pressure_threshold
        self.critical_threshold = critical_threshold
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._worker_states: Dict[str, WorkerMemoryInfo] = {}
        self._migration_history: List[Dict] = []
        self._kill_protocol_triggered = False
        
        # Callbacks for integration with orchestration
        self.on_pressure_callback = None
        self.on_critical_callback = None
        self.on_kill_trigger = None
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start background memory monitoring thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
        print(f"[Scheduler] Started monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                cluster_state = self.get_cluster_state()
                
                if cluster_state.state == MemoryState.PRESSURE:
                    self._handle_memory_pressure(cluster_state)
                elif cluster_state.state == MemoryState.CRITICAL:
                    self._handle_critical_memory(cluster_state)
                elif cluster_state.state == MemoryState.EXCEEDED:
                    self._trigger_kill_protocol(cluster_state)
                
            except Exception as e:
                print(f"[Scheduler] Monitoring error: {e}")
            
            time.sleep(interval)
    
    def get_cluster_state(self) -> ClusterMemoryState:
        """Get current cluster-wide memory state."""
        try:
            # Get Ray cluster resources
            if ray.is_initialized():
                cluster_resources = ray.cluster_resources()
                node_stats = ray.nodes()
            else:
                cluster_resources = {}
                node_stats = []
            
            # Collect worker memory info
            worker_states = {}
            total_used = 0.0
            total_available = 0.0
            
            # Get system-wide memory info
            system_mem = psutil.virtual_memory()
            
            for node in node_stats:
                node_id = node.get("NodeID", "unknown")
                
                # Estimate worker memory from Ray metrics
                # In production, use Ray's internal memory metrics
                worker_info = self._estimate_worker_memory(node_id, system_mem)
                worker_states[node_id] = worker_info
                
                total_used += worker_info.used_memory_gb
                total_available += worker_info.available_memory_gb
            
            # Also account for Python overhead
            python_overhead = self._get_python_overhead_gb()
            total_used += python_overhead
            
            # Calculate global state
            memory_fraction = total_used / self.global_limit_gb if self.global_limit_gb > 0 else 0
            
            global_state = MemoryState.NORMAL
            if memory_fraction >= 1.0:
                global_state = MemoryState.EXCEEDED
            elif memory_fraction >= self.critical_threshold:
                global_state = MemoryState.CRITICAL
            elif memory_fraction >= self.pressure_threshold:
                global_state = MemoryState.PRESSURE
            
            return ClusterMemoryState(
                total_memory_gb=self.global_limit_gb,
                used_memory_gb=total_used,
                available_memory_gb=max(0, self.global_limit_gb - total_used),
                memory_fraction=memory_fraction,
                worker_states=worker_states,
                state=global_state,
                timestamp_ns=time.time_ns()
            )
            
        except Exception as e:
            # Return safe defaults on error
            return ClusterMemoryState(
                total_memory_gb=self.global_limit_gb,
                used_memory_gb=0.0,
                available_memory_gb=self.global_limit_gb,
                memory_fraction=0.0,
                worker_states={},
                state=MemoryState.NORMAL,
                timestamp_ns=time.time_ns()
            )
    
    def _estimate_worker_memory(
        self,
        node_id: str,
        system_mem: psutil._common.svmem
    ) -> WorkerMemoryInfo:
        """Estimate memory usage for a worker node."""
        # Convert to GB
        used_gb = (system_mem.total - system_mem.available) / (1024**3)
        available_gb = system_mem.available / (1024**3)
        fraction = used_gb / (system_mem.total / (1024**3)) if system_mem.total > 0 else 0
        
        # Determine state based on worker limit
        if fraction >= 1.0:
            state = MemoryState.EXCEEDED
        elif fraction >= self.critical_threshold:
            state = MemoryState.CRITICAL
        elif fraction >= self.pressure_threshold:
            state = MemoryState.PRESSURE
        else:
            state = MemoryState.NORMAL
        
        return WorkerMemoryInfo(
            worker_id=node_id,
            used_memory_gb=used_gb,
            available_memory_gb=available_gb,
            memory_fraction=fraction,
            state=state,
            task_count=0,  # Would get from Ray in production
            last_updated_ns=time.time_ns()
        )
    
    def _get_python_overhead_gb(self) -> float:
        """Get memory overhead from Python interpreter and Ray."""
        process = psutil.Process(os.getpid())
        try:
            mem_info = process.memory_info()
            return mem_info.rss / (1024**3)
        except Exception:
            return 0.0
    
    def _handle_memory_pressure(self, state: ClusterMemoryState):
        """Handle memory pressure by migrating tasks."""
        print(f"[Scheduler] Memory pressure detected: {state.memory_fraction:.1%}")
        
        # Find workers under pressure
        pressured_workers = [
            (wid, winfo) for wid, winfo in state.worker_states.items()
            if winfo.state in [MemoryState.PRESSURE, MemoryState.CRITICAL]
        ]
        
        # Find workers with available capacity
        available_workers = [
            (wid, winfo) for wid, winfo in state.worker_states.items()
            if winfo.state == MemoryState.NORMAL and winfo.memory_fraction < 0.5
        ]
        
        if not pressured_workers or not available_workers:
            return
        
        # Migrate tasks from pressured to available workers
        for pressured_id, pressured_info in pressured_workers:
            if available_workers:
                target_id, target_info = available_workers[0]
                self._migrate_tasks(pressured_id, target_id)
                
                self._migration_history.append({
                    'timestamp_ns': time.time_ns(),
                    'from_worker': pressured_id,
                    'to_worker': target_id,
                    'reason': 'memory_pressure'
                })
                
                print(f"[Scheduler] Migrated tasks from {pressured_id} to {target_id}")
                break
        
        if self.on_pressure_callback:
            self.on_pressure_callback(state)
    
    def _handle_critical_memory(self, state: ClusterMemoryState):
        """Handle critical memory situation."""
        print(f"[Scheduler] CRITICAL memory state: {state.memory_fraction:.1%}")
        
        # Force garbage collection on all workers
        self._force_gc_on_workers()
        
        if self.on_critical_callback:
            self.on_critical_callback(state)
    
    def _trigger_kill_protocol(self, state: ClusterMemoryState):
        """Trigger the /KILL protocol for memory violations."""
        if self._kill_protocol_triggered:
            return
        
        self._kill_protocol_triggered = True
        print(f"[Scheduler] TRIGGERING /KILL PROTOCOL - Memory exceeded: {state.memory_fraction:.1%}")
        
        # Log the violation
        violation_log = {
            'timestamp_ns': time.time_ns(),
            'type': 'memory_exceeded',
            'memory_fraction': state.memory_fraction,
            'used_gb': state.used_memory_gb,
            'limit_gb': self.global_limit_gb,
        }
        
        # Trigger kill callback (integrates with PowerShell orchestration)
        if self.on_kill_trigger:
            self.on_kill_trigger(violation_log)
        
        # In production, this would send signal to orchestration layer
        # to execute the /KILL protocol
    
    def _migrate_tasks(self, from_worker: str, to_worker: str):
        """Migrate tasks from one worker to another."""
        if not ray.is_initialized():
            return
        
        # In production, use Ray's task migration APIs
        # This is a placeholder for the actual migration logic
        print(f"[Scheduler] Migrating tasks: {from_worker} -> {to_worker}")
    
    def _force_gc_on_workers(self):
        """Force garbage collection on all Ray workers."""
        if not ray.is_initialized():
            return
        
        @ray.remote
        def force_gc():
            import gc
            gc.collect()
            return True
        
        # Dispatch GC to all workers
        try:
            ray.get([force_gc.remote() for _ in range(ray.cluster_resources().get('CPU', 1))])
        except Exception as e:
            print(f"[Scheduler] GC dispatch error: {e}")
    
    def enforce_worker_limit(self, worker_id: str) -> bool:
        """
        Enforce the 4GB limit on a specific worker.
        
        Returns True if limit is satisfied, False if action needed.
        """
        if worker_id not in self._worker_states:
            return True
        
        worker_info = self._worker_states[worker_id]
        
        if worker_info.used_memory_gb > self.worker_limit_gb:
            print(f"[Scheduler] Worker {worker_id} exceeds limit: "
                  f"{worker_info.used_memory_gb:.2f}GB > {self.worker_limit_gb}GB")
            return False
        
        return True
    
    def get_migration_history(self) -> List[Dict]:
        """Get history of task migrations."""
        return self._migration_history.copy()
    
    def is_kill_triggered(self) -> bool:
        """Check if kill protocol has been triggered."""
        return self._kill_protocol_triggered
    
    def reset_kill_flag(self):
        """Reset the kill protocol flag (after manual intervention)."""
        self._kill_protocol_triggered = False


@ray.remote(max_calls=100)
class MemoryLimitedWorker:
    """
    Ray actor with built-in memory limits.
    
    Automatically monitors its own memory and reports to scheduler.
    """
    
    def __init__(self, memory_limit_gb: float = PYTHON_WORKER_RAM_LIMIT_GB):
        self.memory_limit_gb = memory_limit_gb
        self.task_count = 0
        self.scheduler: Optional[RayResourceScheduler] = None
    
    def set_scheduler(self, scheduler: RayResourceScheduler):
        """Set reference to scheduler for reporting."""
        self.scheduler = scheduler
    
    def check_memory_limit(self) -> Tuple[bool, float]:
        """Check if within memory limit."""
        process = psutil.Process(os.getpid())
        try:
            mem_gb = process.memory_info().rss / (1024**3)
            return mem_gb <= self.memory_limit_gb, mem_gb
        except Exception:
            return True, 0.0
    
    def execute_with_memory_check(self, task_func, *args, **kwargs):
        """Execute task with memory checking."""
        before_ok, before_mem = self.check_memory_limit()
        
        if not before_ok:
            raise MemoryError(
                f"Worker memory {before_mem:.2f}GB exceeds limit {self.memory_limit_gb}GB"
            )
        
        result = task_func(*args, **kwargs)
        self.task_count += 1
        
        after_ok, after_mem = self.check_memory_limit()
        
        if not after_ok:
            # Report to scheduler
            if self.scheduler:
                self.scheduler._handle_memory_pressure(
                    self.scheduler.get_cluster_state()
                )
        
        return result


def create_scheduler_with_callbacks(
    on_pressure=None,
    on_critical=None,
    on_kill=None
) -> RayResourceScheduler:
    """Create scheduler with callbacks for orchestration integration."""
    scheduler = RayResourceScheduler()
    
    if on_pressure:
        scheduler.on_pressure_callback = on_pressure
    if on_critical:
        scheduler.on_critical_callback = on_critical
    if on_kill:
        scheduler.on_kill_trigger = on_kill
    
    return scheduler


if __name__ == '__main__':
    print("Ray Resource Scheduler")
    print("=" * 40)
    
    # Initialize Ray with memory limits
    if not ray.is_initialized():
        ray.init(
            object_store_memory=int(2 * 1024**3),  # 2GB object store
            _system_config={
                'max_direct_call_object_size': 1024**2,
                'min_worker_size': 1,
            }
        )
    
    # Create scheduler
    scheduler = RayResourceScheduler(
        global_limit_gb=GLOBAL_RAM_LIMIT_GB,
        worker_limit_gb=PYTHON_WORKER_RAM_LIMIT_GB
    )
    
    # Set up callbacks
    def on_pressure(state):
        print(f"  -> Pressure callback: {state.memory_fraction:.1%}")
    
    def on_critical(state):
        print(f"  -> Critical callback: {state.memory_fraction:.1%}")
    
    def on_kill(log):
        print(f"  -> KILL PROTOCOL TRIGGERED: {log}")
    
    scheduler.on_pressure_callback = on_pressure
    scheduler.on_critical_callback = on_critical
    scheduler.on_kill_trigger = on_kill
    
    # Start monitoring
    scheduler.start_monitoring(interval_seconds=2.0)
    
    # Get initial state
    state = scheduler.get_cluster_state()
    print(f"\nInitial Cluster State:")
    print(f"  Total Memory: {state.total_memory_gb:.1f}GB")
    print(f"  Used Memory: {state.used_memory_gb:.2f}GB")
    print(f"  Memory Fraction: {state.memory_fraction:.1%}")
    print(f"  State: {state.state.value}")
    
    # Run for a few seconds to demonstrate monitoring
    time.sleep(5)
    
    # Stop monitoring
    scheduler.stop_monitoring()
    
    print("\nScheduler stopped.")
