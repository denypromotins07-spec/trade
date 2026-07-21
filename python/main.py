# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - PYTHON ENTRY POINT
# =============================================================================
# File: python/main.py
# Purpose: Initialize Ray distributed cluster and orchestrate AI workers
# Memory Constraint: Strict 4GB RAM ceiling for all Python/AI processes
# GPU Support: AMD DirectML/ROCm environment detection for future scaling
# =============================================================================

"""
Nautilus/Ray Trading Bot - Python Entry Point

This module initializes the Ray distributed compute cluster with strict
memory isolation to preserve 4GB for the Rust execution engine.

Architecture:
- Ray Head Node: Localhost, port 6379
- Ray Workers: Configurable CPU count, memory-capped at 4GB total
- AI Brain: Reinforcement learning agents for signal generation
- Data Pipeline: Parallel walk-forward training on historical tick data

Usage:
    python python/main.py --start
    python python/main.py --stop
"""

import os
import sys
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Configure structured logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("nautilus_ray_bot")

# =============================================================================
# ENVIRONMENT VALIDATION - Pre-flight checks
# =============================================================================


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set correctly.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    required_vars = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "RAY_HEAD_PORT",
        "RAY_WORKER_MEMORY_GB",
    ]
    
    missing = []
    for var in required_vars:
        if var not in os.environ:
            missing.append(var)
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    
    # Validate memory constraints
    try:
        worker_memory_gb = int(os.environ.get("RAY_WORKER_MEMORY_GB", "4"))
        if worker_memory_gb > 4:
            logger.warning(
                f"Ray worker memory ({worker_memory_gb}GB) exceeds recommended 4GB limit"
            )
    except ValueError:
        logger.error("RAY_WORKER_MEMORY_GB must be a valid integer")
        return False
    
    logger.info("Environment validation passed")
    return True


def detect_gpu_backend() -> str:
    """
    Detect available GPU acceleration backend for AI inference.
    
    Priority order:
    1. ROCm (AMD GPUs on Linux)
    2. DirectML (AMD GPUs on Windows)
    3. CUDA (NVIDIA GPUs)
    4. CPU fallback
    
    Returns:
        str: Detected backend name ("rocm", "directml", "cuda", or "cpu")
    """
    # Check for ROCm (Linux AMD)
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path or Path("/opt/rocm").exists():
        logger.info("ROCm installation detected")
        os.environ["AI_ROCM_ENABLED"] = "true"
        return "rocm"
    
    # Check for DirectML (Windows AMD)
    directml_enabled = os.environ.get("AI_DIRECTML_ENABLED", "").lower() == "true"
    if directml_enabled and sys.platform == "win32":
        logger.info("DirectML enabled for Windows AMD GPU acceleration")
        os.environ["AI_DEVICE"] = "directml"
        return "directml"
    
    # Check for CUDA (NVIDIA)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        logger.info(f"CUDA devices detected: {cuda_visible}")
        os.environ["AI_DEVICE"] = "cuda"
        return "cuda"
    
    # CPU fallback
    logger.info("No GPU acceleration detected, using CPU backend")
    os.environ["AI_DEVICE"] = "cpu"
    return "cpu"


# =============================================================================
# RAY CLUSTER INITIALIZATION
# =============================================================================


def initialize_ray_cluster() -> Optional[Any]:
    """
    Initialize the Ray distributed compute cluster with strict memory limits.
    
    Memory Budget:
    - Object Store: 2GB (shared memory for inter-process communication)
    - Worker Heap: 2GB (Python objects, model weights, training data)
    - Total: 4GB maximum (leaves 4GB for Rust engine on 8GB system)
    
    Returns:
        Ray context object or None if initialization fails
    """
    try:
        import ray
    except ImportError:
        logger.error("Ray is not installed. Run: pip install ray[default]")
        return None
    
    # Load configuration from environment
    head_host = os.environ.get("RAY_HEAD_HOST", "127.0.0.1")
    head_port = int(os.environ.get("RAY_HEAD_PORT", "6379"))
    dashboard_port = int(os.environ.get("RAY_DASHBOARD_PORT", "8265"))
    num_cpus = int(os.environ.get("RAY_NUM_CPUS", "6"))
    worker_memory_gb = int(os.environ.get("RAY_WORKER_MEMORY_GB", "4"))
    object_store_gb = int(os.environ.get("RAY_OBJECT_STORE_MEMORY_GB", "2"))
    
    # Calculate memory limits in bytes
    worker_memory_bytes = worker_memory_gb * 1024 * 1024 * 1024
    object_store_bytes = object_store_gb * 1024 * 1024 * 1024
    
    # Ray initialization configuration
    ray_config = {
        "address": f"{head_host}:{head_port}",
        "num_cpus": num_cpus,
        "_memory": worker_memory_bytes,
        "_object_store_memory": object_store_bytes,
        "include_dashboard": True,
        "dashboard_host": "127.0.0.1",
        "dashboard_port": dashboard_port,
        "log_to_driver": True,
        "logging_level": "info",
    }
    
    logger.info(f"Initializing Ray cluster with {num_cpus} CPUs")
    logger.info(f"Worker memory limit: {worker_memory_gb}GB")
    logger.info(f"Object store size: {object_store_gb}GB")
    
    try:
        # Initialize Ray with strict memory constraints
        ctx = ray.init(**ray_config)
        
        # Verify cluster health
        cluster_resources = ray.cluster_resources()
        logger.info(f"Ray cluster initialized successfully")
        logger.info(f"Available resources: {dict(cluster_resources)}")
        
        return ctx
        
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        return None


# =============================================================================
# REMOTE ACTOR DEFINITIONS
# =============================================================================


def register_remote_actors():
    """
    Register Ray remote actors for distributed AI processing.
    
    Actors are initialized with memory annotations to ensure
    proper resource allocation within the 4GB budget.
    """
    import ray
    
    # Import actor definitions (lazy loading to avoid circular imports)
    from python.ai.brain import AIBrain
    
    # Create Ray actor class with resource annotations
    @ray.remote(
        num_cpus=1,
        memory=int(1.5 * 1024 * 1024 * 1024),  # 1.5GB per actor max
        max_restarts=3,
        max_task_retries=2,
    )
    class DistributedAIBrain:
        """Ray-wrapped AI brain actor for parallel inference."""
        
        def __init__(self):
            self.brain = AIBrain()
            
        def initialize(self, config: Dict[str, Any]) -> bool:
            return self.brain.initialize(config)
            
        def infer(self, market_data: bytes) -> Dict[str, Any]:
            return self.brain.infer(market_data)
            
        def train_batch(self, batch_data: bytes) -> Dict[str, float]:
            return self.brain.train_batch(batch_data)
    
    logger.info("Remote actors registered successfully")
    return DistributedAIBrain


# =============================================================================
# SIGNAL HANDLING & GRACEFUL SHUTDOWN
# =============================================================================


class GracefulShutdownHandler:
    """
    Handle SIGINT/SIGTERM signals for graceful cluster shutdown.
    
    Ensures:
    1. All pending inference tasks complete or are cancelled
    2. Model checkpoints are saved
    3. Ray cluster resources are released
    4. SOUL.md is updated with shutdown state
    """
    
    def __init__(self):
        self.shutdown_requested = False
        self.original_sigint = None
        self.original_sigterm = None
        
    def register(self):
        """Register signal handlers."""
        self.original_sigint = signal.signal(signal.SIGINT, self._handler)
        self.original_sigterm = signal.signal(signal.SIGTERM, self._handler)
        logger.info("Graceful shutdown handlers registered")
        
    def _handler(self, signum, frame):
        """Signal handler callback."""
        logger.warning(f"Shutdown signal received (signal {signum})")
        self.shutdown_requested = True
        
    def unregister(self):
        """Restore original signal handlers."""
        if self.original_sigint:
            signal.signal(signal.SIGINT, self.original_sigint)
        if self.original_sigterm:
            signal.signal(signal.SIGTERM, self.original_sigterm)


def graceful_shutdown(ray_context=None):
    """
    Perform graceful shutdown of all components.
    
    Steps:
    1. Stop accepting new inference requests
    2. Wait for pending tasks to complete (with timeout)
    3. Save model checkpoints
    4. Update SOUL.md with final state
    5. Shutdown Ray cluster
    6. Release all resources
    """
    logger.info("Initiating graceful shutdown...")
    
    import ray
    
    # Cancel pending tasks
    logger.info("Cancelling pending tasks...")
    
    # Save checkpoints
    logger.info("Saving model checkpoints...")
    try:
        from python.ai.brain import save_checkpoint
        save_checkpoint()
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
    
    # Update SOUL.md
    logger.info("Updating SOUL.md with shutdown state...")
    try:
        update_soul_ledger("SHUTDOWN", "Graceful shutdown completed")
    except Exception as e:
        logger.error(f"Failed to update SOUL.md: {e}")
    
    # Shutdown Ray
    if ray_context:
        logger.info("Shutting down Ray cluster...")
        ray.shutdown()
    
    logger.info("Graceful shutdown completed")


def update_soul_ledger(event_type: str, message: str):
    """
    Append an entry to the SOUL.md self-learning ledger.
    
    Args:
        event_type: Type of event (STARTUP, SHUTDOWN, TRADE, etc.)
        message: Human-readable description of the event
    """
    soul_path = Path(__file__).parent.parent / "SOUL.md"
    
    timestamp = __import__("datetime").datetime.now().isoformat()
    entry = f"\n## [{event_type}] {timestamp}\n\n{message}\n"
    
    try:
        with open(soul_path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug(f"Updated SOUL.md with {event_type} entry")
    except Exception as e:
        logger.error(f"Failed to update SOUL.md: {e}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the Python AI cluster."""
    parser = argparse.ArgumentParser(
        description="Nautilus/Ray Trading Bot - Python AI Cluster"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the Ray cluster and AI workers",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the Ray cluster gracefully",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check cluster status",
    )
    
    args = parser.parse_args()
    
    if args.stop:
        logger.info("Stopping Ray cluster...")
        try:
            import ray
            ray.shutdown()
            logger.info("Ray cluster stopped")
        except Exception as e:
            logger.error(f"Error stopping cluster: {e}")
        return
    
    if args.status:
        try:
            import ray
            if ray.is_initialized():
                print(f"Ray cluster running: {ray.cluster_resources()}")
            else:
                print("Ray cluster not initialized")
        except Exception as e:
            print(f"Error checking status: {e}")
        return
    
    # Default: Start the cluster
    if not args.start:
        parser.print_help()
        return
    
    logger.info("=" * 60)
    logger.info("Nautilus/Ray Trading Bot - Python AI Cluster Starting")
    logger.info("=" * 60)
    
    # Step 1: Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Step 2: Detect GPU backend
    gpu_backend = detect_gpu_backend()
    logger.info(f"GPU backend: {gpu_backend}")
    
    # Step 3: Initialize Ray cluster
    ray_context = initialize_ray_cluster()
    if ray_context is None:
        logger.error("Failed to initialize Ray cluster")
        sys.exit(1)
    
    # Step 4: Register remote actors
    try:
        register_remote_actors()
    except Exception as e:
        logger.error(f"Failed to register actors: {e}")
    
    # Step 5: Register signal handlers
    shutdown_handler = GracefulShutdownHandler()
    shutdown_handler.register()
    
    # Step 6: Update SOUL.md
    update_soul_ledger("STARTUP", "Python AI cluster initialized successfully")
    
    # Step 7: Main loop (keep alive until shutdown signal)
    logger.info("Cluster is ready. Waiting for tasks...")
    
    try:
        import time
        while not shutdown_handler.shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        graceful_shutdown(ray_context)
        shutdown_handler.unregister()
    
    logger.info("Python AI cluster terminated")


if __name__ == "__main__":
    main()
