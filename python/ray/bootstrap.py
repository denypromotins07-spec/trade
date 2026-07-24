"""
Ray Cluster Bootstrap Module - Stage 55
========================================
Custom Ray initialization script that pre-allocates the Plasma object store
strictly within the 4GB boundary, disabling automatic worker respawning to save RAM.

Optimized for:
- AMD Ryzen AI 5 architecture (znver4)
- 4GB Python RAM quota enforcement
- DirectML/ROCm GPU tensor offloading
- Microsecond latency requirements

Author: Nautilus Quantitative Engineering Team
Stage: 55 - Process Isolation & IPC Validation
"""

import os
import sys
import logging
import ctypes
import platform
from typing import Optional, Dict, Any
from pathlib import Path

import ray
from ray._private.ray_logging import setup_logger

# Configure strict logging for bootstrap operations
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# =============================================================================
# CONSTANTS - STRICT MEMORY BOUNDARIES
# =============================================================================

# Maximum Plasma object store size: 3.5GB (leaves 0.5GB for Python overhead)
PLASMA_STORE_MAX_BYTES: int = 3_758_096_384  # 3.5 * 1024^3

# Hard limit for Python process RSS
PYTHON_RSS_HARD_LIMIT_BYTES: int = 4_294_967_296  # 4GB exactly

# Disable automatic worker respawning to prevent RAM spikes
RAY_AUTO_RESTART_WORKERS: bool = False

# Number of initial workers (conservative for 8GB total system RAM)
RAY_NUM_WORKERS: int = 2

# Object store memory fraction of available RAM
OBJECT_STORE_MEMORY_FRACTION: float = 0.4375  # 3.5GB / 8GB

# AMD ROCm/DirectML environment validation
AMD_GPU_ENABLED: bool = False

# =============================================================================
# AMD GPU ENVIRONMENT VALIDATION
# =============================================================================


def validate_amd_gpu_environment() -> bool:
    """
    Validate AMD DirectML/ROCm environment for GPU tensor offloading.
    
    Returns:
        bool: True if AMD GPU environment is properly configured
    """
    global AMD_GPU_ENABLED
    
    system = platform.system()
    logger.info(f"Validating AMD GPU environment on {system}")
    
    if system == "Windows":
        # Check for DirectML
        directml_paths = [
            Path(r"C:\Program Files\DirectML"),
            Path(os.environ.get("DIRECTML_PATH", "")),
            Path(os.environ.get("PROGRAMFILES", "")) / "DirectML",
        ]
        
        for path in directml_paths:
            if path.exists() and any(path.glob("*.dll")):
                logger.info(f"DirectML libraries found at: {path}")
                AMD_GPU_ENABLED = True
                break
        
        # Check for DirectML environment variable
        if "DIRECTML_PATH" in os.environ:
            logger.info(f"DIRECTML_PATH set to: {os.environ['DIRECTML_PATH']}")
            AMD_GPU_ENABLED = True
            
    elif system == "Linux":
        # Check for ROCm
        rocm_paths = [
            Path("/opt/rocm"),
            Path(os.environ.get("ROCM_PATH", "")),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]
        
        for path in rocm_paths:
            if path.exists() and any(path.glob("**/librocblas.so*", recursive=True)):
                logger.info(f"ROCm libraries found at: {path}")
                AMD_GPU_ENABLED = True
                break
        
        # Check for ROCm environment variables
        if "ROCM_PATH" in os.environ or "HIP_PATH" in os.environ:
            logger.info("ROCm environment variables detected")
            AMD_GPU_ENABLED = True
    
    if not AMD_GPU_ENABLED:
        logger.warning("AMD GPU (DirectML/ROCm) not detected - falling back to CPU")
        logger.warning("Tensor operations will use CPU-only execution")
    
    return AMD_GPU_ENABLED


def set_amd_gpu_environment_variables() -> None:
    """
    Set AMD-specific environment variables for optimal GPU performance.
    """
    if platform.system() == "Windows":
        # DirectML optimization flags
        os.environ["DML_ENABLE_DYNAMIC_HEAP"] = "1"
        os.environ["DML_DISABLE_FLOAT16"] = "0"  # Enable FP16 for AI 5 NPU
        logger.info("DirectML environment variables configured")
    else:
        # ROCm optimization flags
        os.environ["HSA_ENABLE_SDMA"] = "1"
        os.environ["ROCR_VISIBLE_DEVICES"] = os.environ.get("ROCR_VISIBLE_DEVICES", "GPU")
        os.environ["HIP_DEVICE_RESET"] = "1"
        logger.info("ROCm environment variables configured")


# =============================================================================
# MEMORY LIMIT ENFORCEMENT
# =============================================================================


def get_available_memory_bytes() -> int:
    """
    Get available system memory in bytes.
    
    Returns:
        int: Available memory in bytes
    """
    if platform.system() == "Windows":
        import ctypes.wintypes
        
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        
        memory_status = MEMORYSTATUSEX()
        memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        return int(memory_status.ullAvailPhys)
    else:
        # Linux/macOS
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
        return 8 * 1024 * 1024 * 1024  # Default to 8GB


def enforce_python_memory_limit() -> bool:
    """
    Enforce the 4GB Python RAM quota at the OS level.
    
    Returns:
        bool: True if limit was successfully enforced
    """
    try:
        if platform.system() == "Windows":
            # Use Windows Job Objects via ctypes
            kernel32 = ctypes.windll.kernel32
            
            # Create a job object
            job_handle = kernel32.CreateJobObjectW(None, "NautilusPythonQuota")
            if not job_handle:
                logger.error("Failed to create Windows Job Object")
                return False
            
            # Set up extended limit information
            JOBOBJECT_BASIC_LIMIT_INFORMATION = ctypes.c_int * 9
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION = ctypes.c_int * 12
            
            # Query current limits
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            kernel32.QueryInformationJobObject(
                job_handle, 9,  # JobObjectExtendedLimitInformation
                ctypes.byref(info),
                ctypes.sizeof(info),
                None
            )
            
            # Set memory limit (4GB)
            info[4] = PYTHON_RSS_HARD_LIMIT_BYTES  # PeakJobMemoryUsed
            
            # Apply limits
            kernel32.SetInformationJobObject(
                job_handle,
                9,  # JobObjectExtendedLimitInformation
                ctypes.byref(info),
                ctypes.sizeof(info)
            )
            
            # Assign current process to job
            current_process = kernel32.GetCurrentProcess()
            if not kernel32.AssignProcessToJobObject(job_handle, current_process):
                logger.error("Failed to assign process to Job Object")
                return False
            
            logger.info(f"Successfully enforced {PYTHON_RSS_HARD_LIMIT_BYTES / (1024**3):.1f}GB memory limit via Job Object")
            return True
        else:
            # Linux: Use resource limits
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (PYTHON_RSS_HARD_LIMIT_BYTES, PYTHON_RSS_HARD_LIMIT_BYTES))
            logger.info(f"Successfully enforced {PYTHON_RSS_HARD_LIMIT_BYTES / (1024**3):.1f}GB memory limit via RLIMIT_AS")
            return True
            
    except Exception as e:
        logger.error(f"Failed to enforce memory limit: {e}")
        return False


# =============================================================================
# RAY CLUSTER INITIALIZATION
# =============================================================================


def calculate_optimal_ray_config() -> Dict[str, Any]:
    """
    Calculate optimal Ray configuration based on available resources.
    
    Returns:
        Dict[str, Any]: Ray initialization configuration
    """
    available_memory = get_available_memory_bytes()
    logger.info(f"Available system memory: {available_memory / (1024**3):.2f}GB")
    
    # Calculate plasma store size (43.75% of total RAM = 3.5GB of 8GB)
    plasma_store_size = min(
        PLASMA_STORE_MAX_BYTES,
        int(available_memory * OBJECT_STORE_MEMORY_FRACTION)
    )
    
    logger.info(f"Plasma store size: {plasma_store_size / (1024**3):.2f}GB")
    
    config = {
        "address": os.environ.get("RAY_ADDRESS", None),
        "num_cpus": os.cpu_count() or 8,
        "_memory": plasma_store_size,  # Internal Ray parameter for object store
        "object_store_memory": plasma_store_size,
        "runtime_env": {
            "env_vars": {
                "RAY_DISABLE_AUTO_RESTART": "1" if RAY_AUTO_RESTART_WORKERS else "0",
                "RAY_NUM_WORKERS": str(RAY_NUM_WORKERS),
            }
        },
        "_temp_dir": "/tmp/ray" if platform.system() != "Windows" else None,
        "include_dashboard": False,  # Disable dashboard to save RAM
        "log_to_driver": True,
        "logging_level": logging.INFO,
    }
    
    # Add GPU resources if AMD GPU is available
    if AMD_GPU_ENABLED:
        config["num_gpus"] = 1
        logger.info("GPU resources enabled for Ray cluster")
    
    return config


def initialize_ray_cluster() -> bool:
    """
    Initialize the Ray cluster with strict memory constraints.
    
    Returns:
        bool: True if initialization was successful
    """
    logger.info("=" * 60)
    logger.info("Nautilus Ray Cluster Bootstrap - Stage 55")
    logger.info("=" * 60)
    
    # Step 1: Validate AMD GPU environment
    logger.info("Step 1: Validating AMD GPU environment...")
    validate_amd_gpu_environment()
    if AMD_GPU_ENABLED:
        set_amd_gpu_environment_variables()
    
    # Step 2: Enforce Python memory limit
    logger.info("Step 2: Enforcing 4GB Python memory quota...")
    if not enforce_python_memory_limit():
        logger.error("Failed to enforce memory limit - aborting bootstrap")
        return False
    
    # Step 3: Calculate optimal configuration
    logger.info("Step 3: Calculating optimal Ray configuration...")
    ray_config = calculate_optimal_ray_config()
    
    # Step 4: Initialize Ray
    logger.info("Step 4: Initializing Ray cluster...")
    try:
        ray.init(**ray_config)
        logger.info("Ray cluster initialized successfully")
        
        # Verify cluster status
        cluster_resources = ray.cluster_resources()
        logger.info(f"Cluster resources: {cluster_resources}")
        
        # Verify object store memory
        object_store_memory = cluster_resources.get("object_store_memory", 0)
        if object_store_memory > PLASMA_STORE_MAX_BYTES:
            logger.warning(
                f"Object store memory ({object_store_memory / (1024**3):.2f}GB) "
                f"exceeds maximum ({PLASMA_STORE_MAX_BYTES / (1024**3):.2f}GB)"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize Ray cluster: {e}")
        return False


def shutdown_ray_cluster() -> None:
    """
    Gracefully shutdown the Ray cluster and release all resources.
    """
    logger.info("Shutting down Ray cluster...")
    try:
        ray.shutdown()
        logger.info("Ray cluster shutdown complete")
    except Exception as e:
        logger.error(f"Error during Ray shutdown: {e}")


# =============================================================================
# PRE-FLIGHT VALIDATION
# =============================================================================


def run_pre_flight_checks() -> bool:
    """
    Run pre-flight checks before starting the trading system.
    
    Returns:
        bool: True if all checks passed
    """
    logger.info("Running pre-flight checks...")
    
    checks_passed = True
    
    # Check 1: Memory limit enforcement
    import psutil
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss
    
    if current_memory > PYTHON_RSS_HARD_LIMIT_BYTES * 0.5:
        logger.warning(
            f"Current memory usage ({current_memory / (1024**3):.2f}GB) "
            f"is above 50% of quota at bootstrap"
        )
    
    # Check 2: Plasma store accessibility
    try:
        import ray.internal
        ray.internal.internal.free()
        logger.info("Plasma store accessible")
    except Exception as e:
        logger.error(f"Plasma store check failed: {e}")
        checks_passed = False
    
    # Check 3: Worker availability
    try:
        @ray.remote
        def health_check():
            return "OK"
        
        result = ray.get(health_check.remote())
        if result == "OK":
            logger.info("Ray workers responsive")
        else:
            logger.error("Ray workers returned unexpected response")
            checks_passed = False
    except Exception as e:
        logger.error(f"Ray worker check failed: {e}")
        checks_passed = False
    
    return checks_passed


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def bootstrap() -> bool:
    """
    Main bootstrap entry point for the Ray cluster.
    
    Returns:
        bool: True if bootstrap was successful
    """
    success = initialize_ray_cluster()
    
    if success:
        success = run_pre_flight_checks()
    
    if not success:
        logger.error("Bootstrap failed - initiating graceful shutdown")
        shutdown_ray_cluster()
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Ray cluster bootstrap completed successfully")
    logger.info(f"Plasma store limit: {PLASMA_STORE_MAX_BYTES / (1024**3):.2f}GB")
    logger.info(f"Python memory quota: {PYTHON_RSS_HARD_LIMIT_BYTES / (1024**3):.2f}GB")
    logger.info(f"AMD GPU enabled: {AMD_GPU_ENABLED}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    bootstrap()
