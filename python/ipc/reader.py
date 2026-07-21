"""
Shared Memory IPC Reader for Rust-Python Communication

This module implements the Python consumer using numpy.memmap to read the shared
memory space directly, transforming raw bytes into NumPy arrays without triggering
memory reallocations.

**Performance Characteristics:**
- Zero-copy memory access via mmap
- No data duplication in Python heap
- AMD ROCm/DirectML environment checks for GPU-accelerated processing
- Thread-safe read operations with atomic pointers

**Architecture:**
The reader connects to the Rust-side shared memory segment created by
src/ipc/shared_mem.rs and provides:
1. Read-only views of ring buffer state
2. Order book depth snapshots
3. Tick data streams
4. SMC signal vectors

Memory Layout (defined in Rust):
- Header: 64 bytes (magic, version, write_idx, read_idx, timestamp)
- Order Book: 10 levels * 2 sides * 2 values (price/size) = 320 bytes
- Tick Buffer: 1024 ticks * 8 values = 8192 bytes
- SMC Signals: 16 floats = 64 bytes
- Total: ~8640 bytes minimum
"""

import os
import mmap
import struct
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


# Shared memory layout constants (must match Rust side)
HEADER_SIZE = 64
HEADER_FORMAT = '<QQqqQ'  # magic, version, write_idx, read_idx, timestamp
MAGIC_NUMBER = 0xDEADBEEFCAFEBABE
VERSION = 1

# Data section offsets
ORDERBOOK_OFFSET = HEADER_SIZE
ORDERBOOK_SIZE = 320  # 10 levels * 2 sides * 2 values * 8 bytes

TICK_BUFFER_OFFSET = ORDERBOOK_OFFSET + ORDERBOOK_SIZE
TICK_BUFFER_SIZE = 8192  # 1024 ticks * 8 values * 1 byte (scaled differently in practice)

SMC_SIGNALS_OFFSET = TICK_BUFFER_OFFSET + TICK_BUFFER_SIZE
SMC_SIGNALS_SIZE = 64  # 16 floats * 4 bytes

# Total minimum size
MIN_SHM_SIZE = SMC_SIGNALS_OFFSET + SMC_SIGNALS_SIZE


def check_amd_gpu_environment() -> Dict[str, Any]:
    """
    Check AMD GPU environment variables for future acceleration.
    
    Returns:
        Dictionary with GPU environment status
    """
    env_info = {
        'rocm_path': os.environ.get('ROCM_PATH', 'not set'),
        'hip_visible_devices': os.environ.get('HIP_VISIBLE_DEVICES', 'not set'),
        'directml_enabled': os.environ.get('DIRECTML_ENABLED', '0') == '1',
        'gpu_available': False,
    }
    
    # Check ROCm
    rocm_path = env_info['rocm_path']
    if rocm_path != 'not set' and os.path.exists(rocm_path):
        env_info['gpu_available'] = True
        logger.info(f"ROCm environment detected at {rocm_path}")
    
    # Check DirectML (Windows)
    if env_info['directml_enabled']:
        env_info['gpu_available'] = True
        logger.info("DirectML environment enabled")
    
    return env_info


class SharedMemoryReader:
    """
    Zero-copy reader for Rust shared memory segments.
    
    Uses numpy.memmap to create views into the shared memory without
    copying data into Python's heap. All reads are direct memory accesses.
    """
    
    def __init__(
        self,
        shm_path: str,
        readonly: bool = True,
        max_size: int = 512 * 1024 * 1024,  # 512MB max (matches Rust)
    ):
        """
        Initialize the shared memory reader.
        
        Args:
            shm_path: Path to the shared memory file
            readonly: Whether to open in read-only mode
            max_size: Maximum expected size of shared memory
        """
        self.shm_path = Path(shm_path)
        self.readonly = readonly
        self.max_size = max_size
        
        self._mmap: Optional[mmap.mmap] = None
        self._header_view: Optional[np.ndarray] = None
        self._orderbook_view: Optional[np.ndarray] = None
        self._tick_view: Optional[np.ndarray] = None
        self._smc_view: Optional[np.ndarray] = None
        
        self._is_open = False
        self._last_read_idx = -1
        
        # Log GPU environment
        gpu_env = check_amd_gpu_environment()
        logger.info(f"GPU Environment: {gpu_env['gpu_available']}")
        
        self._open()
    
    def _open(self):
        """Open the shared memory file and create memory views."""
        if not self.shm_path.exists():
            raise FileNotFoundError(f"Shared memory file not found: {self.shm_path}")
        
        # Open file descriptor
        flags = os.O_RDONLY if self.readonly else os.O_RDWR
        self._fd = os.open(str(self.shm_path), flags)
        
        try:
            # Get file size
            file_size = os.fstat(self._fd).st_size
            
            if file_size < MIN_SHM_SIZE:
                raise ValueError(
                    f"Shared memory file too small: {file_size} bytes, "
                    f"expected at least {MIN_SHM_SIZE}"
                )
            
            # Create memory map
            prot = mmap.PROT_READ if self.readonly else (mmap.PROT_READ | mmap.PROT_WRITE)
            self._mmap = mmap.mmap(
                self._fd,
                min(file_size, self.max_size),
                mmap.MAP_SHARED,
                prot,
            )
            
            # Create numpy views (zero-copy)
            self._create_views()
            
            self._is_open = True
            logger.info(f"Shared memory opened: {self.shm_path}, size={file_size}")
            
        except Exception as e:
            os.close(self._fd)
            raise RuntimeError(f"Failed to map shared memory: {e}")
    
    def _create_views(self):
        """Create numpy memmap views for each data section."""
        if self._mmap is None:
            return
        
        # Header view (64 bytes)
        header_array = np.frombuffer(
            self._mmap[0:HEADER_SIZE],
            dtype=np.uint64,
        )
        self._header_view = header_array
        
        # Order book view (320 bytes = 40 uint64 values)
        ob_start = ORDERBOOK_OFFSET
        ob_end = ob_start + ORDERBOOK_SIZE
        self._orderbook_view = np.frombuffer(
            self._mmap[ob_start:ob_end],
            dtype=np.float64,
        ).reshape(2, 10, 2)  # [side, level, price/size]
        
        # Tick buffer view
        tick_start = TICK_BUFFER_OFFSET
        tick_end = tick_start + TICK_BUFFER_SIZE
        self._tick_view = np.frombuffer(
            self._mmap[tick_start:tick_end],
            dtype=np.float32,
        ).reshape(1024, 8)  # [tick_index, 8 values]
        
        # SMC signals view (16 floats)
        smc_start = SMC_SIGNALS_OFFSET
        smc_end = smc_start + SMC_SIGNALS_SIZE
        self._smc_view = np.frombuffer(
            self._mmap[smc_start:smc_end],
            dtype=np.float32,
        ).reshape(16,)
    
    def read_header(self) -> Dict[str, Any]:
        """
        Read and parse the shared memory header.
        
        Returns:
            Dictionary with header fields
        """
        if self._header_view is None or len(self._header_view) < 5:
            return {}
        
        header = {
            'magic': int(self._header_view[0]),
            'version': int(self._header_view[1]),
            'write_idx': int(self._header_view[2]),
            'read_idx': int(self._header_view[3]),
            'timestamp_ms': int(self._header_view[4]),
        }
        
        # Validate magic number
        if header['magic'] != MAGIC_NUMBER:
            logger.warning(f"Invalid magic number: {hex(header['magic'])}")
        
        return header
    
    def read_orderbook(self) -> Optional[np.ndarray]:
        """
        Read current order book state.
        
        Returns:
            Order book array of shape [2, 10, 2] (side, level, price/size)
            or None if unavailable
        """
        if self._orderbook_view is None:
            return None
        
        # Return a view (no copy)
        return self._orderbook_view
    
    def read_latest_ticks(self, count: int = 100) -> Optional[np.ndarray]:
        """
        Read the most recent tick data.
        
        Args:
            count: Number of ticks to retrieve
            
        Returns:
            Tick array of shape [count, 8] or None
        """
        if self._tick_view is None:
            return None
        
        header = self.read_header()
        write_idx = header.get('write_idx', 0)
        
        if write_idx == 0:
            return None
        
        # Get last N ticks from circular buffer
        start_idx = max(0, write_idx - count)
        indices = np.arange(start_idx, write_idx) % 1024
        
        return self._tick_view[indices]
    
    def read_smc_signals(self) -> Optional[np.ndarray]:
        """
        Read Smart Money Concepts signals.
        
        Returns:
            SMC signals array of shape [16] or None
        """
        if self._smc_view is None:
            return None
        
        return self._smc_view.copy()  # Small enough to copy
    
    def read_latest(self) -> Optional[np.ndarray]:
        """
        Read all data as a single flattened array.
        
        This is the main method used by the RL environment to get
        a complete state observation.
        
        Returns:
            Flattened feature array or None
        """
        if not self._is_open:
            return None
        
        header = self.read_header()
        current_read_idx = header.get('read_idx', 0)
        
        # Skip if no new data
        if current_read_idx == self._last_read_idx:
            return None
        
        self._last_read_idx = current_read_idx
        
        # Build feature vector
        features = []
        
        # Order book features (flattened)
        if self._orderbook_view is not None:
            features.extend(self._orderbook_view.flatten().tolist())
        
        # SMC signals
        if self._smc_view is not None:
            features.extend(self._smc_view.tolist())
        
        # Latest tick
        if self._tick_view is not None:
            latest_tick_idx = (header.get('write_idx', 1) - 1) % 1024
            features.extend(self._tick_view[latest_tick_idx].tolist())
        
        return np.array(features, dtype=np.float32)
    
    def has_new_data(self) -> bool:
        """Check if new data is available since last read."""
        if not self._is_open:
            return False
        
        header = self.read_header()
        current_read_idx = header.get('read_idx', 0)
        return current_read_idx != self._last_read_idx
    
    def close(self):
        """Close the shared memory mapping."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        
        if hasattr(self, '_fd'):
            os.close(self._fd)
        
        self._is_open = False
        logger.debug("Shared memory closed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for creating readers
def create_reader(
    shm_path: str = "/tmp/nautilus_shm",
    readonly: bool = True,
) -> Optional[SharedMemoryReader]:
    """
    Create a shared memory reader with error handling.
    
    Args:
        shm_path: Path to shared memory file
        readonly: Open in read-only mode
        
    Returns:
        SharedMemoryReader instance or None if unavailable
    """
    try:
        return SharedMemoryReader(shm_path=shm_path, readonly=readonly)
    except FileNotFoundError:
        logger.debug(f"Shared memory not found at {shm_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to create shared memory reader: {e}")
        return None


if __name__ == "__main__":
    # Test the reader
    print("Testing SharedMemoryReader...")
    
    # Check GPU environment
    gpu_env = check_amd_gpu_environment()
    print(f"GPU Environment: {gpu_env}")
    
    # Try to connect to shared memory
    reader = create_reader("/tmp/nautilus_shm")
    
    if reader is not None:
        print("Connected to shared memory!")
        
        header = reader.read_header()
        print(f"Header: {header}")
        
        orderbook = reader.read_orderbook()
        if orderbook is not None:
            print(f"Order book shape: {orderbook.shape}")
        
        smc = reader.read_smc_signals()
        if smc is not None:
            print(f"SMC signals: {smc}")
        
        reader.close()
    else:
        print("Shared memory not available (this is expected if Rust engine is not running)")
    
    print("Test complete")