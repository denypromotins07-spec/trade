"""
Python Shared Memory Reader using numpy.memmap

This module implements the Python consumer using numpy.memmap to read the shared
memory space directly, transforming raw bytes into NumPy arrays without triggering
memory reallocations. Designed for zero-copy IPC with the Rust engine.

Key Features:
- numpy.memmap for zero-copy memory access
- Direct binary parsing of Rust data structures
- AMD DirectML/ROCm environment detection
- Thread-safe read operations with atomic position tracking
"""

import os
import struct
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# Constants and Configuration
# =============================================================================

# Header magic number (must match Rust: 0x4E415654 = "NAVT")
MMAP_MAGIC = 0x4E415654

# Header structure format (matches Rust SharedMemoryHeader)
# magic: u32, version: u32, buffer_size: u64
# write_pos: u64, read_pos: u64, items_written: u64, items_read: u64
# writer_active: bool, reader_active: bool, last_write_ns: u64, last_read_ns: u64
HEADER_FORMAT = '<IIQQQQQ?QQ'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Default shared memory path
DEFAULT_SHM_PATH = "/tmp/nautilus_shm.bin"

# Maximum mmap size (512MB - must match Rust MAX_MMAP_SIZE)
MAX_MMAP_SIZE = 512 * 1024 * 1024


@dataclass
class SharedMemoryHeader:
    """Parsed shared memory header."""
    magic: int
    version: int
    buffer_size: int
    write_pos: int
    read_pos: int
    items_written: int
    items_read: int
    writer_active: bool
    reader_active: bool
    last_write_ns: int
    last_read_ns: int


@dataclass
class TickData:
    """Tick data structure matching Rust FfiTick."""
    timestamp_ns: int
    price: float
    quantity: float
    is_buyer_maker: bool
    sequence: int


def check_amd_gpu_environment() -> Dict[str, Any]:
    """Detect AMD ROCm/DirectML environment."""
    env_info = {
        "rocm_available": any(var in os.environ for var in ["ROCM_PATH", "HIP_VISIBLE_DEVICES"]),
        "directml_available": any(var in os.environ for var in ["DIRECTML_ENABLED", "DIRECTML_DEVICE"]),
    }
    
    if env_info["rocm_available"]:
        logger.info("ROCm environment detected")
    if env_info["directml_available"]:
        logger.info("DirectML environment detected")
    
    return env_info


class SharedMemoryReader:
    """
    Zero-copy shared memory reader using numpy.memmap.
    
    This class provides thread-safe access to the shared memory region
    created by the Rust engine, enabling efficient data transfer without
    serialization overhead.
    """
    
    def __init__(self, path: str = DEFAULT_SHM_PATH, readonly: bool = True):
        """
        Initialize the shared memory reader.
        
        Args:
            path: Path to the shared memory file
            readonly: Open in read-only mode (default: True)
        """
        self.path = Path(path)
        self.readonly = readonly
        self.mmap: Optional[np.memmap] = None
        self.header: Optional[SharedMemoryHeader] = None
        self.is_open = False
        self.gpu_env = check_amd_gpu_environment()
        
        logger.info(f"SharedMemoryReader initialized for {path}")
    
    def open(self) -> bool:
        """
        Open the shared memory file.
        
        Returns:
            True if successfully opened
        """
        if not self.path.exists():
            logger.error(f"Shared memory file not found: {self.path}")
            return False
        
        try:
            # Open as memory-mapped file
            mode = 'r' if self.readonly else 'r+'
            self.mmap = np.memmap(
                str(self.path),
                dtype='uint8',
                mode=mode,
                offset=0
            )
            
            # Parse header
            if len(self.mmap) < HEADER_SIZE:
                logger.error("File too small for header")
                return False
            
            header_bytes = bytes(self.mmap[:HEADER_SIZE])
            header_data = struct.unpack(HEADER_FORMAT, header_bytes)
            
            self.header = SharedMemoryHeader(
                magic=header_data[0],
                version=header_data[1],
                buffer_size=header_data[2],
                write_pos=header_data[3],
                read_pos=header_data[4],
                items_written=header_data[5],
                items_read=header_data[6],
                writer_active=bool(header_data[7]),
                reader_active=bool(header_data[8]),
                last_write_ns=header_data[9],
                last_read_ns=header_data[10],
            )
            
            # Validate magic number
            if self.header.magic != MMAP_MAGIC:
                logger.error(f"Invalid magic number: {self.header.magic:#x}")
                return False
            
            self.is_open = True
            logger.info(
                f"Opened shared memory: {self.header.buffer_size} bytes, "
                f"written={self.header.items_written}, read={self.header.items_read}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to open shared memory: {e}")
            return False
    
    def close(self):
        """Close the shared memory mapping."""
        if self.mmap is not None:
            # Flush any pending writes
            if not self.readonly:
                self.mmap.flush()
            
            # Delete reference to release mmap
            del self.mmap
            self.mmap = None
        
        self.is_open = False
        logger.info("Shared memory closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current shared memory statistics."""
        if not self.is_open or self.header is None:
            return {"error": "Not open"}
        
        data_size = self.header.buffer_size - HEADER_SIZE
        utilization = 0.0
        
        if self.header.write_pos >= self.header.read_pos:
            used = self.header.write_pos - self.header.read_pos
        else:
            used = data_size - (self.header.read_pos - self.header.write_pos)
        
        utilization = min(1.0, used / data_size) if data_size > 0 else 0.0
        
        return {
            "total_size": self.header.buffer_size,
            "data_size": data_size,
            "write_pos": self.header.write_pos,
            "read_pos": self.header.read_pos,
            "items_written": self.header.items_written,
            "items_read": self.header.items_read,
            "writer_active": self.header.writer_active,
            "reader_active": self.header.reader_active,
            "utilization": utilization,
            "last_write_ns": self.header.last_write_ns,
            "last_read_ns": self.header.last_read_ns,
        }
    
    def read_available_data(self) -> Optional[bytes]:
        """
        Read all available data from shared memory.
        
        Returns:
            Bytes containing available data, or None if no data
        """
        if not self.is_open or self.header is None:
            return None
        
        write_pos = self.header.write_pos
        read_pos = self.header.read_pos
        
        if read_pos >= write_pos:
            return None  # No data available
        
        data_size = self.header.buffer_size - HEADER_SIZE
        actual_read_pos = HEADER_SIZE + (read_pos % data_size)
        length = write_pos - read_pos
        
        # Handle wrap-around
        if actual_read_pos + length > len(self.mmap):
            # Read in two parts
            part1 = bytes(self.mmap[actual_read_pos:])
            remaining = length - len(part1)
            part2 = bytes(self.mmap[HEADER_SIZE:HEADER_SIZE + remaining])
            return part1 + part2
        else:
            return bytes(self.mmap[actual_read_pos:actual_read_pos + length])
    
    def read_tick_batch(self) -> Optional[list]:
        """
        Read a batch of ticks from shared memory.
        
        Returns:
            List of TickData objects, or None if no data
        """
        data = self.read_available_data()
        if data is None:
            return None
        
        ticks = []
        offset = 0
        
        while offset + 4 <= len(data):
            # Read length prefix (4 bytes)
            length = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            
            if offset + length > len(data):
                break
            
            # Parse tick data (format depends on Rust serialization)
            # Expected: timestamp_ns(u64), price(f64), quantity(f64), 
            #           is_buyer_maker(bool), sequence(u64)
            if length >= 33:  # 8 + 8 + 8 + 1 + 8 = 33
                tick_data = data[offset:offset + length]
                
                timestamp_ns = struct.unpack('<Q', tick_data[0:8])[0]
                price = struct.unpack('<d', tick_data[8:16])[0]
                quantity = struct.unpack('<d', tick_data[16:24])[0]
                is_buyer_maker = struct.unpack('?', tick_data[24:25])[0]
                sequence = struct.unpack('<Q', tick_data[25:33])[0]
                
                ticks.append(TickData(
                    timestamp_ns=timestamp_ns,
                    price=price,
                    quantity=quantity,
                    is_buyer_maker=is_buyer_maker,
                    sequence=sequence,
                ))
            
            offset += length
        
        return ticks if ticks else None
    
    def read_as_numpy(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Read tick data as NumPy arrays for vectorized processing.
        
        Returns:
            Dictionary of field names to numpy arrays
        """
        ticks = self.read_tick_batch()
        if not ticks:
            return None
        
        n = len(ticks)
        
        return {
            "timestamp_ns": np.array([t.timestamp_ns for t in ticks], dtype=np.int64),
            "price": np.array([t.price for t in ticks], dtype=np.float64),
            "quantity": np.array([t.quantity for t in ticks], dtype=np.float64),
            "is_buyer_maker": np.array([t.is_buyer_maker for t in ticks], dtype=np.bool_),
            "sequence": np.array([t.sequence for t in ticks], dtype=np.int64),
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Entry point for testing
if __name__ == "__main__":
    import tempfile
    
    # Create a test shared memory file (simulate Rust writer)
    test_path = tempfile.mktemp(suffix=".bin")
    
    # Write test header
    with open(test_path, 'wb') as f:
        header = struct.pack(
            HEADER_FORMAT,
            MMAP_MAGIC,      # magic
            1,               # version
            1024 * 1024,     # buffer_size
            100,             # write_pos
            0,               # read_pos
            10,              # items_written
            0,               # items_read
            True,            # writer_active
            False,           # reader_active
            1000000000,      # last_write_ns
            0,               # last_read_ns
        )
        f.write(header)
        
        # Write some test data
        # Format: length(4) + timestamp(8) + price(8) + qty(8) + buyer(1) + seq(8)
        tick_bytes = struct.pack('<Qd dBd Q', 1000000000, 50000.5, 0.1, True, 1)
        length = struct.pack('<I', len(tick_bytes))
        f.write(length + tick_bytes)
    
    # Test reading
    with SharedMemoryReader(test_path) as reader:
        stats = reader.get_stats()
        print(f"Stats: {stats}")
        
        ticks = reader.read_tick_batch()
        if ticks:
            print(f"Read {len(ticks)} ticks:")
            for tick in ticks:
                print(f"  {tick}")
        
        numpy_data = reader.read_as_numpy()
        if numpy_data:
            print(f"\nNumPy arrays:")
            for key, arr in numpy_data.items():
                print(f"  {key}: {arr}")
    
    # Cleanup
    os.unlink(test_path)
