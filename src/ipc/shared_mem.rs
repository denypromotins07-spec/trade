//! Cross-Platform Memory-Mapped File (mmap) for Zero-Copy IPC
//! 
//! This module establishes a memory-mapped file architecture to enable
//! zero-copy data transfer between the Rust hot path and the Python
//! reinforcement learning agents. Optimized for the 8GB RAM limit.
//! 
//! Key Features:
//! - Memory-mapped files for shared memory IPC
//! - Configurable mmap sizes respecting 8GB global limit
//! - Atomic read/write pointers for thread synchronization
//! - Cross-platform support (Windows/Linux/macOS)

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

/// Default shared memory size (32MB)
const DEFAULT_MMAP_SIZE: usize = 32 * 1024 * 1024;

/// Maximum shared memory size (512MB - respecting 8GB total limit)
const MAX_MMAP_SIZE: usize = 512 * 1024 * 1024;

/// Header magic number for validation
const MMAP_MAGIC: u32 = 0x4E415654; // "NAVT" (Nautilus)

/// Shared memory header structure
#[repr(C)]
pub struct SharedMemoryHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Version of the protocol
    pub version: u32,
    /// Total buffer size
    pub buffer_size: u64,
    /// Write position (head)
    pub write_pos: AtomicU64,
    /// Read position (tail)
    pub read_pos: AtomicU64,
    /// Number of items written
    pub items_written: AtomicU64,
    /// Number of items read
    pub items_read: AtomicU64,
    /// Is writer active
    pub writer_active: AtomicBool,
    /// Is reader active
    pub reader_active: AtomicBool,
    /// Last write timestamp (nanoseconds)
    pub last_write_ns: AtomicU64,
    /// Last read timestamp (nanoseconds)
    pub last_read_ns: AtomicU64,
}

/// Memory-mapped shared memory region
pub struct SharedMemory {
    /// Path to the memory-mapped file
    path: String,
    /// Size of the mapped region
    size: usize,
    /// Raw pointer to mapped memory
    ptr: *mut u8,
    /// File handle (kept open for lifetime)
    file: Option<File>,
    /// Header reference
    header: *mut SharedMemoryHeader,
    /// Data region start
    data_start: usize,
    /// Is owner (created the file)
    is_owner: bool,
}

unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    /// Create a new shared memory region
    pub fn create(path: &str, size: usize) -> Result<Self, String> {
        if size < std::mem::size_of::<SharedMemoryHeader>() + 1024 {
            return Err("Size too small for header and data".to_string());
        }
        
        if size > MAX_MMAP_SIZE {
            return Err(format!(
                "Size {} exceeds maximum allowed {}",
                size, MAX_MMAP_SIZE
            ));
        }

        let path_str = path.to_string();
        
        // Create or open the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path_str)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        // Set file size
        file.set_len(size as u64)
            .map_err(|e| format!("Failed to set file size: {}", e))?;

        // Memory map the file
        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(|e| format!("Failed to mmap file: {}", e))?
        };

        // Initialize header
        let header_ptr = mmap.as_mut_ptr() as *mut SharedMemoryHeader;
        let data_start = std::mem::size_of::<SharedMemoryHeader>();

        unsafe {
            std::ptr::write(
                header_ptr,
                SharedMemoryHeader {
                    magic: MMAP_MAGIC,
                    version: 1,
                    buffer_size: size as u64,
                    write_pos: AtomicU64::new(0),
                    read_pos: AtomicU64::new(0),
                    items_written: AtomicU64::new(0),
                    items_read: AtomicU64::new(0),
                    writer_active: AtomicBool::new(true),
                    reader_active: AtomicBool::new(false),
                    last_write_ns: AtomicU64::new(0),
                    last_read_ns: AtomicU64::new(0),
                },
            );
        }

        // Leak the mmap to keep it alive (we manage lifetime manually)
        std::mem::forget(mmap);

        Ok(Self {
            path: path_str,
            size,
            ptr: mmap.as_mut_ptr(),
            file: Some(file),
            header: header_ptr,
            data_start,
            is_owner: true,
        })
    }

    /// Open an existing shared memory region
    pub fn open(path: &str) -> Result<Self, String> {
        let path_str = path.to_string();
        
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path_str)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        let mmap = unsafe {
            memmap2::MmapMut::map_mut(&file)
                .map_err(|e| format!("Failed to mmap file: {}", e))?
        };

        let size = mmap.len();
        let header_ptr = mmap.as_mut_ptr() as *mut SharedMemoryHeader;
        let data_start = std::mem::size_of::<SharedMemoryHeader>();

        // Validate header
        unsafe {
            if (*header_ptr).magic != MMAP_MAGIC {
                return Err("Invalid shared memory magic number".to_string());
            }
        }

        std::mem::forget(mmap);

        Ok(Self {
            path: path_str,
            size,
            ptr: mmap.as_mut_ptr(),
            file: Some(file),
            header: header_ptr,
            data_start,
            is_owner: false,
        })
    }

    /// Get available write space
    pub fn available_write_space(&self) -> usize {
        unsafe {
            let write_pos = (*self.header).write_pos.load(Ordering::Acquire);
            let read_pos = (*self.header).read_pos.load(Ordering::Acquire);
            
            let data_size = self.size - self.data_start;
            
            if write_pos >= read_pos {
                data_size - (write_pos - read_pos) as usize
            } else {
                (read_pos - write_pos) as usize
            }
        }
    }

    /// Write data to shared memory
    pub fn write(&self, data: &[u8]) -> Result<usize, String> {
        let available = self.available_write_space();
        
        if data.len() > available {
            return Err(format!(
                "Not enough space: need {}, available {}",
                data.len(),
                available
            ));
        }

        unsafe {
            let write_pos = (*self.header).write_pos.load(Ordering::Acquire);
            let data_size = self.size - self.data_start;
            
            // Calculate actual write position (wrap around)
            let actual_pos = self.data_start + (write_pos as usize % data_size);
            
            // Write data length first (4 bytes)
            let len_bytes = (data.len() as u32).to_ne_bytes();
            std::ptr::copy_nonoverlapping(
                len_bytes.as_ptr(),
                self.ptr.add(actual_pos),
                4,
            );
            
            // Write data
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.ptr.add(actual_pos + 4),
                data.len(),
            );
            
            // Update write position
            let new_pos = write_pos + (data.len() as u64) + 4;
            (*self.header).write_pos.store(new_pos, Ordering::Release);
            (*self.header).items_written.fetch_add(1, Ordering::Relaxed);
            
            // Update timestamp
            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            (*self.header).last_write_ns.store(now_ns, Ordering::Relaxed);
        }

        Ok(data.len())
    }

    /// Read data from shared memory
    pub fn read(&self) -> Result<Vec<u8>, String> {
        unsafe {
            let write_pos = (*self.header).write_pos.load(Ordering::Acquire);
            let read_pos = (*self.header).read_pos.load(Ordering::Acquire);
            
            if read_pos >= write_pos {
                return Err("No data available".to_string());
            }
            
            let data_size = self.size - self.data_start;
            let actual_pos = self.data_start + (read_pos as usize % data_size);
            
            // Read data length
            let mut len_bytes = [0u8; 4];
            std::ptr::copy_nonoverlapping(
                self.ptr.add(actual_pos),
                len_bytes.as_mut_ptr(),
                4,
            );
            let data_len = u32::from_ne_bytes(len_bytes) as usize;
            
            // Read data
            let mut data = vec![0u8; data_len];
            std::ptr::copy_nonoverlapping(
                self.ptr.add(actual_pos + 4),
                data.as_mut_ptr(),
                data_len,
            );
            
            // Update read position
            let new_pos = read_pos + (data_len as u64) + 4;
            (*self.header).read_pos.store(new_pos, Ordering::Release);
            (*self.header).items_read.fetch_add(1, Ordering::Relaxed);
            
            // Update timestamp
            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            (*self.header).last_read_ns.store(now_ns, Ordering::Relaxed);
            
            Ok(data)
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> SharedMemoryStats {
        unsafe {
            SharedMemoryStats {
                total_size: self.size,
                data_size: self.size - self.data_start,
                write_pos: (*self.header).write_pos.load(Ordering::Acquire),
                read_pos: (*self.header).read_pos.load(Ordering::Acquire),
                items_written: (*self.header).items_written.load(Ordering::Acquire),
                items_read: (*self.header).items_read.load(Ordering::Acquire),
                writer_active: (*self.header).writer_active.load(Ordering::Acquire),
                reader_active: (*self.header).reader_active.load(Ordering::Acquire),
                utilization: self.calculate_utilization(),
            }
        }
    }

    /// Calculate buffer utilization (0.0 to 1.0)
    fn calculate_utilization(&self) -> f64 {
        unsafe {
            let write_pos = (*self.header).write_pos.load(Ordering::Acquire);
            let read_pos = (*self.header).read_pos.load(Ordering::Acquire);
            let data_size = (self.size - self.data_start) as u64;
            
            if write_pos >= read_pos {
                (write_pos - read_pos) as f64 / data_size as f64
            } else {
                1.0 - ((read_pos - write_pos) as f64 / data_size as f64)
            }
        }
    }

    /// Close and cleanup
    pub fn close(&mut self) {
        if self.is_owner {
            unsafe {
                (*self.header).writer_active.store(false, Ordering::Release);
            }
            
            // Try to remove the file
            let _ = std::fs::remove_file(&self.path);
        }
        
        // Re-create mmap to allow proper drop
        // (In production: use proper RAII with Box<MmapMut>)
    }
}

/// Statistics for shared memory region
#[derive(Debug, Clone)]
pub struct SharedMemoryStats {
    pub total_size: usize,
    pub data_size: usize,
    pub write_pos: u64,
    pub read_pos: u64,
    pub items_written: u64,
    pub items_read: u64,
    pub writer_active: bool,
    pub reader_active: bool,
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_open() {
        let path = "/tmp/test_shared_memory.bin";
        
        // Create
        let mut shm = SharedMemory::create(path, 1024 * 1024).unwrap();
        
        // Verify stats
        let stats = shm.get_stats();
        assert_eq!(stats.total_size, 1024 * 1024);
        assert!(stats.writer_active);
        assert!(!stats.reader_active);
        
        // Cleanup
        shm.close();
    }

    #[test]
    fn test_write_and_read() {
        let path = "/tmp/test_shared_memory_rw.bin";
        
        let mut shm = SharedMemory::create(path, 1024 * 1024).unwrap();
        
        // Write data
        let test_data = b"Hello, Shared Memory!";
        let written = shm.write(test_data).unwrap();
        assert_eq!(written, test_data.len());
        
        // Read data
        let read_data = shm.read().unwrap();
        assert_eq!(read_data, test_data);
        
        shm.close();
    }
}
