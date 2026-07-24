// =============================================================================
// Nautilus/Ray Bot - Stage 53: AOT Loader
// File: src/boot/aot_loader.rs
// Purpose: Memory-map pre-compiled ONNX and RL weight binaries directly into
//          execution space, bypassing standard filesystem read overhead.
// Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
// Constraints: 8GB RAM Limit, GPU VRAM pinned during load
// =============================================================================

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum allowed model size (enforced to stay within 8GB total limit)
const MAX_MODEL_SIZE_MB: usize = 512;

/// Descriptor for a loaded model
#[derive(Debug, Clone)]
pub struct LoadedModel {
    pub name: String,
    pub path: PathBuf,
    pub base_ptr: *const u8,
    pub size: usize,
    pub is_mapped: bool,
}

unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

/// Ahead-Of-Time Model Loader
pub struct AotLoader {
    /// Directory containing pre-compiled weights
    weights_dir: PathBuf,
    /// Currently loaded models
    loaded_models: Vec<LoadedModel>,
    /// Total memory mapped by loader
    total_mapped_bytes: AtomicUsize,
    /// Flag indicating if GPU VRAM is locked
    gpu_vram_locked: AtomicBool,
}

impl AotLoader {
    pub fn new(weights_dir: &str) -> Self {
        Self {
            weights_dir: PathBuf::from(weights_dir),
            loaded_models: Vec::with_capacity(8),
            total_mapped_bytes: AtomicUsize::new(0),
            gpu_vram_locked: AtomicBool::new(false),
        }
    }

    /// Initialize the loader and lock GPU VRAM
    pub fn initialize(&mut self) -> Result<(), String> {
        log::info!("Initializing AOT Loader...");
        
        if !self.weights_dir.exists() {
            return Err(format!("Weights directory does not exist: {:?}", self.weights_dir));
        }

        // Lock GPU VRAM for DirectML/ROCm before loading
        self.lock_gpu_vram()?;
        
        log::info!("AOT Loader initialized. GPU VRAM locked.");
        Ok(())
    }

    /// Lock GPU VRAM to prevent paging during inference
    fn lock_gpu_vram(&self) -> Result<(), String> {
        log::info!("Locking GPU VRAM for DirectML/ROCm...");
        
        #[cfg(target_os = "windows")]
        {
            // In production, this would call into DirectML or AMD ROCm APIs
            // to pin the VRAM allocation.
            // Example: DmlCreateDevice with DML_CREATE_DEVICE_FLAG_DEBUG_DISABLED
            // and subsequent buffer allocations with DML_BUFFER_TYPE_MEMORY
            
            log::debug!("Simulating GPU VRAM lock via Windows API...");
            
            // Placeholder: Actual implementation requires FFI to D3D12/DirectML
            // For now, we just set the flag
        }
        
        self.gpu_vram_locked.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Load a single model file via memory mapping
    pub fn load_model(&mut self, model_name: &str) -> Result<&LoadedModel, String> {
        let model_path = self.weights_dir.join(format!("{}.bin", model_name));
        
        if !model_path.exists() {
            return Err(format!("Model file not found: {:?}", model_path));
        }

        // Get file size
        let metadata = std::fs::metadata(&model_path)
            .map_err(|e| format!("Failed to read metadata: {}", e))?;
        
        let file_size = metadata.len() as usize;
        
        // Enforce size limit
        if file_size > MAX_MODEL_SIZE_MB * 1024 * 1024 {
            return Err(format!(
                "Model {} exceeds maximum size limit ({} MB > {} MB)",
                model_name,
                file_size / (1024 * 1024),
                MAX_MODEL_SIZE_MB
            ));
        }

        // Check total memory budget
        let current_total = self.total_mapped_bytes.load(Ordering::Relaxed);
        if current_total + file_size > (4 * 1024 * 1024 * 1024) {
            return Err("Loading this model would exceed the 4GB Rust memory quota".to_string());
        }

        // Memory map the file
        let ptr = self.memory_map_file(&model_path, file_size)?;
        
        let model = LoadedModel {
            name: model_name.to_string(),
            path: model_path.clone(),
            base_ptr: ptr,
            size: file_size,
            is_mapped: true,
        };

        self.total_mapped_bytes.fetch_add(file_size, Ordering::Relaxed);
        self.loaded_models.push(model);
        
        let loaded = self.loaded_models.last().unwrap();
        log::info!("Loaded model '{}' ({} MB) at {:p}", 
                   model_name, file_size / (1024 * 1024), loaded.base_ptr);
        
        Ok(loaded)
    }

    /// Memory map a file into read-only address space
    fn memory_map_file(&self, path: &Path, size: usize) -> Result<*const u8, String> {
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::System::Memory::*;
            use windows::Win32::Foundation::*;
            use windows::Win32::Storage::FileSystem::*;

            unsafe {
                // Open file
                let file = CreateFileW(
                    &windows::core::HSTRING::from(path.as_os_str().to_str().unwrap()),
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    None,
                    OPEN_EXISTING,
                    FILE_ATTRIBUTE_READONLY,
                    HANDLE::default(),
                )?;

                // Create file mapping
                let mapping = CreateFileMappingW(
                    file,
                    None,
                    PAGE_READONLY,
                    0,
                    0,
                    Some(&windows::core::HSTRING::from("NautilusModelMap")),
                )?;

                // Map view
                let ptr = MapViewOfFile(
                    mapping,
                    FILE_MAP_READ,
                    0,
                    0,
                    size,
                );

                if ptr.is_null() {
                    return Err("Failed to map view of file".to_string());
                }

                log::debug!("Memory mapped {} at {:p}", path.display(), ptr);
                Ok(ptr as *const u8)
            }
        }

        #[cfg(target_os = "linux")]
        {
            use libc::{mmap, MAP_PRIVATE, MAP_FAILED, PROT_READ};
            use std::os::unix::io::AsRawFd;
            
            let file = File::open(path)?;
            let fd = file.as_raw_fd();
            
            unsafe {
                let ptr = mmap(
                    std::ptr::null_mut(),
                    size,
                    PROT_READ,
                    MAP_PRIVATE,
                    fd,
                    0,
                );
                
                if ptr == MAP_FAILED {
                    return Err("Failed to mmap file".to_string());
                }
                
                Ok(ptr as *const u8)
            }
        }
    }

    /// Get pointer to loaded model data
    pub fn get_model_data(&self, model_name: &str) -> Option<(*const u8, usize)> {
        for model in &self.loaded_models {
            if model.name == model_name {
                return Some((model.base_ptr, model.size));
            }
        }
        None
    }

    /// Unload all models and release mappings
    pub fn unload_all(&mut self) {
        log::warn!("Unloading all AOT models...");
        
        for model in &self.loaded_models {
            if model.is_mapped && !model.base_ptr.is_null() {
                self.unmap_memory(model.base_ptr, model.size);
            }
        }
        
        self.loaded_models.clear();
        self.total_mapped_bytes.store(0, Ordering::Relaxed);
        self.gpu_vram_locked.store(false, Ordering::SeqCst);
        
        log::info!("All models unloaded. GPU VRAM unlocked.");
    }

    /// Unmap memory region
    fn unmap_memory(&self, ptr: *const u8, size: usize) {
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::System::Memory::UnmapViewOfFile;
            unsafe {
                let _ = UnmapViewOfFile(ptr as *const _);
            }
        }

        #[cfg(target_os = "linux")]
        {
            use libc::{munmap};
            unsafe {
                let _ = munmap(ptr as *mut _, size);
            }
        }
        
        log::debug!("Unmapped {} bytes at {:p}", size, ptr);
    }
}

impl Drop for AotLoader {
    fn drop(&mut self) {
        self.unload_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let loader = AotLoader::new("/tmp/weights");
        assert!(!loader.gpu_vram_locked.load(Ordering::SeqCst));
        assert_eq!(loader.total_mapped_bytes.load(Ordering::Relaxed), 0);
    }
}
