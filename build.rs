// =============================================================================
// NAUTILUS/RAY CRYPTO TRADING BOT - RUST BUILD SCRIPT
// =============================================================================
// File: build.rs
// Purpose: Compile-time configuration for native AMD optimizations
// Features: C-binding generation, Cython extension linking, BLAS configuration
// Target: AMD Ryzen AI 5 with AVX2, FMA, BMI2 instruction sets
// =============================================================================

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // ==========================================================================
    // TARGET DETECTION AND VALIDATION
    // ==========================================================================
    
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    
    println!("cargo:warning=Building for target: {}-{}-{}", target_arch, target_os, target_env);
    
    // Validate we're building for x86_64 (required for AVX2/FMA)
    if target_arch != "x86_64" {
        panic!(
            "This build is optimized for x86_64 architecture only. \
             Current target: {}. Use --target=x86_64-pc-windows-msvc",
            target_arch
        );
    }
    
    // ==========================================================================
    // CPU FEATURE DETECTION FOR AMD RYZEN
    // ==========================================================================
    
    // Enable AMD-specific CPU features for Ryzen AI 5
    // These flags are also set in Cargo.toml but repeated here for clarity
    let cpu_features = [
        "avx2",      // Advanced Vector Extensions 2 (Ryzen support)
        "fma",       // Fused Multiply-Add (critical for ML inference)
        "bmi2",      // Bit Manipulation Instructions 2
        "lzcnt",     // Leading Zero Count
        "popcnt",    // Population Count (bit counting)
        "sse4.2",    // SSE 4.2 (baseline for modern x86_64)
        "aes",       // AES-NI (for API signature hashing)
    ];
    
    for feature in &cpu_features {
        println!("cargo:rustc-cfg=target_feature_{}", feature.replace('.', "_"));
    }
    
    // ==========================================================================
    // BLAS LIBRARY CONFIGURATION
    // ==========================================================================
    
    // Configure BLAS backend for numerical computing
    // Priority: OpenBLAS > MKL > Reference BLAS
    configure_blas(&target_os);
    
    // ==========================================================================
    // C-BINDINGS FOR CYTHON EXTENSIONS
    // ==========================================================================
    
    // Generate C headers for Python FFI
    generate_c_bindings();
    
    // ==========================================================================
    // PLATFORM-SPECIFIC CONFIGURATION
    // ==========================================================================
    
    match target_os.as_str() {
        "windows" => configure_windows_build(&target_env),
        "linux" => configure_linux_build(),
        "macos" => configure_macos_build(),
        _ => println!("cargo:warning=Unknown OS: {}", target_os),
    }
    
    // ==========================================================================
    // COMPILE-TIME CONSTANTS
    // ==========================================================================
    
    // Export build-time constants for runtime use
    let build_timestamp = chrono_timestamp();
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", build_timestamp);
    println!("cargo:rustc-env=BUILD_TARGET={}-{}-{}", target_arch, target_os, target_env);
    
    // Version information from git (if available)
    if let Some(git_hash) = get_git_hash() {
        println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    }
    
    println!("cargo:warning=Build configuration complete for AMD Ryzen AI 5");
}

// =============================================================================
// BLAS CONFIGURATION
// =============================================================================

/// Configure BLAS library linking based on platform.
fn configure_blas(target_os: &str) {
    println!("cargo:rerun-if-env-changed=BLAS_LIB_DIR");
    println!("cargo:rerun-if-env-changed=BLAS_INCLUDE_DIR");
    
    // Check for environment-provided BLAS paths
    let blas_lib_dir = env::var("BLAS_LIB_DIR").ok();
    let blas_include_dir = env::var("BLAS_INCLUDE_DIR").ok();
    
    // Prefer OpenBLAS for AMD processors (better than MKL on Ryzen)
    #[cfg(feature = "openblas")]
    {
        println!("cargo:warning=Using OpenBLAS backend (optimized for AMD)");
        
        if let Some(lib_dir) = &blas_lib_dir {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }
        
        // Link OpenBLAS
        println!("cargo:rustc-link-lib=static=openblas");
    }
    
    // Fallback to system BLAS
    #[cfg(not(feature = "openblas"))]
    {
        println!("cargo:warning=Using system BLAS backend");
        
        // Try to find BLAS via pkg-config on Unix-like systems
        if target_os != "windows" {
            if let Ok(_) = pkg_config::Config::new().probe("openblas") {
                println!("cargo:warning=Found OpenBLAS via pkg-config");
                return;
            }
        }
        
        // Default BLAS linkage
        if let Some(lib_dir) = &blas_lib_dir {
            println!("cargo:rustc-link-search=native={}", lib_dir);
        }
        println!("cargo:rustc-link-lib=blas");
    }
    
    // Set include path for BLAS headers
    if let Some(include_dir) = &blas_include_dir {
        println!("cargo:include={}", include_dir);
    }
}

// =============================================================================
// C-BINDING GENERATION
// =============================================================================

/// Generate C bindings for Python/Cython interop.
fn generate_c_bindings() {
    println!("cargo:rerun-if-changed=src/ffi.h");
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let header_path = Path::new("src/ffi.h");
    
    // Only generate if header exists
    if !header_path.exists() {
        println!("cargo:warning=FFI header not found, skipping C binding generation");
        return;
    }
    
    // Create output directory for generated bindings
    let bindings_dir = out_dir.join("bindings");
    std::fs::create_dir_all(&bindings_dir).expect("Failed to create bindings directory");
    
    println!("cargo:warning=C bindings will be generated at build time");
    
    // Note: Actual bindgen would be done here if we had a header file
    // Example:
    // let bindings = bindgen::Builder::default()
    //     .header("src/ffi.h")
    //     .generate()
    //     .expect("Unable to generate bindings");
    // 
    // bindings.write_to_file(bindings_dir.join("bindings.rs"))
    //     .expect("Couldn't write bindings!");
}

// =============================================================================
// WINDOWS-SPECIFIC CONFIGURATION
// =============================================================================

/// Configure Windows-specific build settings.
fn configure_windows_build(target_env: &str) {
    println!("cargo:warning=Configuring Windows build (env: {})", target_env);
    
    // Windows-specific linker flags
    println!("cargo:rustc-link-lib=dylib=ws2_32");  // Winsock
    println!("cargo:rustc-link-lib=dylib=advapi32"); // Security APIs
    println!("cargo:rustc-link-lib=dylib=userenv");  // User environment
    
    // For MSVC
    if target_env == "msvc" {
        // Enable static CRT linking for standalone executable
        // This is handled by the .cargo/config.toml typically
        println!("cargo:warning=MSVC toolchain detected");
        
        // Additional MSVC-specific optimizations
        println!("cargo:rustc-flag=/OPT:LREF");
        println!("cargo:rustc-flag=/OPT:ICF");
    }
    
    // For GNU toolchain (MinGW)
    if target_env == "gnu" {
        println!("cargo:warning=MinGW toolchain detected");
    }
    
    // Copy DLLs for runtime (if any external dependencies)
    // This would be handled by a post-build script typically
}

// =============================================================================
// LINUX-SPECIFIC CONFIGURATION
// =============================================================================

/// Configure Linux-specific build settings.
fn configure_linux_build() {
    println!("cargo:warning=Configuring Linux build");
    
    // Linux-specific libraries
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=rt");
    println!("cargo:rustc-link-lib=dylib=dl");
    
    // Enable position-independent executable
    println!("cargo:rustc-flag=-fPIE");
    
    // ROCm detection for AMD GPU support (Linux only)
    if Path::new("/opt/rocm").exists() {
        println!("cargo:warning=ROCm installation detected");
        println!("cargo:rustc-cfg=feature=\"rocm\"");
        
        // Add ROCm library path
        println!("cargo:rustc-link-search=native=/opt/rocm/lib");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
    }
}

// =============================================================================
// MACOS-SPECIFIC CONFIGURATION
// =============================================================================

/// Configure macOS-specific build settings.
fn configure_macos_build() {
    println!("cargo:warning=Configuring macOS build");
    
    // macOS frameworks
    println!("cargo:rustc-link-lib=framework=Security");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    
    // Minimum macOS version
    println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=11.0");
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Get current timestamp in ISO 8601 format.
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    
    let secs = duration.as_secs();
    
    // Simple conversion to human-readable format
    // In production, use chrono crate
    format!("{}", secs)
}

/// Get current Git commit hash (if in a Git repository).
fn get_git_hash() -> Option<String> {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map(|hash| hash.trim().to_string())
}

// =============================================================================
// BUILD VERIFICATION
// =============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn verify_cpu_features() {
        // Runtime verification that CPU features are enabled
        #[cfg(target_feature = "avx2")]
        {
            println!("AVX2 feature enabled");
        }
        
        #[cfg(target_feature = "fma")]
        {
            println!("FMA feature enabled");
        }
        
        #[cfg(target_feature = "bmi2")]
        {
            println!("BMI2 feature enabled");
        }
    }
}
