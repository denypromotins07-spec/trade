# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - PYTHON WHEEL BUILDER
# =============================================================================
# Stage 54: Automated Wheel Builder for Python AI Modules
# Target: AMD Ryzen AI 5 with ROCm/DirectML GPU offloading
# Memory Constraint: 4GB Python RAM quota during C-extension compilation
# Purpose: Pre-compile Cython/Numba extensions, cache binaries, eliminate JIT overhead
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("cython", "numba", "pyo3", "all")]
    [string]$BuildType = "all",
    
    [Parameter(Mandatory = $false)]
    [int]$RamLimitMB = 4096,
    
    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "dist/wheels",
    
    [Parameter(Mandatory = $false)]
    [switch]$UseROCm,
    
    [Parameter(Mandatory = $false)]
    [switch]$UseDirectML,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_ROOT
$PYTHON_DIR = Join-Path $PROJECT_ROOT "python"
$WHEEL_OUTPUT = Join-Path $PROJECT_ROOT $OutputDir
$CACHE_DIR = Join-Path $PROJECT_ROOT ".build_cache"
$LOG_FILE = Join-Path $PROJECT_ROOT "logs/wheel_build_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"

# AMD GPU configuration
$ROCM_PATH = $env:ROCM_PATH ?? "C:\Program Files\AMD\ROCm"
$DIRECTML_PATH = $env:DIRECTML_PATH ?? "C:\Program Files\DirectML"

# Memory limit enforcement
$MEMORY_LIMIT_BYTES = $RamLimitMB * 1MB
$MAX_PARALLEL_JOBS = [math]::Max(1, [math]::Floor($RamLimitMB / 1024))  # 1 job per GB RAM

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "WARN") { "Yellow" } else { "Green" })
    if ($Verbose) {
        Add-Content -Path $LOG_FILE -Value $logEntry
    }
}

function Test-PythonEnvironment {
    Write-Log "Validating Python build environment..."
    
    # Check Python version
    $pythonVersion = python --version 2>&1
    if (-not $pythonVersion) {
        throw "Python not found. Please install Python 3.11 or 3.12."
    }
    Write-Log "Python version: $pythonVersion"
    
    # Verify Python version is 3.11 or 3.12
    $versionMatch = $pythonVersion -match 'Python\s+(\d+)\.(\d+)'
    if ($versionMatch) {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -ne 3 -or ($minor -lt 11 -or $minor -gt 12)) {
            throw "Python 3.11 or 3.12 required. Found: $major.$minor"
        }
    }
    
    # Check for required build tools
    $requiredTools = @("pip", "wheel", "cython", "numba")
    foreach ($tool in $requiredTools) {
        $toolCheck = python -c "import $tool; print($tool.__version__)" 2>$null
        if (-not $toolCheck) {
            Write-Log "WARNING: $tool not found or not importable" -Level "WARN"
        } else {
            Write-Log "$tool version: $toolCheck"
        }
    }
    
    # Check virtual environment
    if ($env:VIRTUAL_ENV) {
        Write-Log "Virtual environment: $($env:VIRTUAL_ENV)" -ForegroundColor Green
    } else {
        Write-Log "WARNING: Not running in a virtual environment" -Level "WARN"
    }
    
    return $true
}

function Test-GPUEnvironment {
    Write-Log "Checking GPU acceleration environment..."
    
    $gpuConfig = @{
        ROCmAvailable = $false
        DirectMLAvailable = $false
        CUDA Available = $false
    }
    
    # Check AMD ROCm (Linux primary, Windows via WSL)
    if (Test-Path $ROCM_PATH) {
        $gpuConfig.ROCmAvailable = $true
        Write-Log "AMD ROCm found at: $ROCM_PATH" -ForegroundColor Green
        
        # Check ROCm version
        $rocmVersion = Get-ItemProperty -Path "HKLM:\SOFTWARE\AMD\ROCm" -ErrorAction SilentlyContinue
        if ($rocmVersion) {
            Write-Log "ROCm version: $($rocmVersion.Version)"
        }
    } elseif ($UseROCm) {
        Write-Log "WARNING: ROCm requested but not found at $ROCM_PATH" -Level "WARN"
    }
    
    # Check AMD DirectML (Windows native)
    if (Test-Path $DIRECTML_PATH) {
        $gpuConfig.DirectMLAvailable = $true
        Write-Log "AMD DirectML found at: $DIRECTML_PATH" -ForegroundColor Green
    } elseif ($UseDirectML) {
        Write-Log "WARNING: DirectML requested but not found at $DIRECTML_PATH" -Level "WARN"
    }
    
    # Check NVIDIA CUDA (fallback)
    $cudaPath = $env:CUDA_PATH ?? "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaPath) {
        $gpuConfig.CUDAAvailable = $true
        Write-Log "NVIDIA CUDA found at: $cudaPath" -ForegroundColor Yellow
    }
    
    # Set environment variables based on availability
    if ($gpuConfig.DirectMLAvailable -or $UseDirectML) {
        $env:ONNXRUNTIME_PROVIDER = "directml"
        $env:DML_DISABLE_FASTGEO = "0"
        Write-Log "Configured for DirectML acceleration"
    }
    
    if ($gpuConfig.ROCmAvailable -or $UseROCm) {
        $env:HSA_OVERRIDE_GFX_VERSION = "11.0.0"  # AMD Ryzen AI 5 GFX version
        $env:ROCM_VISIBLE_DEVICES = "0"
        Write-Log "Configured for ROCm acceleration"
    }
    
    return $gpuConfig
}

function Set-MemoryLimit {
    param([int]$LimitMB)
    
    Write-Log "Enforcing Python memory limit: ${LimitMB}MB"
    
    # Set Python-specific memory limits
    $env:PYTHONMALLOC = "malloc"  # Use system malloc for better control
    $env:MALLOC_ARENA_MAX = "2"   # Limit glibc malloc arenas
    
    # For Numba, limit parallel threads based on RAM
    $numThreads = [math]::Min([Environment]::ProcessorCount, [math]::Floor($LimitMB / 512))
    $env:NUMBA_NUM_THREADS = $numThreads.ToString()
    
    # For Ray, limit object store memory
    $rayMemoryLimit = [math]::Floor($LimitMB * 0.5)  # 50% of limit for Ray
    $env:RAY_OBJECT_STORE_MEMORY = $rayMemoryLimit.ToString()
    
    Write-Log "NUMBA_NUM_THREADS set to: $numThreads"
    Write-Log "Ray object store memory limit: ${rayMemoryLimit}MB"
}

function Invoke-CythonBuild {
    Write-Log "=== Building Cython Extensions ===" -ForegroundColor Cyan
    
    $cythonDir = Join-Path $PYTHON_DIR "cython_modules"
    if (-not (Test-Path $cythonDir)) {
        Write-Log "No Cython modules found at $cythonDir. Skipping..." -Level "WARN"
        return
    }
    
    # Find all .pyx files
    $pyxFiles = Get-ChildItem -Path $cythonDir -Filter "*.pyx" -Recurse
    Write-Log "Found $($pyxFiles.Count) Cython source files"
    
    # Build each Cython module with memory constraints
    $buildArgs = @(
        "setup.py",
        "build_ext",
        "--inplace",
        "--parallel=$MAX_PARALLEL_JOBS"
    )
    
    # Set compiler optimization flags for AMD Ryzen
    $env:CFLAGS = "-O3 -march=native -mtune=native"
    $env:CXXFLAGS = "-O3 -march=native -mtune=native"
    
    Write-Log "Compiling Cython extensions with $MAX_PARALLEL_JOBS parallel jobs..."
    
    Push-Location $cythonDir
    try {
        python @buildArgs 2>&1 | ForEach-Object { Write-Log $_ }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Cython build failed with exit code $LASTEXITCODE"
        }
        
        Write-Log "Cython extensions compiled successfully" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

function Invoke-NumbaBuild {
    Write-Log "=== Pre-compiling Numba Functions ===" -ForegroundColor Cyan
    
    $numbaDir = Join-Path $PYTHON_DIR "numba_modules"
    if (-not (Test-Path $numbaDir)) {
        Write-Log "No Numba modules found at $numbaDir. Skipping..." -Level "WARN"
        return
    }
    
    # Create cache directory for Numba compiled functions
    $numbaCacheDir = Join-Path $CACHE_DIR "numba_cache"
    if (-not (Test-Path $numbaCacheDir)) {
        New-Item -ItemType Directory -Force -Path $numbaCacheDir | Out-Null
    }
    
    $env:NUMBA_CACHE_DIR = $numbaCacheDir
    
    # Run the Numba pre-compilation script
    $compileScript = Join-Path $numbaDir "compile_all.py"
    if (Test-Path $compileScript) {
        Write-Log "Running Numba pre-compilation script..."
        python $compileScript 2>&1 | ForEach-Object { Write-Log $_ }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Numba pre-compilation failed"
        }
        
        Write-Log "Numba functions pre-compiled to: $numbaCacheDir" -ForegroundColor Green
    } else {
        Write-Log "No pre-compilation script found. Numba will JIT at runtime." -Level "WARN"
    }
}

function Invoke-PyO3Build {
    Write-Log "=== Building PyO3 Rust Extensions ===" -ForegroundColor Cyan
    
    $pyo3Dir = Join-Path $PYTHON_DIR "ffi"
    if (-not (Test-Path $pyo3Dir)) {
        Write-Log "No PyO3 modules found at $pyo3Dir. Skipping..." -Level "WARN"
        return
    }
    
    # Check for maturin
    $maturinVersion = maturin --version 2>$null
    if (-not $maturinVersion) {
        throw "maturin not found. Install with: pip install maturin"
    }
    Write-Log "Maturin version: $maturinVersion"
    
    # Build with PGO profiles if available
    $pgoProfile = Join-Path $PROJECT_ROOT "target/pgo/profiles/merged.profdata"
    $maturinArgs = @("build", "--release", "--out", $WHEEL_OUTPUT)
    
    if (Test-Path $pgoProfile) {
        Write-Log "Using PGO profile: $pgoProfile"
        $env:RUSTFLAGS = "-Cprofile-use=$pgoProfile"
    }
    
    Push-Location $pyo3Dir
    try {
        Write-Log "Building PyO3 extension with maturin..."
        maturin @maturinArgs 2>&1 | ForEach-Object { Write-Log $_ }
        
        if ($LASTEXITCODE -ne 0) {
            throw "PyO3 build failed with exit code $LASTEXITCODE"
        }
        
        Write-Log "PyO3 extension built successfully" -ForegroundColor Green
        
        # List generated wheels
        $wheels = Get-ChildItem -Path $WHEEL_OUTPUT -Filter "*.whl"
        if ($wheels) {
            Write-Log "Generated wheels:" -ForegroundColor Green
            $wheels | ForEach-Object { Write-Log "  - $($_.Name) ($([math]::Round($_.Length / 1MB, 2))MB)" }
        }
    } finally {
        Pop-Location
    }
}

function Invoke-WheelPackage {
    Write-Log "=== Packaging Python Wheels ===" -ForegroundColor Cyan
    
    # Ensure output directory exists
    if (-not (Test-Path $WHEEL_OUTPUT)) {
        New-Item -ItemType Directory -Force -Path $WHEEL_OUTPUT | Out-Null
    }
    
    # Build wheel for the main Python package
    Push-Location $PROJECT_ROOT
    try {
        Write-Log "Building main package wheel..."
        
        $wheelArgs = @(
            "-m", "build",
            "--wheel",
            "--outdir", $WHEEL_OUTPUT,
            "--no-isolation"  # Use existing environment
        )
        
        python @wheelArgs 2>&1 | ForEach-Object { Write-Log $_ }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Wheel packaging failed"
        }
        
        Write-Log "Wheels packaged successfully" -ForegroundColor Green
    } finally {
        Pop-Location
    }
}

function Invoke-FullBuild {
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "NAUTILUS/RAY PYTHON WHEEL BUILD" -ForegroundColor Cyan
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "Target: AMD Ryzen AI 5"
    Write-Log "RAM Limit: ${RamLimitMB}MB"
    Write-Log "Parallel Jobs: $MAX_PARALLEL_JOBS"
    Write-Log "Output: $WHEEL_OUTPUT"
    Write-Log ""
    
    try {
        # Phase 0: Environment validation
        Test-PythonEnvironment
        $gpuConfig = Test-GPUEnvironment
        
        # Apply memory limits
        Set-MemoryLimit -LimitMB $RamLimitMB
        
        # Ensure directories exist
        if (-not (Test-Path $CACHE_DIR)) {
            New-Item -ItemType Directory -Force -Path $CACHE_DIR | Out-Null
        }
        if (-not (Test-Path $WHEEL_OUTPUT)) {
            New-Item -ItemType Directory -Force -Path $WHEEL_OUTPUT | Out-Null
        }
        if (-not (Test-Path (Split-Path $LOG_FILE -Parent))) {
            New-Item -ItemType Directory -Force -Path (Split-Path $LOG_FILE -Parent) | Out-Null
        }
        
        # Build based on type
        switch ($BuildType) {
            "cython" { Invoke-CythonBuild }
            "numba" { Invoke-NumbaBuild }
            "pyo3" { Invoke-PyO3Build }
            "all" {
                Invoke-CythonBuild
                Invoke-NumbaBuild
                Invoke-PyO3Build
                Invoke-WheelPackage
            }
        }
        
        Write-Log ""
        Write-Log "========================================" -ForegroundColor Green
        Write-Log "WHEEL BUILD COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Log "========================================" -ForegroundColor Green
        Write-Log "Output directory: $WHEEL_OUTPUT"
        Write-Log "Cache directory: $CACHE_DIR"
        Write-Log ""
        Write-Log "GPU Configuration:"
        Write-Log "  ROCm:     $($gpuConfig.ROCmAvailable)"
        Write-Log "  DirectML: $($gpuConfig.DirectMLAvailable)"
        Write-Log "  CUDA:     $($gpuConfig.CUDAAvailable)"
        Write-Log ""
        Write-Log "To install wheels:"
        Write-Log "  pip install --force-reinstall --no-index --find-links=$WHEEL_OUTPUT nautilus-ray-python"
        
    } catch {
        Write-Log "BUILD FAILED: $($_.Exception.Message)" -Level "ERROR" -ForegroundColor Red
        throw
    }
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try {
    Invoke-FullBuild
} catch {
    Write-Host "Fatal error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

exit 0
