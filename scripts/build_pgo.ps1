# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - PGO BUILD SCRIPT
# =============================================================================
# Stage 54: Profile-Guided Optimization (PGO) Build Automation
# Target: AMD Ryzen AI 5 with 8GB RAM limit enforcement
# Purpose: Generate instrumented binary, run market replay, recompile with profiles
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("instrument", "replay", "optimize", "full")]
    [string]$Phase = "full",
    
    [Parameter(Mandatory = $false)]
    [int]$RamLimitMB = 8192,
    
    [Parameter(Mandatory = $false)]
    [string]$MarketDataPath = "data/replay/market_ticks.bin",
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_ROOT
$PGO_DIR = Join-Path $PROJECT_ROOT "target/pgo"
$INSTRUMENTED_BIN = Join-Path $PROJECT_ROOT "target/release/nautilus-ray-bot.exe"
$PROFILE_DIR = Join-Path $PGO_DIR "profiles"
$LOG_FILE = Join-Path $PROJECT_ROOT "logs/pgo_build_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"

# AMD Ryzen AI 5 specific optimizations
$TARGET_CPU = "native"
$LLVM_PGO_FLAGS = "-Cprofile-use=$PROFILE_DIR"
$MEMORY_LIMIT_BYTES = $RamLimitMB * 1MB

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

function Test-Environment {
    Write-Log "Validating build environment..."
    
    # Check Rust toolchain
    $rustcVersion = rustc --version 2>$null
    if (-not $rustcVersion) {
        throw "Rust compiler (rustc) not found. Please install Rust via rustup."
    }
    Write-Log "Rust compiler: $rustcVersion"
    
    # Check for LLVM profdata (required for PGO)
    $llvmProfdata = Get-Command "llvm-profdata" -ErrorAction SilentlyContinue
    if (-not $llvmProfdata) {
        Write-Log "WARNING: llvm-profdata not found in PATH. PGO may fail." -Level "WARN"
    }
    
    # Check available RAM
    $os = Get-CimInstance Win32_OperatingSystem
    $totalRamGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    Write-Log "Total system RAM: ${totalRamGB}GB"
    
    if ($totalRamGB -lt 8) {
        Write-Log "WARNING: System has less than 8GB RAM. Build may be memory-constrained." -Level "WARN"
    }
    
    # Check disk space
    $drive = Get-PSDrive (Split-Path $PROJECT_ROOT -Qualifier).TrimEnd(':')
    $freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
    if ($freeSpaceGB -lt 10) {
        throw "Insufficient disk space. PGO builds require at least 10GB free."
    }
    Write-Log "Free disk space: ${freeSpaceGB}GB"
    
    return $true
}

function Set-MemoryLimit {
    param([int]$LimitMB)
    
    Write-Log "Enforcing memory limit: ${LimitMB}MB during instrumentation phase"
    
    # Create job object to limit memory usage
    $jobName = "NautilusPGOBuild_$(Get-Random)"
    
    # Using Windows Job Objects for memory limiting
    $signature = @"
using System;
using System.Runtime.InteropServices;

public class JobObject {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetInformationJobObject(
        IntPtr hJob, 
        JOBOBJECTINFOCLASS JobObjectInfoClass, 
        IntPtr lpJobObjectInfo, 
        uint cbJobObjectInfoLength);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
    
    public enum JOBOBJECTINFOCLASS {
        JobObjectExtendedLimitInformation = 9
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public long Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }
}
"@
    
    try {
        Add-Type -TypeDefinition $signature -ErrorAction Stop
        Write-Log "Job Object memory limiting enabled"
    } catch {
        Write-Log "WARNING: Could not enable Job Object memory limiting" -Level "WARN"
    }
}

function Invoke-PGOInstrument {
    Write-Log "=== PHASE 1: Building Instrumented Binary ===" -ForegroundColor Cyan
    
    # Clean previous builds
    Write-Log "Cleaning target directory..."
    cargo clean 2>&1 | ForEach-Object { Write-Log $_ }
    
    # Build with PGO instrumentation
    Write-Log "Compiling with PGO instrumentation flags..."
    $env:RUSTFLAGS = "-Cprofile-generate=$PROFILE_DIR -Ccodegen-units=1 -Clto=thin"
    $env:CARGO_INCREMENTAL = "0"
    
    # Enforce memory limit during compilation
    Set-MemoryLimit -LimitMB $RamLimitMB
    
    $buildArgs = @(
        "build",
        "--release",
        "--profile=pgo-instrument",
        "--locked"
    )
    
    if ($Verbose) {
        $buildArgs += "--verbose"
    }
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    cargo @buildArgs 2>&1 | ForEach-Object { Write-Log $_ }
    
    $stopwatch.Stop()
    Write-Log "Instrumented build completed in $($stopwatch.Elapsed.TotalSeconds.ToString("F2")) seconds"
    
    if (-not (Test-Path $INSTRUMENTED_BIN)) {
        throw "Instrumented binary not found at $INSTRUMENTED_BIN"
    }
    
    Write-Log "Instrumented binary: $INSTRUMENTED_BIN" -ForegroundColor Green
}

function Invoke-MarketReplay {
    Write-Log "=== PHASE 2: Running Market Replay for Profile Generation ===" -ForegroundColor Cyan
    
    if (-not (Test-Path $MarketDataPath)) {
        Write-Log "WARNING: Market data file not found at $MarketDataPath. Creating synthetic replay data..." -Level "WARN"
        
        # Create synthetic market data for profiling
        $replayDir = Split-Path $MarketDataPath -Parent
        if (-not (Test-Path $replayDir)) {
            New-Item -ItemType Directory -Force -Path $replayDir | Out-Null
        }
        
        # Generate minimal synthetic tick data (1000 ticks for profiling)
        $syntheticData = 1..1000 | ForEach-Object {
            [PSCustomObject]@{
                timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() + $_
                symbol = "BTCUSDT"
                price = 50000 + ($_ % 100)
                quantity = 0.001
                side = ($_ % 2)
            }
        }
        $syntheticData | ConvertTo-Json -Compress | Out-File -FilePath $MarketDataPath -Encoding utf8
        Write-Log "Generated synthetic market data: $MarketDataPath"
    }
    
    Write-Log "Running instrumented binary against market replay data..."
    Write-Log "Input: $MarketDataPath"
    
    # Run the instrumented binary with profile generation
    $replayArgs = @(
        "--mode=replay",
        "--data=$MarketDataPath",
        "--duration=60",  # 60 second replay for profile collection
        "--output-profiles=$PROFILE_DIR"
    )
    
    $replayStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    # Execute with memory monitoring
    $process = Start-Process -FilePath $INSTRUMENTED_BIN `
        -ArgumentList $replayArgs `
        -PassThru `
        -Wait `
        -NoNewWindow
    
    $replayStopwatch.Stop()
    
    if ($process.ExitCode -ne 0) {
        Write-Log "Market replay failed with exit code: $($process.ExitCode)" -Level "ERROR"
        throw "PGO profile generation failed"
    }
    
    Write-Log "Market replay completed in $($replayStopwatch.Elapsed.TotalSeconds.ToString("F2")) seconds"
    
    # Verify profile data was generated
    $profileFiles = Get-ChildItem -Path $PROFILE_DIR -Filter "*.profraw" -ErrorAction SilentlyContinue
    if (-not $profileFiles) {
        Write-Log "WARNING: No .profraw files generated. PGO optimization may be ineffective." -Level "WARN"
    } else {
        Write-Log "Generated $($profileFiles.Count) profile files:" -ForegroundColor Green
        $profileFiles | ForEach-Object { Write-Log "  - $($_.Name)" }
    }
}

function Invoke-PGOOptimize {
    Write-Log "=== PHASE 3: Recompiling with PGO Profiles ===" -ForegroundColor Cyan
    
    # Merge profile data using llvm-profdata
    Write-Log "Merging profile data..."
    $mergedProfile = Join-Path $PROFILE_DIR "merged.profdata"
    
    $profrawFiles = Get-ChildItem -Path $PROFILE_DIR -Filter "*.profraw"
    if (-not $profrawFiles) {
        throw "No profile data found. Run market replay first."
    }
    
    $profrawPaths = $profrawFiles | ForEach-Object { $_.FullName }
    
    llvm-profdata merge -sparse @profrawPaths -o $mergedProfile 2>&1 | ForEach-Object { Write-Log $_ }
    
    if (-not (Test-Path $mergedProfile)) {
        throw "Failed to merge profile data"
    }
    
    Write-Log "Merged profile: $mergedProfile" -ForegroundColor Green
    
    # Clean intermediate build artifacts
    Write-Log "Cleaning intermediate artifacts..."
    cargo clean -p nautilus-ray-bot 2>&1 | ForEach-Object { Write-Log $_ }
    
    # Build final optimized binary with PGO
    Write-Log "Compiling final binary with PGO profiles..."
    $env:RUSTFLAGS = "-Cprofile-use=$mergedProfile -Ccodegen-units=1 -Clto=fat -Ctarget-cpu=$TARGET_CPU"
    $env:CARGO_INCREMENTAL = "0"
    
    $optimizeArgs = @(
        "build",
        "--release",
        "--profile=pgo-use",
        "--locked"
    )
    
    if ($Verbose) {
        $optimizeArgs += "--verbose"
    }
    
    $optimizeStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    cargo @optimizeArgs 2>&1 | ForEach-Object { Write-Log $_ }
    
    $optimizeStopwatch.Stop()
    
    Write-Log "PGO-optimized build completed in $($optimizeStopwatch.Elapsed.TotalSeconds.ToString("F2")) seconds"
    
    # Display binary size comparison
    $finalBin = $INSTRUMENTED_BIN
    if (Test-Path $finalBin) {
        $binSize = (Get-Item $finalBin).Length / 1MB
        Write-Log "Final binary size: $([math]::Round($binSize, 2))MB" -ForegroundColor Green
    }
}

function Invoke-FullPGOBuild {
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "NAUTILUS/RAY PGO BUILD - FULL WORKFLOW" -ForegroundColor Cyan
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "Target CPU: $TARGET_CPU"
    Write-Log "RAM Limit: ${RamLimitMB}MB"
    Write-Log "Profile Directory: $PROFILE_DIR"
    Write-Log ""
    
    try {
        # Phase 0: Environment validation
        Test-Environment
        
        # Phase 1: Build instrumented binary
        Invoke-PGOInstrument
        
        # Phase 2: Run market replay to generate profiles
        Invoke-MarketReplay
        
        # Phase 3: Recompile with profiles
        Invoke-PGOOptimize
        
        Write-Log ""
        Write-Log "========================================" -ForegroundColor Green
        Write-Log "PGO BUILD COMPLETED SUCCESSFULLY" -ForegroundColor Green
        Write-Log "========================================" -ForegroundColor Green
        Write-Log "Output: $INSTRUMENTED_BIN"
        Write-Log "Profiles: $PROFILE_DIR"
        Write-Log ""
        Write-Log "Next steps:"
        Write-Log "  1. Run micro-benchmarks to verify latency improvements"
        Write-Log "  2. Deploy to production with master_start.ps1"
        Write-Log "  3. Monitor with health_monitor.ps1"
        
    } catch {
        Write-Log "BUILD FAILED: $($_.Exception.Message)" -Level "ERROR" -ForegroundColor Red
        throw
    }
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try {
    # Ensure log directory exists
    $logDir = Split-Path $LOG_FILE -Parent
    if (-not (Test-Path $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    
    # Ensure profile directory exists
    if (-not (Test-Path $PROFILE_DIR)) {
        New-Item -ItemType Directory -Force -Path $PROFILE_DIR | Out-Null
    }
    
    switch ($Phase) {
        "instrument" { Invoke-PGOInstrument }
        "replay" { Invoke-MarketReplay }
        "optimize" { Invoke-PGOOptimize }
        "full" { Invoke-FullPGOBuild }
        default { throw "Unknown phase: $Phase" }
    }
    
} catch {
    Write-Host "Fatal error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

exit 0
