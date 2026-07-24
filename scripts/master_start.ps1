# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - MASTER START ORCHESTRATOR
# =============================================================================
# Stage 54: Master /START Orchestrator
# Purpose: Sequentially initialize AMD OS tweaks, mount huge pages, boot Ray cluster,
#          and ignite the Rust PGO binary
# Target: AMD Ryzen AI 5 with 8GB RAM limit
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [switch]$SkipTweaks,
    
    [Parameter(Mandatory = $false)]
    [switch]$SkipRay,
    
    [Parameter(Mandatory = $false)]
    [int]$RamLimitMB = 8192,
    
    [Parameter(Mandatory = $false)]
    [string]$ConfigPath = "config/trading_config.json",
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_ROOT
$LOG_FILE = Join-Path $PROJECT_ROOT "logs/master_start_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"
$PID_FILE = Join-Path $PROJECT_ROOT ".pids/master.pids"

# Binary paths
$RUST_BINARY = Join-Path $PROJECT_ROOT "target/release/nautilus-ray-bot.exe"
$RAY_START_SCRIPT = Join-Path $PROJECT_ROOT "scripts/start_ray.ps1"

# Memory configuration
$HUGE_PAGE_SIZE_MB = 2
$PYTHON_RAM_QUOTA_MB = 4096
$RUST_RAM_QUOTA_MB = 4096

# Job object name for process group management
$JOB_OBJECT_NAME = "NautilusRayBot_Main_Job"

# Component Scripts
$Scripts = @{
    OSStripper      = "C:\Nautilus\Scripts\os_stripper.ps1"
    KernelTuning    = "C:\Nautilus\Scripts\kernel_tuning.ps1"
    FirewallLockdown = "C:\Nautilus\Scripts\firewall_lockdown.ps1"
    FSReadOnly      = "C:\Nautilus\Scripts\fs_readonly.ps1"
    AMDSecure       = "C:\Nautilus\Scripts\amd_secure.ps1"
    NICTuning       = "C:\Nautilus\Scripts\nic_tuning.ps1"
}

# Process Handles
$Global:RustEngineProcess = $null
$Global:RayHeadProcess = $null
$Global:RayWorkerProcesses = @()

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    if (-not (Test-Path (Split-Path $LogPath))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
    }
    Add-Content -Path $LogPath -Value $logEntry
    
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    Write-Host $logEntry -ForegroundColor $color
}

function Initialize-Backup {
    if (-not (Test-Path $BackupPath)) {
        New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null
    }
    Write-Log "Backup directory ready: $BackupPath"
}

function Test-Prerequisites {
    Write-Log "Testing system prerequisites..."
    
    # Check Admin privileges
    $isAdmin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    
    if (-not $isAdmin) {
        throw "Administrator privileges required. Run as Administrator."
    }
    
    # Check available RAM (must have at least 8GB free)
    $os = Get-WmiObject Win32_OperatingSystem
    $freeRAM = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    
    if ($freeRAM -lt 8) {
        Write-Log "WARNING: Less than 8GB free RAM detected ($freeRAM GB)" -Level "WARN"
    } else {
        Write-Log "Available RAM: $freeRAM GB" -Level "SUCCESS"
    }
    
    # Check for required executables
    $requiredExes = @("rust.exe", "python.exe", "ray.exe")
    foreach ($exe in $requiredExes) {
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
            Write-Log "WARNING: Required executable not found: $exe" -Level "WARN"
        }
    }
    
    Write-Log "Prerequisites check completed."
}

function Invoke-OSTuning {
    Write-Log "=== PHASE 1: OS TUNING ==="
    
    # Run OS Stripper
    if (Test-Path $Scripts.OSStripper) {
        Write-Log "Executing OS Stripper..."
        & $Scripts.OSStripper -DryRun:$DryRun
    }
    
    # Run Kernel Tuning
    if (Test-Path $Scripts.KernelTuning) {
        Write-Log "Executing Kernel Tuning..."
        & $Scripts.KernelTuning -DryRun:$DryRun
    }
    
    # Run NIC Tuning
    if (Test-Path $Scripts.NICTuning) {
        Write-Log "Executing NIC Tuning..."
        & $Scripts.NICTuning -DryRun:$DryRun
    }
    
    Write-Log "OS Tuning phase completed." -Level "SUCCESS"
}

function Invoke-SecurityLockdown {
    Write-Log "=== PHASE 2: SECURITY LOCKDOWN ==="
    
    # Run Firewall Lockdown
    if (Test-Path $Scripts.FirewallLockdown) {
        Write-Log "Configuring Firewall..."
        & $Scripts.FirewallLockdown -DryRun:$DryRun
    }
    
    # Run AMD Secure Config
    if (Test-Path $Scripts.AMDSecure) {
        Write-Log "Configuring AMD Security Features..."
        & $Scripts.AMDSecure -DryRun:$DryRun
    }
    
    # Run Filesystem Read-Only
    if (Test-Path $Scripts.FSReadOnly) {
        Write-Log "Setting Filesystem to Read-Only..."
        & $Scripts.FSReadOnly -TargetDir "C:\Nautilus\Bin" -DryRun:$DryRun
    }
    
    Write-Log "Security Lockdown phase completed." -Level "SUCCESS"
}

function Start-RayCluster {
    Write-Log "=== PHASE 3: RAY CLUSTER INITIALIZATION ==="
    
    # Start Ray Head Node
    Write-Log "Starting Ray Head Node..."
    $rayArgs = @("start", "--head", "--num-cpus=4", "--memory=4294967296") # 4GB limit
    
    if (-not $DryRun) {
        $Global:RayHeadProcess = Start-Process "ray.exe" -ArgumentList $rayArgs -PassThru -NoNewWindow
        Start-Sleep -Seconds 2
        Write-Log "Ray Head Node started (PID: $($Global:RayHeadProcess.Id))" -Level "SUCCESS"
    } else {
        Write-Log "[DRY RUN] Would start Ray Head Node with args: $($rayArgs -join ' ')"
    }
    
    # Start Ray Worker Processes (limited for 8GB total)
    $workerCount = 2
    for ($i = 0; $i -lt $workerCount; $i++) {
        Write-Log "Starting Ray Worker $i..."
        $workerArgs = @("start", "--address=localhost:6379", "--num-cpus=2", "--memory=1073741824") # 1GB per worker
        
        if (-not $DryRun) {
            $worker = Start-Process "ray.exe" -ArgumentList $workerArgs -PassThru -NoNewWindow
            $Global:RayWorkerProcesses += $worker
            Write-Log "Ray Worker $i started (PID: $($worker.Id))" -Level "SUCCESS"
        }
    }
    
    Write-Log "Ray Cluster initialization completed." -Level "SUCCESS"
}

function Start-RustEngine {
    Write-Log "=== PHASE 4: RUST ENGINE IGNITION ==="
    
    $enginePath = "C:\Nautilus\Bin\nautilus_engine.exe"
    
    if (-not (Test-Path $enginePath)) {
        throw "Rust engine not found at: $enginePath"
    }
    
    Write-Log "Launching Rust HFT Engine..."
    
    if (-not $DryRun) {
        # Pin to Core 0-1 (reserved for HFT)
        $Global:RustEngineProcess = Start-Process $enginePath -PassThru -NoNewWindow
        
        # Set affinity to cores 0-1 (mask = 3)
        $process = Get-Process -Id $Global:RustEngineProcess.Id
        $process.ProcessorAffinity = 3
        
        Write-Log "Rust Engine started (PID: $($Global:RustEngineProcess.Id), Affinity: Cores 0-1)" -Level "SUCCESS"
    } else {
        Write-Log "[DRY RUN] Would launch Rust Engine from: $enginePath"
    }
    
    Write-Log "Rust Engine ignition completed." -Level "SUCCESS"
}

function Save-State {
    $state = @{
        Status = "RUNNING"
        StartTime = Get-Date -Format "o"
        RustPID = $Global:RustEngineProcess?.Id
        RayHeadPID = $Global:RayHeadProcess?.Id
        RayWorkerPIDs = $Global:RayWorkerProcesses?.Id
    }
    
    if (-not (Test-Path (Split-Path $StatePath))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $StatePath) | Out-Null
    }
    
    $state | ConvertTo-Json | Set-Content -Path $StatePath
    Write-Log "State saved to: $StatePath"
}

function Invoke-KillSequence {
    Write-Log "!!! KILL SEQUENCE INITIATED !!!" -Level "ERROR"
    
    # Stop Rust Engine
    if ($Global:RustEngineProcess -and $Global:RustEngineProcess.HasExited -eq $false) {
        Write-Log "Stopping Rust Engine..."
        Stop-Process -Id $Global:RustEngineProcess.Id -Force
        Write-Log "Rust Engine stopped."
    }
    
    # Stop Ray Workers
    foreach ($worker in $Global:RayWorkerProcesses) {
        if ($worker -and $worker.HasExited -eq $false) {
            Write-Log "Stopping Ray Worker (PID: $($worker.Id))..."
            Stop-Process -Id $worker.Id -Force
        }
    }
    
    # Stop Ray Head
    if ($Global:RayHeadProcess -and $Global:RayHeadProcess.HasExited -eq $false) {
        Write-Log "Stopping Ray Head..."
        Stop-Process -Id $Global:RayHeadProcess.Id -Force
    }
    
    # Rollback OS Tuning
    Write-Log "Rolling back OS tweaks..."
    & $Scripts.KernelTuning -Rollback
    & $Scripts.OSStripper -Rollback
    & $Scripts.FirewallLockdown -Rollback
    & $Scripts.FSReadOnly -Rollback
    
    # Update State
    $state = @{
        Status = "KILLED"
        KillTime = Get-Date -Format "o"
    }
    $state | ConvertTo-Json | Set-Content -Path $StatePath
    
    Write-Log "!!! KILL SEQUENCE COMPLETE !!!" -Level "SUCCESS"
}

# Main Execution
try {
    Initialize-Backup
    
    if ($Kill) {
        Invoke-KillSequence
        exit 0
    }
    
    Write-Log "=========================================" 
    Write-Log "  NAUTILUS/RAY BOT - MASTER START" 
    Write-Log "  Stage 53: Bare-Metal HFT Edition"
    Write-Log "=========================================" 
    
    Test-Prerequisites
    Invoke-OSTuning
    Invoke-SecurityLockdown
    Start-RayCluster
    Start-RustEngine
    Save-State
    
    Write-Log "=========================================" 
    Write-Log "  SYSTEM ONLINE - TRADING ACTIVE"
    Write-Log "=========================================" -Level "SUCCESS"
    
} catch {
    Write-Log "FATAL ERROR: $_" -Level "ERROR"
    Write-Log "Initiating emergency kill sequence..."
    Invoke-KillSequence
    throw
}
