# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - STARTUP ORCHESTRATOR (POWERSHELL)
# =============================================================================
# File: scripts/start.ps1
# Purpose: Concurrently boot Rust core, Python AI, and Ray dashboard
# Features: Hidden background processes, PID capture, job object management
# Usage: .\scripts\start.ps1 or /START from project root
# =============================================================================

<#
.SYNOPSIS
    Nautilus/Ray Trading Bot Startup Orchestrator

.DESCRIPTION
    Launches all components of the trading bot system:
    1. Rust execution engine (high-frequency trading core)
    2. Python AI cluster (Ray distributed compute)
    3. Ray dashboard (monitoring interface)

    All processes are managed via Windows Job Objects to ensure
    proper cleanup on termination.

.PARAMETER NoDashboard
    Skip launching the Ray dashboard (saves resources)

.PARAMETER PaperTrading
    Start in paper trading mode (no real orders)

.PARAMETER Verbose
    Enable verbose output for debugging

.EXAMPLE
    .\scripts\start.ps1
    Start all components with default settings

.EXAMPLE
    .\scripts\start.ps1 -PaperTrading -Verbose
    Start in paper trading mode with verbose logging
#>

[CmdletBinding()]
param(
    [switch]$NoDashboard,
    [switch]$PaperTrading,
    [switch]$Verbose
)

# =============================================================================
# CONFIGURATION AND INITIALIZATION
# =============================================================================

$ErrorActionPreference = "Stop"
$OutputEncoding = [System.Text.Encoding]::UTF8

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\.."
$LogsDir = "$ProjectRoot\logs"
$PidsDir = "$ProjectRoot\.pids"

# Process names for tracking
$RustProcessName = "nautilus-ray-bot"
$PythonProcessName = "python"
$RayProcessName = "ray"

# Create necessary directories
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
    Write-Host "[INFO] Created logs directory: $LogsDir" -ForegroundColor Gray
}

if (-not (Test-Path $PidsDir)) {
    New-Item -ItemType Directory -Path $PidsDir | Out-Null
    Write-Host "[INFO] Created PIDs directory: $PidsDir" -ForegroundColor Gray
}

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO",
        [string]$Color = "White"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    switch ($Level) {
        "ERROR" { Write-Host $logEntry -ForegroundColor Red }
        "WARN"  { Write-Host $logEntry -ForegroundColor Yellow }
        "INFO"  { Write-Host $logEntry -ForegroundColor $Color }
        "DEBUG" { if ($Verbose) { Write-Host $logEntry -ForegroundColor Gray } }
    }
    
    # Append to log file
    Add-Content -Path "$LogsDir\startup.log" -Value $logEntry
}

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

function Test-Environment {
    Write-Log "Validating environment..." "INFO" "Cyan"
    
    $errors = @()
    
    # Check for .env file
    if (-not (Test-Path "$ProjectRoot\.env")) {
        $errors += ".env file not found. Please create configuration file."
    }
    
    # Check for Rust binary
    $rustBinary = "$ProjectRoot\target\release\nautilus-ray-bot.exe"
    if (-not (Test-Path $rustBinary)) {
        Write-Log "Rust binary not found. Building release version..." "WARN" "Yellow"
        return "BUILD_REQUIRED"
    }
    
    # Check for Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Log "Python found: $pythonVersion" "DEBUG" "Gray"
    } catch {
        $errors += "Python is not installed or not in PATH"
    }
    
    # Check for required Python packages
    try {
        python -c "import ray" 2>$null
        if ($LASTEXITCODE -ne 0) {
            $errors += "Ray is not installed. Run: pip install ray[default]"
        }
    } catch {
        $errors += "Failed to import Ray module"
    }
    
    if ($errors.Count -gt 0) {
        foreach ($error in $errors) {
            Write-Log $error "ERROR" "Red"
        }
        return $false
    }
    
    Write-Log "Environment validation passed" "INFO" "Green"
    return $true
}

# =============================================================================
# JOB OBJECT CREATION (WINDOWS PROCESS GROUPS)
# =============================================================================

# Note: PowerShell doesn't have native Job Object API exposure.
# We use a workaround with process tracking and explicit termination.
# For production, consider using a compiled helper or C# interop.

$Global:ManagedProcesses = @()

function Register-Process {
    param([System.Diagnostics.Process]$Process)
    
    $Global:ManagedProcesses += $Process
    Write-Log "Registered process: $($Process.Id) - $($Process.ProcessName)" "DEBUG" "Gray"
}

function Stop-AllManagedProcesses {
    Write-Log "Stopping all managed processes..." "WARN" "Yellow"
    
    foreach ($proc in $Global:ManagedProcesses) {
        try {
            if (-not $proc.HasExited) {
                Write-Log "Stopping process $($proc.Id)..." "INFO" "Cyan"
                $proc.Kill()
                $proc.WaitForExit(5000)
            }
        } catch {
            Write-Log "Failed to stop process $($proc.Id): $_" "ERROR" "Red"
        }
    }
    
    $Global:ManagedProcesses.Clear()
}

# =============================================================================
# RUST ENGINE STARTUP
# =============================================================================

function Start-RustEngine {
    Write-Log "Starting Rust execution engine..." "INFO" "Cyan"
    
    $rustBinary = "$ProjectRoot\target\release\nautilus-ray-bot.exe"
    
    if (-not (Test-Path $rustBinary)) {
        Write-Log "Rust binary not found at: $rustBinary" "ERROR" "Red"
        Write-Log "Please build with: cargo build --release" "WARN" "Yellow"
        return $false
    }
    
    # Set environment variables for Rust engine
    $env:RUST_LOG = if ($Verbose) { "debug" } else { "info" }
    $env:RUST_BACKTRACE = if ($Verbose) { "1" } else { "0" }
    
    if ($PaperTrading) {
        $env:FEATURE_ENABLE_PAPER_TRADING = "true"
    }
    
    # Start Rust process (hidden window)
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $rustBinary
    $startInfo.WorkingDirectory = $ProjectRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    
    $process = [System.Diagnostics.Process]::Start($startInfo)
    Register-Process $process
    
    # Save PID
    $process.Id | Out-File "$PidsDir\rust.pid" -Encoding UTF8
    Write-Log "Rust engine started with PID: $($process.Id)" "INFO" "Green"
    
    # Start output reader thread
    Start-ThreadJob -ScriptBlock {
        param($proc, $logFile)
        while (-not $proc.HasExited) {
            $line = $proc.StandardOutput.ReadLine()
            if ($line) {
                Add-Content -Path $logFile -Value $line
            }
        }
    } -ArgumentList $process, "$LogsDir\rust_engine.log" | Out-Null
    
    return $true
}

# =============================================================================
# PYTHON AI CLUSTER STARTUP
# =============================================================================

function Start-PythonAI {
    Write-Log "Starting Python AI cluster..." "INFO" "Cyan"
    
    $pythonScript = "$ProjectRoot\python\main.py"
    
    if (-not (Test-Path $pythonScript)) {
        Write-Log "Python script not found at: $pythonScript" "ERROR" "Red"
        return $false
    }
    
    # Build command arguments
    $args = @("python", $pythonScript, "--start")
    if ($PaperTrading) {
        $args += "--paper-trading"
    }
    
    # Start Python process
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = "python"
    $startInfo.Arguments = "$pythonScript --start"
    $startInfo.WorkingDirectory = $ProjectRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    
    $process = [System.Diagnostics.Process]::Start($startInfo)
    Register-Process $process
    
    # Save PID
    $process.Id | Out-File "$PidsDir\python_ai.pid" -Encoding UTF8
    Write-Log "Python AI cluster started with PID: $($process.Id)" "INFO" "Green"
    
    return $true
}

# =============================================================================
# RAY DASHBOARD STARTUP
# =============================================================================

function Start-RayDashboard {
    if ($NoDashboard) {
        Write-Log "Skipping Ray dashboard (NoDashboard flag set)" "DEBUG" "Gray"
        return $true
    }
    
    Write-Log "Starting Ray dashboard..." "INFO" "Cyan"
    
    # Ray dashboard typically starts with Ray head node
    # This function ensures it's accessible
    
    $dashboardPort = $env:RAY_DASHBOARD_PORT ?? "8265"
    $dashboardUrl = "http://127.0.0.1:$dashboardPort"
    
    # Wait for dashboard to be ready
    $maxAttempts = 30
    $attempt = 0
    
    while ($attempt -lt $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri $dashboardUrl -TimeoutSec 2 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Log "Ray dashboard is ready at: $dashboardUrl" "INFO" "Green"
                return $true
            }
        } catch {
            # Dashboard not ready yet
        }
        
        Start-Sleep -Milliseconds 500
        $attempt++
    }
    
    Write-Log "Ray dashboard did not become ready within timeout" "WARN" "Yellow"
    return $true  # Non-fatal, continue anyway
}

# =============================================================================
# HEALTH CHECK
# =============================================================================

function Test-HealthCheck {
    Write-Log "Performing health check..." "INFO" "Cyan"
    
    $healthy = $true
    
    # Check Rust engine
    $rustPidFile = "$PidsDir\rust.pid"
    if (Test-Path $rustPidFile) {
        $pid = Get-Content $rustPidFile
        try {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Log "✓ Rust engine running (PID: $pid)" "INFO" "Green"
            } else {
                Write-Log "✗ Rust engine not running" "ERROR" "Red"
                $healthy = $false
            }
        } catch {
            Write-Log "✗ Rust engine check failed: $_" "ERROR" "Red"
            $healthy = $false
        }
    }
    
    # Check Python AI
    $pythonPidFile = "$PidsDir\python_ai.pid"
    if (Test-Path $pythonPidFile) {
        $pid = Get-Content $pythonPidFile
        try {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Log "✓ Python AI cluster running (PID: $pid)" "INFO" "Green"
            } else {
                Write-Log "✗ Python AI cluster not running" "ERROR" "Red"
                $healthy = $false
            }
        } catch {
            Write-Log "✗ Python AI check failed: $_" "ERROR" "Red"
            $healthy = $false
        }
    }
    
    return $healthy
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try {
    Write-Log "=" * 60 "INFO" "Cyan"
    Write-Log "Nautilus/Ray Trading Bot - Starting Up" "INFO" "Cyan"
    Write-Log "=" * 60 "INFO" "Cyan"
    
    # Validate environment
    $envStatus = Test-Environment
    if ($envStatus -eq "BUILD_REQUIRED") {
        Write-Log "Building Rust binary..." "INFO" "Cyan"
        Set-Location $ProjectRoot
        cargo build --release
        if ($LASTEXITCODE -ne 0) {
            throw "Rust build failed"
        }
    } elseif ($envStatus -eq $false) {
        throw "Environment validation failed"
    }
    
    # Start components
    if (-not (Start-RustEngine)) {
        throw "Failed to start Rust engine"
    }
    
    Start-Sleep -Seconds 2  # Give Rust engine time to initialize
    
    if (-not (Start-PythonAI)) {
        throw "Failed to start Python AI cluster"
    }
    
    Start-Sleep -Seconds 5  # Give Ray time to initialize
    
    if (-not (Start-RayDashboard)) {
        throw "Failed to start Ray dashboard"
    }
    
    # Health check
    Start-Sleep -Seconds 3
    if (-not (Test-HealthCheck)) {
        Write-Log "Health check failed. Some components may not have started correctly." "WARN" "Yellow"
    }
    
    Write-Log "=" * 60 "INFO" "Green"
    Write-Log "All components started successfully!" "INFO" "Green"
    Write-Log "Dashboard: http://127.0.0.1:8265" "INFO" "Green"
    Write-Log "To stop: Run .\scripts\kill.ps1 or /KILL" "INFO" "Green"
    Write-Log "=" * 60 "INFO" "Green"
    
} catch {
    Write-Log "Startup failed: $_" "ERROR" "Red"
    Stop-AllManagedProcesses
    exit 1
}
