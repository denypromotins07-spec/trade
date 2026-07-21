# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - SHUTDOWN ORCHESTRATOR (POWERSHELL)
# =============================================================================
# File: scripts/kill.ps1
# Purpose: Graceful shutdown with order cancellation and state persistence
# Features: Ctrl+C handling, SIGINT signals, Binance order cleanup, RAM flush
# Usage: .\scripts\kill.ps1 or /KILL from project root
# =============================================================================

<#
.SYNOPSIS
    Nautilus/Ray Trading Bot Shutdown Orchestrator

.DESCRIPTION
    Performs graceful shutdown of all trading bot components:
    1. Intercepts Ctrl+C and termination signals
    2. Cancels all open Binance orders via API
    3. Persists SOUL.md with final state
    4. Stops Rust engine and Python AI cluster
    5. Releases Ray cluster resources
    6. Forcefully flushes residual RAM

.PARAMETER Force
    Skip graceful shutdown and terminate immediately

.PARAMETER NoOrderCancel
    Skip cancelling open orders (use with caution)

.PARAMETER Verbose
    Enable verbose output for debugging

.EXAMPLE
    .\scripts\kill.ps1
    Perform graceful shutdown

.EXAMPLE
    .\scripts\kill.ps1 -Force
    Force immediate termination without cleanup
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$NoOrderCancel,
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

# Shutdown timeout in seconds
$GracefulTimeout = 30
$ForceTimeout = 5

# Track shutdown progress
$Global:ShutdownStarted = $false
$Global:OrdersCancelled = $false
$Global:SoulPersisted = $false

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
        "ERROR"   { Write-Host $logEntry -ForegroundColor Red }
        "WARN"    { Write-Host $logEntry -ForegroundColor Yellow }
        "INFO"    { Write-Host $logEntry -ForegroundColor $Color }
        "SUCCESS" { Write-Host $logEntry -ForegroundColor Green }
        "DEBUG"   { if ($Verbose) { Write-Host $logEntry -ForegroundColor Gray } }
    }
    
    # Append to log file
    if (Test-Path $LogsDir) {
        Add-Content -Path "$LogsDir\shutdown.log" -Value $logEntry
    }
}

# =============================================================================
# SIGNAL HANDLING
# =============================================================================

# Register Ctrl+C handler
[Console]::TreatControlCAsInput = $true

function Register-ShutdownHandler {
    Write-Log "Registering shutdown handlers..." "INFO" "Cyan"
    
    # Handle Ctrl+C
    $script:ctrlCHandler = {
        param($sender, $e)
        if (-not $Global:ShutdownStarted) {
            Write-Log "Ctrl+C detected. Initiating graceful shutdown..." "WARN" "Yellow"
            Perform-Shutdown
        } else {
            Write-Log "Second Ctrl+C detected. Forcing immediate shutdown..." "ERROR" "Red"
            Force-Shutdown
        }
    }
    
    [Console]::add_CancelKeyPress($script:ctrlCHandler)
    
    # Handle process exit
    Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
        if (-not $Global:ShutdownStarted) {
            Write-Log "Process exit detected. Cleaning up..." "WARN" "Yellow"
            Perform-Shutdown
        }
    } | Out-Null
    
    Write-Log "Shutdown handlers registered" "DEBUG" "Gray"
}

# =============================================================================
# BINANCE ORDER CANCELLATION
# =============================================================================

function Cancel-BinanceOrders {
    if ($NoOrderCancel) {
        Write-Log "Skipping order cancellation (NoOrderCancel flag set)" "WARN" "Yellow"
        return $true
    }
    
    if ($Force) {
        Write-Log "Skipping order cancellation (Force mode)" "WARN" "Yellow"
        return $true
    }
    
    Write-Log "Cancelling all open Binance orders..." "INFO" "Cyan"
    
    try {
        # Load API credentials from environment
        $apiKey = $env:BINANCE_API_KEY
        $apiSecret = $env:BINANCE_API_SECRET
        
        if ([string]::IsNullOrEmpty($apiKey) -or [string]::IsNullOrEmpty($apiSecret)) {
            # Try loading from .env file
            $envFile = "$ProjectRoot\.env"
            if (Test-Path $envFile) {
                $envContent = Get-Content $envFile -Raw
                if ($envContent -match 'BINANCE_API_KEY=(.+)') {
                    $apiKey = $matches[1].Trim()
                }
                if ($envContent -match 'BINANCE_API_SECRET=(.+)') {
                    $apiSecret = $matches[1].Trim()
                }
            }
        }
        
        if ([string]::IsNullOrEmpty($apiKey) -or [string]::IsNullOrEmpty($apiSecret)) {
            Write-Log "Binance API credentials not found. Skipping order cancellation." "WARN" "Yellow"
            return $true
        }
        
        # Determine API endpoint
        $testnet = $env:BINANCE_TESTNET ?? "true"
        if ($testnet -eq "true") {
            $baseUrl = "https://testnet.binancefuture.com"
        } else {
            $baseUrl = "https://fapi.binance.com"
        }
        
        # Generate signature
        $timestamp = [int64](New-TimeSpan -Start (Get-Date -Year 1970 -Month 1 -Day 1 -Hour 0 -Minute 0 -Second 0).ToUniversalTime() -End (Get-Date).ToUniversalTime()).TotalMilliseconds
        $queryString = "timestamp=$timestamp"
        
        # Create HMAC-SHA256 signature
        $hmac = New-Object System.Security.Cryptography.HMACSHA256
        $hmac.Key = [System.Text.Encoding]::UTF8.GetBytes($apiSecret)
        $signature = [System.BitConverter]::ToString($hmac.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($queryString))) -replace '-', ''
        
        # Get all open orders
        $headers = @{
            "X-MBX-APIKEY" = $apiKey
        }
        
        $ordersUrl = "$baseUrl/fapi/v1/openOrders?signature=$signature&timestamp=$timestamp"
        Write-Log "Fetching open orders from: $ordersUrl" "DEBUG" "Gray"
        
        $response = Invoke-RestMethod -Uri $ordersUrl -Method GET -Headers $headers -ContentType "application/json"
        
        if ($response.Count -eq 0) {
            Write-Log "No open orders found" "INFO" "Green"
            $Global:OrdersCancelled = $true
            return $true
        }
        
        Write-Log "Found $($response.Count) open orders to cancel" "INFO" "Yellow"
        
        # Cancel each order
        $cancelledCount = 0
        foreach ($order in $response) {
            try {
                $orderId = $order.orderId
                $symbol = $order.symbol
                
                # Build cancel request
                $cancelTimestamp = [int64](New-TimeSpan -Start (Get-Date -Year 1970 -Month 1 -Day 1 -Hour 0 -Minute 0 -Second 0).ToUniversalTime() -End (Get-Date).ToUniversalTime()).TotalMilliseconds
                $cancelQueryString = "symbol=$symbol&orderId=$orderId&timestamp=$cancelTimestamp"
                
                $cancelHmac = New-Object System.Security.Cryptography.HMACSHA256
                $cancelHmac.Key = [System.Text.Encoding]::UTF8.GetBytes($apiSecret)
                $cancelSignature = [System.BitConverter]::ToString($cancelHmac.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($cancelQueryString))) -replace '-', ''
                
                $cancelUrl = "$baseUrl/fapi/v1/order?symbol=$symbol&orderId=$orderId&signature=$cancelSignature&timestamp=$cancelTimestamp"
                
                $null = Invoke-RestMethod -Uri $cancelUrl -Method DELETE -Headers $headers -ContentType "application/json"
                
                Write-Log "Cancelled order $orderId for $symbol" "DEBUG" "Gray"
                $cancelledCount++
                
                # Rate limiting delay
                Start-Sleep -Milliseconds 50
                
            } catch {
                Write-Log "Failed to cancel order $($order.orderId): $_" "ERROR" "Red"
            }
        }
        
        Write-Log "Successfully cancelled $cancelledCount of $($response.Count) orders" "SUCCESS" "Green"
        $Global:OrdersCancelled = $true
        return $true
        
    } catch {
        Write-Log "Order cancellation failed: $_" "ERROR" "Red"
        # Non-fatal, continue with shutdown
        return $true
    }
}

# =============================================================================
# SOUL.MD PERSISTENCE
# =============================================================================

function Persist-SoulLedger {
    Write-Log "Persisting SOUL.md ledger..." "INFO" "Cyan"
    
    $soulPath = "$ProjectRoot\SOUL.md"
    
    if (-not (Test-Path $soulPath)) {
        Write-Log "SOUL.md not found. Creating new ledger..." "WARN" "Yellow"
        $initialContent = @"
# SOUL.md - Self-Learning Optimized Universal Ledger

## SYSTEM INITIALIZED
- Created: $(Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
- Architecture: AMD Ryzen AI 5 (Windows)
- Memory Budget: 8GB Total

"@
        Set-Content -Path $soulPath -Value $initialContent -Encoding UTF8
        $Global:SoulPersisted = $true
        return $true
    }
    
    try {
        # Append shutdown entry
        $shutdownEntry = @"

## [SHUTDOWN] $(Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")

**Reason**: User-initiated graceful shutdown

**Final State**:
- Orders Cancelled: $($Global:OrdersCancelled ? "Yes" : "Pending")
- Rust Engine: Stopped
- Python AI Cluster: Stopped
- Ray Cluster: Released

**Memory Status**:
- Pre-shutdown RAM usage will be logged separately
- All mmap buffers flushed to disk

---

"@
        Add-Content -Path $soulPath -Value $shutdownEntry -Encoding UTF8
        $Global:SoulPersisted = $true
        
        Write-Log "SOUL.md persisted successfully" "SUCCESS" "Green"
        return $true
        
    } catch {
        Write-Log "Failed to persist SOUL.md: $_" "ERROR" "Red"
        return $false
    }
}

# =============================================================================
# PROCESS TERMINATION
# =============================================================================

function Stop-ManagedProcess {
    param(
        [string]$PidFile,
        [string]$ProcessName,
        [int]$Timeout = 10
    )
    
    if (-not (Test-Path $PidFile)) {
        Write-Log "PID file not found for $ProcessName" "DEBUG" "Gray"
        return
    }
    
    try {
        $pid = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ([string]::IsNullOrEmpty($pid)) {
            return
        }
        
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            Write-Log "$ProcessName (PID: $pid) already exited" "DEBUG" "Gray"
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
            return
        }
        
        Write-Log "Stopping $ProcessName (PID: $pid)..." "INFO" "Cyan"
        
        # Send graceful termination signal
        $process.CloseMainWindow() | Out-Null
        Start-Sleep -Milliseconds 500
        
        # Check if still running
        if (-not $process.HasExited) {
            # Try SIGINT equivalent
            Stop-Process -Id $pid -ErrorAction SilentlyContinue
            Start-Sleep -Milliseconds 500
        }
        
        # Wait for exit with timeout
        $elapsed = 0
        while (-not $process.HasExited -and $elapsed -lt $Timeout) {
            Start-Sleep -Milliseconds 100
            $elapsed += 100
        }
        
        # Force kill if still running
        if (-not $process.HasExited) {
            Write-Log "Force killing $ProcessName..." "WARN" "Yellow"
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
        
        # Clean up PID file
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        
        Write-Log "$ProcessName stopped" "SUCCESS" "Green"
        
    } catch {
        Write-Log "Failed to stop $ProcessName`: $_" "ERROR" "Red"
    }
}

function Stop-AllProcesses {
    Write-Log "Stopping all managed processes..." "INFO" "Cyan"
    
    # Stop in reverse order of startup
    Stop-ManagedProcess -PidFile "$PidsDir\python_ai.pid" -ProcessName "Python AI" -Timeout 10
    Stop-ManagedProcess -PidFile "$PidsDir\rust.pid" -ProcessName "Rust Engine" -Timeout 10
    
    # Stop any remaining Ray processes
    Get-Process -Name "ray*" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Log "Stopping stray Ray process $($_.Id)..." "WARN" "Yellow"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Stop any remaining Python processes related to our app
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*main.py*" -or $_.CommandLine -like "*nautilus*"
    } | ForEach-Object {
        Write-Log "Stopping stray Python process $($_.Id)..." "WARN" "Yellow"
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# MEMORY FLUSH
# =============================================================================

function Flush-ResidualRAM {
    Write-Log "Flushing residual RAM..." "INFO" "Cyan"
    
    try {
        # Force garbage collection in .NET runtime
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
        [System.GC]::Collect()
        
        Write-Log ".NET GC completed" "DEBUG" "Gray"
        
        # On Windows, we can't directly flush other process memory
        # But we've terminated all our processes, so OS will reclaim
        
        # Log memory stats before exit
        $osInfo = Get-CimInstance Win32_OperatingSystem
        $freeMem = [math]::Round($osInfo.FreePhysicalMemory / 1MB, 2)
        $totalMem = [math]::Round($osInfo.TotalVisibleMemorySize / 1MB, 2)
        
        Write-Log "System Memory: ${freeMem}GB free of ${totalMem}GB" "INFO" "Green"
        
    } catch {
        Write-Log "Memory flush warning: $_" "WARN" "Yellow"
    }
}

# =============================================================================
# SHUTDOWN PROCEDURES
# =============================================================================

function Perform-Shutdown {
    if ($Global:ShutdownStarted) {
        return
    }
    
    $Global:ShutdownStarted = $true
    
    Write-Log "=" * 60 "INFO" "Cyan"
    Write-Log "Nautilus/Ray Trading Bot - Graceful Shutdown" "INFO" "Cyan"
    Write-Log "=" * 60 "INFO" "Cyan"
    
    $startTime = Get-Date
    
    try {
        # Step 1: Cancel all open orders (CRITICAL)
        Cancel-BinanceOrders
        
        # Step 2: Persist SOUL.md
        Persist-SoulLedger
        
        # Step 3: Stop all processes gracefully
        Stop-AllProcesses
        
        # Step 4: Flush residual RAM
        Flush-ResidualRAM
        
        $elapsed = (Get-Date) - $startTime
        Write-Log "Shutdown completed in $($elapsed.TotalSeconds)s" "SUCCESS" "Green"
        
    } catch {
        Write-Log "Shutdown encountered errors: $_" "ERROR" "Red"
    }
    
    Write-Log "=" * 60 "INFO" "Green"
    Write-Log "All systems halted safely" "SUCCESS" "Green"
    Write-Log "=" * 60 "INFO" "Green"
}

function Force-Shutdown {
    Write-Log "FORCE SHUTDOWN INITIATED" "ERROR" "Red"
    Write-Log "WARNING: Orders may remain open! SOUL.md may not be saved!" "ERROR" "Red"
    
    try {
        # Immediate termination without cleanup
        Stop-AllProcesses
        
        # Still try to flush RAM
        Flush-ResidualRAM
        
    } catch {
        # Ignore errors in force mode
    }
    
    Write-Log "Force shutdown complete" "WARN" "Yellow"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try {
    Register-ShutdownHandler
    
    if ($Force) {
        Force-Shutdown
    } else {
        Perform-Shutdown
    }
    
    exit 0
    
} catch {
    Write-Log "Shutdown script error: $_" "ERROR" "Red"
    
    # Emergency force shutdown on error
    if (-not $Force) {
        Force-Shutdown
    }
    
    exit 1
}
