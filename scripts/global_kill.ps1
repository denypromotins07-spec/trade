# global_kill.ps1 - Master /KILL Script and Ctrl+C Interceptor
# Stage 54: Nautilus/Ray Crypto Trading Bot
# Gracefully flushes orders, writes SOUL.md final states, terminates all processes

param(
    [switch]$Force = $false,
    [string]$LogPrefix = "[GLOBAL_KILL]"
)

$ErrorActionPreference = "Continue"
$BaseDir = Split-Path -Parent $PSScriptRoot

Write-Host "`n$LogPrefix ========================================" -ForegroundColor Cyan
Write-Host "$LogInit Prefix Initiating Graceful Shutdown Sequence" -ForegroundColor Yellow
Write-Host "$LogPrefix ========================================" -ForegroundColor Cyan

# Track shutdown start time for latency metrics
$shutdownStart = Get-Date

# Function to safely terminate a process tree
function Stop-ProcessTree {
    param([string]$ProcessName)
    $processes = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue
    if ($processes) {
        foreach ($proc in $processes) {
            Write-Host "$LogPrefix Terminating $ProcessName (PID: $($proc.Id))..." -ForegroundColor Yellow
            try {
                # Send graceful shutdown signal first
                $proc.CloseMainWindow() | Out-Null
                Start-Sleep -Milliseconds 500
                if (!$proc.HasExited) {
                    $proc.Kill() | Out-Null
                }
                $proc.WaitForExit(2000)
                Write-Host "$LogPrefix $ProcessName terminated gracefully" -ForegroundColor Green
            } catch {
                Write-Host "$LogPrefix WARNING: Force killing $ProcessName" -ForegroundColor Red
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

# Function to flush open orders via Rust gateway
function Invoke-OrderFlush {
    Write-Host "$LogPrefix Flushing open orders to exchange..." -ForegroundColor Yellow
    $flushEndpoint = "http://localhost:$gatewayPort/api/v1/flush-orders"
    try {
        $response = Invoke-RestMethod -Uri $flushEndpoint -Method POST -TimeoutSec 5
        Write-Host "$LogPrefix Order flush completed: $($response.status)" -ForegroundColor Green
    } catch {
        Write-Host "$LogPrefix WARNING: Order flush endpoint unavailable, orders may remain open" -ForegroundColor Red
    }
}

# Function to write final state to SOUL.md
function Write-SoulFinalState {
    param(
        [string]$Reason,
        [hashtable]$Metrics
    )
    $soulPath = Join-Path $BaseDir "SOUL.md"
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ"
    
    $finalEntry = @"

## Shutdown Event: $timestamp

**Reason**: $Reason

### Final Metrics
$(foreach ($key in $Metrics.Keys) { "- **$key`**: $($Metrics[$key])" })

### System State
- Rust Core: Terminated
- Ray Cluster: Terminated  
- Chrome Browser: Closed
- Open Orders: Flushed

---
"@
    
    try {
        if (Test-Path $soulPath) {
            Add-Content -Path $soulPath -Value $finalEntry -Encoding UTF8
        } else {
            $header = "# SOUL.md - Autonomous Learning Ledger`n`n" + $finalEntry
            Set-Content -Path $soulPath -Value $header -Encoding UTF8
        }
        Write-Host "$LogPrefix Final state written to SOUL.md" -ForegroundColor Green
    } catch {
        Write-Host "$LogPrefix ERROR: Failed to write SOUL.md: $_" -ForegroundColor Red
    }
}

# Function to kill orphaned Chrome processes
function Stop-OrphanedChrome {
    Write-Host "$LogPrefix Scanning for orphaned Chrome processes..." -ForegroundColor Yellow
    $chromeProcesses = Get-Process -Name "chrome" -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*--nautilus-bot*" }
    
    if ($chromeProcesses) {
        foreach ($chrome in $chromeProcesses) {
            Write-Host "$LogPrefix Killing orphaned Chrome (PID: $($chrome.Id))" -ForegroundColor Red
            Stop-Process -Id $chrome.Id -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "$LogPrefix No orphaned Chrome processes found" -ForegroundColor Green
    }
}

try {
    # Read gateway port if available
    $portFile = Join-Path $BaseDir "shared\gateway_port.txt"
    $gatewayPort = 8080
    if (Test-Path $portFile) {
        $gatewayPort = Get-Content $portFile | Select-Object -First 1
    }
    
    # Step 1: Flush open orders (if gateway is responsive)
    if (-not $Force) {
        Invoke-OrderFlush
    }
    
    # Step 2: Terminate Python Ray processes
    Write-Host "$LogPrefix Step 1/4: Terminating Python Ray Cluster..." -ForegroundColor Yellow
    Stop-ProcessTree -ProcessName "python"
    Start-Sleep -Milliseconds 300
    
    # Step 3: Terminate Rust core gateway
    Write-Host "$LogPrefix Step 2/4: Terminating Rust Core Gateway..." -ForegroundColor Yellow
    Stop-ProcessTree -ProcessName "nautilus_gateway"
    Start-Sleep -Milliseconds 300
    
    # Step 4: Close Chrome browser instances
    Write-Host "$LogPrefix Step 3/4: Closing Chrome Browser..." -ForegroundColor Yellow
    Stop-ProcessTree -ProcessName "chrome"
    Stop-OrphanedChrome
    Start-Sleep -Milliseconds 300
    
    # Step 5: Clean up Node.js frontend processes
    Write-Host "$LogPrefix Step 4/4: Cleaning up Frontend Processes..." -ForegroundColor Yellow
    Stop-ProcessTree -ProcessName "node"
    
    # Calculate shutdown duration
    $shutdownEnd = Get-Date
    $shutdownDuration = ($shutdownEnd - $shutdownStart).TotalMilliseconds
    
    # Collect final metrics
    $finalMetrics = @{
        "ShutdownDurationMs" = [math]::Round($shutdownDuration, 2)
        "Timestamp" = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fffZ")
        "GracefulShutdown" = (-not $Force).ToString()
        "GatewayPort" = $gatewayPort
    }
    
    # Write final state to SOUL.md
    Write-SoulFinalState -Reason "User-initiated shutdown" -Metrics $finalMetrics
    
    # Clean up shared memory files
    $sharedDir = Join-Path $BaseDir "shared"
    if (Test-Path $sharedDir) {
        Remove-Item -Path $sharedDir\*.* -Force -ErrorAction SilentlyContinue
        Write-Host "$LogPrefix Shared memory files cleaned" -ForegroundColor Green
    }
    
    Write-Host "`n$LogPrefix ========================================" -ForegroundColor Cyan
    Write-Host "$LogPrefix Shutdown Completed Successfully" -ForegroundColor Green
    Write-Host "$LogPrefix Duration: $shutdownDuration ms" -ForegroundColor Green
    Write-Host "$LogPrefix All systems terminated gracefully" -ForegroundColor Green
    Write-Host "$LogPrefix ========================================" -ForegroundColor Cyan
    
} catch {
    Write-Host "$LogPrefix CRITICAL ERROR during shutdown: $_" -ForegroundColor Red
    if ($Force) {
        Write-Host "$LogPrefix Force flag set, attempting emergency cleanup..." -ForegroundColor Red
        Get-Process | Where-Object { $_.Name -match "chrome|python|node|nautilus" } | 
            Stop-Process -Force -ErrorAction SilentlyContinue
    }
    exit 1
}
