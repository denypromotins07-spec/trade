# =============================================================================
# Nautilus/Ray Crypto Trading Bot - Stage 55
# File 12: scripts/ray_monitor.ps1
#
# PowerShell background task parsing Ray dashboard telemetry
# Instantly triggers master /KILL sequence if Plasma object store spills to NVMe
# Enforces strict 4GB Python RAM quota for Ray workers
# Optimized for AMD Ryzen AI 5 architecture
# =============================================================================

param(
    [Parameter(Mandatory = $false)]
    [string]$Action = "Start",
    
    [Parameter(Mandatory = $false)]
    [int]$PollIntervalMs = 500,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# Configuration
# =============================================================================

$RayDashboardUrl = "http://localhost:8265"
$PlasmaStoreLimitBytes = 4GB  # Strict 4GB limit for Python/Ray
$PlasmaSpillThresholdPercent = 90  # Trigger kill at 90% capacity
$MemoryQuotaPerWorker = 1GB  # Max 1GB per worker
$MaxWorkers = 4  # Maximum number of Ray workers
$KillScriptPath = Join-Path $PSScriptRoot "master_kill.ps1"

# Logging with microsecond timestamps
function Write-RayLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.ffffff"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "CRITICAL" { "DarkRed" }
        "SUCCESS" { "Green" }
        default { "Cyan" }
    }
    Write-Host "[$timestamp] [$Level] [RayMonitor] $Message" -ForegroundColor $color
}

# =============================================================================
# Ray Dashboard API Functions
# =============================================================================

function Get-RayClusterStats {
    try {
        $response = Invoke-RestMethod -Uri "$RayDashboardUrl/api/ray_get_cluster_status" -TimeoutSec 5 -ErrorAction Stop
        return $response
    }
    catch {
        Write-RayLog "Failed to fetch cluster stats: $_" -Level "WARN"
        return $null
    }
}

function Get-PlasmaStoreUsage {
    try {
        # Query Ray internal metrics for plasma store
        $metrics = Invoke-RestMethod -Uri "$RayDashboardUrl/metrics" -TimeoutSec 5 -ErrorAction SilentlyContinue
        
        if ($metrics) {
            # Parse plasma_store_memory_used metric
            $plasmaUsed = ($metrics | Select-String "ray_object_store_memory_usage" | ForEach-Object {
                ($_ -split ' ')[-1]
            } | Measure-Object -Sum).Sum
            
            return [double]$plasmaUsed
        }
        
        return $null
    }
    catch {
        return $null
    }
}

function Get-RayWorkerMemory {
    try {
        $workers = @()
        
        # Get list of active workers from Ray
        $workerInfo = Invoke-RestMethod -Uri "$RayDashboardUrl/api/workers" -TimeoutSec 5 -ErrorAction SilentlyContinue
        
        if ($workerInfo) {
            foreach ($worker in $workerInfo.workers) {
                $workers += @{
                    WorkerId = $worker.worker_id
                    MemoryUsed = $worker.memory_used
                    Pid = $worker.pid
                }
            }
        }
        
        return $workers
    }
    catch {
        return @()
    }
}

# =============================================================================
# Memory Monitoring and Enforcement
# =============================================================================

function Test-PlasmaStoreSpill {
    param(
        [double]$CurrentUsage,
        [double]$Limit = $PlasmaStoreLimitBytes
    )
    
    if ($null -eq $CurrentUsage) {
        return $false
    }
    
    $usagePercent = ($CurrentUsage / $Limit) * 100
    
    if ($Verbose) {
        Write-RayLog "Plasma Store: $([math]::Round($CurrentUsage / 1MB, 2))MB / $([math]::Round($Limit / 1MB, 2))MB ($([math]::Round($usagePercent, 1))%)"
    }
    
    # Check if approaching spill threshold
    if ($usagePercent -ge $PlasmaSpillThresholdPercent) {
        Write-RayLog "CRITICAL: Plasma store at $([math]::Round($usagePercent, 1))% - imminent spill to NVMe!" -Level "CRITICAL"
        return $true
    }
    
    return $false
}

function Test-WorkerMemoryQuotas {
    param([array]$Workers)
    
    $violations = @()
    
    foreach ($worker in $Workers) {
        if ($worker.MemoryUsed -gt $MemoryQuotaPerWorker) {
            $violations += @{
                WorkerId = $worker.WorkerId
                MemoryUsed = $worker.MemoryUsed
                Limit = $MemoryQuotaPerWorker
                Pid = $worker.Pid
            }
            
            Write-RayLog "Worker $($worker.WorkerId) exceeds memory quota: $([math]::Round($worker.MemoryUsed / 1MB, 2))MB > $([math]::Round($MemoryQuotaPerWorker / 1MB, 2))MB" -Level "WARN"
        }
    }
    
    return $violations
}

function Test-DiskSpillDetected {
    # Check for Ray spill files on disk (indicates OOM condition)
    $spillPaths = @(
        "$env:TEMP\ray_spill",
        "$env:LOCALAPPDATA\ray\spill",
        "/tmp/ray/spill"
    )
    
    foreach ($path in $spillPaths) {
        if (Test-Path $path) {
            $files = Get-ChildItem -Path $path -File -ErrorAction SilentlyContinue
            if ($files.Count -gt 0) {
                $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
                if ($totalSize -gt 0) {
                    Write-RayLog "DISK SPILL DETECTED: $([math]::Round($totalSize / 1MB, 2))MB spilled to $path" -Level "CRITICAL"
                    return $true
                }
            }
        }
    }
    
    return $false
}

# =============================================================================
# Kill Sequence Trigger
# =============================================================================

function Invoke-MasterKill {
    param(
        [string]$Reason,
        [hashtable]$Context = @{}
    )
    
    Write-RayLog "TRIGGERING MASTER KILL SEQUENCE" -Level "CRITICAL"
    Write-RayLog "Reason: $Reason" -Level "CRITICAL"
    
    # Log context information
    foreach ($key in $Context.Keys) {
        Write-RayLog "  $key`: $($Context[$key])" -Level "CRITICAL"
    }
    
    # Execute master kill script
    if (Test-Path $KillScriptPath) {
        try {
            & $KillScriptPath -Reason $Reason -Source "RayMonitor"
            Write-RayLog "Master kill sequence executed successfully" -Level "SUCCESS"
        }
        catch {
            Write-RayLog "Failed to execute master kill: $_" -Level "ERROR"
            
            # Fallback: terminate Ray processes directly
            Write-RayLog "Executing fallback process termination..." -Level "WARN"
            Stop-Process -Name "raylet", "plasma_store", "gcs_server" -Force -ErrorAction SilentlyContinue
        }
    }
    else {
        Write-RayLog "Kill script not found at $KillScriptPath - terminating Ray directly" -Level "WARN"
        Stop-Process -Name "raylet", "plasma_store", "gcs_server", "python" -Force -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# Main Monitoring Loop
# =============================================================================

function Start-RayMonitor {
    Write-RayLog "Starting Ray Monitor background task"
    Write-RayLog "Configuration:"
    Write-RayLog "  - Plasma Store Limit: $([math]::Round($PlasmaStoreLimitBytes / 1GB, 2))GB"
    Write-RayLog "  - Spill Threshold: ${PlasmaSpillThresholdPercent}%"
    Write-RayLog "  - Worker Memory Quota: $([math]::Round($MemoryQuotaPerWorker / 1MB, 2))MB"
    Write-RayLog "  - Max Workers: $MaxWorkers"
    Write-RayLog "  - Poll Interval: ${PollIntervalMs}ms"
    
    $monitoring = $true
    $killTriggered = $false
    
    # Register Ctrl+C handler
    [Console]::TreatControlCAsInput = $true
    
    while ($monitoring -and -not $killTriggered) {
        try {
            # Check for Ctrl+C
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true)
                if ($key.Key -eq 'C' -and $key.Modifiers -eq 'Ctrl') {
                    Write-RayLog "Received interrupt - stopping monitor" -Level "WARN"
                    $monitoring = $false
                    break
                }
            }
            
            # Get current state
            $plasmaUsage = Get-PlasmaStoreUsage
            $workers = Get-RayWorkerMemory
            
            # Check for plasma spill condition
            if (Test-PlasmaStoreSpill -CurrentUsage $plasmaUsage) {
                Invoke-MasterKill -Reason "Plasma store memory exceeded threshold - imminent NVMe spill" -Context @{
                    "PlasmaUsage" = "$([math]::Round($plasmaUsage / 1MB, 2))MB"
                    "Limit" = "$([math]::Round($PlasmaStoreLimitBytes / 1MB, 2))MB"
                }
                $killTriggered = $true
                break
            }
            
            # Check for disk spill
            if (Test-DiskSpillDetected) {
                Invoke-MasterKill -Reason "Ray object store spill detected on disk - OOM condition" -Context @{
                    "Timestamp" = Get-Date -Format "o"
                }
                $killTriggered = $true
                break
            }
            
            # Check worker memory quotas
            $violations = Test-WorkerMemoryQuotas -Workers $workers
            if ($violations.Count -gt 0) {
                $criticalViolation = $violations | Where-Object { $_.MemoryUsed -gt ($MemoryQuotaPerWorker * 1.5) }
                if ($criticalViolation) {
                    Invoke-MasterKill -Reason "Critical worker memory violation - exceeding 150% quota" -Context @{
                        "WorkerId" = $criticalViolation.WorkerId
                        "MemoryUsed" = "$([math]::Round($criticalViolation.MemoryUsed / 1MB, 2))MB"
                    }
                    $killTriggered = $true
                    break
                }
            }
            
            # Status update every 10 seconds
            if ($Verbose) {
                Write-RayLog "Monitoring active - Plasma: $([math]::Round(($plasmaUsage ?? 0) / 1MB, 2))MB, Workers: $($workers.Count)"
            }
            
            Start-Sleep -Milliseconds $PollIntervalMs
        }
        catch {
            Write-RayLog "Monitor loop error: $_" -Level "ERROR"
            Start-Sleep -Seconds 1
        }
    }
    
    Write-RayLog "Ray Monitor stopped" -Level ($killTriggered ? "WARN" : "INFO")
}

# =============================================================================
# Main Execution
# =============================================================================

try {
    switch ($Action) {
        "Start" {
            Start-RayMonitor
        }
        
        "Status" {
            Write-RayLog "Checking Ray cluster status..."
            $plasmaUsage = Get-PlasmaStoreUsage
            $workers = Get-RayWorkerMemory
            
            Write-RayLog "Plasma Store Usage: $([math]::Round(($plasmaUsage ?? 0) / 1MB, 2))MB / $([math]::Round($PlasmaStoreLimitBytes / 1MB, 2))MB"
            Write-RayLog "Active Workers: $($workers.Count)"
            
            foreach ($w in $workers) {
                $status = if ($w.MemoryUsed -gt $MemoryQuotaPerWorker) { "OVER QUOTA" } else { "OK" }
                Write-RayLog "  Worker $($w.WorkerId): $([math]::Round($w.MemoryUsed / 1MB, 2))MB [$status]"
            }
            
            $spillDetected = Test-DiskSpillDetected
            Write-RayLog "Disk Spill Detected: $spillDetected" -Level ($spillDetected ? "CRITICAL" : "INFO")
        }
        
        "Test-Kill" {
            Write-RayLog "Testing kill sequence (dry run)..."
            Write-RayLog "Would trigger master kill with reason: Test triggered"
            # Don't actually kill in test mode
        }
        
        default {
            throw "Unknown action: $Action. Valid actions: Start, Status, Test-Kill"
        }
    }
}
catch {
    Write-RayLog "Fatal error: $_" -Level "ERROR"
    exit 1
}

Write-RayLog "Ray Monitor script completed"
