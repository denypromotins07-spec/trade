<#
.SYNOPSIS
    Health Monitor Watchdog for Nautilus/Ray HFT Bot - Stage 54
    
.DESCRIPTION
    Background watchdog loop monitoring the exact CPU and RAM footprint
    of Rust and Python PIDs, auto-triggering /KILL if the 8GB limit is breached.
    
.PARAMETER PollIntervalMs
    Milliseconds between health checks (default: 1000)
    
.PARAMETER RamLimitGB
    Global RAM ceiling in GB (default: 8)
    
.EXAMPLE
    .\health_monitor.ps1
    .\health_monitor.ps1 -PollIntervalMs 500 -RamLimitGB 8
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [int]$PollIntervalMs = 1000,
    
    [Parameter(Mandatory = $false)]
    [int]$RamLimitGB = 8,
    
    [Parameter(Mandatory = $false)]
    [switch]$RunAsDaemon
)

# =============================================================================
# CONFIGURATION
# =============================================================================
$ErrorActionPreference = "Continue"
$WarningPreference = "Continue"

# Memory limits
$GLOBAL_RAM_LIMIT_BYTES = $RamLimitGB * 1GB
$PYTHON_RAM_QUOTA_BYTES = 4 * 1GB
$RUST_RAM_QUOTA_BYTES = 3 * 1GB

# Thresholds for warnings
$WARNING_THRESHOLD = 0.85      # 85% usage triggers warning
$CRITICAL_THRESHOLD = 0.95     # 95% usage triggers critical alert
$BREACH_THRESHOLD = 1.0        # 100% triggers KILL

# Process patterns to monitor
$MonitoredProcesses = @{
    Rust = @("nautilus_engine", "pgo_instrument")
    Python = @("python.*ray", "python.*nautilus", "gcs_server", "raylet")
}

# Log file
$LogPath = ".\logs\health_monitor.log"
New-Item -ItemType Directory -Path (Split-Path $LogPath) -Force | Out-Null

# Metrics storage
$Global:MetricsHistory = New-Object System.Collections.Generic.List[object]
$Global:LastKillTrigger = $null

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

function Write-MonitorLog {
    <#
    .SYNOPSIS
        Writes timestamped log entries
    #>
    param(
        [string]$Message,
        [ValidateSet("INFO", "WARN", "CRITICAL", "ERROR")]
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Console output with color
    $color = switch ($Level) {
        "INFO" { "White" }
        "WARN" { "Yellow" }
        "CRITICAL" { "Red" }
        "ERROR" { "DarkRed" }
    }
    
    Write-Host $logEntry -ForegroundColor $color
    
    # File logging
    try {
        Add-Content -Path $LogPath -Value $logEntry
    } catch {}
}

# =============================================================================
# PROCESS DISCOVERY AND MONITORING
# =============================================================================

function Get-MonitoredProcesses {
    <#
    .SYNOPSIS
        Discovers all processes that should be monitored
    #>
    [OutputType([System.Collections.Generic.List[System.Diagnostics.Process]])]
    param()
    
    $processes = New-Object System.Collections.Generic.List[System.Diagnostics.Process]
    
    foreach ($category in $MonitoredProcesses.Keys) {
        foreach ($pattern in $MonitoredProcesses[$category]) {
            $matches = Get-Process | Where-Object { $_.Name -match $pattern }
            foreach ($proc in $matches) {
                $processes.Add($proc) | Out-Null
            }
        }
    }
    
    return $processes
}

function Get-ProcessMemoryInfo {
    <#
    .SYNOPSIS
        Gets detailed memory information for a process
    #>
    param([System.Diagnostics.Process]$Process)
    
    try {
        $memInfo = $Process.WorkingSet64
        $privateMem = $Process.PrivateMemorySize64
        
        return @{
            WorkingSet = $memInfo
            PrivateMemory = $privateMem
            PID = $Process.Id
            Name = $Process.Name
        }
    } catch {
        return $null
    }
}

function Get-ProcessCPUInfo {
    <#
    .SYNOPSIS
        Gets CPU usage information for a process
    #>
    param([System.Diagnostics.Process]$Process)
    
    try {
        $cpuTime = $Process.TotalProcessorTime.TotalMilliseconds
        $startTime = $Process.StartTime
        
        return @{
            CPUTimeMs = $cpuTime
            StartTime = $startTime
            PID = $Process.Id
        }
    } catch {
        return $null
    }
}

# =============================================================================
# HEALTH CHECK CORE
# =============================================================================

function Invoke-HealthCheck {
    <#
    .SYNOPSIS
        Performs a single health check iteration
    #>
    [OutputType([hashtable])]
    param()
    
    $result = @{
        Timestamp = Get-Date
        TotalRAM_Bytes = 0
        TotalRAM_GB = 0
        Processes = @()
        Status = "OK"
        KillRequired = $false
    }
    
    $processes = Get-MonitoredProcesses
    $totalRAM = 0
    
    foreach ($proc in $processes) {
        $memInfo = Get-ProcessMemoryInfo -Process $proc
        $cpuInfo = Get-ProcessCPUInfo -Process $proc
        
        if ($memInfo) {
            $totalRAM += $memInfo.WorkingSet
            
            $procData = @{
                PID = $memInfo.PID
                Name = $memInfo.Name
                WorkingSet_MB = [math]::Round($memInfo.WorkingSet / 1MB, 2)
                PrivateMB = [math]::Round($memInfo.PrivateMemory / 1MB, 2)
                CPUTime_Ms = if ($cpuInfo) { $cpuInfo.CPUTimeMs } else { 0 }
            }
            
            $result.Processes += $procData
        }
    }
    
    $result.TotalRAM_Bytes = $totalRAM
    $result.TotalRAM_GB = [math]::Round($totalRAM / 1GB, 3)
    
    # Calculate usage ratio
    $usageRatio = $totalRAM / $GLOBAL_RAM_LIMIT_BYTES
    
    # Determine status
    if ($usageRatio >= $BREACH_THRESHOLD) {
        $result.Status = "BREACH"
        $result.KillRequired = $true
    } elseif ($usageRatio >= $CRITICAL_THRESHOLD) {
        $result.Status = "CRITICAL"
    } elseif ($usageRatio >= $WARNING_THRESHOLD) {
        $result.Status = "WARNING"
    }
    
    # Store in history
    $Global:MetricsHistory.Add($result)
    
    # Keep only last 1000 entries
    if ($Global:MetricsHistory.Count -gt 1000) {
        $Global:MetricsHistory.RemoveAt(0)
    }
    
    return $result
}

# =============================================================================
# ALERT AND RESPONSE FUNCTIONS
# =============================================================================

function Invoke-Alert {
    <#
    .SYNOPSIS
        Issues alerts based on health status
    #>
    param([hashtable]$HealthResult)
    
    switch ($HealthResult.Status) {
        "WARNING" {
            Write-MonitorLog "WARNING: RAM at $($HealthResult.TotalRAM_GB)GB / ${RamLimitGB}GB ($([math]::Round($HealthResult.TotalRAM_Bytes / $GLOBAL_RAM_LIMIT_BYTES * 100, 1))%)" -Level WARN
        }
        
        "CRITICAL" {
            Write-MonitorLog "CRITICAL: RAM at $($HealthResult.TotalRAM_GB)GB / ${RamLimitGB}GB - Immediate action required!" -Level CRITICAL
            
            # Sound system alert (Windows)
            try {
                [Console]::Beep(800, 200)
                [Console]::Beep(1000, 200)
            } catch {}
        }
        
        "BREACH" {
            Write-MonitorLog "BREACH DETECTED: RAM at $($HealthResult.TotalRAM_GB)GB exceeds ${RamLimitGB}GB limit!" -Level CRITICAL
            
            # Triple beep alert
            try {
                [Console]::Beep(1200, 300)
                Start-Sleep -Milliseconds 100
                [Console]::Beep(1200, 300)
                Start-Sleep -Milliseconds 100
                [Console]::Beep(1200, 500)
            } catch {}
            
            return $true
        }
    }
    
    return $false
}

function Invoke-AutoKill {
    <#
    .SYNOPSIS
        Triggers the master kill script when limits are breached
    #>
    Write-MonitorLog "AUTO-KILL TRIGGERED: Breached ${RamLimitGB}GB RAM limit" -Level CRITICAL
    
    # Rate limit kills (prevent rapid cycling)
    if ($Global:LastKillTrigger) {
        $timeSinceLastKill = (Get-Date) - $Global:LastKillTrigger
        
        if ($timeSinceLastKill.TotalSeconds -lt 60) {
            Write-MonitorLog "RATE LIMITED: Kill triggered too recently ($($timeSinceLastKill.TotalSeconds)s ago)" -Level WARN
            return $false
        }
    }
    
    $Global:LastKillTrigger = Get-Date
    
    # Execute master kill
    $killScript = ".\scripts\master_kill.ps1"
    
    if (Test-Path $killScript) {
        Write-MonitorLog "Executing: $killScript" -Level INFO
        
        try {
            & $killScript -Force
            Write-MonitorLog "Master kill completed" -Level INFO
            return $true
        } catch {
            Write-MonitorLog "Master kill failed: $_" -Level ERROR
            return $false
        }
    } else {
        Write-MonitorLog "Master kill script not found at: $killScript" -Level ERROR
        
        # Emergency fallback: kill all monitored processes directly
        Write-MonitorLog "Emergency fallback: Direct process termination" -Level WARN
        
        $processes = Get-MonitoredProcesses
        foreach ($proc in $processes) {
            try {
                Stop-Process -Id $proc.Id -Force
                Write-MonitorLog "Killed PID $($proc.Id) ($($proc.Name))" -Level INFO
            } catch {}
        }
        
        return $true
    }
}

# =============================================================================
# METRICS REPORTING
# =============================================================================

function Get-MetricsReport {
    <#
    .SYNOPSIS
        Generates a summary report of collected metrics
    #>
    if ($Global:MetricsHistory.Count -eq 0) {
        return "No metrics collected yet"
    }
    
    $latest = $Global:MetricsHistory[-1]
    $avgRAM = ($Global:MetricsHistory | Measure-Object -Property TotalRAM_Bytes -Average).Average
    $maxRAM = ($Global:MetricsHistory | Measure-Object -Property TotalRAM_Bytes -Maximum).Maximum
    
    $report = @"
=== HEALTH MONITOR METRICS REPORT ===
Monitoring Duration: $($Global:MetricsHistory.Count * $PollIntervalMs / 1000)s
Samples Collected:   $($Global:MetricsHistory.Count)

Current Status:
  RAM Usage:    $($latest.TotalRAM_GB) GB / ${RamLimitGB} GB
  Status:       $($latest.Status)
  Processes:    $($latest.Processes.Count)

Historical Averages:
  Avg RAM:      $([math]::Round($avgRAM / 1GB, 3)) GB
  Max RAM:      $([math]::Round($maxRAM / 1GB, 3)) GB

Process Details:
"@
    
    foreach ($proc in $latest.Processes) {
        $report += "`n  PID $($proc.PID): $($proc.Name) - $($proc.WorkingSet_MB) MB"
    }
    
    $report += "`n====================================="
    
    return $report
}

# =============================================================================
# MAIN MONITOR LOOP
# =============================================================================

function Start-HealthMonitor {
    <#
    .SYNOPSIS
        Main monitoring loop
    #>
    Write-MonitorLog "Health Monitor starting..." -Level INFO
    Write-MonitorLog "Configuration:" -Level INFO
    Write-MonitorLog "  Poll Interval: ${PollIntervalMs}ms" -Level INFO
    Write-MonitorLog "  RAM Limit: ${RamLimitGB}GB" -Level INFO
    Write-MonitorLog "  Warning Threshold: $([math]::Round($WARNING_THRESHOLD * 100, 0))%" -Level INFO
    Write-MonitorLog "  Critical Threshold: $([math]::Round($CRITICAL_THRESHOLD * 100, 0))%" -Level INFO
    Write-MonitorLog "  Breach Threshold: $([math]::Round($BREACH_THRESHOLD * 100, 0))%" -Level INFO
    
    $iterationCount = 0
    
    while ($true) {
        $iterationCount++
        
        # Perform health check
        $health = Invoke-HealthCheck
        
        # Log periodic summary
        if ($iterationCount % 60 -eq 0) {
            Write-MonitorLog "Health check #$iterationCount - RAM: $($health.TotalRAM_GB)GB - Status: $($health.Status)" -Level INFO
        }
        
        # Check for alerts
        $breachDetected = Invoke-Alert -HealthResult $health
        
        # Trigger auto-kill if breach detected
        if ($breachDetected -or $health.KillRequired) {
            Write-MonitorLog "Initiating auto-kill sequence..." -Level CRITICAL
            Invoke-AutoKill
            
            if ($RunAsDaemon) {
                Write-MonitorLog "Restarting monitor after kill..." -Level INFO
                Start-Sleep -Seconds 5
            } else {
                Write-MonitorLog "Monitor exiting after kill" -Level INFO
                break
            }
        }
        
        # Wait for next poll
        Start-Sleep -Milliseconds $PollIntervalMs
    }
}

# =============================================================================
# SIGNAL HANDLING
# =============================================================================

# Handle Ctrl+C gracefully
$cancelEvent = Register-ObjectEvent -InputObject ([System.Console]) `
    -EventName CancelKeyPress `
    -Action {
        Write-MonitorLog "Received shutdown signal" -Level INFO
        Write-MonitorLog (Get-MetricsReport) -Level INFO
        $global:monitorRunning = $false
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

Write-Host "`n################################################################" -ForegroundColor Cyan
Write-Host "#      NAUTILUS/RAY HFT BOT - HEALTH MONITOR v5.4             #" -ForegroundColor Cyan
Write-Host "#      RAM Watchdog | Auto-Kill | ${RamLimitGB}GB Ceiling              #" -ForegroundColor Cyan
Write-Host "################################################################`n" -ForegroundColor Cyan

try {
    $global:monitorRunning = $true
    Start-HealthMonitor
} catch {
    Write-MonitorLog "Monitor crashed: $_" -Level ERROR
    exit 1
} finally {
    Unregister-Event -SubscriptionId $cancelEvent.Id -ErrorAction SilentlyContinue
    Write-MonitorLog "Health Monitor stopped" -Level INFO
}
