# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - HEALTH MONITOR
# =============================================================================
# Stage 54: Background Watchdog Loop
# Purpose: Monitor exact CPU and RAM footprint of Rust and Python PIDs,
#          auto-trigger /KILL if the 8GB limit is breached
# Target: AMD Ryzen AI 5 with strict 8GB global RAM ceiling
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [int]$WatchPID = -1,
    
    [Parameter(Mandatory = $false)]
    [int]$RamLimitMB = 8192,
    
    [Parameter(Mandatory = $false)]
    [int]$CheckIntervalSeconds = 5,
    
    [Parameter(Mandatory = $false)]
    [double]$CpuThresholdPercent = 90.0,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_ROOT
$LOG_FILE = Join-Path $PROJECT_ROOT "logs/health_monitor_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"
$PID_FILE = Join-Path $PROJECT_ROOT ".pids/master.pids"
$KILL_SCRIPT = Join-Path $SCRIPT_ROOT "master_kill.ps1"

# Memory thresholds
$MEMORY_WARNING_THRESHOLD_MB = $RamLimitMB * 0.85   # 85% = warning
$MEMORY_CRITICAL_THRESHOLD_MB = $RamLimitMB * 0.95  # 95% = critical
$MEMORY_KILL_THRESHOLD_MB = $RamLimitMB             # 100% = kill

# CPU thresholds
$CPU_WARNING_THRESHOLD = 80.0
$CPU_CRITICAL_THRESHOLD = 95.0

# Alert cooldown (prevent alert spam)
$ALERT_COOLDOWN_SECONDS = 60

# =============================================================================
# GLOBAL STATE
# =============================================================================
$Global:LastWarningTime = $null
$Global:LastCriticalTime = $null
$Global:ConsecutiveViolations = 0
$Global:MonitoredPIDs = @()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "WARN") { "Yellow" } elseif ($Level -eq "CRITICAL") { "Magenta" } elseif ($Level -eq "SUCCESS") { "Green" } else { "White" })
    
    if ($Verbose) {
        $logDir = Split-Path $LOG_FILE -Parent
        if (-not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        }
        Add-Content -Path $LOG_FILE -Value $logEntry
    }
}

function Get-ProcessMemoryInfo {
    <#
    .SYNOPSIS
        Gets detailed memory information for a process including working set,
        private bytes, and virtual memory.
    #>
    param([int]$ProcessId)
    
    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $proc) {
            return $null
        }
        
        return @{
            ProcessId      = $proc.Id
            ProcessName    = $proc.ProcessName
            WorkingSetMB   = [math]::Round($proc.WorkingSet64 / 1MB, 2)
            PrivateBytesMB = [math]::Round($proc.PrivateMemorySize64 / 1MB, 2)
            VirtualMB      = [math]::Round($proc.VirtualMemorySize64 / 1MB, 2)
            CPUSeconds     = [math]::Round($proc.TotalProcessorTime.TotalSeconds, 2)
            Threads        = $proc.Threads.Count
            Handles        = $proc.HandleCount
        }
    } catch {
        return $null
    }
}

function Get-SystemMemoryInfo {
    <#
    .SYNOPSIS
        Gets system-wide memory statistics.
    #>
    $os = Get-CimInstance Win32_OperatingSystem
    
    $totalMem = $os.TotalVisibleMemorySize / 1MB
    $freeMem = $os.FreePhysicalMemory / 1MB
    $usedMem = $totalMem - $freeMem
    
    return @{
        TotalMB       = [math]::Round($totalMem, 2)
        UsedMB        = [math]::Round($usedMem, 2)
        FreeMB        = [math]::Round($freeMem, 2)
        PercentUsed   = [math]::Round(($usedMem / $totalMem) * 100, 2)
        AvailableMB   = [math]::Round($freeMem, 2)
    }
}

function Get-ProcessCPUUsage {
    <#
    .SYNOPSIS
        Calculates current CPU usage percentage for a process.
        Uses two samples to compute the rate.
    #>
    param([int]$ProcessId, [int]$SampleIntervalMs = 1000)
    
    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $proc) {
            return 0
        }
        
        # First sample
        $cpu1 = $proc.TotalProcessorTime.TotalMilliseconds
        $time1 = [DateTime]::Now
        
        Start-Sleep -Milliseconds $SampleIntervalMs
        
        # Refresh process info
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $proc) {
            return 0
        }
        
        # Second sample
        $cpu2 = $proc.TotalProcessorTime.TotalMilliseconds
        $time2 = [DateTime]::Now
        
        # Calculate CPU percentage
        $elapsedMs = ($time2 - $time1).TotalMilliseconds
        $cpuElapsedMs = $cpu2 - $cpu1
        
        # Account for multiple cores
        $numProcessors = [Environment]::ProcessorCount
        $cpuPercent = ($cpuElapsedMs / $elapsedMs) * 100 / $numProcessors
        
        return [math]::Min([math]::Max($cpuPercent, 0), 100)
    } catch {
        return 0
    }
}

function Send-Alert {
    param(
        [string]$AlertType,
        [string]$Message,
        [string]$Severity = "WARNING"
    )
    
    $now = Get-Date
    
    # Check cooldown
    $cooldown = $ALERT_COOLDOWN_SECONDS
    $lastAlertTime = switch ($Severity) {
        "CRITICAL" { $Global:LastCriticalTime }
        "WARNING"  { $Global:LastWarningTime }
        default    { $Global:LastWarningTime }
    }
    
    if ($lastAlertTime -and ($now - $lastAlertTime).TotalSeconds -lt $cooldown) {
        return  # Suppress duplicate alerts within cooldown period
    }
    
    # Update last alert time
    switch ($Severity) {
        "CRITICAL" { $Global:LastCriticalTime = $now }
        "WARNING"  { $Global:LastWarningTime = $now }
    }
    
    Write-Log "ALERT [$Severity]: $AlertType - $Message" -ForegroundColor $(if ($Severity -eq "CRITICAL") { "Magenta" } else { "Yellow" })
    
    # In production, this could send to external monitoring systems
    # Example: Send to Slack, PagerDuty, etc.
}

function Invoke-KillSequence {
    <#
    .SYNOPSIS
        Triggers the master kill script when memory limits are breached.
    #>
    param([string]$Reason)
    
    Write-Log "INITIATING EMERGENCY KILL: $Reason" -ForegroundColor Red
    
    if (Test-Path $KILL_SCRIPT) {
        & $KILL_SCRIPT -Force -Verbose:$Verbose
    } else {
        Write-Log "Kill script not found, force-killing processes..." -Level "ERROR"
        
        foreach ($pid in $Global:MonitoredPIDs) {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
    
    exit 1
}

function Initialize-Monitoring {
    <#
    .SYNOPSIS
        Discovers and initializes the list of PIDs to monitor.
    #>
    Write-Log "Initializing health monitoring..."
    
    $pids = @()
    
    # Use provided PID if specified
    if ($WatchPID -gt 0) {
        $pids += $WatchPID
    }
    
    # Load PIDs from file
    if (Test-Path $PID_FILE) {
        try {
            $pidData = Get-Content $PID_FILE -Raw | ConvertFrom-Json
            
            if ($pidData.RustPID) {
                $pids += $pidData.RustPID
            }
            
            if ($pidData.RayPIDs) {
                $pids += $pidData.RayPIDs
            }
        } catch {
            Write-Log "Failed to read PID file: $_" -Level "WARN"
        }
    }
    
    # Find nautilus-ray-bot processes
    $procs = Get-Process -Name "nautilus-ray-bot*" -ErrorAction SilentlyContinue
    foreach ($proc in $procs) {
        if ($pids -notcontains $proc.Id) {
            $pids += $proc.Id
        }
    }
    
    # Find Ray processes related to our bot
    $rayProcs = Get-Process -Name "*python*" -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_.CommandLine -like "*ray*" -or $_.CommandLine -like "*nautilus*"
        } catch {
            $false
        }
    }
    foreach ($proc in $rayProcs) {
        if ($pids -notcontains $proc.Id) {
            $pids += $proc.Id
        }
    }
    
    # Remove duplicates and filter valid processes
    $Global:MonitoredPIDs = $pids | Select-Object -Unique | Where-Object {
        Get-Process -Id $_ -ErrorAction SilentlyContinue
    }
    
    if ($Global:MonitoredPIDs.Count -eq 0) {
        Write-Log "No processes found to monitor. Waiting for startup..." -Level "WARN"
        return $false
    }
    
    Write-Log "Monitoring $($Global:MonitoredPIDs.Count) PIDs: $($Global:MonitoredPIDs -join ', ')" -ForegroundColor Green
    return $true
}

# =============================================================================
# MAIN MONITORING LOOP
# =============================================================================

function Start-HealthMonitor {
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "NAUTILUS/RAY HEALTH MONITOR" -ForegroundColor Cyan
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "Timestamp:    $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Log "RAM Limit:    ${RamLimitMB}MB"
    Write-Log "Check Interval: ${CheckIntervalSeconds}s"
    Write-Log ""
    
    # Initialize
    if (-not (Initialize-Monitoring)) {
        Write-Log "Waiting 30 seconds for processes to start..."
        Start-Sleep -Seconds 30
        
        if (-not (Initialize-Monitoring)) {
            Write-Log "No processes found after waiting. Exiting." -Level "WARN"
            exit 0
        }
    }
    
    Write-Log "Starting monitoring loop..." -ForegroundColor Green
    Write-Log ""
    
    $iterationCount = 0
    
    while ($true) {
        $iterationCount++
        $checkTime = Get-Date
        
        # Collect metrics for all monitored processes
        $totalMemoryMB = 0
        $totalCPU = 0
        $violations = @()
        
        foreach ($pid in $Global:MonitoredPIDs) {
            $memInfo = Get-ProcessMemoryInfo -ProcessId $pid
            $cpuUsage = Get-ProcessCPUUsage -ProcessId $pid
            
            if ($memInfo) {
                $totalMemoryMB += $memInfo.WorkingSetMB
                
                # Check memory threshold
                if ($memInfo.WorkingSetMB -gt $MEMORY_KILL_THRESHOLD_MB) {
                    $violations += "PID $pid memory: $($memInfo.WorkingSetMB)MB > ${MEMORY_KILL_THRESHOLD_MB}MB"
                } elseif ($memInfo.WorkingSetMB -gt $MEMORY_CRITICAL_THRESHOLD_MB) {
                    $violations += "PID $pid memory CRITICAL: $($memInfo.WorkingSetMB)MB"
                } elseif ($memInfo.WorkingSetMB -gt $MEMORY_WARNING_THRESHOLD_MB) {
                    $violations += "PID $pid memory WARNING: $($memInfo.WorkingSetMB)MB"
                }
                
                # Check CPU threshold
                if ($cpuUsage -gt $CPU_CRITICAL_THRESHOLD) {
                    $violations += "PID $pid CPU CRITICAL: $([math]::Round($cpuUsage, 1))%"
                    $totalCPU += $cpuUsage
                }
            }
        }
        
        # Get system memory info
        $sysMem = Get-SystemMemoryInfo
        
        # Log status every 10 iterations (or if verbose)
        if ($Verbose -or ($iterationCount % 10 -eq 0)) {
            Write-Log "Status: Total=${totalMemoryMB}MB | System=$($sysMem.PercentUsed)% | PIDs=$($Global:MonitoredPIDs.Count)"
        }
        
        # Handle violations
        if ($violations.Count -gt 0) {
            $Global:ConsecutiveViolations++
            
            foreach ($violation in $violations) {
                if ($violation -like "*CRITICAL*") {
                    Send-Alert -AlertType "RESOURCE" -Message $violation -Severity "CRITICAL"
                } elseif ($violation -like "*WARNING*") {
                    Send-Alert -AlertType "RESOURCE" -Message $violation -Severity "WARNING"
                } else {
                    # This is a kill-level violation
                    Send-Alert -AlertType "RESOURCE" -Message $violation -Severity "CRITICAL"
                    Invoke-KillSequence -Reason $violation
                }
            }
            
            # Check for consecutive violations pattern
            if ($Global:ConsecutiveViolations -ge 3) {
                Write-Log "Three consecutive violations detected - possible memory leak" -Level "CRITICAL"
                Invoke-KillSequence -Reason "Consecutive resource violations (possible memory leak)"
            }
        } else {
            $Global:ConsecutiveViolations = 0
        }
        
        # Check total system memory
        if ($sysMem.UsedMB -gt $MEMORY_KILL_THRESHOLD_MB) {
            Write-Log "SYSTEM MEMORY EXCEEDED: $($sysMem.UsedMB)MB > ${MEMORY_KILL_THRESHOLD_MB}MB" -ForegroundColor Red
            Invoke-KillSequence -Reason "System memory limit exceeded"
        }
        
        # Verify monitored processes are still running
        $alivePIDs = $Global:MonitoredPIDs | Where-Object {
            Get-Process -Id $_ -ErrorAction SilentlyContinue
        }
        
        if ($alivePIDs.Count -eq 0) {
            Write-Log "All monitored processes have exited" -Level "WARN"
            Write-Log "Health monitor exiting"
            exit 0
        }
        
        # Update monitored PIDs list
        $Global:MonitoredPIDs = $alivePIDs
        
        # Wait for next check
        Start-Sleep -Seconds $CheckIntervalSeconds
    }
}

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Write-Log "Health monitor received exit signal" -Level "WARN"
}

# Trap for unhandled errors
trap {
    Write-Log "Unhandled error in health monitor: $_" -Level "ERROR"
    continue
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try {
    Start-HealthMonitor
} catch {
    Write-Log "Health monitor failed: $_" -Level "ERROR"
    exit 1
}
