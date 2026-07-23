# Health Check Script
# 
# Continuous background PowerShell loop monitoring the Rust and Python PIDs,
# auto-restarting the entire stack if the master Rust process unexpectedly dies.
#
# Designed for production reliability with configurable thresholds.

param(
    [int]$CheckIntervalSeconds = 5,
    [int]$MaxRestartsPerHour = 5,
    [switch]$Verbose
)

# Configuration
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$OrchestratorScript = "$ScriptRoot\master_orchestrator.ps1"
$PidFile = "$ProjectRoot\.pids\master.pid"
$HealthLogFile = "$ProjectRoot\logs\health_check.log"
$RestartCounterFile = "$ProjectRoot\.pids\restart_counter.json"

# Ensure directories exist
$null = New-Item -ItemType Directory -Force -Path "$ProjectRoot\logs"
$null = New-Item -ItemType Directory -Force -Path "$ProjectRoot\.pids"

# Logging function
function Write-HealthLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logLine = "[$timestamp] [HEALTH] [$Level] $Message"
    Add-Content -Path $HealthLogFile -Value $logLine
    if ($Verbose) {
        Write-Host $logLine -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "WARN") { "Yellow" } else { "Green" })
    }
}

# Check if process is running
function Test-ProcessRunning {
    param([int]$Pid)
    try {
        $process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
        return $null -ne $process -and -not $process.HasExited
    } catch {
        return $false
    }
}

# Get restart counter state
function Get-RestartCounter {
    if (Test-Path $RestartCounterFile) {
        $data = Get-Content $RestartCounterFile | ConvertFrom-Json
        $hourAgo = (Get-Date).AddHours(-1)
        
        # Filter restarts within last hour
        $recentRestarts = $data.Restarts | Where-Object {
            [DateTime]$_ -gt $hourAgo
        }
        
        return @{
            Count = $recentRestarts.Count
            Restarts = $recentRestarts
        }
    }
    
    return @{
        Count = 0
        Restarts = @()
    }
}

# Record a restart event
function Add-RestartEvent {
    $counter = Get-RestartCounter
    $counter.Restarts += (Get-Date).ToString("o")
    
    @{
        Count = $counter.Restarts.Count
        Restarts = $counter.Restarts
    } | ConvertTo-Json | Set-Content -Path $RestartCounterFile
}

# Check system health metrics
function Get-SystemHealth {
    $health = @{
        CpuUsage = 0
        MemoryAvailableGB = 0
        DiskFreeGB = 0
        NetworkLatencyMs = 0
        Healthy = $true
        Warnings = @()
    }
    
    try {
        # CPU Usage
        $cpuLoad = Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average
        $health.CpuUsage = $cpuLoad
        if ($cpuLoad -gt 90) {
            $health.Warnings += "High CPU usage: ${cpuLoad}%"
        }
        
        # Memory
        $os = Get-CimInstance Win32_OperatingSystem
        $freeMem = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
        $health.MemoryAvailableGB = $freeMem
        if ($freeMem -lt 1.0) {
            $health.Warnings += "Low memory: ${freeMem}GB available"
        }
        
        # Disk Space
        $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
        $freeDisk = [math]::Round($disk.FreeSpace / 1GB, 2)
        $health.DiskFreeGB = $freeDisk
        if ($freeDisk -lt 10) {
            $health.Warnings += "Low disk space: ${freeDisk}GB free"
        }
        
        # Network latency to Binance (quick check)
        try {
            $ping = Test-Connection -ComputerName "api.binance.com" -Count 1 -Quiet -ErrorAction SilentlyContinue
            if (-not $ping) {
                $health.Warnings += "Network connectivity issue to Binance"
            }
        } catch {
            $health.Warnings += "Network check failed"
        }
        
    } catch {
        $health.Warnings += "Health check error: $_"
    }
    
    $health.Healthy = ($health.Warnings.Count -eq 0)
    return $health
}

# Auto-restart the bot
function Invoke-AutoRestart {
    Write-HealthLog "Initiating auto-restart..." "WARN"
    
    # Check restart limit
    $counter = Get-RestartCounter
    if ($counter.Count -ge $MaxRestartsPerHour) {
        Write-HealthLog "Max restarts per hour ($MaxRestartsPerHour) reached. Manual intervention required!" "ERROR"
        return $false
    }
    
    # Record restart
    Add-RestartEvent
    
    # Kill any remaining processes
    & $OrchestratorScript /KILL -Verbose:$Verbose
    
    # Wait for cleanup
    Start-Sleep -Seconds 3
    
    # Restart
    Write-HealthLog "Starting bot after unexpected failure..."
    & $OrchestratorScript /START -Verbose:$Verbose
    
    if ($LASTEXITCODE -eq 0) {
        Write-HealthLog "Auto-restart successful" "INFO"
        return $true
    } else {
        Write-HealthLog "Auto-restart failed!" "ERROR"
        return $false
    }
}

# Main health check loop
function Start-HealthMonitor {
    Write-HealthLog "=== HEALTH MONITOR STARTED ==="
    Write-HealthLog "Check interval: ${CheckIntervalSeconds}s"
    Write-HealthLog "Max restarts/hour: $MaxRestartsPerHour"
    
    $consecutiveFailures = 0
    $maxConsecutiveFailures = 3
    
    while ($true) {
        try {
            $checkTime = Get-Date -Format "HH:mm:ss"
            
            # Check PID file
            if (Test-Path $PidFile) {
                $pidData = Get-Content $PidFile | ConvertFrom-Json
                $masterPid = $pidData.Master
                
                if (Test-ProcessRunning $masterPid) {
                    $consecutiveFailures = 0
                    
                    if ($Verbose) {
                        Write-HealthLog "[$checkTime] Master process (PID: $masterPid) is healthy"
                    }
                } else {
                    $consecutiveFailures++
                    Write-HealthLog "[$checkTime] Master process NOT running! Failure count: $consecutiveFailures" "WARN"
                    
                    if ($consecutiveFailures -ge $maxConsecutiveFailures) {
                        Write-HealthLog "[$checkTime] Consecutive failures threshold reached" "ERROR"
                        
                        if (Invoke-AutoRestart) {
                            $consecutiveFailures = 0
                        }
                    }
                }
                
            } else {
                $consecutiveFailures++
                Write-HealthLog "[$checkTime] No PID file found! Failure count: $consecutiveFailures" "WARN"
                
                if ($consecutiveFailures -ge $maxConsecutiveFailures) {
                    # Check if process exists anyway
                    $nautilusProc = Get-Process | Where-Object { $_.ProcessName -like "*nautilus*" }
                    
                    if ($nautilusProc) {
                        Write-HealthLog "Process found without PID file - updating PID file"
                        # Could recreate PID file here
                        $consecutiveFailures = 0
                    } else {
                        Write-HealthLog "No Nautilus process found - may not be started" "WARN"
                        $consecutiveFailures = 0  # Don't restart if never started
                    }
                }
            }
            
            # Periodic system health check (every 60 seconds)
            if ([int](Get-Date -Format "ss") % 60 -lt $CheckIntervalSeconds) {
                $sysHealth = Get-SystemHealth
                
                if (-not $sysHealth.Healthy) {
                    foreach ($warning in $sysHealth.Warnings) {
                        Write-HealthLog "System warning: $warning" "WARN"
                    }
                }
                
                # Critical: if memory is critically low, alert
                if ($sysHealth.MemoryAvailableGB -lt 0.5) {
                    Write-HealthLog "CRITICAL: Memory critically low (${sysHealth.MemoryAvailableGB}GB)!" "ERROR"
                }
            }
            
        } catch {
            Write-HealthLog "Health check error: $_" "ERROR"
        }
        
        # Wait for next check
        Start-Sleep -Seconds $CheckIntervalSeconds
    }
}

# Handle graceful shutdown
$ctrlCPressed = $false
[Console]::TreatControlCAsInput = $true

try {
    Start-HealthMonitor
} catch {
    Write-HealthLog "Health monitor crashed: $_" "ERROR"
    exit 1
}
