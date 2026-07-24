# Degradation Monitor: PowerShell monitor tracking OS-level fallbacks, thermal 
# shedding events, and memory evictions. Pushes critical alerts to frontend UI
# to inform user of automated system preservation actions.
# Compatible with /START and /KILL PowerShell orchestration.

param(
    [string]$ConfigPath = "config/degradation_monitor.json",
    [int]$PollIntervalMs = 1000,
    [string]$LogPath = "logs/degradation_monitor.log",
    [switch]$Verbose
)

# Ensure log directory exists
$logDir = Split-Path $LogPath -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

# Configuration defaults
$DefaultConfig = @{
    CpuTempThresholdCelsius = 85
    MemoryThresholdPercent = 90
    CriticalMemoryThresholdPercent = 95
    AlertCooldownSeconds = 30
    MaxAlertsPerMinute = 10
    WebSocketUrl = "ws://localhost:8080/ws/alerts"
    EnableEmailAlerts = $false
    EmailRecipient = ""
}

# Load or create configuration
if (Test-Path $ConfigPath) {
    $Config = Get-Content $ConfigPath | ConvertFrom-Json
} else {
    $Config = $DefaultConfig
    $Config | ConvertTo-Json | Set-Content $ConfigPath
}

# Global state
$Script:LastAlertTime = @{}
$Script:AlertCountThisMinute = 0
$Script:WebSocketClient = $null
$Script:IsRunning = $true
$Script:TotalThermalEvents = 0
$Script:TotalMemoryEvents = 0
$Script:TotalCpuEvents = 0

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Write to file
    Add-Content -Path $LogPath -Value $logEntry
    
    # Write to console if verbose
    if ($Verbose) {
        switch ($Level) {
            "ERROR" { Write-Host $logEntry -ForegroundColor Red }
            "WARN" { Write-Host $logEntry -ForegroundColor Yellow }
            "INFO" { Write-Host $logEntry -ForegroundColor Green }
            default { Write-Host $logEntry }
        }
    }
}

# Send alert to frontend via WebSocket
function Send-FrontendAlert {
    param(
        [string]$AlertType,
        [string]$Severity,
        [string]$Message,
        [hashtable]$Details = @{}
    )
    
    $alertData = @{
        type = $AlertType
        severity = $Severity
        message = $Message
        timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        details = $Details
    } | ConvertTo-Json
    
    try {
        if ($Script:WebSocketClient -and $Script:WebSocketClient.ReadyState -eq 1) {
            $Script:WebSocketClient.Send($alertData)
        }
        
        # Also write to alert log for persistence
        $alertLogPath = "logs/alerts_$(Get-Date -Format 'yyyyMMdd').json"
        if (-not (Test-Path (Split-Path $alertLogPath -Parent))) {
            New-Item -ItemType Directory -Force -Path (Split-Path $alertLogPath -Parent) | Out-Null
        }
        Add-Content -Path $alertLogPath -Value ($alertData + "`n")
        
        Write-Log "Alert sent: $AlertType - $Message" "WARN"
    } catch {
        Write-Log "Failed to send alert: $_" "ERROR"
    }
}

# Check rate limiting for alerts
function Test-AlertRateLimit {
    param([string]$AlertKey)
    
    $now = Get-Date
    $cooldownSeconds = $Config.AlertCooldownSeconds
    
    # Check if this alert type is in cooldown
    if ($Script:LastAlertTime.ContainsKey($AlertKey)) {
        $lastTime = $Script:LastAlertTime[$AlertKey]
        $elapsed = ($now - $lastTime).TotalSeconds
        
        if ($elapsed -lt $cooldownSeconds) {
            return $false
        }
    }
    
    # Check global rate limit
    if ($Script:AlertCountThisMinute -ge $Config.MaxAlertsPerMinute) {
        return $false
    }
    
    $Script:LastAlertTime[$AlertKey] = $now
    $Script:AlertCountThisMinute++
    
    return $true
}

# Reset alert counter every minute
Start-Job -ScriptBlock {
    while ($true) {
        Start-Sleep -Seconds 60
        $global:Script:AlertCountThisMinute = 0
    }
} | Out-Null

# Get CPU temperature (AMD Ryzen specific)
function Get-CpuTemperature {
    try {
        # Try OpenHardwareMonitorLib first (most reliable)
        $ohmPath = "C:\Program Files\Open Hardware Monitor\OpenHardwareMonitorLib.dll"
        if (Test-Path $ohmPath) {
            Add-Type -Path $ohmPath
            $computer = New-Object OpenHardwareMonitor.Hardware.Computer
            $computer.CPUEnabled = $true
            $computer.Open()
            
            foreach ($hardware in $computer.Hardware) {
                if ($hardware.HardwareType -eq "CPU") {
                    foreach ($sensor in $hardware.Sensors) {
                        if ($sensor.SensorType -eq "Temperature") {
                            return [math]::Round($sensor.Value, 2)
                        }
                    }
                }
            }
        }
        
        # Fallback: WMI (may not work on all systems)
        $temp = Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace "root/wmi" -ErrorAction SilentlyContinue
        if ($temp) {
            return [math]::Round(($temp.CurrentTemperature - 2732) / 10, 2)
        }
        
        # Final fallback: Performance Counter
        $counter = "\Thermal Zone Information\_SYSTEM_ Thermal_Zone_0\Temperature"
        if ([System.Diagnostics.PerformanceCounterCategory]::GetCategories() -match "Thermal Zone Information") {
            $perf = New-Object System.Diagnostics.PerformanceCounter("Thermal Zone Information", "Temperature", "_SYSTEM_ Thermal_Zone_0")
            return [math]::Round($perf.NextValue(), 2)
        }
    } catch {
        Write-Log "Failed to get CPU temperature: $_" "WARN"
    }
    
    return $null
}

# Get memory usage percentage
function Get-MemoryUsagePercent {
    $os = Get-CimInstance Win32_OperatingSystem
    $totalMem = $os.TotalVisibleMemorySize
    $freeMem = $os.FreePhysicalMemory
    $usedMem = $totalMem - $freeMem
    
    return [math]::Round(($usedMem / $totalMem) * 100, 2)
}

# Get CPU usage percentage
function Get-CpuUsagePercent {
    $cpu = Get-CimInstance Win32_Processor
    $load = ($cpu | Measure-Object -Property LoadPercentage -Average).Average
    return [math]::Round($load, 2)
}

# Check for trading bot process status
function Get-BotProcessStatus {
    $processes = @()
    
    # Check for Rust binary
    $rustProc = Get-Process -Name "nautilus_bot" -ErrorAction SilentlyContinue
    if ($rustProc) {
        $processes += @{
            Name = "nautilus_bot"
            Id = $rustProc.Id
            Cpu = [math]::Round($rustProc.CPU, 2)
            MemoryMB = [math]::Round($rustProc.WorkingSet / 1MB, 2)
        }
    }
    
    # Check for Python Ray workers
    $pythonProcs = Get-Process -Name "python" -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*ray*" }
    foreach ($proc in $pythonProcs) {
        $processes += @{
            Name = "ray_worker_$($proc.Id)"
            Id = $proc.Id
            Cpu = [math]::Round($proc.CPU, 2)
            MemoryMB = [math]::Round($proc.WorkingSet / 1MB, 2)
        }
    }
    
    return $processes
}

# Main monitoring loop
function Start-Monitoring {
    Write-Log "Starting degradation monitor..." "INFO"
    Write-Log "Configuration: CPU Threshold=$($Config.CpuTempThresholdCelsius)°C, Memory Threshold=$($Config.MemoryThresholdPercent)%"
    
    try {
        # Initialize WebSocket connection if configured
        if ($Config.WebSocketUrl) {
            try {
                # Note: Requires PowerShell WebSocket module or custom implementation
                # For production, use a proper WebSocket client library
                Write-Log "WebSocket alerts enabled: $($Config.WebSocketUrl)"
            } catch {
                Write-Log "Failed to initialize WebSocket: $_" "WARN"
            }
        }
        
        while ($Script:IsRunning) {
            $startTime = Get-Date
            
            # Monitor CPU Temperature
            $cpuTemp = Get-CpuTemperature
            if ($cpuTemp -ne $null) {
                if ($cpuTemp -ge $Config.CpuTempThresholdCelsius) {
                    $Script:TotalCpuEvents++
                    
                    if (Test-AlertRateLimit "cpu_temp") {
                        Send-FrontendAlert `
                            -AlertType "THERMAL_WARNING" `
                            -Severity "CRITICAL" `
                            -Message "CPU temperature at $($cpuTemp)°C - thermal shedding may activate" `
                            -Details @{ Temperature = $cpuTemp; Threshold = $Config.CpuTempThresholdCelsius }
                    }
                } elseif ($cpuTemp -ge ($Config.CpuTempThresholdCelsius - 5)) {
                    if (Test-AlertRateLimit "cpu_temp_warning") {
                        Send-FrontendAlert `
                            -AlertType "THERMAL_ADVISORY" `
                            -Severity "WARNING" `
                            -Message "CPU temperature approaching threshold: $($cpuTemp)°C" `
                            -Details @{ Temperature = $cpuTemp }
                    }
                }
            }
            
            # Monitor Memory Usage
            $memUsage = Get-MemoryUsagePercent
            if ($memUsage -ge $Config.CriticalMemoryThresholdPercent) {
                $Script:TotalMemoryEvents++
                
                if (Test-AlertRateLimit "memory_critical") {
                    Send-FrontendAlert `
                        -AlertType "MEMORY_CRITICAL" `
                        -Severity "CRITICAL" `
                        -Message "Memory usage at $($memUsage)% - eviction active" `
                        -Details @{ UsagePercent = $memUsage; Threshold = $Config.CriticalMemoryThresholdPercent }
                }
            } elseif ($memUsage -ge $Config.MemoryThresholdPercent) {
                if (Test-AlertRateLimit "memory_warning") {
                    Send-FrontendAlert `
                        -AlertType "MEMORY_WARNING" `
                        -Severity "WARNING" `
                        -Message "Memory usage elevated: $($memUsage)%" `
                        -Details @{ UsagePercent = $memUsage }
                }
            }
            
            # Monitor CPU Usage
            $cpuUsage = Get-CpuUsagePercent
            if ($cpuUsage -ge 95) {
                if (Test-AlertRateLimit "cpu_usage") {
                    Send-FrontendAlert `
                        -AlertType "CPU_SATURATION" `
                        -Severity "WARNING" `
                        -Message "CPU usage at $($cpuUsage)%" `
                        -Details @{ UsagePercent = $cpuUsage }
                }
            }
            
            # Monitor Bot Processes
            $botStatus = Get-BotProcessStatus
            if ($botStatus.Count -eq 0) {
                if (Test-AlertRateLimit "bot_down") {
                    Send-FrontendAlert `
                        -AlertType "BOT_NOT_RUNNING" `
                        -Severity "CRITICAL" `
                        -Message "Trading bot processes not detected" `
                        -Details @{}
                }
            }
            
            # Calculate elapsed time and sleep remainder
            $elapsed = (Get-Date) - $startTime
            $sleepMs = [math]::Max(0, $PollIntervalMs - $elapsed.TotalMilliseconds)
            
            if ($sleepMs -gt 0) {
                Start-Sleep -Milliseconds $sleepMs
            }
        }
    } catch {
        Write-Log "Monitoring error: $_" "ERROR"
        throw
    }
}

# Cleanup function
function Stop-Monitoring {
    Write-Log "Stopping degradation monitor..." "INFO"
    $Script:IsRunning = $false
    
    # Send final status
    Send-FrontendAlert `
        -AlertType "MONITOR_STOPPED" `
        -Severity "INFO" `
        -Message "Degradation monitor stopped" `
        -Details @{
            TotalThermalEvents = $Script:TotalThermalEvents
            TotalMemoryEvents = $Script:TotalMemoryEvents
            TotalCpuEvents = $Script:TotalCpuEvents
        }
}

# Handle Ctrl+C gracefully
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action {
    Stop-Monitoring
} | Out-Null

# Export functions for external use
Export-ModuleMember -Function Start-Monitoring, Stop-Monitoring, Send-FrontendAlert -ErrorAction SilentlyContinue

# Run if executed directly (not imported as module)
if ($MyInvocation.ScriptName -and (Get-Variable -Name MyInvocation).Value.InvocationName -eq $MyInvocation.ScriptName) {
    try {
        Start-Monitoring
    } finally {
        Stop-Monitoring
    }
}

Write-Log "Degradation monitor script loaded successfully" "INFO"
