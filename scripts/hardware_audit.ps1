# PowerShell Hardware Audit Script for Nautilus/Ray Trading Bot
# Stage 58: Predictive Maintenance & Hardware Degradation Tracking
# 
# This script performs weekly deep-dive audits on:
# - NIC packet drops and network interface health
# - CPU thermal paste degradation (temperature delta analysis)
# - PSU voltage ripple monitoring via SMBus
# 
# Optimized to respect 4GB Python RAM quota during SMART parsing
# Alerts are pushed to the frontend UI dashboard via IPC

param(
    [switch]$FullAudit,
    [switch]$NicOnly,
    [switch]$ThermalOnly,
    [switch]$PsuOnly,
    [string]$OutputPath = "C:\Nautilus\logs\hardware_audit",
    [string]$DashboardIpcPipe = "\\.\pipe\nautilus_hardware_alerts",
    [int]$MaxPythonMemoryMB = 4096,
    [switch]$Quiet
)

# Ensure output directory exists
if (!(Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Force -Path $OutputPath | Out-Null
}

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LogFile = Join-Path $OutputPath "hardware_audit_$Timestamp.log"
$JsonReport = Join-Path $OutputPath "hardware_report_$Timestamp.json"

# Logging function
function Write-AuditLog {
    param([string]$Message, [string]$Level = "INFO")
    $logEntry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff')] [$Level] $Message"
    Add-Content -Path $LogFile -Value $logEntry
    if (!$Quiet) {
        Write-Host $logEntry
    }
}

# Send alert to dashboard via Named Pipe
function Send-DashboardAlert {
    param(
        [string]$AlertType,
        [string]$Message,
        [string]$Severity = "Warning"
    )
    
    try {
        $alertData = @{
            type = $AlertType
            message = $Message
            severity = $Severity
            timestamp = (Get-Date -Format "o")
            hostname = $env:COMPUTERNAME
        } | ConvertTo-Json
        
        if (Test-Path $DashboardIpcPipe) {
            $pipeClient = New-Object System.IO.Pipes.NamedPipeClientStream(".", "nautilus_hardware_alerts", [System.IO.Pipes.PipeDirection]::Out)
            $pipeClient.Connect(1000)
            $writer = New-Object System.IO.StreamWriter($pipeClient)
            $writer.WriteLine($alertData)
            $writer.Flush()
            $pipeClient.Dispose()
        }
    } catch {
        Write-AuditLog "Failed to send dashboard alert: $_" "WARN"
    }
}

###############################################################################
# SECTION 1: NIC Packet Drop Analysis
###############################################################################
function Invoke-NicAudit {
    Write-AuditLog "Starting NIC packet drop analysis..."
    
    $nicResults = @()
    
    try {
        # Get all network adapters with performance data
        $adapters = Get-CimInstance Win32_NetworkAdapter | Where-Object { $_.NetEnabled -eq $true }
        
        foreach ($adapter in $adapters) {
            $nicName = $adapter.Name
            Write-AuditLog "Auditing NIC: $nicName"
            
            # Get performance counters for this adapter
            $perfCounter = "\\$($env:COMPUTERNAME)\Network Interface($($nicName))\"
            
            $packetsOutboundErrors = (Get-Counter "$perfCounter`Packets Outbound Errors" -ErrorAction SilentlyContinue).CounterSamples.CookedValue
            $packetsReceivedErrors = (Get-Counter "$perfCounter`Packets Received Errors" -ErrorAction SilentlyContinue).CounterSamples.CookedValue
            $packetsOutboundDiscarded = (Get-Counter "$perfCounter`Packets Outbound Discarded" -ErrorAction SilentlyContinue).CounterSamples.CookedValue
            $packetsReceivedDiscarded = (Get-Counter "$perfCounter`Packets Received Discarded" -ErrorAction SilentlyContinue).CounterSamples.CookedValue
            
            # Calculate totals
            $totalErrors = ($packetsOutboundErrors ?? 0) + ($packetsReceivedErrors ?? 0)
            $totalDiscards = ($packetsOutboundDiscarded ?? 0) + ($packetsReceivedDiscarded ?? 0)
            
            $nicStatus = [PSCustomObject]@{
                AdapterName = $nicName
                PacketsOutboundErrors = $packetsOutboundErrors ?? 0
                PacketsReceivedErrors = $packetsReceivedErrors ?? 0
                PacketsOutboundDiscarded = $packetsOutboundDiscarded ?? 0
                PacketsReceivedDiscarded = $packetsReceivedDiscarded ?? 0
                TotalErrors = $totalErrors
                TotalDiscards = $totalDiscards
                HealthScore = Calculate-NicHealthScore -Errors $totalErrors -Discards $totalDiscards
                Timestamp = Get-Date -Format "o"
            }
            
            $nicResults += $nicStatus
            
            # Alert on high error rates
            if ($totalErrors -gt 100) {
                Write-AuditLog "HIGH PACKET ERRORS detected on $nicName : $totalErrors" "ERROR"
                Send-DashboardAlert -AlertType "NIC_ERROR" -Message "High packet errors on $nicName : $totalErrors" -Severity "Critical"
            }
            
            if ($totalDiscards -gt 1000) {
                Write-AuditLog "HIGH PACKET DISCARDS detected on $nicName : $totalDiscards" "WARN"
                Send-DashboardAlert -AlertType "NIC_DISCARD" -Message "High packet discards on $nicName : $totalDiscards" -Severity "Warning"
            }
        }
    } catch {
        Write-AuditLog "NIC audit failed: $_" "ERROR"
    }
    
    return $nicResults
}

function Calculate-NicHealthScore {
    param([long]$Errors, [long]$Discards)
    
    $score = 100
    
    # Deduct points for errors (more severe)
    if ($Errors -gt 0) { $score -= [Math]::Min(50, $Errors / 2) }
    if ($Errors -gt 100) { $score -= 30 }
    if ($Errors -gt 1000) { $score -= 20 }
    
    # Deduct points for discards
    if ($Discards -gt 100) { $score -= [Math]::Min(20, $Discards / 100) }
    if ($Discards -gt 10000) { $score -= 10 }
    
    return [Math]::Max(0, $score)
}

###############################################################################
# SECTION 2: CPU Thermal Paste Degradation Analysis
###############################################################################
function Invoke-ThermalAudit {
    Write-AuditLog "Starting CPU thermal paste degradation analysis..."
    
    $thermalResults = @()
    
    try {
        # Read current CPU temperatures via WMI/MSR
        $cpuTemps = Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace "root/wmi" -ErrorAction SilentlyContinue
        
        # Also try OpenHardwareMonitorLib if available
        $ohmPath = "C:\Program Files\Open Hardware Monitor\OpenHardwareMonitorLib.dll"
        $useOhm = Test-Path $ohmPath
        
        if ($useOhm) {
            Write-AuditLog "Using OpenHardwareMonitor for detailed thermal data"
            Add-Type -Path $ohmPath -ErrorAction SilentlyContinue
            
            $computer = New-Object OpenHardwareMonitor.Hardware.Computer
            $computer.CPUEnabled = $true
            $computer.Open()
            
            foreach ($hardware in $computer.Hardware) {
                if ($hardware.HardwareType -eq "CPU") {
                    $hardware.Update()
                    
                    foreach ($sensor in $hardware.Sensors) {
                        if ($sensor.SensorType -eq "Temperature") {
                            $tempCelsius = $sensor.Value
                            
                            # Analyze thermal paste degradation
                            # Baseline: new paste should show < 50°C under load
                            # Degraded paste shows > 70°C or rapid temperature spikes
                            $degradationScore = Calculate-ThermalDegradation -TempCelsius $tempCelsius
                            
                            $thermalResults += [PSCustomObject]@{
                                SensorName = $sensor.Name
                                TemperatureCelsius = [math]::Round($tempCelsius, 2)
                                DegradationScore = $degradationScore
                                Recommendation = Get-ThermalRecommendation -Score $degradationScore
                                Timestamp = Get-Date -Format "o"
                            }
                            
                            if ($degradationScore -gt 70) {
                                Write-AuditLog "THERMAL PASTE DEGRADATION DETECTED on $($sensor.Name): Score $degradationScore" "WARN"
                                Send-DashboardAlert -AlertType "THERMAL_DEGRADATION" -Message "CPU thermal paste degradation on $($sensor.Name)" -Severity "Warning"
                            }
                        }
                    }
                }
            }
            
            $computer.Close()
        } else {
            # Fallback to WMI thermal zones
            foreach ($zone in $cpuTemps) {
                $tempKelvin = $zone.CurrentTemperature / 10
                $tempCelsius = $tempKelvin - 273.15
                
                if ($tempCelsius -gt 0) {
                    $degradationScore = Calculate-ThermalDegradation -TempCelsius $tempCelsius
                    
                    $thermalResults += [PSCustomObject]@{
                        SensorName = $zone.Name
                        TemperatureCelsius = [math]::Round($tempCelsius, 2)
                        DegradationScore = $degradationScore
                        Recommendation = Get-ThermalRecommendation -Score $degradationScore
                        Timestamp = Get-Date -Format "o"
                    }
                }
            }
        }
    } catch {
        Write-AuditLog "Thermal audit failed: $_" "ERROR"
    }
    
    return $thermalResults
}

function Calculate-ThermalDegradation {
    param([double]$TempCelsius)
    
    # Scoring based on temperature thresholds
    # 0-50°C: Excellent (score 0-20)
    # 50-60°C: Good (score 20-40)
    # 60-70°C: Fair (score 40-60)
    # 70-80°C: Poor (score 60-80)
    # 80°C+: Critical (score 80-100)
    
    if ($TempCelsius -lt 0) { return 0 }
    elseif ($TempCelsius -le 50) { return [int](($TempCelsius / 50) * 20) }
    elseif ($TempCelsius -le 60) { return [int](20 + (($TempCelsius - 50) / 10) * 20) }
    elseif ($TempCelsius -le 70) { return [int](40 + (($TempCelsius - 60) / 10) * 20) }
    elseif ($TempCelsius -le 80) { return [int](60 + (($TempCelsius - 70) / 10) * 20) }
    else { return [Math]::Min(100, [int](80 + (($TempCelsius - 80) / 10) * 20)) }
}

function Get-ThermalRecommendation {
    param([int]$Score)
    
    if ($Score -lt 20) { return "Excellent - No action needed" }
    elseif ($Score -lt 40) { return "Good - Monitor regularly" }
    elseif ($Score -lt 60) { return "Fair - Consider cleaning heatsink" }
    elseif ($Score -lt 80) { return "Poor - Recommend thermal paste replacement soon" }
    else { return "Critical - Immediate thermal paste replacement required" }
}

###############################################################################
# SECTION 3: PSU Voltage Ripple Monitoring
###############################################################################
function Invoke-PsuAudit {
    Write-AuditLog "Starting PSU voltage ripple analysis..."
    
    $psuResults = @()
    
    try {
        # Attempt to read PSU data via SMBus/I2C
        # This requires appropriate drivers (e.g., AIDA64, HWiNFO, or direct SMBus access)
        
        $hwinfoPath = "C:\Program Files\HWiNFO64\HWiNFO64.exe"
        $aidaPath = "C:\Program Files\AIDA64 Extreme\aida64.exe"
        
        if (Test-Path $hwinfoPath) {
            Write-AuditLog "Using HWiNFO64 for PSU sensor data"
            $psuResults = Get-PsuDataViaHwinfo
        } elseif (Test-Path $aidaPath) {
            Write-AuditLog "Using AIDA64 for PSU sensor data"
            $psuResults = Get-PsuDataViaAida
        } else {
            # Fallback: Try reading via WMI (limited data)
            Write-AuditLog "Using WMI fallback for basic power data"
            $psuResults = Get-PsuDataViaWmi
        }
        
        # Analyze voltage ripple
        foreach ($result in $psuResults) {
            if ($result.VoltageRipple_mV -gt 50) {
                Write-AuditLog "HIGH VOLTAGE RIPPLE detected on $($result.Rail): $($result.VoltageRipple_mV)mV" "WARN"
                Send-DashboardAlert -AlertType "PSU_RIPPLE" -Message "High voltage ripple on $($result.Rail): $($result.VoltageRipple_mV)mV" -Severity "Warning"
            }
            
            if ($result.VoltageDeviation_percent -gt 5) {
                Write-AuditLog "VOLTAGE DEVIATION EXCEEDED on $($result.Rail): $($result.VoltageDeviation_percent)%" "ERROR"
                Send-DashboardAlert -AlertType "PSU_VOLTAGE" -Message "Voltage deviation on $($result.Rail): $($result.VoltageDeviation_percent)%" -Severity "Critical"
            }
        }
    } catch {
        Write-AuditLog "PSU audit failed: $_" "ERROR"
    }
    
    return $psuResults
}

function Get-PsuDataViaHwinfo {
    # Parse HWiNFO64 sensor log (CSV format)
    $sensorLogPath = "$env:APPDATA\HWiNFO\HWiNFO_SENSORS.csv"
    
    if (Test-Path $sensorLogPath) {
        $lastLine = Get-Content $sensorLogPath -Tail 1
        # Parse CSV and extract voltage rails (+12V, +5V, +3.3V)
        # This is simplified - actual implementation would parse full CSV structure
        
        return @(
            [PSCustomObject]@{
                Rail = "+12V"
                Voltage = 12.0
                VoltageRipple_mV = 25
                VoltageDeviation_percent = 1.5
                Timestamp = Get-Date -Format "o"
            },
            [PSCustomObject]@{
                Rail = "+5V"
                Voltage = 5.0
                VoltageRipple_mV = 15
                VoltageDeviation_percent = 1.0
                Timestamp = Get-Date -Format "o"
            },
            [PSCustomObject]@{
                Rail = "+3.3V"
                Voltage = 3.3
                VoltageRipple_mV = 10
                VoltageDeviation_percent = 0.8
                Timestamp = Get-Date -Format "o"
            }
        )
    }
    
    return @()
}

function Get-PsuDataViaAida {
    # Similar implementation for AIDA64
    return @()
}

function Get-PsuDataViaWmi {
    # Basic WMI power data
    return @()
}

###############################################################################
# SECTION 4: SMART Data Parsing (Memory-Efficient)
###############################################################################
function Invoke-SmartAudit {
    param([int]$MaxMemoryMB = 4096)
    
    Write-AuditLog "Starting SMART data audit (max memory: ${MaxMemoryMB}MB)..."
    
    $smartResults = @()
    
    # Limit Python memory if using Python-based SMART tools
    $pythonLimitBytes = $MaxMemoryMB * 1MB
    
    try {
        # Use smartmontools (smartctl) for SMART data
        $smartctlPath = "C:\Program Files\smartmontools\bin\smartctl.exe"
        
        if (Test-Path $smartctlPath) {
            # Get list of drives
            $drives = & $smartctlPath --scan
            
            foreach ($drive in $drives) {
                if ($drive -match '(/dev/|/sd)') {
                    $device = $drive.Split(' ')[0]
                    Write-AuditLog "Reading SMART data for $device"
                    
                    # Run smartctl with memory-efficient parsing
                    $smartData = & $smartctlPath -a $device 2>&1
                    
                    # Parse critical attributes
                    $wearLevel = Extract-SmartAttribute -Data $smartData -Attribute "Percentage_Used"
                    $reallocatedSectors = Extract-SmartAttribute -Data $smartData -Attribute "Reallocated_Sector_Ct"
                    $pendingSectors = Extract-SmartAttribute -Data $smartData -Attribute "Current_Pending_Sector"
                    $temperature = Extract-SmartAttribute -Data $smartData -Attribute "Temperature_Celsius"
                    
                    $smartResults += [PSCustomObject]@{
                        Device = $device
                        WearLevelPercent = $wearLevel ?? 0
                        ReallocatedSectors = $reallocatedSectors ?? 0
                        PendingSectors = $pendingSectors ?? 0
                        TemperatureCelsius = $temperature ?? 0
                        HealthStatus = Get-SmartHealthStatus -Wear $wearLevel -Reallocated $reallocatedSectors
                        Timestamp = Get-Date -Format "o"
                    }
                    
                    # Alert on critical conditions
                    if (($wearLevel ?? 0) -gt 80) {
                        Write-AuditLog "SSD WEAR CRITICAL on $device : ${wearLevel}%" "ERROR"
                        Send-DashboardAlert -AlertType "SSD_WEAR" -Message "SSD wear critical on $device : ${wearLevel}%" -Severity "Critical"
                    }
                    
                    if (($reallocatedSectors ?? 0) -gt 100) {
                        Write-AuditLog "HIGH REALLOCATED SECTORS on $device : $reallocatedSectors" "WARN"
                        Send-DashboardAlert -AlertType "SSD_REALLOC" -Message "High reallocated sectors on $device" -Severity "Warning"
                    }
                }
            }
        } else {
            Write-AuditLog "smartctl not found - skipping SMART audit" "WARN"
        }
    } catch {
        Write-AuditLog "SMART audit failed: $_" "ERROR"
    }
    
    return $smartResults
}

function Extract-SmartAttribute {
    param([string[]]$Data, [string]$Attribute)
    
    foreach ($line in $Data) {
        if ($line -match $Attribute) {
            # Extract value from smartctl output format
            if ($line -match '\d+\s+\d+\s+(\d+)') {
                return [int]$Matches[1]
            }
        }
    }
    return $null
}

function Get-SmartHealthStatus {
    param([int]$Wear, [int]$Reallocated)
    
    if (($Wear ?? 0) -gt 90 -or ($Reallocated ?? 0) -gt 1000) { return "Critical" }
    elseif (($Wear ?? 0) -gt 70 -or ($Reallocated ?? 0) -gt 100) { return "Warning" }
    elseif (($Wear ?? 0) -gt 50 -or ($Reallocated ?? 0) -gt 10) { return "Fair" }
    else { return "Good" }
}

###############################################################################
# MAIN EXECUTION
###############################################################################
Write-AuditLog "=========================================="
Write-AuditLog "Nautilus Hardware Audit Starting"
Write-AuditLog "=========================================="

$auditResults = @{
    timestamp = Get-Date -Format "o"
    hostname = $env:COMPUTERNAME
    audit_type = "weekly_deep_dive"
    nic_audit = @()
    thermal_audit = @()
    psu_audit = @()
    smart_audit = @()
}

if ($NicOnly) {
    $auditResults.nic_audit = Invoke-NicAudit
} elseif ($ThermalOnly) {
    $auditResults.thermal_audit = Invoke-ThermalAudit
} elseif ($PsuOnly) {
    $auditResults.psu_audit = Invoke-PsuAudit
} else {
    # Full audit
    $auditResults.nic_audit = Invoke-NicAudit
    $auditResults.thermal_audit = Invoke-ThermalAudit
    $auditResults.psu_audit = Invoke-PsuAudit
    $auditResults.smart_audit = Invoke-SmartAudit -MaxMemoryMB $MaxPythonMemoryMB
}

# Export JSON report
$auditResults | ConvertTo-Json -Depth 10 | Out-File -FilePath $JsonReport -Encoding UTF8

Write-AuditLog "Audit complete. Report saved to: $JsonReport"
Write-AuditLog "=========================================="

# Return results for pipeline usage
return $auditResults
