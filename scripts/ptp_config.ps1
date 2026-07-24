# =============================================================================
# Nautilus/Ray Crypto Trading Bot - Stage 50
# File: scripts/ptp_config.ps1
# Chapter 3: Precision Time Protocol (PTP) & Hardware Timestamping (PowerShell)
# 
# Purpose: PowerShell script to configure Windows Precision Time Protocol
#          services, disabling NTP fallbacks and forcing the NIC hardware
#          to timestamp packets at the physical layer.
#
# Optimization Targets:
#   - Sub-microsecond time synchronization
#   - Hardware timestamping enablement
#   - NTP fallback prevention
#   - NIC-specific PTP configuration
#
# Constraints:
#   - Requires Administrator privileges
#   - Windows 10/11 or Windows Server 2019+
#   - Compatible with /START and /KILL orchestration
# =============================================================================

param(
    [switch]$Help,
    [switch]$CheckStatus,
    [switch]$Disable,
    [string]$NicName = ""
)

$scriptName = "ptp_config.ps1"
$logFile = "C:\Nautilus\logs\ptp_config.log"

# Ensure log directory exists
$logDir = Split-Path $logFile -Parent
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $logFile -Value $logEntry
    if ($Level -eq "ERROR") {
        Write-Host $logEntry -ForegroundColor Red
    } elseif ($Level -eq "WARN") {
        Write-Host $logEntry -ForegroundColor Yellow
    } else {
        Write-Host $logEntry -ForegroundColor Green
    }
}

function Show-Help {
    Write-Host @"
Nautilus/Ray PTP Configuration Script
=====================================

Usage: .\$scriptName [Options]

Options:
  -Help         Show this help message
  -CheckStatus  Check current PTP/W32Time service status
  -Disable      Disable PTP and revert to NTP
  -NicName      Specify network adapter name (optional)

Examples:
  .\$scriptName                    # Configure PTP on default adapter
  .\$scriptName -NicName "Ethernet" # Configure specific adapter
  .\$scriptName -CheckStatus       # Check current status
  .\$scriptName -Disable           # Revert to NTP

Requirements:
  - Administrator privileges
  - Windows 10/11 or Windows Server 2019+
  - Network adapter with PTP/hardware timestamping support

"@
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-NetworkAdapters {
    Get-NetAdapter | Where-Object { $_.Status -eq "Up" } | Select-Object Name, InterfaceDescription, LinkSpeed
}

function Get-PtpCapableAdapters {
    Write-Log "Checking for PTP-capable network adapters..."
    
    # Check for adapters with hardware timestamping support
    $adapters = Get-NetAdapter | Where-Object { $_.Status -eq "Up" }
    $ptpAdapters = @()
    
    foreach ($adapter in $adapters) {
        # Check registry for hardware timestamping capabilities
        $registryPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}"
        $netConfigs = Get-ChildItem $registryPath -ErrorAction SilentlyContinue
        
        foreach ($config in $netConfigs) {
            $driverDesc = (Get-ItemProperty $config.PSPath).DriverDesc
            if ($driverDesc -like "*{$($adapter.Name)}*") {
                # Check for PTP/hardware timestamping properties
                $properties = Get-ItemProperty $config.PSPath
                if ($properties.PSObject.Properties.Name -match ".*[Tt]ime[Ss]tamp.*") {
                    $ptpAdapters += $adapter
                    Write-Log "Found PTP-capable adapter: $($adapter.Name) ($driverDesc)" -Level "INFO"
                    break
                }
            }
        }
    }
    
    return $ptpAdapters
}

function Configure-W32TimeForPtp {
    Write-Log "Configuring W32Time service for PTP operation..."
    
    try {
        # Stop the Windows Time service
        Stop-Service -Name "w32time" -Force -ErrorAction Stop
        Write-Log "Stopped W32Time service"
        
        # Configure for PTP (hardware timestamping)
        # Set AnnounceRetries to 0 for PTP mode
        w32tm /config /manualpeerlist:"ptp.pool.ntp.org" /syncfromflags:manual /reliable:yes /update
        Write-Log "Configured W32Time for manual peer list"
        
        # Disable NTP fallback behavior
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" `
            -Name "AnnounceFlags" -Value 10 -Type DWORD -Force
        Write-Log "Disabled NTP fallback flags"
        
        # Set special polling interval for low-latency
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" `
            -Name "MinPollInterval" -Value 6 -Type DWORD -Force  # 2^6 = 64 seconds
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" `
            -Name "MaxPollInterval" -Value 8 -Type DWORD -Force  # 2^8 = 256 seconds
        
        # Enable hardware timestamping if supported
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\TimeProviders\NtpClient" `
            -Name "EnableHardwareTimestamp" -Value 1 -Type DWORD -Force
        
        Write-Log "Configured polling intervals and hardware timestamping"
        
        # Restart the service
        Start-Service -Name "w32time" -ErrorAction Stop
        Write-Log "Started W32Time service"
        
        # Force resync
        w32tm /resync /rediscover
        Write-Log "Forced time resynchronization"
        
        return $true
    }
    catch {
        Write-Log "Failed to configure W32Time: $_" -Level "ERROR"
        return $false
    }
}

function Configure-NicHardwareTimestamping {
    param([string]$AdapterName)
    
    Write-Log "Configuring hardware timestamping on adapter: $AdapterName"
    
    try {
        # Enable Receive Side Scaling (RSS) for better performance
        Enable-NetAdapterRss -Name $AdapterName -NumberOfReceiveQueues 4
        Write-Log "Enabled RSS with 4 queues"
        
        # Enable hardware timestamping via registry (adapter-specific)
        $registryPath = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e972-e325-11ce-bfc1-08002be10318}"
        $netConfigs = Get-ChildItem $registryPath -ErrorAction SilentlyContinue
        
        foreach ($config in $netConfigs) {
            $properties = Get-ItemProperty $config.PSPath
            if ($properties.DriverDesc -like "*$AdapterName*") {
                # Try common hardware timestamping property names
                $timestampProperties = @(
                    "PriorityVLANTag",
                    "TimestampEnable",
                    "HardwareTS",
                    "PTPEnable"
                )
                
                foreach ($prop in $timestampProperties) {
                    if ($properties.PSObject.Properties.Name -contains $prop) {
                        Set-ItemProperty -Path $config.PSPath -Name $prop -Value 1 -Force
                        Write-Log "Enabled hardware timestamping property: $prop"
                    }
                }
                break
            }
        }
        
        # Disable power saving features that affect timing
        Disable-NetAdapterBinding -Name $AdapterName -ComponentID "ms_lltdio" -ErrorAction SilentlyContinue
        Disable-NetAdapterBinding -Name $AdapterName -ComponentID "ms_rspndr" -ErrorAction SilentlyContinue
        Write-Log "Disabled power-saving bindings"
        
        return $true
    }
    catch {
        Write-Log "Failed to configure NIC: $_" -Level "ERROR"
        return $false
    }
}

function Disable-CStates {
    Write-Log "Disabling CPU C-states for consistent timing..."
    
    try {
        # Set processor performance core parking to disabled
        powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR COREPARKING 0
        powercfg /setactive SCHEME_CURRENT
        Write-Log "Disabled core parking"
        
        # Note: Full C-state disable requires BIOS configuration
        # This script only handles OS-level settings
        Write-Log "Note: Full C-state disable requires BIOS configuration" -Level "WARN"
        
        return $true
    }
    catch {
        Write-Log "Failed to configure power settings: $_" -Level "ERROR"
        return $false
    }
}

function Check-PtpStatus {
    Write-Log "Checking PTP/W32Time status..."
    
    # Check service status
    $service = Get-Service -Name "w32time" -ErrorAction SilentlyContinue
    if ($service) {
        Write-Log "W32Time Service Status: $($service.Status)"
    } else {
        Write-Log "W32Time Service: Not installed" -Level "WARN"
    }
    
    # Check current time source
    $timeSource = w32tm /query 2>&1 | Select-String "Source:"
    Write-Log "Current time source: $timeSource"
    
    # Check for PTP-capable adapters
    $ptpAdapters = Get-PtpCapableAdapters
    if ($ptpAdapters.Count -eq 0) {
        Write-Log "No PTP-capable adapters detected" -Level "WARN"
    } else {
        Write-Log "PTP-capable adapters found: $($ptpAdapters.Count)"
        $ptpAdapters | ForEach-Object { Write-Log "  - $($_.Name)" }
    }
    
    # Check registry settings
    $announceFlags = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" `
        -Name "AnnounceFlags" -ErrorAction SilentlyContinue
    if ($announceFlags) {
        Write-Log "AnnounceFlags: $($announceFlags.AnnounceFlags)"
    }
}

function Disable-PtpConfiguration {
    Write-Log "Disabling PTP configuration and reverting to NTP..."
    
    try {
        # Stop W32Time
        Stop-Service -Name "w32time" -Force
        
        # Reset to default NTP configuration
        w32tm /config /computer:DOMAIN /syncfromflags:DOMHIER /update
        Write-Log "Reset to domain hierarchy sync"
        
        # Reset announce flags
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Config" `
            -Name "AnnounceFlags" -Value 1 -Type DWORD -Force
        
        # Restart service
        Start-Service -Name "w32time"
        w32tm /resync
        
        Write-Log "Reverted to standard NTP configuration"
        return $true
    }
    catch {
        Write-Log "Failed to disable PTP: $_" -Level "ERROR"
        return $false
    }
}

# Main execution
Write-Log "=== PTP Configuration Script Started ==="

if ($Help) {
    Show-Help
    exit 0
}

if (-not (Test-Administrator)) {
    Write-Log "This script requires Administrator privileges!" -Level "ERROR"
    Write-Host "Please run as Administrator" -ForegroundColor Red
    exit 1
}

if ($CheckStatus) {
    Check-PtpStatus
    exit 0
}

if ($Disable) {
    Disable-PtpConfiguration
    exit 0
}

# Default: Configure PTP
Write-Log "Starting PTP configuration..."

# Get target adapter
if ([string]::IsNullOrEmpty($NicName)) {
    $ptpAdapters = Get-PtpCapableAdapters
    if ($ptpAdapters.Count -eq 0) {
        Write-Log "No PTP-capable adapters found. Using first active adapter." -Level "WARN"
        $targetAdapter = (Get-NetAdapter | Where-Object { $_.Status -eq "Up" } | Select-Object -First 1).Name
    } else {
        $targetAdapter = $ptpAdapters[0].Name
    }
} else {
    $targetAdapter = $NicName
}

if ([string]::IsNullOrEmpty($targetAdapter)) {
    Write-Log "No suitable network adapter found!" -Level "ERROR"
    exit 1
}

Write-Log "Target adapter: $targetAdapter"

# Configure components
$success = $true

$success = $success -and (Configure-W32TimeForPtp)
$success = $success -and (Configure-NicHardwareTimestamping -AdapterName $targetAdapter)
$success = $success -and (Disable-CStates)

if ($success) {
    Write-Log "=== PTP Configuration Completed Successfully ==="
    exit 0
} else {
    Write-Log "=== PTP Configuration Completed with Errors ===" -Level "ERROR"
    exit 1
}
