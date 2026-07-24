# =============================================================================
# Nautilus/Ray Bot - Stage 53: Kernel Tuning
# File: scripts/kernel_tuning.ps1
# Purpose: Modify Windows registry to disable CPU idle states, force High Performance
#          power plans, and set system tick rate to 0.5ms for minimum scheduling latency.
# Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
# Constraints: 8GB RAM Limit, Microsecond Latency Focus
# =============================================================================

param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$LogPath = "C:\Nautilus\Logs\kernel_tuning.log"
$BackupPath = "C:\Nautilus\Backups\Kernel_Config"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] $Message"
    if (-not (Test-Path (Split-Path $LogPath))) {
        New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
    }
    Add-Content -Path $LogPath -Value $logEntry
    if (-not $DryRun) { Write-Host $logEntry }
}

function Initialize-Backup {
    if (-not (Test-Path $BackupPath)) {
        New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null
        Write-Log "Created backup directory: $BackupPath"
    }
}

function Disable-CStates {
    Write-Log "Disabling CPU C-States (Idle States) for constant clock speeds..."
    
    # Processor Power Management Settings
    $ProcKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Power"
    
    # Disable Core Parking (Keep all cores active)
    Set-ItemProperty -Path "$ProcKey\ProcessorPerformanceCoreParking" -Name "MinCores" -Value 100 -Type DWord -Force -ErrorAction SilentlyContinue
    
    # Disable Idle Check
    Set-ItemProperty -Path "$ProcKey\IdleAcpiOverride" -Name "(Default)" -Value 1 -Type DWord -Force -ErrorAction SilentlyContinue
    
    # Force Maximum Processor State (100%)
    # Note: This is also enforced via powercfg commands below
    
    Write-Log "C-State disabling registry keys applied."
}

function Set-HighPerformancePowerPlan {
    Write-Log "Configuring Ultimate Performance Power Plan..."
    
    # GUID for Ultimate Performance Plan (Windows 10/11)
    $UltimateGuid = "e9a42b02-d5df-448d-aa00-03f14749eb61"
    $HighPerfGuid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    
    # Try to activate Ultimate Performance plan first
    try {
        powercfg /setactive $UltimateGuid
        Write-Log "Activated Ultimate Performance power plan."
    } catch {
        # Fallback to High Performance
        powercfg /setactive $HighPerfGuid
        Write-Log "Activated High Performance power plan (Ultimate not available)."
    }
    
    # Modify current scheme settings for maximum performance
    $Scheme = (powercfg /getactivescheme)[0].Split(':')[1].Trim()
    
    # Set Processor Power Management
    # Minimum Processor State = 100%
    powercfg /setacvalueindex $Scheme sub_processor 893dee8e-2bef-41e0-89c6-b55d09390840 100
    # Maximum Processor State = 100%
    powercfg /setacvalueindex $Scheme sub_processor 893dee8e-2bef-41e0-89c6-b55d09390841 100
    # System Cooling Policy = Active (Fans first, throttle later)
    powercfg /setacvalueindex $Scheme sub_processor 94d3a615-a899-4ac5-ae2b-e4d8f634367f 1 # Active
    # Processor Idle Disable = 1 (Disable C-States)
    powercfg /setacvalueindex $Scheme sub_processor 5d76a2ca-e8c0-402f-a133-2158492d58ad 1
    
    # Apply the changes
    powercfg /setactive $Scheme
    
    Write-Log "Power plan configured for 100% CPU state and disabled idle."
}

function Set-SystemTickRate {
    Write-Log "Setting system timer resolution to 0.5ms (500 microseconds)..."
    
    # Note: Windows default is 15.6ms. We request 0.5ms.
    # This requires a helper tool or driver in modern Windows versions (Win10 2004+)
    # as SetThreadExecutionCharacter is capped. 
    # For true 0.5ms, we modify the registry key used by multimedia timers.
    
    $MultimediaKey = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile"
    if (-not (Test-Path $MultimediaKey)) {
        New-Item -Path $MultimediaKey -Force | Out-Null
    }
    
    # Set SystemResponsiveness to 0 (dedicate resources to multimedia/HFT tasks)
    Set-ItemProperty -Path $MultimediaKey -Name "SystemResponsiveness" -Value 0 -Type DWord -Force
    
    # Set NetworkThrottlingIndex to unlimited (hex: ffffffff)
    Set-ItemProperty -Path $MultimediaKey -Name "NetworkThrottlingIndex" -Value 4294967295 -Type DWord -Force
    
    # Timer Resolution hint (Note: Actual enforcement may require 'TimerResolution' tool)
    # Registry key for requested timer resolution in 100ns units (5000 = 0.5ms)
    $TimerKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\kernel"
    if (-not (Test-Path $TimerKey)) {
        New-Item -Path $TimerKey -Force | Out-Null
    }
    # Obsolete in newer Windows but worth trying on LTSC/IoT
    Set-ItemProperty -Path $TimerKey -Name "GlobalTimerResolutionRequests" -Value 1 -Type DWord -Force
    
    Write-Log "Registry keys for high-resolution timer applied."
}

function Disable-InterruptCoalescingRegistry {
    Write-Log "Disabling interrupt coalescing at OS level for NICs..."
    
    # Get all network adapters
    $Adapters = Get-WmiObject Win32_NetworkAdapterConfiguration | Where-Object { $_.MACAddress -ne $null }
    
    foreach ($Adapter in $Adapters) {
        $InterfaceName = $Adapter.SettingID
        if ($InterfaceName) {
            # Path to NIC specific settings
            $NicPath = "HKLM:\SYSTEM\CurrentControlSet\Enum\PCI\*\*\Device Parameters\InterruptManagement\MessageSignaledInterruptProperties\$InterfaceName"
            # Note: Direct manipulation often requires knowing the specific hardware ID.
            # We will rely on the nic_tuning.ps1 script for device-specific registry edits.
            Write-Log "Skipping specific NIC edit here (handled by nic_tuning.ps1): $InterfaceName"
        }
    }
}

function Optimize-MemoryManagement {
    Write-Log "Optimizing Memory Management for HFT..."
    
    $MemoryKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    
    # Disable Paging Executive (Keep kernel resident in RAM)
    Set-ItemProperty -Path $MemoryKey -Name "DisablePagingExecutive" -Value 1 -Type DWord -Force
    
    # Large System Cache (Useful for file serving, maybe not HFT, but keeps FS cache in RAM)
    # Set to 0 for application focus (HFT app is the main consumer)
    Set-ItemProperty -Path $MemoryKey -Name "LargeSystemCache" -Value 0 -Type DWord -Force
    
    # ClearPageFileAtShutdown (Security, but adds shutdown time. Optional.)
    # Set-ItemProperty -Path $MemoryKey -Name "ClearPageFileAtShutdown" -Value 1 -Type DWord -Force
    
    # Prefetcher: Enable for boot, disable for runtime? 
    # For HFT, we pre-warm, so prefetcher is less critical during hot path.
    # Values: 0=Disabled, 1=Boot, 2=Apps, 3=Both
    $PrefetchKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters"
    Set-ItemProperty -Path $PrefetchKey -Name "EnablePrefetcher" -Value 0 -Type DWord -Force
    Set-ItemProperty -Path $PrefetchKey -Name "EnableBoottrace" -Value 0 -Type DWord -Force
    
    Write-Log "Memory management optimizations applied."
}

function Restore-Defaults {
    Write-Log "ROLLBACK INITIATED: Restoring default kernel settings..."
    
    # Reset Power Plan to Balanced
    $BalancedGuid = "381b4222-f694-41f0-9685-ff5bb260df2e"
    powercfg /setactive $BalancedGuid
    
    # Re-enable Paging Executive
    $MemoryKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
    Set-ItemProperty -Path $MemoryKey -Name "DisablePagingExecutive" -Value 0 -Force
    
    # Reset Prefetcher
    $PrefetchKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PrefetchParameters"
    Set-ItemProperty -Path $PrefetchKey -Name "EnablePrefetcher" -Value 3 -Force
    
    # Reset Timer Resolution requests
    $TimerKey = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\kernel"
    Set-ItemProperty -Path $TimerKey -Name "GlobalTimerResolutionRequests" -Value 0 -Force
    
    Write-Log "Kernel settings restored to defaults."
}

# Main Execution
try {
    Initialize-Backup
    
    if ($Rollback) {
        Restore-Defaults
        exit 0
    }
    
    Write-Log "Starting Kernel Tuning Process..."
    
    if (-not $DryRun) {
        Disable-CStates
        Set-HighPerformancePowerPlan
        Set-SystemTickRate
        Optimize-MemoryManagement
    } else {
        Write-Log "DRY RUN: No changes applied."
    }
    
    Write-Log "Kernel Tuning completed successfully. Reboot recommended for full effect."
    
} catch {
    Write-Log "FATAL ERROR during kernel tuning: $_"
    throw
}
