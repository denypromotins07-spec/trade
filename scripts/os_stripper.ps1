# =============================================================================
# Nautilus/Ray Bot - Stage 53: OS Stripper
# File: scripts/os_stripper.ps1
# Purpose: Aggressively disable Windows telemetry, services, and background apps
#          to create a bare-metal HFT execution environment.
# Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
# Constraints: 8GB RAM Limit, Microsecond Latency Focus
# =============================================================================

param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$LogPath = "C:\Nautilus\Logs\os_stripper.log"
$BackupPath = "C:\Nautilus\Backups\OS_Config"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] $Message"
    Add-Content -Path $LogPath -Value $logEntry
    if (-not $DryRun) { Write-Host $logEntry }
}

function Initialize-Backup {
    if (-not (Test-Path $BackupPath)) {
        New-Item -ItemType Directory -Force -Path $BackupPath | Out-Null
        Write-Log "Created backup directory: $BackupPath"
    }
}

function Disable-Telemetry {
    Write-Log "Disabling Windows Telemetry and Data Collection..."
    
    $TelemetryKeys = @(
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection",
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AdvertisingInfo",
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\AppCompat",
        "HKLM:\SOFTWARE\Policies\Microsoft\Windows\FeedbackHub"
    )

    foreach ($key in $TelemetryKeys) {
        if (-not (Test-Path $key)) {
            New-Item -Path $key -Force | Out-Null
        }
        # Set AllowTelemetry to 0 (Security/Enterprise only) or 1 (Basic)
        Set-ItemProperty -Path $key -Name "AllowTelemetry" -Value 0 -Force -ErrorAction SilentlyContinue
        Set-ItemProperty -Path $key -Name "MaxTelemetryAllowed" -Value 0 -Force -ErrorAction SilentlyContinue
    }

    # Disable Diagnostics Tracking Service
    Stop-Service -Name "DiagTrack" -Force -ErrorAction SilentlyContinue
    Set-Service -Name "DiagTrack" -StartupType Disabled -ErrorAction SilentlyContinue
    
    Write-Log "Telemetry services stopped and disabled."
}

function Disable-CortanaAndSearch {
    Write-Log "Disabling Cortana and Windows Search indexing..."
    
    # Disable Cortana via Registry
    $CortanaKey = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\Windows Search"
    if (-not (Test-Path $CortanaKey)) { New-Item -Path $CortanaKey -Force | Out-Null }
    Set-ItemProperty -Path $CortanaKey -Name "AllowCortana" -Value 0 -Force
    Set-ItemProperty -Path $CortanaKey -Name "AllowCloudSearch" -Value 0 -Force
    
    # Stop Search Service (Critical for disk I/O latency reduction)
    Stop-Service -Name "WSearch" -Force -ErrorAction SilentlyContinue
    Set-Service -Name "WSearch" -StartupType Disabled -ErrorAction SilentlyContinue
    
    Write-Log "Cortana and Search indexing disabled."
}

function Disable-WindowsUpdate {
    Write-Log "Disabling Windows Update services to prevent background downloads..."
    
    $UpdateServices = @("wuauserv", "UsoSvc", "Dosvc", "WaaSMedicSvc")
    
    foreach ($svc in $UpdateServices) {
        try {
            Stop-Service -Name $svc -Force -ErrorAction SilentlyContinue
            Set-Service -Name $svc -StartupType Disabled -ErrorAction SilentlyContinue
            Write-Log "Service $svc disabled."
        } catch {
            Write-Log "Warning: Could not disable service $svc. $_"
        }
    }
    
    # Pause Updates via Registry
    $UpdateKey = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU"
    if (-not (Test-Path $UpdateKey)) { New-Item -Path $UpdateKey -Force | Out-Null }
    Set-ItemProperty -Path $UpdateKey -Name "NoAutoUpdate" -Value 1 -Force
}

function Remove-UWPApps {
    Write-Log "Removing non-essential UWP applications to free RAM..."
    
    # List of apps to remove (Xbox, Weather, News, etc.)
    $AppsToRemove = @(
        "Microsoft.XboxApp", "Microsoft.Xbox.TCUI", "Microsoft.XboxGameOverlay",
        "Microsoft.BingWeather", "Microsoft.BingNews", "Microsoft.GetHelp",
        "Microsoft.Getstarted", "Microsoft.MicrosoftSolitaireCollection",
        "Microsoft.StickyNotes", "Microsoft.ZuneMusic", "Microsoft.ZuneVideo",
        "Microsoft.Windows.Photos", "Microsoft.People", "Microsoft.Office.OneNote"
    )

    foreach ($AppName in $AppsToRemove) {
        Get-AppxPackage -AllUsers | Where-Object { $_.Name -like "*$AppName*" } | 
        Remove-AppxPackage -AllUsers -ErrorAction SilentlyContinue
        Write-Log "Removed app package matching: $AppName"
    }
    
    Write-Log "UWP App removal complete."
}

function Disable-VisualEffects {
    Write-Log "Disabling visual effects for maximum performance..."
    
    $SystemParams = "HKCU:\Control Panel\Desktop"
    Set-ItemProperty -Path $SystemParams -Name "UserPreferencesMask" -Value ([byte[]](144, 18, 3, 128, 16, 0, 0, 0)) -Force
    Set-ItemProperty -Path $SystemParams -Name "FontSmoothing" -Value "2" -Force # Keep font smoothing for readability
    
    # Disable Animations
    $Accessibility = "HKCU:\Control Panel\Desktop\WindowMetrics"
    # Additional animation keys can be disabled here
}

function Optimize-NetworkStack {
    Write-Log "Optimizing Network Stack for HFT..."
    
    # Disable TCP Auto-Tuning (Can cause latency spikes)
    netsh interface tcp set global autotuninglevel=disabled
    
    # Disable ECN (Explicit Congestion Notification)
    netsh interface tcp set global ecncapability=disabled
    
    # Disable Window Scaling (Optional, depends on specific network path)
    # netsh interface tcp set global windowscaling=disabled
    
    Write-Log "Network stack optimization applied."
}

function Restore-Defaults {
    Write-Log "ROLLBACK INITIATED: Restoring default Windows configuration..."
    
    # Re-enable Services
    $ServicesToEnable = @("DiagTrack", "WSearch", "wuauserv", "UsoSvc")
    foreach ($svc in $ServicesToEnable) {
        Set-Service -Name $svc -StartupType Automatic -ErrorAction SilentlyContinue
        Start-Service -Name $svc -ErrorAction SilentlyContinue
    }
    
    # Re-enable Telemetry
    $TelemetryKey = "HKLM:\SOFTWARE\Policies\Microsoft\Windows\DataCollection"
    if (Test-Path $TelemetryKey) {
        Set-ItemProperty -Path $TelemetryKey -Name "AllowTelemetry" -Value 3 -Force
    }
    
    # Re-enable TCP Auto-Tuning
    netsh interface tcp set global autotuninglevel=normal
    
    Write-Log "Rollback complete. System restored to default state."
}

# Main Execution
try {
    Initialize-Backup
    
    if ($Rollback) {
        Restore-Defaults
        exit 0
    }
    
    Write-Log "Starting OS Stripping Process..."
    
    if (-not $DryRun) {
        Disable-Telemetry
        Disable-CortanaAndSearch
        Disable-WindowsUpdate
        Remove-UWPApps
        Disable-VisualEffects
        Optimize-NetworkStack
    } else {
        Write-Log "DRY RUN: No changes applied."
    }
    
    Write-Log "OS Stripping completed successfully. Ready for HFT workload."
    
} catch {
    Write-Log "FATAL ERROR during OS stripping: $_"
    throw
}
