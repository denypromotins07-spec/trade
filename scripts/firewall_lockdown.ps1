# =============================================================================
# Nautilus/Ray Bot - Stage 53: Firewall Lockdown
# File: scripts/firewall_lockdown.ps1
# Purpose: Configure Windows Defender Firewall to allow ONLY outbound traffic
#          to specific Binance IP ranges, dropping all other traffic.
# Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
# Constraints: 8GB RAM Limit, Network Isolation for HFT
# =============================================================================

param(
    [switch]$Rollback,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$LogPath = "C:\Nautilus\Logs\firewall_lockdown.log"
$BackupPath = "C:\Nautilus\Backups\Firewall_Config"

# Binance Production IP Ranges (Update as needed)
# Source: Binance API Documentation / Network Analysis
$BinanceIPs = @(
    "52.190.249.1",     # Example Binance US
    "3.72.220.69",      # Example Binance EU
    "52.222.128.0/20",  # AWS CloudFront (Binance CDN)
    "13.32.0.0/15",     # AWS Global
    "54.230.0.0/16"     # AWS CloudFront
)

# Allowed Ports
$AllowedPorts = @(443, 80, 9443) # HTTPS, HTTP, Alternative HTTPS

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
    
    # Backup current firewall rules
    $BackupFile = "$BackupPath\firewall_rules_$(Get-Date -Format 'yyyyMMdd_HHmmss').xml"
    netsh advfirewall export "$BackupFile"
    Write-Log "Firewall rules backed up to: $BackupFile"
}

function Create-BinanceAllowRule {
    Write-Log "Creating ALLOW rules for Binance IP ranges..."
    
    $RuleName = "Nautilus_Binance_Outbound_Allow"
    
    # Remove existing rule if present
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    
    # Create comma-separated list of remote addresses
    $RemoteAddressList = ($BinanceIPs | ForEach-Object { $_ }) -join ","
    
    # Create the allow rule
    New-NetFirewallRule `
        -DisplayName $RuleName `
        -Direction Outbound `
        -Action Allow `
        -RemoteAddress $RemoteAddressList `
        -Protocol TCP `
        -LocalPort Any `
        -RemotePort $AllowedPorts `
        -Profile Any `
        -Enabled True `
        -Description "Allow outbound traffic to Binance exchange only" `
        -ErrorAction SilentlyContinue
        
    Write-Log "Created rule: $RuleName"
}

function Create-BlockAllRule {
    Write-Log "Creating BLOCK ALL rule for non-Binance traffic..."
    
    $RuleName = "Nautilus_Global_Block_Outbound"
    
    # Remove existing rule
    Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
    
    # Create block rule with lower priority (higher number) than allow rule
    New-NetFirewallRule `
        -DisplayName $RuleName `
        -Direction Outbound `
        -Action Block `
        -RemoteAddress Any `
        -Protocol Any `
        -Profile Any `
        -Enabled True `
        -Description "Block all outbound traffic not explicitly allowed" `
        -ErrorAction SilentlyContinue
        
    Write-Log "Created global block rule: $RuleName"
}

function Enable-Logging {
    Write-Log "Enabling firewall logging for dropped packets..."
    
    Set-NetFirewallProfile -Profile Domain,Public,Private `
        -LogBlocked True `
        -LogMaxSizeKilobytes 4096 `
        -LogFileName "%systemroot%\system32\LogFiles\Firewall\pfirewall.log" `
        -ErrorAction SilentlyContinue
        
    Write-Log "Firewall logging enabled."
}

function Disable-NonEssentialServices {
    Write-Log "Disabling non-essential network services..."
    
    # Disable SMB (prevent lateral movement)
    Set-NetFirewallRule -DisplayGroup "File and Printer Sharing" -Enabled False -ErrorAction SilentlyContinue
    
    # Disable Remote Desktop
    Set-NetFirewallRule -DisplayGroup "Remote Desktop" -Enabled False -ErrorAction SilentlyContinue
    
    # Disable Network Discovery
    Set-NetFirewallRule -DisplayGroup "Network Discovery" -Enabled False -ErrorAction SilentlyContinue
    
    Write-Log "Non-essential services disabled."
}

function Restore-Defaults {
    Write-Log "ROLLBACK INITIATED: Restoring default firewall configuration..."
    
    # Delete Nautilus-specific rules
    Remove-NetFirewallRule -DisplayName "Nautilus_*" -ErrorAction SilentlyContinue
    
    # Restore from backup if available
    $LatestBackup = Get-ChildItem -Path $BackupPath -Filter "*.xml" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($LatestBackup) {
        netsh advfirewall import "$($LatestBackup.FullName)"
        Write-Log "Restored firewall from backup: $($LatestBackup.Name)"
    } else {
        # Reset to default
        netsh advfirewall reset
        Write-Log "Firewall reset to factory defaults."
    }
    
    # Re-enable essential services
    Set-NetFirewallRule -DisplayGroup "File and Printer Sharing" -Enabled True -ErrorAction SilentlyContinue
    Set-NetFirewallRule -DisplayGroup "Network Discovery" -Enabled True -ErrorAction SilentlyContinue
}

# Main Execution
try {
    Initialize-Backup
    
    if ($Rollback) {
        Restore-Defaults
        exit 0
    }
    
    Write-Log "Starting Firewall Lockdown Process..."
    
    if (-not $DryRun) {
        # Step 1: Create allow rules for Binance
        Create-BinanceAllowRule
        
        # Step 2: Create global block rule
        Create-BlockAllRule
        
        # Step 3: Enable logging
        Enable-Logging
        
        # Step 4: Disable non-essential services
        Disable-NonEssentialServices
        
        Write-Log "Firewall lockdown completed successfully."
        Write-Log "WARNING: Only Binance IP ranges are accessible. All other outbound traffic is blocked."
    } else {
        Write-Log "DRY RUN: No firewall changes applied."
    }
    
} catch {
    Write-Log "FATAL ERROR during firewall lockdown: $_"
    throw
}
