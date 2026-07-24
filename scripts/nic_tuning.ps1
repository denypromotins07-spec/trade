# Stage 51: NIC Hardware Tuning PowerShell Script
# 
# Disables Energy Efficient Ethernet (EEE), enables Jumbo Frames,
# and forces RSS queues to specific AMD Ryzen P-cores for minimum latency.
#
# Optimized for AMD Ryzen AI 5 architecture with Windows networking stack.
# Maintains /START and /KILL orchestration compatibility.

param(
    [Parameter(Mandatory = $false)]
    [string]$AdapterName = "Ethernet",
    
    [Parameter(Mandatory = $false)]
    [switch]$Revert = $false,
    
    [Parameter(Mandatory = $false)]
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$scriptName = "NIC Tuning Script"

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "SUCCESS") { "Green" } else { "Cyan" })
}

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Log "This script must be run as Administrator" -Level "ERROR"
    exit 1
}

Write-Log "$scriptName starting..."
Write-Log "Target Adapter: $AdapterName"
Write-Log "Mode: $(if ($Revert) { 'REVERT' } elseif ($DryRun) { 'DRY RUN' } else { 'APPLY' })"

# Get network adapter
try {
    $adapter = Get-NetAdapter -Name $AdapterName -ErrorAction Stop
    Write-Log "Found adapter: $($adapter.Name) (Status: $($adapter.Status))" -Level "SUCCESS"
} catch {
    Write-Log "Adapter '$AdapterName' not found. Available adapters:" -Level "ERROR"
    Get-NetAdapter | ForEach-Object { Write-Log "  - $($_.Name)" }
    exit 1
}

# Backup current settings
$backupFile = "$env:TEMP\nic_settings_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"

function Save-CurrentSettings {
    Write-Log "Backing up current NIC settings to $backupFile"
    
    $settings = @{
        AdapterName = $adapter.Name
        Timestamp = Get-Date -Format "o"
        EEE = $null
        JumboPacket = $null
        InterruptModeration = $null
        RSS = $null
        NumRssQueues = $null
        PriorityFlowControl = $null
    }
    
    try {
        # Get current EEE setting
        $eee = Get-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Energy Efficient Ethernet" -ErrorAction SilentlyContinue
        if ($eee) { $settings.EEE = $eee.DisplayValue }
        
        # Get current Jumbo Packet setting
        $jumbo = Get-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Jumbo Packet" -ErrorAction SilentlyContinue
        if ($jumbo) { $settings.JumboPacket = $jumbo.DisplayValue }
        
        # Get current Interrupt Moderation setting
        $moderation = Get-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Interrupt Moderation" -ErrorAction SilentlyContinue
        if ($moderation) { $settings.InterruptModeration = $moderation.DisplayValue }
        
        # Get current RSS setting
        $rss = Get-NetAdapterRss -Name $AdapterName -ErrorAction SilentlyContinue
        if ($rss) {
            $settings.RSS = $rss.Enabled
            $settings.NumRssQueues = $rss.NumberOfReceiveQueues
        }
        
        # Get Priority Flow Control
        $pfc = Get-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Priority Flow Control" -ErrorAction SilentlyContinue
        if ($pfc) { $settings.PriorityFlowControl = $pfc.DisplayValue }
        
        $settings | ConvertTo-Json | Out-File -FilePath $backupFile -Encoding UTF8
        Write-Log "Settings backup saved successfully" -Level "SUCCESS"
    } catch {
        Write-Log "Failed to backup settings: $_" -Level "WARNING"
    }
}

function Restore-Settings {
    Write-Log "Restoring NIC settings from backup..."
    
    if (-not (Test-Path $backupFile)) {
        Write-Log "No backup file found at $backupFile" -Level "ERROR"
        return
    }
    
    $settings = Get-Content $backupFile | ConvertFrom-Json
    
    try {
        # Restore EEE
        if ($settings.EEE) {
            Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Energy Efficient Ethernet" -DisplayValue $settings.EEE -ErrorAction SilentlyContinue
            Write-Log "Restored EEE: $($settings.EEE)"
        }
        
        # Restore Jumbo Packet
        if ($settings.JumboPacket) {
            Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Jumbo Packet" -DisplayValue $settings.JumboPacket -ErrorAction SilentlyContinue
            Write-Log "Restored Jumbo Packet: $($settings.JumboPacket)"
        }
        
        # Restore Interrupt Moderation
        if ($settings.InterruptModeration) {
            Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Interrupt Moderation" -DisplayValue $settings.InterruptModeration -ErrorAction SilentlyContinue
            Write-Log "Restored Interrupt Moderation: $($settings.InterruptModeration)"
        }
        
        # Restore RSS
        if ($null -ne $settings.RSS) {
            Set-NetAdapterRss -Name $AdapterName -Enabled $settings.RSS -NumberOfReceiveQueues $settings.NumRssQueues -ErrorAction SilentlyContinue
            Write-Log "Restored RSS: Enabled=$($settings.RSS), Queues=$($settings.NumRssQueues)"
        }
        
        Write-Log "Settings restored successfully" -Level "SUCCESS"
    } catch {
        Write-Log "Failed to restore settings: $_" -Level "ERROR"
    }
}

function Apply-LowLatencySettings {
    Write-Log "Applying low-latency NIC optimizations..."
    
    if ($DryRun) {
        Write-Log "[DRY RUN] Would apply the following settings:"
        Write-Log "  - Disable Energy Efficient Ethernet (EEE)"
        Write-Log "  - Enable Jumbo Frames (9014 bytes)"
        Write-Log "  - Disable Interrupt Moderation"
        Write-Log "  - Enable RSS with max queues"
        Write-Log "  - Disable Flow Control"
        return
    }
    
    $changesApplied = 0
    
    # 1. Disable Energy Efficient Ethernet (EEE) - causes latency spikes
    try {
        Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Energy Efficient Ethernet" -DisplayValue "Disabled" -ErrorAction SilentlyContinue
        Write-Log "Disabled Energy Efficient Ethernet (EEE)" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not disable EEE: $_" -Level "WARNING"
    }
    
    # 2. Enable Jumbo Frames (9014 bytes) - reduces packet overhead
    try {
        Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Jumbo Packet" -DisplayValue "9014 Bytes" -ErrorAction SilentlyContinue
        Write-Log "Enabled Jumbo Frames (9014 bytes)" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not enable Jumbo Frames: $_" -Level "WARNING"
    }
    
    # 3. Disable Interrupt Moderation - critical for microsecond latency
    try {
        Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Interrupt Moderation" -DisplayValue "Disabled" -ErrorAction SilentlyContinue
        Write-Log "Disabled Interrupt Moderation" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not disable Interrupt Moderation: $_" -Level "WARNING"
    }
    
    # 4. Configure RSS for AMD Ryzen CCD topology
    try {
        # Get number of physical cores (P-cores only for Ryzen AI)
        $physicalCores = (Get-CimInstance Win32_Processor).NumberOfCores
        $logicalProcessors = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
        
        # Calculate optimal RSS queues (use P-cores, avoid efficiency cores)
        $optimalQueues = [Math]::Min($physicalCores, 8) # Cap at 8 queues
        
        Set-NetAdapterRss -Name $AdapterName -Enabled $true -NumberOfReceiveQueues $optimalQueues -ErrorAction SilentlyContinue
        Write-Log "Configured RSS: Enabled=True, Queues=$optimalQueues (Physical Cores: $physicalCores)" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not configure RSS: $_" -Level "WARNING"
    }
    
    # 5. Disable Flow Control - prevents backpressure latency
    try {
        Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Flow Control" -DisplayValue "Disabled" -ErrorAction SilentlyContinue
        Write-Log "Disabled Flow Control" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not disable Flow Control: $_" -Level "WARNING"
    }
    
    # 6. Enable Transmit Buffers (reduce TX underruns)
    try {
        Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Transmit Buffers" -DisplayValue "2048" -ErrorAction SilentlyContinue
        Write-Log "Increased Transmit Buffers to 2048" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not adjust Transmit Buffers: $_" -Level "WARNING"
    }
    
    # 7. Enable Receive Buffers (reduce RX drops)
    try {
        Set-NetAdapterAdvancedProperty -Name $AdapterName -DisplayName "Receive Buffers" -DisplayValue "2048" -ErrorAction SilentlyContinue
        Write-Log "Increased Receive Buffers to 2048" -Level "SUCCESS"
        $changesApplied++
    } catch {
        Write-Log "Could not adjust Receive Buffers: $_" -Level "WARNING"
    }
    
    # Restart the adapter to apply changes
    Write-Log "Restarting network adapter to apply changes..."
    try {
        Restart-NetAdapter -Name $AdapterName -ErrorAction Stop
        Start-Sleep -Seconds 2
        Write-Log "Adapter restarted successfully" -Level "SUCCESS"
    } catch {
        Write-Log "Failed to restart adapter: $_" -Level "WARNING"
    }
    
    Write-Log "Applied $changesApplied low-latency optimizations" -Level "SUCCESS"
}

function Show-CurrentSettings {
    Write-Log "Current NIC Settings for '$AdapterName':"
    Write-Host ""
    
    Get-NetAdapterAdvancedProperty -Name $AdapterName | 
        Where-Object { $_.DisplayName -match "Energy|Jumbo|Interrupt|Flow|Buffer|RSS" } |
        Format-Table DisplayName, DisplayValue -AutoSize
    
    Write-Host ""
    Write-Log "RSS Configuration:"
    Get-NetAdapterRss -Name $AdapterName | Format-List *
}

# Main execution
try {
    if ($Revert) {
        # Revert to original settings
        Restore-Settings
    } else {
        # Save current settings first
        Save-CurrentSettings
        
        # Apply low-latency optimizations
        Apply-LowLatencySettings
        
        # Show resulting configuration
        Show-CurrentSettings
    }
    
    Write-Log ""
    Write-Log "$scriptName completed successfully" -Level "SUCCESS"
    Write-Log ""
    Write-Log "=========================================="
    Write-Log "IMPORTANT: After running this script:"
    Write-Log "1. Verify network connectivity"
    Write-Log "2. Run '/KILL' script to revert changes"
    Write-Log "3. Backup file location: $backupFile"
    Write-Log "=========================================="
    
} catch {
    Write-Log "Script failed with error: $_" -Level "ERROR"
    exit 1
}
