# =============================================================================
# Nautilus/Ray Bot - Stage 53: Filesystem Read-Only Lockdown
# File: scripts/fs_readonly.ps1
# Purpose: Remount core Nautilus execution directories as read-only at OS level
#          to prevent ransomware or rogue scripts from corrupting binaries.
# Target: AMD Ryzen AI 5 / Windows 10/11 IoT Enterprise LTSC
# Constraints: 8GB RAM Limit, Security Focus
# =============================================================================

param(
    [switch]$Rollback,
    [switch]$DryRun,
    [string]$TargetDir = "C:\Nautilus\Bin"
)

$ErrorActionPreference = "Stop"
$LogPath = "C:\Nautilus\Logs\fs_readonly.log"
$BackupPath = "C:\Nautilus\Backups\ACL_Config"

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
    }
    
    # Backup current ACLs
    $AclBackup = "$BackupPath\acl_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    icacls $TargetDir /save "$AclBackup"
    Write-Log "ACLs backed up to: $AclBackup"
}

function Set-DirectoryReadOnly {
    param([string]$Path)
    
    Write-Log "Setting directory to Read-Only: $Path"
    
    if (-not (Test-Path $Path)) {
        Write-Log "ERROR: Path does not exist: $Path"
        return
    }
    
    # Remove Write permissions for Everyone and Users
    # Keep Read & Execute for SYSTEM and Administrators
    
    # Deny Write to Everyone (explicit deny takes precedence)
    icacls $Path /deny Everyone:(OI)(CI)W /T
    icacls $Path /deny Users:(OI)(CI)W /T
    
    # Allow Read & Execute for Everyone (so app can still run)
    icacls $Path /allow Everyone:(OI)(CI)RX /T
    
    # Ensure Administrators have Full Control (for /KILL rollback)
    icacls $Path /allow Administrators:(OI)(CI)F /T
    icacls $Path /allow SYSTEM:(OI)(CI)F /T
    
    Write-Log "Permissions updated for: $Path"
}

function Set-FileImmutable {
    param([string]$FilePath)
    
    Write-Log "Setting file attributes to ReadOnly + System: $FilePath"
    
    if (Test-Path $FilePath) {
        # Set ReadOnly attribute
        Set-ItemProperty -Path $FilePath -Name IsReadOnly -Value $true
        
        # Set System attribute (harder to modify accidentally)
        attrib +S +R "$FilePath"
        
        Write-Log "File locked: $FilePath"
    }
}

function Protect-RegistryKeys {
    Write-Log "Protecting critical registry keys..."
    
    # Example: Protect the startup key from modification
    $KeyPath = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
    
    $Acl = Get-Acl $KeyPath
    $Rule = New-Object System.Security.AccessControl.RegistryAccessRule(
        "Everyone",
        [System.Security.AccessControl.RegistryRights]::WriteKey,
        [System.Security.AccessControl.AccessControlType]::Deny
    )
    $Acl.AddAccessRule($Rule)
    Set-Acl $KeyPath $Acl
    
    Write-Log "Registry protection applied."
}

function Restore-Defaults {
    Write-Log "ROLLBACK INITIATED: Restoring write permissions..."
    
    # Restore ACLs from backup
    $LatestBackup = Get-ChildItem -Path $BackupPath -Filter "acl_backup_*.txt" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    
    if ($LatestBackup) {
        icacls $TargetDir /restore "$($LatestBackup.FullName)"
        Write-Log "ACLs restored from: $($LatestBackup.Name)"
    } else {
        # Reset to default inheritance
        icacls $TargetDir /reset /T
        icacls $TargetDir /grant:r Everyone:(OI)(CI)M /T
        Write-Log "Permissions reset to defaults."
    }
    
    # Remove explicit denies
    icacls $TargetDir /remove:d Everyone /T
    icacls $TargetDir /remove:d Users /T
    
    Write-Log "Write permissions restored."
}

# Main Execution
try {
    if (-not (Test-Path $TargetDir)) {
        Write-Log "Creating target directory: $TargetDir"
        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    }

    Initialize-Backup
    
    if ($Rollback) {
        Restore-Defaults
        exit 0
    }
    
    Write-Log "Starting Filesystem Read-Only Lockdown..."
    
    if (-not $DryRun) {
        # Step 1: Set directory permissions
        Set-DirectoryReadOnly -Path $TargetDir
        
        # Step 2: Lock individual executable files
        Get-ChildItem -Path $TargetDir -Filter "*.exe" | ForEach-Object {
            Set-FileImmutable -FilePath $_.FullName
        }
        
        Get-ChildItem -Path $TargetDir -Filter "*.dll" | ForEach-Object {
            Set-FileImmutable -FilePath $_.FullName
        }
        
        # Step 3: Protect registry
        Protect-RegistryKeys
        
        Write-Log "Filesystem lockdown completed."
        Write-Log "WARNING: Binaries are now read-only. Run fs_readonly.ps1 -Rollback to modify."
    } else {
        Write-Log "DRY RUN: No filesystem changes applied."
    }
    
} catch {
    Write-Log "FATAL ERROR during filesystem lockdown: $_"
    throw
}
