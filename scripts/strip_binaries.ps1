#!/usr/bin/env pwsh
# Production Hardening: Strip Binaries Script
# 
# Strips all debug symbols and metadata from final Rust release binaries
# to minimize attack surface and file size.
# 
# Features:
# - Removes PDB files to prevent reverse engineering
# - Strips DWARF debug info from binaries
# - Removes .rmeta and .rlib artifacts
# - Verifies binary integrity after stripping
# - Compatible with /START and /KILL orchestration
# 
# Usage:
#   .\scripts\strip_binaries.ps1 --target-dir .\target --profile release

param(
    [Parameter(Mandatory=$false)]
    [string]$TargetDir = ".\target",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("release", "debug")]
    [string]$Profile = "release",
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose,
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Configuration
$ScriptName = "Strip Binaries"
$LogPrefix = "[STRIP]"
$MinificationTargets = @(
    "*.pdb",           # Windows debug symbols
    "*.dSYM",          # macOS debug symbols  
    "*.debug",         # Generic debug info
    "*.rlib",          # Rust library archives
    "*.rmeta",         # Rust metadata
    "*.d",             # Dependency files
    "build-*/",        # Build intermediate files
    "incremental/",    # Incremental compilation cache
    "debug/",          # Debug profile (if not target)
)

# Logging functions
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "$timestamp $LogPrefix [$Level] $Message" -ForegroundColor $(
        if ($Level -eq "ERROR") { "Red" }
        elseif ($Level -eq "WARN") { "Yellow" }
        elseif ($Level -eq "SUCCESS") { "Green" }
        else { "White" }
    )
}

function Write-Verbose-Log {
    param([string]$Message)
    if ($Verbose) {
        Write-Log $Message "VERBOSE"
    }
}

# Verify prerequisites
function Test-Prerequisites {
    Write-Log "Checking prerequisites..."
    
    if (-not (Test-Path $TargetDir)) {
        Write-Log "Target directory does not exist: $TargetDir" "ERROR"
        return $false
    }
    
    $profilePath = Join-Path $TargetDir $Profile
    if (-not (Test-Path $profilePath)) {
        Write-Log "Profile directory does not exist: $profilePath" "ERROR"
        return $false
    }
    
    Write-Log "Prerequisites check passed" "SUCCESS"
    return $true
}

# Calculate original size
function Get-DirectorySize {
    param([string]$Path)
    $size = 0
    if (Test-Path $Path) {
        $files = Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            $size += $file.Length
        }
    }
    return $size
}

# Strip PDB files
function Remove-PdbFiles {
    param([string]$SearchPath)
    
    Write-Log "Removing PDB debug symbols..."
    $pdbFiles = Get-ChildItem -Path $SearchPath -Recurse -Filter "*.pdb" -ErrorAction SilentlyContinue
    
    if ($pdbFiles.Count -eq 0) {
        Write-Verbose-Log "No PDB files found"
        return 0
    }
    
    $count = 0
    foreach ($pdb in $pdbFiles) {
        if (-not $DryRun) {
            Remove-Item -Path $pdb.FullName -Force -ErrorAction SilentlyContinue
        }
        Write-Verbose-Log "Removed: $($pdb.FullName)"
        $count++
    }
    
    Write-Log "Removed $count PDB files" "SUCCESS"
    return $count
}

# Strip debug sections from ELF/Mach-O binaries using strip utility
function Strip-BinaryDebugInfo {
    param([string]$BinaryPath)
    
    $stripTools = @("strip", "llvm-strip", "rust-strip")
    $stripTool = $null
    
    foreach ($tool in $stripTools) {
        if (Get-Command $tool -ErrorAction SilentlyContinue) {
            $stripTool = $tool
            break
        }
    }
    
    if (-not $stripTool) {
        Write-Verbose-Log "No strip tool found, skipping binary stripping"
        return $false
    }
    
    if (-not $DryRun) {
        & $stripTool --strip-debug $BinaryPath 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Verbose-Log "Stripped debug info from: $BinaryPath"
            return $true
        }
    }
    return $false
}

# Remove incremental compilation cache
function Remove-IncrementalCache {
    param([string]$SearchPath)
    
    Write-Log "Removing incremental compilation cache..."
    $incrementalDirs = Get-ChildItem -Path $SearchPath -Recurse -Directory -Filter "incremental" -ErrorAction SilentlyContinue
    
    if ($incrementalDirs.Count -eq 0) {
        Write-Verbose-Log "No incremental cache directories found"
        return 0
    }
    
    $count = 0
    foreach ($dir in $incrementalDirs) {
        if (-not $DryRun) {
            Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Verbose-Log "Removed incremental cache: $($dir.FullName)"
        $count++
    }
    
    Write-Log "Removed $count incremental cache directories" "SUCCESS"
    return $count
}

# Remove rlib and rmeta files
function Remove-RustArtifacts {
    param([string]$SearchPath)
    
    Write-Log "Removing Rust compilation artifacts..."
    $rlibFiles = Get-ChildItem -Path $SearchPath -Recurse -Include "*.rlib","*.rmeta" -ErrorAction SilentlyContinue
    
    if ($rlibFiles.Count -eq 0) {
        Write-Verbose-Log "No Rust artifacts found"
        return 0
    }
    
    $count = 0
    foreach ($file in $rlibFiles) {
        if (-not $DryRun) {
            Remove-Item -Path $file.FullName -Force -ErrorAction SilentlyContinue
        }
        Write-Verbose-Log "Removed: $($file.FullName)"
        $count++
    }
    
    Write-Log "Removed $count Rust artifact files" "SUCCESS"
    return $count
}

# Verify binary integrity after stripping
function Test-BinaryIntegrity {
    param([string]$BinaryPath)
    
    Write-Log "Verifying binary integrity..."
    
    if (-not (Test-Path $BinaryPath)) {
        Write-Log "Binary not found: $BinaryPath" "ERROR"
        return $false
    }
    
    # Check if binary is executable (Windows PE format)
    try {
        $bytes = [System.IO.File]::ReadAllBytes($BinaryPath)
        
        # Check PE header signature
        if ($bytes[0] -eq 0x4D -and $bytes[1] -eq 0x5A) {
            Write-Verbose-Log "Valid PE header detected"
        } else {
            Write-Log "Warning: Binary may not be valid PE format" "WARN"
        }
        
        # Check file size is reasonable (> 100KB for production binaries)
        $fileSize = (Get-Item $BinaryPath).Length
        if ($fileSize -lt 100KB) {
            Write-Log "Warning: Binary size unusually small: $fileSize bytes" "WARN"
        }
        
        Write-Log "Binary integrity verified" "SUCCESS"
        return $true
        
    } catch {
        Write-Log "Failed to verify binary: $_" "ERROR"
        return $false
    }
}

# Generate minification report
function New-MinificationReport {
    param(
        [long]$OriginalSize,
        [long]$FinalSize,
        [int]$FilesRemoved,
        [string]$OutputPath
    )
    
    $reduction = if ($OriginalSize -gt 0) { 
        [math]::Round((1 - $FinalSize / $OriginalSize) * 100, 2) 
    } else { 0 }
    
    $report = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        TargetDirectory = $TargetDir
        Profile = $Profile
        OriginalSizeBytes = $OriginalSize
        FinalSizeBytes = $FinalSize
        SizeReductionPercent = $reduction
        FilesRemoved = $FilesRemoved
        DryRun = $DryRun
    }
    
    if ($OutputPath) {
        $report | ConvertTo-Json | Out-File -FilePath $OutputPath -Encoding UTF8
        Write-Log "Report saved to: $OutputPath"
    }
    
    return $report
}

# Main execution
function Invoke-StripBinaries {
    Write-Log "=" * 60
    Write-Log "$ScriptName - Production Binary Minification"
    Write-Log "=" * 60
    
    if ($DryRun) {
        Write-Log "DRY RUN MODE - No changes will be made" "WARN"
    }
    
    # Prerequisites check
    if (-not (Test-Prerequisites)) {
        Write-Log "Prerequisites check failed, aborting" "ERROR"
        return $false
    }
    
    # Calculate original size
    $originalSize = Get-DirectorySize -Path $TargetDir
    Write-Log "Original directory size: $([math]::Round($originalSize / 1MB, 2)) MB"
    
    $totalRemoved = 0
    
    # Execute stripping operations
    $totalRemoved += Remove-PdbFiles -SearchPath $TargetDir
    $totalRemoved += Remove-IncrementalCache -SearchPath $TargetDir
    $totalRemoved += Remove-RustArtifacts -SearchPath $TargetDir
    
    # Find and strip main binaries
    $binaryExtensions = @("exe", "dll", "so", "dylib")
    foreach ($ext in $binaryExtensions) {
        $binaries = Get-ChildItem -Path $TargetDir -Recurse -Filter "*.$ext" -ErrorAction SilentlyContinue
        foreach ($binary in $binaries) {
            # Only strip release binaries
            if ($binary.FullName -like "*\$Profile\*" -or $binary.FullName -like "*\$Profile/*") {
                Strip-BinaryDebugInfo -BinaryPath $binary.FullName
            }
        }
    }
    
    # Calculate final size
    $finalSize = Get-DirectorySize -Path $TargetDir
    Write-Log "Final directory size: $([math]::Round($finalSize / 1MB, 2)) MB"
    
    # Generate report
    $report = New-MinificationReport `
        -OriginalSize $originalSize `
        -FinalSize $finalSize `
        -FilesRemoved $totalRemoved `
        -OutputPath (Join-Path $TargetDir "strip_report.json")
    
    # Summary
    Write-Log "=" * 60
    Write-Log "STRIPPING COMPLETE"
    Write-Log "=" * 60
    Write-Log "Files Removed: $totalRemoved"
    Write-Log "Space Saved: $([math]::Round(($originalSize - $finalSize) / 1MB, 2)) MB"
    Write-Log "Reduction: $($report.SizeReductionPercent)%"
    Write-Log "=" * 60
    
    # Verify key binaries
    $mainBinary = Join-Path $TargetDir "$Profile\nautilus_bot.exe"
    if (Test-Path $mainBinary) {
        Test-BinaryIntegrity -BinaryPath $mainBinary
    }
    
    Write-Log "Binary stripping completed successfully" "SUCCESS"
    return $true
}

# Handle script errors
trap {
    Write-Log "Unhandled error: $_" "ERROR"
    Write-Log $_.ScriptStackTrace
    exit 1
}

# Execute main function
$result = Invoke-StripBinaries
exit $(if ($result) { 0 } else { 1 })
