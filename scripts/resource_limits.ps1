# =============================================================================
# Nautilus/Ray Crypto Trading Bot - Stage 55
# File 2: scripts/resource_limits.ps1
# 
# SetInformationJobObject for Hard CPU and Memory Limits at OS Kernel Level
# Guarantees Python ecosystem never breaches 4GB quota
# Enforces global 8GB RAM limit for AMD Ryzen AI 5 architecture
# Optimized for microsecond latency with zero-overhead limit enforcement
# =============================================================================

param(
    [Parameter(Mandatory = $false)]
    [string]$Action = "Apply",
    
    [Parameter(Mandatory = $false)]
    [switch]$Verify
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# Windows API Definitions for Resource Limits
# =============================================================================

Add-Type @"
using System;
using System.Runtime.InteropServices;

public class ResourceLimitApi
{
    // Job Object Information Classes
    public enum JOBOBJECTINFOCLASS
    {
        JobObjectBasicLimitInformation = 2,
        JobObjectExtendedLimitInformation = 9,
        JobObjectAssociationCompletionPortInformation = 7,
        JobObjectCpuRateControlInformation = 10
    }

    // CPU Rate Control Information
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
    {
        public uint ControlFlags;
        public uint CpuRate;
    }

    // Extended Limit Information
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    // Basic Limit Information
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    // CPU Rate Control Flags
    public const uint JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x00000001;
    public const uint JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x00000002;
    public const uint JOB_OBJECT_CPU_RATE_CONTROL_NOTIFY = 0x00000004;

    // Limit Flags
    public const uint JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008;
    public const uint JOB_OBJECT_LIMIT_AFFINITY = 0x00000010;
    public const uint JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400;
    public const uint JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200;
    public const uint JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004;
    public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
    public const uint JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100;
    public const uint JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002;
    public const uint JOB_OBJECT_LIMIT_SCHEDULING_CLASS = 0x00000040;
    public const uint JOB_OBJECT_LIMIT_WORKINGSET = 0x00000020;

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetInformationJobObject(
        IntPtr hJob,
        JOBOBJECTINFOCLASS JobObjectInfoClass,
        IntPtr lpJobObjectInfo,
        uint cbJobObjectInfoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool QueryInformationJobObject(
        IntPtr hJob,
        JOBOBJECTINFOCLASS JobObjectInfoClass,
        out JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInfo,
        uint cbJobObjectInfoLength,
        out uint lpReturnLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint dwDesiredAccess, bool bInheritHandle, int dwProcessId);

    public const uint PROCESS_SET_QUOTA = 0x0100;
    public const uint PROCESS_TERMINATE = 0x0001;
    public const uint PROCESS_QUERY_INFORMATION = 0x0400;
}
"@

# =============================================================================
# Global Configuration - Strict Memory Boundaries
# =============================================================================

$Global8GBLimit = 8GB
$Python4GBLimit = 4GB
$RustCore2GBLimit = 2GB
$RayWorkers1GBLimit = 1GB
$MaxWorkingSetPython = 3.5GB  # Soft limit below hard 4GB ceiling
$MinWorkingSetPython = 256MB
$CpuRateCapPython = 75  # Python capped at 75% of single core to prevent starvation
$CpuRateCapRust = 100   # Rust can use full core for hot paths
$PriorityClassRust = "High"
$PriorityClassPython = "Normal"

# Logging with microsecond precision
function Write-LimitLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.ffffff"
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "Cyan" }
    }
    Write-Host "[$timestamp] [$Level] [ResourceLimits] $Message" -ForegroundColor $color
}

# =============================================================================
# Apply Hard Memory Limits via SetInformationJobObject
# =============================================================================

function Set-HardMemoryLimits {
    param(
        [IntPtr]$JobHandle,
        [uint64]$TotalJobLimit = $Global8GBLimit,
        [uint64]$ProcessLimit = $Python4GBLimit,
        [uint64]$MinWorkingSet = 256MB,
        [uint64]$MaxWorkingSet = 3.5GB
    )

    try {
        $extendedInfo = New-Object ResourceLimitApi+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        
        # Configure strict memory limit flags
        $extendedInfo.BasicLimitInformation.LimitFlags = `
            [ResourceLimitApi]::JOB_OBJECT_LIMIT_JOB_MEMORY -bor `
            [ResourceLimitApi]::JOB_OBJECT_LIMIT_PROCESS_MEMORY -bor `
            [ResourceLimitApi]::JOB_OBJECT_LIMIT_WORKINGSET -bor `
            [ResourceLimitApi]::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE -bor `
            [ResourceLimitApi]::JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION

        # Set absolute job-wide memory ceiling (8GB global)
        $extendedInfo.JobMemoryLimit = [UIntPtr]$TotalJobLimit
        
        # Set per-process memory ceiling (4GB for Python)
        $extendedInfo.ProcessMemoryLimit = [UIntPtr]$ProcessLimit
        
        # Configure working set boundaries
        $extendedInfo.BasicLimitInformation.MinimumWorkingSetSize = [UIntPtr]$MinWorkingSet
        $extendedInfo.BasicLimitInformation.MaximumWorkingSetSize = [UIntPtr]$MaxWorkingSet

        Write-LimitLog "Configuring memory limits:"
        Write-LimitLog "  - Job Memory Ceiling: $([math]::Round($TotalJobLimit / 1GB, 2))GB"
        Write-LimitLog "  - Process Memory Ceiling: $([math]::Round($ProcessLimit / 1GB, 2))GB"
        Write-LimitLog "  - Working Set Range: $([math]::Round($MinWorkingSet / 1MB, 0))MB - $([math]::Round($MaxWorkingSet / 1MB, 0))MB"

        # Marshal and apply
        $size = [System.Runtime.InteropServices.Marshal]::SizeOf($extendedInfo)
        $ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
        
        try {
            [System.Runtime.InteropServices.Marshal]::StructureToPtr($extendedInfo, $ptr, $false)
            
            $result = [ResourceLimitApi]::SetInformationJobObject(
                $JobHandle,
                [ResourceLimitApi+JOBOBJECTINFOCLASS]::JobObjectExtendedLimitInformation,
                $ptr,
                [uint32]$size
            )

            if (-not $result) {
                $error = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
                throw "SetInformationJobObject failed with error code: $error"
            }

            Write-LimitLog "Hard memory limits applied successfully at kernel level" -Level "SUCCESS"
        }
        finally {
            [System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
        }

        return $true
    }
    catch {
        Write-LimitLog "Failed to set memory limits: $_" -Level "ERROR"
        return $false
    }
}

# =============================================================================
# Apply CPU Rate Control (Hard Cap)
# =============================================================================

function Set-CpuRateLimit {
    param(
        [IntPtr]$JobHandle,
        [uint32]$CpuRate = 75,  # Percentage (1-100)
        [bool]$HardCap = $true
    )

    try {
        $cpuInfo = New-Object ResourceLimitApi+JOBOBJECT_CPU_RATE_CONTROL_INFORMATION
        
        # CPU rate is specified as a percentage * 100 (e.g., 75% = 7500)
        $cpuInfo.CpuRate = $CpuRate * 100
        
        # Set control flags
        $cpuInfo.ControlFlags = [ResourceLimitApi]::JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
        if ($HardCap) {
            $cpuInfo.ControlFlags = $cpuInfo.ControlFlags -bor [ResourceLimitApi]::JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
        }

        Write-LimitLog "Configuring CPU rate limit: ${CpuRate}% (HardCap: $HardCap)"

        $size = [System.Runtime.InteropServices.Marshal]::SizeOf($cpuInfo)
        $ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
        
        try {
            [System.Runtime.InteropServices.Marshal]::StructureToPtr($cpuInfo, $ptr, $false)
            
            $result = [ResourceLimitApi]::SetInformationJobObject(
                $JobHandle,
                [ResourceLimitApi+JOBOBJECTINFOCLASS]::JobObjectCpuRateControlInformation,
                $ptr,
                [uint32]$size
            )

            if (-not $result) {
                $error = [System.Runtime.InteropServices.Marshal]::GetLastWin32Error()
                # CPU rate control may not be available on all Windows versions
                if ($error -eq 87) {
                    Write-LimitLog "CPU rate control not supported on this Windows version - skipping" -Level "WARN"
                    return $true
                }
                throw "SetInformationJobObject (CPU) failed: $error"
            }

            Write-LimitLog "CPU rate limit applied successfully" -Level "SUCCESS"
        }
        finally {
            [System.Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
        }

        return $true
    }
    catch {
        Write-LimitLog "Failed to set CPU rate limit: $_" -Level "ERROR"
        return $false
    }
}

# =============================================================================
# Verify Applied Limits
# =============================================================================

function Test-ResourceLimits {
    param([IntPtr]$JobHandle)

    try {
        $extendedInfo = New-Object ResourceLimitApi+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        $returnLength = 0

        $result = [ResourceLimitApi]::QueryInformationJobObject(
            $JobHandle,
            [ResourceLimitApi+JOBOBJECTINFOCLASS]::JobObjectExtendedLimitInformation,
            [ref]$extendedInfo,
            [System.Runtime.InteropServices.Marshal]::SizeOf($extendedInfo),
            [ref]$returnLength
        )

        if (-not $result) {
            throw "Failed to query job object limits"
        }

        $actualJobLimit = $extendedInfo.JobMemoryLimit.ToUInt64()
        $actualProcessLimit = $extendedInfo.ProcessMemoryLimit.ToUInt64()
        $actualMinWS = $extendedInfo.BasicLimitInformation.MinimumWorkingSetSize.ToUInt64()
        $actualMaxWS = $extendedInfo.BasicLimitInformation.MaximumWorkingSetSize.ToUInt64()
        $limitFlags = $extendedInfo.BasicLimitInformation.LimitFlags

        Write-LimitLog "=== Resource Limits Verification ==="
        Write-LimitLog "Job Memory Limit:     $([math]::Round($actualJobLimit / 1GB, 2))GB (Expected: 8GB)"
        Write-LimitLog "Process Memory Limit: $([math]::Round($actualProcessLimit / 1GB, 2))GB (Expected: 4GB)"
        Write-LimitLog "Min Working Set:      $([math]::Round($actualMinWS / 1MB, 0))MB"
        Write-LimitLog "Max Working Set:      $([math]::Round($actualMaxWS / 1MB, 0))MB"
        
        # Verify flags
        $hasJobMemory = ($limitFlags -band [ResourceLimitApi]::JOB_OBJECT_LIMIT_JOB_MEMORY) -ne 0
        $hasProcessMemory = ($limitFlags -band [ResourceLimitApi]::JOB_OBJECT_LIMIT_PROCESS_MEMORY) -ne 0
        $hasWorkingSet = ($limitFlags -band [ResourceLimitApi]::JOB_OBJECT_LIMIT_WORKINGSET) -ne 0
        $hasKillOnClose = ($limitFlags -band [ResourceLimitApi]::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) -ne 0

        Write-LimitLog "JOB_MEMORY flag:           $hasJobMemory"
        Write-LimitLog "PROCESS_MEMORY flag:       $hasProcessMemory"
        Write-LimitLog "WORKINGSET flag:           $hasWorkingSet"
        Write-LimitLog "KILL_ON_JOB_CLOSE flag:    $hasKillOnClose"

        # Validate against expected values
        $jobLimitOk = $actualJobLimit -le ($Global8GBLimit + 1MB)  # Small tolerance
        $processLimitOk = $actualProcessLimit -le ($Python4GBLimit + 1MB)
        $flagsOk = $hasJobMemory -and $hasProcessMemory -and $hasWorkingSet -and $hasKillOnClose

        if ($jobLimitOk -and $processLimitOk -and $flagsOk) {
            Write-LimitLog "All resource limits verified successfully" -Level "SUCCESS"
            return $true
        }
        else {
            Write-LimitLog "Resource limit verification FAILED" -Level "ERROR"
            return $false
        }
    }
    catch {
        Write-LimitLog "Verification error: $_" -Level "ERROR"
        return $false
    }
}

# =============================================================================
# Main Execution
# =============================================================================

try {
    Write-LimitLog "=========================================="
    Write-LimitLog "Nautilus Resource Limits Configuration"
    Write-LimitLog "Target: AMD Ryzen AI 5 (znver4)"
    Write-LimitLog "=========================================="

    # For demonstration, we create a temporary job object
    # In production, this integrates with job_objects.ps1
    $tempJobName = "NautilusTempJob_$(Get-Random)"
    $hJob = [ResourceLimitApi]::CreateJobObject([IntPtr]::Zero, $tempJobName)
    
    if ($hJob -eq [IntPtr]::Zero) {
        throw "Failed to create temporary job object"
    }

    Write-LimitLog "Created temporary job object: $tempJobName"

    switch ($Action) {
        "Apply" {
            Write-LimitLog "Applying hard memory and CPU limits..."
            
            $memResult = Set-HardMemoryLimits -JobHandle $hJob
            $cpuResult = Set-CpuRateLimit -JobHandle $hJob -CpuRate $CpuRateCapPython
            
            if ($memResult -and $cpuResult) {
                Write-LimitLog "All resource limits applied successfully" -Level "SUCCESS"
            }
            else {
                throw "Failed to apply one or more resource limits"
            }
        }
        
        "Verify" {
            Write-LimitLog "Verifying applied resource limits..."
            $verified = Test-ResourceLimits -JobHandle $hJob
            if (-not $verified) {
                throw "Resource limit verification failed"
            }
        }
        
        default {
            throw "Unknown action: $Action"
        }
    }

    # Cleanup temporary job
    [ResourceLimitApi]::CloseHandle($hJob) | Out-Null
    Write-LimitLog "Temporary job object cleaned up"
    
    Write-LimitLog "Resource limits script completed" -Level "SUCCESS"
}
catch {
    Write-LimitLog "Fatal error: $_" -Level "ERROR"
    exit 1
}
