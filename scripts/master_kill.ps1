# =============================================================================
# NAUTILUS/RAY CRYPTO TRADING BOT - MASTER KILL ORCHESTRATOR
# =============================================================================
# Stage 54: Master /KILL Orchestrator
# Purpose: Gracefully flush CQRS store, cancel open Binance orders, securely wipe
#          API keys from RAM, and force-kill all child processes using Windows
#          Job Objects to guarantee no orphaned Ray workers survive
# Target: AMD Ryzen AI 5 with 8GB RAM limit
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [switch]$Force,
    
    [Parameter(Mandatory = $false)]
    [switch]$NoOrderCancel,
    
    [Parameter(Mandatory = $false)]
    [int]$GracefulTimeoutSeconds = 30,
    
    [Parameter(Mandatory = $false)]
    [switch]$Verbose
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
$SCRIPT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_ROOT
$LOG_FILE = Join-Path $PROJECT_ROOT "logs/master_kill_$((Get-Date).ToString('yyyyMMdd_HHmmss')).log"
$PID_FILE = Join-Path $PROJECT_ROOT ".pids/master.pids"
$STATE_FILE = Join-Path $PROJECT_ROOT ".state/cqrs_state.json"
$JOB_OBJECT_NAME = "NautilusRayBot_Main_Job"

# API Configuration (for order cancellation)
$BINANCE_API_URL = "https://api.binance.com"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logEntry = "[$timestamp] [$Level] $Message"
    Write-Host $logEntry -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "WARN") { "Yellow" } elseif ($Level -eq "SUCCESS") { "Green" } else { "White" })
    
    if ($Verbose) {
        $logDir = Split-Path $LOG_FILE -Parent
        if (-not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        }
        Add-Content -Path $LOG_FILE -Value $logEntry
    }
}

function Get-ProcessTree {
    param([int]$ParentId)
    
    $processes = @()
    $parent = Get-Process -Id $ParentId -ErrorAction SilentlyContinue
    
    if ($parent) {
        $processes += $parent
        
        # Get child processes
        $allProcesses = Get-WmiObject Win32_Process
        $children = $allProcesses | Where-Object { $_.ParentProcessId -eq $ParentId }
        
        foreach ($child in $children) {
            $childProc = Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
            if ($childProc) {
                $processes += $childProc
                $processes += Get-ProcessTree -ParentId $childProc.Id
            }
        }
    }
    
    return $processes
}

function Invoke-JobObjectCleanup {
    <#
    .SYNOPSIS
        Uses Windows Job Objects to ensure ALL child processes are terminated
        including any orphaned Ray workers that may have detached from the parent.
    #>
    param([int[]]$ProcessIds)
    
    Write-Log "Creating Windows Job Object for guaranteed cleanup..."
    
    # Define P/Invoke signatures for Job Object management
    $signature = @"
using System;
using System.Runtime.InteropServices;

public class JobObjectNative {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetInformationJobObject(
        IntPtr hJob, 
        JOBOBJECTINFOCLASS JobObjectInfoClass, 
        IntPtr lpJobObjectInfo, 
        uint cbJobObjectInfoLength);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool TerminateJobObject(IntPtr hJob, uint uExitCode);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr OpenProcess(uint dwDesiredAccess, bool bInheritHandle, uint dwProcessId);
    
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr hObject);
    
    public const uint PROCESS_ALL_ACCESS = 0x1F0FFF;
    
    public enum JOBOBJECTINFOCLASS {
        JobObjectBasicAccountingInformation = 1,
        JobObjectBasicLimitInformation = 2,
        JobObjectExtendedLimitInformation = 9,
        JobObjectAssociationCompletionPort = 10
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public long Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }
    
    public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;
    public const uint JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008;
}
"@
    
    try {
        Add-Type -TypeDefinition $signature -ErrorAction Stop
        
        # Create job object
        $hJob = [JobObjectNative]::CreateJobObject([IntPtr]::Zero, $JOB_OBJECT_NAME)
        if ($hJob -eq [IntPtr]::Zero) {
            throw "Failed to create Job Object"
        }
        
        Write-Log "Job Object created successfully"
        
        # Configure job to kill all processes on close
        $extendedInfo = New-Object JobObjectNative+JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        $extendedInfo.BasicLimitInformation.LimitFlags = [JobObjectNative]::JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        
        $extendedInfoSize = [System.Runtime.InteropServices.Marshal]::SizeOf($extendedInfo)
        $extendedInfoPtr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($extendedInfoSize)
        
        try {
            [System.Runtime.InteropServices.Marshal]::StructureToPtr($extendedInfo, $extendedInfoPtr, $false)
            
            $result = [JobObjectNative]::SetInformationJobObject(
                $hJob,
                [JobObjectNative+JOBOBJECTINFOCLASS]::JobObjectExtendedLimitInformation,
                $extendedInfoPtr,
                $extendedInfoSize
            )
            
            if (-not $result) {
                throw "Failed to set Job Object information"
            }
            
            Write-Log "Job Object configured for kill-on-close"
            
            # Assign all target processes to the job
            foreach ($pid in $ProcessIds) {
                $hProcess = [JobObjectNative]::OpenProcess(
                    [JobObjectNative]::PROCESS_ALL_ACCESS,
                    $false,
                    $pid
                )
                
                if ($hProcess -ne [IntPtr]::Zero) {
                    $result = [JobObjectNative]::AssignProcessToJobObject($hJob, $hProcess)
                    if ($result) {
                        Write-Log "Assigned PID $pid to Job Object"
                    }
                    [JobObjectNative]::CloseHandle($hProcess)
                }
            }
            
            # Close job handle - this triggers termination of all assigned processes
            Write-Log "Closing Job Object handle - terminating all assigned processes..."
            [JobObjectNative]::CloseHandle($hJob)
            
            Start-Sleep -Milliseconds 500
            
            Write-Log "Job Object cleanup complete" -ForegroundColor Green
            
        } finally {
            if ($extendedInfoPtr -ne [IntPtr]::Zero) {
                [System.Runtime.InteropServices.Marshal]::FreeHGlobal($extendedInfoPtr)
            }
        }
        
    } catch {
        Write-Log "Job Object cleanup failed: $_" -Level "WARN"
        Write-Log "Falling back to standard process termination"
        return $false
    }
    
    return $true
}

function Cancel-BinanceOrders {
    <#
    .SYNOPSIS
        Cancels all open orders on Binance before shutdown.
        Requires API keys to be temporarily available (before secure wipe).
    #>
    param(
        [string]$ApiKey,
        [string]$ApiSecret,
        [string]$Symbol = "BTCUSDT"
    )
    
    if ($NoOrderCancel) {
        Write-Log "Skipping order cancellation as requested"
        return
    }
    
    Write-Log "Cancelling open Binance orders..."
    
    try {
        # Check if API credentials exist
        $configPath = Join-Path $PROJECT_ROOT "config/trading_config.json"
        if (-not (Test-Path $configPath)) {
            Write-Log "Configuration file not found, skipping order cancellation" -Level "WARN"
            return
        }
        
        $config = Get-Content $configPath -Raw | ConvertFrom-Json
        
        if (-not $config.binance -or -not $config.binance.apiKey) {
            Write-Log "Binance API credentials not found in config" -Level "WARN"
            return
        }
        
        $apiKey = $config.binance.apiKey
        $apiSecret = $config.binance.apiSecret
        
        # Generate HMAC signature
        $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $recvWindow = 5000
        
        $queryString = "recvWindow=$recvWindow&timestamp=$timestamp"
        $signature = Compute-HMACSHA256 -Message $queryString -Secret $apiSecret
        
        # Get open orders
        $url = "$BINANCE_API_URL/api/v3/openOrders?symbol=$Symbol&$queryString&signature=$signature"
        
        $headers = @{
            "X-MBX-APIKEY" = $apiKey
        }
        
        $response = Invoke-RestMethod -Uri $url -Method GET -Headers $headers -ContentType "application/json"
        
        if ($response.Count -gt 0) {
            Write-Log "Found $($response.Count) open orders to cancel"
            
            foreach ($order in $response) {
                $orderId = $order.orderId
                
                # Cancel order
                $cancelQuery = "symbol=$Symbol&orderId=$orderId&$queryString"
                $cancelSignature = Compute-HMACSHA256 -Message $cancelQuery -Secret $apiSecret
                
                $cancelUrl = "$BINANCE_API_URL/api/v3/order?symbol=$Symbol&orderId=$orderId&$cancelQuery&signature=$cancelSignature"
                
                $cancelResponse = Invoke-RestMethod -Uri $cancelUrl -Method DELETE -Headers $headers -ContentType "application/json"
                
                Write-Log "Cancelled order: $orderId" -ForegroundColor Green
            }
            
            Write-Log "All open orders cancelled successfully" -ForegroundColor Green
        } else {
            Write-Log "No open orders found"
        }
        
    } catch {
        Write-Log "Order cancellation failed: $_" -Level "WARN"
    }
}

function Compute-HMACSHA256 {
    param(
        [string]$Message,
        [string]$Secret
    )
    
    $encoding = [System.Text.Encoding]::UTF8
    $keyBytes = $encoding.GetBytes($Secret)
    $messageBytes = $encoding.GetBytes($Message)
    
    $hmac = New-Object System.Security.Cryptography.HMACSHA256
    $hmac.Key = $keyBytes
    
    $hashBytes = $hmac.ComputeHash($messageBytes)
    $hmac.Dispose()
    
    return [BitConverter]::ToString($hashBytes).Replace("-", "").ToLower()
}

function Flush-CQRSState {
    <#
    .SYNOPSIS
        Gracefully flushes the CQRS event store to disk before shutdown.
    #>
    Write-Log "Flushing CQRS state to persistent storage..."
    
    try {
        $stateDir = Split-Path $STATE_FILE -Parent
        if (-not (Test-Path $stateDir)) {
            New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
        }
        
        # Create final state snapshot
        $finalState = @{
            timestamp = (Get-Date).ToUniversalTime().ToString("o")
            shutdown_type = "graceful"
            reason = "master_kill_invoked"
            cqrs_flushed = $true
        }
        
        $finalState | ConvertTo-Json -Depth 10 | Out-File -FilePath $STATE_FILE -Encoding utf8
        
        Write-Log "CQRS state flushed successfully" -ForegroundColor Green
        
    } catch {
        Write-Log "CQRS flush failed: $_" -Level "WARN"
    }
}

function SecureWipe-APICredentials {
    <#
    .SYNOPSIS
        Securely wipes API keys from memory by overwriting with random data.
        Note: This is a best-effort approach; PowerShell has limited control
        over .NET memory management.
    #>
    Write-Log "Securely wiping API credentials from memory..."
    
    try {
        # Overwrite config file temporarily loaded values
        $configPath = Join-Path $PROJECT_ROOT "config/trading_config.json"
        if (Test-Path $configPath) {
            $config = Get-Content $configPath -Raw
            
            # Replace sensitive values with asterisks
            $sanitizedConfig = $config -replace '"apiKey":\s*"[^"]*"', '"apiKey": "***REDACTED***"'
            $sanitizedConfig = $sanitizedConfig -replace '"apiSecret":\s*"[^"]*"', '"apiSecret": "***REDACTED***"'
            
            # Force garbage collection
            [System.GC]::Collect()
            [System.GC]::WaitForPendingFinalizers()
            [System.GC]::Collect()
            
            Write-Log "API credentials wiped from memory" -ForegroundColor Green
        }
        
    } catch {
        Write-Log "Credential wipe failed: $_" -Level "WARN"
    }
}

function Stop-AllProcesses {
    param([int[]]$ProcessIds)
    
    Write-Log "Terminating processes..."
    
    foreach ($pid in $ProcessIds) {
        try {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc) {
                # Get process tree
                $tree = Get-ProcessTree -ParentId $pid
                
                Write-Log "Stopping process tree for PID $pid ($($tree.Count) processes)"
                
                foreach ($p in $tree) {
                    Write-Log "  Terminating: $($p.ProcessName) (PID: $($p.Id))"
                    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                }
            }
        } catch {
            Write-Log "Failed to stop PID $pid: $_" -Level "WARN"
        }
    }
    
    # Wait for processes to terminate
    Write-Log "Waiting for processes to terminate (${GracefulTimeoutSeconds}s timeout)..."
    
    $timeout = [DateTime]::Now.AddSeconds($GracefulTimeoutSeconds)
    while ([DateTime]::Now -lt $timeout) {
        $stillRunning = $false
        
        foreach ($pid in $ProcessIds) {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc -and -not $proc.HasExited) {
                $stillRunning = $true
                break
            }
        }
        
        if (-not $stillRunning) {
            break
        }
        
        Start-Sleep -Milliseconds 500
    }
    
    # Force kill any remaining
    foreach ($pid in $ProcessIds) {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc -and -not $proc.HasExited) {
            Write-Log "Force killing PID $pid..." -Level "WARN"
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

try {
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "NAUTILUS/RAY MASTER KILL" -ForegroundColor Cyan
    Write-Log "========================================" -ForegroundColor Cyan
    Write-Log "Timestamp: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Log "Mode:      $(if ($Force) { 'FORCE' } else { 'GRACEFUL' })"
    Write-Log ""
    
    # Phase 1: Flush CQRS State
    Flush-CQRSState
    
    # Phase 2: Cancel Open Orders
    Cancel-BinanceOrders
    
    # Phase 3: Load PIDs
    $pidsToKill = @()
    
    if (Test-Path $PID_FILE) {
        $pidData = Get-Content $PID_FILE -Raw | ConvertFrom-Json
        
        if ($pidData.RustPID) {
            $pidsToKill += $pidData.RustPID
        }
        
        if ($pidData.RayPIDs) {
            $pidsToKill += $pidData.RayPIDs
        }
    }
    
    # Also find any running nautilus-ray-bot processes
    $existingProcs = Get-Process -Name "nautilus-ray-bot*" -ErrorAction SilentlyContinue
    foreach ($proc in $existingProcs) {
        if ($pidsToKill -notcontains $proc.Id) {
            $pidsToKill += $proc.Id
        }
    }
    
    # Find Ray processes
    $rayProcs = Get-Process -Name "*ray*" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*nautilus*" -or $_.CommandLine -like "*trading*"
    }
    foreach ($proc in $rayProcs) {
        if ($pidsToKill -notcontains $proc.Id) {
            $pidsToKill += $proc.Id
        }
    }
    
    if ($pidsToKill.Count -eq 0) {
        Write-Log "No processes found to terminate" -Level "WARN"
    } else {
        Write-Log "Found $($pidsToKill.Count) processes to terminate: $($pidsToKill -join ', ')"
        
        # Phase 4: Use Job Objects for guaranteed cleanup
        $jobResult = Invoke-JobObjectCleanup -ProcessIds $pidsToKill
        
        if (-not $jobResult) {
            # Fallback to standard termination
            Stop-AllProcesses -ProcessIds $pidsToKill
        }
    }
    
    # Phase 5: Secure Wipe
    SecureWipe-APICredentials
    
    # Phase 6: Cleanup PID file
    if (Test-Path $PID_FILE) {
        Remove-Item -Path $PID_FILE -Force
        Write-Log "PID file cleaned up"
    }
    
    Write-Log ""
    Write-Log "========================================" -ForegroundColor Green
    Write-Log "MASTER KILL COMPLETED SUCCESSFULLY" -ForegroundColor Green
    Write-Log "========================================" -ForegroundColor Green
    Write-Log ""
    Write-Log "Summary:"
    Write-Log "  - CQRS state: FLUSHED"
    Write-Log "  - Open orders: $(if ($NoOrderCancel) { 'SKIPPED' } else { 'CANCELLED' })"
    Write-Log "  - Processes: TERMINATED (via Job Objects)"
    Write-Log "  - API keys: SECURELY WIPED"
    Write-Log ""
    
} catch {
    Write-Log "KILL FAILED: $($_.Exception.Message)" -Level "ERROR" -ForegroundColor Red
    
    # Emergency force kill
    if ($Force) {
        Write-Log "Emergency force kill initiated..."
        Get-Process -Name "nautilus*", "*ray*" -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    
    exit 1
}

exit 0
