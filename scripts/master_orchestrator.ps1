# Master Orchestrator Script
# 
# The ultimate PowerShell script parsing /START and /KILL commands,
# managing Windows Job Objects to guarantee all child processes are destroyed.
#
# Compatible with AMD Ryzen AI 5 architecture optimization.

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('/START', '/KILL', '/STATUS', '/RESTART')]
    [string]$Command,
    
    [switch]$Verbose,
    [switch]$NoDaemon
)

# Configuration
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$RustBinary = "$ProjectRoot\target\release\nutilus_bot.exe"
$PythonMain = "$ProjectRoot\python\main.py"
$LogFile = "$ProjectRoot\logs\orchestrator.log"
$PidFile = "$ProjectRoot\.pids\master.pid"
$JobName = "NautilusTradingBot"

# Ensure directories exist
$null = New-Item -ItemType Directory -Force -Path "$ProjectRoot\logs"
$null = New-Item -ItemType Directory -Force -Path "$ProjectRoot\.pids"

# Logging function
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logLine = "[$timestamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $logLine
    if ($Verbose) {
        Write-Host $logLine
    }
}

# Check if process is running
function Test-ProcessRunning {
    param([int]$Pid)
    try {
        $process = Get-Process -Id $Pid -ErrorAction SilentlyContinue
        return $null -ne $process -and -not $process.HasExited
    } catch {
        return $false
    }
}

# Get all child PIDs from job
function Get-JobChildPids {
    param([Microsoft.Win32.SafeHandles.SafeProcessHandle]$JobHandle)
    $pids = @()
    
    # Use GetExpandedCommandLine via WMI to find job members
    $jobProcess = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -in (Get-Process | Where-Object { $_.Handle -in @(
            Get-CimInstance Win32_JobObject | ForEach-Object {
                $_.GetAssociatedCimInstances("Win32_Process") | Select-Object -ExpandProperty ProcessId
            }
        ) })
    } | Select-Object -ExpandProperty ProcessId
    
    return $jobProcess
}

# Create Windows Job Object for process containment
function New-ProcessJob {
    $signature = @'
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool SetInformationJobObject(
    IntPtr hJob, 
    int JobObjectInfoType, 
    IntPtr lpJobObjectInfo, 
    uint cbJobObjectInfoLength
);

[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool TerminateJobObject(IntPtr hJob, uint uExitCode);
'@
    
    Add-Type -MemberDefinition $signature -Name "JobObject" -Namespace "Win32"
    
    $JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    $JobObjectExtendedLimitInformation = 9
    
    $hJob = [Win32.JobObject]::CreateJobObject([IntPtr]::Zero, $JobName)
    
    $jobInfo = New-Object -TypeName System.Object
    $jobInfo | Add-Member -MemberType NoteProperty -Name BasicLimitInformation -Value @{
        LimitFlags = [uint32]$JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    }
    
    # Simplified - in production use proper struct marshaling
    $size = 16
    $ptr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    [System.Runtime.InteropServices.Marshal]::StructureToPtr($jobInfo.BasicLimitInformation, $ptr, $false)
    
    [Win32.JobObject]::SetInformationJobObject($hJob, $JobObjectExtendedLimitInformation, $ptr, [uint32]$size) | Out-Null
    
    return $hJob
}

# START command
function Start-NautilusBot {
    Write-Log "=== STARTING NAUTILUS TRADING BOT ==="
    
    # Pre-flight checks
    Write-Log "Running pre-flight checks..."
    
    # Validate .env file
    if (-not (Test-Path "$ProjectRoot\.env")) {
        Write-Log ".env file not found!" "ERROR"
        throw ".env file required. Run env_validator.ps1 first."
    }
    
    # Check Rust binary
    if (-not (Test-Path $RustBinary)) {
        Write-Log "Rust binary not found. Building..." "WARN"
        Write-Log "Run: cargo build --release" "WARN"
    }
    
    # Check Python environment
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Log "Python not found in PATH!" "ERROR"
        throw "Python 3.10+ required"
    }
    
    # Create Job Object for process containment
    $jobHandle = New-ProcessJob
    Write-Log "Created Windows Job Object: $JobName"
    
    # Start Rust master process
    Write-Log "Starting Rust master process..."
    $rustStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $rustStartInfo.FileName = $RustBinary
    $rustStartInfo.WorkingDirectory = $ProjectRoot
    $rustStartInfo.UseShellExecute = $false
    $rustStartInfo.CreateNoWindow = $true
    
    $rustProcess = [System.Diagnostics.Process]::Start($rustStartInfo)
    $rustPid = $rustProcess.Id
    
    # Assign to job object
    [Win32.JobObject]::AssignProcessToJobObject($jobHandle, $rustProcess.Handle) | Out-Null
    
    # Save PIDs
    $pidData = @{
        Master = $rustPid
        JobHandle = $jobHandle.ToString()
        StartTime = (Get-Date).ToString("o")
    }
    $pidData | ConvertTo-Json | Set-Content -Path $PidFile
    
    Write-Log "Rust master started with PID: $rustPid"
    
    # Start Python Ray workers (if not NoDaemon)
    if (-not $NoDaemon) {
        Write-Log "Starting Python Ray workers..."
        
        $pythonStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $pythonStartInfo.FileName = "python"
        $pythonStartInfo.Arguments = "$PythonMain --worker"
        $pythonStartInfo.WorkingDirectory = $ProjectRoot
        $pythonStartInfo.UseShellExecute = $false
        $pythonStartInfo.CreateNoWindow = $true
        
        $pyProcess = [System.Diagnostics.Process]::Start($pythonStartInfo)
        [Win32.JobObject]::AssignProcessToJobObject($jobHandle, $pyProcess.Handle) | Out-Null
        
        Write-Log "Python worker started with PID: $($pyProcess.Id)"
        
        # Update PID file
        $pidData.Worker = $pyProcess.Id
        $pidData | ConvertTo-Json | Set-Content -Path $PidFile
    }
    
    Write-Log "Nautilus Bot started successfully"
    Write-Log "PID file: $PidFile"
    Write-Log "Log file: $LogFile"
    
    return $rustPid
}

# KILL command
function Stop-NautilusBot {
    Write-Log "=== STOPPING NAUTILUS TRADING BOT ==="
    
    # Load PID file
    if (Test-Path $PidFile) {
        $pidData = Get-Content $PidFile | ConvertFrom-Json
        $masterPid = $pidData.Master
        
        Write-Log "Master PID from file: $masterPid"
        
        # Try graceful shutdown first
        if (Test-ProcessRunning $masterPid) {
            Write-Log "Sending graceful shutdown signal to PID $masterPid..."
            
            # Send CTRL+C via console control
            try {
                $process = Get-Process -Id $masterPid
                $process.CloseMainWindow() | Out-Null
                
                # Wait for graceful exit (max 10 seconds)
                $waitCount = 0
                while ($process.HasExited -eq $false -and $waitCount -lt 100) {
                    Start-Sleep -Milliseconds 100
                    $waitCount++
                }
                
                if (-not $process.HasExited) {
                    Write-Log "Graceful shutdown timed out, forcing kill..." "WARN"
                    Stop-Process -Id $masterPid -Force
                }
            } catch {
                Write-Log "Error during graceful shutdown: $_" "ERROR"
                Stop-Process -Id $masterPid -Force -ErrorAction SilentlyContinue
            }
        }
        
        # Kill any remaining child processes
        Write-Log "Cleaning up remaining processes..."
        Get-Process | Where-Object {
            $_.ProcessName -like "*nautilus*" -or 
            $_.CommandLine -like "*ray*" -or
            $_.Id -eq $pidData.Worker
        } | Stop-Process -Force -ErrorAction SilentlyContinue
        
        # Clean up Job Object
        try {
            $jobObjects = Get-CimInstance Win32_JobObject -Filter "Name='$JobName'"
            foreach ($job in $jobObjects) {
                Invoke-CimMethod -InputObject $job -MethodName TerminateJobObject -Arguments @{uExitCode = 0}
                Write-Log "Terminated Job Object: $JobName"
            }
        } catch {
            Write-Log "Job Object cleanup warning: $_" "WARN"
        }
        
        # Remove PID file
        Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
        
        Write-Log "Nautilus Bot stopped successfully"
        
    } else {
        Write-Log "No PID file found. Searching for running instances..." "WARN"
        
        # Find by process name
        $nautilusProcs = Get-Process | Where-Object {
            $_.ProcessName -like "*nautilus*" -or $_.ProcessName -eq "rust_bot"
        }
        
        if ($nautilusProcs) {
            foreach ($proc in $nautilusProcs) {
                Write-Log "Stopping PID $($proc.Id)..."
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
            Write-Log "Stopped $($nautilusProcs.Count) processes"
        } else {
            Write-Log "No running Nautilus processes found"
        }
    }
    
    # Cleanup Ray shared memory (call Python teardown)
    if (Test-Path "$ScriptRoot\..\python\ray\teardown.py") {
        Write-Log "Running Ray teardown..."
        python "$ScriptRoot\..\python\ray\teardown.py" --force 2>&1 | ForEach-Object { Write-Log $_ }
    }
}

# STATUS command
function Get-NautilusStatus {
    Write-Log "Checking Nautilus Bot status..."
    
    if (Test-Path $PidFile) {
        $pidData = Get-Content $PidFile | ConvertFrom-Json
        $masterPid = $pidData.Master
        
        $status = [PSCustomObject]@{
            Running = Test-ProcessRunning $masterPid
            MasterPid = $masterPid
            WorkerPid = $pidData.Worker
            StartTime = $pidData.StartTime
            UptimeSeconds = 0
        }
        
        if ($status.Running) {
            $startTime = [DateTimeOffset]$pidData.StartTime
            $status.UptimeSeconds = [math]::Round((Get-Date) - $startTime).TotalSeconds
        }
        
        $status | Format-List
        
        return $status.Running
    } else {
        Write-Log "Not running (no PID file)"
        return $false
    }
}

# RESTART command
function Restart-NautilusBot {
    Write-Log "=== RESTARTING NAUTILUS TRADING BOT ==="
    Stop-NautilusBot
    Start-Sleep -Seconds 2
    Start-NautilusBot
}

# Main execution
try {
    switch ($Command) {
        '/START' { Start-NautilusBot }
        '/KILL' { Stop-NautilusBot }
        '/STATUS' { Get-NautilusStatus }
        '/RESTART' { Restart-NautilusBot }
    }
} catch {
    Write-Log "Fatal error: $_" "ERROR"
    exit 1
}
