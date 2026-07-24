# global_start.ps1 - Master Lifecycle Orchestration /START Script
# Stage 54: Nautilus/Ray Crypto Trading Bot
# Optimized for AMD Ryzen AI 5, 8GB RAM limit, microsecond latency

$ErrorActionPreference = "Stop"
$LogPrefix = "[GLOBAL_START]"
$BaseDir = Split-Path -Parent $PSScriptRoot

Write-Host "$LogPrefix Initializing Nautilus/Ray Trading Bot - Stage 54" -ForegroundColor Cyan

# Function to check if port is in use
function Test-PortAvailable {
    param([int]$Port)
    $tcpListener = $null
    try {
        $tcpListener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $tcpListener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($tcpListener -ne $null) {
            $tcpListener.Stop()
        }
    }
}

# Function to wait for process readiness
function Wait-ProcessReady {
    param(
        [string]$ProcessName,
        [int]$TimeoutSeconds = 30,
        [string]$PortFile = ""
    )
    $elapsed = 0
    while ($elapsed -lt $TimeoutSeconds) {
        if (-not [string]::IsNullOrEmpty($PortFile) -and (Test-Path $PortFile)) {
            Write-Host "$LogPrefix $ProcessName ready (port file exists)" -ForegroundColor Green
            return $true
        }
        if (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue) {
            Write-Host "$LogPrefix $ProcessName running" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Milliseconds 500
        $elapsed += 0.5
    }
    Write-Host "$LogPrefix WARNING: $ProcessName startup timeout" -ForegroundColor Yellow
    return $false
}

try {
    # Step 1: Boot Rust Core Gateway
    Write-Host "$LogPrefix Step 1/4: Booting Rust Core Gateway..." -ForegroundColor Yellow
    $rustCorePath = Join-Path $BaseDir "target\release\nautilus_gateway.exe"
    if (Test-Path $rustCorePath) {
        $rustJob = Start-Job -ScriptBlock {
            Set-Location $using:BaseDir
            & $using:rustCorePath
        }
        Write-Host "$LogPrefix Rust core job started (PID: $($rustJob.Id))" -ForegroundColor Green
    } else {
        Write-Host "$LogPrefix Building Rust core in release mode..." -ForegroundColor Yellow
        Push-Location $BaseDir
        cargo build --release
        Pop-Location
        $rustJob = Start-Job -ScriptBlock {
            Set-Location $using:BaseDir
            & "$using\BaseDir\target\release\nautilus_gateway.exe"
        }
    }
    
    # Wait for port allocator to write shared memory file
    $portFile = Join-Path $BaseDir "shared\gateway_port.txt"
    Wait-ProcessReady -ProcessName "nautilus_gateway" -PortFile $portFile -TimeoutSeconds 45
    
    # Read allocated port for Chrome launcher
    $gatewayPort = 8080
    if (Test-Path $portFile) {
        $gatewayPort = Get-Content $portFile | Select-Object -First 1
        Write-Host "$LogPrefix Rust gateway bound to port: $gatewayPort" -ForegroundColor Green
    }
    
    # Step 2: Initialize Python Ray Cluster
    Write-Host "$LogPrefix Step 2/4: Initializing Python Ray Cluster..." -ForegroundColor Yellow
    $rayInitScript = Join-Path $BaseDir "python\soul\continuous_learner.py"
    if (Test-Path $rayInitScript) {
        $rayJob = Start-Job -ScriptBlock {
            Set-Location $using:BaseDir
            # Enforce 4GB RAM quota for Python processes
            $env:RAY_MEMORY_LIMIT = "4294967296"
            python -c "import ray; ray.init(num_cpus=4, object_store_memory=2147483648); from soul.continuous_learner import start_learner; start_learner()"
        }
        Write-Host "$LogPrefix Ray cluster job started (PID: $($rayJob.Id))" -ForegroundColor Green
    } else {
        throw "$LogPrefix CRITICAL: Python learner script not found at $rayInitScript"
    }
    
    # Step 3: Compile Next.js Frontend
    Write-Host "$LogPrefix Step 3/4: Compiling Next.js Frontend..." -ForegroundColor Yellow
    $frontendDir = Join-Path $BaseDir "frontend"
    if (Test-Path (Join-Path $frontendDir "package.json")) {
        Push-Location $frontendDir
        npm install --silent
        npm run build --silent
        Pop-Location
        Write-Host "$LogPrefix Next.js build completed" -ForegroundColor Green
    } else {
        Write-Host "$LogPrefix WARNING: Frontend package.json not found, skipping build" -ForegroundColor Yellow
    }
    
    # Step 4: Launch Chrome in Kiosk Mode with Hardware Acceleration
    Write-Host "$LogPrefix Step 4/4: Launching Chrome Automation..." -ForegroundColor Yellow
    $chromeLauncher = Join-Path $BaseDir "scripts\chrome_launcher.ps1"
    if (Test-Path $chromeLauncher) {
        & $chromeLauncher -GatewayPort $gatewayPort
        Write-Host "$LogPrefix Chrome launched in kiosk mode on port $gatewayPort" -ForegroundColor Green
    } else {
        throw "$LogPrefix CRITICAL: Chrome launcher script not found at $chromeLauncher"
    }
    
    # Final status
    Write-Host "`n$LogPrefix ========================================" -ForegroundColor Cyan
    Write-Host "$LogPrefix Nautilus/Ray Bot Successfully Started" -ForegroundColor Green
    Write-Host "$LogPrefix Gateway Port: $gatewayPort" -ForegroundColor Green
    Write-Host "$LogPrefix Rust Core: Running" -ForegroundColor Green
    Write-Host "$LogPrefix Ray Cluster: Running" -ForegroundColor Green
    Write-Host "$LogPrefix Frontend: Compiled" -ForegroundColor Green
    Write-Host "$LogPrefix Chrome: Kiosk Mode Active" -ForegroundColor Green
    Write-Host "$LogPrefix ========================================" -ForegroundColor Cyan
    Write-Host "`n$LogPrefix Press Ctrl+C or run global_kill.ps1 to shutdown" -ForegroundColor White
    
    # Keep script alive to intercept Ctrl+C
    Register-EngineEvent -SourceIdentifier Microsoft.PowerShell.Engine.Signals -SupportEvent
    while ($true) {
        Start-Sleep -Seconds 1
    }
    
} catch {
    Write-Host "$LogPrefix FATAL ERROR: $_" -ForegroundColor Red
    # Invoke kill script on failure
    $killScript = Join-Path $BaseDir "scripts\global_kill.ps1"
    if (Test-Path $killScript) {
        & $killScript -Force
    }
    exit 1
}
