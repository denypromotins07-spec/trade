# chrome_launcher.ps1 - Chrome Automation with Hardware Acceleration
# Stage 54: Nautilus/Ray Crypto Trading Bot
# Binds to Rust gateway port, disables background throttling, AMD Ryzen AI 5 optimized

param(
    [int]$GatewayPort = 8080,
    [string]$LogPrefix = "[CHROME_LAUNCHER]"
)

$ErrorActionPreference = "Stop"
$BaseDir = Split-Path -Parent $PSScriptRoot

Write-Host "$LogPrefix Initializing Chrome Launcher for Nautilus Bot" -ForegroundColor Cyan
Write-Host "$LogPrefix Target Gateway Port: $GatewayPort" -ForegroundColor Yellow

# AMD Ryzen AI 5 specific optimizations
# Leverages hardware video decoding and DirectML acceleration where available
$chromeArgs = @(
    # Kiosk mode for trading terminal display
    "--kiosk"
    "--kiosk-printing"
    
    # Bind to exact localhost port allocated by Rust gateway
    "--app=http://localhost:$GatewayPort"
    
    # Hardware acceleration flags for AMD Ryzen AI 5
    "--use-gl=angle"
    "--enable-gpu-rasterization"
    "--enable-zero-copy"
    "--gpu-memory-buffer-size=262144"
    "--num-raster-threads=4"
    
    # Disable background throttling for real-time trading updates
    "--disable-background-timer-throttling"
    "--disable-backgrounding-occluded-windows"
    "--disable-renderer-backgrounding"
    "--disable-background-networking"
    
    # Performance optimizations for microsecond latency
    "--disable-features=TranslateUI"
    "--disable-features=InterestFeedContentSuggestions"
    "--disable-features=MediaRouter"
    "--disable-extensions"
    "--disable-component-extensions-with-background-pages"
    "--disable-default-apps"
    
    # Memory management within 8GB system limit
    "--js-flags='--max-old-space-size=2048'"
    "--max-old-space-size=2048"
    
    # Security and isolation for trading session
    "--no-first-run"
    "--no-default-browser-check"
    "--disable-breakpad"
    "--disable-crash-reporter"
    "--disable-client-side-phishing-detection"
    
    # Nautilus bot specific identifier for process tracking
    "--user-data-dir=$(Join-Path $BaseDir 'chrome_profile')"
    "--disk-cache-dir=$(Join-Path $BaseDir 'chrome_cache')"
    "--window-name=Nautilus-Trading-Terminal"
    
    # Enable DirectML for potential GPU-accelerated computations
    "--enable-features=WebGPU"
    "--use-vulkan=natural"
)

# Function to find Chrome installation path
function Get-ChromePath {
    $possiblePaths = @(
        "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe",
        "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
        "${env:LOCALAPPDATA}\Google\Chrome\Application\chrome.exe",
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    throw "Google Chrome not found in standard installation paths"
}

# Function to kill existing Chrome instances for this bot
function Stop-ExistingChrome {
    Write-Host "$LogPrefix Checking for existing Chrome instances..." -ForegroundColor Yellow
    $existingChrome = Get-Process -Name "chrome" -ErrorAction SilentlyContinue | 
        Where-Object { $_.CommandLine -like "*--window-name=Nautilus-Trading-Terminal*" }
    
    if ($existingChrome) {
        Write-Host "$LogPrefix Terminating existing Chrome instance..." -ForegroundColor Red
        foreach ($proc in $existingChrome) {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 1000
    }
}

# Function to verify gateway connectivity before launching Chrome
function Test-GatewayReady {
    param([int]$Port, [int]$MaxRetries = 20)
    $retryCount = 0
    while ($retryCount -lt $MaxRetries) {
        try {
            $tcpClient = New-Object System.Net.Sockets.TcpClient("localhost", $Port)
            $tcpClient.Close()
            Write-Host "$LogPrefix Gateway ready on port $Port" -ForegroundColor Green
            return $true
        } catch {
            $retryCount++
            Start-Sleep -Milliseconds 250
        }
    }
    Write-Host "$LogPrefix WARNING: Gateway not responding after $MaxRetries attempts" -ForegroundColor Red
    return $false
}

try {
    # Step 1: Kill any orphaned Chrome processes from previous runs
    Stop-ExistingChrome
    
    # Step 2: Verify gateway is ready before launching browser
    Write-Host "$LogPrefix Verifying gateway connectivity..." -ForegroundColor Yellow
    if (-not (Test-GatewayReady -Port $GatewayPort)) {
        Write-Host "$LogPrefix Proceeding anyway - gateway may start shortly" -ForegroundColor Yellow
    }
    
    # Step 3: Create necessary directories for Chrome profile
    $profileDir = Join-Path $BaseDir "chrome_profile"
    $cacheDir = Join-Path $BaseDir "chrome_cache"
    if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir | Out-Null }
    if (-not (Test-Path $cacheDir)) { New-Item -ItemType Directory -Path $cacheDir | Out-Null }
    
    # Step 4: Get Chrome executable path
    $chromePath = Get-ChromePath
    Write-Host "$LogPrefix Chrome executable: $chromePath" -ForegroundColor Green
    
    # Step 5: Launch Chrome with optimized arguments
    Write-Host "$LogPrefix Launching Chrome in kiosk mode with AMD optimizations..." -ForegroundColor Cyan
    $chromeProcess = Start-Process -FilePath $chromePath `
        -ArgumentList $chromeArgs `
        -PassThru `
        -WindowStyle Normal
    
    Write-Host "$LogPrefix Chrome launched successfully (PID: $($chromeProcess.Id))" -ForegroundColor Green
    
    # Step 6: Wait for Chrome to fully load the trading interface
    Write-Host "$LogPrefix Waiting for trading interface to load..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    # Verify Chrome is running
    if ($chromeProcess.HasExited) {
        throw "Chrome process exited unexpectedly"
    }
    
    Write-Host "`n$LogPrefix ========================================" -ForegroundColor Cyan
    Write-Host "$LogPrefix Chrome Launcher Successfully Completed" -ForegroundColor Green
    Write-Host "$LogPrefix Process ID: $($chromeProcess.Id)" -ForegroundColor Green
    Write-Host "$LogPrefix Gateway Port: $GatewayPort" -ForegroundColor Green
    Write-Host "$LogPrefix Hardware Acceleration: Enabled" -ForegroundColor Green
    Write-Host "$LogPrefix Background Throttling: Disabled" -ForegroundColor Green
    Write-Host "$LogPrefix ========================================" -ForegroundColor Cyan
    
    # Return process ID for parent script tracking
    return $chromeProcess.Id
    
} catch {
    Write-Host "$LogPrefix FATAL ERROR: $_" -ForegroundColor Red
    # Attempt cleanup on failure
    Stop-ExistingChrome
    exit 1
}
