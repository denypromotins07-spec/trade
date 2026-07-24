# =============================================================================
# chrome_kiosk.ps1 - Advanced Chrome Kiosk Launcher
# Nautilus/Ray Trading Bot - Stage 60
# =============================================================================
# Purpose: Launches Chrome with strict GPU rasterization flags, disables
#          background tabs, and forces AMD Radeon GPU for all WebGL/Canvas.
# Constraints: Optimized for 60FPS rendering, microsecond UI response.
# Compatibility: Works with kiosk_bind.ts frontend lockdown.
# =============================================================================

param(
    [string]$Url = "http://localhost:3000/live_dashboard",
    [string]$ChromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe",
    [switch]$DisableGPU, # Fallback flag if GPU causes issues
    [string]$UserDataDir = "$env:TEMP\NautilusKioskProfile"
)

$ErrorActionPreference = "Stop"

Write-Host "[CHROME_KIOSK] Initializing advanced kiosk launcher..." -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# 1. Prepare Clean User Data Directory
# -----------------------------------------------------------------------------
if (Test-Path $UserDataDir) {
    Write-Host "[CHROME_KIOSK] Cleaning existing kiosk profile..." -ForegroundColor Gray
    Remove-Item -Path $UserDataDir -Recurse -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Path $UserDataDir -Force | Out-Null
Write-Host "[CHROME_KIOSK] Profile directory created: $UserDataDir" -ForegroundColor Green

# -----------------------------------------------------------------------------
# 2. Detect AMD GPU and Configure Flags
# -----------------------------------------------------------------------------
$gpuVendor = ""
try {
    $adapters = Get-WmiObject Win32_VideoController
    foreach ($adapter in $adapters) {
        if ($adapter.Name -like "*AMD*" -or $adapter.Name -like "*Radeon*") {
            $gpuVendor = $adapter.Name
            Write-Host "[CHROME_KIOSK] AMD GPU detected: $gpuVendor" -ForegroundColor Green
            break
        }
    }
    
    if (-not $gpuVendor) {
        Write-Warning "[CHROME_KIOSK] No AMD GPU detected. Falling back to default GPU."
    }
} catch {
    Write-Warning "[CHROME_KIOSK] Failed to detect GPU: $_"
}

# -----------------------------------------------------------------------------
# 3. Build Chrome Launch Arguments
# -----------------------------------------------------------------------------
$argsList = @(
    # Kiosk Mode Essentials
    "--kiosk",
    "--kiosk-printing",
    "--no-first-run",
    "--disable-features=TranslateUI",
    "--disable-ipc-flooding-protection",
    
    # Performance & Latency Optimization
    "--disable-gpu-vsync",              # Disable VSync for lowest latency
    "--enable-fast-unload",             # Faster tab closing
    "--disable-background-timer-throttling", # Prevent background throttling
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    
    # GPU Acceleration (AMD Specific)
    "--use-gl=angle",                   # Use ANGLE for better compatibility
    "--ignore-gpu-blocklist",           # Force GPU even if blacklisted
    "--enable-gpu-rasterization",       # GPU-based rasterization
    "--enable-zero-copy",               # Zero-copy texture uploads
    "--gpu-memory-buffer=0",            # Let Chrome manage GPU memory
    
    # Security & Hardening
    "--disable-dev-shm-usage",
    "--no-sandbox",                     # Required for some automation scenarios
    "--disable-web-security",           # Allow local WS connections
    "--allow-running-insecure-content",
    
    # UI Cleanup
    "--hide-scrollbars",
    "--mute-audio",
    "--disable-component-update",
    "--disable-check-for-update",
    
    # User Data
    "--user-data-dir=`"$UserDataDir`"",
    
    # Target URL
    $Url
)

# Add AMD-specific forcing if GPU detected
if ($gpuVendor -and -not $DisableGPU) {
    $argsList += "--force-high-performance-gpu"
    Write-Host "[CHROME_KIOSK] High-performance GPU mode enabled." -ForegroundColor Gray
}

if ($DisableGPU) {
    $argsList += "--disable-gpu"
    Write-Warning "[CHROME_KIOSK] GPU acceleration disabled by user flag."
}

# -----------------------------------------------------------------------------
# 4. Launch Chrome Process
# -----------------------------------------------------------------------------
Write-Host "[CHROME_KIOSK] Launching Chrome with $($argsList.Count) flags..." -ForegroundColor Yellow

try {
    $processInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processInfo.FileName = $ChromePath
    $processInfo.Arguments = $argsList -join " "
    $processInfo.WorkingDirectory = Split-Path $ChromePath
    $processInfo.UseShellExecute = $true
    $processInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Maximized
    
    $process = [System.Diagnostics.Process]::Start($processInfo)
    
    Write-Host "[CHROME_KIOSK] Chrome launched successfully (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "[CHROME_KIOSK] Kiosk URL: $Url" -ForegroundColor Green
    
    if ($gpuVendor) {
        Write-Host "[CHROME_KIOSK] Rendering on: $gpuVendor" -ForegroundColor Cyan
    }
    
} catch {
    Write-Error "[CHROME_KIOSK] Failed to launch Chrome: $_"
    exit 1
}

# -----------------------------------------------------------------------------
# 5. Monitor and Auto-Restart (Optional Watchdog)
# -----------------------------------------------------------------------------
# In production, you might want a watchdog to restart Chrome if it crashes
# For now, we just exit and let ULTIMATE_START.ps1 handle supervision

Write-Host "[CHROME_KIOSK] Launcher complete. Monitoring disabled (handled by supervisor)." -ForegroundColor Gray
