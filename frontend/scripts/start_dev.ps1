# Nautilus/Ray Frontend Development Server
# PowerShell script to concurrently boot Next.js server and inject local .env variables
# Integrates seamlessly with the master /START backend orchestrator

param(
    [switch]$Production,
    [switch]$Analyze,
    [string]$Port = "3000",
    [switch]$NoCache,
    [switch]$Verbose
)

# Cyberpunk header
Write-Host @"

╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗ ██╗   ██╗██╗     ███████╗███████╗               ║
║   ██╔══██╗██║   ██║██║     ██╔════╝██╔════╝               ║
║   ██████╔╝██║   ██║██║     █████╗  ███████╗               ║
║   ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║               ║
║   ██████╔╝╚██████╔╝███████╗███████╗███████║               ║
║   ╚═════╝  ╚═════╝ ╚══════╝╚══════╝╚══════╝               ║
║                                                           ║
║   FRONTEND DEVELOPMENT SERVER                             ║
║   Stage 48 - Build Orchestration                          ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# Configuration
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$FRONTEND_DIR = Split-Path -Parent $SCRIPT_DIR
$ENV_FILE = Join-Path $FRONTEND_DIR ".env.local"
$NEXT_PORT = $Port

# Colors for output
$COLOR_INFO = "Cyan"
$COLOR_SUCCESS = "Green"
$COLOR_WARNING = "Yellow"
$COLOR_ERROR = "Red"

function Write-Status {
    param([string]$Message, [string]$Color = $COLOR_INFO)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message" -ForegroundColor $Color
}

# Check Node.js version
Write-Status "Checking Node.js version..."
try {
    $nodeVersion = node --version
    Write-Status "Node.js: $nodeVersion" $COLOR_SUCCESS
} catch {
    Write-Status "ERROR: Node.js not found. Please install Node.js 18+." $COLOR_ERROR
    exit 1
}

# Check npm version
try {
    $npmVersion = npm --version
    Write-Status "npm: $npmVersion" $COLOR_SUCCESS
} catch {
    Write-Status "ERROR: npm not found." $COLOR_ERROR
    exit 1
}

# Generate/Update .env.local if needed
Write-Status "Configuring environment variables..."

$envVars = @{
    "NEXT_PUBLIC_APP_NAME" = "Nautilus/Ray Trading Bot"
    "NEXT_PUBLIC_APP_VERSION" = "0.1.0"
    "NEXT_PUBLIC_WS_URL" = "ws://localhost:8080/rpc"
    "NEXT_PUBLIC_API_URL" = "http://localhost:8080/api"
    "NEXT_PUBLIC_RPC_TIMEOUT" = "5000"
    "NEXT_PUBLIC_MAX_RETRIES" = "3"
    "NEXT_PUBLIC_ENABLE_PWA" = "true"
    "NEXT_PUBLIC_ENABLE_PUSH" = "true"
    "NEXT_PUBLIC_VAPID_PUBLIC_KEY" = ""
    "NEXT_PUBLIC_SENTRY_DSN" = ""
    "NEXT_PUBLIC_ANALYTICS_ID" = ""
    "NODE_ENV" = $(if ($Production) { "production" } else { "development" })
    "NEXT_TELEMETRY_DISABLED" = "1"
    "NEXT_BUNDLE_ANALYZE" = $(if ($Analyze) { "true" } else { "false" })
}

# Read existing env file if present
$existingEnv = @{}
if (Test-Path $ENV_FILE) {
    Get-Content $ENV_FILE | ForEach-Object {
        if ($_ -match '^([^#=]+)=(.*)$') {
            $existingEnv[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
}

# Merge with defaults (existing values take precedence)
$finalEnv = $envVars.Clone()
foreach ($key in $existingEnv.Keys) {
    $finalEnv[$key] = $existingEnv[$key]
}

# Write updated .env.local
$envContent = $finalEnv.GetEnumerator() | ForEach-Object {
    "$($_.Key)=$($_.Value)"
}
$envContent | Set-Content -Path $ENV_FILE -Encoding UTF8

Write-Status "Environment file: $ENV_FILE" $COLOR_SUCCESS

# Install dependencies if node_modules missing
if (-not (Test-Path (Join-Path $FRONTEND_DIR "node_modules"))) {
    Write-Status "Installing dependencies..."
    npm install --prefer-offline
}

# Clear Next.js cache if requested
if ($NoCache) {
    Write-Status "Clearing Next.js cache..."
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $FRONTEND_DIR ".next")
    Write-Status "Cache cleared" $COLOR_SUCCESS
}

# Build command based on mode
$buildCmd = if ($Production) {
    Write-Status "Building for PRODUCTION..." $COLOR_WARNING
    
    # Run build first
    npm run build
    
    if ($LASTEXITCODE -ne 0) {
        Write-Status "Build failed!" $COLOR_ERROR
        exit 1
    }
    
    Write-Status "Build completed successfully" $COLOR_SUCCESS
    "npm start -- -p $NEXT_PORT"
} elseif ($Analyze) {
    Write-Status "Building with bundle analysis..." $COLOR_WARNING
    $env:NEXT_BUNDLE_ANALYZE = "true"
    "npm run build-and-analyze"
} else {
    Write-Status "Starting DEVELOPMENT server..."
    "npm run dev -- -p $NEXT_PORT"
}

# Start the server
Write-Status "Launching Next.js server on port $NEXT_PORT..."
Write-Status ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" 
Write-Host "  Local:   http://localhost:$NEXT_PORT"
Write-Host "  Network: http://$($env:COMPUTERNAME).local:$NEXT_PORT"
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Write-Host ""

# Set environment variables for the process
$env:NEXT_PUBLIC_WS_URL = $finalEnv["NEXT_PUBLIC_WS_URL"]
$env:NEXT_PUBLIC_API_URL = $finalEnv["NEXT_PUBLIC_API_URL"]
$env:NEXT_TELEMETRY_DISABLED = "1"

if ($Verbose) {
    $env:DEBUG = "nautilus:*"
}

# Execute the server command
Invoke-Expression $buildCmd

# Post-exit handling
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    Write-Status "Server stopped gracefully" $COLOR_SUCCESS
} else {
    Write-Status "Server exited with code $exitCode" $COLOR_WARNING
}

# Check for pending crash reports
$crashIndex = Get-Content -Path (Join-Path $FRONTEND_DIR ".next" "pending_crash_index") -ErrorAction SilentlyContinue
if ($crashIndex) {
    Write-Status ""
    Write-Status "⚠️  PENDING CRASH REPORTS DETECTED" $COLOR_WARNING
    Write-Status "Run 'npm run flush-crashes' to send pending reports" $COLOR_WARNING
}

# Integration with master /START orchestrator
Write-Status ""
Write-Status "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" $COLOR_INFO
Write-Status "  To integrate with master orchestrator:" $COLOR_INFO
Write-Status "  - Backend: ./scripts/start_backend.ps1 /START" $COLOR_INFO
Write-Status "  - Kill All: ./scripts/kill_all.ps1 /KILL" $COLOR_INFO
Write-Status "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" $COLOR_INFO

exit $exitCode
