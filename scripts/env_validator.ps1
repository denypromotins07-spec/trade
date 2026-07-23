# Environment Validator Script
# 
# Pre-boot PowerShell script that strictly validates the .env file schema,
# ensuring no malformed API keys can trigger silent failures in the Rust core.
#
# Validates Binance API key format, required fields, and security constraints.

param(
    [string]$EnvFilePath = ".env",
    [switch]$Strict,
    [switch]$Verbose
)

# Configuration
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
$ValidatorLog = "$ProjectRoot\logs\env_validator.log"

# Ensure log directory exists
$null = New-Item -ItemType Directory -Force -Path "$ProjectRoot\logs"

# Logging function
function Write-ValidatorLog {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
    $logLine = "[$timestamp] [VALIDATOR] [$Level] $Message"
    Add-Content -Path $ValidatorLog -Value $logLine
    if ($Verbose) {
        Write-Host $logLine -ForegroundColor $(if ($Level -eq "ERROR") { "Red" } elseif ($Level -eq "WARN") { "Yellow" } else { "Green" })
    }
}

# Validation result structure
class ValidationResult {
    [bool]$Valid
    [string]$Field
    [string]$Message
    [string]$Level
    
    ValidationResult($valid, $field, $message, $level) {
        $this.Valid = $valid
        $this.Field = $field
        $this.Message = $message
        $this.Level = $level
    }
}

# Parse .env file
function Parse-EnvFile {
    param([string]$Path)
    
    $envVars = @{}
    
    if (-not (Test-Path $Path)) {
        throw ".env file not found at: $Path"
    }
    
    $lines = Get-Content $Path -Encoding UTF8
    
    foreach ($line in $lines) {
        # Skip comments and empty lines
        $line = $line.Trim()
        if ([string]::IsNullOrEmpty($line) -or $line.StartsWith("#")) {
            continue
        }
        
        # Parse KEY=VALUE
        if ($line -match '^([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            
            # Remove surrounding quotes if present
            if ($value -match '^["''](.*)["'']$') {
                $value = $matches[1]
            }
            
            $envVars[$key] = $value
        }
    }
    
    return $envVars
}

# Validate Binance API key format
function Test-BinanceApiKey {
    param([string]$Key)
    
    # Binance API keys are typically 60-64 alphanumeric characters
    if ([string]::IsNullOrEmpty($Key)) {
        return $false
    }
    
    if ($Key.Length -lt 60 -or $Key.Length -gt 128) {
        return $false
    }
    
    # Should be alphanumeric only
    if ($Key -notmatch '^[A-Za-z0-9]+$') {
        return $false
    }
    
    return $true
}

# Validate Binance Secret Key format
function Test-BinanceSecretKey {
    param([string]$Key)
    
    # Secret keys are typically 64 characters
    if ([string]::IsNullOrEmpty($Key)) {
        return $false
    }
    
    if ($Key.Length -ne 64) {
        return $false
    }
    
    # Should be hexadecimal
    if ($Key -notmatch '^[A-Fa-f0-9]{64}$') {
        return $false
    }
    
    return $true
}

# Validate environment variable is not empty
function Test-NotEmpty {
    param([string]$Value)
    return -not [string]::IsNullOrEmpty($Value)
}

# Main validation function
function Invoke-EnvValidation {
    Write-ValidatorLog "=== ENVIRONMENT VALIDATION STARTED ==="
    Write-ValidatorLog "Env file path: $EnvFilePath"
    
    $allValid = $true
    $results = @()
    
    # Check if file exists
    if (-not (Test-Path $EnvFilePath)) {
        Write-ValidatorLog ".env file NOT FOUND!" "ERROR"
        return $false
    }
    
    try {
        $envVars = Parse-EnvFile $EnvFilePath
        Write-ValidatorLog "Parsed $($envVars.Count) environment variables"
    } catch {
        Write-ValidatorLog "Failed to parse .env file: $_" "ERROR"
        return $false
    }
    
    # Required fields for Nautilus bot
    $requiredFields = @(
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "RAY_REDIS_ADDRESS",
        "LOG_LEVEL"
    )
    
    # Optional but recommended fields
    $optionalFields = @(
        "BINANCE_TESTNET",
        "MAX_POSITION_SIZE",
        "RISK_PER_TRADE",
        "TRADING_PAIRS",
        "WS_HEARTBEAT_INTERVAL_MS"
    )
    
    # Validate required fields exist
    foreach ($field in $requiredFields) {
        if (-not $envVars.ContainsKey($field)) {
            $results += [ValidationResult]::new($false, $field, "Required field missing", "ERROR")
            $allValid = $false
            Write-ValidatorLog "MISSING required field: $field" "ERROR"
        } elseif (-not (Test-NotEmpty $envVars[$field])) {
            $results += [ValidationResult]::new($false, $field, "Field is empty", "ERROR")
            $allValid = $false
            Write-ValidatorLog "EMPTY required field: $field" "ERROR"
        } else {
            $results += [ValidationResult]::new($true, $field, "Present", "OK")
        }
    }
    
    # Validate BINANCE_API_KEY format
    if ($envVars.ContainsKey("BINANCE_API_KEY") -and (Test-NotEmpty $envVars["BINANCE_API_KEY"])) {
        $apiKey = $envVars["BINANCE_API_KEY"]
        if (-not (Test-BinanceApiKey $apiKey)) {
            $results += [ValidationResult]::new($false, "BINANCE_API_KEY", "Invalid format (expected 60-128 alphanumeric chars)", "ERROR")
            $allValid = $false
            Write-ValidatorLog "INVALID Binance API key format" "ERROR"
        } else {
            Write-ValidatorLog "Binance API key format valid" "OK"
        }
    }
    
    # Validate BINANCE_API_SECRET format
    if ($envVars.ContainsKey("BINANCE_API_SECRET") -and (Test-NotEmpty $envVars["BINANCE_API_SECRET"])) {
        $secret = $envVars["BINANCE_API_SECRET"]
        if (-not (Test-BinanceSecretKey $secret)) {
            $results += [ValidationResult]::new($false, "BINANCE_API_SECRET", "Invalid format (expected 64 hex chars)", "ERROR")
            $allValid = $false
            Write-ValidatorLog "INVALID Binance secret key format" "ERROR"
        } else {
            Write-ValidatorLog "Binance secret key format valid" "OK"
        }
    }
    
    # Security check: warn if testnet flag is not set in strict mode
    if ($Strict) {
        if (-not $envVars.ContainsKey("BINANCE_TESTNET")) {
            $results += [ValidationResult]::new($false, "BINANCE_TESTNET", "Not set (required in strict mode)", "WARN")
            Write-ValidatorLog "WARNING: BINANCE_TESTNET not set in strict mode" "WARN"
        } elseif ($envVars["BINANCE_TESTNET"] -ne "true" -and $envVars["BINANCE_TESTNET"] -ne "1") {
            $results += [ValidationResult]::new($false, "BINANCE_TESTNET", "Live trading enabled in strict mode!", "WARN")
            Write-ValidatorLog "WARNING: Live trading mode detected in strict mode!" "WARN"
        }
    }
    
    # Validate numeric fields
    $numericFields = @("MAX_POSITION_SIZE", "RISK_PER_TRADE", "WS_HEARTBEAT_INTERVAL_MS")
    foreach ($field in $numericFields) {
        if ($envVars.ContainsKey($field) -and (Test-NotEmpty $envVars[$field])) {
            $value = $envVars[$field]
            if ($value -notmatch '^-?\d+(\.\d+)?$') {
                $results += [ValidationResult]::new($false, $field, "Must be a valid number", "ERROR")
                $allValid = $false
                Write-ValidatorLog "INVALID numeric value for $field: $value" "ERROR"
            }
        }
    }
    
    # Validate LOG_LEVEL
    if ($envVars.ContainsKey("LOG_LEVEL")) {
        $validLevels = @("trace", "debug", "info", "warn", "error")
        $logLevel = $envVars["LOG_LEVEL"].ToLower()
        if ($validLevels -notcontains $logLevel) {
            $results += [ValidationResult]::new($false, "LOG_LEVEL", "Must be one of: trace, debug, info, warn, error", "WARN")
            Write-ValidatorLog "INVALID log level: $logLevel" "WARN"
        }
    }
    
    # Check for potentially dangerous patterns
    $dangerousPatterns = @(
        "eval",
        "exec",
        "`$",
        "&&",
        "||",
        ";",
        "|"
    )
    
    foreach ($kv in $envVars.GetEnumerator()) {
        foreach ($pattern in $dangerousPatterns) {
            if ($kv.Value -like "*$pattern*") {
                $results += [ValidationResult]::new($false, $kv.Key, "Contains potentially dangerous character: $pattern", "WARN")
                Write-ValidatorLog "WARNING: Field $($kv.Key) contains dangerous pattern" "WARN"
            }
        }
    }
    
    # Summary
    Write-ValidatorLog "=== VALIDATION SUMMARY ==="
    $errorCount = ($results | Where-Object { $_.Level -eq "ERROR" }).Count
    $warnCount = ($results | Where-Object { $_.Level -eq "WARN" }).Count
    $okCount = ($results | Where-Object { $_.Level -eq "OK" }).Count
    
    Write-ValidatorLog "Passed: $okCount"
    Write-ValidatorLog "Warnings: $warnCount"
    Write-ValidatorLog "Errors: $errorCount"
    
    if ($allValid) {
        Write-ValidatorLog "ENVIRONMENT VALIDATION PASSED" "OK"
    } else {
        Write-ValidatorLog "ENVIRONMENT VALIDATION FAILED - Fix errors before starting" "ERROR"
    }
    
    return $allValid
}

# Export validated environment (creates .env.validated)
function Export-ValidatedEnv {
    param([string]$SourcePath, [string]$OutputPath)
    
    if (-not (Invoke-EnvValidation -EnvFilePath $SourcePath)) {
        Write-ValidatorLog "Cannot export - validation failed" "ERROR"
        return $false
    }
    
    $envVars = Parse-EnvFile $SourcePath
    
    # Write validated file with header
    $header = @"
# Validated Environment File
# Generated: $(Get-Date -Format "o")
# DO NOT EDIT MANUALLY - Re-run env_validator.ps1 to regenerate
#
"@
    
    Set-Content -Path $OutputPath -Value $header
    
    foreach ($kv in $envVars.GetEnumerator()) {
        Add-Content -Path $OutputPath -Value "$($kv.Key)=$($kv.Value)"
    }
    
    Write-ValidatorLog "Validated environment exported to: $OutputPath"
    return $true
}

# Main execution
try {
    $fullPath = if (Test-Path $EnvFilePath -PathType Leaf) {
        (Resolve-Path $EnvFilePath).Path
    } elseif (Test-Path "$ProjectRoot\$EnvFilePath") {
        (Resolve-Path "$ProjectRoot\$EnvFilePath").Path
    } else {
        $EnvFilePath
    }
    
    $result = Invoke-EnvValidation -EnvFilePath $fullPath
    
    if (-not $result) {
        exit 1
    }
    
    exit 0
    
} catch {
    Write-ValidatorLog "Validation error: $_" "ERROR"
    exit 1
}
