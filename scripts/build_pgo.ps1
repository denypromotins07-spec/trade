# PGO Build Script - Stage 54
# Profile-Guided Optimization workflow for Rust matching engine
param(
    [ValidateSet("instrument", "replay", "optimize", "full")]
    [string]$Mode = "full",
    [int]$RamLimitGB = 8
)

$ErrorActionPreference = "Stop"
$Global:MaxMemoryBytes = $RamLimitGB * 1GB

function Test-MemoryLimit {
    $usage = (Get-Process -Id $PID).WorkingSet64
    if ($usage -gt $Global:MaxMemoryBytes) {
        throw "Memory limit exceeded: $([math]::Round($usage/1GB,2))GB > ${RamLimitGB}GB"
    }
    Write-Host "[MEM] Usage: $([math]::Round($usage/1MB,0))MB / ${RamLimitGB}GB" -ForegroundColor Cyan
}

function Invoke-PGOInstrument {
    Write-Host "`n[STAGE 1] Building instrumented binary..." -ForegroundColor Green
    $env:RUSTFLAGS = "-Cprofile-generate=./target/pgo-data"
    cargo build --release --profile pgo-instrument --bin pgo_instrument
    Test-MemoryLimit
}

function Invoke-MarketReplay {
    Write-Host "`n[STAGE 2] Running market replay for profile collection..." -ForegroundColor Green
    New-Item -ItemType Directory -Path "./target/pgo-data" -Force | Out-Null
    & ./target/pgo-instrument/pgo_instrument --replay --input=./data/market_replay
    Test-MemoryLimit
}

function Invoke-PGOOptimize {
    Write-Host "`n[STAGE 3] Building PGO-optimized binary..." -ForegroundColor Green
    $env:RUSTFLAGS = "-Cprofile-use=./target/pgo-data/merged.profdata -Ctarget-cpu=znver4"
    cargo build --release --bin nautilus_engine
    Write-Host "`n[COMPLETE] PGO build finished" -ForegroundColor Green
}

Write-Host "`n=== NAUTILUS PGO BUILD v5.4 ===" -ForegroundColor Magenta
Write-Host "RAM Limit: ${RamLimitGB}GB (enforced)`n" -ForegroundColor Cyan

switch ($Mode) {
    "instrument" { Invoke-PGOInstrument }
    "replay" { Invoke-MarketReplay }
    "optimize" { Invoke-PGOOptimize }
    "full" { 
        Invoke-PGOInstrument
        [GC]::Collect()
        Invoke-MarketReplay
        [GC]::Collect()
        Invoke-PGOOptimize
    }
}
Write-Host "`nBuild complete!`n" -ForegroundColor Green
