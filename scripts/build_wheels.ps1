# Python Wheel Builder - Stage 54
# Compiles Cython/Numba extensions against AMD ROCm/DirectML
param(
    [ValidateSet("win_amd64", "linux_x86_64")]
    [string]$TargetArch = "win_amd64",
    [int]$RamLimitGB = 4,
    [switch]$UseCache = $true
)

$ErrorActionPreference = "Stop"
$Global:MaxMemoryBytes = $RamLimitGB * 1GB
$AMD_DIRECTML = "C:\Program Files\DirectML"
$AMD_ROCM = "C:\Program Files\AMD\ROCm"

Write-Host "`n=== NAUTILUS WHEEL BUILDER v5.4 ===" -ForegroundColor Magenta
Write-Host "RAM Limit: ${RamLimitGB}GB (Python quota)`n" -ForegroundColor Cyan

function Test-Memory {
    $usage = (Get-Process -Id $PID).WorkingSet64
    if ($usage -gt $Global:MaxMemoryBytes) { throw "Python RAM quota exceeded!" }
    Write-Host "[MEM] $([math]::Round($usage/1MB,0))MB / ${RamLimitGB}GB" -ForegroundColor DarkGray
}

function Test-AMDGPU {
    Write-Host "[GPU] Checking AMD libraries..." -ForegroundColor Cyan
    if (Test-Path $AMD_DIRECTML) { 
        Write-Host "[GPU] DirectML found: $AMD_DIRECTML" -ForegroundColor Green
        $env:DML_PATH = $AMD_DIRECTML
        return "DirectML"
    } elseif (Test-Path $AMD_ROCM) {
        Write-Host "[GPU] ROCm found: $AMD_ROCM" -ForegroundColor Green
        return "ROCm"
    } else {
        Write-Host "[GPU] No AMD GPU, using CPU fallback" -ForegroundColor Yellow
        return "CPU"
    }
}

function Invoke-CythonBuild {
    Write-Host "`n[CYTHON] Compiling extensions..." -ForegroundColor Yellow
    $cythonFlags = "--3str --directive boundscheck=False --directive wraparound=False"
    Get-ChildItem -Path "./python/cython" -Filter "*.pyx" -Recurse | ForEach-Object {
        Write-Host "  Compiling: $($_.Name)" -ForegroundColor DarkGray
        cython $cythonFlags -o $($_.FullName -replace "\.pyx$", ".c") $_.FullName
        Test-Memory
    }
}

function Invoke-NumbaAOT {
    Write-Host "`n[NUMBA] AOT compilation..." -ForegroundColor Yellow
    $env:NUMBA_CPU_NAME = "znver4"
    if (Test-Path "./python/numba_kernels/compile_aot.py") {
        python ./python/numba_kernels/compile_aot.py
        Test-Memory
    }
}

function Invoke-WheelBuild {
    Write-Host "`n[WHEEL] Building distribution..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "./dist/wheels" -Force | Out-Null
    python -m build --wheel -C--plat-name=$TargetArch
    Test-Memory
}

function Invoke-Cache {
    if ($UseCache) {
        Write-Host "`n[CACHE] Caching wheels..." -ForegroundColor Yellow
        Copy-Item ./dist/*.whl ./build_cache/wheels/ -Force -ErrorAction SilentlyContinue
    }
}

# Main execution
$gpuBackend = Test-AMDGPU
Invoke-CythonBuild
[GC]::Collect()
Invoke-NumbaAOT
[GC]::Collect()
Invoke-WheelBuild
Invoke-Cache

Write-Host "`n=== BUILD COMPLETE ===" -ForegroundColor Green
Write-Host "GPU Backend: $gpuBackend" -ForegroundColor Cyan
Write-Host "Output: ./dist/wheels`n" -ForegroundColor Cyan
