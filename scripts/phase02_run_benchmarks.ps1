# Phase 02 M0 Base Benchmark - Run all four configs
# Run this script from the sft-dpo-HomeChef-Agent directory
# Usage: powershell -ExecutionPolicy Bypass -File scripts/phase02_run_benchmarks.ps1

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$configs = @(
    "configs/evaluation/phase02_base_0_6b_frozen.yaml",
    "configs/evaluation/phase02_base_0_6b_diagnostic.yaml",
    "configs/evaluation/phase02_base_1_7b_frozen.yaml",
    "configs/evaluation/phase02_base_1_7b_diagnostic.yaml"
)

foreach ($config in $configs) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  RUNNING: $config" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    $start = Get-Date
    uv run homechef-eval --config $config 2>&1 | Out-Null
    $elapsed = (Get-Date) - $start
    Write-Host "  COMPLETED in $($elapsed.TotalSeconds)s" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ALL 4 BENCHMARKS COMPLETE" -ForegroundColor Green
Write-Host "  Generating combined summary..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

uv run python scripts/eval/run_base_benchmark.py `
    --config configs/evaluation/phase02_base_0_6b_frozen.yaml `
    --config configs/evaluation/phase02_base_0_6b_diagnostic.yaml `
    --config configs/evaluation/phase02_base_1_7b_frozen.yaml `
    --config configs/evaluation/phase02_base_1_7b_diagnostic.yaml

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PHASE 02 M0 BASE BENCHMARK COMPLETE" -ForegroundColor Green
Write-Host "  Output: reports/generated/phase02/" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
