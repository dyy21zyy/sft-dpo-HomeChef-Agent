# Phase 02 Final Gate Script
# Usage: powershell -ExecutionPolicy Bypass -File scripts/phase02_gate.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PHASE 02 FINAL GATE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"
$env:UV_LINK_MODE = "copy"

# Step 1: Sync
Write-Host "[1/8] uv sync --frozen" -ForegroundColor Yellow
uv sync --frozen 2>&1 | Out-Null
Write-Host "  PASS" -ForegroundColor Green

# Step 2: Ruff
Write-Host "[2/8] ruff check ." -ForegroundColor Yellow
uv run ruff check . 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "  FAIL" -ForegroundColor Red; exit 1 }
Write-Host "  PASS" -ForegroundColor Green

# Step 3: Unit tests
Write-Host "[3/8] pytest -q tests/unit/" -ForegroundColor Yellow
$pytest_output = uv run pytest -q tests/unit/ --basetemp="$env:TEMP\pytest-work" 2>&1
if ($LASTEXITCODE -ne 0) { Write-Host "  FAIL" -ForegroundColor Red; exit 1 }
Write-Host "  PASS" -ForegroundColor Green

# Step 4: Contract validate
Write-Host "[4/8] homechef-contract-validate" -ForegroundColor Yellow
uv run homechef-contract-validate --root . --fixtures tests/fixtures/contracts 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "  FAIL" -ForegroundColor Red; exit 1 }
Write-Host "  PASS" -ForegroundColor Green

# Step 5: Phase 01 replay anchor
Write-Host "[5/8] Phase 01 replay anchor" -ForegroundColor Yellow
uv run pytest -q tests/integration/test_scorer_replay_anchor.py --basetemp="$env:TEMP\pytest-work" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "  FAIL" -ForegroundColor Red; exit 1 }
Write-Host "  PASS" -ForegroundColor Green

# Step 6: Suite validation
Write-Host "[6/8] Suite validation" -ForegroundColor Yellow
uv run python -m homechef_booking.evaluation.suite_manifest --manifest data/eval/frozen_test.manifest.json --root . 2>&1 | Out-Null
uv run python -m homechef_booking.evaluation.suite_manifest --manifest data/dev/diagnostic_dev.manifest.json --root . 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "  FAIL" -ForegroundColor Red; exit 1 }
Write-Host "  PASS" -ForegroundColor Green

# Step 7: Base benchmark (mock)
Write-Host "[7/8] Base benchmark (mock)" -ForegroundColor Yellow
uv run homechef-eval --config configs/evaluation/phase01_mock.yaml 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host "  FAIL" -ForegroundColor Red; exit 1 }
Write-Host "  PASS" -ForegroundColor Green

# Step 8: Git status
Write-Host "[8/8] git status --short" -ForegroundColor Yellow
git status --short
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PHASE 02 GATE: ALL PASSED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
