# Phase 02 llama.cpp CPU Smoke Test
#
# Runs 1 Frozen Test case per model through the full pipeline:
#   GGUF -> llama-server.exe -> HomeChef eval -> verify artifacts
#
# Usage:
#   .\scripts\eval\smoke_phase02_llama_cpp_cpu.ps1
#
# Prerequisites:
#   - GGUF models in models/gguf/
#   - llama-server.exe in deployment/llama_cpp/bin/
#   - Python venv with homechef-eval installed

param(
    [int]$Port = 8080,
    [int]$CtxSize = 2048,
    [string]$HostAddr = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."

$LlamaServer = "$ProjectRoot\deployment\llama_cpp\bin\llama-server.exe"
$ModelsDir = "$ProjectRoot\models\gguf"
$SmokeCasesPath = "$ProjectRoot\data\eval\frozen_test.jsonl"

# Verify prerequisites
if (-not (Test-Path $LlamaServer)) {
    Write-Error "llama-server.exe not found at $LlamaServer"
    exit 1
}

# Model definitions for smoke test
$Models = @(
    @{
        name = "0.6B"
        model_id = "Qwen/Qwen3-0.6B-Base"
        gguf = "$ModelsDir\Qwen3-0.6B-Base-Q8_0.gguf"
        config = "configs/evaluation/phase02_base_0_6b_frozen.yaml"
        timeout = 300
    },
    @{
        name = "1.7B"
        model_id = "Qwen/Qwen3-1.7B-Base"
        gguf = "$ModelsDir\Qwen3-1.7B-Base-Q8_0.gguf"
        config = "configs/evaluation/phase02_base_1_7b_frozen.yaml"
        timeout = 600
    },
    @{
        name = "4B"
        model_id = "Qwen/Qwen3-4B-Base"
        gguf = "$ModelsDir\Qwen3-4B-Base-Q4_K_M.gguf"
        config = "configs/evaluation/phase02_base_4b_frozen.yaml"
        timeout = 900
    }
)

# Get CPU thread count for --threads auto
$CpuThreads = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
if (-not $CpuThreads) { $CpuThreads = 4 }

Write-Host "============================================================"
Write-Host "Phase 02 llama.cpp CPU Smoke Test"
Write-Host "============================================================"
Write-Host "Host: ${HostAddr}:${Port}"
Write-Host "CPU threads: $CpuThreads"
Write-Host ""

$AllPassed = $true

foreach ($Model in $Models) {
    Write-Host "============================================================"
    Write-Host "[SMOKE] $($Model.name) - $($Model.model_id)"
    Write-Host "============================================================"

    # Check GGUF exists
    if (-not (Test-Path $Model.gguf)) {
        Write-Error "GGUF not found: $($Model.gguf)"
        $AllPassed = $false
        continue
    }
    $ggufSize = [math]::Round((Get-Item $Model.gguf).Length / 1MB, 1)
    Write-Host "[GGUF] $($Model.gguf) ($ggufSize MB)"

    # Prepare 1-case smoke dataset
    $SmokeDataDir = "$ProjectRoot\data\tmp"
    New-Item -ItemType Directory -Force -Path $SmokeDataDir | Out-Null
    $SmokeCaseFile = "$SmokeDataDir\smoke_frozen_1case.jsonl"

    # Extract first case from frozen_test.jsonl
    $FirstSmokeCase = [System.IO.File]::ReadLines(
        $SmokeCasesPath,
        [System.Text.Encoding]::UTF8
    ) | Select-Object -First 1
    [System.IO.File]::WriteAllText(
        $SmokeCaseFile,
        $FirstSmokeCase + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding($false))
    )

    # Create temporary smoke eval config
    $SmokeConfigDir = "$ProjectRoot\experiments\phase02\smoke"
    New-Item -ItemType Directory -Force -Path $SmokeConfigDir | Out-Null
    $SmokeConfigFile = "$SmokeConfigDir\smoke_$($Model.name)_frozen.yaml"
    $BackendConfigPath = switch ($Model.name) {
        "0.6B" { "configs/inference/llama_cpp_0_6b_cpu.yaml" }
        "1.7B" { "configs/inference/llama_cpp_1_7b_cpu.yaml" }
        "4B"   { "configs/inference/llama_cpp_4b_cpu.yaml" }
        default { throw "Unknown smoke model name: $($Model.name)" }
    }

    @"
run_id: smoke_phase02_base_$($Model.name)_frozen
cases_path: $SmokeCaseFile
backend_config_path: $BackendConfigPath
manifest_path: data/eval/frozen_test.manifest.json
output_dir: experiments/phase02/smoke/base_$($Model.name)/frozen_test
model_id: $($Model.model_id)
suite_id: phase02_frozen_test_v1
device: cpu
runtime: llama.cpp
model_format: gguf
quantization: Q8_0
gpu_layers: 0
notes: "Phase 02 Smoke: $($Model.model_id) on llama.cpp CPU (1 case)"
tags: ["smoke", "base_benchmark", "$($Model.name)", "frozen", "llama_cpp", "cpu"]
"@ | Set-Content $SmokeConfigFile -Encoding UTF8

    # Start llama-server
    Write-Host "[SERVER] Starting llama-server.exe..."
    $ServerStdoutLog = "$SmokeConfigDir\server_$($Model.name).stdout.log"
    $ServerStderrLog = "$SmokeConfigDir\server_$($Model.name).stderr.log"
    $ServerProcess = Start-Process -FilePath $LlamaServer -ArgumentList @(
        "-m", $Model.gguf,
        "--host", $HostAddr,
        "--port", $Port,
        "--ctx-size", $CtxSize,
        "--threads", $CpuThreads,
        "-ngl", "0"
    ) -NoNewWindow -PassThru -RedirectStandardOutput $ServerStdoutLog -RedirectStandardError $ServerStderrLog

    # Wait for server to be healthy
    Write-Host "[SERVER] Waiting for health check..."
    $Healthy = $false
    $MaxWait = 60
    for ($i = 0; $i -lt $MaxWait; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri "http://${HostAddr}:${Port}/v1/models" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
            if ($resp.StatusCode -eq 200) {
                $Healthy = $true
                Write-Host "[SERVER] Healthy after $i seconds"
                break
            }
        } catch {}
        Start-Sleep -Seconds 1
    }

    if (-not $Healthy) {
        Write-Error "llama-server failed to become healthy within ${MaxWait}s"
        Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
        $AllPassed = $false
        continue
    }

    # Run evaluation
    Write-Host "[EVAL] Running smoke evaluation..."
    try {
        Push-Location $ProjectRoot
        & uv run homechef-eval --config $SmokeConfigFile
        $EvalExit = $LASTEXITCODE
        Pop-Location

        if ($EvalExit -ne 0) {
            Write-Error "Evaluation failed with exit code $EvalExit"
            $AllPassed = $false
        } else {
            Write-Host "[PASS] Smoke evaluation completed successfully"
        }
    } catch {
        Write-Error "Evaluation exception: $_"
        $AllPassed = $false
    } finally {
        Pop-Location -ErrorAction SilentlyContinue
    }

    # Stop server
    Write-Host "[SERVER] Stopping llama-server..."
    Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Host "[SERVER] Stopped."
    Write-Host ""
}

Write-Host "============================================================"
if ($AllPassed) {
    Write-Host "[RESULT] All smoke tests PASSED"
} else {
    Write-Host "[RESULT] Some smoke tests FAILED - check output above"
}
Write-Host "============================================================"
