# Phase 02 M0 Frozen Benchmark - llama.cpp CPU
#
# Runs 2 models (1.7B, 4B) against Frozen Test (120 cases each).
# Manages llama-server.exe lifecycle per model.
#
# Usage:
#   .\scripts\eval\run_phase02_frozen_cpu.ps1
#
# Prerequisites:
#   - GGUF models prepared: .\scripts\models\prepare_phase02_base_gguf.ps1
#   - llama-server.exe in deployment/llama_cpp/bin/
#   - Python venv with homechef-eval

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

# Verify llama-server exists
if (-not (Test-Path $LlamaServer)) {
    Write-Error "llama-server.exe not found at $LlamaServer. Build/download llama.cpp first."
    exit 1
}

# Get llama.cpp version
$OldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"

$VersionOutput = & $LlamaServer --version 2>&1

$ErrorActionPreference = $OldErrorActionPreference

Write-Host "llama.cpp version: $($VersionOutput | Out-String)"

# Hardware info
$CpuInfo = Get-CimInstance Win32_Processor | Select-Object -First 1
$OsInfo = Get-CimInstance Win32_OperatingSystem
$CpuThreads = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors

if (-not $CpuThreads) {
    $CpuThreads = 4
}

Write-Host "============================================================"
Write-Host "Phase 02 M0 Frozen Benchmark - llama.cpp CPU"
Write-Host "============================================================"
Write-Host "OS: $($OsInfo.Caption)"
Write-Host "CPU: $($CpuInfo.Name)"
Write-Host "Logical cores: $CpuThreads"
Write-Host "RAM: $([math]::Round($OsInfo.TotalVisibleMemorySize / 1MB, 1)) GB"
Write-Host "Host: ${HostAddr}:${Port}"
Write-Host ""

# Model definitions: Frozen Test only, 2 models
$Models = @(
    @{
        name = "1_7b"
        label = "1.7B"
        model_id = "Qwen/Qwen3-1.7B-Base"
        gguf = "$ModelsDir\Qwen3-1.7B-Base-Q8_0.gguf"
        config = "configs/evaluation/phase02_base_1_7b_frozen.yaml"
        timeout = 600
    },
    @{
        name = "4b"
        label = "4B"
        model_id = "Qwen/Qwen3-4B-Base"
        gguf = "$ModelsDir\Qwen3-4B-Base-Q4_K_M.gguf"
        config = "configs/evaluation/phase02_base_4b_frozen.yaml"
        timeout = 900
    }
)

$AllResults = @()
$OverallStart = Get-Date

foreach ($Model in $Models) {
    Write-Host "============================================================"
    Write-Host "[RUN] $($Model.label) - $($Model.model_id)"
    Write-Host "============================================================"

    # Check GGUF
    if (-not (Test-Path $Model.gguf)) {
        Write-Error "GGUF not found: $($Model.gguf). Run prepare_phase02_base_gguf.ps1 first."
        continue
    }

    $ggufSize = [math]::Round((Get-Item $Model.gguf).Length / 1MB, 1)
    Write-Host "[GGUF] $($Model.gguf) ($ggufSize MB)"

    # Prepare server logs
    $ServerLogDir = "$ProjectRoot\reports\generated\phase02\base_$($Model.name)\frozen_test"

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $ServerLogDir |
        Out-Null

    $ServerStdoutLog = "$ServerLogDir\server.stdout.log"
    $ServerStderrLog = "$ServerLogDir\server.stderr.log"

    # Start llama-server
    Write-Host "[SERVER] Starting llama-server.exe..."

    $ServerProcess = Start-Process `
        -FilePath $LlamaServer `
        -ArgumentList @(
            "-m", $Model.gguf,
            "--host", $HostAddr,
            "--port", $Port,
            "--ctx-size", $CtxSize,
            "--threads", $CpuThreads,
            "-ngl", "0"
        ) `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $ServerStdoutLog `
        -RedirectStandardError $ServerStderrLog

    # Wait for health
    Write-Host "[SERVER] Waiting for health check..."

    $Healthy = $false
    $MaxWait = 120

    for ($i = 0; $i -lt $MaxWait; $i++) {
        try {
            $resp = Invoke-WebRequest `
                -Uri "http://${HostAddr}:${Port}/v1/models" `
                -UseBasicParsing `
                -TimeoutSec 3 `
                -ErrorAction SilentlyContinue

            if ($resp.StatusCode -eq 200) {
                $Healthy = $true
                Write-Host "[SERVER] Healthy after $i seconds"
                break
            }
        }
        catch {
        }

        Start-Sleep -Seconds 1
    }

    if (-not $Healthy) {
        Write-Error "llama-server failed to become healthy within ${MaxWait}s"

        Stop-Process `
            -Id $ServerProcess.Id `
            -Force `
            -ErrorAction SilentlyContinue

        continue
    }

    # Run evaluation
    Write-Host "[EVAL] Running Frozen Test (120 cases)..."

    $RunStart = Get-Date

    try {
        Push-Location $ProjectRoot

        $EvalOutput = uv run homechef-eval `
            --config $Model.config 2>&1

        $EvalExit = $LASTEXITCODE

        Pop-Location

        $RunElapsed = (Get-Date) - $RunStart

        Write-Host "[EVAL] Completed in $($RunElapsed.TotalSeconds.ToString('F1'))s"

        if ($EvalExit -ne 0) {
            Write-Error "Evaluation failed with exit code $EvalExit"
            Write-Host ($EvalOutput | Out-String)
        }
        else {
            Write-Host ($EvalOutput | Out-String)

            $AllResults += @{
                model = $Model.label
                model_id = $Model.model_id
                status = "PASS"
                elapsed = $RunElapsed.TotalSeconds
            }
        }
    }
    catch {
        Write-Error "Evaluation exception: $_"

        Pop-Location -ErrorAction SilentlyContinue
    }

    # Stop server
    Write-Host "[SERVER] Stopping llama-server..."

    Stop-Process `
        -Id $ServerProcess.Id `
        -Force `
        -ErrorAction SilentlyContinue

    Start-Sleep -Seconds 3

    Write-Host "[SERVER] Stopped."
    Write-Host ""
}

$OverallElapsed = (Get-Date) - $OverallStart

# Generate combined summary
Write-Host "============================================================"
Write-Host "Generating combined summary..."
Write-Host "============================================================"

Push-Location $ProjectRoot

$SummaryResult = uv run python scripts/eval/run_base_benchmark.py `
    --config configs/evaluation/phase02_base_1_7b_frozen.yaml `
    --config configs/evaluation/phase02_base_4b_frozen.yaml 2>&1

$SummaryExit = $LASTEXITCODE

Pop-Location

Write-Host ($SummaryResult | Out-String)

Write-Host "============================================================"
Write-Host "Phase 02 M0 Frozen Benchmark Complete"
Write-Host "============================================================"
Write-Host "Total elapsed: $($OverallElapsed.TotalMinutes.ToString('F1')) minutes"
Write-Host ""
Write-Host "Results: reports/generated/phase02/"
Write-Host "  base_benchmark_summary.json"
Write-Host "  base_benchmark_summary.md"
Write-Host ""

$ResultFiles = Get-ChildItem "$ProjectRoot\reports\generated\phase02" -Recurse -File

foreach ($ResultFile in $ResultFiles) {
    Write-Host "  $($ResultFile.FullName.Replace($ProjectRoot, ''))"
}
