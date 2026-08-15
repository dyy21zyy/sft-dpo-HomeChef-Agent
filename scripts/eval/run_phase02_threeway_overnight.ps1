# ============================================================
# HomeChef Phase 02 Three-way Overnight Benchmark
#
# A. Qwen3-1.7B-Base Q8_0
#    + JSON Schema constrained decoding
#
# B. Qwen3-4B-Instruct-2507 Q4_K_M
#    + unconstrained generation
#
# C. Qwen3-4B-Instruct-2507 Q4_K_M
#    + JSON Schema constrained decoding
#
# Frozen Test: 120 cases
# ============================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest


# ============================================================
# Project
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path "$ScriptDir\..\..").Path

Set-Location $ProjectRoot

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

$LogDir = Join-Path `
    $ProjectRoot `
    "reports\generated\phase02\overnight_logs\$Timestamp"

$TempConfigDir = Join-Path `
    $ProjectRoot `
    "reports\generated\phase02\overnight_configs\$Timestamp"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $LogDir |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path $TempConfigDir |
    Out-Null


# ============================================================
# Helper: find first existing file
# ============================================================

function Resolve-FirstExisting {

    param(
        [string[]]$Candidates,
        [string]$Description
    )

    foreach ($Candidate in $Candidates) {

        if (Test-Path $Candidate) {

            return (Resolve-Path $Candidate).Path
        }
    }

    throw "Cannot find ${Description}. Checked: $($Candidates -join ', ')"
}


# ============================================================
# llama.cpp + GGUF
# Prefer project-local runtime; fall back to H:
# ============================================================

$LlamaServer = Resolve-FirstExisting `
    -Description "llama-server.exe" `
    -Candidates @(
        "$ProjectRoot\deployment\llama_cpp\bin\llama-server.exe",
        "H:\deployment\llama_cpp\bin\llama-server.exe"
    )

$Model17B = Resolve-FirstExisting `
    -Description "Qwen3-1.7B-Base-Q8_0.gguf" `
    -Candidates @(
        "$ProjectRoot\models\gguf\Qwen3-1.7B-Base-Q8_0.gguf",
        "H:\models\gguf\Qwen3-1.7B-Base-Q8_0.gguf"
    )

$Model4B = Resolve-FirstExisting `
    -Description "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" `
    -Candidates @(
        "$ProjectRoot\models\gguf\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "H:\models\gguf\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    )


# ============================================================
# Project inputs
# ============================================================

$CasesPath = "data/eval/frozen_test.jsonl"
$ManifestPath = "data/eval/frozen_test.manifest.json"

$Inference17BStructured =
    "configs/inference/llama_cpp_1_7b_cpu_structured.yaml"

$Inference4BUnstructured =
    "configs/inference/llama_cpp_4b_instruct_2507_cpu.yaml"

$Inference4BStructured =
    "configs/inference/llama_cpp_4b_instruct_2507_cpu_structured.yaml"


# ============================================================
# Output directories
# ============================================================

$Result17BStructured =
    "reports/generated/phase02/structured/1_7b/frozen_test"

$Result4BUnstructured =
    "reports/generated/phase02/unstructured/4b_instruct_2507/frozen_test"

$Result4BStructured =
    "reports/generated/phase02/structured/4b_instruct_2507/frozen_test"


# ============================================================
# PREFLIGHT
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "PHASE 02 THREE-WAY BENCHMARK - PREFLIGHT" -ForegroundColor Cyan
Write-Host "============================================================"

Write-Host "[OK] Project:"
Write-Host "     $ProjectRoot"

Write-Host "[OK] llama-server:"
Write-Host "     $LlamaServer"

Write-Host "[OK] 1.7B GGUF:"
Write-Host "     $Model17B"

Write-Host "[OK] 4B Instruct GGUF:"
Write-Host "     $Model4B"

$RequiredProjectFiles = @(
    $CasesPath,
    $ManifestPath,
    $Inference17BStructured,
    $Inference4BUnstructured,
    $Inference4BStructured
)

foreach ($RelativePath in $RequiredProjectFiles) {

    $FullPath = Join-Path $ProjectRoot $RelativePath

    if (-not (Test-Path $FullPath)) {

        throw "Missing required project file: $FullPath"
    }

    Write-Host "[OK] $RelativePath"
}


# ============================================================
# Verify Frozen Test = 120
# ============================================================

$FrozenCount = (
    Get-Content (Join-Path $ProjectRoot $CasesPath) |
    Where-Object {
        -not [string]::IsNullOrWhiteSpace($_)
    }
).Count

Write-Host ""
Write-Host "[DATA] Frozen Test cases: $FrozenCount"

if ($FrozenCount -ne 120) {

    throw "Frozen Test must contain exactly 120 non-empty lines. Found $FrozenCount."
}


# ============================================================
# Verify uv + real CLI
# ============================================================

$UvCommand = Get-Command uv -ErrorAction Stop

Write-Host ""
Write-Host "[CLI] Checking homechef-eval..."

$CliCheck = Start-Process `
    -FilePath $UvCommand.Source `
    -ArgumentList @(
        "run",
        "homechef-eval",
        "--help"
    ) `
    -WorkingDirectory $ProjectRoot `
    -Wait `
    -PassThru `
    -NoNewWindow

if ($CliCheck.ExitCode -ne 0) {

    throw "homechef-eval --help failed."
}

Write-Host "[OK] homechef-eval available."
Write-Host "[OK] Expected invocation: homechef-eval --config <evaluation.yaml>"


# ============================================================
# Generate three evaluation YAML configs
# ============================================================

$Eval17BStructured =
    Join-Path `
        $TempConfigDir `
        "phase02_1_7b_structured_frozen.yaml"

$Eval4BUnstructured =
    Join-Path `
        $TempConfigDir `
        "phase02_4b_instruct_unstructured_frozen.yaml"

$Eval4BStructured =
    Join-Path `
        $TempConfigDir `
        "phase02_4b_instruct_structured_frozen.yaml"


# ------------------------------------------------------------
# A. 1.7B Structured
# ------------------------------------------------------------

@"
run_id: phase02_1_7b_structured_frozen
cases_path: data/eval/frozen_test.jsonl
backend_config_path: configs/inference/llama_cpp_1_7b_cpu_structured.yaml
manifest_path: data/eval/frozen_test.manifest.json
output_dir: $Result17BStructured
model_id: Qwen/Qwen3-1.7B-Base
suite_id: phase02_frozen_test_v1
device: cpu
runtime: llama.cpp
model_format: gguf
quantization: Q8_0
gpu_layers: 0
notes: "Phase 02 Frozen Test: Qwen3-1.7B-Base Q8_0 with JSON Schema constrained decoding"
tags: ["structured", "1.7b", "frozen", "llama_cpp", "cpu"]
"@ |
Set-Content `
    -Path $Eval17BStructured `
    -Encoding UTF8


# ------------------------------------------------------------
# B. 4B Instruct Unstructured
# ------------------------------------------------------------

@"
run_id: phase02_4b_instruct_unstructured_frozen
cases_path: data/eval/frozen_test.jsonl
backend_config_path: configs/inference/llama_cpp_4b_instruct_2507_cpu.yaml
manifest_path: data/eval/frozen_test.manifest.json
output_dir: $Result4BUnstructured
model_id: Qwen/Qwen3-4B-Instruct-2507
suite_id: phase02_frozen_test_v1
device: cpu
runtime: llama.cpp
model_format: gguf
quantization: Q4_K_M
gpu_layers: 0
notes: "Phase 02 Frozen Test: Qwen3-4B-Instruct-2507 Q4_K_M without constrained decoding"
tags: ["unstructured", "4b", "instruct_2507", "frozen", "llama_cpp", "cpu"]
"@ |
Set-Content `
    -Path $Eval4BUnstructured `
    -Encoding UTF8


# ------------------------------------------------------------
# C. 4B Instruct Structured
# ------------------------------------------------------------

@"
run_id: phase02_4b_instruct_structured_frozen
cases_path: data/eval/frozen_test.jsonl
backend_config_path: configs/inference/llama_cpp_4b_instruct_2507_cpu_structured.yaml
manifest_path: data/eval/frozen_test.manifest.json
output_dir: $Result4BStructured
model_id: Qwen/Qwen3-4B-Instruct-2507
suite_id: phase02_frozen_test_v1
device: cpu
runtime: llama.cpp
model_format: gguf
quantization: Q4_K_M
gpu_layers: 0
notes: "Phase 02 Frozen Test: Qwen3-4B-Instruct-2507 Q4_K_M with JSON Schema constrained decoding"
tags: ["structured", "4b", "instruct_2507", "frozen", "llama_cpp", "cpu"]
"@ |
Set-Content `
    -Path $Eval4BStructured `
    -Encoding UTF8


Write-Host ""
Write-Host "[CONFIG] Generated:"
Write-Host " A: $Eval17BStructured"
Write-Host " B: $Eval4BUnstructured"
Write-Host " C: $Eval4BStructured"


# ============================================================
# CPU
# ============================================================

$CpuThreads = (
    Get-CimInstance Win32_ComputerSystem
).NumberOfLogicalProcessors

if (-not $CpuThreads) {

    $CpuThreads = 4
}

Write-Host ""
Write-Host "[CPU] Logical processors: $CpuThreads"


# ============================================================
# Stop llama-server
# ============================================================

function Stop-LlamaServer {

    $Processes = @(
        Get-Process `
            "llama-server" `
            -ErrorAction SilentlyContinue
    )

    foreach ($Process in $Processes) {

        Write-Host "[SERVER] Stopping PID $($Process.Id)"

        Stop-Process `
            -Id $Process.Id `
            -Force `
            -ErrorAction SilentlyContinue
    }

    Start-Sleep -Seconds 3
}


# ============================================================
# Start llama-server
# ============================================================

function Start-LlamaServer {

    param(
        [string]$ModelPath,
        [string]$Alias,
        [string]$LogPrefix
    )

    Stop-LlamaServer

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[SERVER] STARTING $Alias" -ForegroundColor Cyan
    Write-Host "============================================================"

    Write-Host "[MODEL] $ModelPath"

    $ServerStdout =
        Join-Path `
            $LogDir `
            "$LogPrefix.server.stdout.log"

    $ServerStderr =
        Join-Path `
            $LogDir `
            "$LogPrefix.server.stderr.log"

    $Process = Start-Process `
        -FilePath $LlamaServer `
        -ArgumentList @(
            "-m",
            $ModelPath,
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--ctx-size",
            "2048",
            "--threads",
            "$CpuThreads",
            "-ngl",
            "0",
            "--alias",
            $Alias
        ) `
        -PassThru `
        -RedirectStandardOutput $ServerStdout `
        -RedirectStandardError $ServerStderr

    Write-Host "[SERVER] PID: $($Process.Id)"
    Write-Host "[SERVER] Waiting for health..."

    $Healthy = $false

    for ($i = 1; $i -le 120; $i++) {

        if ($Process.HasExited) {

            Write-Host ""
            Write-Host "----- SERVER STDERR -----" -ForegroundColor Red

            if (Test-Path $ServerStderr) {

                Get-Content $ServerStderr |
                    Select-Object -Last 100
            }

            throw "llama-server exited before becoming healthy."
        }

        try {

            $Response = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8080/v1/models" `
                -TimeoutSec 3 `
                -ErrorAction Stop

            $LoadedIds = @(
                $Response.data |
                ForEach-Object {
                    $_.id
                }
            )

            if ($LoadedIds -contains $Alias) {

                $Healthy = $true

                Write-Host "[SERVER] HEALTHY: $Alias" -ForegroundColor Green

                break
            }
        }
        catch {
        }

        Start-Sleep -Seconds 1
    }

    if (-not $Healthy) {

        Stop-Process `
            -Id $Process.Id `
            -Force `
            -ErrorAction SilentlyContinue

        throw "llama-server failed health check for $Alias."
    }

    return $Process
}


# ============================================================
# Check server before evaluation
# ============================================================

function Assert-ServerModel {

    param(
        [string]$ExpectedAlias
    )

    try {

        $Response = Invoke-RestMethod `
            -Uri "http://127.0.0.1:8080/v1/models" `
            -TimeoutSec 5 `
            -ErrorAction Stop

        $Ids = @(
            $Response.data |
            ForEach-Object {
                $_.id
            }
        )

        if ($Ids -notcontains $ExpectedAlias) {

            throw "Expected $ExpectedAlias, found: $($Ids -join ', ')"
        }
    }
    catch {

        throw "llama-server health check failed: $_"
    }

    Write-Host "[SERVER] Confirmed model: $ExpectedAlias"
}


# ============================================================
# Check output artifacts
# ============================================================

function Assert-EvalArtifacts {

    param(
        [string]$ReportDir,
        [string]$RunName
    )

    $FullReportDir =
        Join-Path `
            $ProjectRoot `
            $ReportDir

    if (-not (Test-Path $FullReportDir)) {

        throw "${RunName}: result directory not created: $FullReportDir"
    }

    $Files = @(
        Get-ChildItem `
            $FullReportDir `
            -Recurse `
            -File
    )

    $Scorecards = @(
        $Files |
        Where-Object {
            $_.Name -match "scorecard"
        }
    )

    $CaseResults = @(
        $Files |
        Where-Object {
            $_.Name -match "case.*result"
        }
    )

    if ($Scorecards.Count -eq 0) {

        throw "${RunName}: no scorecard file generated."
    }

    if ($CaseResults.Count -eq 0) {

        throw "${RunName}: no case-results file generated."
    }

    Write-Host ""
    Write-Host "[ARTIFACT] Scorecard:"
    $Scorecards |
        ForEach-Object {
            Write-Host "  $($_.FullName)"
        }

    Write-Host "[ARTIFACT] Case results:"
    $CaseResults |
        ForEach-Object {
            Write-Host "  $($_.FullName)"
        }
}


# ============================================================
# Run one 120-case evaluation
#
# IMPORTANT:
# Real CLI is:
#
# uv run homechef-eval --config <evaluation.yaml>
# ============================================================

function Run-Evaluation {

    param(
        [string]$RunName,
        [string]$EvalConfig,
        [string]$ReportDir,
        [string]$LogPrefix,
        [string]$ExpectedAlias
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[EVAL] $RunName" -ForegroundColor Cyan
    Write-Host "============================================================"

    Assert-ServerModel `
        -ExpectedAlias $ExpectedAlias

    Write-Host "[CONFIG] $EvalConfig"
    Write-Host "[OUTPUT] $ReportDir"
    Write-Host "[CASES] 120"

    $EvalStdout =
        Join-Path `
            $LogDir `
            "$LogPrefix.eval.stdout.log"

    $EvalStderr =
        Join-Path `
            $LogDir `
            "$LogPrefix.eval.stderr.log"

    $Started = Get-Date

    $Process = Start-Process `
        -FilePath $UvCommand.Source `
        -ArgumentList @(
            "run",
            "homechef-eval",
            "--config",
            $EvalConfig
        ) `
        -WorkingDirectory $ProjectRoot `
        -Wait `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $EvalStdout `
        -RedirectStandardError $EvalStderr

    $Elapsed = (Get-Date) - $Started

    Write-Host ""
    Write-Host "[EVAL] Exit code: $($Process.ExitCode)"
    Write-Host "[EVAL] Elapsed: $($Elapsed.TotalMinutes.ToString('F1')) minutes"

    if (Test-Path $EvalStdout) {

        Write-Host ""
        Write-Host "----- EVAL STDOUT (last 100 lines) -----"

        Get-Content $EvalStdout |
            Select-Object -Last 100
    }

    if ($Process.ExitCode -ne 0) {

        Write-Host ""
        Write-Host "----- EVAL STDERR (last 150 lines) -----" -ForegroundColor Red

        if (Test-Path $EvalStderr) {

            Get-Content $EvalStderr |
                Select-Object -Last 150
        }

        throw "${RunName}: evaluation failed with exit code $($Process.ExitCode)."
    }

    Assert-EvalArtifacts `
        -ReportDir $ReportDir `
        -RunName $RunName

    Write-Host ""
    Write-Host "[PASS] $RunName" -ForegroundColor Green
}


# ============================================================
# Clean ONLY the three new target result directories
#
# Does NOT touch old baseline results.
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "CLEAN TARGET DIRECTORIES"
Write-Host "============================================================"

foreach ($RelativeDir in @(
    $Result17BStructured,
    $Result4BUnstructured,
    $Result4BStructured
)) {

    $FullDir =
        Join-Path `
            $ProjectRoot `
            $RelativeDir

    if (Test-Path $FullDir) {

        Write-Host "[CLEAN] $FullDir"

        Remove-Item `
            $FullDir `
            -Recurse `
            -Force
    }
}


# ============================================================
# Execute
# ============================================================

$OverallStarted = Get-Date

try {

    # ========================================================
    # RUN A
    # 1.7B Base + Structured
    # ========================================================

    $Server17 = Start-LlamaServer `
        -ModelPath $Model17B `
        -Alias "Qwen/Qwen3-1.7B-Base" `
        -LogPrefix "01_1.7b_structured"

    Run-Evaluation `
        -RunName "A - 1.7B Base Q8_0 + Structured" `
        -EvalConfig $Eval17BStructured `
        -ReportDir $Result17BStructured `
        -LogPrefix "01_1.7b_structured" `
        -ExpectedAlias "Qwen/Qwen3-1.7B-Base"

    Stop-LlamaServer


    # ========================================================
    # Start 4B Instruct
    # ========================================================

    $Server4 = Start-LlamaServer `
        -ModelPath $Model4B `
        -Alias "Qwen/Qwen3-4B-Instruct-2507" `
        -LogPrefix "02_4b_instruct"


    # ========================================================
    # RUN B
    # 4B Instruct + Unstructured
    # ========================================================

    Run-Evaluation `
        -RunName "B - 4B Instruct-2507 Q4_K_M + Unstructured" `
        -EvalConfig $Eval4BUnstructured `
        -ReportDir $Result4BUnstructured `
        -LogPrefix "02_4b_unstructured" `
        -ExpectedAlias "Qwen/Qwen3-4B-Instruct-2507"


    # ========================================================
    # Make sure 4B is still healthy.
    #
    # If server died after Run B, restart same 4B before Run C.
    # ========================================================

    $NeedRestart4B = $false

    try {

        Assert-ServerModel `
            -ExpectedAlias "Qwen/Qwen3-4B-Instruct-2507"
    }
    catch {

        $NeedRestart4B = $true
    }

    if ($NeedRestart4B) {

        Write-Host ""
        Write-Host "[SERVER] Restarting 4B before Structured run..." -ForegroundColor Yellow

        $Server4 = Start-LlamaServer `
            -ModelPath $Model4B `
            -Alias "Qwen/Qwen3-4B-Instruct-2507" `
            -LogPrefix "03_4b_instruct_restart"
    }


    # ========================================================
    # RUN C
    # 4B Instruct + Structured
    # ========================================================

    Run-Evaluation `
        -RunName "C - 4B Instruct-2507 Q4_K_M + Structured" `
        -EvalConfig $Eval4BStructured `
        -ReportDir $Result4BStructured `
        -LogPrefix "03_4b_structured" `
        -ExpectedAlias "Qwen/Qwen3-4B-Instruct-2507"


    # ========================================================
    # SUCCESS
    # ========================================================

    $OverallElapsed = (Get-Date) - $OverallStarted

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "ALL 3 FROZEN TESTS PASSED EXECUTION" -ForegroundColor Green
    Write-Host "============================================================"

    Write-Host ""
    Write-Host "Total elapsed:"
    Write-Host "  $($OverallElapsed.TotalHours.ToString('F2')) hours"

    Write-Host ""
    Write-Host "A. 1.7B Structured"
    Write-Host "  $Result17BStructured"

    Write-Host ""
    Write-Host "B. 4B Instruct Unstructured"
    Write-Host "  $Result4BUnstructured"

    Write-Host ""
    Write-Host "C. 4B Instruct Structured"
    Write-Host "  $Result4BStructured"

    Write-Host ""
    Write-Host "Logs:"
    Write-Host "  $LogDir"

    Write-Host ""
    Write-Host "Evaluation configs:"
    Write-Host "  $TempConfigDir"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "OVERNIGHT PHASE 02 COMPLETE"
    Write-Host "============================================================"
}
finally {

    Stop-LlamaServer
}
