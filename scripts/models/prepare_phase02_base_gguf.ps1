# Phase 02: Prepare Qwen3 Base GGUF models for llama.cpp CPU benchmark
#
# Downloads exact HF revisions, converts to GGUF, quantizes.
# Revisions are pinned in configs/models/phase02_base_models.yaml.
#
# Usage:
#   .\scripts\models\prepare_phase02_base_gguf.ps1           # skip if GGUF already exists
#   .\scripts\models\prepare_phase02_base_gguf.ps1 -Force    # force rebuild
#
# Prerequisites:
#   - Python 3.12+ with transformers, torch, huggingface_hub installed
#   - llama.cpp b10357 in deployment\llama_cpp\bin\
#   - llama.cpp source with convert_hf_to_gguf.py at deployment\llama_cpp\llama_cpp_src\
#   - Optional: set $env:HOMECHEF_PYTHON to an explicit Python executable
#
# Output:
#   models/gguf/Qwen3-0.6B-Base-Q8_0.gguf
#   models/gguf/Qwen3-1.7B-Base-Q8_0.gguf
#   models/gguf/Qwen3-4B-Base-Q4_K_M.gguf

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."
$ModelsDir = "$ProjectRoot\models\gguf"
$LlamaCppDir = "$ProjectRoot\deployment\llama_cpp"

# Ensure output directory exists
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# Model definitions with pinned HF revisions from configs/models/phase02_base_models.yaml
$Models = @(
    @{
        hf = "Qwen/Qwen3-0.6B-Base"
        rev = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
        file = "Qwen3-0.6B-Base-Q8_0.gguf"
        quant = "Q8_0"
    },
    @{
        hf = "Qwen/Qwen3-1.7B-Base"
        rev = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
        file = "Qwen3-1.7B-Base-Q8_0.gguf"
        quant = "Q8_0"
    },
    @{
        hf = "Qwen/Qwen3-4B-Base"
        rev = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
        file = "Qwen3-4B-Base-Q4_K_M.gguf"
        quant = "Q4_K_M"
    }
)

# Check llama.cpp tools
$ConvertScript = "$LlamaCppDir\llama_cpp_src\convert_hf_to_gguf.py"
$QuantizeExe = "$LlamaCppDir\bin\llama-quantize.exe"
$GgufPyPath = "$LlamaCppDir\llama_cpp_src\gguf-py"
$LlamaCppSrc = "$LlamaCppDir\llama_cpp_src"
$PythonExe = if ($env:HOMECHEF_PYTHON) { $env:HOMECHEF_PYTHON } else { "python" }

if (-not (Test-Path $QuantizeExe)) {
    Write-Error "llama-quantize.exe not found at $QuantizeExe."
    exit 1
}
if (-not (Test-Path $ConvertScript)) {
    Write-Error "convert_hf_to_gguf.py not found at $ConvertScript. Clone llama.cpp source first."
    exit 1
}

foreach ($Model in $Models) {
    $OutputPath = "$ModelsDir\$($Model.file)"
    $TempFp16 = "$ModelsDir\$($Model.hf -replace '/', '_')-fp16.gguf"

    if ((Test-Path $OutputPath) -and (-not $Force)) {
        $size = (Get-Item $OutputPath).Length
        Write-Host "[SKIP] $($Model.file) already exists ($([math]::Round($size/1MB, 1)) MB). Use -Force to rebuild."
        continue
    }

    Write-Host "============================================================"
    Write-Host "Preparing: $($Model.hf) @ $($Model.rev) -> $($Model.quant)"
    Write-Host "============================================================"

    # Step 1: Download HF model at exact revision, then convert to FP16 GGUF
    if ((Test-Path $TempFp16) -and (-not $Force)) {
        Write-Host "[SKIP] FP16 GGUF already exists: $TempFp16"
    } else {
        Write-Host "[DOWNLOAD] $($Model.hf) @ $($Model.rev)"
$env:PYTHONPATH = "$GgufPyPath;$LlamaCppSrc"

# Pass values through environment variables instead of interpolating them
# into `python -c`. This avoids Windows PowerShell native-argument quoting issues.
$env:HOMECHEF_HF_MODEL_ID = $Model.hf
$env:HOMECHEF_HF_REVISION = $Model.rev
$env:HOMECHEF_HF_LOCAL_DIR = "models/hf_cache/$($Model.hf -replace '/', '_')"

@'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["HOMECHEF_HF_MODEL_ID"]
revision = os.environ["HOMECHEF_HF_REVISION"]
local_dir = os.environ["HOMECHEF_HF_LOCAL_DIR"]

print(f"Downloading {model_id} @ {revision} to {local_dir}...")

snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=local_dir,
)

print("Download complete.")
'@ | & $PythonExe -
        if ($LASTEXITCODE -ne 0) {
            Write-Error "HF download failed for $($Model.hf) @ $($Model.rev)"
            exit 1
        }

        Write-Host "[CONVERT] models/hf_cache/$($Model.hf -replace '/', '_') -> FP16 GGUF"
        $env:PYTHONPATH = "$GgufPyPath;$LlamaCppSrc"
        & $PythonExe "$ConvertScript" "models/hf_cache/$($Model.hf -replace '/', '_')" --outfile "$TempFp16" --outtype f16
        if ($LASTEXITCODE -ne 0) {
            Write-Error "convert_hf_to_gguf.py failed for $($Model.hf)"
            exit 1
        }
    }

    # Step 2: Quantize
    if (Test-Path $OutputPath) {
        Write-Host "[SKIP] Quantized GGUF already exists: $OutputPath"
    } else {
        Write-Host "[QUANTIZE] $TempFp16 -> $($Model.quant)"
        & $QuantizeExe "$TempFp16" "$OutputPath" $Model.quant
        if ($LASTEXITCODE -ne 0) {
            Write-Error "llama-quantize failed for $TempFp16"
            exit 1
        }
    }

    $size = (Get-Item $OutputPath).Length
    Write-Host "[DONE] $($Model.file) ($([math]::Round($size/1MB, 1)) MB)"
    Write-Host ""
}

Write-Host "============================================================"
Write-Host "All Phase 02 Base GGUF models prepared."
Write-Host "Output: $ModelsDir"
Get-ChildItem $ModelsDir -Filter "*.gguf" | ForEach-Object {
    Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB, 1)) MB"
}
