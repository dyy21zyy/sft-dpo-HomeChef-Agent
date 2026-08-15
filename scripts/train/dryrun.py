"""Phase 04 Task 3 Amendment 3 — Training dry-run script (CPU engineering dry-run).

Refuses to run without --approval-file (Checkpoint B gate).
After approval, executes a real LLaMA-Factory CPU engineering dry-run:
- SFT: 1 row, max_steps=2, CPU
- DPO: 1 pair, max_steps=2, pref_beta=0.1, pref_loss=sigmoid, CPU

This is an ENGINEERING smoke test only — it does NOT claim training
effectiveness, DPO improvement, or precision gains.

The local representative model is Qwen/Qwen3-1.7B-Base (offline HF weights).
4B is validated by static config/template/contract tests only; real 4B training
is left to AutoDL.

Dry-run supports a local model path override (--local-model-path) so the formal
canonical model id (Qwen/Qwen3-1.7B-Base) is preserved in provenance while the
rendered LLaMA-Factory config uses the resolved local HF path. If the local path
is missing, it is a HARD FAIL (no HF download, no 0.6B fallback).

Temp datasets go under experiments/phase04/dryrun/.
DPO pairs are converted to LLaMA-Factory 0.9.5 standard format.

Usage (after Checkpoint B approval) — formal Phase04 models are 1.7B and 4B:
  python scripts/train/dryrun.py --stage sft --config configs/training/phase04_sft_qwen3_1_7b.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json --local-model-path models/hf_cache/Qwen_Qwen3-1.7B-Base
  python scripts/train/dryrun.py --stage dpo --config configs/training/phase04_dpo_qwen3_1_7b_beta_0_1.yaml --sample-count 1 --max-steps 2 --approval-file project-log/phase04_dryrun_approval.json --local-model-path models/hf_cache/Qwen_Qwen3-1.7B-Base
  (4B uses configs/training/phase04_sft_qwen3_4b.yaml / phase04_dpo_qwen3_4b_beta_0_1.yaml)

The output_dir is derived from the config's model size (e.g. experiments/phase04/dryrun/sft_1_7b).
0.6B is historical / non-formal only and is NOT used as a formal dry-run example.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from homechef_booking.training.config import load_training_run_spec, validate_training_run_spec
from homechef_booking.training.dataset_adapter import (
    convert_sft_row_to_llamafactory_format,
    validate_dpo_for_training,
    validate_sft_for_training,
)
from homechef_booking.training.renderer import (
    LLAMAFACTORY_ALLOWED_ARGS,
    assert_rendered_config,
)

# ── Training renderer contract (fail-closed) ─────────────────────────────────
# do_train is EXECUTION CONTROL, not an experiment hyperparameter. A rendered
# LLaMA-Factory training config must set do_train=true or LLaMA-Factory's
# "if training_args.do_train" branch is skipped -> silent no-op (exit 0,
# global_step=0, no adapter). This is the confirmed ROOT CAUSE of the earlier
# CPU dry-run no-op. We enforce it before launching llamafactory-cli.


# LLaMA-Factory native allowlist and fail-closed contract are shared with the
# formal renderer (imported from homechef_booking.training.renderer):
#   LLAMAFACTORY_ALLOWED_ARGS / Phase04TrainingConfigError / assert_rendered_config

# Windows consoles default to a legacy codec (e.g. cp1252) that cannot encode
# non-ASCII characters such as the Chinese workspace path printed in command
# lines. Reconfigure stdout/stderr to UTF-8 so dry-run output is safe anywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ── Constants ───────────────────────────────────────────────────────────────

DRYRUN_BASE = Path("experiments/phase04/dryrun")
MANIFEST_PATH = Path("project-log/phase04_dryrun_manifest.json")

# Temp datasets go under data/ because LLaMA-Factory resolves file_name relative to
# dataset_dir=data/ and cannot handle non-ASCII paths. They are clearly named
# phase04_dryrun_* to distinguish from permanent datasets.
SFT_TEMP_DATASET = Path("data") / "phase04_dryrun_sft_1row.jsonl"
SFT_TEMP_CONFIG = DRYRUN_BASE / "phase04_dryrun_sft_1row.yaml"

DPO_TEMP_DATASET = Path("data") / "phase04_dryrun_dpo_1pair.jsonl"


def _dpo_temp_config(beta: float) -> Path:
    beta_tag = f"{beta:g}".replace(".", "_")
    return DRYRUN_BASE / f"phase04_dryrun_dpo_1pair_beta_{beta_tag}.yaml"

# ── Model key derivation (no hardcoded 0_6b) ─────────────────────────────────
#
# The dry-run output_dir is derived from the training config's model size/key,
# NOT hardcoded to a specific size. Formal Phase04 models are 1.7B and 4B;
# 0.6B is historical/non-formal only. model identity is independent of the
# sft/dpo experiment stage (no model_size -> training_stage fallback).


def _derive_model_key(model_name_or_path: str) -> str:
    """Extract the model-size key from the Phase02 formal model id.

    e.g. "Qwen/Qwen3-1.7B-Base"        -> "1_7b"
         "Qwen/Qwen3-4B-Instruct-2507" -> "4b"
         "Qwen/Qwen3-0.6B-Base"        -> "0_6b" (historical only)

    The 4B formal variant is Instruct-2507 (NOT 4B-Base), so we key on the
    distinctive "instruct-2507" marker rather than a bare "4b" substring.
    """
    lowered = (model_name_or_path or "").lower()
    if "1.7b" in lowered:
        return "1_7b"
    if "instruct-2507" in lowered and "4b" in lowered:
        return "4b"
    if "0.6b" in lowered:
        return "0_6b"
    raise ValueError(
        f"Cannot derive model key from model_name_or_path: {model_name_or_path!r}"
    )


def _dryrun_output_dir(stage: str, model_key: str, pref_beta: float | None = None) -> Path:
    """Compute the per-model dry-run output dir.

    e.g. ("sft", "1_7b")               -> experiments/phase04/dryrun/sft_1_7b
         ("dpo", "1_7b", 0.1)          -> experiments/phase04/dryrun/dpo_1_7b_beta_0_1
         ("dpo", "1_7b", 0.3)          -> experiments/phase04/dryrun/dpo_1_7b_beta_0_3
    """
    if stage == "dpo" and pref_beta is not None:
        beta_tag = f"{pref_beta:g}".replace(".", "_")
        return DRYRUN_BASE / f"dpo_{model_key}_beta_{beta_tag}"
    return DRYRUN_BASE / f"{stage}_{model_key}"


def _validate_approval(approval_path: Path) -> dict:
    """Validate the Checkpoint B approval file. Returns the parsed JSON."""
    if not approval_path.exists():
        print(f"ERROR: Approval file not found: {approval_path}")
        print("  Checkpoint B is not approved.")
        sys.exit(1)
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("ERROR: Approval file is not valid JSON.")
        sys.exit(1)
    if not approval.get("approved"):
        print("ERROR: Approval file exists but 'approved' is not true.")
        print("  Checkpoint B is not approved.")
        sys.exit(1)
    return approval


def _convert_dpo_pair_to_llamafactory_format(pair: dict) -> dict:
    """Convert a DPO pair to LLaMA-Factory 0.9.5 compatible ShareGPT ranking format.

    LLaMA-Factory's SharegptDatasetConverter enforces strict alternating tags
    (user/observation on odd positions, assistant/function_call on even positions).
    Multi-turn conversations with tool calls violate this pattern.

    For the engineering dry-run, we serialize the full prompt context into a single
    user message, producing a clean single-turn shape:

    {
      "conversations": [{"from": "user", "value": "<serialized prompt>"}],
      "chosen": {"from": "assistant", "value": "<chosen JSON>"},
      "rejected": {"from": "assistant", "value": "<rejected JSON>"}
    }
    """
    prompt = pair.get("prompt", [])

    # Serialize the full prompt context into one user message
    serialized_prompt = json.dumps(prompt, ensure_ascii=False)
    conversations = [{"from": "user", "value": serialized_prompt}]

    chosen_raw = pair.get("chosen", "")
    rejected_raw = pair.get("rejected", "")

    return {
        "conversations": conversations,
        "chosen": {"from": "assistant", "value": chosen_raw},
        "rejected": {"from": "assistant", "value": rejected_raw},
    }


def _create_dpo_dryrun_dataset(source_path: Path, count: int) -> Path:
    """Copy the first DPO pair, convert to LLaMA-Factory format, write to temp file.

    Also updates data/dataset_info.json with the absolute path to the temp file
    so LLaMA-Factory can find it.
    """
    lines = source_path.read_text(encoding="utf-8").splitlines()
    first_row = json.loads(lines[0].strip())
    converted = _convert_dpo_pair_to_llamafactory_format(first_row)
    DPO_TEMP_DATASET.parent.mkdir(parents=True, exist_ok=True)
    DPO_TEMP_DATASET.write_text(json.dumps(converted, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Temporary DPO dataset: {DPO_TEMP_DATASET} (1 pair, LLaMA-Factory format)")

    # Register in dataset_info.json with relative path from data/
    _register_temp_dataset("phase04_dryrun_dpo_1pair", DPO_TEMP_DATASET, is_dpo=True)

    return DPO_TEMP_DATASET


def _create_sft_dryrun_dataset(source_path: Path, count: int) -> Path:
    """Copy the first `count` rows from the SFT train dataset into a temp file,
    converted to LLaMA-Factory 0.9.5 sharegpt format (shared converter).

    Also updates data/dataset_info.json with the absolute path.
    """
    lines = source_path.read_text(encoding="utf-8").splitlines()
    first_row = json.loads(lines[0].strip())
    converted = convert_sft_row_to_llamafactory_format(first_row)
    SFT_TEMP_DATASET.parent.mkdir(parents=True, exist_ok=True)
    SFT_TEMP_DATASET.write_text(json.dumps(converted, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Temporary SFT dataset: {SFT_TEMP_DATASET} (1 row, LLaMA-Factory format)")

    _register_temp_dataset("phase04_dryrun_sft_1row", SFT_TEMP_DATASET, is_dpo=False)

    return SFT_TEMP_DATASET


def _register_temp_dataset(name: str, temp_path: Path, is_dpo: bool) -> None:
    """Register a temp dry-run dataset in data/dataset_info.json.

    Uses a relative path from data/ since LLaMA-Factory resolves file_name
    relative to dataset_dir (data/). The temp file must be under data/ to
    avoid non-ASCII path encoding issues with LLaMA-Factory.
    """
    info_path = Path("data/dataset_info.json")
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    # Use relative path from data/ — temp file is directly under data/
    relative_path = temp_path.name

    entry: dict = {
        "file_name": relative_path,
        "formatting": "sharegpt",
    }

    if is_dpo:
        entry.update({
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        })
    else:
        entry.update({
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        })

    info[name] = entry
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_dryrun_config(
    source_config: Path,
    output_path: Path,
    output_dir: Path,
    stage: str,
    local_model_path: Path | None = None,
    adapter_override: Path | None = None,
    extra_overrides: dict | None = None,
) -> Path:
    """Generate a temporary dry-run YAML config (LLaMA-Factory native only).

    Systematic renderer contract: only keys in ``LLAMAFACTORY_ALLOWED_ARGS`` are
    forwarded. Project-only metadata (experiment_class, train_dataset_path,
    eval_dataset_path, etc.) is never passed to llamafactory-cli.

    ``local_model_path`` overrides model_name_or_path in the RENDERED config
    while the canonical model id is preserved in the manifest/provenance.

    ``adapter_override`` sets adapter_name_or_path in the RENDERED config so a
    DPO dry-run continues from the real SFT dry-run adapter (not the formal
    SFT checkpoint path).
    """
    raw = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}

    # Pass through only LLaMA-Factory native args (shared contract).
    rendered = {k: v for k, v in raw.items() if k in LLAMAFACTORY_ALLOWED_ARGS}

    # Adapter override: DPO dry-run continues from the SFT dry-run adapter.
    if adapter_override is not None:
        rendered["adapter_name_or_path"] = str(adapter_override)

    # SFT-only flags are invalid for DPO (LLaMA-Factory rejects mask_history /
    # train_on_prompt except SFT).
    if stage != "sft":
        rendered.pop("mask_history", None)
        rendered.pop("train_on_prompt", None)

    # dataset / eval_dataset are project paths converted to LLaMA-Factory names.
    if stage == "dpo":
        rendered["dataset"] = "phase04_dryrun_dpo_1pair"
        rendered["eval_dataset"] = "phase04_dryrun_dpo_1pair"
    else:
        rendered["dataset"] = "phase04_dryrun_sft_1row"
        rendered["eval_dataset"] = "phase04_dryrun_sft_1row"

    # CPU engineering dry-run forced params (do NOT touch formal configs).
    # do_train=true is REQUIRED: without it LLaMA-Factory skips the training
    # branch -> silent no-op. do_eval=false because dry-run validates the
    # engineering pipeline only (not eval effectiveness).
    rendered["do_train"] = True
    rendered["do_eval"] = False
    rendered["max_steps"] = 2
    rendered["num_train_epochs"] = 1
    rendered["save_total_limit"] = 1
    rendered["output_dir"] = str(output_dir)
    rendered["per_device_train_batch_size"] = 1
    rendered["per_device_eval_batch_size"] = 1
    rendered["bf16"] = False
    rendered["fp16"] = False
    rendered["use_cpu"] = True
    # Force step-based checkpointing so the LoRA adapter IS saved after the
    # 2-step dry-run (epoch-based saving with max_steps=2 on a 1-row dataset
    # would otherwise complete 0 epochs and save no adapter). Disable eval to
    # avoid the trainer stalling on a metric/best-model schedule. Use
    # gradient_accumulation_steps=1 so the single 1-row sample yields a real
    # optimizer step that max_steps can iterate (with grad-acc=8 a 1-sample
    # dataset would produce 0 optimizer steps).
    rendered["save_strategy"] = "steps"
    rendered["save_steps"] = 2
    rendered["eval_strategy"] = "no"
    rendered["load_best_model_at_end"] = False
    rendered["gradient_accumulation_steps"] = 1
    rendered.pop("metric_for_best_model", None)
    rendered.pop("greater_is_better", None)
    rendered.pop("save_on_each_node", None)

    # Local model path override (offline dry-run, no HF download).
    if local_model_path is not None:
        rendered["model_name_or_path"] = str(local_model_path)

    if extra_overrides:
        rendered.update(extra_overrides)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(rendered, default_flow_style=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"  Temporary config: {output_path}")
    return output_path


def _require_local_model_path(local_model_path: Path) -> Path:
    """Fail-closed: the local HF model dir must exist (offline dry-run).

    No HF auto-download, no silent fallback to another model, no 0.6B fallback.
    """
    if not local_model_path.exists():
        raise FileNotFoundError(
            f"Local model path not found: {local_model_path}. "
            "No HF download / model fallback is allowed for the offline CPU dry-run."
        )
    return local_model_path


def _run_llamafactory_train(config_path: Path, log_path: Path) -> int:
    """Invoke llamafactory-cli train with the given config, streaming output live.

    Streams stdout/stderr line-by-line to the terminal AND writes the same to a
    log file (cross-platform, no shell-specific tee). Returns the child exit code.
    KeyboardInterrupt (Ctrl+C) terminates the child process.

    Fail-closed preflight: refuses to launch if the rendered config does not
    set do_train=true (prevents silent no-op with exit 0 + no training).
    """
    # Fail-closed: do_train must be true, or LLaMA-Factory skips training.
    rendered = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assert_rendered_config(rendered, stage=rendered.get("stage"))

    scripts_dir = Path(sys.executable).parent
    cli_path = scripts_dir / "llamafactory-cli.exe"
    if not cli_path.exists():
        cli_path = scripts_dir / "llamafactory-cli"
    if not cli_path.exists():
        print("  ERROR: llamafactory-cli not found in venv Scripts directory.")
        print(f"  Checked: {cli_path}")
        return 1
    cmd = [str(cli_path), "train", str(config_path)]
    print(f"  Running: {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in proc.stdout:
                line = line.rstrip()
                print(line)
                logf.write(line + "\n")
            proc.stdout.close()
            proc.wait()
    except KeyboardInterrupt:
        # Ctrl+C: terminate the child process cleanly.
        print("\n[KeyboardInterrupt] terminating child process...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # pragma: no cover - process may already be gone
            proc.kill()
        return 130
    return proc.returncode


def _load_existing_manifest() -> dict:
    """Load existing manifest if present, otherwise return empty base dict."""
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "checkpoint_b_approved": True,
        "frozen_test_used_for_training": False,
        "diagnostic_dev_used_for_training": False,
        "phase02_output_used_for_training": False,
        "no_training_effectiveness_claim": True,
    }


def _update_sft_manifest(
    sft_passed: bool,
    sft_sample_count: int,
    sft_max_steps: int,
    approval_file: str,
    sft_config: str,
    sft_output_dir: Path,
    sft_train_dataset: str,
    sft_eval_dataset: str,
    canonical_model_id: str,
    resolved_model_path: str,
) -> dict:
    """Update SFT fields in the dry-run manifest, preserving DPO fields.

    ``sft_output_dir`` records the REAL dry-run model size (derived from the
    training config), never a hardcoded 0_6b. Provenance records both the
    canonical formal model id and the resolved local HF path.
    """
    manifest = _load_existing_manifest()
    manifest["sft_passed"] = sft_passed
    manifest["sft_sample_count"] = sft_sample_count
    manifest["sft_max_steps"] = sft_max_steps
    manifest["approval_file"] = approval_file
    manifest["sft_config"] = sft_config
    manifest["sft_temp_dataset"] = str(SFT_TEMP_DATASET)
    manifest["sft_temp_config"] = str(SFT_TEMP_CONFIG)
    manifest["sft_output_dir"] = str(sft_output_dir)
    manifest["sft_train_dataset"] = sft_train_dataset
    manifest["sft_eval_dataset"] = sft_eval_dataset
    manifest["canonical_model_id"] = canonical_model_id
    manifest["resolved_model_path"] = resolved_model_path
    manifest["created_at"] = datetime.now(UTC).isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDry-run manifest updated (SFT) -> {MANIFEST_PATH}")
    return manifest


def _update_dpo_manifest(
    dpo_passed: bool,
    dpo_pair_count: int,
    dpo_max_steps: int,
    approval_file: str,
    dpo_config: str,
    dpo_output_dir: Path,
    dpo_train_dataset: str,
    dpo_eval_dataset: str,
    canonical_model_id: str,
    resolved_model_path: str,
    pref_beta: float,
) -> dict:
    """Update DPO fields in the dry-run manifest, preserving SFT fields.

    ``dpo_output_dir`` records the REAL dry-run model size (derived from the
    training config), never a hardcoded 0_6b. Provenance records the canonical
    model id, resolved local path, and pref_beta (beta sweep).
    """
    manifest = _load_existing_manifest()
    manifest["dpo_passed"] = dpo_passed
    manifest["dpo_pair_count"] = dpo_pair_count
    manifest["dpo_max_steps"] = dpo_max_steps
    manifest["approval_file"] = approval_file
    manifest["dpo_config"] = dpo_config
    manifest["dpo_temp_dataset"] = str(DPO_TEMP_DATASET)
    manifest["dpo_temp_config"] = str(_dpo_temp_config(pref_beta))
    manifest["dpo_output_dir"] = str(dpo_output_dir)
    manifest["dpo_train_dataset"] = dpo_train_dataset
    manifest["dpo_eval_dataset"] = dpo_eval_dataset
    manifest["canonical_model_id"] = canonical_model_id
    manifest["resolved_model_path"] = resolved_model_path
    manifest["pref_beta"] = pref_beta
    manifest["created_at"] = datetime.now(UTC).isoformat()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDry-run manifest updated (DPO) -> {MANIFEST_PATH}")
    return manifest


def _adapter_artifacts_present(output_dir: Path) -> tuple[bool, list[str]]:
    """A dry-run PASS must be based on real adapter artifacts, not just exit code.

    Checks adapter_config.json and adapter_model.safetensors exist.
    """
    missing: list[str] = []
    cfg = output_dir / "adapter_config.json"
    weights = output_dir / "adapter_model.safetensors"
    if not cfg.exists():
        missing.append(f"adapter_config.json missing: {cfg}")
    if not weights.exists():
        missing.append(f"adapter_model.safetensors missing: {weights}")
    return (not missing, missing)


def _run_sft_dryrun(
    config_path: Path,
    spec,
    approval_file: str,
    sample_count: int,
    max_steps: int,
    local_model_path: Path | None,
) -> None:
    print("\nPreparing SFT dry-run (CPU engineering):")
    print(f"  sample_count: {sample_count}")
    print(f"  max_steps: {max_steps}")

    sft_train = spec.train_dataset_path
    if sft_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)

    model_key = _derive_model_key(spec.model_name_or_path or "")
    output_dir = _dryrun_output_dir("sft", model_key)
    print(f"  output_dir: {output_dir} (model_key={model_key})")

    _create_sft_dryrun_dataset(sft_train, sample_count)
    _generate_dryrun_config(
        config_path, SFT_TEMP_CONFIG, output_dir, "sft", local_model_path=local_model_path,
    )
    log_path = DRYRUN_BASE / "logs" / "sft_llamafactory.log"
    exit_code = _run_llamafactory_train(SFT_TEMP_CONFIG, log_path)

    # PASS requires exit_code==0 AND real adapter artifacts.
    artifacts_ok, missing = _adapter_artifacts_present(output_dir)
    sft_passed = exit_code == 0 and artifacts_ok
    print(f"\nSFT dry-run {'PASSED' if sft_passed else f'FAILED (exit code: {exit_code})'}.")
    if missing:
        for m in missing:
            print(f"  MISSING: {m}")

    _update_sft_manifest(
        sft_passed=sft_passed,
        sft_sample_count=sample_count,
        sft_max_steps=max_steps,
        approval_file=approval_file,
        sft_config=str(config_path),
        sft_output_dir=output_dir,
        sft_train_dataset=str(sft_train),
        sft_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
        canonical_model_id=spec.model_name_or_path or "",
        resolved_model_path=str(local_model_path) if local_model_path else (spec.model_name_or_path or ""),
    )
    if not sft_passed:
        sys.exit(1)


def _run_dpo_dryrun(
    config_path: Path,
    spec,
    approval_file: str,
    sample_count: int,
    max_steps: int,
    local_model_path: Path | None,
    pref_beta: float,
) -> None:
    print("\nPreparing DPO dry-run (CPU engineering):")
    print(f"  pair_count: {sample_count}")
    print(f"  max_steps: {max_steps}")
    print(f"  pref_beta: {pref_beta}")

    dpo_train = spec.train_dataset_path
    if dpo_train is None:
        print("ERROR: No train dataset path in config.")
        sys.exit(1)

    model_key = _derive_model_key(spec.model_name_or_path or "")
    output_dir = _dryrun_output_dir("dpo", model_key, pref_beta)
    print(f"  output_dir: {output_dir} (model_key={model_key}, beta={pref_beta})")

    # DPO continues from the REAL SFT dry-run adapter (not the formal path).
    sft_dryrun_adapter = _dryrun_output_dir("sft", model_key)
    print(f"  DPO continues from SFT dry-run adapter: {sft_dryrun_adapter}")
    if not (sft_dryrun_adapter / "adapter_config.json").exists():
        print(f"  ERROR: SFT dry-run adapter missing: {sft_dryrun_adapter}")
        sys.exit(1)

    _create_dpo_dryrun_dataset(dpo_train, sample_count)
    temp_config = _dpo_temp_config(pref_beta)
    _generate_dryrun_config(
        config_path, temp_config, output_dir, "dpo",
        local_model_path=local_model_path,
        adapter_override=sft_dryrun_adapter,
        extra_overrides={"pref_beta": pref_beta, "pref_loss": "sigmoid"},
    )
    log_path = DRYRUN_BASE / "logs" / f"dpo_llamafactory_beta_{pref_beta:g}.log"
    exit_code = _run_llamafactory_train(temp_config, log_path)

    # PASS requires exit_code==0 AND real adapter artifacts.
    artifacts_ok, missing = _adapter_artifacts_present(output_dir)
    dpo_passed = exit_code == 0 and artifacts_ok
    print(f"\nDPO dry-run {'PASSED' if dpo_passed else f'FAILED (exit code: {exit_code})'}.")
    if missing:
        for m in missing:
            print(f"  MISSING: {m}")

    _update_dpo_manifest(
        dpo_passed=dpo_passed,
        dpo_pair_count=sample_count,
        dpo_max_steps=max_steps,
        approval_file=approval_file,
        dpo_config=str(config_path),
        dpo_output_dir=output_dir,
        dpo_train_dataset=str(dpo_train),
        dpo_eval_dataset=str(spec.eval_dataset_path) if spec.eval_dataset_path else "",
        canonical_model_id=spec.model_name_or_path or "",
        resolved_model_path=str(local_model_path) if local_model_path else (spec.model_name_or_path or ""),
        pref_beta=pref_beta,
    )
    if not dpo_passed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training dry-run (approval-gated, executes LLaMA-Factory)")
    parser.add_argument("--stage", required=True, choices=["sft", "dpo"], type=str)
    parser.add_argument("--config", required=True, type=str, help="Path to training YAML config")
    parser.add_argument("--sample-count", required=True, type=int, help="Number of samples/pairs to use")
    parser.add_argument("--max-steps", required=True, type=int, help="Max training steps")
    parser.add_argument("--approval-file", type=str, default=None,
                        help="Path to Checkpoint B approval JSON (required to proceed)")
    parser.add_argument("--local-model-path", type=str, default=None,
                        help="Resolved local HF model path for offline CPU dry-run "
                             "(fail-closed if missing; no HF download/fallback)")
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="DPO pref_beta for the dry-run (default 0.1; 0.3 accepted)")
    args = parser.parse_args()

    if not args.approval_file:
        print("ERROR: --approval-file is required. Checkpoint B is not approved.")
        sys.exit(1)

    # Approval gate: exits non-zero unless the Checkpoint B approval is valid.
    _validate_approval(Path(args.approval_file))

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    spec = load_training_run_spec(config_path)
    config_errors = validate_training_run_spec(spec)
    if config_errors:
        print("Config validation FAILED:")
        for err in config_errors:
            print(f"  - {err}")
        sys.exit(1)

    # Local model path override: fail-closed if provided but missing.
    local_model_path: Path | None = None
    if args.local_model_path:
        local_model_path = _require_local_model_path(Path(args.local_model_path))

    print(f"Config loaded: stage={spec.stage}, model={spec.model_name_or_path}")
    print(f"  Train dataset: {spec.train_dataset_path}")
    print(f"  Eval dataset: {spec.eval_dataset_path}")

    if args.stage == "dpo":
        if spec.train_dataset_path:
            train_report = validate_dpo_for_training(spec.train_dataset_path, expected_count=216)
            if not train_report.passed:
                print("DPO train dataset validation FAILED:")
                for err in train_report.errors:
                    print(f"  - {err}")
                sys.exit(1)
            print(f"  Train dataset OK: {train_report.total_rows} pairs")
        if spec.eval_dataset_path:
            eval_report = validate_dpo_for_training(spec.eval_dataset_path, expected_count=24)
            if not eval_report.passed:
                print("DPO eval dataset validation FAILED:")
                for err in eval_report.errors:
                    print(f"  - {err}")
                sys.exit(1)
            print(f"  Eval dataset OK: {eval_report.total_rows} pairs")
    else:
        if spec.train_dataset_path:
            train_report = validate_sft_for_training(spec.train_dataset_path, expected_count=540)
            if not train_report.passed:
                print("Train dataset validation FAILED:")
                for err in train_report.errors:
                    print(f"  - {err}")
                sys.exit(1)
            print(f"  Train dataset OK: {train_report.total_rows} rows")
        if spec.eval_dataset_path:
            eval_report = validate_sft_for_training(spec.eval_dataset_path, expected_count=60)
            if not eval_report.passed:
                print("Eval dataset validation FAILED:")
                for err in eval_report.errors:
                    print(f"  - {err}")
                sys.exit(1)
            print(f"  Eval dataset OK: {eval_report.total_rows} rows")

    if args.stage == "dpo":
        _run_dpo_dryrun(
            config_path, spec, args.approval_file, args.sample_count, args.max_steps,
            local_model_path, args.dpo_beta,
        )
    else:
        _run_sft_dryrun(
            config_path, spec, args.approval_file, args.sample_count, args.max_steps,
            local_model_path,
        )


if __name__ == "__main__":
    main()
