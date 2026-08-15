"""Phase 04 Formal Experiment Boundary & 8-Run Evaluation Matrix.

Enforces the Codex Phase04 Release Review rules:

- FORMAL_MODELS = [1.7B, 4B]; FORMAL_DATASET_VERSION = v0.3.
- 0.6B and v0.1 must NOT enter formal discovery / run-all / evaluation matrix /
  comparison / release gate (they stay available only as historical configs).
- The Phase02 final actual model_id is the sole training-start-point source:
      - 1.7B -> Qwen/Qwen3-1.7B-Base
      - 4B   -> Qwen/Qwen3-4B-Instruct-2507   (NOT Qwen3-4B-Base)
- DPO must continue from the corresponding best SFT adapter/checkpoint. If the
  SFT checkpoint is missing the DPO config FAILS (no Base fallback).
- Structured/Unstructured share the SAME trained checkpoint; they differ ONLY by
  the runtime structured-output constraint. Structured runtime must explicitly
  load the SFT/DPO adapter and prove (via adapter path) that it is not the Base.

This module is pure configuration/validation: it never generates reports,
never trains, and never writes adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from homechef_booking.training.config import TrainingRunSpec

# ── Phase 04 formal boundary ─────────────────────────────────────────────────
FORMAL_DATASET_VERSION = "v0.3"

# Phase02 final actual model_id -> formal training start point.
# key -> (label, model_id)
FORMAL_MODELS: dict[str, dict[str, str]] = {
    "1_7b": {"label": "1.7B", "model_id": "Qwen/Qwen3-1.7B-Base"},
    "4b": {"label": "4B", "model_id": "Qwen/Qwen3-4B-Instruct-2507"},
}

FORMAL_MODEL_IDS = {m["model_id"] for m in FORMAL_MODELS.values()}

# Historical / non-formal model ids that must never appear in a formal spec.
NON_FORMAL_MODEL_IDS = {"Qwen/Qwen3-0.6B-Base"}

# The 4B must use the Phase02 final Instruct-2507 variant, NOT the Base variant.
_4B_BASE_ID = "Qwen/Qwen3-4B-Base"
_4B_INSTRUCT_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Formal frozen dataset paths (v0.3 only). Counts per spec.
FORMAL_DATASETS = {
    "sft": {
        "train": Path("data/processed/sft/v0.3/train.jsonl"),
        "eval": Path("data/processed/sft/v0.3/val.jsonl"),
        "train_count": 540,
        "eval_count": 60,
    },
    "dpo": {
        "train": Path("data/processed/dpo/v0.3/train.jsonl"),
        "eval": Path("data/processed/dpo/v0.3/val.jsonl"),
        "train_count": 216,
        "eval_count": 24,
    },
}


# model_size is a model-identity property, independent of training stage.
_MODEL_KEY_SIZE = {"1_7b": "1.7B", "4b": "4B"}


def _posix(p: Path) -> str:
    """Serialize a repo-relative path with POSIX separators (Linux portability)."""
    return str(p).replace("\\", "/")


@dataclass(frozen=True)
class Phase04EvalRun:
    """One Phase 04 evaluation run.

    Structured and Unstructured runs share the same ``checkpoint_path`` and
    ``adapter_name_or_path``; they differ ONLY in ``use_structured_output``.

    ``pref_beta`` is None for SFT runs and one of (0.1, 0.3) for DPO runs
    (DPO beta sweep). Each run carries a ``backend_config_path`` so it is
    directly executable via ``evaluation/runner.py`` (cases_path +
    backend_config_path).
    """

    run_id: str
    model_key: str            # "1_7b" | "4b"
    model_id: str             # Phase02 final model_id
    stage: str                # "sft" | "dpo"
    variant: str              # "u" (unstructured) | "s" (structured)
    checkpoint_path: Path     # the trained checkpoint (adapter) to evaluate
    adapter_name_or_path: Path  # explicit adapter dir (non-Base proof)
    use_structured_output: bool
    output_dir: Path
    backend_config_path: Path  # phase04_hf backend config (executable)
    pref_beta: float | None = None  # None for SFT; 0.1/0.3 for DPO

    @property
    def label(self) -> str:
        return f"{self.model_key}.{self.stage}.{self.variant}"

    @property
    def training_stage(self) -> str:
        # training_stage is an EXPERIMENT-STAGE property, not model identity.
        # It MUST come directly from run.stage ("sft" | "dpo"), never derived
        # from the model key or model size.
        return self.stage

    @property
    def model_size(self) -> str:
        # model_size is a model-identity property, independent of training stage.
        return _MODEL_KEY_SIZE.get(self.model_key, "")


def _sft_checkpoint_path(model_key: str) -> Path:
    return Path("experiments/phase04") / f"sft_qwen3_{model_key}" / "checkpoint-best"


def _dpo_checkpoint_path(model_key: str, beta: float) -> Path:
    beta_tag = f"{beta:g}".replace(".", "_")
    return Path("experiments/phase04") / f"dpo_qwen3_{model_key}_beta_{beta_tag}" / "checkpoint-best"


# DPO beta sweep values (formal Phase04).
DPO_BETAS = (0.1, 0.3)

# ── Exploratory Current-Data evaluation matrix ───────────────────────────────
# This is the Phase04 Frozen Test evaluation run for exploratory_currentdata.
# 6 checkpoints x 2 runtime modes (U/S) = 12 cells.
# Structured/Unstructured are RUNTIME evaluation modes, NOT separate checkpoints.
#
# experiment_class = exploratory_currentdata (NOT formal release):
#   known limitation: SFT canonical 540/60 but LLaMA-Factory actual 314/36;
#   tool-context trajectories are skipped. formal_release_eligible = false.
EXPLORATORY_CLASS = "exploratory_currentdata"
EXPLORATORY_CHECKPOINT_ROOT = Path("experiments/phase04/exploratory_currentdata")
KNOWN_DATA_LIMITATION = (
    "SFT canonical 540/60 but LLaMA-Factory actual 314/36; tool-context trajectories skipped"
)
FROZEN_TEST_PATH = Path("data/eval/frozen_test.jsonl")
FROZEN_TEST_MANIFEST_PATH = Path("data/eval/frozen_test.manifest.json")
FROZEN_TEST_SHA256 = "c67bb58f5d390598ea5e7470b806f8d555964af9165c75ef04d05f671f5ca8ed"
FROZEN_TEST_TOTAL_CASES = 120


def _exploratory_adapter_path(model_key: str, stage: str, beta: float | None = None) -> Path:
    root = EXPLORATORY_CHECKPOINT_ROOT
    if stage == "sft":
        return root / f"sft_qwen3_{model_key}"
    if stage == "dpo":
        beta_tag = f"{beta:g}".replace(".", "_") if beta is not None else "01"
        return root / f"dpo_qwen3_{model_key}_beta_{beta_tag}"
    raise ValueError(f"Unknown stage: {stage}")


def build_phase04_exploratory_matrix(
    output_root: Path = Path("reports/generated/phase04/exploratory_currentdata"),
) -> list[Phase04EvalRun]:
    """Build the 12-cell exploratory Frozen Test matrix.

    Fixed run_ids (per Codex review):
        phase04_1_7b_sft_u/s, phase04_1_7b_dpo_b01_u/s, phase04_1_7b_dpo_b03_u/s
        phase04_4b_sft_u/s,   phase04_4b_dpo_b01_u/s,   phase04_4b_dpo_b03_u/s

    Each checkpoint appears exactly twice: one U (use_structured_output=False)
    and one S (use_structured_output=True), sharing base model + adapter.
    """
    runs: list[Phase04EvalRun] = []
    for model_key, meta in FORMAL_MODELS.items():
        # SFT
        sft_adapter = _exploratory_adapter_path(model_key, "sft")
        for variant, structured in (("u", False), ("s", True)):
            run_id = f"phase04_{model_key}_sft_{variant}"
            runs.append(Phase04EvalRun(
                run_id=run_id, model_key=model_key, model_id=meta["model_id"],
                stage="sft", variant=variant, checkpoint_path=sft_adapter,
                adapter_name_or_path=sft_adapter, use_structured_output=structured,
                output_dir=output_root / run_id,
                backend_config_path=Path("configs/phase04/exploratory_backends") / f"{run_id}_backend.yaml",
                pref_beta=None,
            ))
        # DPO beta sweep
        for beta in DPO_BETAS:
            bshort = f"b{beta:g}".replace(".", "")      # "b01"/"b03"
            adapter = _exploratory_adapter_path(model_key, "dpo", beta)
            for variant, structured in (("u", False), ("s", True)):
                run_id = f"phase04_{model_key}_dpo_{bshort}_{variant}"
                runs.append(Phase04EvalRun(
                    run_id=run_id, model_key=model_key, model_id=meta["model_id"],
                    stage="dpo", variant=variant, checkpoint_path=adapter,
                    adapter_name_or_path=adapter, use_structured_output=structured,
                    output_dir=output_root / run_id,
                    backend_config_path=Path("configs/phase04/exploratory_backends") / f"{run_id}_backend.yaml",
                    pref_beta=beta,
                ))
    return runs


def validate_exploratory_matrix(runs: list[Phase04EvalRun]) -> list[str]:
    """Validate the 12-cell exploratory matrix.

    - exactly 12 runs, 12 unique run_ids.
    - exactly 6 unique adapters; each adapter has exactly one U and one S.
    - 1.7B/4B each contribute 6 cells.
    - DPO has both beta 0.1 and 0.3.
    - U/S share adapter; differ only in use_structured_output.
    """
    errors: list[str] = []
    if len(runs) != 12:
        errors.append(f"Expected 12 runs in the exploratory matrix, got {len(runs)}.")
    run_ids = [r.run_id for r in runs]
    if len(set(run_ids)) != len(run_ids):
        errors.append(f"run_ids are not unique: {[r for r in run_ids if run_ids.count(r) > 1]}")

    # 6 unique adapters, each U+S.
    adapter_to_runs: dict[str, list[Phase04EvalRun]] = {}
    for r in runs:
        adapter_to_runs.setdefault(str(r.adapter_name_or_path), []).append(r)
    if len(adapter_to_runs) != 6:
        errors.append(f"Expected 6 unique adapters, got {len(adapter_to_runs)}.")
    for adapter, group in adapter_to_runs.items():
        if len(group) != 2:
            errors.append(f"Adapter {adapter} must have exactly 2 runs (U/S), got {len(group)}.")
        variants = {r.variant for r in group}
        if variants != {"u", "s"}:
            errors.append(f"Adapter {adapter} must have one 'u' and one 's', got {variants}.")

    # Per-model 6 cells.
    for mk in ("1_7b", "4b"):
        n = sum(1 for r in runs if r.model_key == mk)
        if n != 6:
            errors.append(f"model {mk} must contribute 6 cells, got {n}.")

    # DPO beta both 0.1 and 0.3.
    betas = {r.pref_beta for r in runs if r.stage == "dpo"}
    if betas != {0.1, 0.3}:
        errors.append(f"DPO beta set must be {{0.1, 0.3}}, got {betas}.")

    # U/S share adapter, differ only in structured flag.
    for r in runs:
        if r.adapter_name_or_path != r.checkpoint_path:
            errors.append(f"{r.run_id}: adapter != checkpoint; U/S must share one checkpoint.")

    # Fixed run_id set.
    expected_ids = {
        "phase04_1_7b_sft_u", "phase04_1_7b_sft_s",
        "phase04_1_7b_dpo_b01_u", "phase04_1_7b_dpo_b01_s",
        "phase04_1_7b_dpo_b03_u", "phase04_1_7b_dpo_b03_s",
        "phase04_4b_sft_u", "phase04_4b_sft_s",
        "phase04_4b_dpo_b01_u", "phase04_4b_dpo_b01_s",
        "phase04_4b_dpo_b03_u", "phase04_4b_dpo_b03_s",
    }
    if set(run_ids) != expected_ids:
        errors.append(f"run_ids do not match the fixed exploratory set. Got {sorted(set(run_ids))}.")
    return errors


def exploratory_matrix_to_yaml(runs: list[Phase04EvalRun]) -> list[dict[str, Any]]:
    """Serialize the exploratory matrix to eval-config dicts (one per run)."""
    return [
        {
            "run_id": r.run_id,
            "model_key": r.model_key,
            "model_id": r.model_id,
            "stage": r.stage,
            "variant": r.variant,
            "pref_beta": r.pref_beta,
            "use_structured_output": r.use_structured_output,
            "adapter_name_or_path": _posix(r.adapter_name_or_path),
            "output_dir": _posix(r.output_dir),
            "cases_path": "data/eval/frozen_test.jsonl",
            "manifest_path": "data/eval/frozen_test.manifest.json",
        }
        for r in runs
    ]


def build_phase04_eval_matrix(
    base_dir: Path = Path("reports/generated/phase04/matrix"),
    backend_dir: Path = Path("configs/phase04/backends"),
    eval_dir: Path = Path("configs/evaluation/phase04_matrix"),
) -> list[Phase04EvalRun]:
    """Build the 12-run Phase04 evaluation matrix.

    [1.7B, 4B] x [SFT, DPO-beta(0.1), DPO-beta(0.3)] x [U, S] = 12 runs.

    - SFT: pref_beta is None.
    - DPO: pref_beta in {0.1, 0.3} (beta sweep).

    U/S runs for a given (model, stage, beta) share the SAME checkpoint; they
    differ only in use_structured_output. Every run is executable via
    ``evaluation/runner.py`` (backend_config_path present).
    """
    runs: list[Phase04EvalRun] = []
    for model_key, meta in FORMAL_MODELS.items():
        # SFT (pref_beta=None)
        sft_checkpoint = _sft_checkpoint_path(model_key)
        for variant, structured in (("u", False), ("s", True)):
            run_id = f"phase04_{model_key}_sft_{variant}"
            runs.append(
                Phase04EvalRun(
                    run_id=run_id,
                    model_key=model_key,
                    model_id=meta["model_id"],
                    stage="sft",
                    variant=variant,
                    checkpoint_path=sft_checkpoint,
                    adapter_name_or_path=sft_checkpoint,
                    use_structured_output=structured,
                    output_dir=base_dir / run_id,
                    backend_config_path=backend_dir / f"{run_id}_backend.yaml",
                    pref_beta=None,
                )
            )
        # DPO beta sweep (pref_beta in {0.1, 0.3})
        for beta in DPO_BETAS:
            dpo_checkpoint = _dpo_checkpoint_path(model_key, beta)
            beta_tag = f"{beta:g}".replace(".", "_")
            for variant, structured in (("u", False), ("s", True)):
                run_id = f"phase04_{model_key}_dpo_beta_{beta_tag}_{variant}"
                runs.append(
                    Phase04EvalRun(
                        run_id=run_id,
                        model_key=model_key,
                        model_id=meta["model_id"],
                        stage="dpo",
                        variant=variant,
                        checkpoint_path=dpo_checkpoint,
                        adapter_name_or_path=dpo_checkpoint,
                        use_structured_output=structured,
                        output_dir=base_dir / run_id,
                        backend_config_path=backend_dir / f"{run_id}_backend.yaml",
                        pref_beta=beta,
                    )
                )
    return runs


def backend_config_to_yaml(run: Phase04EvalRun) -> dict[str, object]:
    """Serialize the Phase04 HF backend config for a matrix run.

    Structured and Unstructured share model_id + adapter_name_or_path; they
    differ ONLY in use_structured_output. Device/torch_dtype defaults mirror
    the frozen Phase04 formal training settings.
    """
    return {
        "backend": "phase04_hf",
        "model_id": run.model_id,
        "adapter_name_or_path": _posix(run.adapter_name_or_path),
        "use_structured_output": run.use_structured_output,
        "device": "auto",
        "torch_dtype": "bfloat16",
        "max_model_length": 2048,
        "max_new_tokens": 512,
        "model_key": run.model_key,
        "training_stage": run.training_stage,
        "model_size": run.model_size,
        # pref_beta: None for SFT, 0.1/0.3 for DPO (beta sweep provenance).
        "pref_beta": run.pref_beta,
    }


def _is_v03_dataset_path(p: Path) -> bool:
    """A formal training dataset path must live under a v0.3 directory."""
    parts = [part for part in p.parts if part]
    # Accept "sft/v0.3/..." or "dpo/v0.3/...".
    return any(part == "v0.3" for part in parts)


def validate_phase04_formal_spec(spec: TrainingRunSpec) -> list[str]:
    """Fail-closed validation of a Phase 04 formal training config.

    Rejects:
    - any model not in FORMAL_MODELS (0.6B and v0.1-free enforced here);
    - the 4B Base variant (must be Instruct-2507 per Phase02 final);
    - non-v0.3 dataset paths (v0.1 is NOT allowed in the formal experiment).
    """
    errors: list[str] = []

    if spec.experiment_class != "formal":
        errors.append(
            "experiment_class must be 'formal' for a Phase 04 formal config."
        )

    model = spec.model_name_or_path or ""
    if model not in FORMAL_MODEL_IDS:
        errors.append(
            f"model_name_or_path '{model}' is not a Phase 04 formal model. "
            f"FORMAL_MODELS = {sorted(FORMAL_MODEL_IDS)}."
        )
    if model == _4B_BASE_ID:
        errors.append(
            f"4B must use the Phase02 final model {_4B_INSTRUCT_ID}, not {_4B_BASE_ID}."
        )
    if model in NON_FORMAL_MODEL_IDS:
        errors.append(
            f"model '{model}' is historical-only and must not enter the formal experiment."
        )

    # Dataset version boundary: formal uses v0.3 only.
    for field_name, path in (
        ("train_dataset_path", spec.train_dataset_path),
        ("eval_dataset_path", spec.eval_dataset_path),
    ):
        if path is None:
            errors.append(f"{field_name} is required for a formal config.")
            continue
        if not _is_v03_dataset_path(path):
            errors.append(
                f"{field_name} '{path}' is not under FORMAL_DATASET_VERSION "
                f"'{FORMAL_DATASET_VERSION}'. v0.1/historical data is not allowed "
                f"in the formal experiment."
            )

    return errors


def validate_dpo_source_checkpoint(spec: TrainingRunSpec) -> list[str]:
    """Fail-closed: a formal DPO config MUST continue from an existing SFT
    adapter/checkpoint. If the SFT checkpoint is missing, FAIL — never fall
    back to the Base model.
    """
    errors: list[str] = []

    if spec.stage != "dpo":
        return errors

    adapter = spec.adapter_name_or_path
    if adapter is None or str(adapter) == "":
        errors.append(
            "DPO requires adapter_name_or_path (the best-SFT checkpoint). "
            "DPO must NOT start from Base."
        )
        return errors

    adapter = Path(adapter)
    # adapter_config.json is the canonical LLaMA-Factory adapter marker. The
    # checkpoint dir may also exist without it during dry-run, but a formal
    # DPO must resolve to a real adapter to be runnable.
    marker = adapter / "adapter_config.json"
    if not adapter.exists() or not marker.exists():
        errors.append(
            f"DPO source SFT checkpoint not found: {adapter} (no adapter_config.json). "
            f"Refusing to fall back to Base. Train the SFT checkpoint first."
        )

    return errors


def validate_structured_adapter_loaded(run: Phase04EvalRun) -> list[str]:
    """Prove a structured run actually evaluates the adapter, not Base.

    Rules:
    - The structured run must reference a non-empty adapter_name_or_path.
    - The adapter path must equal the trained checkpoint (so U/S share it).
    - model_id must be the Phase02 final model (not a bare placeholder).
    """
    errors: list[str] = []
    if not run.adapter_name_or_path or str(run.adapter_name_or_path) == "":
        errors.append(
            f"{run.run_id}: structured/unstructured run must reference an explicit "
            "adapter_name_or_path (non-Base proof)."
        )
    if run.adapter_name_or_path != run.checkpoint_path:
        errors.append(
            f"{run.run_id}: structured and unstructured runs must share the SAME "
            "checkpoint (U/S differ only by the runtime structured constraint). "
            f"got adapter={run.adapter_name_or_path}, checkpoint={run.checkpoint_path}."
        )
    if run.model_id not in FORMAL_MODEL_IDS:
        errors.append(f"{run.run_id}: model_id '{run.model_id}' is not a formal model.")
    return errors


def validate_all_matrix_runs(runs: list[Phase04EvalRun]) -> list[str]:
    """Validate the full 12-run matrix.

    Returns a flat list of errors; empty list means the matrix is consistent.
    """
    errors: list[str] = []
    seen_checkpoints: dict[tuple[str, str, float | None], Phase04EvalRun] = {}

    for run in runs:
        errors.extend(validate_structured_adapter_loaded(run))

        # A (model, stage, beta) tuple must appear exactly twice: one U, one S.
        key = (run.model_key, run.stage, run.pref_beta)
        if key in seen_checkpoints:
            prev = seen_checkpoints[key]
            if run.checkpoint_path != prev.checkpoint_path:
                errors.append(
                    f"{run.run_id} and {prev.run_id} share (model, stage, beta) but use "
                    f"different checkpoints; U/S must share one checkpoint."
                )
            variants = {prev.variant, run.variant}
            if "u" not in variants or "s" not in variants:
                errors.append(
                    f"{key}: expected both an 'u' and an 's' run, got {sorted(variants)}."
                )
        else:
            seen_checkpoints[key] = run

    # Expect exactly 12 runs (2 models x [SFT + 2 DPO betas] x 2 variants).
    if len(runs) != 12:
        errors.append(f"Expected 12 runs in the Phase04 matrix, got {len(runs)}.")

    # SFT runs: pref_beta must be None. DPO runs: pref_beta in DPO_BETAS.
    sft_count = sum(1 for r in runs if r.stage == "sft")
    dpo_count = sum(1 for r in runs if r.stage == "dpo")
    if sft_count != 4:
        errors.append(f"Expected 4 SFT runs, got {sft_count}.")
    if dpo_count != 8:
        errors.append(f"Expected 8 DPO runs, got {dpo_count}.")
    for r in runs:
        if r.stage == "sft" and r.pref_beta is not None:
            errors.append(f"{r.run_id}: SFT run must have pref_beta=None, got {r.pref_beta}.")
        if r.stage == "dpo" and r.pref_beta not in DPO_BETAS:
            errors.append(f"{r.run_id}: DPO run must have pref_beta in {DPO_BETAS}, got {r.pref_beta}.")

    model_keys = {r.model_key for r in runs}
    if model_keys != set(FORMAL_MODELS):
        errors.append(f"Matrix must cover FORMAL_MODELS={sorted(FORMAL_MODELS)}, got {sorted(model_keys)}.")

    return errors


def matrix_to_yaml(runs: list[Phase04EvalRun]) -> list[dict[str, Any]]:
    """Serialize the matrix to a list of EVAL-config dicts (one per run).

    Each eval config is executable via ``evaluation/runner.py``: it carries a
    ``backend_config_path`` (plus cases_path/manifest) and uses POSIX separators
    for all repo-relative paths (Linux portability).

    Each run also emits a backend config (``backend_config_to_yaml``) that the
    eval config points at.
    """
    return [
        {
            "run_id": run.run_id,
            "model_key": run.model_key,
            "model_id": run.model_id,
            "stage": run.stage,
            "variant": run.variant,
            "use_structured_output": run.use_structured_output,
            "checkpoint_path": _posix(run.checkpoint_path),
            "adapter_name_or_path": _posix(run.adapter_name_or_path),
            "output_dir": _posix(run.output_dir),
            "backend_config_path": _posix(run.backend_config_path),
            "pref_beta": run.pref_beta,
            "cases_path": "data/eval/frozen_test.jsonl",
            "manifest_path": "data/eval/frozen_test.manifest.json",
        }
        for run in runs
    ]
