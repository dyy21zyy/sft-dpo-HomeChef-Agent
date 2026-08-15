"""Phase 04 formal experiment boundary + 8-run matrix tests.

Covers Codex Phase04 Release Review Blockers:
- FORMAL_MODELS=[1.7B, 4B]; 0.6B & v0.1 excluded from formal.
- Phase02 final actual model_id is the sole training start point.
- DPO must continue from best SFT checkpoint (fail-closed, no Base fallback).
- 8-run matrix; U/S share one checkpoint.
- Structured runtime explicitly loads adapter (non-Base proof).
"""

from pathlib import Path

import yaml

from homechef_booking.inference.phase04_hf_backend import Phase04HFConfig
from homechef_booking.training.config import load_training_run_spec
from homechef_booking.training.formal_matrix import (
    FORMAL_DATASET_VERSION,
    FORMAL_MODEL_IDS,
    FORMAL_MODELS,
    Phase04EvalRun,
    build_phase04_eval_matrix,
    validate_all_matrix_runs,
    validate_dpo_source_checkpoint,
    validate_phase04_formal_spec,
    validate_structured_adapter_loaded,
)

# ── Blocker 2: Phase02 final actual model_id is the only training start ──────


def test_formal_models_match_phase02_final_actual_ids():
    assert FORMAL_MODELS["1_7b"]["model_id"] == "Qwen/Qwen3-1.7B-Base"
    # 4B MUST be Instruct-2507 (Phase02 final), NOT 4B-Base.
    assert FORMAL_MODELS["4b"]["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert "Qwen/Qwen3-4B-Base" not in FORMAL_MODEL_IDS


def test_formal_dataset_version_is_v03():
    assert FORMAL_DATASET_VERSION == "v0.3"


# ── Blocker 1: 4 formal configs exist and are loadable ──────────────────────


def test_four_formal_configs_exist():
    configs = [
        "configs/training/phase04_sft_qwen3_1_7b.yaml",
        "configs/training/phase04_sft_qwen3_4b.yaml",
        "configs/training/phase04_dpo_qwen3_1_7b.yaml",
        "configs/training/phase04_dpo_qwen3_4b.yaml",
    ]
    for c in configs:
        assert Path(c).exists(), f"missing formal config {c}"


def test_sft_configs_are_formal_and_use_v03():
    for path in (
        Path("configs/training/phase04_sft_qwen3_1_7b.yaml"),
        Path("configs/training/phase04_sft_qwen3_4b.yaml"),
    ):
        spec = load_training_run_spec(path)
        assert spec.experiment_class == "formal"
        assert spec.stage == "sft"
        assert spec.train_dataset_path == Path("data/processed/sft/v0.3/train.jsonl")
        assert spec.eval_dataset_path == Path("data/processed/sft/v0.3/val.jsonl")
        # 4B SFT must be Instruct-2507, not Base.
        if "4b" in path.name:
            assert spec.model_name_or_path == "Qwen/Qwen3-4B-Instruct-2507"
        else:
            assert spec.model_name_or_path == "Qwen/Qwen3-1.7B-Base"
        assert validate_phase04_formal_spec(spec) == []


def test_dpo_configs_are_formal_and_reference_best_sft_checkpoint():
    for path in (
        Path("configs/training/phase04_dpo_qwen3_1_7b.yaml"),
        Path("configs/training/phase04_dpo_qwen3_4b.yaml"),
    ):
        spec = load_training_run_spec(path)
        assert spec.experiment_class == "formal"
        assert spec.stage == "dpo"
        assert spec.train_dataset_path == Path("data/processed/dpo/v0.3/train.jsonl")
        assert spec.eval_dataset_path == Path("data/processed/dpo/v0.3/val.jsonl")
        assert spec.adapter_name_or_path is not None
        assert validate_phase04_formal_spec(spec) == []
        # 4B DPO must be Instruct-2507.
        if "4b" in path.name:
            assert spec.model_name_or_path == "Qwen/Qwen3-4B-Instruct-2507"


# ── Blocker 6: 0.6B and v0.1 excluded from formal ───────────────────────────


def test_formal_spec_rejects_0_6b_model():
    spec = load_training_run_spec(Path("configs/training/phase04_sft_qwen3_0_6b.yaml"))
    errors = validate_phase04_formal_spec(spec)
    assert any("historical-only" in e for e in errors)
    assert any("not a Phase 04 formal model" in e for e in errors)


def test_formal_spec_rejects_4b_base_variant(tmp_path: Path):
    path = tmp_path / "bad_4b_base.yaml"
    path.write_text(
        yaml.dump({
            "stage": "sft",
            "experiment_class": "formal",
            "model_name_or_path": "Qwen/Qwen3-4B-Base",
            "train_dataset_path": "data/processed/sft/v0.3/train.jsonl",
            "eval_dataset_path": "data/processed/sft/v0.3/val.jsonl",
        }),
        encoding="utf-8",
    )
    spec = load_training_run_spec(path)
    errors = validate_phase04_formal_spec(spec)
    assert any("Instruct-2507" in e and "Base" in e for e in errors)


def test_formal_spec_rejects_v01_dataset(tmp_path: Path):
    path = tmp_path / "bad_v01.yaml"
    path.write_text(
        yaml.dump({
            "stage": "sft",
            "experiment_class": "formal",
            "model_name_or_path": "Qwen/Qwen3-1.7B-Base",
            "train_dataset_path": "data/processed/sft/v0.1/train.jsonl",
            "eval_dataset_path": "data/processed/sft/v0.3/val.jsonl",
        }),
        encoding="utf-8",
    )
    spec = load_training_run_spec(path)
    errors = validate_phase04_formal_spec(spec)
    assert any("v0.1" in e for e in errors)


def test_formal_spec_requires_experiment_class(tmp_path: Path):
    path = tmp_path / "no_class.yaml"
    path.write_text(
        yaml.dump({
            "stage": "sft",
            "model_name_or_path": "Qwen/Qwen3-1.7B-Base",
            "train_dataset_path": "data/processed/sft/v0.3/train.jsonl",
            "eval_dataset_path": "data/processed/sft/v0.3/val.jsonl",
        }),
        encoding="utf-8",
    )
    spec = load_training_run_spec(path)
    errors = validate_phase04_formal_spec(spec)
    assert any("experiment_class must be 'formal'" in e for e in errors)


# ── Blocker 3: DPO fail-closed on missing SFT checkpoint ────────────────────


def test_dpo_missing_adapter_fails(tmp_path: Path):
    path = tmp_path / "dpo_no_adapter.yaml"
    path.write_text(
        yaml.dump({
            "stage": "dpo",
            "experiment_class": "formal",
            "model_name_or_path": "Qwen/Qwen3-1.7B-Base",
            "train_dataset_path": "data/processed/dpo/v0.3/train.jsonl",
            "eval_dataset_path": "data/processed/dpo/v0.3/val.jsonl",
            # No adapter_name_or_path: must fail, not fall back to Base.
        }),
        encoding="utf-8",
    )
    spec = load_training_run_spec(path)
    errors = validate_dpo_source_checkpoint(spec)
    assert any("adapter_name_or_path" in e for e in errors)
    assert any("must NOT start from Base" in e for e in errors)


def test_dpo_nonexistent_sft_checkpoint_fails(tmp_path: Path):
    path = tmp_path / "dpo_bad_adapter.yaml"
    path.write_text(
        yaml.dump({
            "stage": "dpo",
            "experiment_class": "formal",
            "model_name_or_path": "Qwen/Qwen3-1.7B-Base",
            "train_dataset_path": "data/processed/dpo/v0.3/train.jsonl",
            "eval_dataset_path": "data/processed/dpo/v0.3/val.jsonl",
            "adapter_name_or_path": "experiments/phase04/nope/checkpoint-best",
        }),
        encoding="utf-8",
    )
    spec = load_training_run_spec(path)
    errors = validate_dpo_source_checkpoint(spec)
    assert any("Refusing to fall back to Base" in e for e in errors)


def test_dpo_sft_stage_not_checked():
    # validate_dpo_source_checkpoint must be a no-op for SFT stage.
    sft_spec = load_training_run_spec(Path("configs/training/phase04_sft_qwen3_4b.yaml"))
    assert validate_dpo_source_checkpoint(sft_spec) == []


# ── Blocker 4: 12-run matrix (beta sweep) ────────────────────────────────────


def test_matrix_has_12_runs():
    runs = build_phase04_eval_matrix()
    assert len(runs) == 12
    # Distribution: SFT=4, DPO beta=0.1=4, DPO beta=0.3=4.
    sft = [r for r in runs if r.stage == "sft"]
    dpo_01 = [r for r in runs if r.stage == "dpo" and r.pref_beta == 0.1]
    dpo_03 = [r for r in runs if r.stage == "dpo" and r.pref_beta == 0.3]
    assert len(sft) == 4
    assert len(dpo_01) == 4
    assert len(dpo_03) == 4
    # Unique run_ids.
    assert len({r.run_id for r in runs}) == 12


def test_matrix_validation_passes():
    runs = build_phase04_eval_matrix()
    assert validate_all_matrix_runs(runs) == []


def test_matrix_uses_only_formal_models():
    runs = build_phase04_eval_matrix()
    model_ids = {r.model_id for r in runs}
    assert model_ids == FORMAL_MODEL_IDS
    assert "Qwen/Qwen3-0.6B-Base" not in model_ids
    assert "Qwen/Qwen3-4B-Base" not in model_ids


# ── Blocker 5: U/S share the same checkpoint (per model/stage/beta) ──────────


def test_structured_unstructured_share_checkpoint():
    runs = build_phase04_eval_matrix()
    by_key = {}
    for r in runs:
        by_key.setdefault((r.model_key, r.stage, r.pref_beta), []).append(r)
    for _key, group in by_key.items():
        assert len(group) == 2
        variants = {r.variant for r in group}
        assert variants == {"u", "s"}
        # Same checkpoint, same adapter, same pref_beta.
        assert group[0].checkpoint_path == group[1].checkpoint_path
        assert group[0].adapter_name_or_path == group[1].adapter_name_or_path
        assert group[0].pref_beta == group[1].pref_beta
        # Differ ONLY by structured flag.
        assert group[0].use_structured_output != group[1].use_structured_output


# ── Blocker 6: structured runtime explicitly loads adapter (non-Base proof) ─


def test_each_matrix_run_has_explicit_adapter():
    runs = build_phase04_eval_matrix()
    for r in runs:
        assert r.adapter_name_or_path is not None
        assert str(r.adapter_name_or_path) != ""
        assert r.adapter_name_or_path == r.checkpoint_path
        assert validate_structured_adapter_loaded(r) == []


def test_structured_run_without_adapter_fails():
    run = Phase04EvalRun(
        run_id="phase04_1_7b_sft_s",
        model_key="1_7b",
        model_id="Qwen/Qwen3-1.7B-Base",
        stage="sft",
        variant="s",
        checkpoint_path=Path("experiments/phase04/sft_qwen3_1_7b/checkpoint-best"),
        adapter_name_or_path=None,
        use_structured_output=True,
        output_dir=Path("reports/generated/phase04/matrix/phase04_1_7b_sft_s"),
        backend_config_path=Path("configs/phase04/backends/phase04_1_7b_sft_s_backend.yaml"),
    )
    errors = validate_structured_adapter_loaded(run)
    assert any("explicit" in e for e in errors)


def test_phase04_hf_config_requires_adapter():
    cfg = Phase04HFConfig(model_id="Qwen/Qwen3-4B-Instruct-2507", adapter_name_or_path=None)
    errors = cfg.validate_for_run()
    assert any("adapter_name_or_path is required" in e for e in errors)
    assert any("non-Base proof" in e for e in errors)


def test_phase04_hf_config_rejects_4b_base():
    cfg = Phase04HFConfig(
        model_id="Qwen/Qwen3-4B-Base",
        adapter_name_or_path="experiments/phase04/sft_qwen3_4b/checkpoint-best",
    )
    errors = cfg.validate_for_run()
    assert any("not a Phase02 final model" in e for e in errors)


def test_phase04_hf_config_accepts_formal_models_with_adapter():
    for model_id in ("Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Instruct-2507"):
        cfg = Phase04HFConfig(
            model_id=model_id,
            adapter_name_or_path="experiments/phase04/sft_qwen3_4b/checkpoint-best",
            training_stage="sft",
        )
        assert cfg.validate_for_run() == []


def test_all_matrix_configs_writeable_and_validate():
    # Confirm every written eval YAML is loadable and well-formed.
    runs = build_phase04_eval_matrix()
    for r in runs:
        yaml_path = Path("configs/evaluation/phase04_matrix") / f"{r.run_id}.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert data["run_id"] == r.run_id
        assert data["model_id"] == r.model_id
        assert data["use_structured_output"] == r.use_structured_output
        assert data["adapter_name_or_path"].replace("\\", "/") == str(r.adapter_name_or_path).replace("\\", "/")
        # No non-formal models / versions leaked into the written configs.
        assert "v0.1" not in yaml_path.read_text(encoding="utf-8")
