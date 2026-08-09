"""Task 8 Phase 02: Data card and report metadata tests."""

import json
from pathlib import Path

from homechef_booking.evaluation.benchmark_runner import BenchmarkConfig, run_benchmark
from homechef_booking.evaluation.data_card import generate_data_card


def test_data_card_contains_all_required_fields(tmp_path: Path):
    config = BenchmarkConfig(
        run_id="test_datacard",
        cases_path=Path("tests/fixtures/evaluation/phase01_mock_cases.jsonl"),
        backend_config_path=Path("configs/inference/mock.yaml"),
        predictions_path=Path("tests/fixtures/evaluation/phase01_mock_predictions.json"),
        output_dir=tmp_path,
        model_id="Qwen/Qwen3-0.6B-Base",
        suite_id="phase01_mock",
        manifest_path=None,
    )
    result = run_benchmark(config)
    card_path = generate_data_card(result, config, tmp_path / "data_card.json")

    assert card_path.exists()
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["benchmark_name"] == "HomeChef-Booking-M0"
    assert card["phase"] == "02"
    assert card["model_id"] == "Qwen/Qwen3-0.6B-Base"
    assert card["suite_id"] == "phase01_mock"
    assert card["results"]["total_cases"] == 19
    assert card["results"]["effective_pass_rate"] is not None
    assert "hardware" in card
    assert card["hardware"]["device"] == "auto"
    assert "generated_at" in card
    assert "contract_version" in card
    assert card["contract_version"] == "homechef-booking-v1"
    assert card["frozen_suite"] is False
    assert "no_sft_dpo_training" in card
    assert card["no_sft_dpo_training"] is True
    assert "no_model_downloaded" in card
    assert "no_real_inference" in card
    assert "no_training" in card
