"""Task 8 Phase 03: Raw generation CLI tests."""

from homechef_booking.data.generator import GenerationPlan, build_generation_prompt


def test_generation_prompt_names_contract_and_forbids_frozen_eval_training_use():
    prompt = build_generation_prompt(GenerationPlan(mode="smoke", count=25, seed=3001))
    assert "BOOKING_MACHINE_CONTRACT_v1" in prompt
    assert "Frozen Test" in prompt
    assert "never be used as training data" in prompt
    assert "must never be used as training data" in prompt or "never be used as training data" in prompt
