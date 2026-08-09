"""Task 5: Assertions tests."""

from homechef_booking.evaluation.assertions import run_assertions
from homechef_booking.evaluation.evidence import derive_tool_evidence


def test_assertion_registry_executes_real_assertions(valid_final_case):
    evidence = derive_tool_evidence(valid_final_case.input)
    prediction = valid_final_case.expected.model_dump(mode="json", exclude_none=False)
    results = run_assertions(valid_final_case.model_copy(update={"assertions": ["tool_fact_grounded"]}), prediction, evidence)

    assert results[0].name == "tool_fact_grounded"
    assert results[0].passed is True
