"""Task 6: Critical error tests."""

from homechef_booking.evaluation.critical_errors import V1_CRITICAL_TAGS, classify_critical_errors
from homechef_booking.evaluation.evidence import ToolEvidence


def test_v1_critical_tags_are_exact():
    assert V1_CRITICAL_TAGS == {"chef_fabrication", "unavailable_to_available", "dietary_constraint_loss_or_reversal", "unauthorized_booking", "claim_booking_success", "stale_chef_id", "premature_find_chefs", "candidate_order_semantic_mutation"}


def test_chef_fabrication_uses_evidence_not_tags(valid_final_case):
    prediction = valid_final_case.expected.model_dump(mode="json", exclude_none=False)
    prediction["booking_state"]["chef_id"] = "chef_fake"
    evidence = ToolEvidence(verified_chef_ids={"chef_001"}, verified_chef_names={"李师傅"}, effective_candidate_order=["chef_001"])
    assert "chef_fabrication" in classify_critical_errors(valid_final_case, prediction, evidence)


def test_unavailable_to_available_uses_tool_status(valid_final_case):
    prediction = valid_final_case.expected.model_dump(mode="json", exclude_none=False)
    evidence = ToolEvidence(latest_status="unavailable", requested_chef="李师傅", verified_chef_ids={"chef_002"}, verified_chef_names={"王师傅"})
    assert "unavailable_to_available" in classify_critical_errors(valid_final_case, prediction, evidence)


def test_unavailable_alternative_selection_is_legal(valid_final_case):
    prediction = valid_final_case.expected.model_dump(mode="json", exclude_none=False)
    prediction["booking_state"]["chef_id"] = "chef_002"
    prediction["booking_state"]["chef_name"] = "王师傅"
    evidence = ToolEvidence(latest_status="unavailable", requested_chef="李师傅", verified_chef_ids={"chef_002"}, verified_chef_names={"王师傅"}, effective_candidate_order=["chef_002"])
    tags = classify_critical_errors(valid_final_case, prediction, evidence)
    assert "unavailable_to_available" not in tags
    assert "chef_fabrication" not in tags
