from __future__ import annotations

from homechef_booking.evaluation.evidence import ToolEvidence

FIELD_PHRASES = {"service_date": ["日期", "哪天", "服务日期"], "start_time": ["时间", "几点", "开始时间"], "people": ["人数", "几个人"], "address": ["地址", "服务地址", "哪里"]}
REPLY_EXPECTATION_KEYS = {
    "asks_confirmation_for",
    "asks_missing_info",
    "handoff",
    "mentions_candidates",
    "mentions_missing_info",
    "mentions_no_match",
    "mentions_out_of_service_area",
    "mentions_requery_needed",
    "mentions_tool_error",
    "mentions_unavailable",
    "no_success_claim",
    "preserves_candidate_order",
    "preserves_dietary",
    "rejects_unavailable_requested_chef",
    "rejects_unverified_selection",
    "requires_affirmative_allowlist",
    "uses_authorization_not_success_claim",
}
SUCCESS_CLAIM_PHRASES = ("预约成功", "预订成功", "下单成功", "订单已创建", "已经完成预订", "已创建订单")
UNAVAILABLE_PHRASES = ("不可用", "没空", "无法服务", "约不了")
HANDOFF_PHRASES = ("不是上门私厨预约", "切换到相应服务", "转人工", "人工客服", "转接")
REQUERY_PHRASES = ("重新查询", "再查", "需要重新")
CONFIRMATION_PHRASES = ("请确认", "是否预约", "确认", "可以", "好的", "就这样", "确认预约")


def score_reply(reply: str | None, expectations: dict[str, object], evidence: ToolEvidence) -> float:
    if not expectations:
        return 1.0
    unknown = set(expectations) - REPLY_EXPECTATION_KEYS
    if unknown:
        raise ValueError(f"Unknown reply expectation keys: {', '.join(sorted(unknown))}")
    text = reply or ""
    checks: list[bool] = []
    missing = expectations.get("mentions_missing_info") or expectations.get("asks_missing_info")
    if isinstance(missing, list):
        checks.append(all(any(phrase in text for phrase in FIELD_PHRASES.get(str(field), [str(field)])) for field in missing))
    chef = expectations.get("asks_confirmation_for")
    if isinstance(chef, str):
        checks.append(chef in text and any(phrase in text for phrase in CONFIRMATION_PHRASES))
    candidates = expectations.get("mentions_candidates")
    if isinstance(candidates, list):
        checks.append(all(str(name) in text for name in candidates))
    unavailable = expectations.get("mentions_unavailable")
    if isinstance(unavailable, str):
        checks.append(unavailable in text and any(phrase in text for phrase in UNAVAILABLE_PHRASES))
    if expectations.get("mentions_no_match") is True:
        checks.append("没有匹配" in text or ("没有" in text and "厨师" in text))
    if expectations.get("mentions_out_of_service_area") is True:
        checks.append("服务范围" in text or "不在服务" in text)
    if expectations.get("mentions_tool_error") is True:
        checks.append("暂时不可用" in text or "稍后再试" in text or ("查询" in text and "不可用" in text))
    if expectations.get("mentions_requery_needed") is True:
        checks.append(any(phrase in text for phrase in REQUERY_PHRASES))
    if expectations.get("handoff") is True:
        checks.append(any(phrase in text for phrase in HANDOFF_PHRASES))
    if expectations.get("rejects_unverified_selection") is True:
        checks.append("已验证" in text or "候选" in text)
    if expectations.get("rejects_unavailable_requested_chef") is True:
        checks.append(any(phrase in text for phrase in UNAVAILABLE_PHRASES) and ("替代" in text or "候选" in text))
    if expectations.get("requires_affirmative_allowlist") is True:
        checks.append(any(phrase in text for phrase in CONFIRMATION_PHRASES))
    if expectations.get("no_success_claim") is True or expectations.get("uses_authorization_not_success_claim") is True:
        checks.append(not any(phrase in text for phrase in SUCCESS_CLAIM_PHRASES))
    dietary = expectations.get("preserves_dietary")
    if isinstance(dietary, list):
        checks.append(all(str(item) in text for item in dietary))
    order = expectations.get("preserves_candidate_order")
    if isinstance(order, list):
        checks.append(evidence.effective_candidate_order == [str(item) for item in order])
    return 1.0 if not checks else sum(1 for item in checks if item) / len(checks)
