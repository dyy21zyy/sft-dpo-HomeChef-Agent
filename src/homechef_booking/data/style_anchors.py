"""Phase 03 v0.2.1 — Official Approved Style/Difficulty Anchors.

These 25 anchors were produced by the Qwen strong-model surface realizer and
human-reviewed/approved (reports/generated/phase03/v0.2.1/
qwen_surface_preview_25_revised.json). They are the official Chinese
style/difficulty anchors for Strong Model Surface Realization.

Distribution: Easy 8 / Medium 9 / Hard 8.

Usage:
  - `select_anchors(difficulty, scenario, max_count)` MUST return 2-4 anchors,
    preferring same-scenario, then same-difficulty. Never returns 0.
  - Anchors are style references ONLY — never copied verbatim into generated
    600 Raw (unless an independently-generated sample passes all data gates).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StyleAnchor:
    """A single approved style/difficulty anchor."""
    id: int
    difficulty: str  # easy | medium | hard
    scenario: str
    text: str
    language_style: str = ""


# 25 official approved anchors (Easy 8 / Medium 9 / Hard 8).
# Source: qwen_surface_preview_25_revised.json (human-approved).
STYLE_ANCHORS: tuple[StyleAnchor, ...] = (
    # ── Easy (8) ──
    StyleAnchor(1, "easy", "missing_required_slots", "想请个厨师上门做川菜，4个人，地址在上海市徐汇区。", "自然口语"),
    StyleAnchor(2, "easy", "missing_required_slots", "这周六想在北京市朝阳区请个师傅来家里做饭，一共6个人，想吃粤菜。", "自然口语"),
    StyleAnchor(3, "easy", "missing_required_slots", "明天晚上七点，地点在杭州市西湖区，想找个会做江浙菜的师傅。", "简洁口语"),
    StyleAnchor(4, "easy", "missing_required_slots", "后天晚上六点半，5个人，预算800元左右，想吃家常菜。", "自然口语"),
    StyleAnchor(5, "easy", "valid_search_tool_call", "8月20日晚上六点，4个人，在南京市鼓楼区，预算600元左右，想吃淮扬菜。", "信息完整"),
    StyleAnchor(6, "easy", "specific_chef", "我想约张师傅，周五晚上七点，3个人，地址在深圳市南山区。", "直接请求"),
    StyleAnchor(7, "easy", "dietary_preservation", "明晚六点，4个人，在苏州市吴中区，其中有一位客人花生过敏。", "补充约束"),
    StyleAnchor(8, "easy", "explicit_confirmation", "可以，就订李师傅吧。", "简短确认"),
    # ── Medium (9) ──
    StyleAnchor(9, "medium", "relative_time", "这周六晚上想找个做粤菜的师傅，6个人，地址在广州市天河区。", "相对时间"),
    StyleAnchor(10, "medium", "relative_time", "下周末家里聚餐，大概8个人，预算1000元左右，想吃江浙菜。", "相对时间+预算"),
    StyleAnchor(11, "medium", "state_inheritance", "地址改成上海市浦东新区世纪大道附近，其他条件都不变。", "修改地址"),
    StyleAnchor(12, "medium", "state_inheritance", "六点还是有点早，改到晚上七点吧，其他安排照旧。", "修改时间"),
    StyleAnchor(13, "medium", "state_inheritance", "刚确认了一下，不是4个人，是6个人，其他都按之前的来。", "修改人数"),
    StyleAnchor(14, "medium", "dietary_preservation", "对了，再补充一下，有位客人不吃香菜，而且要少辣。", "追加忌口"),
    StyleAnchor(15, "medium", "modification_requires_requery", "川菜先不要了，换成粤菜吧，时间和地址都不变。", "修改菜系"),
    StyleAnchor(16, "medium", "tool_result", "王师傅如果没空，就帮我看看同一时间还有没有其他做家常菜的师傅。", "不可用后继续搜索"),
    StyleAnchor(17, "medium", "candidate_selection", "第二位陈师傅看起来挺合适的，就选他吧。", "候选选择"),
    # ── Hard (8) ──
    StyleAnchor(18, "hard", "relative_time_correction", "不是明天，我刚才说错了，是后天晚上七点，其他条件别改。", "时间纠正"),
    StyleAnchor(19, "hard", "modification_requires_requery", "地址换到杭州市滨江区，人数还是6个，预算和忌口都按之前的，重新帮我查一下。", "地址修改+继承"),
    StyleAnchor(20, "hard", "tool_result_requery", "张师傅周六有空是吧？那我改成周日晚上，还能约他吗？", "查询后改日期"),
    StyleAnchor(21, "hard", "tool_result_dietary_requery", "先别订，刚知道有位客人海鲜过敏。其他条件都不变，麻烦重新帮我看看合适的师傅。", "新增忌口后重查"),
    StyleAnchor(22, "hard", "multi_field_modification", "时间从六点改到七点半，人数改成8个，预算提高到1200元，菜系还是粤菜。", "多字段修改"),
    StyleAnchor(23, "hard", "candidate_rejection_research", "这几个师傅都不太合适，还有别的吗？时间、地点和预算都不用改。", "拒绝候选后重搜"),
    StyleAnchor(24, "hard", "confirmation_reversal", "先别确认预约，我想把地址换到上海市长宁区，再帮我确认一下这个师傅还能不能上门。", "确认前反悔"),
    StyleAnchor(25, "hard", "multi_constraint_relative_time", "下周六给老人过生日，7个人，在成都市武侯区，想吃清淡一点的家常菜，有人乳糖不耐，也不能吃太辣，预算控制在1000元以内。", "复杂综合"),
)

# Index by difficulty and scenario for few-shot selection.
_ANCHORS_BY_DIFFICULTY: dict[str, list[StyleAnchor]] = {}
_ANCHORS_BY_SCENARIO: dict[str, list[StyleAnchor]] = {}
for _a in STYLE_ANCHORS:
    _ANCHORS_BY_DIFFICULTY.setdefault(_a.difficulty, []).append(_a)
    _ANCHORS_BY_SCENARIO.setdefault(_a.scenario, []).append(_a)

# Official distribution.
DIFFICULTY_DISTRIBUTION: dict[str, int] = {
    "easy": len(_ANCHORS_BY_DIFFICULTY.get("easy", [])),
    "medium": len(_ANCHORS_BY_DIFFICULTY.get("medium", [])),
    "hard": len(_ANCHORS_BY_DIFFICULTY.get("hard", [])),
}


def select_anchors(difficulty: str, scenario: str, max_count: int = 4) -> list[StyleAnchor]:
    """Select 2-4 anchors for few-shot prompting.

    Priority:
      1. same scenario
      2. same difficulty
      3. any remaining anchor (same difficulty first)

    Guarantees 2 <= len(result) <= max_count (never empty).
    """
    if max_count < 2:
        max_count = 2
    selected: list[StyleAnchor] = []
    seen: set[int] = set()

    # 1. Same scenario.
    for _a in _ANCHORS_BY_SCENARIO.get(scenario, []):
        if len(selected) >= max_count:
            break
        if _a.id not in seen:
            selected.append(_a)
            seen.add(_a.id)

    # 2. Same difficulty.
    if len(selected) < max_count:
        for _a in _ANCHORS_BY_DIFFICULTY.get(difficulty, []):
            if len(selected) >= max_count:
                break
            if _a.id not in seen:
                selected.append(_a)
                seen.add(_a.id)

    # 3. Any remaining (guarantee >= 2).
    if len(selected) < 2:
        for _a in STYLE_ANCHORS:
            if len(selected) >= max_count:
                break
            if _a.id not in seen:
                selected.append(_a)
                seen.add(_a.id)

    return selected[:max_count]


def anchor_skeleton(text: str) -> str:
    """Skeleton normalization for similarity gate (Chinese chars → X)."""
    import re
    skel = re.sub(r'[\u4e00-\u9fff]+', 'X', text)
    skel = re.sub(r'\d+', 'N', skel)
    return skel


def anchor_normalized(text: str) -> str:
    """Normalize by removing punctuation and whitespace."""
    import re
    norm = re.sub(r'[，。！？、；：\s]', '', text)
    return norm


__all__ = [
    "StyleAnchor",
    "STYLE_ANCHORS",
    "DIFFICULTY_DISTRIBUTION",
    "select_anchors",
    "anchor_skeleton",
    "anchor_normalized",
]
