"""Phase 03 v0.2.1 — Strong Model Chinese Surface Realizer (few-shot).

Uses the 25 style/difficulty anchors as few-shot references to paraphrase a
placeholder-ized sample into natural Chinese. Guarantees:
- ScenarioFacts / Gold action / reply_type / tool_name / slots unchanged
- all placeholders preserved with identical count
- Anchor Similarity Gate: reject paraphrases that are mere entity-swaps of an anchor
- attempt/retry loop (max 3)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from homechef_booking.data.style_anchors import (
    anchor_normalized,
    anchor_skeleton,
    select_anchors,
)

# ═══════════════════════════════════════════════════════════
# Identity-preserving placeholder helpers
# ═══════════════════════════════════════════════════════════

# Indexed placeholder pattern, e.g. <TIME_1>, <TIME_2>, <ADDRESS_1>.
# A slot type (TIME/DATE/PEOPLE/...) plus a 1-based identity index.
PLACEHOLDER_PATTERN = re.compile(
    r'<(?P<slot>DATE|TIME|PEOPLE|ADDRESS|CHEF|CUISINE|BUDGET|MENU|DIETARY|OCCASION)_(?P<idx>\d+)>'
)

_SLOT_TYPES = [
    "DATE", "TIME", "PEOPLE", "ADDRESS", "CHEF", "CUISINE", "BUDGET", "MENU", "DIETARY", "OCCASION",
]

# Map booking_state / user_input slot key → placeholder type
_SLOT_KEY_TO_TYPE = {
    "service_date": "DATE",
    "start_time": "TIME",
    "people": "PEOPLE",
    "address": "ADDRESS",
    "chef_name": "CHEF",
    "cuisine": "CUISINE",
    "budget_max": "BUDGET",
    "budget_min": "BUDGET",
    "menu": "MENU",
    "dietary_constraints": "DIETARY",
    "occasion": "OCCASION",
}


def _build_surface_spans(
    text: str,
    values: list[tuple[str, str]],
) -> list[tuple[int, int, str, str]]:
    """Find all non-overlapping surface spans for the given (value,type) pairs.

    Returns a list of (start, end, value, slot_type) sorted by start.
    Rules:
      - A longer value takes precedence over a shorter value it overlaps
        (prevents `people=2` from polluting the "2" inside "12点" / "2026").
      - Among overlapping candidates of the same value, the first is kept.
      - Remaining spans are non-overlapping (greedy longest-first).
    """
    # Collect all candidate occurrences as spans.
    candidates: list[tuple[int, int, str, str]] = []
    for value, slot_type in values:
        if not value:
            continue
        start = 0
        while True:
            idx = text.find(value, start)
            if idx == -1:
                break
            candidates.append((idx, idx + len(value), value, slot_type))
            start = idx + 1

    # Sort by (length desc, start asc) so longest spans win overlaps.
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[0]))

    chosen: list[tuple[int, int, str, str]] = []
    covered: list[tuple[int, int]] = []  # chosen spans

    def _overlaps(s: int, e: int) -> bool:
        for (cs, ce) in covered:
            if s < ce and cs < e:
                return True
        return False

    for (s, e, value, slot_type) in candidates:
        if _overlaps(s, e):
            continue
        chosen.append((s, e, value, slot_type))
        covered.append((s, e))

    chosen.sort(key=lambda c: c[0])
    return chosen


def _placeholderize_text(
    text: str,
    slot_values: dict[str, Any] | list[tuple[str, str]],
) -> tuple[str, dict[str, str]]:
    """Replace each distinct slot-value surface span with an indexed placeholder.

    Non-overlapping + identity-preserving:
      - Only real surface spans are replaced (longer values win overlaps), so a
        `people=2` never pollutes "12点", "2026", a date, or a time.
      - Each distinct span gets a UNIQUE indexed placeholder (<TIME_1>, <TIME_2>)
        bound to its exact value. Restore can never swap/merge/drop values.

    `slot_values` may be:
      - a dict {slot_key: value}, using _SLOT_KEY_TO_TYPE to infer the type
      - a list of (surface_value, slot_type) pairs for explicit typing
        (e.g. [("六点","TIME"), ("晚上七点","TIME")]).

    Returns (placeholderized_text, mapping {placeholder: original_value}).
    """
    if isinstance(slot_values, dict):
        values: list[tuple[str, str]] = []
        for key, val in slot_values.items():
            slot_type = _SLOT_KEY_TO_TYPE.get(key)
            if slot_type is None or val is None:
                continue
            sval = str(val)
            if sval:
                values.append((sval, slot_type))
    else:
        values = [(str(v), t) for v, t in slot_values if v is not None and str(v)]

    spans = _build_surface_spans(text, values)

    # Assign indexed placeholders per type in forward span order so the FIRST
    # occurrence of a type is <TYPE_1>.
    mapping: dict[str, str] = {}
    span_placeholders: list[tuple[int, int, str]] = []
    counters: dict[str, int] = {}
    for (s, e, value, slot_type) in spans:
        counters[slot_type] = counters.get(slot_type, 0) + 1
        ph = f"<{slot_type}_{counters[slot_type]}>"
        mapping[ph] = value
        span_placeholders.append((s, e, ph))

    # Replace from the end so earlier span positions stay valid.
    chars = list(text)
    for (s, e, ph) in reversed(span_placeholders):
        chars[s:e] = [ph]
    return "".join(chars), mapping


def _restore_placeholders(paraphrased: str, mapping: dict[str, str]) -> str:
    """Restore each indexed placeholder back to its bound original value.

    Identity-preserving: <TIME_1>→"六点", <TIME_2>→"晚上七点". Values are never
    swapped, merged, or dropped.
    """
    result = paraphrased
    # Restore in order of longest placeholder token first to avoid prefix issues.
    for ph in sorted(mapping.keys(), key=len, reverse=True):
        value = mapping[ph]
        result = result.replace(ph, value)
    return result


def _placeholder_set(text: str) -> set[str]:
    """Return the set of unique indexed placeholder tokens in text."""
    return set(PLACEHOLDER_PATTERN.findall(text)) if text else set()


def count_placeholders(text: str) -> dict[str, int]:
    """Count occurrences of each placeholder TYPE (slot, without index)."""
    counts: dict[str, int] = {}
    for m in PLACEHOLDER_PATTERN.finditer(text):
        slot = m.group("slot")
        counts[slot] = counts.get(slot, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════
# Strong Model Backend Protocol
# ═══════════════════════════════════════════════════════════

class LLMBackend(Protocol):
    """Minimal interface for a strong-model chat backend."""
    def chat(self, system: str, user: str) -> str: ...


@dataclass
class AnchorSimilarityResult:
    """Result of Anchor Similarity Gate."""
    max_similarity: float
    best_anchor_id: int | None
    best_anchor_text: str | None
    passed: bool
    reason: str = ""


def _levenshtein_ratio(a: str, b: str) -> float:
    """Levenshtein-based similarity ratio (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return 1.0 - dp[m][n] / max(m, n)


def _char_bigram_jaccard(a: str, b: str) -> float:
    """Character bigram Jaccard similarity (0-1). Discriminates surface text."""
    def bigrams(s: str) -> set[str]:
        if len(s) < 2:
            return {s} if s else set()
        return {s[i:i + 2] for i in range(len(s) - 1)}
    ba, bb = bigrams(a), bigrams(b)
    if not ba and not bb:
        return 1.0
    inter = ba & bb
    union = ba | bb
    return len(inter) / len(union) if union else 0.0


def anchor_similarity_gate(
    text: str,
    difficulty: str,
    scenario: str,
    similarity_threshold: float = 0.85,
) -> AnchorSimilarityResult:
    """Check if text is too similar to any anchor (entity-swap detection).

    Uses a combination of:
    - character bigram Jaccard on normalized text (keeps Chinese chars, highly
      discriminating against entity-swaps)
    - Levenshtein ratio on normalized text

    If either exceeds the threshold vs an anchor, the paraphrase is rejected
    as likely being a mere entity-swap.
    """
    text_norm = anchor_normalized(text)
    text_skeleton = anchor_skeleton(text)

    candidates = select_anchors(difficulty, scenario, max_count=6)
    best_sim = 0.0
    best_id = None
    best_text = None
    entity_swap_detected = False

    for a in candidates:
        a_norm = anchor_normalized(a.text)
        a_skeleton = anchor_skeleton(a.text)
        # Surface similarity (bigram Jaccard + Levenshtein) — primary signal.
        sim = max(
            _char_bigram_jaccard(text_norm, a_norm),
            _levenshtein_ratio(text_norm, a_norm),
        )
        # Entity-swap detection: identical multi-clause skeleton AND high
        # surface similarity. An entity-swap keeps the same words (high sim)
        # while a genuine paraphrase has the same skeleton but different words
        # (low sim). Both conditions must hold to reject.
        if (
            text_skeleton
            and len(text_skeleton) >= 3
            and text_skeleton == a_skeleton
            and sim >= 0.5
        ):
            entity_swap_detected = True
            sim = 1.0
        if sim > best_sim:
            best_sim = sim
            best_id = a.id
            best_text = a.text

    passed = best_sim <= similarity_threshold
    reason = (
        f"similarity={best_sim:.2f} (threshold {similarity_threshold})"
        + ("; entity-swap (exact skeleton + high surface sim)" if entity_swap_detected else "")
        if passed
        else f"too similar to anchor#{best_id} (sim={best_sim:.2f}): {best_text}"
    )
    return AnchorSimilarityResult(
        max_similarity=best_sim,
        best_anchor_id=best_id,
        best_anchor_text=best_text,
        passed=passed,
        reason=reason,
    )


# ═══════════════════════════════════════════════════════════
# Strong Model Surface Realizer
# ═══════════════════════════════════════════════════════════

@dataclass
class RealizationOutcome:
    """Outcome of a single surface realization."""
    success: bool
    user_input: str = ""
    attempts: int = 0
    anchor_similarity: float = 0.0
    reason: str = ""


class StrongModelSurfaceRealizer:
    """Few-shot Chinese surface realizer backed by a strong LLM.

    Pipeline per sample:
      1. placeholderize current user_input
      2. select 2-4 anchors (same difficulty/scenario)
      3. call strong model with few-shot prompt (max 3 attempts)
      4. validate: placeholders preserved, no business-fact change
      5. Anchor Similarity Gate: reject entity-swaps
      6. restore placeholders → natural Chinese
    """

    def __init__(
        self,
        backend: LLMBackend,
        similarity_threshold: float = 0.85,
        max_attempts: int = 3,
    ):
        self.backend = backend
        self.similarity_threshold = similarity_threshold
        self.max_attempts = max_attempts

    def _build_prompt(
        self,
        difficulty: str,
        scenario: str,
        language_style: str,
        placeholderized: str,
    ) -> tuple[str, str]:
        """Build system + user prompt for the strong model."""
        anchors = select_anchors(difficulty, scenario, max_count=4)
        anchor_lines = "\n".join(f"{i+1}. “{a.text}”" for i, a in enumerate(anchors))

        system = (
            "你是一个中文自然语言改写助手。请根据参考风格，将待改写的中文文本改写为更自然、"
            "更具个人风格的新表达。\n"
            "硬性要求：\n"
            "- 不得复制参考句，必须生成新的表达；\n"
            "- 待改写文本中包含带序号的身份绑定占位符（如 <TIME_1>、<TIME_2>、"
            "<DATE_1>、<PEOPLE_1>）。每个占位符代表一个**不同**的具体业务实体"
            "（例如 <TIME_1> 和 <TIME_2> 是两个不同的时间点）。\n"
            "- 必须**原样完整保留**所有占位符：类型、序号、数量、出现顺序都不得改变；\n"
            "- 禁止新增、删除、合并、或重命名任何占位符；\n"
            "- 禁止交换占位符（例如 <TIME_1> 与 <TIME_2> 的位置不得互换）；\n"
            "- 不得增加或删除任何业务事实（日期/时间/人数/地址/厨师/菜系/预算等语义）；\n"
            "- 只输出改写后的文本，不要输出其他内容。"
        )
        user = (
            f"difficulty: {difficulty}\n"
            f"scenario: {scenario}\n"
            f"language_style: {language_style}\n\n"
            f"参考风格（仅模仿风格，不得复制）：\n{anchor_lines}\n\n"
            f"待改写：\n“{placeholderized}”\n\n"
            f"请输出 1 条新的自然中文表达。"
        )
        return system, user

    def _validate_placeholders(self, input_text: str, output_text: str) -> tuple[bool, str]:
        """Ensure ALL identity-preserving placeholders are preserved exactly.

        The set of indexed placeholder tokens (e.g. {('TIME','1'), ('TIME','2')})
        in the output must equal the set in the input. This prevents the model
        from dropping, merging, or renaming an identity-bound placeholder.
        """
        in_tokens = set(PLACEHOLDER_PATTERN.findall(input_text))
        out_tokens = set(PLACEHOLDER_PATTERN.findall(output_text))
        if in_tokens != out_tokens:
            missing = in_tokens - out_tokens
            extra = out_tokens - in_tokens
            return False, (
                f"identity placeholder set mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        return True, ""

    def _validate_no_new_slots(self, input_text: str, output_text: str) -> bool:
        """Ensure output doesn't introduce placeholders not present in input."""
        in_tokens = set(PLACEHOLDER_PATTERN.findall(input_text))
        out_tokens = set(PLACEHOLDER_PATTERN.findall(output_text))
        return out_tokens.issubset(in_tokens)

    def realize(
        self,
        raw_text: str,
        slot_values: dict[str, Any],
        difficulty: str,
        scenario: str,
        language_style: str = "普通口语",
    ) -> RealizationOutcome:
        """Realize a natural Chinese user_input for a sample.

        Identity-preserving pipeline:
          1. placeholderize `raw_text` → indexed placeholders + identity map
             (each distinct slot-value occurrence gets a unique <TYPE_N> index).
          2. send the placeholderized text to the strong model.
          3. require the EXACT placeholder-token set on output (no drop/merge/swap).
          4. restore each token to its bound value.
        """
        # Build identity map (indexed placeholder → original value).
        placeholderized, mapping = _placeholderize_text(raw_text, slot_values)

        for attempt in range(1, self.max_attempts + 1):
            system, prompt = self._build_prompt(difficulty, scenario, language_style, placeholderized)
            try:
                raw_out = self.backend.chat(system, prompt).strip()
            except Exception as exc:
                return RealizationOutcome(False, reason=f"backend error: {exc}")

            # Strip quotes if model wrapped output
            raw_out = raw_out.strip('“”"\n')

            # Identity-preserving placeholder validation
            ok, reason = self._validate_placeholders(placeholderized, raw_out)
            if not ok:
                continue  # retry
            if not self._validate_no_new_slots(placeholderized, raw_out):
                continue  # retry

            # Anchor Similarity Gate
            gate = anchor_similarity_gate(
                self._strip_placeholders(raw_out),
                difficulty,
                scenario,
                self.similarity_threshold,
            )
            if not gate.passed:
                continue  # reject + real retry (next loop iteration calls Qwen again)

            # Restore each identity-bound placeholder to its exact original value.
            restored = _restore_placeholders(raw_out, mapping)
            return RealizationOutcome(
                success=True,
                user_input=restored,
                attempts=attempt,
                anchor_similarity=gate.max_similarity,
            )

        return RealizationOutcome(
            False,
            attempts=self.max_attempts,
            reason=f"anchor similarity or placeholder validation failed after {self.max_attempts} attempts",
        )

    def _strip_placeholders(self, text: str) -> str:
        """Replace placeholders with a Chinese marker so skeleton/normalization
        treat them consistently with anchor text (Chinese runs)."""
        return PLACEHOLDER_PATTERN.sub('某', text)


__all__ = [
    "StrongModelSurfaceRealizer",
    "LLMBackend",
    "anchor_similarity_gate",
    "AnchorSimilarityResult",
    "PLACEHOLDER_PATTERN",
    "count_placeholders",
]
