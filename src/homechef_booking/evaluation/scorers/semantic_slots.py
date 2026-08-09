from __future__ import annotations

import re
import unicodedata
from typing import Protocol

SEMANTIC_SLOT_FIELDS = {"cuisine", "menu", "dietary_constraints", "occasion"}


class TextEmbedder(Protocol):
    def similarity(self, left: str, right: str) -> float:
        raise NotImplementedError


class Phase01DeterministicEmbedder:
    def similarity(self, left: str, right: str) -> float:
        equivalent_pairs = {
            frozenset({"不吃花生", "花生过敏"}),
            frozenset({"川菜", "四川菜"}),
            frozenset({"生日", "生日宴"}),
        }
        if left == right:
            return 1.0
        if frozenset({left, right}) in equivalent_pairs:
            return 0.91
        return 0.0


def normalize_slot_text(text: str, aliases: dict[str, str] | None = None) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = re.sub(r"[\s,，。.!！?？、]+", "", normalized)
    return (aliases or {}).get(normalized, normalized)


def semantic_slot_f1(expected: list[str], predicted: list[str], embedder: TextEmbedder, aliases: dict[str, str] | None = None, threshold: float = 0.70) -> dict[str, float]:
    expected_values = [normalize_slot_text(value, aliases) for value in expected]
    predicted_values = [normalize_slot_text(value, aliases) for value in predicted]
    if not expected_values and not predicted_values:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    matches: list[tuple[float, int, int]] = []
    for expected_index, expected_value in enumerate(expected_values):
        for predicted_index, predicted_value in enumerate(predicted_values):
            if _reversed_polarity(expected_value, predicted_value):
                continue
            similarity = 1.0 if expected_value == predicted_value else embedder.similarity(expected_value, predicted_value)
            if similarity >= threshold:
                matches.append((similarity, expected_index, predicted_index))
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    for _similarity, expected_index, predicted_index in sorted(matches, reverse=True):
        if expected_index not in used_expected and predicted_index not in used_predicted:
            used_expected.add(expected_index)
            used_predicted.add(predicted_index)
    precision = len(used_predicted) / len(predicted_values) if predicted_values else 0.0
    recall = len(used_expected) / len(expected_values) if expected_values else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _reversed_polarity(left: str, right: str) -> bool:
    left_negative = any(marker in left for marker in ("不吃", "不要", "不能", "过敏", "忌"))
    right_positive = any(marker in right for marker in ("可以吃", "能吃", "接受"))
    return left_negative and right_positive
