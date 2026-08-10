"""Phase 03 contamination check against Frozen Test and Diagnostic Dev."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from homechef_booking.data.raw_sample import parse_raw_sample_line
from homechef_booking.evaluation.sample import load_eval_cases
from homechef_booking.prompts import PromptBuilder


def canonical_training_fingerprint(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_eval_fingerprint_set(paths: list[Path]) -> set[str]:
    fingerprints: set[str] = set()
    for path in paths:
        for case in load_eval_cases(path):
            prompt = PromptBuilder().build_messages(case.input)
            fingerprints.add(canonical_training_fingerprint(prompt))
            fingerprints.add(canonical_training_fingerprint(case.expected.model_dump(mode="json", exclude_none=False)))
    return fingerprints


class ContaminationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    raw_path: str
    raw_count: int
    eval_paths: list[str] = Field(default_factory=list)
    overlap_count: int = 0
    overlapping_ids: list[str] = Field(default_factory=list)


def check_raw_contamination(raw_path: Path, eval_paths: list[Path]) -> ContaminationReport:
    eval_fingerprints = build_eval_fingerprint_set(eval_paths)
    overlapping: list[str] = []
    raw_count = 0
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw_count += 1
        try:
            sample = parse_raw_sample_line(line)
        except Exception:
            continue
        prompt = PromptBuilder().build_messages(sample.input)
        raw_fp = canonical_training_fingerprint(prompt)
        expected_fp = canonical_training_fingerprint(sample.expected.model_dump(mode="json", exclude_none=False))
        if raw_fp in eval_fingerprints or expected_fp in eval_fingerprints:
            overlapping.append(sample.id)
    return ContaminationReport(
        raw_path=str(raw_path),
        raw_count=raw_count,
        eval_paths=[str(p) for p in eval_paths],
        overlap_count=len(overlapping),
        overlapping_ids=overlapping,
    )
