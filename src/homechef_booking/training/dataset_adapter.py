"""Phase 04 Task 2 — Dataset adapter and training preflight validators.

Validates Phase 03 SFT and DPO datasets are trainable inputs and rejects
eval suites (Frozen Test, Diagnostic Dev) and Phase 02 benchmark outputs.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from homechef_booking.data.dataset_schema import DpoPair, SftSample

# Row IDs containing any of these substrings are forbidden as training data
_FORBIDDEN_ID_SUBSTRINGS = [
    "frozen",
    "diagnostic",
    "missing_mock_prediction",
    "phase02",
    "base_benchmark",
]


class DatasetPreflightReport(BaseModel):
    """Result of a training dataset preflight validation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    dataset_type: str = Field(description="'sft' or 'dpo'")
    total_rows: int = Field(ge=0)
    expected_count: int = Field(ge=0)
    passed: bool
    errors: list[str] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Convenience: number of validation errors."""
        return len(self.errors)


def _is_forbidden_id(row_id: str) -> bool:
    """Check if a row ID contains any forbidden substring."""
    rid_lower = row_id.lower()
    for substr in _FORBIDDEN_ID_SUBSTRINGS:
        if substr in rid_lower:
            return True
    return False


def validate_sft_for_training(path: Path, expected_count: int) -> DatasetPreflightReport:
    """Validate an SFT JSONL file for training readiness.

    Checks:
    - Every row is a valid SftSample.
    - No row ID contains forbidden substrings (frozen, diagnostic, etc.).
    - Total row count matches expected_count.
    """
    errors: list[str] = []
    total = 0

    if not path.exists():
        return DatasetPreflightReport(
            dataset_type="sft",
            total_rows=0,
            expected_count=expected_count,
            passed=False,
            errors=[f"File not found: {path}"],
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Row {total}: invalid JSON — {e}")
            continue
        try:
            sample = SftSample(**row)
        except ValidationError as e:
            errors.append(f"Row {total}: SftSample validation error — {e}")
            continue
        if _is_forbidden_id(sample.id):
            errors.append(
                f"Row {total} ({sample.id}): forbidden ID — contains eval-only or benchmark substring."
            )

    passed = total == expected_count and len(errors) == 0
    if total != expected_count:
        errors.append(
            f"Count mismatch: got {total} rows, expected {expected_count}."
        )

    return DatasetPreflightReport(
        dataset_type="sft",
        total_rows=total,
        expected_count=expected_count,
        passed=passed,
        errors=errors,
    )


def validate_dpo_for_training(path: Path, expected_count: int) -> DatasetPreflightReport:
    """Validate a DPO JSONL file for training readiness.

    Checks:
    - Every row is a valid DpoPair.
    - No row ID contains forbidden substrings (frozen, diagnostic, etc.).
    - Total row count matches expected_count.
    """
    errors: list[str] = []
    total = 0

    if not path.exists():
        return DatasetPreflightReport(
            dataset_type="dpo",
            total_rows=0,
            expected_count=expected_count,
            passed=False,
            errors=[f"File not found: {path}"],
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Row {total}: invalid JSON — {e}")
            continue
        try:
            pair = DpoPair(**row)
        except ValidationError as e:
            errors.append(f"Row {total}: DpoPair validation error — {e}")
            continue
        if _is_forbidden_id(pair.id):
            errors.append(
                f"Row {total} ({pair.id}): forbidden ID — contains eval-only or benchmark substring."
            )

    passed = total == expected_count and len(errors) == 0
    if total != expected_count:
        errors.append(
            f"Count mismatch: got {total} rows, expected {expected_count}."
        )

    return DatasetPreflightReport(
        dataset_type="dpo",
        total_rows=total,
        expected_count=expected_count,
        passed=passed,
        errors=errors,
    )


# ── Shared SFT row -> LLaMA-Factory sharegpt converter ───────────────────────
# This is the SINGLE converter shared by the formal Phase04 derived-dataset
# builder and the CPU dry-run (scripts/train/dryrun.py). Do NOT write a third
# conversion logic.

_TOOL_RESULT_MARKER = "[TOOL_RESULT]"
_USER_FOLLOWUP_MARKER = "[USER_FOLLOWUP]"
_ASSISTANT_MARKER = "[ASSISTANT]"


def convert_sft_row_to_llamafactory_format(row: dict) -> dict:
    """Convert a canonical Phase03 SFT row into LLaMA-Factory 0.9.5 sharegpt.

    Deterministic context adaptation for tool-context samples:

        system
        user
        assistant(content="")     <- empty placeholder, DELETED
        tool(result)              <- folded into user as [TOOL_RESULT]
        user(followup)            <- folded into user as [USER_FOLLOWUP]
        assistant(target)         <- UNCHANGED final assistant target

    Output:
        system: <original system>
        user:   <original user>[TOOL_RESULT]<tool>[USER_FOLLOWUP]<followup>
        assistant: <final target>

    Constraints:
    - NEVER fabricates function_call / tool_calls from a tool result.
    - Tool result info is preserved (not lost).
    - Final assistant target is byte-identical to source.
    - Empty assistant placeholders are removed.
    - Rows WITHOUT tool history (system/user/assistant) keep equivalent
      training semantics.
    """
    messages = row.get("messages") or []

    # Final assistant target = last assistant message.
    last_asst_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_asst_idx = i
            break
    if last_asst_idx is None:
        target = ""
        ctx_msgs = messages
    else:
        target = messages[last_asst_idx].get("content", "")
        ctx_msgs = messages[:last_asst_idx]

    # Last system message (the booking contract) -> system role.
    system_content = None
    for m in ctx_msgs:
        if m.get("role") == "system":
            system_content = m.get("content", "")

    # Build the user context deterministically.
    user_parts: list[str] = []
    for m in ctx_msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            continue  # handled as the system message
        if role == "assistant" and content in ("", None):
            continue  # delete empty assistant placeholder
        if role == "tool":
            if isinstance(content, (dict, list)):
                content = json.dumps(content, ensure_ascii=False)
            user_parts.append(f"{_TOOL_RESULT_MARKER}\n{content}")
        elif role == "user":
            user_parts.append(f"{_USER_FOLLOWUP_MARKER}\n{content}")
        elif role == "assistant":
            # Non-empty intermediate assistant turn (rare) — keep it.
            user_parts.append(f"{_ASSISTANT_MARKER}\n{content}")
        else:
            user_parts.append(str(content))

    # First user message should not carry the [USER_FOLLOWUP] prefix (it is the
    # request, not a followup). Re-format: the first user part is the request.
    if user_parts and user_parts[0].startswith(_USER_FOLLOWUP_MARKER + "\n"):
        user_parts[0] = user_parts[0][len(_USER_FOLLOWUP_MARKER + "\n"):]

    user_content = "\n\n".join(p for p in user_parts if p.strip())

    out: dict = {"messages": []}
    if system_content:
        out["messages"].append({"role": "system", "content": system_content})
    out["messages"].append({"role": "user", "content": user_content})
    out["messages"].append({"role": "assistant", "content": target})
    return out


def build_sft_derived_dataset(
    canonical_path: Path,
    out_path: Path,
    manifest_path: Path | None = None,
) -> dict:
    """Build a 1:1 derived SFT dataset for LLaMA-Factory.

    Maps each canonical source row to EXACTLY ONE derived sharegpt row
    (no skip / drop / dedup / merge). Returns a small manifest dict:
    {"source_rows": N, "derived_rows": N, "mapping": [source_index] }
    """
    mapping: list[int] = []
    with canonical_path.open(encoding="utf-8") as fin, \
            out_path.open("w", encoding="utf-8") as fout:
        for source_idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            derived = convert_sft_row_to_llamafactory_format(row)
            fout.write(json.dumps(derived, ensure_ascii=False) + "\n")
            mapping.append(source_idx)

    manifest = {
        "canonical_path": str(canonical_path),
        "derived_path": str(out_path),
        "source_rows": len(mapping),
        "derived_rows": len(mapping),
        "mapping": mapping,  # source row index -> derived row index (same order)
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
