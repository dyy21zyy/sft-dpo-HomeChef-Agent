"""Phase 02 suite manifest and integrity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class SuiteManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    suite_id: str
    contract_id: str
    case_file: str
    case_count: int
    sha256: str
    frozen: bool = False
    versioned: bool = False
    created_at: str | None = None
    approved_by: str | None = None
    notes: str | None = None


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_suite_manifest(path: Path) -> SuiteManifest:
    return SuiteManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_suite_manifest(
    manifest_path: Path,
    root: Path | None = None,
    resolve_case_file: Path | None = None,
    require_frozen: bool = False,
) -> SuiteManifest:
    manifest = load_suite_manifest(manifest_path)
    case_file_path = resolve_case_file or (root / manifest.case_file if root else Path(manifest.case_file))
    if not case_file_path.exists():
        raise ValueError(f"Case file not found: {case_file_path}")
    actual_lines = len([line for line in case_file_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    if manifest.case_count != actual_lines:
        raise ValueError(f"case_count mismatch: manifest says {manifest.case_count}, file has {actual_lines}")
    actual_sha = compute_sha256(case_file_path)
    if manifest.sha256 != actual_sha:
        raise ValueError(f"sha256 mismatch: manifest says {manifest.sha256}, file has {actual_sha}")
    if require_frozen and not manifest.frozen:
        raise ValueError(f"Suite {manifest.suite_id} requires frozen: true but got frozen: {manifest.frozen}")
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = validate_suite_manifest(Path(args.manifest), root=Path(args.root))
    print(f"Suite {result.suite_id}: case_count={result.case_count}, sha256={result.sha256[:12]}..., frozen={result.frozen}")
