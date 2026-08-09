"""Contract manifest and hash validation utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def normalized_utf8_lf_sha256(data: bytes) -> str:
    """Compute SHA256 of *data* after normalizing CRLF to LF."""
    normalized = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def load_contract_manifest(path: Path) -> dict[str, Any]:
    """Load the contract manifest YAML and return it as a dict."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be a mapping: {path}")
    return data


def validate_manifest_source_hashes(
    manifest: dict[str, Any],
    project_root: Path,
) -> list[str]:
    """Validate that source document hashes match the manifest.

    Returns a list of mismatch error strings. Empty list means all hashes match.
    """
    errors: list[str] = []
    source_docs = manifest.get("source_documents", {})
    for doc_name, doc_meta in source_docs.items():
        doc_path = project_root / doc_name
        if not doc_path.exists():
            errors.append(f"Source document not found: {doc_name}")
            continue
        expected_hash = doc_meta["sha256"] if isinstance(doc_meta, dict) else doc_meta
        actual_hash = normalized_utf8_lf_sha256(doc_path.read_bytes())
        if actual_hash != expected_hash:
            errors.append(
                f"Hash mismatch for {doc_name}: expected {expected_hash}, got {actual_hash}"
            )
    return errors


__all__ = [
    "normalized_utf8_lf_sha256",
    "load_contract_manifest",
    "validate_manifest_source_hashes",
]
