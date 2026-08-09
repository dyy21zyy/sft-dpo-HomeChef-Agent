from pathlib import Path

from homechef_booking.contracts.manifest import (
    load_contract_manifest,
    normalized_utf8_lf_sha256,
    validate_manifest_source_hashes,
)


def test_manifest_loads_frozen_contract_with_normalized_hash_mode() -> None:
    manifest = load_contract_manifest(Path("contracts/contract_manifest.yaml"))
    assert manifest["contract_id"] == "homechef-booking-v1"
    assert manifest["status"] == "FROZEN"
    assert manifest["version"] == "v1"
    assert manifest["hash_mode"] == "normalized_utf8_lf"


def test_manifest_hashes_match_source_documents() -> None:
    manifest = load_contract_manifest(Path("contracts/contract_manifest.yaml"))
    assert validate_manifest_source_hashes(manifest, Path(".")) == []


def test_normalized_hash_treats_crlf_and_lf_as_equal() -> None:
    assert normalized_utf8_lf_sha256(b"a\r\nb\r\n") == (
        normalized_utf8_lf_sha256(b"a\nb\n")
    )
