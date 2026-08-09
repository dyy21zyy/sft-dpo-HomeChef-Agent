from pathlib import Path

import homechef_booking


def test_package_imports() -> None:
    assert homechef_booking.__version__ == "0.0.0"


def test_pyproject_declares_base_config_without_cli() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in text
    assert 'testpaths = ["tests"]' in text
    assert "homechef-contract-validate" not in text


def test_gitignore_does_not_hide_data_or_reports_roots() -> None:
    lines = Path(".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" not in lines
    assert "reports/" not in lines
    assert "data/generated/" in lines
    assert "reports/generated/" in lines
