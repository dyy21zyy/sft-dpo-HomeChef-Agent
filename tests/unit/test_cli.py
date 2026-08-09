"""Task 8: CLI entry point tests."""

import subprocess
from pathlib import Path


def test_homechef_eval_help():
    result = subprocess.run(["uv", "run", "homechef-eval", "--help"], capture_output=True, text=True, cwd=Path.cwd())
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_homechef_eval_mock_run():
    result = subprocess.run(["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase01_mock.yaml"], capture_output=True, text=True, cwd=Path.cwd())
    assert result.returncode == 0
    assert "Total cases" in result.stdout
    assert "19" in result.stdout
