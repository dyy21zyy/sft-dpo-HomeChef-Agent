"""Task 7 Phase 02: CLI compatibility tests for Phase 02 benchmark commands."""

import subprocess
import sys
from pathlib import Path


def test_homechef_eval_phase01_still_works():
    env = {"PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase01_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(),
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout
    assert "19" in result.stdout


def test_homechef_eval_phase02_frozen_mock_works():
    env = {"PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase02_frozen_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(),
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout


def test_homechef_eval_phase02_diag_mock_works():
    env = {"PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase02_diag_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(),
        env={**__import__("os").environ, **env},
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout
