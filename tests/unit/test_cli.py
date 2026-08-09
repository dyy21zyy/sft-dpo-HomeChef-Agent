"""Task 8: CLI entry point tests."""

import subprocess
import sys
from pathlib import Path


def test_homechef_eval_help():
    env = {"PYTHONIOENCODING": "utf-8", "PATH": sys.prefix + ";" + (Path.cwd() / ".venv" / "Scripts").as_posix()}
    result = subprocess.run(["uv", "run", "homechef-eval", "--help"], capture_output=True, text=True, cwd=Path.cwd(), env={**__import__("os").environ, **env})
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_homechef_eval_mock_run():
    env = {"PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase01_mock.yaml"], capture_output=True, text=True, cwd=Path.cwd(), env={**__import__("os").environ, **env})
    assert result.returncode == 0
    assert "Total cases" in result.stdout
    assert "19" in result.stdout
