"""Task 7 Phase 02: CLI compatibility tests for Phase 02 benchmark commands."""

import os
import subprocess
import sys
from pathlib import Path


def _cli_env() -> dict:
    """Cross-platform environment for invoking the `uv` CLI.

    - Uses ``os.pathsep`` (Windows `;`, Linux `:`) — never a hardcoded separator.
    - Prepends ``Path(sys.executable).parent`` (Windows `Scripts/`, Linux `bin/`).
    - Preserves the original ``os.environ["PATH"]`` so the system PATH / current
      venv bin are not dropped.
    """
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PATH": os.pathsep.join(
            [
                str(Path(sys.executable).parent),
                os.environ.get("PATH", ""),
            ]
        ),
    }


def test_homechef_eval_phase01_still_works():
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase01_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(), env=_cli_env(),
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout
    assert "19" in result.stdout


def test_homechef_eval_phase02_frozen_mock_works():
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase02_frozen_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(), env=_cli_env(),
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout


def test_homechef_eval_phase02_diag_mock_works():
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase02_diag_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(), env=_cli_env(),
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout
