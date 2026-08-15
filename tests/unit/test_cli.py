"""Task 8: CLI entry point tests."""

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


def test_homechef_eval_help():
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--help"],
        capture_output=True, text=True, cwd=Path.cwd(), env=_cli_env(),
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_homechef_eval_mock_run():
    result = subprocess.run(
        ["uv", "run", "homechef-eval", "--config", "configs/evaluation/phase01_mock.yaml"],
        capture_output=True, text=True, cwd=Path.cwd(), env=_cli_env(),
    )
    assert result.returncode == 0
    assert "Total cases" in result.stdout
    assert "19" in result.stdout


def test_cli_env_is_cross_platform():
    """Prove the CLI env PATH is cross-platform (Windows + Linux)."""
    env = _cli_env()

    # PYTHONIOENCODING is set for UTF-8-safe subprocess output.
    assert env["PYTHONIOENCODING"] == "utf-8"

    path = env["PATH"]
    # Uses os.pathsep (never a hardcoded ';' or ':').
    assert os.pathsep in path
    assert env["PATH"] != ";"
    assert env["PATH"] != ":"

    entries = path.split(os.pathsep)
    # Path(sys.executable).parent is PREPENDED (Scripts/ on Windows, bin/ on
    # Linux) — derived from sys.executable, NOT a hardcoded ".venv/Scripts".
    assert entries[0] == str(Path(sys.executable).parent)

    # Preserves the original PATH (system + current venv bin not dropped).
    original = os.environ.get("PATH", "")
    for entry in original.split(os.pathsep):
        if entry:
            assert entry in entries, f"original PATH entry lost: {entry}"
