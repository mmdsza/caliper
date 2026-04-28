"""Subprocess-level test of the sandbox driver.

Runs ``_sandbox_driver.py`` as a child Python process with a JSON payload
on stdin and asserts the JSON envelope on stdout. This proves the wire
contract end-to-end without paying for a real E2B sandbox in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "easytrain_server"
    / "_sandbox_driver.py"
)

OK_GRADER = '''
from easytrain_sdk import grader, Rollout, GraderResult


@grader(name="x", version="9.9.9")
def g(rollout: Rollout) -> float:
    return 1.0 if rollout.response == "yes" else 0.0
'''


def _run(payload: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, str(DRIVER)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture
def two_rows() -> list[dict]:
    return [
        {"id": "r0", "rollout": {"prompt": "?", "response": "yes"}},
        {"id": "r1", "rollout": {"prompt": "?", "response": "no"}},
    ]


def test_driver_happy_path(two_rows: list[dict]) -> None:
    out = _run({"source": OK_GRADER, "rows": two_rows})
    assert out["kind"] == "ok"
    assert out["name"] == "x"
    assert out["version"] == "9.9.9"
    report = out["report"]
    assert report["n_rows"] == 2
    assert report["n_succeeded"] == 2
    assert report["mean_score"] == 0.5
    scores = sorted(r["score"] for r in report["results"])
    assert scores == [0.0, 1.0]


def test_driver_load_error_returns_envelope(two_rows: list[dict]) -> None:
    out = _run({"source": "def broken(:\n    pass\n", "rows": two_rows})
    assert out["kind"] == "load_error"
    assert "syntax error" in out["message"]


def test_driver_no_grader_returns_envelope(two_rows: list[dict]) -> None:
    out = _run({"source": "x = 1\n", "rows": two_rows})
    assert out["kind"] == "load_error"
    assert "no @grader-decorated function" in out["message"]


def test_driver_per_row_failure_stays_in_report(two_rows: list[dict]) -> None:
    """Grader exceptions become RowFailure entries — not load errors."""
    src = '''
from easytrain_sdk import grader, Rollout

@grader
def boom(rollout: Rollout) -> float:
    raise RuntimeError("nope")
'''
    out = _run({"source": src, "rows": two_rows})
    assert out["kind"] == "ok"
    report = out["report"]
    assert report["n_succeeded"] == 0
    assert report["n_failed"] == 2
    assert all(f["error_type"] == "RuntimeError" for f in report["failures"])
