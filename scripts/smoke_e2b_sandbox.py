"""Live smoke for the E2B sandbox executor.

Spins a real Firecracker microVM via the E2B SDK, ships the workspace
``easytrain-sdk`` wheel into it, runs the legal-citation sample grader
against the ``legal_v3`` fixture (10 rows), and asserts ``mean=0.5``
(same expected histogram as ``smoke_dry_run.py``).

Skips cleanly when:

* ``E2B_API_KEY`` is unset.
* ``e2b-code-interpreter`` is not installed (optional ``[e2b]`` extra).

Run from repo root:

    EASYTRAIN_GRADER_EXECUTOR=e2b E2B_API_KEY=... \\
        uv run --extra e2b python scripts/smoke_e2b_sandbox.py

Not run in CI by default — `uv sync` doesn't pull the e2b extra and we
shouldn't depend on E2B_API_KEY for the unit suite.
"""

from __future__ import annotations

import os
import sys

from easytrain_server._seed import LEGAL_V3_ROWS
from easytrain_server.executor import E2BExecutor


SAMPLE_GRADER_SOURCE = '''
from easytrain_sdk import grader, Rollout, GraderResult


@grader(name="legal_citation_grader", version="0.1.0")
def legal_citation_grader(rollout: Rollout) -> GraderResult:
    response = rollout.response
    if not response.startswith("<think>"):
        return GraderResult(score=0.0, explanation="missing <think> gate")
    if rollout.gold is not None and rollout.gold in response:
        return GraderResult(score=1.0, explanation="gold present in response")
    return GraderResult(score=0.0, explanation="gold absent")
'''


def _skip(reason: str) -> int:
    print(f"[smoke] SKIP — {reason}")
    return 0


def main() -> int:
    if not os.environ.get("E2B_API_KEY"):
        return _skip("E2B_API_KEY not set")

    try:
        executor = E2BExecutor()
    except ImportError as e:
        return _skip(str(e))

    rows = LEGAL_V3_ROWS
    print(f"[smoke] running E2BExecutor against legal_v3 ({len(rows)} rows)…")

    result = executor.execute(SAMPLE_GRADER_SOURCE, rows)
    report = result.report

    print(
        f"[smoke] grader={result.name}@{result.version}  "
        f"rows={report.n_succeeded}/{report.n_rows}  "
        f"failed={report.n_failed}  mean={report.mean_score:.3f}"
    )

    if report.n_rows != 10 or report.n_failed != 0:
        print("[smoke] FAIL — expected 10 rows, 0 failures")
        return 1
    if abs(report.mean_score - 0.5) > 1e-9:
        print(f"[smoke] FAIL — expected mean_score=0.5, got {report.mean_score}")
        return 1

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
