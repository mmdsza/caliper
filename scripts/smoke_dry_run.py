"""End-to-end smoke: spin up server, POST the editor's sample grader, assert.

Run from repo root via `uv run python scripts/smoke_dry_run.py`.

Mirrors what the editor does at runtime — same /api/dry-run shape (minus the
Next.js rewrite prefix; we hit the FastAPI server directly on 127.0.0.1:8000).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from json import dumps, loads
from pathlib import Path
from urllib.error import HTTPError

REPO = Path(__file__).resolve().parent.parent
EDITOR_SAMPLE = REPO / "packages" / "editor" / "lib" / "sample-grader.ts"
SERVER_PORT = 8765
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"


def extract_sample_source() -> str:
    """Pull the `source` field out of `lib/sample-grader.ts`.

    The TS file uses a backtick template literal — easy to slice without a parser.
    """
    text = EDITOR_SAMPLE.read_text()
    match = re.search(r"source:\s*`([^`]*)`", text, re.DOTALL)
    if match is None:
        raise SystemExit("could not find `source: \\`...\\`` block in sample-grader.ts")
    return match.group(1)


def wait_for_health(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=1) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.2)
    raise SystemExit(f"server did not start within {timeout}s")


def post_dry_run(source: str) -> dict:
    body = dumps({"source": source, "eval_set": "legal_v3"}).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/dry-run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return loads(r.read())
    except HTTPError as e:
        raise SystemExit(f"dry-run failed: HTTP {e.code} — {e.read().decode()}") from e


def main() -> int:
    source = extract_sample_source()
    print(f"[smoke] sample grader source: {len(source)} chars")

    print(f"[smoke] starting server on :{SERVER_PORT}…")
    proc = subprocess.Popen(
        ["uv", "run", "caliper-server", "--port", str(SERVER_PORT)],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        wait_for_health()
        print("[smoke] server up")

        result = post_dry_run(source)
        print(f"[smoke] dry-run OK — grader={result['grader_name']} v{result['grader_version']}")

        report = result["report"]
        n_rows = report["n_rows"]
        n_succeeded = report["n_succeeded"]
        n_failed = report["n_failed"]
        mean = report["mean_score"]

        print(f"[smoke] rows={n_succeeded}/{n_rows}  failed={n_failed}  mean={mean:.3f}")

        # Assertions: sample grader on legal_v3 should produce
        #   - all 10 rows scored
        #   - zero failures
        #   - non-trivial distribution (mean strictly between 0 and 1)
        assert n_rows == 10, f"expected 10 rows, got {n_rows}"
        assert n_succeeded == 10, f"expected 10 succeeded, got {n_succeeded}"
        assert n_failed == 0, f"expected 0 failures, got {n_failed}"
        assert 0.0 < mean < 1.0, f"expected non-trivial mean, got {mean}"
        assert len(result["rows"]) == 10
        assert len(report["results"]) == 10

        print("[smoke] PASS")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
