"""Sandbox driver for E2BExecutor.

Runs inside the sandbox (or, in tests, as a local subprocess). Reads a
JSON payload ``{"source": str, "rows": [EvalRow.model_dump, ...]}`` from
stdin, loads the grader via the SDK, runs it, and writes a single JSON
envelope to stdout:

* ``{"kind": "ok", "name": str, "version": str, "report": <RunReport.model_dump>}``
* ``{"kind": "load_error", "message": str}``

Has only ``easytrain_sdk`` as a dependency — does not import anything from
``easytrain_server``.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    source = payload["source"]
    raw_rows = payload["rows"]

    from easytrain_sdk import (
        EvalRow,
        GraderLoadError,
        Runner,
        load_grader_from_source,
    )

    rows = [EvalRow.model_validate(r) for r in raw_rows]

    try:
        loaded = load_grader_from_source(source)
    except GraderLoadError as e:
        json.dump({"kind": "load_error", "message": str(e)}, sys.stdout)
        sys.stdout.write("\n")
        return 0

    report = Runner(loaded.grader).run(rows)
    json.dump(
        {
            "kind": "ok",
            "name": loaded.name,
            "version": loaded.version,
            "report": report.model_dump(),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
