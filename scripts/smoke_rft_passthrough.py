"""End-to-end smoke for the RFT passthrough.

Compiles a real grader spec, submits via FakeRFTClient, polls until terminal,
prints the audit trail. No network. Used as the CI surrogate for OpenAI's
real fine-tuning API — the real client (`OpenAIRFTClient`) is unit-tested
against a mocked SDK in `tests/test_openai_client.py`.

Run from repo root: `uv run python scripts/smoke_rft_passthrough.py`.
"""

from __future__ import annotations

import json
import sys

from caliper_rft import (
    JobStatus,
    MultiGrader,
    PythonGrader,
    RFTRunner,
    ScoreModelGrader,
    StringCheckGrader,
    build_payload,
)
from caliper_rft.testing import FakeRFTClient


def main() -> int:
    # 1. Author a real, multi-grader spec — exactly the shape a customer
    #    would author in the editor. This is the "format gate + LLM judge
    #    + python check" composite from `docs/03-graders-and-rewards.md`.
    grader = MultiGrader(
        graders={
            "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            "correctness": ScoreModelGrader(
                model="gpt-4.1-mini",
                prompt="Is this answer correct? Reply 0.0 to 1.0.",
            ),
            "code_runs": PythonGrader(
                source="def grade(sample, item):\n    return 1.0\n",
            ),
        },
        calculate_output="0.3 * format + 0.5 * correctness + 0.2 * code_runs",
    )

    # 2. Inspect the wire payload before submission. This is what we'd
    #    POST to /v1/fine_tuning/jobs.
    payload = build_payload(
        grader,
        model="gpt-4o-mini-2024-07-18",
        training_file="file-PRE-UPLOADED-abc",
        hyperparameters={"n_epochs": 1, "compute_multiplier": 1.0},
    )
    print("[smoke] submission payload (truncated):")
    print(json.dumps(payload, indent=2)[:600] + "\n  ...")
    assert payload["method"]["type"] == "reinforcement"
    assert payload["method"]["reinforcement"]["grader"]["type"] == "multi"

    # 3. Submit + poll via the FakeRFTClient — scriptable status progression
    #    standing in for the real OpenAI lifecycle.
    client = FakeRFTClient(
        status_sequence=["queued", "running", "running", "running", "succeeded"]
    )
    runner = RFTRunner(client)

    job = runner.submit(
        grader,
        model="gpt-4o-mini-2024-07-18",
        training_file="file-PRE-UPLOADED-abc",
        hyperparameters={"n_epochs": 1, "compute_multiplier": 1.0},
    )
    print(f"[smoke] submitted: id={job.id}  initial_status={job.status.value}")

    transitions: list[JobStatus] = []
    final = runner.wait_for_completion(
        job.id,
        poll_interval=0.0,
        sleep=lambda _s: None,
        on_status_change=lambda j: transitions.append(j.status),
    )

    print(
        f"[smoke] transitions: {' → '.join(s.value for s in transitions)}"
    )
    print(
        f"[smoke] final: status={final.status.value}  "
        f"fine_tuned_model={final.fine_tuned_model}"
    )

    if final.status is not JobStatus.SUCCEEDED:
        print(f"[smoke] FAIL — expected SUCCEEDED, got {final.status.value}")
        return 1
    if final.fine_tuned_model is None:
        print("[smoke] FAIL — fine_tuned_model not populated on success")
        return 1
    if transitions != [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED]:
        print(f"[smoke] FAIL — unexpected transition trail: {transitions}")
        return 1

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
