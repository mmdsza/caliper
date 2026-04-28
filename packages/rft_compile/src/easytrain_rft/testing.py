"""Test doubles for the RFT runner.

`FakeRFTClient` is exported as a public-API testing helper so customers can
exercise their own RFT integrations (and the EasyTrain runner) without
touching the OpenAI network. It has scriptable status progression: each
call to `get_job` advances the status sequence, making polling-loop tests
deterministic.
"""

from __future__ import annotations

import itertools
from typing import Any

from easytrain_rft.runner import Job, JobStatus


class FakeRFTClient:
    """An in-memory `RFTClient` for tests and dev work.

    Usage:

        client = FakeRFTClient(
            base_model="gpt-4o-mini-2024-07-18",
            status_sequence=["queued", "running", "running", "succeeded"],
        )
        runner = RFTRunner(client)
        job = runner.submit(grader, model="gpt-4o-mini-2024-07-18",
                            training_file="file-abc")
        final = runner.wait_for_completion(job.id, sleep=lambda s: None)

    The first `submit_job` returns a job with status `queued`. Each
    subsequent `get_job(job.id)` returns the next status in the sequence;
    if the sequence is exhausted, the last status sticks. Statuses can be
    raw strings (coerced to JobStatus) or JobStatus instances.
    """

    def __init__(
        self,
        *,
        base_model: str = "gpt-4o-mini-2024-07-18",
        status_sequence: list[str | JobStatus] | None = None,
        fine_tuned_model: str = "ft:gpt-4o-mini:fake:abc123",
    ) -> None:
        self._statuses: list[JobStatus] = [
            JobStatus(s) if not isinstance(s, JobStatus) else s
            for s in (status_sequence or [JobStatus.SUCCEEDED])
        ]
        if not self._statuses:
            self._statuses = [JobStatus.SUCCEEDED]
        self._iters: dict[str, itertools.chain[JobStatus]] = {}
        self._jobs: dict[str, Job] = {}
        self._base_model = base_model
        self._fine_tuned_model = fine_tuned_model

        # Recorded calls for assertions in tests.
        self.submit_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self.cancel_calls: list[str] = []

    # ------------------------------------------------------------------ helpers

    def _make_iter(self) -> itertools.chain[JobStatus]:
        # Cycle through the sequence then stick on the last one.
        seq = iter(self._statuses)
        last = self._statuses[-1]
        return itertools.chain(seq, itertools.repeat(last))

    def _next_id(self) -> str:
        return f"ftjob-fake-{len(self._jobs) + 1:04d}"

    # ------------------------------------------------------------------ RFTClient

    def submit_job(self, payload: dict[str, Any]) -> Job:
        self.submit_calls.append(payload)
        job_id = self._next_id()
        self._iters[job_id] = self._make_iter()
        # First state is the initial submitted state — the next call to
        # get_job advances to status_sequence[0].
        job = Job(
            id=job_id,
            status=JobStatus.VALIDATING_FILES,
            model=payload.get("model", self._base_model),
            training_file=payload.get("training_file", "file-fake"),
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job:
        self.get_calls.append(job_id)
        if job_id not in self._jobs:
            raise KeyError(f"unknown job_id: {job_id}")
        status = next(self._iters[job_id])
        prev = self._jobs[job_id]
        fine_tuned = (
            self._fine_tuned_model if status is JobStatus.SUCCEEDED else None
        )
        updated = prev.model_copy(
            update={"status": status, "fine_tuned_model": fine_tuned}
        )
        self._jobs[job_id] = updated
        return updated

    def cancel_job(self, job_id: str) -> Job:
        self.cancel_calls.append(job_id)
        if job_id not in self._jobs:
            raise KeyError(f"unknown job_id: {job_id}")
        prev = self._jobs[job_id]
        updated = prev.model_copy(update={"status": JobStatus.CANCELLED})
        self._jobs[job_id] = updated
        return updated
