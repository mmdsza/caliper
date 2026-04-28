"""OpenAI RFT passthrough runner.

This module is the SDK-side glue that turns an EasyTrain grader spec into a
submitted OpenAI fine-tuning job and polls it to completion. The submission
target is OpenAI's reinforcement fine-tuning surface — `POST /v1/fine_tuning/
jobs` with `method.type == "reinforcement"` carrying the compiled grader DSL
in `method.reinforcement.grader`.

Design choices
--------------

* **Pluggable client.** `RFTClient` is a Protocol. The default real impl is
  `easytrain_rft.clients.OpenAIRFTClient`; a `FakeRFTClient` (in
  `easytrain_rft.testing`) is used for unit tests and any caller who wants
  to exercise the runner without hitting the network. Customers running on
  Azure Foundry or similar can write their own client.

* **Pure compile + submit separation.** The runner accepts an `AnyGrader`
  (the typed spec from `easytrain_rft.spec`), compiles it via
  `compile_to_rft`, and forwards the dict to the client. The grader spec
  validation already ran at construction time.

* **Polling without a thread pool.** `wait_for_completion` is a synchronous
  poll loop with caller-provided `poll_interval` and `timeout`. Async is a
  later concern — most callers wrap this in a job runner of their own.

* **No file upload here.** OpenAI requires the training file to exist as
  an uploaded `file-...` ID. We accept the ID as a parameter; uploading is
  a separate concern handled by `OpenAIRFTClient.upload_training_file()`
  (or the user's own pre-upload).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from easytrain_rft.compile import compile_to_rft
from easytrain_rft.spec import AnyGrader


class JobStatus(StrEnum):
    """OpenAI fine-tuning job lifecycle states.

    Ordered by typical progression. `cancelled` and `failed` are terminal
    along with `succeeded`.
    """

    QUEUED = "queued"
    VALIDATING_FILES = "validating_files"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
)


class Job(BaseModel):
    """A normalized fine-tuning job record.

    Mirrors the OpenAI `FineTuningJob` shape but trimmed to the fields the
    runner actually consumes. Extra fields are tolerated so the model
    survives upstream API additions without churn.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    status: JobStatus
    model: str
    training_file: str
    fine_tuned_model: str | None = None
    error: dict[str, Any] | None = None
    created_at: int | None = None
    finished_at: int | None = None


@runtime_checkable
class RFTClient(Protocol):
    """Structural type for any RFT submission client."""

    def submit_job(self, payload: dict[str, Any]) -> Job: ...
    def get_job(self, job_id: str) -> Job: ...
    def cancel_job(self, job_id: str) -> Job: ...


class RFTRunPayload(BaseModel):
    """The body sent to OpenAI's fine-tuning endpoint.

    Constructed by `RFTRunner.submit` from (grader, model, training_file,
    hyperparameters). Exposed as a Pydantic model so callers can introspect
    the payload before submission (useful for dry-runs and audit logs).
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    training_file: str
    method: dict[str, Any] = Field(default_factory=dict)


def build_payload(
    grader: AnyGrader,
    *,
    model: str,
    training_file: str,
    hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile a grader spec into the OpenAI RFT job submission payload.

    Returns the raw dict that gets sent over the wire. Pure — no network.
    """
    grader_json = compile_to_rft(grader)
    reinforcement: dict[str, Any] = {"grader": grader_json}
    if hyperparameters:
        reinforcement["hyperparameters"] = hyperparameters

    return {
        "model": model,
        "training_file": training_file,
        "method": {"type": "reinforcement", "reinforcement": reinforcement},
    }


class RFTRunner:
    """Orchestrate compile → submit → poll for OpenAI RFT.

    The runner is client-agnostic — pass in any `RFTClient` (the real
    `OpenAIRFTClient`, the `FakeRFTClient` for tests, or a custom impl).
    """

    def __init__(self, client: RFTClient) -> None:
        self.client = client

    def submit(
        self,
        grader: AnyGrader,
        *,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
    ) -> Job:
        payload = build_payload(
            grader,
            model=model,
            training_file=training_file,
            hyperparameters=hyperparameters,
        )
        return self.client.submit_job(payload)

    def get(self, job_id: str) -> Job:
        return self.client.get_job(job_id)

    def cancel(self, job_id: str) -> Job:
        return self.client.cancel_job(job_id)

    def wait_for_completion(
        self,
        job_id: str,
        *,
        poll_interval: float = 30.0,
        timeout: float | None = None,
        on_status_change: Callable[[Job], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> Job:
        """Poll until `job_id` reaches a terminal status or `timeout` elapses.

        `sleep` and `clock` are injectable so tests can run without real
        wall-clock waits. `on_status_change` fires once per observed status
        transition (including the first observation), useful for logs / UI.
        """
        deadline = clock() + timeout if timeout is not None else None
        last_status: JobStatus | None = None

        while True:
            job = self.client.get_job(job_id)
            if job.status != last_status:
                if on_status_change is not None:
                    on_status_change(job)
                last_status = job.status

            if job.status in TERMINAL_STATUSES:
                return job

            if deadline is not None and clock() >= deadline:
                raise TimeoutError(
                    f"job {job_id} did not reach a terminal status within "
                    f"{timeout}s (last status: {job.status.value})"
                )

            sleep(poll_interval)
