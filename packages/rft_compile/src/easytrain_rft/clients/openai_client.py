"""Real `RFTClient` backed by the OpenAI Python SDK.

The openai SDK is **not** a hard dependency of this package — we lazy-import
it inside the constructor so users running with a `FakeRFTClient` (or a
custom non-OpenAI client) don't need to install it.

Customers integrating against Azure OpenAI Foundry or another OpenAI-
compatible endpoint can pass an `openai_client` already configured for
that base URL; the runner code is otherwise identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from easytrain_rft.runner import Job, JobStatus

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIRFTClient:
    """Thin adapter from the EasyTrain `RFTClient` Protocol to OpenAI's SDK.

    Pass `api_key` to construct an SDK instance, or pass `openai_client`
    to reuse one you've already configured (custom base_url, retries, etc.).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: OpenAI | None = None,
    ) -> None:
        if openai_client is not None:
            self._client = openai_client
            return

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAIRFTClient requires the `openai` package. "
                "Install with `pip install openai` or use FakeRFTClient for tests."
            ) from e

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @staticmethod
    def _to_job(api_obj: Any) -> Job:
        """Convert an openai SDK FineTuningJob (or dict) into our `Job` model."""
        if hasattr(api_obj, "model_dump"):
            data = api_obj.model_dump()
        elif isinstance(api_obj, dict):
            data = api_obj
        else:
            # Fall back to attribute access for older SDK versions / namedtuples.
            data = {
                "id": getattr(api_obj, "id", ""),
                "status": getattr(api_obj, "status", "queued"),
                "model": getattr(api_obj, "model", ""),
                "training_file": getattr(api_obj, "training_file", ""),
                "fine_tuned_model": getattr(api_obj, "fine_tuned_model", None),
                "error": getattr(api_obj, "error", None),
                "created_at": getattr(api_obj, "created_at", None),
                "finished_at": getattr(api_obj, "finished_at", None),
            }
        # OpenAI sometimes returns "succeeded" / "queued" / etc. directly,
        # but also returns statuses we don't model (e.g. "running" → kept).
        # We normalize via JobStatus(...); unknown values raise loudly so we
        # find out about new statuses rather than silently dropping them.
        if "status" in data and not isinstance(data["status"], JobStatus):
            data["status"] = JobStatus(data["status"])
        return Job.model_validate(data)

    # ------------------------------------------------------------------ Protocol

    def submit_job(self, payload: dict[str, Any]) -> Job:
        api_obj = self._client.fine_tuning.jobs.create(**payload)
        return self._to_job(api_obj)

    def get_job(self, job_id: str) -> Job:
        api_obj = self._client.fine_tuning.jobs.retrieve(job_id)
        return self._to_job(api_obj)

    def cancel_job(self, job_id: str) -> Job:
        api_obj = self._client.fine_tuning.jobs.cancel(job_id)
        return self._to_job(api_obj)

    # ------------------------------------------------------------------ helpers

    def upload_training_file(self, path: Path | str) -> str:
        """Upload a JSONL training file with `purpose="fine-tune"`. Returns the file id."""
        path = Path(path)
        with path.open("rb") as fh:
            file_obj = self._client.files.create(file=fh, purpose="fine-tune")
        return file_obj.id
