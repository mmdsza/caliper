"""OpenAIRFTClient tests — mock the openai SDK; no network, no API key.

We verify that the adapter:
  * forwards `submit_job` to `client.fine_tuning.jobs.create(**payload)` verbatim
  * uses `retrieve(job_id)` for `get_job`
  * uses `cancel(job_id)` for `cancel_job`
  * normalizes the SDK return shape (whether it's a Pydantic model with
    `.model_dump()` or a plain dict) into our `Job` model
  * raises a clean ImportError when openai isn't installed AND no client
    is passed in
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from caliper_rft import Job, JobStatus
from caliper_rft.clients import OpenAIRFTClient


def _make_mock_openai_client() -> MagicMock:
    """Construct a mock standing in for `openai.OpenAI()`."""
    client = MagicMock(name="OpenAI")
    client.fine_tuning.jobs.create = MagicMock(name="jobs.create")
    client.fine_tuning.jobs.retrieve = MagicMock(name="jobs.retrieve")
    client.fine_tuning.jobs.cancel = MagicMock(name="jobs.cancel")
    client.files.create = MagicMock(name="files.create")
    return client


def _job_dict(**overrides: object) -> dict[str, object]:
    base = {
        "id": "ftjob-real-123",
        "status": "queued",
        "model": "gpt-4o-mini-2024-07-18",
        "training_file": "file-abc",
        "fine_tuned_model": None,
        "error": None,
        "created_at": 1700000000,
        "finished_at": None,
    }
    base.update(overrides)
    return base


def test_submit_job_forwards_payload_to_create() -> None:
    sdk = _make_mock_openai_client()
    sdk.fine_tuning.jobs.create.return_value = _job_dict()
    rft = OpenAIRFTClient(openai_client=sdk)

    payload = {
        "model": "gpt-4o-mini-2024-07-18",
        "training_file": "file-abc",
        "method": {"type": "reinforcement", "reinforcement": {"grader": {"type": "string_check"}}},
    }
    job = rft.submit_job(payload)

    sdk.fine_tuning.jobs.create.assert_called_once_with(**payload)
    assert isinstance(job, Job)
    assert job.id == "ftjob-real-123"
    assert job.status == JobStatus.QUEUED


def test_get_job_uses_retrieve() -> None:
    sdk = _make_mock_openai_client()
    sdk.fine_tuning.jobs.retrieve.return_value = _job_dict(
        status="succeeded",
        fine_tuned_model="ft:gpt-4o-mini:custom:abc",
        finished_at=1700001000,
    )
    rft = OpenAIRFTClient(openai_client=sdk)

    job = rft.get_job("ftjob-real-123")
    sdk.fine_tuning.jobs.retrieve.assert_called_once_with("ftjob-real-123")
    assert job.status == JobStatus.SUCCEEDED
    assert job.fine_tuned_model == "ft:gpt-4o-mini:custom:abc"


def test_cancel_job_uses_cancel() -> None:
    sdk = _make_mock_openai_client()
    sdk.fine_tuning.jobs.cancel.return_value = _job_dict(status="cancelled")
    rft = OpenAIRFTClient(openai_client=sdk)

    job = rft.cancel_job("ftjob-real-123")
    sdk.fine_tuning.jobs.cancel.assert_called_once_with("ftjob-real-123")
    assert job.status == JobStatus.CANCELLED


def test_to_job_handles_pydantic_model_dump() -> None:
    """SDK returns objects with `.model_dump()`. Verify we use it."""
    sdk = _make_mock_openai_client()
    fake_pydantic_obj = MagicMock()
    fake_pydantic_obj.model_dump.return_value = _job_dict(status="running")
    sdk.fine_tuning.jobs.retrieve.return_value = fake_pydantic_obj
    rft = OpenAIRFTClient(openai_client=sdk)

    job = rft.get_job("ftjob-real-123")
    fake_pydantic_obj.model_dump.assert_called_once()
    assert job.status == JobStatus.RUNNING


def test_unknown_status_raises() -> None:
    """An unmodelled status value blows up loudly — we want to find out."""
    sdk = _make_mock_openai_client()
    sdk.fine_tuning.jobs.retrieve.return_value = _job_dict(status="moonshot_state")
    rft = OpenAIRFTClient(openai_client=sdk)
    with pytest.raises(ValueError, match="moonshot_state"):
        rft.get_job("ftjob-real-123")


def test_upload_training_file(tmp_path) -> None:
    """File upload calls files.create with `purpose="fine-tune"` and returns id."""
    sdk = _make_mock_openai_client()
    sdk.files.create.return_value = MagicMock(id="file-uploaded-xyz")
    rft = OpenAIRFTClient(openai_client=sdk)

    train = tmp_path / "train.jsonl"
    train.write_text('{"messages": []}\n')

    file_id = rft.upload_training_file(train)
    assert file_id == "file-uploaded-xyz"
    _, kwargs = sdk.files.create.call_args
    assert kwargs["purpose"] == "fine-tune"
    # `file=` must be a file-like (not a path string) — verify it's not str/bytes.
    assert hasattr(kwargs["file"], "read")


def test_construct_without_openai_installed(monkeypatch) -> None:
    """When `openai` isn't importable AND no client is passed, raise cleanly."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="requires the `openai` package"):
        OpenAIRFTClient()
