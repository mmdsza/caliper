"""RFTRunner orchestration tests, run against the in-memory FakeRFTClient."""

from __future__ import annotations

import json

import pytest
from easytrain_rft import (
    Job,
    JobStatus,
    MultiGrader,
    RFTClient,
    RFTRunner,
    ScoreModelGrader,
    StringCheckGrader,
    build_payload,
)
from easytrain_rft.runner import RFTRunPayload
from easytrain_rft.testing import FakeRFTClient


@pytest.fixture
def grader() -> MultiGrader:
    return MultiGrader(
        graders={
            "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            "correctness": ScoreModelGrader(model="gpt-4.1", prompt="Is this right?"),
        },
        calculate_output="0.3 * format + 0.7 * correctness",
    )


def test_fake_client_satisfies_protocol() -> None:
    client = FakeRFTClient()
    assert isinstance(client, RFTClient)


def test_build_payload_shape(grader: MultiGrader) -> None:
    """Payload matches OpenAI's reinforcement fine-tuning spec shape."""
    payload = build_payload(
        grader,
        model="gpt-4o-mini-2024-07-18",
        training_file="file-abc",
        hyperparameters={"n_epochs": 1, "compute_multiplier": 1.0},
    )
    assert payload["model"] == "gpt-4o-mini-2024-07-18"
    assert payload["training_file"] == "file-abc"
    assert payload["method"]["type"] == "reinforcement"
    grader_json = payload["method"]["reinforcement"]["grader"]
    assert grader_json["type"] == "multi"
    assert set(grader_json["graders"]) == {"format", "correctness"}
    assert payload["method"]["reinforcement"]["hyperparameters"] == {
        "n_epochs": 1,
        "compute_multiplier": 1.0,
    }
    # Round-trips through json without error.
    json.dumps(payload)


def test_build_payload_omits_hyperparameters_when_none(grader: MultiGrader) -> None:
    payload = build_payload(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )
    assert "hyperparameters" not in payload["method"]["reinforcement"]


def test_run_payload_model_validates(grader: MultiGrader) -> None:
    payload_dict = build_payload(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )
    payload = RFTRunPayload.model_validate(payload_dict)
    assert payload.model == "gpt-4o-mini-2024-07-18"


def test_runner_submit_calls_client_with_compiled_payload(grader: MultiGrader) -> None:
    client = FakeRFTClient()
    runner = RFTRunner(client)

    job = runner.submit(
        grader,
        model="gpt-4o-mini-2024-07-18",
        training_file="file-abc",
        hyperparameters={"n_epochs": 1},
    )

    assert isinstance(job, Job)
    assert job.id.startswith("ftjob-fake-")
    assert job.status == JobStatus.VALIDATING_FILES

    assert len(client.submit_calls) == 1
    sent = client.submit_calls[0]
    assert sent["model"] == "gpt-4o-mini-2024-07-18"
    assert sent["method"]["type"] == "reinforcement"
    assert sent["method"]["reinforcement"]["grader"]["type"] == "multi"


def test_wait_for_completion_polls_until_terminal(grader: MultiGrader) -> None:
    client = FakeRFTClient(
        status_sequence=["queued", "running", "running", "succeeded"]
    )
    runner = RFTRunner(client)
    job = runner.submit(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )

    sleeps: list[float] = []
    final = runner.wait_for_completion(
        job.id,
        poll_interval=0.05,
        sleep=sleeps.append,
    )

    assert final.status == JobStatus.SUCCEEDED
    assert final.fine_tuned_model == "ft:gpt-4o-mini:fake:abc123"
    # Three polls before terminal: queued, running, running. Then succeeded
    # (terminal) so no further sleep.
    assert len(sleeps) == 3
    assert all(s == 0.05 for s in sleeps)


def test_wait_for_completion_records_status_changes(grader: MultiGrader) -> None:
    client = FakeRFTClient(
        status_sequence=["queued", "running", "running", "succeeded"]
    )
    runner = RFTRunner(client)
    job = runner.submit(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )

    transitions: list[JobStatus] = []
    runner.wait_for_completion(
        job.id,
        poll_interval=0.0,
        sleep=lambda _s: None,
        on_status_change=lambda j: transitions.append(j.status),
    )

    # Each *distinct* status fires the callback exactly once.
    assert transitions == [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED]


def test_wait_for_completion_terminal_failed(grader: MultiGrader) -> None:
    client = FakeRFTClient(status_sequence=["running", "failed"])
    runner = RFTRunner(client)
    job = runner.submit(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )

    final = runner.wait_for_completion(
        job.id, poll_interval=0.0, sleep=lambda _s: None
    )
    assert final.status == JobStatus.FAILED
    assert final.fine_tuned_model is None


def test_wait_for_completion_timeout(grader: MultiGrader) -> None:
    """Timeout fires when the job never reaches a terminal state."""
    # Sequence of running-only would otherwise loop forever.
    client = FakeRFTClient(status_sequence=["running"])
    runner = RFTRunner(client)
    job = runner.submit(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )

    # Fake clock increments by `poll_interval` on every sleep; deadline is
    # 1.0s; sleeps of 0.5s → first sleep brings clock to 0.5 (under deadline,
    # OK), second sleep to 1.0 (== deadline, raises on next loop iteration).
    fake_now = [0.0]

    def fake_sleep(s: float) -> None:
        fake_now[0] += s

    with pytest.raises(TimeoutError, match="did not reach a terminal status"):
        runner.wait_for_completion(
            job.id,
            poll_interval=0.5,
            timeout=1.0,
            sleep=fake_sleep,
            clock=lambda: fake_now[0],
        )


def test_get_and_cancel(grader: MultiGrader) -> None:
    client = FakeRFTClient(status_sequence=["running"])
    runner = RFTRunner(client)
    job = runner.submit(
        grader, model="gpt-4o-mini-2024-07-18", training_file="file-abc"
    )

    fetched = runner.get(job.id)
    assert fetched.id == job.id
    assert fetched.status == JobStatus.RUNNING

    cancelled = runner.cancel(job.id)
    assert cancelled.status == JobStatus.CANCELLED
    assert client.cancel_calls == [job.id]


def test_unknown_job_raises() -> None:
    client = FakeRFTClient()
    runner = RFTRunner(client)
    with pytest.raises(KeyError):
        runner.get("ftjob-does-not-exist")
