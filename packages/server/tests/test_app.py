"""FastAPI route tests via TestClient."""

from caliper_sdk import (
    EvalRow,
    GraderLoadError,
    RowResult,
    RunReport,
)
from caliper_server import create_app
from caliper_server.executor import ExecutorResult, SandboxError
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_eval_sets(client: TestClient) -> None:
    r = client.get("/eval-sets")
    assert r.status_code == 200
    assert "legal_v3" in r.json()["names"]


def test_get_eval_set(client: TestClient) -> None:
    r = client.get("/eval-sets/legal_v3")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "legal_v3"
    assert body["n_rows"] == 10
    assert len(body["version"]) == 12
    assert body["description"]
    assert all(set(row) == {"id", "prompt", "response", "gold"} for row in body["rows"])


def test_list_eval_set_versions(client: TestClient) -> None:
    r = client.get("/eval-sets/legal_v3/versions")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "legal_v3"
    assert len(body["versions"]) == 1
    v = body["versions"][0]
    assert v["name"] == "legal_v3"
    assert len(v["version"]) == 12
    assert v["n_rows"] == 10


def test_list_eval_set_versions_unknown_name(client: TestClient) -> None:
    r = client.get("/eval-sets/missing/versions")
    assert r.status_code == 404


def test_dry_run_response_includes_eval_set_version(
    client: TestClient, sample_grader_source: str
) -> None:
    r = client.post(
        "/dry-run",
        json={"source": sample_grader_source, "eval_set": "legal_v3"},
    )
    assert r.status_code == 200
    assert len(r.json()["eval_set_version"]) == 12


def test_dry_run_with_pinned_version(
    client: TestClient, sample_grader_source: str
) -> None:
    """Pinning to the current latest version returns the same result."""
    versions_resp = client.get("/eval-sets/legal_v3/versions")
    pinned = versions_resp.json()["versions"][0]["version"]

    r = client.post(
        "/dry-run",
        json={
            "source": sample_grader_source,
            "eval_set": "legal_v3",
            "eval_set_version": pinned,
        },
    )
    assert r.status_code == 200
    assert r.json()["eval_set_version"] == pinned


def test_dry_run_with_unknown_pinned_version(
    client: TestClient, sample_grader_source: str
) -> None:
    r = client.post(
        "/dry-run",
        json={
            "source": sample_grader_source,
            "eval_set": "legal_v3",
            "eval_set_version": "deadbeefdead",
        },
    )
    assert r.status_code == 404


def test_get_eval_set_not_found(client: TestClient) -> None:
    r = client.get("/eval-sets/missing")
    assert r.status_code == 404


def test_dry_run_happy_path(client: TestClient, sample_grader_source: str) -> None:
    r = client.post(
        "/dry-run",
        json={"source": sample_grader_source, "eval_set": "legal_v3"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grader_name"] == "legal_citation_grader"
    assert body["grader_version"] == "0.1.0"
    assert body["eval_set"] == "legal_v3"

    report = body["report"]
    assert report["n_rows"] == 10
    assert report["n_succeeded"] == 10
    assert report["n_failed"] == 0
    # 5 of 10 fixture rows pass the format-gate-and-gold-present test
    # (rows 001, 002, 003, 009, 010 — see eval_sets/legal_v3.py).
    assert report["mean_score"] == 0.5
    assert len(report["results"]) == 10

    assert len(body["rows"]) == 10
    assert {row["id"] for row in body["rows"]} == {
        f"legal_v3-{i:03d}" for i in range(1, 11)
    }


def test_dry_run_syntax_error(client: TestClient) -> None:
    r = client.post(
        "/dry-run",
        json={"source": "def broken(:\n    pass\n", "eval_set": "legal_v3"},
    )
    assert r.status_code == 422
    assert "syntax error" in r.json()["detail"]


def test_dry_run_no_grader(client: TestClient) -> None:
    r = client.post(
        "/dry-run",
        json={"source": "x = 1\n", "eval_set": "legal_v3"},
    )
    assert r.status_code == 422
    assert "no @grader-decorated function" in r.json()["detail"]


def test_dry_run_unknown_eval_set(client: TestClient, sample_grader_source: str) -> None:
    r = client.post(
        "/dry-run",
        json={"source": sample_grader_source, "eval_set": "nope"},
    )
    assert r.status_code == 404


def test_dry_run_grader_raises_per_row(client: TestClient) -> None:
    src = """
from caliper_sdk import grader, Rollout, GraderResult

@grader
def explodes(r: Rollout) -> float:
    raise RuntimeError("nope")
"""
    r = client.post("/dry-run", json={"source": src, "eval_set": "legal_v3"})
    assert r.status_code == 200
    report = r.json()["report"]
    assert report["n_rows"] == 10
    assert report["n_succeeded"] == 0
    assert report["n_failed"] == 10
    assert all(f["error_type"] == "RuntimeError" for f in report["failures"])


class _StubExecutor:
    """Records calls and returns a fixed ExecutorResult — proves the route
    is executor-agnostic (doesn't reach into in-process loader internals)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[EvalRow]]] = []

    def execute(self, source: str, rows: list[EvalRow]) -> ExecutorResult:
        self.calls.append((source, rows))
        report = RunReport(
            grader_name="stub",
            grader_version="0.0.0",
            n_rows=len(rows),
            n_succeeded=len(rows),
            n_failed=0,
            mean_score=1.0,
            std_score=0.0,
            min_score=1.0,
            max_score=1.0,
            results=[
                RowResult(row_id=row.id, score=1.0, explanation="stub")
                for row in rows
            ],
            failures=[],
        )
        return ExecutorResult(name="stub", version="0.0.0", report=report)


def test_dry_run_uses_injected_executor(sample_grader_source: str) -> None:
    stub = _StubExecutor()
    client = TestClient(create_app(executor=stub))

    r = client.post(
        "/dry-run",
        json={"source": sample_grader_source, "eval_set": "legal_v3"},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grader_name"] == "stub"
    assert body["report"]["mean_score"] == 1.0
    assert len(stub.calls) == 1
    source, rows = stub.calls[0]
    assert source == sample_grader_source
    assert len(rows) == 10


class _LoadErrorExecutor:
    def execute(self, source: str, rows: list[EvalRow]) -> ExecutorResult:
        raise GraderLoadError("custom load error from executor")


def test_dry_run_executor_load_error_maps_to_422(sample_grader_source: str) -> None:
    client = TestClient(create_app(executor=_LoadErrorExecutor()))
    r = client.post(
        "/dry-run",
        json={"source": sample_grader_source, "eval_set": "legal_v3"},
    )
    assert r.status_code == 422
    assert "custom load error" in r.json()["detail"]


class _SandboxErrorExecutor:
    def execute(self, source: str, rows: list[EvalRow]) -> ExecutorResult:
        raise SandboxError("e2b OOMKilled")


def test_dry_run_executor_sandbox_error_maps_to_502(sample_grader_source: str) -> None:
    client = TestClient(create_app(executor=_SandboxErrorExecutor()))
    r = client.post(
        "/dry-run",
        json={"source": sample_grader_source, "eval_set": "legal_v3"},
    )
    assert r.status_code == 502
    assert "OOMKilled" in r.json()["detail"]
