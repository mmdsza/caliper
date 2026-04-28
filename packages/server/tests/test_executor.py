"""Tests for the executor module.

Covers:

* :class:`InProcessExecutor` happy path / load errors / per-row failures.
* :class:`E2BExecutor` against a fake Sandbox: payload assembly, stdout
  parsing, ``kill()`` called on every path, ``SandboxError`` paths.
* :func:`build_from_env` dispatch including unknown values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from easytrain_sdk import EvalRow, GraderLoadError, Rollout
from easytrain_server.executor import (
    E2BExecutor,
    InProcessExecutor,
    SandboxError,
    build_from_env,
)

OK_GRADER = '''
from easytrain_sdk import grader, Rollout, GraderResult


@grader(name="ok_grader", version="1.2.3")
def ok_grader(rollout: Rollout) -> float:
    return 1.0 if rollout.response == rollout.gold else 0.0
'''


@pytest.fixture
def rows() -> list[EvalRow]:
    return [
        EvalRow(id="r0", rollout=Rollout(prompt="q", response="hi", gold="hi")),
        EvalRow(id="r1", rollout=Rollout(prompt="q", response="bye", gold="hi")),
    ]


# ---------- InProcessExecutor ----------


class TestInProcessExecutor:
    def test_happy_path(self, rows: list[EvalRow]) -> None:
        result = InProcessExecutor().execute(OK_GRADER, rows)
        assert result.name == "ok_grader"
        assert result.version == "1.2.3"
        assert result.report.n_rows == 2
        assert result.report.n_succeeded == 2
        assert result.report.mean_score == 0.5

    def test_load_error_propagates(self, rows: list[EvalRow]) -> None:
        with pytest.raises(GraderLoadError, match="syntax error"):
            InProcessExecutor().execute("def broken(:\n    pass\n", rows)

    def test_per_row_failure_in_report(self, rows: list[EvalRow]) -> None:
        src = '''
from easytrain_sdk import grader, Rollout

@grader
def explodes(r: Rollout) -> float:
    raise RuntimeError("nope")
'''
        result = InProcessExecutor().execute(src, rows)
        assert result.report.n_succeeded == 0
        assert result.report.n_failed == 2
        assert all(f.error_type == "RuntimeError" for f in result.report.failures)


# ---------- E2BExecutor (fake Sandbox) ----------


@dataclass
class _FakeCmdResult:
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class _FakeFiles:
    writes: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str, content: Any) -> None:
        self.writes[path] = content


@dataclass
class _FakeCommands:
    install_result: _FakeCmdResult = field(default_factory=_FakeCmdResult)
    run_result: _FakeCmdResult = field(default_factory=_FakeCmdResult)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        cmd: str,
        *,
        stdin: str | None = None,
        timeout: int | None = None,
    ) -> _FakeCmdResult:
        self.calls.append({"cmd": cmd, "stdin": stdin, "timeout": timeout})
        if cmd.startswith("pip install"):
            return self.install_result
        return self.run_result


class _FakeSandbox:
    """Imitates the e2b_code_interpreter.Sandbox surface E2BExecutor uses."""

    def __init__(
        self,
        *,
        install_result: _FakeCmdResult | None = None,
        run_result: _FakeCmdResult | None = None,
        kill_raises: bool = False,
        timeout: int | None = None,
    ) -> None:
        self.files = _FakeFiles()
        self.commands = _FakeCommands(
            install_result=install_result or _FakeCmdResult(),
            run_result=run_result or _FakeCmdResult(),
        )
        self.kill_raises = kill_raises
        self.killed = False
        self.timeout = timeout

    def kill(self) -> None:
        self.killed = True
        if self.kill_raises:
            raise RuntimeError("sandbox already terminated")


def _ok_envelope(name: str = "g", version: str = "0.1.0") -> str:
    """Mimic what the real driver writes to stdout."""
    report = {
        "grader_name": name,
        "grader_version": version,
        "n_rows": 2,
        "n_succeeded": 2,
        "n_failed": 0,
        "mean_score": 0.5,
        "std_score": 0.5,
        "min_score": 0.0,
        "max_score": 1.0,
        "results": [
            {"row_id": "r0", "score": 1.0, "explanation": None, "components": {}},
            {"row_id": "r1", "score": 0.0, "explanation": None, "components": {}},
        ],
        "failures": [],
    }
    return json.dumps(
        {"kind": "ok", "name": name, "version": version, "report": report}
    ) + "\n"


def _make_executor(
    sandbox: _FakeSandbox,
    tmp_path: Path,
) -> E2BExecutor:
    """Build an E2BExecutor wired to a fake sandbox + a fake wheel file.

    Skipping the real wheel build keeps the test offline and fast — the
    wheel-build path is exercised separately in the live smoke.
    """
    wheel = tmp_path / "easytrain_sdk-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"fake-wheel-bytes")
    return E2BExecutor(
        sandbox_factory=lambda timeout: sandbox,
        sdk_wheel_path=wheel,
    )


class TestE2BExecutor:
    def test_happy_path_assembles_payload_and_parses_output(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        sandbox = _FakeSandbox(run_result=_FakeCmdResult(stdout=_ok_envelope("g", "0.1.0")))
        executor = _make_executor(sandbox, tmp_path)

        result = executor.execute(OK_GRADER, rows)

        # Files written: driver, source, wheel.
        assert E2BExecutor.DRIVER_PATH in sandbox.files.writes
        assert E2BExecutor.SOURCE_PATH in sandbox.files.writes
        assert E2BExecutor.WHEEL_PATH in sandbox.files.writes
        assert sandbox.files.writes[E2BExecutor.SOURCE_PATH] == OK_GRADER
        assert sandbox.files.writes[E2BExecutor.WHEEL_PATH] == b"fake-wheel-bytes"

        # Two commands: pip install, then python driver.
        assert len(sandbox.commands.calls) == 2
        assert sandbox.commands.calls[0]["cmd"].startswith("pip install")
        run_call = sandbox.commands.calls[1]
        assert run_call["cmd"] == f"python {E2BExecutor.DRIVER_PATH}"

        # Stdin was a JSON payload with source + rows.
        payload = json.loads(run_call["stdin"])
        assert payload["source"] == OK_GRADER
        assert [r["id"] for r in payload["rows"]] == ["r0", "r1"]

        # Result reflects the parsed envelope.
        assert result.name == "g"
        assert result.version == "0.1.0"
        assert result.report.n_rows == 2

        # Sandbox killed after happy path.
        assert sandbox.killed

    def test_pip_install_failure_raises_sandbox_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        sandbox = _FakeSandbox(
            install_result=_FakeCmdResult(exit_code=1, stderr="connection refused"),
        )
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(SandboxError, match="pip install failed"):
            executor.execute(OK_GRADER, rows)

        # Driver shouldn't run if install fails.
        assert len(sandbox.commands.calls) == 1
        # Cleanup still happens.
        assert sandbox.killed

    def test_driver_nonzero_exit_raises_sandbox_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        sandbox = _FakeSandbox(
            run_result=_FakeCmdResult(exit_code=137, stderr="OOMKilled"),
        )
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(SandboxError, match="exited 137"):
            executor.execute(OK_GRADER, rows)
        assert sandbox.killed

    def test_load_error_envelope_raises_grader_load_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        envelope = json.dumps({"kind": "load_error", "message": "syntax error at line 2"}) + "\n"
        sandbox = _FakeSandbox(run_result=_FakeCmdResult(stdout=envelope))
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(GraderLoadError, match="syntax error at line 2"):
            executor.execute(OK_GRADER, rows)
        assert sandbox.killed

    def test_garbage_stdout_raises_sandbox_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        sandbox = _FakeSandbox(run_result=_FakeCmdResult(stdout="not json at all"))
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(SandboxError, match="not JSON"):
            executor.execute(OK_GRADER, rows)
        assert sandbox.killed

    def test_empty_stdout_raises_sandbox_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        sandbox = _FakeSandbox(run_result=_FakeCmdResult(stdout=""))
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(SandboxError, match="no JSON output"):
            executor.execute(OK_GRADER, rows)
        assert sandbox.killed

    def test_unknown_envelope_kind_raises_sandbox_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        envelope = json.dumps({"kind": "wat"}) + "\n"
        sandbox = _FakeSandbox(run_result=_FakeCmdResult(stdout=envelope))
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(SandboxError, match="unexpected envelope"):
            executor.execute(OK_GRADER, rows)
        assert sandbox.killed

    def test_kill_failure_does_not_mask_real_error(
        self, rows: list[EvalRow], tmp_path: Path
    ) -> None:
        sandbox = _FakeSandbox(
            run_result=_FakeCmdResult(exit_code=1, stderr="bad"),
            kill_raises=True,
        )
        executor = _make_executor(sandbox, tmp_path)

        with pytest.raises(SandboxError, match="exited 1"):
            executor.execute(OK_GRADER, rows)
        assert sandbox.killed  # was attempted


# ---------- build_from_env ----------


class TestBuildFromEnv:
    def test_default_is_in_process(self) -> None:
        executor = build_from_env(env={})
        assert isinstance(executor, InProcessExecutor)

    def test_explicit_in_process(self) -> None:
        executor = build_from_env(env={"EASYTRAIN_GRADER_EXECUTOR": "in_process"})
        assert isinstance(executor, InProcessExecutor)

    def test_aliases_accepted(self) -> None:
        for alias in ("inprocess", "local", "IN_PROCESS"):
            executor = build_from_env(env={"EASYTRAIN_GRADER_EXECUTOR": alias})
            assert isinstance(executor, InProcessExecutor)

    def test_e2b_requires_extra(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # E2B SDK isn't installed in the dev workspace, so building one
        # surfaces a clean ImportError naming the extra.
        with pytest.raises(ImportError, match="e2b-code-interpreter"):
            build_from_env(env={"EASYTRAIN_GRADER_EXECUTOR": "e2b"})

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown EASYTRAIN_GRADER_EXECUTOR"):
            build_from_env(env={"EASYTRAIN_GRADER_EXECUTOR": "modal"})
