"""Pluggable grader executor.

The dry-run server delegates `(source, rows) -> RunReport` to an executor.
Two implementations:

* :class:`InProcessExecutor` — exec()s the source in this Python process.
  Fast, deterministic, offline. Suitable when the user IS the author
  (solo dev, local CI). Default for ``uv run pytest`` so the suite stays
  hermetic.

* :class:`E2BExecutor` — runs each grader inside a Firecracker microVM via
  the `e2b-code-interpreter` SDK. Suitable for shared/team workflows where
  graders may come from teammates or untrusted sources. Requires
  ``E2B_API_KEY`` and the optional ``e2b`` extra.

Selection is by environment:

    EASYTRAIN_GRADER_EXECUTOR=in_process  # default
    EASYTRAIN_GRADER_EXECUTOR=e2b
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from easytrain_sdk import (
    EvalRow,
    GraderLoadError,
    Runner,
    RunReport,
    load_grader_from_source,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Imported lazily inside _build_sdk_wheel so unit tests don't need uv on PATH.
_DEFAULT_E2B_TIMEOUT_S = 300


class SandboxError(RuntimeError):
    """Raised when the sandbox itself fails (process crash, bad output, etc.)."""


@dataclass(frozen=True)
class ExecutorResult:
    name: str
    version: str
    report: RunReport


class GraderExecutor(Protocol):
    """Strategy interface for running a grader against rows.

    Implementations must raise :class:`GraderLoadError` for source-level
    failures (syntax, missing decorator, multiple graders) and
    :class:`SandboxError` for executor-level failures.
    """

    def execute(self, source: str, rows: list[EvalRow]) -> ExecutorResult: ...


class InProcessExecutor:
    """Runs the grader in this Python process.

    Mirrors the Sprint 0 behavior bit-for-bit: ``load_grader_from_source``
    plus ``Runner(grader).run(rows)``. Kept as the default so offline tests
    don't require an E2B API key.
    """

    def execute(self, source: str, rows: list[EvalRow]) -> ExecutorResult:
        loaded = load_grader_from_source(source)
        report = Runner(loaded.grader).run(rows)
        return ExecutorResult(name=loaded.name, version=loaded.version, report=report)


class E2BExecutor:
    """Runs the grader inside an E2B Firecracker sandbox.

    The flow per ``execute`` call:

    1. Build a wheel of the workspace ``easytrain-sdk`` (cached across calls
       on this executor instance).
    2. Open a sandbox.
    3. Upload wheel + sandbox driver + user source.
    4. ``pip install`` the wheel.
    5. Run the driver, piping ``{source, rows}`` JSON on stdin.
    6. Parse the driver's stdout into an :class:`ExecutorResult`.
    7. Kill the sandbox (always — even on error).

    Cold start is dominated by ``pip install`` (~10-20s); per-grader
    iteration is fast. Pooling / a custom E2B template are deferred - see
    ADR-0004.
    """

    DRIVER_PATH = "/tmp/easytrain_driver.py"  # noqa: S108 - sandbox tmpfs, not host
    SOURCE_PATH = "/tmp/easytrain_grader.py"  # noqa: S108 - sandbox tmpfs, not host
    WHEEL_PATH = "/tmp/easytrain_sdk.whl"  # noqa: S108 - sandbox tmpfs, not host

    def __init__(
        self,
        *,
        sandbox_factory: object | None = None,
        sdk_wheel_path: Path | None = None,
        timeout_s: int = _DEFAULT_E2B_TIMEOUT_S,
    ) -> None:
        # `sandbox_factory` is injected for tests; in production we look up
        # `e2b_code_interpreter.Sandbox` lazily so the import isn't required
        # unless the executor is actually selected.
        if sandbox_factory is None:
            try:
                from e2b_code_interpreter import Sandbox  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "E2BExecutor requires the `e2b-code-interpreter` extra. "
                    "Install with: pip install 'easytrain-server[e2b]'"
                ) from e
            sandbox_factory = Sandbox
        self._sandbox_factory = sandbox_factory
        self._sdk_wheel_path = sdk_wheel_path
        self._timeout_s = timeout_s

    def _ensure_sdk_wheel(self) -> Path:
        if self._sdk_wheel_path is not None and self._sdk_wheel_path.exists():
            return self._sdk_wheel_path
        wheel = _build_sdk_wheel()
        # Cache on the instance — Path is immutable but the dataclass isn't frozen.
        self._sdk_wheel_path = wheel
        return wheel

    def execute(self, source: str, rows: list[EvalRow]) -> ExecutorResult:
        driver_src = _read_driver_source()
        wheel = self._ensure_sdk_wheel()

        sandbox = self._sandbox_factory(timeout=self._timeout_s)  # type: ignore[operator]
        try:
            sandbox.files.write(self.DRIVER_PATH, driver_src)
            sandbox.files.write(self.SOURCE_PATH, source)
            with wheel.open("rb") as f:
                sandbox.files.write(self.WHEEL_PATH, f.read())

            install = sandbox.commands.run(
                f"pip install --quiet {self.WHEEL_PATH}",
                timeout=self._timeout_s,
            )
            if install.exit_code != 0:
                raise SandboxError(
                    f"sandbox pip install failed (exit {install.exit_code}): "
                    f"{install.stderr or install.stdout!r}"
                )

            payload = json.dumps(
                {"source": source, "rows": [row.model_dump() for row in rows]}
            )
            run = sandbox.commands.run(
                f"python {self.DRIVER_PATH}",
                stdin=payload,
                timeout=self._timeout_s,
            )
            if run.exit_code != 0:
                raise SandboxError(
                    f"sandbox driver exited {run.exit_code}: {run.stderr or run.stdout!r}"
                )

            return _parse_driver_output(run.stdout)
        finally:
            try:
                sandbox.kill()
            except Exception:  # noqa: S110 - cleanup must not mask the real error
                pass


def build_from_env(env: dict[str, str] | None = None) -> GraderExecutor:
    """Pick an executor based on ``EASYTRAIN_GRADER_EXECUTOR``.

    Defaults to :class:`InProcessExecutor` so ``uv run pytest`` works
    without any external setup.
    """
    chosen = (env or os.environ).get("EASYTRAIN_GRADER_EXECUTOR", "in_process").strip().lower()
    if chosen in {"", "in_process", "inprocess", "local"}:
        return InProcessExecutor()
    if chosen == "e2b":
        return E2BExecutor()
    raise ValueError(
        f"unknown EASYTRAIN_GRADER_EXECUTOR={chosen!r}; expected 'in_process' or 'e2b'"
    )


def _parse_driver_output(stdout: str) -> ExecutorResult:
    """Parse the JSON envelope the sandbox driver writes to stdout.

    Driver contract (also implemented in `_sandbox_driver.py`):

        {"kind": "ok", "name": str, "version": str, "report": <RunReport.model_dump>}
        {"kind": "load_error", "message": str}

    Anything else is a sandbox-level fault.
    """
    line = _last_json_line(stdout)
    if line is None:
        raise SandboxError(f"sandbox driver produced no JSON output; stdout={stdout!r}")
    try:
        envelope = json.loads(line)
    except json.JSONDecodeError as e:
        raise SandboxError(f"sandbox driver stdout was not JSON: {line!r}") from e

    kind = envelope.get("kind")
    if kind == "load_error":
        raise GraderLoadError(envelope.get("message", "load error"))
    if kind != "ok":
        raise SandboxError(f"sandbox driver returned unexpected envelope: {envelope!r}")

    try:
        return ExecutorResult(
            name=envelope["name"],
            version=envelope["version"],
            report=RunReport.model_validate(envelope["report"]),
        )
    except KeyError as e:
        raise SandboxError(f"sandbox envelope missing field: {e}") from e


def _last_json_line(stdout: str) -> str | None:
    """The driver may emit warnings before the JSON envelope. Take the last
    non-empty line and trust it — driver contract is one envelope per run."""
    for line in reversed(_split_nonempty(stdout.splitlines())):
        return line
    return None


def _split_nonempty(lines: Iterable[str]) -> list[str]:
    return [ln for ln in lines if ln.strip()]


def _build_sdk_wheel() -> Path:
    """Build a wheel of the workspace `easytrain-sdk` package.

    Uses ``uv build`` if available, falling back to ``python -m build``.
    The wheel is written to a process-wide tempdir and reused.
    """
    sdk_dir = Path(__file__).resolve().parents[3] / "sdk"
    if not (sdk_dir / "pyproject.toml").exists():
        raise SandboxError(
            f"could not locate easytrain-sdk source at {sdk_dir} — "
            "E2BExecutor must run inside a workspace checkout"
        )

    out_dir = Path(tempfile.gettempdir()) / "easytrain-sandbox-wheels"
    out_dir.mkdir(exist_ok=True)

    builder = shutil.which("uv")
    if builder is not None:
        cmd = [builder, "build", "--wheel", "--out-dir", str(out_dir), str(sdk_dir)]
    else:  # pragma: no cover - uv is the supported tool
        cmd = ["python", "-m", "build", "--wheel", "--outdir", str(out_dir), str(sdk_dir)]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603 - args are constructed from trusted constants
    if proc.returncode != 0:
        raise SandboxError(f"failed to build easytrain-sdk wheel: {proc.stderr or proc.stdout}")

    wheels = sorted(out_dir.glob("easytrain_sdk-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SandboxError(f"build produced no wheel in {out_dir}")
    return wheels[-1]


def _driver_path() -> Path:
    return Path(__file__).resolve().parent / "_sandbox_driver.py"


def _read_driver_source() -> str:
    return _driver_path().read_text(encoding="utf-8")


__all__ = [
    "E2BExecutor",
    "ExecutorResult",
    "GraderExecutor",
    "InProcessExecutor",
    "SandboxError",
    "build_from_env",
]
