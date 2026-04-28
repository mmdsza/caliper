"""In-process grader runner.

The ``Runner`` executes a single grader against an eval set, collects per-row
results and per-row failures, and produces a ``RunReport`` with summary
statistics. Determinism is preserved by iterating the eval set in input order
and never touching wall-clock or RNG state.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from easytrain_sdk.types import EvalRow, GraderCallable


class RowResult(BaseModel):
    """One successful grader invocation."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    score: float
    explanation: str | None = None
    components: dict[str, float] = Field(default_factory=dict)


class RowFailure(BaseModel):
    """One grader invocation that raised."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    error_type: str
    error_message: str


class RunReport(BaseModel):
    """Aggregate report from running a grader over an eval set."""

    model_config = ConfigDict(extra="forbid")

    grader_name: str
    grader_version: str
    n_rows: int
    n_succeeded: int
    n_failed: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    results: list[RowResult] = Field(default_factory=list)
    failures: list[RowFailure] = Field(default_factory=list)


def _summarize(scores: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, std, min, max) using a population std.

    Empty input yields zeros — callers carry the n_succeeded count themselves
    so an all-zeros summary is unambiguous.
    """
    if not scores:
        return 0.0, 0.0, 0.0, 0.0
    n = len(scores)
    mean = sum(scores) / n
    if n == 1:
        std = 0.0
    else:
        # Population std (N divisor). Matches numpy.std default.
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(variance)
    return mean, std, min(scores), max(scores)


class Runner:
    """Bind a grader and execute it over eval sets."""

    def __init__(self, grader: GraderCallable) -> None:
        self.grader = grader
        # Cache name/version so the report stays stable even if the grader
        # is mutated between construction and run().
        self._name: str = getattr(grader, "name", grader.__class__.__name__)
        self._version: str = getattr(grader, "version", "0.0.0")

    def run(self, eval_set: list[EvalRow]) -> RunReport:
        results: list[RowResult] = []
        failures: list[RowFailure] = []

        for row in eval_set:
            try:
                graded = self.grader(row.rollout)
            except Exception as exc:
                failures.append(
                    RowFailure(
                        row_id=row.id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue

            results.append(
                RowResult(
                    row_id=row.id,
                    score=graded.score,
                    explanation=graded.explanation,
                    components=dict(graded.components),
                )
            )

        scores = [r.score for r in results]
        mean, std, lo, hi = _summarize(scores)

        return RunReport(
            grader_name=self._name,
            grader_version=self._version,
            n_rows=len(eval_set),
            n_succeeded=len(results),
            n_failed=len(failures),
            mean_score=mean,
            std_score=std,
            min_score=lo,
            max_score=hi,
            results=results,
            failures=failures,
        )


__all__: list[str] = [
    "RowFailure",
    "RowResult",
    "RunReport",
    "Runner",
]
