"""Pydantic models describing a grader-vs-grader diff report.

These structures are returned by `diff_graders` and consumed by the CLI
formatter as well as (eventually) the editor's diff panel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Classification = Literal["unchanged", "promoted", "demoted"]
FailureSide = Literal["v1", "v2", "both"]


class RowDiff(BaseModel):
    """Per-row scoring delta between two grader versions."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    v1_score: float
    v2_score: float
    delta: float
    classification: Classification


class RowDiffFailure(BaseModel):
    """Captures a row where one (or both) graders raised an exception.

    Failed rows are excluded from summary statistics. We keep the error
    message (not a full traceback) because the differ runs both graders
    in-process and we don't want to leak unrelated stack frames into UI.
    """

    model_config = ConfigDict(extra="forbid")

    row_id: str
    side: FailureSide
    error_type: str
    error_message: str


class DiffReport(BaseModel):
    """Aggregate result of running two grader versions on the same eval set.

    Summary statistics (`mean_delta`, `median_delta`, `p95_delta`,
    `max_promotion`, `max_demotion`) are computed only across rows where
    both graders returned a `GraderResult`. Rows in `failures` are not
    counted in `n_changed`/`n_promoted`/`n_demoted` either.
    """

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_changed: int
    n_promoted: int
    n_demoted: int
    mean_delta: float
    median_delta: float
    p95_delta: float
    max_promotion: float
    max_demotion: float
    per_row: list[RowDiff] = Field(default_factory=list)
    failures: list[RowDiffFailure] = Field(default_factory=list)
