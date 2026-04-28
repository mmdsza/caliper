"""Pydantic models for mutation-test outcomes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ViolationKind = Literal["score_increased", "score_decreased", "score_changed", "grader_raised"]


class MutationOutcome(BaseModel):
    """Per (row x mutation) result from running the grader on baseline and mutated rollouts."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    mutation: str
    baseline_score: float
    mutated_score: float
    delta: float  # mutated_score - baseline_score
    skipped_no_op: bool = False
    violation: ViolationKind | None = None
    note: str | None = None


class MutationFailure(BaseModel):
    """Per (row x mutation) hard failure (e.g. grader raised on baseline or mutated row)."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    mutation: str
    side: Literal["baseline", "mutated", "both"]
    error_type: str
    error_message: str


class MutationSummary(BaseModel):
    """Per-mutation aggregate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    expected_effect: str
    n_applied: int
    n_skipped_no_op: int
    n_violations: int
    n_failures: int
    passed: bool


class MutationReport(BaseModel):
    """Top-level mutation-test report for one grader against one eval set."""

    model_config = ConfigDict(extra="forbid")

    grader_name: str
    grader_version: str
    n_rows: int
    n_mutations: int
    passed: bool
    per_mutation: list[MutationSummary] = Field(default_factory=list)
    outcomes: list[MutationOutcome] = Field(default_factory=list)
    failures: list[MutationFailure] = Field(default_factory=list)
