"""Partial-labeling and grader-failure handling for compute_kappa."""

from __future__ import annotations

from typing import Any

from easytrain_kappa import (
    LabeledRow,
    LabelSet,
    compute_kappa,
)


def _ls(name: str, **scores: float) -> LabelSet:
    return LabelSet(
        name=name,
        rows=[LabeledRow(row_id=k, label_score=v) for k, v in scores.items()],
    )


def test_partial_labels_only_score_labeled_rows(
    make_eval_row: Any, score_for_id: Any
) -> None:
    rows = [make_eval_row(f"r{i}") for i in range(6)]
    grader = score_for_id({f"r{i}": 1.0 for i in range(6)})
    labels = _ls("partial", r0=1.0, r2=1.0, r4=1.0)  # only 3 of 6 labeled

    report = compute_kappa(grader, rows, labels)

    assert report.n_eval_rows == 6
    assert report.n_labeled == 3
    assert report.n_unlabeled == 3
    assert report.n_scored == 3
    assert {p.row_id for p in report.per_row} == {"r0", "r2", "r4"}


def test_grader_exception_excluded_from_kappa(
    make_eval_row: Any, score_for_id: Any
) -> None:
    rows = [make_eval_row(f"r{i}") for i in range(4)]
    # r1 raises; r0/r2/r3 score normally.
    grader = score_for_id({"r0": 0.0, "r2": 1.0, "r3": 1.0})  # r1 missing -> KeyError
    labels = _ls("any", r0=0.0, r1=0.5, r2=1.0, r3=1.0)

    report = compute_kappa(grader, rows, labels)

    assert report.n_labeled == 4
    assert report.n_scored == 3
    assert len(report.failures) == 1
    assert report.failures[0].row_id == "r1"
    assert report.failures[0].error_type == "KeyError"
    # Only the 3 successful rows enter per_row.
    assert {p.row_id for p in report.per_row} == {"r0", "r2", "r3"}


def test_out_of_range_label_raises_at_validation_time() -> None:
    """LabeledRow.label_score is constrained to [0, 1] by Pydantic. Pushing
    a bad value should fail at construction, not silently corrupt kappa."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LabeledRow(row_id="r0", label_score=1.5)


def test_out_of_range_grader_score_becomes_failure(make_eval_row: Any) -> None:
    """A grader that somehow yields out-of-range (e.g. via a bug bypassing
    the decorator's validation) — currently impossible because @grader
    enforces [0,1] — but if it happened, bucketize would raise, and we
    surface it as a row failure. We simulate by skipping the @grader
    decorator and bypassing validation.
    """
    from easytrain_sdk import GraderResult, Rollout

    def raw_grader(rollout: Rollout) -> GraderResult:
        # Bypass GraderResult validation by using model_construct.
        return GraderResult.model_construct(score=1.5, explanation=None, components={}, metadata={})

    # Attach minimal grader-like attributes so the loader-style duck check
    # in compute_kappa's _safe_call doesn't matter — compute_kappa calls
    # the grader directly without any introspection.
    raw_grader.name = "raw"
    raw_grader.version = "0.0.0"

    rows = [make_eval_row("r0")]
    labels = _ls("any", r0=0.5)

    report = compute_kappa(raw_grader, rows, labels)
    assert report.n_scored == 0
    assert len(report.failures) == 1
    assert report.failures[0].row_id == "r0"
    assert "outside bin range" in report.failures[0].error_message
