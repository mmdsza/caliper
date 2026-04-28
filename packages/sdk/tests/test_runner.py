"""Tests for the Runner / RunReport."""

from __future__ import annotations

import statistics

import pytest
from caliper_sdk import (
    EvalRow,
    GraderResult,
    Rollout,
    RowFailure,
    RowResult,
    Runner,
    RunReport,
    grader,
)


@grader(name="length_grader", version="1.0.0")
def length_grader(rollout: Rollout) -> float:
    """Score based on response length, normalized to [0, 1]. Deterministic."""
    return min(len(rollout.response) / 10.0, 1.0)


@grader(name="index_grader", version="1.0.0")
def index_grader(rollout: Rollout) -> GraderResult:
    """Score = i/10 where the response is "a{i}". Deterministic and ordered."""
    # response shape is "a{i}" — extract i
    i = int(rollout.response[1:])
    return GraderResult(
        score=i / 10.0,
        explanation=f"row index {i}",
        components={"index": float(i)},
    )


def test_run_report_basic_stats(eval_set_10: list[EvalRow]) -> None:
    runner = Runner(index_grader)
    report = runner.run(eval_set_10)

    expected_scores = [i / 10.0 for i in range(10)]

    assert isinstance(report, RunReport)
    assert report.grader_name == "index_grader"
    assert report.grader_version == "1.0.0"
    assert report.n_rows == 10
    assert report.n_succeeded == 10
    assert report.n_failed == 0

    # Compare against statistics module ground truth (population std, matches numpy.std).
    assert report.mean_score == pytest.approx(statistics.fmean(expected_scores))
    assert report.std_score == pytest.approx(statistics.pstdev(expected_scores))
    assert report.min_score == pytest.approx(min(expected_scores))
    assert report.max_score == pytest.approx(max(expected_scores))

    assert [r.row_id for r in report.results] == [f"row-{i}" for i in range(10)]
    assert [r.score for r in report.results] == pytest.approx(expected_scores)
    assert report.results[3].explanation == "row index 3"
    assert report.results[3].components == {"index": 3.0}


def test_run_report_isolates_failures(eval_set_10: list[EvalRow]) -> None:
    @grader(name="explodes_on_5", version="0.0.1")
    def explodes_on_5(rollout: Rollout) -> float:
        if rollout.response == "a5":
            raise RuntimeError("boom on row 5")
        return 0.5

    runner = Runner(explodes_on_5)
    report = runner.run(eval_set_10)

    assert report.n_rows == 10
    assert report.n_succeeded == 9
    assert report.n_failed == 1

    assert len(report.failures) == 1
    failure = report.failures[0]
    assert isinstance(failure, RowFailure)
    assert failure.row_id == "row-5"
    assert failure.error_type == "RuntimeError"
    assert "boom on row 5" in failure.error_message

    # The other 9 rows succeed cleanly, all scoring 0.5.
    assert {r.row_id for r in report.results} == {f"row-{i}" for i in range(10) if i != 5}
    assert all(r.score == 0.5 for r in report.results)
    assert report.mean_score == pytest.approx(0.5)
    assert report.std_score == pytest.approx(0.0)
    assert report.min_score == pytest.approx(0.5)
    assert report.max_score == pytest.approx(0.5)


def test_run_is_deterministic(eval_set_10: list[EvalRow]) -> None:
    runner = Runner(length_grader)
    a = runner.run(eval_set_10)
    b = runner.run(eval_set_10)

    # Pydantic equality compares field-by-field, so this is the strongest check.
    assert a.results == b.results
    assert a.failures == b.failures
    assert a == b


def test_run_preserves_row_order(eval_set_10: list[EvalRow]) -> None:
    runner = Runner(index_grader)
    # Reverse the eval set; results should mirror the input order.
    reversed_set = list(reversed(eval_set_10))
    report = runner.run(reversed_set)
    assert [r.row_id for r in report.results] == [f"row-{i}" for i in reversed(range(10))]


def test_empty_eval_set_returns_zero_summary() -> None:
    runner = Runner(length_grader)
    report = runner.run([])
    assert report.n_rows == 0
    assert report.n_succeeded == 0
    assert report.n_failed == 0
    assert report.mean_score == 0.0
    assert report.std_score == 0.0
    assert report.min_score == 0.0
    assert report.max_score == 0.0
    assert report.results == []
    assert report.failures == []


def test_row_result_has_expected_fields(eval_set_10: list[EvalRow]) -> None:
    runner = Runner(index_grader)
    report = runner.run(eval_set_10[:1])
    assert len(report.results) == 1
    row = report.results[0]
    assert isinstance(row, RowResult)
    assert row.row_id == "row-0"
    assert row.score == 0.0
    assert row.explanation == "row index 0"
    assert row.components == {"index": 0.0}


def test_runner_captures_grader_name_and_version() -> None:
    @grader(name="custom_one", version="3.1.4")
    def g(rollout: Rollout) -> float:
        return 1.0

    runner = Runner(g)
    report = runner.run([])
    assert report.grader_name == "custom_one"
    assert report.grader_version == "3.1.4"
