"""Exceptions in either grader should produce structured failure entries."""

from __future__ import annotations

from caliper_differ import diff_graders
from caliper_sdk.types import GraderResult, Rollout


def v1_raises_on_3_and_5(rollout: Rollout) -> GraderResult:
    i = int(rollout.prompt[1:])
    if i == 3:
        raise ValueError("v1 boom on row 3")
    if i == 5:
        raise RuntimeError("v1 boom on row 5")
    return GraderResult(score=0.5)


def v2_raises_on_5_and_7(rollout: Rollout) -> GraderResult:
    i = int(rollout.prompt[1:])
    if i == 5:
        raise ValueError("v2 boom on row 5")
    if i == 7:
        raise RuntimeError("v2 boom on row 7")
    return GraderResult(score=0.5)


def test_failure_handling_segregates_sides(eval_set_10):
    report = diff_graders(v1_raises_on_3_and_5, v2_raises_on_5_and_7, eval_set_10)

    # 3 failures: row 3 (v1 only), row 5 (both), row 7 (v2 only).
    assert report.n_rows == 10
    assert len(report.failures) == 3
    assert len(report.per_row) == 7  # 10 - 3 failures

    by_row = {f.row_id: f for f in report.failures}
    assert by_row["3"].side == "v1"
    assert by_row["3"].error_type == "ValueError"
    assert "row 3" in by_row["3"].error_message

    assert by_row["5"].side == "both"
    # `both` keeps v1's error per implementation contract.
    assert by_row["5"].error_type == "RuntimeError"

    assert by_row["7"].side == "v2"
    assert by_row["7"].error_type == "RuntimeError"
    assert "row 7" in by_row["7"].error_message

    # Both graders return 0.5 on succeeded rows -> all unchanged, zero stats.
    assert report.n_changed == 0
    assert report.n_promoted == 0
    assert report.n_demoted == 0
    assert report.mean_delta == 0.0
    assert report.median_delta == 0.0
    assert report.p95_delta == 0.0
    assert report.max_promotion == 0.0
    assert report.max_demotion == 0.0
