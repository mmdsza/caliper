"""Two identical graders should report zero changes."""

from __future__ import annotations

from easytrain_differ import diff_graders
from easytrain_sdk.types import GraderResult, Rollout


def constant_grader(rollout: Rollout) -> GraderResult:
    return GraderResult(score=0.42)


def test_identical_graders_produce_no_changes(eval_set_10):
    report = diff_graders(constant_grader, constant_grader, eval_set_10)

    assert report.n_rows == 10
    assert report.n_changed == 0
    assert report.n_promoted == 0
    assert report.n_demoted == 0
    assert report.mean_delta == 0.0
    assert report.median_delta == 0.0
    assert report.p95_delta == 0.0
    assert report.max_promotion == 0.0
    assert report.max_demotion == 0.0
    assert len(report.per_row) == 10
    assert all(r.classification == "unchanged" for r in report.per_row)
    assert report.failures == []
