"""Verify counts and summary stats for a small hand-computed perturbation."""

from __future__ import annotations

from math import isclose

from easytrain_differ import diff_graders
from easytrain_sdk.types import EvalRow, GraderResult, Rollout


def make_v1(_rollout: Rollout) -> GraderResult:
    return GraderResult(score=0.5)


def make_v2_factory(eval_set: list[EvalRow]):
    """v2 returns 0.5 + 0.1 * (i%3 - 1) where i is the row index in eval_set."""
    index_by_id = {row.id: i for i, row in enumerate(eval_set)}

    def v2(rollout: Rollout) -> GraderResult:
        # Recover i from the prompt "q{i}", since Rollout has no id.
        i = int(rollout.prompt[1:])
        # Sanity: should match index_by_id by construction.
        assert index_by_id[str(i)] == i
        return GraderResult(score=0.5 + 0.1 * (i % 3 - 1))

    return v2


def test_perturbed_grader_counts_and_stats(eval_set_10):
    v2 = make_v2_factory(eval_set_10)
    report = diff_graders(make_v1, v2, eval_set_10)

    # Hand-computed: i%3 == 0 -> -0.1 (4 rows: i=0,3,6,9)
    #                i%3 == 1 -> 0.0  (3 rows: i=1,4,7)
    #                i%3 == 2 -> +0.1 (3 rows: i=2,5,8)
    assert report.n_rows == 10
    assert report.n_promoted == 3
    assert report.n_demoted == 4
    assert report.n_changed == 7  # promoted + demoted
    assert isclose(report.mean_delta, (3 * 0.1 + 4 * -0.1) / 10, abs_tol=1e-9)
    assert isclose(report.max_promotion, 0.1, abs_tol=1e-9)
    assert isclose(report.max_demotion, -0.1, abs_tol=1e-9)
    assert report.failures == []
    # Sanity on per-row classifications.
    classifications = {r.row_id: r.classification for r in report.per_row}
    assert classifications["0"] == "demoted"
    assert classifications["1"] == "unchanged"
    assert classifications["2"] == "promoted"
