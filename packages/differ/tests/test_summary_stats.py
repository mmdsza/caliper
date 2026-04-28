"""Hand-constructed deltas exercise mean/median/p95/max_promotion/max_demotion."""

from __future__ import annotations

from math import isclose

from caliper_differ import diff_graders
from caliper_sdk.types import GraderResult, Rollout

# Indexed by row id "0".."9" — chosen so v1 + delta stays in [0, 1].
_DELTAS = [-0.4, -0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6]
_V1_BASE = 0.4


def v1(_rollout: Rollout) -> GraderResult:
    return GraderResult(score=_V1_BASE)


def v2(rollout: Rollout) -> GraderResult:
    i = int(rollout.prompt[1:])
    return GraderResult(score=_V1_BASE + _DELTAS[i])


def test_summary_stats_match_hand_computed(eval_set_10):
    report = diff_graders(v1, v2, eval_set_10)

    # mean = sum(deltas) / 10 = 1.4 / 10 = 0.14
    assert isclose(report.mean_delta, 0.14, abs_tol=1e-9)
    # sorted deltas median for n=10: avg of 5th and 6th -> (0.1 + 0.15)/2 = 0.125
    assert isclose(report.median_delta, 0.125, abs_tol=1e-9)
    # |delta| sorted: [0.0, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    # p95 with linear interpolation, rank = 0.95 * 9 = 8.55 -> 0.5 + 0.55*(0.6-0.5) = 0.555
    assert isclose(report.p95_delta, 0.555, abs_tol=1e-9)
    assert isclose(report.max_promotion, 0.6, abs_tol=1e-9)
    assert isclose(report.max_demotion, -0.4, abs_tol=1e-9)
    # 1 unchanged (delta 0.0 at i=2), so n_changed = 9.
    assert report.n_changed == 9
    assert report.n_promoted == 7  # +0.05, +0.1, +0.15, +0.2, +0.3, +0.5, +0.6
    assert report.n_demoted == 2  # -0.4, -0.1
    assert report.failures == []
