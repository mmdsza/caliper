"""Core differ: run two grader versions on an eval set and summarise deltas."""

from __future__ import annotations

import statistics
from collections.abc import Callable

from easytrain_sdk.types import EvalRow, GraderResult, Rollout

from ._report import DiffReport, RowDiff, RowDiffFailure

# Tolerance for treating two scores as equal. Below this, we classify the
# row as "unchanged" rather than promoted/demoted. 1e-6 is comfortably
# below the precision most graders care about while still cleaning up
# float noise from any arithmetic the grader might do internally.
_UNCHANGED_TOL = 1e-6


def _safe_call(
    grader: Callable[[Rollout], GraderResult], rollout: Rollout
) -> tuple[GraderResult | None, BaseException | None]:
    """Invoke a grader, returning either the result or the captured exception.

    We deliberately catch `BaseException` so a misbehaving grader that
    raises e.g. `KeyboardInterrupt` inside its body doesn't take down a
    long batch run. (Real Ctrl-C from the user still works because
    pytest/CLI invokes us at a higher level.)
    """
    try:
        return grader(rollout), None
    except Exception as exc:
        # Intentionally broad: a grader is user-supplied code; we want
        # one bad row to surface as a failure entry, not abort the run.
        return None, exc


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over a pre-sorted list.

    `pct` is in [0, 1]. Matches the "linear" / type-7 convention used by
    NumPy's default `np.percentile`. We roll our own instead of using
    `statistics.quantiles` so the behaviour is well-defined for tiny
    samples and we don't have to special-case n==1 (quantiles raises).
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    rank = pct * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def diff_graders(
    grader_v1: Callable[[Rollout], GraderResult],
    grader_v2: Callable[[Rollout], GraderResult],
    eval_set: list[EvalRow],
) -> DiffReport:
    """Run two graders against the same eval set and return a `DiffReport`.

    Each grader is invoked exactly once per row. Exceptions from either
    grader are captured as `RowDiffFailure`s; failed rows do not
    contribute to summary statistics. The returned report is
    deterministic for fixed inputs.
    """
    per_row: list[RowDiff] = []
    failures: list[RowDiffFailure] = []

    for row in eval_set:
        v1_res, v1_err = _safe_call(grader_v1, row.rollout)
        v2_res, v2_err = _safe_call(grader_v2, row.rollout)

        if v1_err is not None or v2_err is not None:
            if v1_err is not None and v2_err is not None:
                side = "both"
                err = v1_err  # surface v1's error for both-failed rows
            elif v1_err is not None:
                side = "v1"
                err = v1_err
            else:
                side = "v2"
                assert v2_err is not None
                err = v2_err
            failures.append(
                RowDiffFailure(
                    row_id=row.id,
                    side=side,
                    error_type=type(err).__name__,
                    error_message=str(err),
                )
            )
            continue

        assert v1_res is not None and v2_res is not None
        delta = v2_res.score - v1_res.score
        if abs(delta) <= _UNCHANGED_TOL:
            classification = "unchanged"
        elif delta > 0:
            classification = "promoted"
        else:
            classification = "demoted"
        per_row.append(
            RowDiff(
                row_id=row.id,
                v1_score=v1_res.score,
                v2_score=v2_res.score,
                delta=delta,
                classification=classification,
            )
        )

    deltas = [r.delta for r in per_row]
    n_changed = sum(1 for r in per_row if r.classification != "unchanged")
    n_promoted = sum(1 for r in per_row if r.classification == "promoted")
    n_demoted = sum(1 for r in per_row if r.classification == "demoted")

    if deltas:
        mean_delta = statistics.fmean(deltas)
        median_delta = statistics.median(deltas)
        max_promotion = max(deltas) if max(deltas) > 0 else 0.0
        max_demotion = min(deltas) if min(deltas) < 0 else 0.0
    else:
        mean_delta = 0.0
        median_delta = 0.0
        max_promotion = 0.0
        max_demotion = 0.0

    # p95 is computed over |delta| (magnitude of change), since the use
    # case is "how big are the worst-shifted rows". Below 2 successful
    # rows there's no meaningful percentile, so we explicitly return 0.0
    # — this keeps the type stable for downstream JSON consumers.
    if len(deltas) < 2:
        p95_delta = 0.0
    else:
        sorted_abs = sorted(abs(d) for d in deltas)
        p95_delta = _percentile(sorted_abs, 0.95)

    return DiffReport(
        n_rows=len(eval_set),
        n_changed=n_changed,
        n_promoted=n_promoted,
        n_demoted=n_demoted,
        mean_delta=mean_delta,
        median_delta=median_delta,
        p95_delta=p95_delta,
        max_promotion=max_promotion,
        max_demotion=max_demotion,
        per_row=per_row,
        failures=failures,
    )
