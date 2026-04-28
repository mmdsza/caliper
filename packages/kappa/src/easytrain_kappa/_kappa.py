"""Cohen's kappa between a grader and a labeled reference set.

Implements the classical and weighted variants. For ordinal scores in
[0, 1] (the EasyTrain default), quadratic weighting is the right call —
it penalizes a 0-vs-1 mismatch quadratically more than a 0-vs-0.5
mismatch, matching how a labeler would feel about each kind of error.

The math (weighted form):

    kappa = 1 - sum_ij(w_ij * O_ij) / sum_ij(w_ij * E_ij)

where ``O_ij`` is the observed count for (grader bucket=i, label bucket=j),
``E_ij`` is the expected count under independence given the marginals, and
``w_ij`` is the disagreement weight. ``w_ii = 0`` (diagonal is free).

The unweighted form falls out of this with ``w_ij = 0 if i==j else 1``.

For ``len(buckets) == 1`` the metric is undefined (no possible
disagreement). For perfectly chance-distributed observations
``expected == observed`` so the denominator is zero — also undefined.
Both cases set ``kappa=0.0, kappa_undefined=True`` and force callers to
notice via the flag.
"""

from __future__ import annotations

from collections.abc import Callable

from easytrain_sdk.types import EvalRow, GraderResult, Rollout

from ._bucketize import (
    DEFAULT_BIN_EDGES,
    bucketize,
    n_buckets,
    validate_bin_edges,
)
from ._types import (
    KappaReport,
    KappaWeighting,
    LabelSet,
    RowKappa,
    RowKappaFailure,
)


def _weight_matrix(n: int, weighting: KappaWeighting) -> list[list[float]]:
    """Build the ``n x n`` disagreement-weight matrix.

    Diagonal is always zero. Off-diagonal is normalized by ``(n-1)``
    (linear) or ``(n-1)**2`` (quadratic) so the worst possible
    disagreement weighs exactly 1.0 for any bucket count — keeps kappa on
    a comparable scale across different bin choices.
    """
    if weighting is KappaWeighting.UNWEIGHTED:
        return [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]

    denom = (n - 1) if n > 1 else 1
    if weighting is KappaWeighting.LINEAR:
        return [[abs(i - j) / denom for j in range(n)] for i in range(n)]
    # Quadratic
    denom_sq = denom * denom
    return [[((i - j) ** 2) / denom_sq for j in range(n)] for i in range(n)]


def _safe_call(
    grader: Callable[[Rollout], GraderResult], rollout: Rollout
) -> tuple[GraderResult | None, BaseException | None]:
    """Mirror differ._safe_call: catch broadly, return (result, exc) tuple."""
    try:
        return grader(rollout), None
    except Exception as exc:
        # User-supplied code; one bad row should not abort the run.
        return None, exc


def compute_kappa(
    grader: Callable[[Rollout], GraderResult],
    eval_set: list[EvalRow],
    labels: LabelSet,
    *,
    bin_edges: list[float] | None = None,
    weighting: KappaWeighting = KappaWeighting.QUADRATIC,
) -> KappaReport:
    """Compute Cohen's kappa between a grader's scores and a label set.

    Only rows that appear both in ``eval_set`` and ``labels`` (matched by
    ``row_id``) participate in the computation. Rows where the grader
    raises are surfaced in :attr:`KappaReport.failures` and excluded from
    the kappa numerator.

    The function is deterministic: input order is preserved everywhere
    that order is observable (per_row, failures), and confusion matrix
    indices follow ``bin_edges`` order.
    """
    edges = list(bin_edges) if bin_edges is not None else list(DEFAULT_BIN_EDGES)
    validate_bin_edges(edges)
    n = n_buckets(edges)

    label_by_id = {row.row_id: row for row in labels.rows}
    confusion = [[0 for _ in range(n)] for _ in range(n)]
    per_row: list[RowKappa] = []
    failures: list[RowKappaFailure] = []
    n_labeled = 0

    for eval_row in eval_set:
        labeled = label_by_id.get(eval_row.id)
        if labeled is None:
            continue
        n_labeled += 1

        result, err = _safe_call(grader, eval_row.rollout)
        if err is not None:
            failures.append(
                RowKappaFailure(
                    row_id=eval_row.id,
                    error_type=type(err).__name__,
                    error_message=str(err),
                )
            )
            continue

        assert result is not None
        try:
            grader_bucket = bucketize(result.score, edges)
            label_bucket = bucketize(labeled.label_score, edges)
        except ValueError as exc:
            # Out-of-range score on either side. Treat as a row failure
            # rather than aborting the whole run — the SDET workflow is
            # "find the broken rows", not "give up at the first one".
            failures.append(
                RowKappaFailure(
                    row_id=eval_row.id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue

        confusion[grader_bucket][label_bucket] += 1
        per_row.append(
            RowKappa(
                row_id=eval_row.id,
                grader_score=result.score,
                label_score=labeled.label_score,
                grader_bucket=grader_bucket,
                label_bucket=label_bucket,
            )
        )

    n_scored = sum(sum(row) for row in confusion)
    weights = _weight_matrix(n, weighting)

    kappa, observed_agreement, expected_agreement, undefined = _kappa_from_confusion(
        confusion, weights, n_scored
    )

    return KappaReport(
        weighting=weighting,
        bin_edges=edges,
        n_eval_rows=len(eval_set),
        n_labeled=n_labeled,
        n_scored=n_scored,
        n_unlabeled=len(eval_set) - n_labeled,
        kappa=kappa,
        kappa_undefined=undefined,
        observed_agreement=observed_agreement,
        expected_agreement=expected_agreement,
        confusion=confusion,
        per_row=per_row,
        failures=failures,
    )


def _kappa_from_confusion(
    confusion: list[list[int]],
    weights: list[list[float]],
    n_scored: int,
) -> tuple[float, float, float, bool]:
    """Compute (kappa, observed_agreement, expected_agreement, undefined).

    Observed and expected agreement are reported in the *unweighted*
    sense (fraction of diagonal mass) so they stay interpretable
    independent of the weighting choice. Kappa itself uses the weights.
    """
    n = len(confusion)

    if n_scored == 0:
        # No data at all — nothing to say.
        return 0.0, 0.0, 0.0, True

    # Marginals.
    row_marginals = [sum(confusion[i]) for i in range(n)]
    col_marginals = [sum(confusion[i][j] for i in range(n)) for j in range(n)]

    # Observed agreement (unweighted, for reporting).
    diag = sum(confusion[i][i] for i in range(n))
    observed_agreement = diag / n_scored

    # Expected agreement under independence (unweighted, for reporting).
    expected_diag = sum(
        (row_marginals[i] * col_marginals[i]) / n_scored for i in range(n)
    )
    expected_agreement = expected_diag / n_scored

    if n == 1:
        # Single bucket — no possible disagreement; kappa undefined.
        return 0.0, observed_agreement, expected_agreement, True

    # Weighted numerator and denominator.
    # Numerator = sum w_ij * O_ij      (observed disagreement, weighted)
    # Denominator = sum w_ij * E_ij    (expected disagreement, weighted)
    # kappa = 1 - num / denom
    num = 0.0
    denom = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = weights[i][j]
            o = confusion[i][j]
            e = (row_marginals[i] * col_marginals[j]) / n_scored
            num += w * o
            denom += w * e

    if denom == 0.0:
        # Either everything is on the diagonal AND the marginals collapse
        # so expected disagreement is zero, or some pathological all-in-
        # one-row/col case. Either way kappa is undefined.
        return 0.0, observed_agreement, expected_agreement, True

    kappa = 1.0 - num / denom
    return kappa, observed_agreement, expected_agreement, False


__all__ = ["compute_kappa"]
