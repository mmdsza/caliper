"""How weighting changes kappa: unweighted vs linear vs quadratic."""

from __future__ import annotations

from typing import Any

from easytrain_kappa import KappaWeighting, LabeledRow, LabelSet, compute_kappa
from easytrain_kappa._kappa import _weight_matrix

# 3-bucket bins so distance-1 vs distance-2 errors are distinguishable.
TERNARY_EDGES = [0.0, 0.34, 0.67, 1.0]


def _ls(name: str, **scores: float) -> LabelSet:
    return LabelSet(
        name=name,
        rows=[LabeledRow(row_id=k, label_score=v) for k, v in scores.items()],
    )


class TestWeightMatrix:
    """The matrix shape — separate from kappa, so we don't conflate bugs."""

    def test_unweighted_diag_zero_off_diag_one(self) -> None:
        m = _weight_matrix(3, KappaWeighting.UNWEIGHTED)
        assert m == [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]

    def test_linear_distance_normalized(self) -> None:
        m = _weight_matrix(3, KappaWeighting.LINEAR)
        # max distance = 2 (n-1), so off-diagonal corners weight 1.0.
        assert m[0][2] == 1.0
        assert m[2][0] == 1.0
        # Distance 1 weights = 1/2 = 0.5.
        assert m[0][1] == 0.5
        assert m[1][2] == 0.5
        # Diagonal still zero.
        assert m[0][0] == 0.0

    def test_quadratic_distance_squared_normalized(self) -> None:
        m = _weight_matrix(3, KappaWeighting.QUADRATIC)
        assert m[0][2] == 1.0
        # Distance 1 weights = 1/4 = 0.25.
        assert m[0][1] == 0.25
        assert m[1][2] == 0.25


class TestWeightingChangesKappa:
    """Same data, three weightings — assert they differ in the expected order."""

    def _setup_distance_1_dominant(
        self, make_eval_row: Any, score_for_id: Any
    ) -> tuple[Any, list[Any], LabelSet]:
        """A confusion matrix dominated by adjacent-bucket errors.

        With 6 rows ternary-bucketed:
        - 2 rows: bucket 0/0 (agree, low)
        - 1 row : bucket 1/0 (off by one)
        - 1 row : bucket 0/1 (off by one)
        - 2 rows: bucket 2/2 (agree, high)

        confusion =
            [[2, 1, 0],
             [1, 0, 0],
             [0, 0, 2]]
        Off-diagonal mass (2 rows) is entirely at distance 1.
        """
        ids = [f"r{i}" for i in range(6)]
        eval_rows = [make_eval_row(rid) for rid in ids]
        grader = score_for_id(
            {"r0": 0.1, "r1": 0.1, "r2": 0.5, "r3": 0.1, "r4": 0.9, "r5": 0.9}
        )
        labels = _ls(
            "near_errors",
            r0=0.1,
            r1=0.1,
            r2=0.1,  # label says low; grader said mid (distance-1 error)
            r3=0.5,  # label says mid; grader said low (distance-1 error)
            r4=0.9,
            r5=0.9,
        )
        return grader, eval_rows, labels

    def test_linear_kappa_lower_than_quadratic_for_near_errors(
        self, make_eval_row: Any, score_for_id: Any
    ) -> None:
        grader, eval_rows, labels = self._setup_distance_1_dominant(
            make_eval_row, score_for_id
        )

        unweighted = compute_kappa(
            grader,
            eval_rows,
            labels,
            bin_edges=TERNARY_EDGES,
            weighting=KappaWeighting.UNWEIGHTED,
        )
        linear = compute_kappa(
            grader,
            eval_rows,
            labels,
            bin_edges=TERNARY_EDGES,
            weighting=KappaWeighting.LINEAR,
        )
        quadratic = compute_kappa(
            grader,
            eval_rows,
            labels,
            bin_edges=TERNARY_EDGES,
            weighting=KappaWeighting.QUADRATIC,
        )

        # Sanity — confusion is the same across weightings.
        assert unweighted.confusion == linear.confusion == quadratic.confusion
        # Sanity — observed agreement is unchanged.
        assert (
            unweighted.observed_agreement
            == linear.observed_agreement
            == quadratic.observed_agreement
        )

        # When errors are exclusively at distance 1:
        # - Quadratic weights distance-1 errors as 1/(n-1)^2 = 1/4 of the
        #   maximum possible disagreement.
        # - Linear weights them as 1/(n-1) = 1/2.
        # So quadratic *under*weights distance-1 mistakes vs linear: the
        # quadratic kappa is *more forgiving*, i.e., closer to 1.
        assert quadratic.kappa > linear.kappa
        # And linear is more forgiving than unweighted (which treats every
        # off-diagonal cell as a full-magnitude error).
        assert linear.kappa > unweighted.kappa

    def test_corner_only_errors_quadratic_punishes_more_than_linear(
        self, make_eval_row: Any, score_for_id: Any
    ) -> None:
        """When observed errors are exclusively at distance 2 (corners),
        both linear and quadratic give those cells weight 1.0 — they
        agree on the *numerator*. The denominator differs:

        * Linear weights the (mostly distance-1) *expected* mass at 0.5.
        * Quadratic weights the same expected mass at 0.25.

        Smaller denominator => bigger num/denom => smaller kappa.
        Result: corner-only observed errors are punished *more* by
        quadratic than by linear. This is the canonical "ordinal
        agreement is harder than nominal" regime.
        """
        ids = [f"r{i}" for i in range(8)]
        eval_rows = [make_eval_row(rid) for rid in ids]
        # confusion = [[3, 0, 1], [0, 3, 0], [1, 0, 0]] — 8 rows total,
        # off-diagonal mass at the corners (distance 2) only.
        grader = score_for_id(
            {
                "r0": 0.1, "r1": 0.1, "r2": 0.1,
                "r3": 0.1,  # corner: grader low, label high
                "r4": 0.5, "r5": 0.5, "r6": 0.5,
                "r7": 0.9,  # corner: grader high, label low
            }
        )
        labels = _ls(
            "far_errors",
            r0=0.1, r1=0.1, r2=0.1,
            r3=0.9,
            r4=0.5, r5=0.5, r6=0.5,
            r7=0.1,
        )

        unweighted = compute_kappa(
            grader, eval_rows, labels,
            bin_edges=TERNARY_EDGES,
            weighting=KappaWeighting.UNWEIGHTED,
        )
        linear = compute_kappa(
            grader, eval_rows, labels,
            bin_edges=TERNARY_EDGES,
            weighting=KappaWeighting.LINEAR,
        )
        quadratic = compute_kappa(
            grader, eval_rows, labels,
            bin_edges=TERNARY_EDGES,
            weighting=KappaWeighting.QUADRATIC,
        )

        assert unweighted.confusion == [[3, 0, 1], [0, 3, 0], [1, 0, 0]]
        for r in (unweighted, linear, quadratic):
            assert not r.kappa_undefined
            assert -1.0 <= r.kappa <= 1.0

        # The ordering for corner-only errors is the OPPOSITE of the
        # near-diagonal case: quadratic < linear < unweighted.
        assert quadratic.kappa < linear.kappa
        assert linear.kappa < unweighted.kappa
