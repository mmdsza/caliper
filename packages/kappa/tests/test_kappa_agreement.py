"""Cohen's kappa: identity / chance / total disagreement / undefined cases."""

from __future__ import annotations

from typing import Any

from caliper_kappa import (
    KappaWeighting,
    LabeledRow,
    LabelSet,
    compute_kappa,
)


def _ls(name: str, **scores: float) -> LabelSet:
    return LabelSet(
        name=name,
        rows=[LabeledRow(row_id=k, label_score=v) for k, v in scores.items()],
    )


def test_perfect_agreement_returns_kappa_one(
    make_eval_row: Any, score_for_id: Any
) -> None:
    """4 rows, two in each bucket, grader and labels match exactly => kappa=1.0."""
    rows = [make_eval_row(f"r{i}") for i in range(4)]
    grader = score_for_id({"r0": 0.0, "r1": 0.0, "r2": 1.0, "r3": 1.0})
    labels = _ls("perfect", r0=0.0, r1=0.0, r2=1.0, r3=1.0)

    report = compute_kappa(grader, rows, labels)

    assert report.kappa == 1.0
    assert report.kappa_undefined is False
    assert report.observed_agreement == 1.0
    assert report.expected_agreement == 0.5
    assert report.confusion == [[2, 0], [0, 2]]
    assert report.n_scored == 4
    assert report.n_labeled == 4
    assert report.n_unlabeled == 0


def test_total_binary_disagreement_returns_kappa_minus_one(
    make_eval_row: Any, score_for_id: Any
) -> None:
    """Every grader/label pair lands in the *wrong* bucket => kappa=-1.0."""
    rows = [make_eval_row(f"r{i}") for i in range(4)]
    grader = score_for_id({"r0": 0.0, "r1": 0.0, "r2": 1.0, "r3": 1.0})
    labels = _ls("flipped", r0=1.0, r1=1.0, r2=0.0, r3=0.0)

    report = compute_kappa(grader, rows, labels, weighting=KappaWeighting.UNWEIGHTED)

    assert report.kappa == -1.0
    assert report.kappa_undefined is False
    assert report.observed_agreement == 0.0
    assert report.confusion == [[0, 2], [2, 0]]


def test_chance_agreement_returns_kappa_zero(
    make_eval_row: Any, score_for_id: Any
) -> None:
    """Off-diagonal balanced exactly => observed = expected => kappa=0.0."""
    rows = [make_eval_row(f"r{i}") for i in range(4)]
    # confusion = [[1,1],[1,1]]: marginals are [2,2] on both axes,
    # observed = 0.5, expected = 0.5.
    grader = score_for_id({"r0": 0.0, "r1": 0.0, "r2": 1.0, "r3": 1.0})
    labels = _ls("balanced", r0=0.0, r1=1.0, r2=0.0, r3=1.0)

    report = compute_kappa(grader, rows, labels, weighting=KappaWeighting.UNWEIGHTED)

    assert abs(report.kappa - 0.0) < 1e-12
    assert report.confusion == [[1, 1], [1, 1]]
    assert report.observed_agreement == 0.5
    assert report.expected_agreement == 0.5


def test_single_bucket_input_kappa_undefined(
    make_eval_row: Any, score_for_id: Any
) -> None:
    """All scores in one bucket => no possible disagreement => undefined."""
    rows = [make_eval_row(f"r{i}") for i in range(3)]
    grader = score_for_id({"r0": 0.9, "r1": 0.8, "r2": 0.7})
    labels = _ls("all_pass", r0=0.95, r1=0.75, r2=0.85)
    # bin_edges = [0.5, 1.0] so there's only ONE bucket
    report = compute_kappa(grader, rows, labels, bin_edges=[0.5, 1.0])

    assert report.kappa == 0.0
    assert report.kappa_undefined is True
    assert report.confusion == [[3]]
    assert report.observed_agreement == 1.0
    assert report.expected_agreement == 1.0


def test_unanimous_diagonal_with_two_buckets_is_defined(
    make_eval_row: Any, score_for_id: Any
) -> None:
    """All rows on the diagonal but spanning >1 bucket => kappa=1.0, defined."""
    rows = [make_eval_row(f"r{i}") for i in range(4)]
    grader = score_for_id({"r0": 0.1, "r1": 0.2, "r2": 0.9, "r3": 0.8})
    labels = _ls("diag", r0=0.0, r1=0.3, r2=1.0, r3=0.7)

    report = compute_kappa(grader, rows, labels)

    assert report.kappa == 1.0
    assert report.kappa_undefined is False


def test_no_overlap_between_eval_set_and_labels_returns_undefined(
    make_eval_row: Any, score_for_id: Any
) -> None:
    rows = [make_eval_row(f"eval_{i}") for i in range(3)]
    grader = score_for_id({})  # never invoked because nothing matches
    labels = _ls("orphans", different_id_1=1.0, different_id_2=0.0)

    report = compute_kappa(grader, rows, labels)

    assert report.n_labeled == 0
    assert report.n_scored == 0
    assert report.kappa_undefined is True
