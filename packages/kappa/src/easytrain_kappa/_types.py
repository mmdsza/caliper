"""Pydantic models for the Cohen's kappa report.

These structures are returned by :func:`compute_kappa` and consumed by
the smoke script as well as (eventually) the editor's "agreement with
labels" panel.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class KappaWeighting(StrEnum):
    """Weighting scheme for off-diagonal disagreements.

    * ``unweighted`` — classical Cohen's kappa. Any disagreement counts
      equally regardless of how far apart the two buckets are.
    * ``linear`` — disagreement penalty grows linearly with bucket distance.
    * ``quadratic`` — penalty grows with the square of bucket distance.
      Standard for ordinal data with a meaningful ordering (medical
      diagnosis grades, Kaggle ordinal-target competitions).
    """

    UNWEIGHTED = "unweighted"
    LINEAR = "linear"
    QUADRATIC = "quadratic"


class LabeledRow(BaseModel):
    """A single human-labeled reference score for an eval-set row.

    ``label_score`` is a float in [0, 1] using the same scale as a grader's
    output. Multiple :class:`LabelSet` instances can label the same eval
    set (per-labeler, golden, consensus).
    """

    model_config = ConfigDict(extra="forbid")

    row_id: str
    label_score: float = Field(ge=0.0, le=1.0)


class LabelSet(BaseModel):
    """A named collection of labeled rows.

    The name is metadata only — useful for "labeler-A vs labeler-B" or
    "human-consensus" comparisons in reports.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    rows: list[LabeledRow] = Field(default_factory=list)


class RowKappa(BaseModel):
    """Per-row contribution to the kappa computation.

    Carries both raw scores and the bucketized indices so a downstream UI
    can render either the heatmap (categorical) or the scatter (continuous).
    """

    model_config = ConfigDict(extra="forbid")

    row_id: str
    grader_score: float
    label_score: float
    grader_bucket: int
    label_bucket: int


class RowKappaFailure(BaseModel):
    """A row excluded from the kappa computation due to a grader exception."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    error_type: str
    error_message: str


class KappaReport(BaseModel):
    """Aggregate Cohen's kappa report.

    Coverage:

    * ``n_eval_rows`` — total rows in the input eval set.
    * ``n_labeled`` — eval rows with a matching :class:`LabeledRow`.
    * ``n_scored`` — labeled rows where the grader returned successfully
      (this is the denominator of the kappa computation).
    * ``n_unlabeled`` = ``n_eval_rows`` - ``n_labeled``.

    Stats:

    * ``kappa`` — the agreement statistic. Equal to ``observed - expected) /
      (1 - expected)``. By convention, returned as ``0.0`` when undefined.
    * ``kappa_undefined`` — ``True`` when the metric is mathematically
      undefined (everything in one bucket, or perfect chance agreement).
      Callers MUST check this flag rather than reading ``kappa`` blindly.
    * ``observed_agreement`` / ``expected_agreement`` — components, useful
      for debugging which side moved.
    * ``confusion`` — n_buckets x n_buckets matrix; ``confusion[i][j]`` is
      the count of rows where grader bucket = i and label bucket = j.
    """

    model_config = ConfigDict(extra="forbid")

    weighting: KappaWeighting
    bin_edges: list[float]
    n_eval_rows: int
    n_labeled: int
    n_scored: int
    n_unlabeled: int
    kappa: float
    kappa_undefined: bool
    observed_agreement: float
    expected_agreement: float
    confusion: list[list[int]]
    per_row: list[RowKappa] = Field(default_factory=list)
    failures: list[RowKappaFailure] = Field(default_factory=list)


__all__ = [
    "KappaReport",
    "KappaWeighting",
    "LabelSet",
    "LabeledRow",
    "RowKappa",
    "RowKappaFailure",
]
