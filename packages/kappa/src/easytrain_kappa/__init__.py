"""EasyTrain kappa — Cohen's kappa between a grader and a labeled reference set.

Public surface:
    compute_kappa     — the core entry point
    KappaReport       — result type
    KappaWeighting    — unweighted | linear | quadratic
    LabelSet          — named collection of labeled rows
    LabeledRow        — (row_id, label_score)
    RowKappa          — per-row contribution
    RowKappaFailure   — per-row exclusion record
    DEFAULT_BIN_EDGES — pass/fail default ([0.0, 0.5, 1.0])
    bucketize         — score -> bucket index
"""

from ._bucketize import DEFAULT_BIN_EDGES, bucketize
from ._kappa import compute_kappa
from ._types import (
    KappaReport,
    KappaWeighting,
    LabeledRow,
    LabelSet,
    RowKappa,
    RowKappaFailure,
)

__all__ = [
    "DEFAULT_BIN_EDGES",
    "KappaReport",
    "KappaWeighting",
    "LabelSet",
    "LabeledRow",
    "RowKappa",
    "RowKappaFailure",
    "bucketize",
    "compute_kappa",
]
