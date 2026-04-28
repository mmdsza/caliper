"""EasyTrain differ — compare two grader versions on the same eval set."""

from ._diff import diff_graders
from ._format import format_report
from ._report import DiffReport, RowDiff, RowDiffFailure

__all__ = [
    "DiffReport",
    "RowDiff",
    "RowDiffFailure",
    "diff_graders",
    "format_report",
]
