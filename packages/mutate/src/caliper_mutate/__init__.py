"""Caliper mutation-testing for graders.

Public surface (filled in by the build-out):
    Mutation, MutationExpectation        — Protocol + enum (types.py)
    MutationOutcome, MutationFailure,
    MutationSummary, MutationReport      — report shapes (report.py)
    Built-in mutations (mutations.py)    — DropFormatGate, WrongAnswer, ...
    run_mutations, format_mutation_report (runner.py)
"""

from caliper_mutate.format import format_mutation_report
from caliper_mutate.mutations import (
    AppendGarbage,
    CasePerturb,
    DropFormatGate,
    HardcodeGold,
    PrependSycophancy,
    TruncateResponse,
    WhitespacePerturb,
    WrongAnswer,
)
from caliper_mutate.report import (
    MutationFailure,
    MutationOutcome,
    MutationReport,
    MutationSummary,
)
from caliper_mutate.runner import run_mutations
from caliper_mutate.types import Mutation, MutationExpectation

__all__ = [
    "AppendGarbage",
    "CasePerturb",
    "DropFormatGate",
    "HardcodeGold",
    "Mutation",
    "MutationExpectation",
    "MutationFailure",
    "MutationOutcome",
    "MutationReport",
    "MutationSummary",
    "PrependSycophancy",
    "TruncateResponse",
    "WhitespacePerturb",
    "WrongAnswer",
    "format_mutation_report",
    "run_mutations",
]
