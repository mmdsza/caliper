"""EasyTrain Grader SDK.

Public surface:
    @grader                  — decorator that wraps a function into a typed grader
    Runner                   — executes graders against an eval set
    RunReport                — aggregate output of Runner.run()
    RowResult                — per-row success record
    RowFailure               — per-row failure record
    Rollout                  — input record
    GraderResult             — output record
    EvalRow                  — id + Rollout
    GraderCallable           — structural type for decorated graders
    load_grader_from_source  — compile + exec source, return the decorated grader
    LoadedGrader             — return type of load_grader_from_source
    GraderLoadError          — raised when source cannot be loaded
"""

from easytrain_sdk._decorator import grader
from easytrain_sdk._loader import GraderLoadError, LoadedGrader, load_grader_from_source
from easytrain_sdk._runner import RowFailure, RowResult, Runner, RunReport
from easytrain_sdk.types import EvalRow, GraderCallable, GraderResult, Rollout

__all__ = [
    "EvalRow",
    "GraderCallable",
    "GraderLoadError",
    "GraderResult",
    "LoadedGrader",
    "Rollout",
    "RowFailure",
    "RowResult",
    "RunReport",
    "Runner",
    "grader",
    "load_grader_from_source",
]
