"""Caliper dry-run backend.

Public surface:
    create_app()                  — FastAPI app factory
    load_grader_from_source(s)    — exec source, return the (single) decorated grader
    eval_sets                     — built-in fixture registry
"""

from caliper_server.app import create_app
from caliper_server.grader_loader import GraderLoadError, LoadedGrader, load_grader_from_source

__all__ = ["GraderLoadError", "LoadedGrader", "create_app", "load_grader_from_source"]
