"""load_grader_from_source: exec source, find @grader, edge cases."""

import pytest
from caliper_sdk import GraderLoadError, Rollout, load_grader_from_source

SAMPLE_GRADER_SOURCE = '''
from caliper_sdk import grader, Rollout, GraderResult


@grader(name="legal_citation_grader", version="0.1.0")
def legal_citation_grader(rollout: Rollout) -> GraderResult:
    response = rollout.response
    if not response.startswith("<think>"):
        return GraderResult(score=0.0, explanation="missing <think> gate")
    if rollout.gold is not None and rollout.gold in response:
        return GraderResult(score=1.0, explanation="gold present in response")
    return GraderResult(score=0.0, explanation="gold absent")
'''


def test_loads_simple_grader() -> None:
    loaded = load_grader_from_source(SAMPLE_GRADER_SOURCE)
    assert loaded.name == "legal_citation_grader"
    assert loaded.version == "0.1.0"
    assert callable(loaded.grader)


def test_syntax_error_reports_line() -> None:
    bad = "from caliper_sdk import grader\ndef foo(:\n    pass\n"
    with pytest.raises(GraderLoadError, match="syntax error at line"):
        load_grader_from_source(bad)


def test_runtime_error_during_import() -> None:
    bad = "raise ValueError('boom')"
    with pytest.raises(GraderLoadError, match="ValueError during import: boom"):
        load_grader_from_source(bad)


def test_no_grader_found() -> None:
    src = "x = 1\ndef plain():\n    return 0\n"
    with pytest.raises(GraderLoadError, match="no @grader-decorated function found"):
        load_grader_from_source(src)


def test_multiple_graders_rejected() -> None:
    src = """
from caliper_sdk import grader, Rollout, GraderResult

@grader
def first(r: Rollout) -> float:
    return 0.5

@grader
def second(r: Rollout) -> float:
    return 0.7
"""
    with pytest.raises(GraderLoadError, match="multiple graders"):
        load_grader_from_source(src)


def test_grader_with_explicit_name_and_version() -> None:
    src = """
from caliper_sdk import grader, Rollout, GraderResult

@grader(name="custom_name", version="2.5.0")
def whatever(r: Rollout) -> float:
    return 1.0
"""
    loaded = load_grader_from_source(src)
    assert loaded.name == "custom_name"
    assert loaded.version == "2.5.0"


def test_loaded_grader_runs() -> None:
    """End-to-end: load a grader, then actually invoke it on a Rollout."""
    src = """
from caliper_sdk import grader, Rollout, GraderResult

@grader
def hello(r: Rollout) -> float:
    return 1.0 if r.response == "hi" else 0.0
"""
    loaded = load_grader_from_source(src)
    assert loaded.grader(Rollout(prompt="q", response="hi")).score == 1.0
    assert loaded.grader(Rollout(prompt="q", response="bye")).score == 0.0
