"""StringCheckGrader: all five operations compile, missing fields raise."""

from __future__ import annotations

import pytest
from caliper_rft import StringCheckGrader, compile_to_rft
from pydantic import ValidationError


@pytest.mark.parametrize(
    "operation",
    ["eq", "ne", "starts_with", "ends_with", "contains"],
)
def test_each_operation_compiles(operation: str) -> None:
    g = StringCheckGrader(operation=operation, reference="ref")
    out = compile_to_rft(g)
    assert out == {"type": "string_check", "operation": operation, "reference": "ref"}


def test_missing_reference_raises() -> None:
    with pytest.raises(ValidationError):
        StringCheckGrader(operation="eq")  # type: ignore[call-arg]


def test_missing_operation_raises() -> None:
    with pytest.raises(ValidationError):
        StringCheckGrader(reference="ref")  # type: ignore[call-arg]


def test_unknown_operation_raises() -> None:
    with pytest.raises(ValidationError):
        StringCheckGrader(operation="regex_match", reference="ref")  # type: ignore[arg-type]


def test_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        StringCheckGrader(  # type: ignore[call-arg]
            operation="eq",
            reference="ref",
            extra="nope",
        )
