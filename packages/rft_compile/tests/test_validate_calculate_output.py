"""Direct tests of the calculate_output expression validator."""

from __future__ import annotations

import pytest
from easytrain_rft import validate_calculate_output

VALID_NAMES = {"a", "b"}


@pytest.mark.parametrize(
    "expr",
    [
        "a + b",
        "0.5 * a + 0.5 * b",
        "(a + b) / 2",
        "a",
        "1.0",
        "-a + b",
        "a - b",
        "(a) * (b)",
    ],
)
def test_valid_expressions(expr: str) -> None:
    validate_calculate_output(expr, VALID_NAMES)


def test_rejects_dunder_import() -> None:
    with pytest.raises(ValueError):
        validate_calculate_output("__import__('os')", VALID_NAMES)


def test_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown sub-grader"):
        validate_calculate_output("a + c", VALID_NAMES)


def test_rejects_pow_operator() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        validate_calculate_output("a ** 2", VALID_NAMES)


def test_rejects_eval_call() -> None:
    with pytest.raises(ValueError):
        validate_calculate_output("eval('a')", VALID_NAMES)


def test_rejects_modulo() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        validate_calculate_output("a % b", VALID_NAMES)


def test_rejects_floor_div() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        validate_calculate_output("a // b", VALID_NAMES)


def test_rejects_string_literal() -> None:
    with pytest.raises(ValueError):
        validate_calculate_output("'hello'", VALID_NAMES)


def test_rejects_attribute_access() -> None:
    with pytest.raises(ValueError):
        validate_calculate_output("a.b", VALID_NAMES)


def test_rejects_syntax_error() -> None:
    with pytest.raises(ValueError, match="not a valid expression"):
        validate_calculate_output("a +", VALID_NAMES)
