"""Shared fixtures for mutation tests."""

from __future__ import annotations

import pytest
from caliper_sdk import EvalRow, Rollout


@pytest.fixture
def formatted_rollout() -> Rollout:
    """A rollout with a `<think>` reasoning block and a final answer."""
    return Rollout(
        prompt="What is 6 * 7?",
        response="<think>Step 1.</think>The answer is 42",
        gold="42",
    )


@pytest.fixture
def bare_rollout() -> Rollout:
    """A rollout with no reasoning structure and no whitespace in the response."""
    return Rollout(
        prompt="What is 6 * 7?",
        response="42",
        gold="42",
    )


@pytest.fixture
def eval_set() -> list[EvalRow]:
    """A 5-row eval set used by the runner tests.

    Responses are intentionally longer than 50 characters so that
    length-sensitive graders in `test_runner.py` exercise both the
    "stays above threshold" and "falls below threshold" branches when
    a `_ChopHalf`-style mutation halves the response.
    """
    return [
        EvalRow(
            id=f"row-{i}",
            rollout=Rollout(
                prompt=f"q{i}",
                response=(
                    f"good answer is {i}, "
                    "and here is some additional context to keep it long"
                ),
                gold=str(i),
            ),
        )
        for i in range(5)
    ]
