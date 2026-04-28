"""Shared fixtures for the SDK test suite."""

from __future__ import annotations

import pytest
from caliper_sdk import EvalRow, Rollout


@pytest.fixture
def eval_set_10() -> list[EvalRow]:
    """A deterministic 10-row eval set.

    Row ids are ``"row-0"`` ... ``"row-9"``. Prompts and responses follow a
    simple template so graders can be written that yield reproducible scores.
    """
    return [
        EvalRow(
            id=f"row-{i}",
            rollout=Rollout(
                prompt=f"q{i}",
                response=f"a{i}",
                gold=f"a{i}" if i % 2 == 0 else f"a{i + 1}",
            ),
        )
        for i in range(10)
    ]
