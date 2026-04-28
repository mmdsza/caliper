"""Shared fixtures for differ tests."""

from __future__ import annotations

import pytest
from easytrain_sdk.types import EvalRow, Rollout


@pytest.fixture
def eval_set_10() -> list[EvalRow]:
    """10-row deterministic eval set: prompts q0..q9, responses a0..a9."""
    return [
        EvalRow(id=str(i), rollout=Rollout(prompt=f"q{i}", response=f"a{i}"))
        for i in range(10)
    ]
