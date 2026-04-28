"""Shared fixtures for the store test suite."""

from __future__ import annotations

import pytest
from caliper_sdk import EvalRow, Rollout


@pytest.fixture
def two_rows() -> list[EvalRow]:
    return [
        EvalRow(id="r-001", rollout=Rollout(prompt="q1", response="a1", gold="a1")),
        EvalRow(id="r-002", rollout=Rollout(prompt="q2", response="a2", gold="b2")),
    ]


@pytest.fixture
def three_rows() -> list[EvalRow]:
    return [
        EvalRow(id="r-001", rollout=Rollout(prompt="q1", response="a1")),
        EvalRow(id="r-002", rollout=Rollout(prompt="q2", response="a2")),
        EvalRow(id="r-003", rollout=Rollout(prompt="q3", response="a3")),
    ]
