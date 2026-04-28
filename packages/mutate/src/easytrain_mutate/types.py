"""Shared types for grader mutation testing.

A `Mutation` is an adversarial transformation on a `Rollout`. Each carries an
`expected_effect` describing how a *robust* grader's score should respond to
the perturbation:

    LOWER   — the mutation degrades the response; score should not increase
    EQUAL   — the mutation is semantics-preserving; score should not change
    HIGHER  — the mutation strictly improves the response (rare)

The runner runs the grader on (baseline, mutated) pairs and flags any score
delta that violates the expected direction. Each violation is a candidate
reward-hack — the grader is sensitive to a property the mutation says it
should be invariant to (or vice versa).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from easytrain_sdk import Rollout


class MutationExpectation(StrEnum):
    """Direction the score should move under a mutation, for a robust grader."""

    LOWER = "lower"
    EQUAL = "equal"
    HIGHER = "higher"


@runtime_checkable
class Mutation(Protocol):
    """Structural type any built-in or user-supplied mutation must satisfy."""

    name: str
    description: str
    expected_effect: MutationExpectation

    def apply(self, rollout: Rollout) -> Rollout: ...
