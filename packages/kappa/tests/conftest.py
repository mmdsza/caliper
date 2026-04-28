"""Shared fixtures for the kappa test suite."""

from __future__ import annotations

import pytest
from caliper_sdk import EvalRow, GraderResult, Rollout, grader


@pytest.fixture
def four_row_eval_set() -> list[EvalRow]:
    """4 rows, ids r0..r3. Responses match gold for r0/r1, differ for r2/r3.

    Useful when paired with a grader that returns 1.0 for response==gold —
    you get exactly 2 rows in each bucket under the default pass/fail bins.
    """
    return [
        EvalRow(id="r0", rollout=Rollout(prompt="?", response="a", gold="a")),
        EvalRow(id="r1", rollout=Rollout(prompt="?", response="a", gold="a")),
        EvalRow(id="r2", rollout=Rollout(prompt="?", response="b", gold="a")),
        EvalRow(id="r3", rollout=Rollout(prompt="?", response="b", gold="a")),
    ]


@pytest.fixture
def six_row_eval_set() -> list[EvalRow]:
    """6 rows, ids r0..r5. Score-pinning fixture (see ``score_for_id``)."""
    return [
        EvalRow(id=f"r{i}", rollout=Rollout(prompt="?", response="x"))
        for i in range(6)
    ]


@pytest.fixture
def exact_match_grader():
    """Returns 1.0 when response == gold, else 0.0."""

    @grader
    def g(rollout: Rollout) -> GraderResult:
        score = 1.0 if rollout.gold is not None and rollout.response == rollout.gold else 0.0
        return GraderResult(score=score)

    return g


@pytest.fixture
def score_for_id():
    """Factory that builds a grader keyed by ``rollout.metadata['row_id']``.

    The kappa core invokes graders with a Rollout (no row id), so tests
    that need to assert per-row behavior steer the score by stuffing the
    id into metadata before invocation.
    """

    def _build(scores: dict[str, float]):
        @grader
        def g(rollout: Rollout) -> GraderResult:
            rid = rollout.metadata.get("row_id")
            if rid not in scores:
                raise KeyError(f"no scripted score for row id {rid!r}")
            return GraderResult(score=scores[rid])

        return g

    return _build


@pytest.fixture
def make_eval_row():
    """Factory: build an EvalRow whose Rollout carries its row id in metadata."""

    def _make(row_id: str) -> EvalRow:
        return EvalRow(
            id=row_id,
            rollout=Rollout(prompt="?", response="x", metadata={"row_id": row_id}),
        )

    return _make
