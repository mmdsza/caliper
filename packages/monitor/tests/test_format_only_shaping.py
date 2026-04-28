"""Tests for ``FormatOnlyShapingRubric``."""

from __future__ import annotations

from easytrain_monitor import FormatOnlyShapingRubric
from easytrain_sdk.types import Rollout


def test_empty_think_block_with_boxed_is_flagged() -> None:
    rubric = FormatOnlyShapingRubric()
    rollout = Rollout(
        prompt="What is 6*7?",
        response="<think></think>\\boxed{42}",
    )
    verdict = rubric.check(rollout)
    assert verdict.flagged is True
    assert verdict.rubric_name == "format_only_shaping"
    assert verdict.confidence > 0.5
    assert "format gate matched" in verdict.reason


def test_real_reasoning_inside_think_block_is_not_flagged() -> None:
    rubric = FormatOnlyShapingRubric()
    rollout = Rollout(
        prompt="What is 6*7?",
        response=(
            "<think>Step 1: I need to compute 6 multiplied by 7. "
            "Step 2: 6*7 equals 42 by direct computation. "
            "Step 3: confirm by 6+6+6+6+6+6+6 = 42.</think>"
            "\\boxed{42}"
        ),
    )
    verdict = rubric.check(rollout)
    assert verdict.flagged is False
    assert verdict.confidence == 0.0


def test_no_format_gate_returns_no_opinion() -> None:
    rubric = FormatOnlyShapingRubric()
    rollout = Rollout(
        prompt="What is 6*7?",
        response="The answer is 42 because 6 times 7 is 42.",
    )
    verdict = rubric.check(rollout)
    assert verdict.flagged is False
    assert "no format gate" in verdict.reason


def test_boxed_only_with_no_reasoning_is_flagged() -> None:
    rubric = FormatOnlyShapingRubric()
    rollout = Rollout(
        prompt="What is 6*7?",
        response="\\boxed{42}",
    )
    verdict = rubric.check(rollout)
    assert verdict.flagged is True
