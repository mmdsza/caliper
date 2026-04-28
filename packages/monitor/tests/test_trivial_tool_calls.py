"""Tests for ``TrivialToolCallsRubric``."""

from __future__ import annotations

from easytrain_monitor import TrivialToolCallsRubric
from easytrain_sdk.types import Rollout


def _trivial_trace() -> list[dict]:
    # Four tool calls, all with empty arguments — exactly the dummy-call
    # pathology described in the reward-hack table.
    trace: list[dict] = []
    for _ in range(4):
        trace.append({"role": "assistant", "type": "tool_call", "arguments": ""})
        trace.append({"role": "tool", "content": "ok"})
    return trace


def _substantive_trace() -> list[dict]:
    # Four real tool calls with non-empty arguments AND results that are
    # subsequently quoted in the assistant's reply.
    trace: list[dict] = []
    payloads = [
        ("search", {"query": "what is the population of Iceland"}, "Iceland population is 393000"),
        ("calc", {"expression": "393000 / 1000"}, "result: 393.0"),
        ("lookup", {"id": "iceland-2024"}, "data: latest census 2024 confirms"),
        ("translate", {"text": "thanks"}, "translation: takk"),
    ]
    for tool, args, result in payloads:
        trace.append(
            {
                "role": "assistant",
                "type": "tool_call",
                "tool_call": {"name": tool, "arguments": args},
            }
        )
        trace.append({"role": "tool", "content": result})
        # The agent quotes the leading 20 chars of the result back, simulating
        # genuine downstream use.
        trace.append({"role": "assistant", "content": f"Got: {result[:20]} — using it."})
    return trace


def test_all_trivial_tool_calls_are_flagged() -> None:
    rubric = TrivialToolCallsRubric()
    rollout = Rollout(prompt="q", response="r", trace=_trivial_trace())
    verdict = rubric.check(rollout)
    assert verdict.flagged is True
    assert verdict.rubric_name == "trivial_tool_calls"
    assert "4/4" in verdict.reason


def test_substantive_tool_calls_are_not_flagged() -> None:
    rubric = TrivialToolCallsRubric()
    rollout = Rollout(prompt="q", response="r", trace=_substantive_trace())
    verdict = rubric.check(rollout)
    assert verdict.flagged is False
    assert verdict.confidence == 0.0


def test_empty_trace_returns_no_signal() -> None:
    rubric = TrivialToolCallsRubric()
    rollout = Rollout(prompt="q", response="r", trace=[])
    verdict = rubric.check(rollout)
    assert verdict.flagged is False
    assert verdict.confidence == 0.0
    assert "empty trace" in verdict.reason


def test_trace_with_no_tool_calls_returns_no_signal() -> None:
    rubric = TrivialToolCallsRubric()
    trace = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    ]
    rollout = Rollout(prompt="q", response="r", trace=trace)
    verdict = rubric.check(rollout)
    assert verdict.flagged is False
    assert "no tool calls" in verdict.reason
