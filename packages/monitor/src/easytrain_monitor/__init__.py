"""EasyTrain reward-hack monitor.

Public surface:
    Monitor                 — composes N rubrics into a single verdict
    MonitorVerdict          — composite output (flagged + per-rubric verdicts)
    RubricVerdict           — single-rubric output
    LLMClient               — Protocol for pluggable LLM clients
    Rubric                  — Protocol every rubric satisfies
    FormatOnlyShapingRubric — flags format-gated empty reasoning
    TrivialToolCallsRubric  — flags traces dominated by dummy tool calls
    TestMonkeyPatchRubric   — flags responses that mutate tests / hardcode answers
    LLMJudgeRubric          — generic LLM-as-judge wrapper
"""

from easytrain_monitor.monitor import Monitor
from easytrain_monitor.rubrics import (
    FormatOnlyShapingRubric,
    LLMJudgeRubric,
    TestMonkeyPatchRubric,
    TrivialToolCallsRubric,
)
from easytrain_monitor.types import LLMClient, MonitorVerdict, Rubric, RubricVerdict

__all__ = [
    "FormatOnlyShapingRubric",
    "LLMClient",
    "LLMJudgeRubric",
    "Monitor",
    "MonitorVerdict",
    "Rubric",
    "RubricVerdict",
    "TestMonkeyPatchRubric",
    "TrivialToolCallsRubric",
]
