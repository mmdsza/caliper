"""Built-in reward-hack rubrics.

Each rubric targets a documented hack pattern from
``docs/03-graders-and-rewards.md`` ("Reward hacking — the unsolved attack
surface"). Pure-programmatic rubrics live alongside an LLM-as-judge generic
that takes any pluggable ``LLMClient``.
"""

from caliper_monitor.rubrics.format_only_shaping import FormatOnlyShapingRubric
from caliper_monitor.rubrics.llm_judge import LLMJudgeRubric
from caliper_monitor.rubrics.test_monkey_patch import TestMonkeyPatchRubric
from caliper_monitor.rubrics.trivial_tool_calls import TrivialToolCallsRubric

__all__ = [
    "FormatOnlyShapingRubric",
    "LLMJudgeRubric",
    "TestMonkeyPatchRubric",
    "TrivialToolCallsRubric",
]
