"""Public types for the reward-hack monitor.

The monitor consumes a ``Rollout`` (from ``caliper_sdk.types``) and a list of
``Rubric`` instances, and emits a composite ``MonitorVerdict``. Each rubric
returns its own ``RubricVerdict``; the composite ``flagged`` is the OR over
sub-rubrics so a single suspicious signal is never silently dropped.

LLM-backed rubrics depend on the ``LLMClient`` Protocol — never on a concrete
SDK — so tests can drive them with a deterministic mock and the real wiring is
deferred to a later milestone.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from caliper_sdk.types import Rollout
from pydantic import BaseModel, ConfigDict, Field


class RubricVerdict(BaseModel):
    """Output of a single rubric on a single rollout."""

    model_config = ConfigDict(extra="forbid")

    rubric_name: str
    flagged: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class MonitorVerdict(BaseModel):
    """Composite output of the ``Monitor`` over all rubrics."""

    model_config = ConfigDict(extra="forbid")

    flagged: bool
    verdicts: list[RubricVerdict] = Field(default_factory=list)
    flagged_rubrics: list[str] = Field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal synchronous LLM client surface used by judge-style rubrics.

    Concrete implementations (OpenAI, Anthropic, vLLM, ...) live elsewhere; the
    monitor only depends on this Protocol so tests stay hermetic.
    """

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str: ...


@runtime_checkable
class Rubric(Protocol):
    """A pluggable judge that turns a rollout into a ``RubricVerdict``."""

    name: str

    def check(
        self, rollout: Rollout, primary_score: float | None = None
    ) -> RubricVerdict: ...
