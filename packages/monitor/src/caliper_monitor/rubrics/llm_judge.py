"""Generic LLM-as-judge rubric.

Wraps any ``LLMClient`` Protocol into a rubric. The judge prompt is a
``str.format()`` template — we deliberately avoid a Jinja2 dependency for the
common case. The judge is expected to emit two parsable lines anywhere in its
response::

    FLAGGED: yes
    CONFIDENCE: 0.85

Parsing is case-insensitive and tolerant of surrounding whitespace. If neither
marker is found, the rubric returns a non-flagged verdict at confidence 0.0
with a reason that carries the first 100 chars of the offending response, so
callers can debug judge prompt drift.
"""

from __future__ import annotations

import re

from caliper_sdk.types import Rollout

from caliper_monitor.types import LLMClient, RubricVerdict

# Forgiving line-anchored matchers. The judge may interleave prose; we only
# require one ``FLAGGED:`` and one ``CONFIDENCE:`` line.
_FLAGGED_RE = re.compile(
    r"^\s*FLAGGED\s*[:=]\s*(yes|no|true|false|y|n|1|0)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CONFIDENCE_RE = re.compile(
    r"^\s*CONFIDENCE\s*[:=]\s*([0-9]*\.?[0-9]+)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TRUTHY = {"yes", "true", "y", "1"}


class LLMJudgeRubric:
    """A pluggable LLM-as-judge rubric.

    Parameters
    ----------
    client:
        Anything satisfying the ``LLMClient`` Protocol — tests pass a mock,
        production passes a real provider client.
    model:
        Forwarded verbatim to ``client.complete``.
    prompt_template:
        ``str.format()`` template; receives ``prompt``, ``response``,
        ``gold``, and ``primary_score`` keyword arguments. Missing keys are
        rendered as empty strings rather than raising — judges shouldn't break
        because a rollout lacks a gold reference.
    name:
        Rubric identifier surfaced in ``RubricVerdict.rubric_name``.
    max_tokens:
        Forwarded to the client. Default 512.
    """

    def __init__(
        self,
        client: LLMClient,
        model: str,
        prompt_template: str,
        name: str,
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.name = name
        self.max_tokens = max_tokens

    def _render(self, rollout: Rollout, primary_score: float | None) -> str:
        # Use a defaultdict-style mapping so missing template keys render as
        # empty strings rather than raising KeyError.
        class _Defaulting(dict):  # type: ignore[type-arg]
            def __missing__(self, key: str) -> str:
                return ""

        ctx = _Defaulting(
            prompt=rollout.prompt,
            response=rollout.response,
            gold=rollout.gold or "",
            primary_score="" if primary_score is None else f"{primary_score:.4f}",
        )
        return self.prompt_template.format_map(ctx)

    def check(
        self, rollout: Rollout, primary_score: float | None = None
    ) -> RubricVerdict:
        prompt = self._render(rollout, primary_score)
        completion = self.client.complete(
            prompt, model=self.model, max_tokens=self.max_tokens
        )

        flag_match = _FLAGGED_RE.search(completion)
        conf_match = _CONFIDENCE_RE.search(completion)

        if flag_match is None or conf_match is None:
            snippet = completion.strip().replace("\n", " ")[:100]
            return RubricVerdict(
                rubric_name=self.name,
                flagged=False,
                confidence=0.0,
                reason=f"judge response unparseable: {snippet}",
            )

        flagged = flag_match.group(1).lower() in _TRUTHY
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            snippet = completion.strip().replace("\n", " ")[:100]
            return RubricVerdict(
                rubric_name=self.name,
                flagged=False,
                confidence=0.0,
                reason=f"judge response unparseable: {snippet}",
            )

        # Clamp to [0, 1] rather than rejecting; judges occasionally emit 1.5
        # and we'd rather record the verdict than throw.
        confidence = max(0.0, min(1.0, confidence))

        return RubricVerdict(
            rubric_name=self.name,
            flagged=flagged,
            confidence=confidence,
            reason=f"llm judge ({self.model}) returned FLAGGED={flagged}",
        )
