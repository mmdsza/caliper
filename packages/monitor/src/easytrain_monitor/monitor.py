"""Composite reward-hack monitor.

Runs every supplied ``Rubric`` over a single rollout and produces a composite
``MonitorVerdict``. A rubric that raises is captured into a non-flagging
verdict carrying the exception message — one broken rubric must never crash
the monitor (the monitor sits in CI; one bad regex shouldn't redline a build).
"""

from __future__ import annotations

from easytrain_sdk.types import Rollout

from easytrain_monitor.types import MonitorVerdict, Rubric, RubricVerdict


class Monitor:
    """Compose N rubrics into one verdict."""

    def __init__(self, rubrics: list[Rubric]) -> None:
        self.rubrics = list(rubrics)

    def check(
        self, rollout: Rollout, primary_score: float | None = None
    ) -> MonitorVerdict:
        verdicts: list[RubricVerdict] = []
        for rubric in self.rubrics:
            try:
                verdict = rubric.check(rollout, primary_score)
            except Exception as exc:
                verdicts.append(
                    RubricVerdict(
                        rubric_name=getattr(rubric, "name", repr(rubric)),
                        flagged=False,
                        confidence=0.0,
                        reason=f"rubric raised: {exc}",
                    )
                )
                continue
            verdicts.append(verdict)

        flagged_rubrics = [v.rubric_name for v in verdicts if v.flagged]
        return MonitorVerdict(
            flagged=bool(flagged_rubrics),
            verdicts=verdicts,
            flagged_rubrics=flagged_rubrics,
        )
