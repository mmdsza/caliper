"""Tests for the ``Monitor`` composer."""

from __future__ import annotations

from easytrain_monitor import Monitor, RubricVerdict
from easytrain_sdk.types import Rollout


class _AlwaysFlagRubric:
    name = "always_flag"

    def check(self, rollout: Rollout, primary_score: float | None = None) -> RubricVerdict:
        return RubricVerdict(
            rubric_name=self.name,
            flagged=True,
            confidence=0.8,
            reason="forced flag",
        )


class _NeverFlagRubric:
    name = "never_flag"

    def check(self, rollout: Rollout, primary_score: float | None = None) -> RubricVerdict:
        return RubricVerdict(
            rubric_name=self.name,
            flagged=False,
            confidence=0.0,
            reason="forced no-flag",
        )


class _BoomRubric:
    name = "boom"

    def check(self, rollout: Rollout, primary_score: float | None = None) -> RubricVerdict:
        raise RuntimeError("kaboom")


def _rollout() -> Rollout:
    return Rollout(prompt="q", response="a")


def test_broken_rubric_does_not_crash_monitor() -> None:
    monitor = Monitor(rubrics=[_NeverFlagRubric(), _BoomRubric(), _AlwaysFlagRubric()])
    verdict = monitor.check(_rollout())
    # All three rubrics get a verdict back.
    assert len(verdict.verdicts) == 3
    by_name = {v.rubric_name: v for v in verdict.verdicts}
    boom = by_name["boom"]
    assert boom.flagged is False
    assert boom.confidence == 0.0
    assert boom.reason.startswith("rubric raised:")
    assert "kaboom" in boom.reason


def test_any_flag_makes_composite_flagged() -> None:
    monitor = Monitor(rubrics=[_NeverFlagRubric(), _AlwaysFlagRubric()])
    verdict = monitor.check(_rollout())
    assert verdict.flagged is True
    assert verdict.flagged_rubrics == ["always_flag"]


def test_all_clean_means_composite_not_flagged() -> None:
    monitor = Monitor(rubrics=[_NeverFlagRubric(), _NeverFlagRubric()])
    verdict = monitor.check(_rollout())
    assert verdict.flagged is False
    assert verdict.flagged_rubrics == []
    assert len(verdict.verdicts) == 2


def test_monitor_with_no_rubrics_is_not_flagged() -> None:
    monitor = Monitor(rubrics=[])
    verdict = monitor.check(_rollout())
    assert verdict.flagged is False
    assert verdict.verdicts == []
    assert verdict.flagged_rubrics == []


def test_monitor_passes_primary_score_through() -> None:
    seen: list[float | None] = []

    class _SpyRubric:
        name = "spy"

        def check(
            self, rollout: Rollout, primary_score: float | None = None
        ) -> RubricVerdict:
            seen.append(primary_score)
            return RubricVerdict(
                rubric_name=self.name,
                flagged=False,
                confidence=0.0,
                reason="spy",
            )

    monitor = Monitor(rubrics=[_SpyRubric()])
    monitor.check(_rollout(), primary_score=0.42)
    assert seen == [0.42]
