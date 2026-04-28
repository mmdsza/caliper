"""Tests for `easytrain_mutate.runner.run_mutations` and the formatter.

We deliberately do NOT import from `easytrain_mutate.mutations` — that
module is owned by another agent. All stand-in mutations live in this
file so the runner tests are self-contained and order-independent
relative to the mutations test suite.
"""

from __future__ import annotations

from easytrain_mutate.format import format_mutation_report
from easytrain_mutate.report import MutationReport
from easytrain_mutate.runner import run_mutations
from easytrain_mutate.types import MutationExpectation
from easytrain_sdk import EvalRow, GraderResult, Rollout

# ---------------------------------------------------------------------------
# Stand-in mutations (do not depend on the real `mutations.py`).
# ---------------------------------------------------------------------------


class _NoOp:
    """Returns the rollout unchanged. Forces the no-op skip path."""

    name = "noop"
    description = "returns input unchanged"
    expected_effect = MutationExpectation.EQUAL

    def apply(self, rollout: Rollout) -> Rollout:
        return rollout.model_copy()


class _ChopHalf:
    """Halves the response. Expected to *lower* score on length-sensitive graders."""

    name = "chop_half"
    description = "drop last half of response"
    expected_effect = MutationExpectation.LOWER

    def apply(self, rollout: Rollout) -> Rollout:
        return rollout.model_copy(
            update={"response": rollout.response[: len(rollout.response) // 2]}
        )


class _GarbageEqual:
    """Appends a garbage suffix; asserted EQUAL (grader should be invariant)."""

    name = "garbage_equal"
    description = "append garbage; assert grader is invariant"
    expected_effect = MutationExpectation.EQUAL

    def apply(self, rollout: Rollout) -> Rollout:
        return rollout.model_copy(update={"response": rollout.response + " ###"})


class _NakedGold:
    """Replaces response with the gold answer alone — strips any reasoning trace.

    Mirrors the canary scenario from `docs/03-graders-and-rewards.md`:
    a robust grader (which requires `<think>` plus the gold) should
    drop hard, while a naive `gold in response` grader will not move.
    """

    name = "naked_gold"
    description = "response := gold; strips any reasoning"
    expected_effect = MutationExpectation.LOWER

    def apply(self, rollout: Rollout) -> Rollout:
        gold = rollout.gold or ""
        return rollout.model_copy(update={"response": gold})


# ---------------------------------------------------------------------------
# Grader stubs.
# ---------------------------------------------------------------------------


def _result(score: float) -> GraderResult:
    return GraderResult(score=score)


def _grader_with_attrs(
    fn, *, name: str = "stub_grader", version: str = "1.2.3"
):
    """Attach name/version attrs so the runner can pick them up."""
    fn.name = name
    fn.version = version
    return fn


def _good_grader(rollout: Rollout) -> GraderResult:
    """1.0 if "good" appears in response, else 0.0."""
    return _result(1.0 if "good" in rollout.response else 0.0)


def _length_grader(rollout: Rollout) -> GraderResult:
    """1.0 if len(response) > 50 else 0.0 — used to test LOWER compliance."""
    return _result(1.0 if len(rollout.response) > 50 else 0.0)


def _short_is_better_grader(rollout: Rollout) -> GraderResult:
    """Perverse: shorter responses score higher. _ChopHalf should *raise* score."""
    return _result(1.0 if len(rollout.response) <= 50 else 0.0)


def _hash_grader(rollout: Rollout) -> GraderResult:
    """Deterministic but extremely sensitive: hash the response into [0, 1].

    Any change to the response moves the score, which lets us exercise
    EQUAL violations under `_GarbageEqual`.
    """
    h = abs(hash(rollout.response)) % 1000
    return _result(h / 1000.0)


def _robust_canary_grader(rollout: Rollout) -> GraderResult:
    """Awards 1.0 only if response has a `<think>` prefix AND the gold."""
    has_think = "<think>" in rollout.response
    has_gold = rollout.gold is not None and rollout.gold in rollout.response
    return _result(1.0 if (has_think and has_gold) else 0.0)


def _naive_canary_grader(rollout: Rollout) -> GraderResult:
    """Naive: just checks `gold in response`. Vulnerable to naked-gold hack."""
    has_gold = rollout.gold is not None and rollout.gold in rollout.response
    return _result(1.0 if has_gold else 0.0)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_robust_grader_passes_all(eval_set: list[EvalRow]) -> None:
    """A grader insensitive to garbage suffix and no-op should pass cleanly."""
    grader = _grader_with_attrs(_good_grader, name="good_grader", version="0.1.0")
    report = run_mutations(grader, [_NoOp(), _GarbageEqual()], eval_set)

    assert isinstance(report, MutationReport)
    assert report.passed is True
    assert report.grader_name == "good_grader"
    assert report.grader_version == "0.1.0"
    assert report.n_rows == len(eval_set)
    assert report.n_mutations == 2
    assert all(s.n_violations == 0 for s in report.per_mutation)
    assert all(s.n_failures == 0 for s in report.per_mutation)


def test_lower_compliant_then_violation(eval_set: list[EvalRow]) -> None:
    """LOWER is honoured by length grader; perverse short-is-better grader violates."""
    # First: compliant case. Length grader scores 1.0 baseline (responses
    # are >50 chars) and 0.0 mutated (chopped well below 50). delta is
    # strictly negative, which under our strict-LOWER semantics is the
    # only thing that satisfies `expected_effect=LOWER`.
    compliant = run_mutations(
        _grader_with_attrs(_length_grader, name="len_grader"),
        [_ChopHalf()],
        eval_set,
    )
    assert compliant.passed is True
    chop_summary = compliant.per_mutation[0]
    assert chop_summary.n_violations == 0
    # Spot-check that all deltas are strictly negative.
    chop_outcomes = [o for o in compliant.outcomes if o.mutation == "chop_half"]
    assert chop_outcomes  # not all skipped
    assert all(o.delta < 0 for o in chop_outcomes if not o.skipped_no_op)

    # Now: perverse grader rewards shorter responses. _ChopHalf should
    # therefore *raise* the score → `score_increased` violation under
    # expected LOWER.
    perverse = run_mutations(
        _grader_with_attrs(_short_is_better_grader, name="short_grader"),
        [_ChopHalf()],
        eval_set,
    )
    assert perverse.passed is False
    perverse_summary = perverse.per_mutation[0]
    assert perverse_summary.n_violations >= 1
    violations = [o for o in perverse.outcomes if o.violation is not None]
    assert violations
    assert all(o.violation == "score_increased" for o in violations)


def test_equal_violation_caught(eval_set: list[EvalRow]) -> None:
    """A hash-based grader violates EQUAL when garbage is appended."""
    report = run_mutations(
        _grader_with_attrs(_hash_grader, name="hash_grader"),
        [_GarbageEqual()],
        eval_set,
    )
    assert report.passed is False
    summary = report.per_mutation[0]
    # Hashing is essentially never invariant to the change, so we expect
    # a violation on most or all rows.
    assert summary.n_violations >= 1
    violations = [o for o in report.outcomes if o.violation is not None]
    assert all(o.violation == "score_changed" for o in violations)


def test_no_op_is_skipped(eval_set: list[EvalRow]) -> None:
    """A no-op mutation should produce only `skipped_no_op=True` outcomes."""
    report = run_mutations(
        _grader_with_attrs(_good_grader),
        [_NoOp()],
        eval_set,
    )
    summary = report.per_mutation[0]
    assert summary.n_skipped_no_op == len(eval_set)
    assert summary.n_applied == 0
    assert summary.n_violations == 0
    assert summary.n_failures == 0
    # No violation count should be set even though expected_effect=EQUAL
    # — the skip short-circuits classification.
    assert all(o.skipped_no_op for o in report.outcomes)
    assert all(o.violation is None for o in report.outcomes)
    assert all(
        o.note == "mutation was a no-op on this rollout" for o in report.outcomes
    )
    # And report.passed should still be True — a no-op gives no signal,
    # but it isn't a violation.
    assert report.passed is True


def test_baseline_raise(eval_set: list[EvalRow]) -> None:
    """Baseline failure on one row records `side="baseline"` and skips its mutations."""

    def grader(rollout: Rollout) -> GraderResult:
        if rollout.prompt == "q3":  # corresponds to row-3
            raise RuntimeError("baseline boom")
        return _result(1.0)

    grader.name = "boom_baseline"
    grader.version = "0.0.1"

    report = run_mutations(grader, [_GarbageEqual()], eval_set)
    summary = report.per_mutation[0]
    assert summary.n_failures == 1
    failure = report.failures[0]
    assert failure.row_id == "row-3"
    assert failure.side == "baseline"
    assert failure.error_type == "RuntimeError"
    assert "baseline boom" in failure.error_message
    # Other rows still got scored — 4 successful outcomes.
    non_skipped = [o for o in report.outcomes if not o.skipped_no_op]
    assert len(non_skipped) == 4
    # No mutated outcome on the broken row.
    assert all(o.row_id != "row-3" for o in report.outcomes)
    # And the run failed overall.
    assert report.passed is False


def test_mutated_raise(eval_set: list[EvalRow]) -> None:
    """A grader that explodes on the mutated input gives `side="mutated"` per row."""

    def grader(rollout: Rollout) -> GraderResult:
        if "###" in rollout.response:
            raise ValueError("garbage tripped me up")
        return _result(1.0)

    grader.name = "boom_mutated"
    grader.version = "0.0.1"

    report = run_mutations(grader, [_GarbageEqual()], eval_set)
    summary = report.per_mutation[0]
    assert summary.n_failures == len(eval_set)
    assert all(f.side == "mutated" for f in report.failures)
    # Every baseline succeeded → no `side="baseline"` failures.
    assert not any(f.side == "baseline" for f in report.failures)
    # No outcomes recorded since every mutated call raised.
    assert report.outcomes == []
    assert report.passed is False


def test_report_pass_flag(eval_set: list[EvalRow]) -> None:
    """`report.passed` is True iff every per-mutation summary passed."""
    clean = run_mutations(
        _grader_with_attrs(_good_grader),
        [_GarbageEqual()],
        eval_set,
    )
    assert clean.passed is True

    dirty = run_mutations(
        _grader_with_attrs(_hash_grader),
        [_GarbageEqual()],
        eval_set,
    )
    assert dirty.passed is False


def test_format_mutation_report(eval_set: list[EvalRow]) -> None:
    """Formatter contains key fields, status, and respects `max_outcomes`."""
    pass_report = run_mutations(
        _grader_with_attrs(_good_grader, name="good_grader", version="9.9.9"),
        [_GarbageEqual()],
        eval_set,
    )
    text = format_mutation_report(pass_report)
    assert isinstance(text, str)
    assert text  # non-empty
    assert "PASS" in text
    assert "good_grader" in text
    assert "garbage_equal" in text

    fail_report = run_mutations(
        _grader_with_attrs(_hash_grader, name="hash_grader"),
        [_GarbageEqual()],
        eval_set,
    )
    full_text = format_mutation_report(fail_report)
    assert "FAIL" in full_text
    assert "hash_grader" in full_text
    assert "garbage_equal" in full_text
    # Should mention at least one violating row id.
    violation_rows = {o.row_id for o in fail_report.outcomes if o.violation}
    assert any(rid in full_text for rid in violation_rows)

    # `max_outcomes=2` caps the top-violation block at 2 rows.
    capped = format_mutation_report(fail_report, max_outcomes=2)
    top_block = capped.split("Top violations")[1] if "Top violations" in capped else ""
    # Count the violation rows we surfaced (lines starting with "  row=").
    top_lines = [
        ln for ln in top_block.splitlines() if ln.strip().startswith("row=")
    ]
    assert len(top_lines) <= 2


def test_lower_floor_case_is_not_flagged() -> None:
    """Baseline already at 0 + LOWER mutation that doesn't move score → no signal.

    This is the carve-out documented in the runner module docstring: when
    the score has nowhere to go, "no movement" is not a violation. The
    canary case (baseline > 0 with no movement) is exercised separately
    in `test_canary_hardcode_gold_pattern`.
    """
    rows = [
        EvalRow(
            id=f"floor-{i}",
            rollout=Rollout(prompt=f"q{i}", response="bad", gold=str(i)),
        )
        for i in range(3)
    ]

    def always_zero(rollout: Rollout) -> GraderResult:
        return _result(0.0)

    always_zero.name = "always_zero"
    always_zero.version = "0.1.0"

    report = run_mutations(always_zero, [_ChopHalf()], rows)

    summary = report.per_mutation[0]
    assert summary.n_applied >= 1
    assert summary.n_violations == 0, "floor + no-movement should not be a violation"
    assert report.passed is True


def test_lower_score_increase_from_floor_still_flagged(eval_set: list[EvalRow]) -> None:
    """An *active* score increase from the floor is still a violation.

    The floor carve-out only suppresses the "delta == 0 from baseline 0"
    case. If the mutation actively raises the score, that's the perverse-
    grader case (`_short_is_better_grader`) and must still be flagged.
    """
    perverse = run_mutations(
        _grader_with_attrs(_short_is_better_grader, name="short_grader"),
        [_ChopHalf()],
        eval_set,
    )
    assert perverse.passed is False
    violations = [o for o in perverse.outcomes if o.violation is not None]
    assert violations
    assert all(o.violation == "score_increased" for o in violations)


def test_canary_hardcode_gold_pattern(eval_set: list[EvalRow]) -> None:
    """Reward-hack canary: naked-gold mutation flags non-robust graders.

    The robust grader requires `<think>` AND the gold; on this eval set
    no row has `<think>` in its response, so baseline is 0.0 across the
    board. After `_NakedGold` strips down to just the gold, the score
    stays 0.0 — delta == 0 with expected LOWER, which under our STRICT
    semantics is itself a violation. To get a clean compliance signal
    on the robust grader, we therefore use a custom eval set where the
    baseline already includes `<think>` (so baseline=1.0, mutated=0.0,
    delta strictly negative → compliant).

    The naive grader (`gold in response`) keeps scoring 1.0 on the
    naked-gold mutated response, giving delta=0 with expected LOWER →
    violation `score_increased`. That's the canary firing.
    """
    canary_eval = [
        EvalRow(
            id=f"canary-{i}",
            rollout=Rollout(
                prompt=f"q{i}",
                response=f"<think>working it out...</think>The answer is {i}",
                gold=str(i),
            ),
        )
        for i in range(3)
    ]

    robust = run_mutations(
        _grader_with_attrs(_robust_canary_grader, name="robust_canary"),
        [_NakedGold()],
        canary_eval,
    )
    # Baselines all 1.0 (think+gold), mutated all 0.0 (think stripped).
    # delta = -1.0 < -tolerance → compliant under strict LOWER.
    assert robust.passed is True
    robust_outcomes = [o for o in robust.outcomes if o.mutation == "naked_gold"]
    assert robust_outcomes
    assert all(o.delta < 0 for o in robust_outcomes)
    assert all(o.violation is None for o in robust_outcomes)

    naive = run_mutations(
        _grader_with_attrs(_naive_canary_grader, name="naive_canary"),
        [_NakedGold()],
        canary_eval,
    )
    # Baseline 1.0 (gold present), mutated 1.0 (gold still trivially
    # present — that's the bug). delta == 0 fails strict LOWER.
    assert naive.passed is False
    naive_outcomes = [o for o in naive.outcomes if o.mutation == "naked_gold"]
    assert naive_outcomes
    assert all(o.delta == 0.0 for o in naive_outcomes)
    assert all(o.violation == "score_increased" for o in naive_outcomes)
