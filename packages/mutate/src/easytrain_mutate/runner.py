"""Mutation-test runner for graders.

The runner takes a grader, a list of `Mutation`s, and an eval set; runs the
grader on each baseline rollout and on each mutation's perturbed rollout;
and compares the score deltas against each mutation's `expected_effect`.
The output is a `MutationReport` flagging the (row x mutation) pairs whose
score delta violates the declared direction.

This is the productizable wedge described in
`docs/03-graders-and-rewards.md` ("SDET → grader engineering map"): the
grader is the artefact under test, the mutations are the adversarial
inputs, and `MutationReport.passed` is the CI gate.

LOWER / HIGHER violation semantics
----------------------------------

We use **strict** comparison with a **floor/ceiling carve-out** so that
rows with no possible signal don't pollute the report:

  * `LOWER`  — a `score_increased` violation fires when:
                 (a) `delta > tolerance` (the score actually went up), OR
                 (b) `delta >= -tolerance` AND `baseline > tolerance`
                     (the score didn't move, but it could have — this is
                     the canonical "naked-gold canary" failure mode where
                     the grader is invariant to the degradation).
               When `baseline <= tolerance` and `delta` is also near
               zero, the score had nowhere to go, so we record the
               outcome as compliant (no violation) rather than a false
               positive.

  * `HIGHER` — symmetric: violation when `delta < -tolerance` outright,
               OR when `delta <= tolerance` and `baseline < 1.0 -
               tolerance` (the canary). At-ceiling baselines with no
               movement are compliant.

  * `EQUAL`  — score must not move. `abs(delta) > tolerance` is a
               `score_changed` violation. No boundary carve-out — any
               movement is signal regardless of starting score.

A `tolerance` of 1e-6 swallows ordinary float noise (it sits well below
the precision any real grader would surface) without masking real
behavioural changes.
"""

from __future__ import annotations

from collections.abc import Callable

from easytrain_sdk import EvalRow, GraderResult, Rollout

from easytrain_mutate.report import (
    MutationFailure,
    MutationOutcome,
    MutationReport,
    MutationSummary,
)
from easytrain_mutate.types import Mutation, MutationExpectation


def _classify(
    expected: MutationExpectation,
    baseline_score: float,
    delta: float,
    tolerance: float,
) -> str | None:
    """Return the violation kind for a delta against an expectation.

    Returns `None` when the delta complies with the expected effect.
    See module docstring for the strict + boundary-carve-out semantics.
    """
    if expected is MutationExpectation.LOWER:
        # Real violation: score actively went up.
        if delta > tolerance:
            return "score_increased"
        # If the baseline was already at the floor, no-movement is no
        # signal — the score had nowhere to go.
        if baseline_score <= tolerance:
            return None
        # Canary: baseline could have been lowered but the grader stayed
        # invariant to the (real, non-no-op) mutation.
        if delta >= -tolerance:
            return "score_increased"
        return None

    if expected is MutationExpectation.HIGHER:
        if delta < -tolerance:
            return "score_decreased"
        if baseline_score >= 1.0 - tolerance:
            return None  # at ceiling; no movement possible
        if delta <= tolerance:
            return "score_decreased"
        return None

    # EQUAL — any movement is signal.
    if abs(delta) > tolerance:
        return "score_changed"
    return None


def run_mutations(
    grader: Callable[[Rollout], GraderResult],
    mutations: list[Mutation],
    eval_set: list[EvalRow],
    *,
    tolerance: float = 1e-6,
) -> MutationReport:
    """Run `grader` against `eval_set` x `mutations` and return a `MutationReport`.

    For each row we score the baseline rollout once, then for each
    mutation we apply the mutation and re-score. The (baseline, mutated)
    delta is compared to the mutation's declared `expected_effect`; any
    violation is recorded as a `MutationOutcome` with a non-None
    `violation`. Grader exceptions become `MutationFailure`s — a baseline
    failure short-circuits all mutations on that row (no comparison is
    possible without a baseline), but other rows continue.

    The returned report is deterministic for fixed inputs (mutations
    iterate in argument order, rows in eval-set order).
    """
    grader_name = getattr(grader, "name", "<unknown>")
    grader_version = getattr(grader, "version", "0.0.0")

    outcomes: list[MutationOutcome] = []
    failures: list[MutationFailure] = []

    for row in eval_set:
        # 1. Baseline. If this raises, we cannot compute a delta for any
        #    mutation on this row, so we synthesise a baseline failure
        #    for every mutation and move on.
        try:
            baseline = grader(row.rollout)
        except Exception as exc:
            for mutation in mutations:
                failures.append(
                    MutationFailure(
                        row_id=row.id,
                        mutation=mutation.name,
                        side="baseline",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
            continue

        baseline_score = baseline.score

        for mutation in mutations:
            mutated_rollout = mutation.apply(row.rollout)

            # 2. No-op detection: if the mutation didn't actually change
            #    the response text, there's nothing to compare. Record
            #    the baseline score on both sides with delta=0 and a
            #    skip flag so summaries can distinguish "no signal" from
            #    "compliant signal".
            if mutated_rollout.response == row.rollout.response:
                outcomes.append(
                    MutationOutcome(
                        row_id=row.id,
                        mutation=mutation.name,
                        baseline_score=baseline_score,
                        mutated_score=baseline_score,
                        delta=0.0,
                        skipped_no_op=True,
                        violation=None,
                        note="mutation was a no-op on this rollout",
                    )
                )
                continue

            # 3. Score the mutated rollout. A grader exception here gives
            #    a `side="mutated"` failure; we don't synthesise an
            #    outcome since the comparison is undefined.
            try:
                mutated = grader(mutated_rollout)
            except Exception as exc:
                failures.append(
                    MutationFailure(
                        row_id=row.id,
                        mutation=mutation.name,
                        side="mutated",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue

            delta = mutated.score - baseline_score
            violation = _classify(
                mutation.expected_effect, baseline_score, delta, tolerance
            )
            outcomes.append(
                MutationOutcome(
                    row_id=row.id,
                    mutation=mutation.name,
                    baseline_score=baseline_score,
                    mutated_score=mutated.score,
                    delta=delta,
                    skipped_no_op=False,
                    violation=violation,
                )
            )

    # Per-mutation summaries — preserve insertion order of `mutations`.
    summaries: list[MutationSummary] = []
    for mutation in mutations:
        m_outcomes = [o for o in outcomes if o.mutation == mutation.name]
        m_failures = [f for f in failures if f.mutation == mutation.name]
        n_skipped = sum(1 for o in m_outcomes if o.skipped_no_op)
        n_applied = len(m_outcomes) - n_skipped
        n_violations = sum(1 for o in m_outcomes if o.violation is not None)
        n_failures = len(m_failures)
        summaries.append(
            MutationSummary(
                name=mutation.name,
                expected_effect=mutation.expected_effect.value,
                n_applied=n_applied,
                n_skipped_no_op=n_skipped,
                n_violations=n_violations,
                n_failures=n_failures,
                passed=(n_violations == 0 and n_failures == 0),
            )
        )

    overall_passed = all(s.passed for s in summaries)

    return MutationReport(
        grader_name=grader_name,
        grader_version=grader_version,
        n_rows=len(eval_set),
        n_mutations=len(mutations),
        passed=overall_passed,
        per_mutation=summaries,
        outcomes=outcomes,
        failures=failures,
    )
