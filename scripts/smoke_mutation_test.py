"""End-to-end smoke for the mutation-test runner.

Runs the 8 built-in mutations against two graders on the legal_v3 fixture:

  1. The robust sample grader (format gate AND gold present)
  2. A naked-gold-accepting grader (just `gold in response`)

Asserts:
  - The robust grader survives EQUAL mutations (whitespace, sycophancy, garbage)
  - The robust grader's HardcodeGold mutation produces the expected canary
    behavior — score drops because the format gate is missing
  - The naked-gold-accepting grader is caught by HardcodeGold (mutated score
    stays high, expected LOWER, violation flagged)

Run from repo root: `uv run python scripts/smoke_mutation_test.py`.
"""

from __future__ import annotations

import sys

from easytrain_mutate import (
    AppendGarbage,
    CasePerturb,
    DropFormatGate,
    HardcodeGold,
    PrependSycophancy,
    TruncateResponse,
    WhitespacePerturb,
    WrongAnswer,
    format_mutation_report,
    run_mutations,
)
from easytrain_sdk import GraderResult, Rollout, grader
from easytrain_server._seed import LEGAL_V3_ROWS


@grader(name="robust_legal", version="0.1.0")
def robust_grader(rollout: Rollout) -> GraderResult:
    """Format gate + gold-substring match. Mirrors the editor's sample grader."""
    if not rollout.response.startswith("<think>"):
        return GraderResult(score=0.0, explanation="missing <think> gate")
    if rollout.gold is not None and rollout.gold in rollout.response:
        return GraderResult(score=1.0, explanation="gold present")
    return GraderResult(score=0.0, explanation="gold absent")


@grader(name="naked_gold", version="0.1.0")
def naked_gold_grader(rollout: Rollout) -> GraderResult:
    """Reward-hackable: only checks gold-substring; ignores format and structure."""
    if rollout.gold is not None and rollout.gold in rollout.response:
        return GraderResult(score=1.0, explanation="gold present")
    return GraderResult(score=0.0, explanation="gold absent")


def main() -> int:
    mutations = [
        DropFormatGate(),
        WrongAnswer(),
        TruncateResponse(),
        WhitespacePerturb(),
        PrependSycophancy(),
        AppendGarbage(),
        CasePerturb(),
        HardcodeGold(),
    ]

    print("=== Robust grader ===")
    robust_report = run_mutations(robust_grader, mutations, LEGAL_V3_ROWS)
    print(format_mutation_report(robust_report, max_outcomes=15))

    print("\n=== Naked-gold (reward-hackable) grader ===")
    naked_report = run_mutations(naked_gold_grader, mutations, LEGAL_V3_ROWS)
    print(format_mutation_report(naked_report, max_outcomes=15))

    # The naked-gold grader must be caught — at minimum on HardcodeGold,
    # which is the canary mutation.
    naked_summaries = {s.name: s for s in naked_report.per_mutation}
    canary = naked_summaries["hardcode_gold"]
    if canary.passed:
        print("\n[smoke] FAIL — naked-gold grader passed HardcodeGold (canary missed!)")
        return 1
    if canary.n_violations == 0:
        print("\n[smoke] FAIL — HardcodeGold reported zero violations on the hackable grader")
        return 1

    print(
        f"\n[smoke] canary working: HardcodeGold flagged {canary.n_violations} "
        f"violation(s) on the naked-gold grader"
    )
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
