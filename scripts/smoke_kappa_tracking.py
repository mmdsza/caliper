"""End-to-end smoke for Cohen's kappa tracking.

Runs the sample legal-citation grader against a hand-built reference
label set on the ``legal_v3`` fixture and prints kappa under all three
weighting schemes. The label set is designed so the grader and the
labeler *disagree on a few rows in interesting ways*:

* Both agree on the obvious passes (001, 002, 003, 009, 010) — bucket 2.
* Both agree on the missing-format-gate rows (007, 008) — bucket 0.
* Labeler gives partial credit (0.5 → bucket 1) on the format-gate-but-
  gold-mismatch rows (004, 005, 006) where the strict grader returned 0.0.

That puts 3 distance-1 disagreements on the confusion matrix in the
ternary binning, which is exactly the regime where quadratic kappa is
more forgiving than linear kappa (and both are more forgiving than
unweighted).

No network, no API keys. Run from repo root:
    uv run python scripts/smoke_kappa_tracking.py
"""

from __future__ import annotations

import sys

from easytrain_kappa import (
    KappaWeighting,
    LabeledRow,
    LabelSet,
    compute_kappa,
)
from easytrain_sdk import GraderResult, Rollout, grader
from easytrain_server._seed import LEGAL_V3_ROWS


@grader(name="legal_citation_grader", version="0.1.0")
def legal_citation_grader(rollout: Rollout) -> GraderResult:
    response = rollout.response
    if not response.startswith("<think>"):
        return GraderResult(score=0.0, explanation="missing <think> gate")
    if rollout.gold is not None and rollout.gold in response:
        return GraderResult(score=1.0, explanation="gold present in response")
    return GraderResult(score=0.0, explanation="gold absent")


def _build_labels() -> LabelSet:
    """Hand-crafted reference scores reflecting how a careful human reader
    might rate each ``legal_v3`` row. See module docstring for design."""
    return LabelSet(
        name="legal_v3_human_v1",
        rows=[
            # Format gate + exact match — humans agree these are great.
            LabeledRow(row_id="legal_v3-001", label_score=1.0),
            LabeledRow(row_id="legal_v3-002", label_score=1.0),
            LabeledRow(row_id="legal_v3-003", label_score=1.0),
            # Format gate but citation imperfect — humans give partial credit.
            LabeledRow(row_id="legal_v3-004", label_score=0.5),
            LabeledRow(row_id="legal_v3-005", label_score=0.5),
            LabeledRow(row_id="legal_v3-006", label_score=0.5),
            # No format gate — humans agree these fail.
            LabeledRow(row_id="legal_v3-007", label_score=0.0),
            LabeledRow(row_id="legal_v3-008", label_score=0.0),
            # Format gate + match — pass again.
            LabeledRow(row_id="legal_v3-009", label_score=1.0),
            LabeledRow(row_id="legal_v3-010", label_score=1.0),
        ],
    )


# Ternary bins so the partial-credit rows (0.5) live in their own bucket.
TERNARY_EDGES = [0.0, 0.34, 0.67, 1.0]


def main() -> int:
    rows = LEGAL_V3_ROWS
    labels = _build_labels()

    print(f"[smoke] grader=legal_citation_grader@0.1.0  eval_set=legal_v3 ({len(rows)} rows)")
    print(f"[smoke] labels={labels.name} ({len(labels.rows)} labeled)")
    print(f"[smoke] bins={TERNARY_EDGES}\n")

    reports = {
        weighting.value: compute_kappa(
            legal_citation_grader,
            rows,
            labels,
            bin_edges=TERNARY_EDGES,
            weighting=weighting,
        )
        for weighting in (
            KappaWeighting.UNWEIGHTED,
            KappaWeighting.LINEAR,
            KappaWeighting.QUADRATIC,
        )
    }

    sample = next(iter(reports.values()))
    print(f"[smoke] confusion (rows=grader, cols=label):")
    for i, conf_row in enumerate(sample.confusion):
        print(f"  bucket={i}: {conf_row}")
    print(
        f"[smoke] coverage: n_eval={sample.n_eval_rows}  "
        f"n_labeled={sample.n_labeled}  n_scored={sample.n_scored}  "
        f"failed={len(sample.failures)}"
    )
    print(
        f"[smoke] observed_agreement={sample.observed_agreement:.3f}  "
        f"expected_agreement={sample.expected_agreement:.3f}\n"
    )

    for name, report in reports.items():
        flag = " (UNDEFINED)" if report.kappa_undefined else ""
        print(f"[smoke] kappa[{name:<10}] = {report.kappa:+.4f}{flag}")

    # Sanity assertions for the SDET wedge: the metric had better not be
    # degenerate, and the documented ordering had better hold.
    unw = reports["unweighted"]
    lin = reports["linear"]
    qua = reports["quadratic"]

    if unw.n_scored != 10:
        print(f"[smoke] FAIL — expected 10 scored rows, got {unw.n_scored}")
        return 1
    for name, r in reports.items():
        if r.kappa_undefined:
            print(f"[smoke] FAIL — kappa[{name}] returned undefined; smoke fixture should produce a defined kappa")
            return 1
    # The 3 distance-1 disagreements => quadratic > linear > unweighted.
    if not (qua.kappa > lin.kappa > unw.kappa):
        print(
            f"[smoke] FAIL — expected quadratic > linear > unweighted; "
            f"got quadratic={qua.kappa} linear={lin.kappa} unweighted={unw.kappa}"
        )
        return 1

    print("\n[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
