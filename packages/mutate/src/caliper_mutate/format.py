"""Plain-text formatter for `MutationReport`.

Used by the CLI / dry-run debug path. No colour, no rich — just stable
text so snapshots diff cleanly and any terminal renders identically.
"""

from __future__ import annotations

from caliper_mutate.report import MutationReport

_TICK = "✓"  # ✓
_CROSS = "✗"  # ✗


def format_mutation_report(
    report: MutationReport, *, max_outcomes: int = 10
) -> str:
    """Render a `MutationReport` as a multi-line human-readable string.

    Layout:
      - header (grader name/version, row count)
      - PASS / FAIL line with a one-glance violation tally
      - per-mutation block (insertion order, one line each)
      - top-`max_outcomes` violations sorted by `abs(delta)` descending

    The per-mutation block uses Unicode tick / cross so the eye lands on
    the failed row immediately when scrolling a long CI log.
    """
    lines: list[str] = []
    lines.append("Caliper mutation test")
    lines.append("=======================")
    lines.append(f"Grader: {report.grader_name} v{report.grader_version}")
    lines.append(f"Rows:   {report.n_rows}")

    if report.passed:
        lines.append("Status: PASS")
    else:
        n_violating_mutations = sum(
            1 for s in report.per_mutation if s.n_violations > 0 or s.n_failures > 0
        )
        n_total_violations = sum(s.n_violations for s in report.per_mutation)
        n_total_failures = sum(s.n_failures for s in report.per_mutation)
        # Surface failures in the same fail line so a grader that just
        # explodes on every row doesn't read as "0 violations".
        bits = [f"{n_total_violations} violations"]
        if n_total_failures:
            bits.append(f"{n_total_failures} failures")
        lines.append(
            f"Status: FAIL: {', '.join(bits)} across {n_violating_mutations} mutations"
        )

    lines.append("")
    lines.append("Per-mutation")
    if report.per_mutation:
        # Pad the name column so columns line up; small enough sets that
        # we just compute the width from the data rather than hard-code.
        name_w = max(len(s.name) for s in report.per_mutation)
        effect_w = max(len(s.expected_effect.upper()) for s in report.per_mutation)
        for s in report.per_mutation:
            mark = _TICK if s.passed else _CROSS
            total = s.n_applied + s.n_skipped_no_op
            lines.append(
                f"  {s.name:<{name_w}}  "
                f"[{s.expected_effect.upper():<{effect_w}}]  "
                f"{mark}  "
                f"applied {s.n_applied}/{total}  "
                f"violations {s.n_violations}  "
                f"failures {s.n_failures}"
            )
    else:
        lines.append("  (no mutations supplied)")

    violations = [o for o in report.outcomes if o.violation is not None]
    if violations:
        ranked = sorted(violations, key=lambda o: abs(o.delta), reverse=True)
        shown = ranked[: max(max_outcomes, 0)]
        lines.append("")
        lines.append(f"Top violations (first {len(shown)})")
        for o in shown:
            lines.append(
                f"  row={o.row_id} mutation={o.mutation}  "
                f"delta={o.delta:+.3f}  ({o.violation})"
            )

    if report.failures:
        lines.append("")
        lines.append(f"Failures ({len(report.failures)})")
        for f in report.failures:
            lines.append(
                f"  row={f.row_id} mutation={f.mutation} side={f.side}  "
                f"{f.error_type}: {f.error_message}"
            )

    return "\n".join(lines)
