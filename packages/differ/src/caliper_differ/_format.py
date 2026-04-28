"""Plain-text formatter for `DiffReport` — used by the CLI and editor diff panel."""

from __future__ import annotations

from ._report import DiffReport


def format_report(report: DiffReport, top_k: int = 10) -> str:
    """Render a `DiffReport` as a multi-line human-readable string.

    Layout:
      - header line with row count and # of failures
      - summary stats block (n_changed, promoted, demoted, mean/median/p95
        delta, max promotion/demotion)
      - top-k rows by |delta|, descending
      - failure list (if any)

    No colour, no rich; just predictable text so it diffs cleanly in
    test snapshots and renders the same in any terminal.
    """
    lines: list[str] = []

    lines.append(
        f"Grader Diff Report — n_rows={report.n_rows} "
        f"(succeeded={len(report.per_row)}, failed={len(report.failures)})"
    )
    lines.append("=" * 72)
    lines.append("Summary:")
    lines.append(f"  n_changed    : {report.n_changed}")
    lines.append(f"  promoted     : {report.n_promoted}")
    lines.append(f"  demoted      : {report.n_demoted}")
    lines.append(f"  mean_delta   : {report.mean_delta:+.6f}")
    lines.append(f"  median_delta : {report.median_delta:+.6f}")
    lines.append(f"  p95_|delta|  : {report.p95_delta:.6f}")
    lines.append(f"  max_promotion: {report.max_promotion:+.6f}")
    lines.append(f"  max_demotion : {report.max_demotion:+.6f}")

    if report.per_row:
        ranked = sorted(report.per_row, key=lambda r: abs(r.delta), reverse=True)
        shown = ranked[: max(top_k, 0)]
        lines.append("")
        lines.append(f"Top {len(shown)} rows by |delta|:")
        lines.append(
            f"  {'row_id':<24} {'v1':>10} {'v2':>10} {'delta':>10}  class"
        )
        for r in shown:
            lines.append(
                f"  {r.row_id:<24} {r.v1_score:>10.4f} {r.v2_score:>10.4f} "
                f"{r.delta:>+10.4f}  {r.classification}"
            )

    if report.failures:
        lines.append("")
        lines.append(f"Failures ({len(report.failures)}):")
        for f in report.failures:
            lines.append(
                f"  {f.row_id:<24} side={f.side:<4} {f.error_type}: {f.error_message}"
            )

    return "\n".join(lines)
