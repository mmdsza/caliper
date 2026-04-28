"""`format_report` should produce a usable text dump."""

from __future__ import annotations

from easytrain_differ import diff_graders, format_report
from easytrain_sdk.types import GraderResult, Rollout


def v1(_rollout: Rollout) -> GraderResult:
    return GraderResult(score=0.5)


def v2(rollout: Rollout) -> GraderResult:
    i = int(rollout.prompt[1:])
    # Vary score so we have non-trivial top-k content.
    return GraderResult(score=min(1.0, max(0.0, 0.5 + 0.05 * (i - 4))))


def test_format_report_contains_required_labels(eval_set_10):
    report = diff_graders(v1, v2, eval_set_10)
    text = format_report(report)

    assert isinstance(text, str)
    assert text  # non-empty
    lower = text.lower()
    for needle in ("n_rows", "promoted", "demoted", "mean_delta"):
        assert needle in lower, f"missing {needle!r} in formatted output"


def test_format_report_top_k_caps_rows(eval_set_10):
    report = diff_graders(v1, v2, eval_set_10)
    top_k = 3
    text = format_report(report, top_k=top_k)

    # The "Top N rows by |delta|" section should list at most top_k row entries.
    # Each row entry begins with two spaces and a row id; we count the lines
    # under the table header that look like data rows.
    lines = text.splitlines()
    # Find the "Top N rows by |delta|:" section header.
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Top "))
    # The "row_id" header line follows directly after.
    table_header_idx = header_idx + 1
    assert "row_id" in lines[table_header_idx]
    # Count subsequent indented lines until a blank or end-of-output.
    data_rows = 0
    for line in lines[table_header_idx + 1 :]:
        if not line.startswith("  "):
            break
        if line.lstrip().startswith("row_id"):
            continue
        data_rows += 1
    assert data_rows <= top_k


def test_format_report_zero_top_k_emits_no_data_rows(eval_set_10):
    report = diff_graders(v1, v2, eval_set_10)
    text = format_report(report, top_k=0)

    # With top_k=0 there should still be a non-empty summary, just no
    # row table data lines.
    assert "n_rows" in text.lower()
