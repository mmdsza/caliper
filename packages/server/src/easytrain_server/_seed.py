"""Seed data for the eval-set store.

Sprint 0/1.x served the ``legal_v3`` fixture out of an in-process Python
module. Sprint 1.6 moves the canonical store to a SQLAlchemy backend; the
fixture's row payload now lives here and is loaded into whatever store
the app boots with via :func:`seed_default_eval_sets`.

The 10 rows are designed so the sample legal-citation grader (see
``packages/editor/lib/sample-grader.ts``) yields exactly 5 of 10 — a
non-trivial dry-run distribution on first load.
"""

from __future__ import annotations

from easytrain_sdk import EvalRow, Rollout
from easytrain_store import EvalSetStore

LEGAL_V3_DESCRIPTION = (
    "10-row synthetic legal-citation eval set. Designed so the sample "
    "<think>-gated grader scores exactly 5 of 10 rows for a non-trivial "
    "dry-run histogram. Mix of format-gate-pass-and-match, gate-pass-and-"
    "mismatch, and missing-gate cases."
)

LEGAL_V3_ROWS: list[EvalRow] = [
    # Format gate present, gold match — score 1.0
    EvalRow(
        id="legal_v3-001",
        rollout=Rollout(
            prompt="Cite the holding in Marbury v. Madison.",
            response=(
                "<think>The case established judicial review.</think>"
                "Marbury v. Madison, 5 U.S. 137 (1803)"
            ),
            gold="Marbury v. Madison, 5 U.S. 137 (1803)",
        ),
    ),
    EvalRow(
        id="legal_v3-002",
        rollout=Rollout(
            prompt="What is the citation for Brown v. Board of Education?",
            response=(
                "<think>Landmark desegregation case.</think>"
                "Brown v. Board of Education, 347 U.S. 483 (1954)"
            ),
            gold="Brown v. Board of Education, 347 U.S. 483 (1954)",
        ),
    ),
    EvalRow(
        id="legal_v3-003",
        rollout=Rollout(
            prompt="Cite Roe v. Wade.",
            response="<think>1973 abortion rights case.</think>Roe v. Wade, 410 U.S. 113 (1973)",
            gold="Roe v. Wade, 410 U.S. 113 (1973)",
        ),
    ),
    # Format gate present, gold mismatch — score 0.0
    EvalRow(
        id="legal_v3-004",
        rollout=Rollout(
            prompt="Cite Miranda v. Arizona.",
            response="<think>Self-incrimination case.</think>Miranda v. Arizona, 384 U.S. 436",
            gold="Miranda v. Arizona, 384 U.S. 436 (1966)",
        ),
    ),
    EvalRow(
        id="legal_v3-005",
        rollout=Rollout(
            prompt="What is the citation for Gideon v. Wainwright?",
            response=(
                "<think>Right-to-counsel case.</think>"
                "Gideon v. Wainwright, 372 US 335 (1963)"
            ),
            gold="Gideon v. Wainwright, 372 U.S. 335 (1963)",
        ),
    ),
    EvalRow(
        id="legal_v3-006",
        rollout=Rollout(
            prompt="Cite Mapp v. Ohio.",
            response="<think>Exclusionary rule.</think>Mapp v. Ohio (1961)",
            gold="Mapp v. Ohio, 367 U.S. 643 (1961)",
        ),
    ),
    # Missing format gate — score 0.0 (hard fail per grader pattern 1)
    EvalRow(
        id="legal_v3-007",
        rollout=Rollout(
            prompt="Cite Tinker v. Des Moines.",
            response="Tinker v. Des Moines, 393 U.S. 503 (1969)",
            gold="Tinker v. Des Moines, 393 U.S. 503 (1969)",
        ),
    ),
    EvalRow(
        id="legal_v3-008",
        rollout=Rollout(
            prompt="What is the citation for New York Times v. Sullivan?",
            response="That would be NYT v. Sullivan, 376 U.S. 254 (1964).",
            gold="New York Times v. Sullivan, 376 U.S. 254 (1964)",
        ),
    ),
    # Format gate, gold match — another pass
    EvalRow(
        id="legal_v3-009",
        rollout=Rollout(
            prompt="Cite Loving v. Virginia.",
            response=(
                "<think>Interracial marriage case.</think>"
                "Loving v. Virginia, 388 U.S. 1 (1967)"
            ),
            gold="Loving v. Virginia, 388 U.S. 1 (1967)",
        ),
    ),
    # Empty think gate, gold present in response — also passes the simple grader
    EvalRow(
        id="legal_v3-010",
        rollout=Rollout(
            prompt="Cite Obergefell v. Hodges.",
            response="<think></think>Obergefell v. Hodges, 576 U.S. 644 (2015)",
            gold="Obergefell v. Hodges, 576 U.S. 644 (2015)",
        ),
    ),
]


def seed_default_eval_sets(store: EvalSetStore) -> None:
    """Ensure ``legal_v3`` exists in ``store``.

    Idempotent — relies on :meth:`EvalSetStore.create`'s content-hash
    idempotency. Calling this on every app startup is cheap and safe.
    """
    if "legal_v3" not in store.list_names():
        store.create("legal_v3", LEGAL_V3_ROWS, description=LEGAL_V3_DESCRIPTION)


__all__ = ["LEGAL_V3_DESCRIPTION", "LEGAL_V3_ROWS", "seed_default_eval_sets"]
