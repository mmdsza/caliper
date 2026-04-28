"""compute_version: stability, determinism, sensitivity."""

from __future__ import annotations

from caliper_sdk import EvalRow, Rollout
from caliper_store import VERSION_HASH_LEN, compute_version


def test_hash_length_and_hex() -> None:
    h = compute_version([EvalRow(id="r0", rollout=Rollout(prompt="?", response="x"))])
    assert len(h) == VERSION_HASH_LEN == 12
    int(h, 16)  # raises if not valid hex


def test_hash_stable_across_runs(two_rows: list[EvalRow]) -> None:
    assert compute_version(two_rows) == compute_version(two_rows)


def test_hash_independent_of_input_order(two_rows: list[EvalRow]) -> None:
    h1 = compute_version(two_rows)
    h2 = compute_version(list(reversed(two_rows)))
    assert h1 == h2


def test_hash_changes_when_a_row_changes() -> None:
    a = [EvalRow(id="r0", rollout=Rollout(prompt="q", response="a"))]
    b = [EvalRow(id="r0", rollout=Rollout(prompt="q", response="b"))]
    assert compute_version(a) != compute_version(b)


def test_hash_changes_when_a_row_id_changes() -> None:
    a = [EvalRow(id="r0", rollout=Rollout(prompt="q", response="a"))]
    b = [EvalRow(id="r1", rollout=Rollout(prompt="q", response="a"))]
    assert compute_version(a) != compute_version(b)


def test_hash_changes_when_a_row_is_added() -> None:
    a = [EvalRow(id="r0", rollout=Rollout(prompt="q", response="a"))]
    b = [*a, EvalRow(id="r1", rollout=Rollout(prompt="q", response="b"))]
    assert compute_version(a) != compute_version(b)


def test_empty_eval_set_has_a_stable_hash() -> None:
    """The hash of zero rows is well-defined and stable — not a special case."""
    assert compute_version([]) == compute_version([])
