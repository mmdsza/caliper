"""Protocol conformance tests run against every backend.

Same test body, parametrized over :class:`InMemoryStore` and
:class:`SqlStore` backed by SQLite. Adding a new backend means adding it
here — the test surface stays unified.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from easytrain_sdk import EvalRow, Rollout
from easytrain_store import (
    EvalSetNotFound,
    EvalSetStore,
    InMemoryStore,
    SqlStore,
    compute_version,
)

BackendFactory = Callable[[], EvalSetStore]


def _in_memory() -> EvalSetStore:
    return InMemoryStore()


def _sqlite() -> EvalSetStore:
    # ``:memory:`` is per-connection; multiple Sessions opened against the
    # same engine share state because SQLAlchemy holds the connection pool.
    return SqlStore("sqlite:///:memory:")


@pytest.fixture(params=[_in_memory, _sqlite], ids=["in_memory", "sqlite"])
def store(request: Any) -> EvalSetStore:
    factory: BackendFactory = request.param
    return factory()


# ---------- list_names / list_versions on empty store ----------


class TestEmptyStore:
    def test_list_names_returns_empty_list(self, store: EvalSetStore) -> None:
        assert store.list_names() == []

    def test_list_versions_for_unknown_name_returns_empty_list(
        self, store: EvalSetStore
    ) -> None:
        assert store.list_versions("nope") == []

    def test_get_unknown_name_raises(self, store: EvalSetStore) -> None:
        with pytest.raises(EvalSetNotFound, match="'nope'"):
            store.get("nope")


# ---------- create() ----------


class TestCreate:
    def test_create_returns_metadata_with_content_hash(
        self, store: EvalSetStore, two_rows: list[EvalRow]
    ) -> None:
        meta = store.create("legal_v3", two_rows, description="initial")
        assert meta.name == "legal_v3"
        assert meta.version == compute_version(two_rows)
        assert meta.n_rows == len(two_rows)
        assert meta.description == "initial"
        assert meta.created_at is not None

    def test_create_makes_the_eval_set_visible(
        self, store: EvalSetStore, two_rows: list[EvalRow]
    ) -> None:
        store.create("legal_v3", two_rows)
        assert "legal_v3" in store.list_names()

    def test_create_idempotent_on_identical_rows(
        self, store: EvalSetStore, two_rows: list[EvalRow]
    ) -> None:
        first = store.create("legal_v3", two_rows, description="initial")
        # Same content, different description — content hash wins, the
        # second call returns the original metadata.
        second = store.create("legal_v3", two_rows, description="ignored")
        assert first.version == second.version
        assert second.description == "initial"
        # Only one version exists.
        assert len(store.list_versions("legal_v3")) == 1

    def test_create_two_different_versions(
        self,
        store: EvalSetStore,
        two_rows: list[EvalRow],
        three_rows: list[EvalRow],
    ) -> None:
        v1 = store.create("legal_v3", two_rows, description="v1")
        time.sleep(0.005)  # ensure distinct created_at on coarse-grained clocks
        v2 = store.create("legal_v3", three_rows, description="v2")
        assert v1.version != v2.version
        versions = store.list_versions("legal_v3")
        assert {v.version for v in versions} == {v1.version, v2.version}
        # Newest first.
        assert versions[0].version == v2.version

    def test_create_independent_of_row_input_order(
        self, store: EvalSetStore, two_rows: list[EvalRow]
    ) -> None:
        v1 = store.create("a", two_rows)
        v2 = store.create("a", list(reversed(two_rows)))
        assert v1.version == v2.version
        assert len(store.list_versions("a")) == 1


# ---------- get() ----------


class TestGet:
    def test_get_no_version_returns_latest(
        self,
        store: EvalSetStore,
        two_rows: list[EvalRow],
        three_rows: list[EvalRow],
    ) -> None:
        store.create("legal_v3", two_rows, description="v1")
        time.sleep(0.005)
        latest_meta = store.create("legal_v3", three_rows, description="v2")

        eval_set = store.get("legal_v3")
        assert eval_set.version.version == latest_meta.version
        assert eval_set.version.description == "v2"
        assert len(eval_set.rows) == len(three_rows)

    def test_get_explicit_version(
        self,
        store: EvalSetStore,
        two_rows: list[EvalRow],
        three_rows: list[EvalRow],
    ) -> None:
        v1 = store.create("legal_v3", two_rows, description="v1")
        time.sleep(0.005)
        store.create("legal_v3", three_rows, description="v2")

        eval_set = store.get("legal_v3", version=v1.version)
        assert eval_set.version.version == v1.version
        assert len(eval_set.rows) == len(two_rows)

    def test_get_unknown_version_raises(
        self, store: EvalSetStore, two_rows: list[EvalRow]
    ) -> None:
        store.create("legal_v3", two_rows)
        with pytest.raises(EvalSetNotFound, match="@deadbeef"):
            store.get("legal_v3", version="deadbeef0000")

    def test_get_returns_rows_in_id_sorted_order(
        self, store: EvalSetStore
    ) -> None:
        rows = [
            EvalRow(id="r-005", rollout=Rollout(prompt="?", response="x")),
            EvalRow(id="r-001", rollout=Rollout(prompt="?", response="x")),
            EvalRow(id="r-003", rollout=Rollout(prompt="?", response="x")),
        ]
        store.create("ordered", rows)
        eval_set = store.get("ordered")
        assert [r.id for r in eval_set.rows] == ["r-001", "r-003", "r-005"]

    def test_get_round_trips_full_rollout_payload(
        self, store: EvalSetStore
    ) -> None:
        original = [
            EvalRow(
                id="r-001",
                rollout=Rollout(
                    prompt="please cite Marbury",
                    response="<think>...</think>Marbury v. Madison, 5 U.S. 137 (1803)",
                    gold="Marbury v. Madison, 5 U.S. 137 (1803)",
                    metadata={"source": "manual", "score_hint": 1.0},
                ),
            ),
        ]
        store.create("legal_v3", original)
        loaded = store.get("legal_v3")
        assert loaded.rows == original


# ---------- list_names / list_versions on populated store ----------


class TestList:
    def test_list_names_alphabetized(
        self, store: EvalSetStore, two_rows: list[EvalRow]
    ) -> None:
        store.create("zebra", two_rows)
        store.create("alpha", two_rows)
        store.create("middle", two_rows)
        assert store.list_names() == ["alpha", "middle", "zebra"]

    def test_list_versions_newest_first(
        self,
        store: EvalSetStore,
        two_rows: list[EvalRow],
        three_rows: list[EvalRow],
    ) -> None:
        v1 = store.create("legal_v3", two_rows)
        time.sleep(0.005)
        v2 = store.create("legal_v3", three_rows)

        versions = store.list_versions("legal_v3")
        assert [v.version for v in versions] == [v2.version, v1.version]
        # Both carry their original n_rows.
        by_hash = {v.version: v for v in versions}
        assert by_hash[v1.version].n_rows == len(two_rows)
        assert by_hash[v2.version].n_rows == len(three_rows)
