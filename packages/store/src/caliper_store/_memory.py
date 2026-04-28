"""In-memory store implementation.

Useful for tests, dev work, and the default app-startup state when no
``CALIPER_STORE_URL`` is configured. Not durable across processes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from caliper_sdk import EvalRow

from ._hash import compute_version
from ._types import EvalSet, EvalSetVersion
from .exceptions import EvalSetNotFound


class InMemoryStore:
    """Thread-unsafe; suitable for single-process FastAPI workers."""

    def __init__(self) -> None:
        # Outer key: name. Inner key: version. Value: (metadata, rows).
        self._versions: dict[str, dict[str, tuple[EvalSetVersion, list[EvalRow]]]] = {}

    def list_names(self) -> list[str]:
        return sorted(name for name, versions in self._versions.items() if versions)

    def list_versions(self, name: str) -> list[EvalSetVersion]:
        versions = self._versions.get(name, {}).values()
        return sorted(
            (meta for meta, _ in versions),
            key=lambda v: v.created_at,
            reverse=True,
        )

    def get(self, name: str, version: str | None = None) -> EvalSet:
        bucket = self._versions.get(name)
        if not bucket:
            raise EvalSetNotFound(f"eval set {name!r} not found")

        if version is None:
            latest_meta, rows = max(bucket.values(), key=lambda pair: pair[0].created_at)
            return EvalSet(version=latest_meta, rows=list(rows))

        pair = bucket.get(version)
        if pair is None:
            raise EvalSetNotFound(f"eval set {name!r}@{version} not found")
        meta, rows = pair
        return EvalSet(version=meta, rows=list(rows))

    def create(
        self, name: str, rows: list[EvalRow], description: str = ""
    ) -> EvalSetVersion:
        version_hash = compute_version(rows)
        bucket = self._versions.setdefault(name, {})

        existing = bucket.get(version_hash)
        if existing is not None:
            # Idempotent: return the stored metadata. Description on the
            # original wins; the second create() call doesn't overwrite.
            return existing[0]

        # Sort rows by id for deterministic iteration order downstream.
        sorted_rows = sorted(rows, key=lambda r: r.id)
        meta = EvalSetVersion(
            name=name,
            version=version_hash,
            created_at=datetime.now(UTC),
            description=description,
            n_rows=len(sorted_rows),
        )
        bucket[version_hash] = (meta, sorted_rows)
        return meta


__all__ = ["InMemoryStore"]
