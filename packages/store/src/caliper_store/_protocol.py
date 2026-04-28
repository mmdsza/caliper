"""The :class:`EvalSetStore` Protocol that all backends satisfy.

The contract is intentionally narrow:

* :meth:`get` — fetch an eval set by name, optionally pinned to a version.
* :meth:`list_names` — names with at least one version.
* :meth:`list_versions` — all versions of a named set, newest first.
* :meth:`create` — write a new version. Idempotent on identical content
  (returns the pre-existing version metadata if the hash already exists).

Diff between versions is deliberately out of scope for this slice — see
ADR-0006.
"""

from __future__ import annotations

from typing import Protocol

from caliper_sdk import EvalRow

from ._types import EvalSet, EvalSetVersion


class EvalSetStore(Protocol):
    def list_names(self) -> list[str]:
        """All eval-set names with at least one version, sorted alphabetically."""

    def list_versions(self, name: str) -> list[EvalSetVersion]:
        """All versions of ``name``, newest first by ``created_at``.

        Returns an empty list if ``name`` has no versions; never raises.
        """

    def get(self, name: str, version: str | None = None) -> EvalSet:
        """Fetch an eval set.

        With ``version=None`` returns the latest version (by ``created_at``).
        With an explicit version returns that exact version. Raises
        :class:`~caliper_store.exceptions.EvalSetNotFound` if either
        the name or the (name, version) pair is unknown.
        """

    def create(
        self, name: str, rows: list[EvalRow], description: str = ""
    ) -> EvalSetVersion:
        """Write a new version of ``name`` containing ``rows``.

        The version is the content hash of the rows. If a version with the
        same hash already exists for this name, returns the existing
        :class:`EvalSetVersion` unchanged (idempotent).
        """


__all__ = ["EvalSetStore"]
