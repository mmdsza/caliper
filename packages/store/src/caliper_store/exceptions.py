"""Typed exceptions raised by EvalSetStore implementations."""

from __future__ import annotations


class EvalSetStoreError(Exception):
    """Base for all store-level errors."""


class EvalSetNotFound(EvalSetStoreError):  # noqa: N818 — domain "not found", not an error suffix
    """No eval set exists with the given (name, version) tuple."""


class VersionAlreadyExists(EvalSetStoreError):  # noqa: N818 — reads as a state, not an error suffix
    """A `create()` call produced a content hash that's already present.

    Idempotency guarantee: this is what callers see when they re-create
    an identical eval set. Implementations should still return the
    pre-existing :class:`EvalSetVersion` rather than raising — this
    exception is reserved for the rare case where the metadata differs
    (e.g. the description) but the content hash collides.
    """


__all__ = [
    "EvalSetNotFound",
    "EvalSetStoreError",
    "VersionAlreadyExists",
]
