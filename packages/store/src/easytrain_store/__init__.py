"""Eval-set storage for EasyTrain.

Public surface:
    EvalSetStore        — Protocol satisfied by all backends
    InMemoryStore       — process-local, used in tests and as the dev default
    SqlStore            — SQLAlchemy 2.x backend (SQLite locally, Postgres in prod)
    EvalSet             — fetched eval set: metadata + rows
    EvalSetVersion      — metadata for one version (name, version, created_at, ...)
    compute_version     — content hash helper (12-char SHA-256 prefix)
    VERSION_HASH_LEN    — the 12 above, exported for callers that format hashes
    EvalSetNotFound     — raised by get() when name or version is unknown
    EvalSetStoreError   — base exception
    VersionAlreadyExists — reserved for content-hash collisions with metadata mismatch
"""

from ._hash import VERSION_HASH_LEN, compute_version
from ._memory import InMemoryStore
from ._protocol import EvalSetStore
from ._sql import SqlStore
from ._types import EvalSet, EvalSetVersion
from .exceptions import (
    EvalSetNotFound,
    EvalSetStoreError,
    VersionAlreadyExists,
)

__all__ = [
    "VERSION_HASH_LEN",
    "EvalSet",
    "EvalSetNotFound",
    "EvalSetStore",
    "EvalSetStoreError",
    "EvalSetVersion",
    "InMemoryStore",
    "SqlStore",
    "VersionAlreadyExists",
    "compute_version",
]
