"""Build the eval-set store from environment configuration.

Selection rule (single env var):

    EASYTRAIN_STORE_URL unset                       -> InMemoryStore
    EASYTRAIN_STORE_URL=sqlite:///path/to/file.db   -> SqlStore (SQLite)
    EASYTRAIN_STORE_URL=postgresql+psycopg://...    -> SqlStore (Postgres)

Defaulting to in-memory keeps ``uv run pytest`` hermetic and offline —
same posture as the executor in ADR-0004. Production deployments set
the URL.
"""

from __future__ import annotations

import os

from easytrain_store import EvalSetStore, InMemoryStore, SqlStore


def build_store_from_env(env: dict[str, str] | None = None) -> EvalSetStore:
    url = (env or os.environ).get("EASYTRAIN_STORE_URL", "").strip()
    if not url:
        return InMemoryStore()
    return SqlStore(url)


__all__ = ["build_store_from_env"]
