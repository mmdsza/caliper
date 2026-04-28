"""Content hashing for eval-set versions.

The hash is a stable function of the rows alone — independent of insertion
order, the description, the timestamp, or anything else. Two ``create()``
calls with the same rows always produce the same version string.

We use SHA-256 truncated to 12 hex chars (48 bits). Collision risk is
negligible at any sane corpus size and short hashes are easier to read in
logs and CLI output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from caliper_sdk import EvalRow

VERSION_HASH_LEN = 12


def compute_version(rows: Iterable[EvalRow]) -> str:
    """Return the 12-char content version for an iterable of rows.

    Determinism rules:

    * Rows are sorted by ``id`` before hashing, so the same set of rows in
      any input order produces the same hash.
    * ``model_dump`` is stringified with ``sort_keys=True`` and the most
      compact JSON separators, so equivalent dicts always produce the
      same bytes.
    """
    canonical = json.dumps(
        [r.model_dump(mode="json") for r in sorted(rows, key=lambda r: r.id)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:VERSION_HASH_LEN]


__all__ = ["VERSION_HASH_LEN", "compute_version"]
