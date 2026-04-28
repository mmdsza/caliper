"""Pydantic types describing eval-set metadata and content.

Rows themselves are :class:`caliper_sdk.EvalRow` — we deliberately do
not redefine the row shape here. The store owns versioning and storage;
the SDK owns the row schema.
"""

from __future__ import annotations

from datetime import datetime

from caliper_sdk import EvalRow
from pydantic import BaseModel, ConfigDict, Field


class EvalSetVersion(BaseModel):
    """Metadata for one immutable version of a named eval set.

    ``version`` is a 12-char hex string from :func:`compute_version`.
    Together with ``name`` it uniquely identifies the contents.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    created_at: datetime
    description: str = ""
    n_rows: int = Field(ge=0)


class EvalSet(BaseModel):
    """An :class:`EvalSetVersion` plus the rows it contains.

    Returned by :meth:`EvalSetStore.get`. Iteration order matches insertion
    order (stable per content hash, since the rows are sorted by id at
    hash time and stored in that order).
    """

    model_config = ConfigDict(extra="forbid")

    version: EvalSetVersion
    rows: list[EvalRow] = Field(default_factory=list)


__all__ = ["EvalSet", "EvalSetVersion"]
