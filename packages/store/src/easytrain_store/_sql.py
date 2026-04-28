"""SQLAlchemy 2.x backend for the eval-set store.

Works against any SQLAlchemy URL — SQLite for local + tests, Postgres
in production. The schema uses ``JSON`` (mapped to TEXT in SQLite,
JSONB in Postgres) so the same code path serves both.

We keep the SQL deliberately narrow: no JSONB operators, no GIN indexes,
no Postgres-specific syntax. When we need them later (e.g. for full-text
search across rollouts) we'll branch on the dialect at that point.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from easytrain_sdk import EvalRow
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKeyConstraint,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from ._hash import compute_version
from ._types import EvalSet, EvalSetVersion
from .exceptions import EvalSetNotFound


class _Base(DeclarativeBase):
    """Per-engine declarative base.

    We avoid a module-level base so test code can spin up multiple
    isolated engines (e.g. one per test) without the metadata catalog
    bleeding state across them.
    """


class _EvalSetRow(_Base):
    __tablename__ = "eval_sets"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    description: Mapped[str] = mapped_column(String, default="")

    rows: Mapped[list[_EvalRowRecord]] = relationship(
        back_populates="eval_set",
        cascade="all, delete-orphan",
        order_by="_EvalRowRecord.row_id",
    )


class _EvalRowRecord(_Base):
    __tablename__ = "eval_set_rows"

    eval_set_name: Mapped[str] = mapped_column(String, primary_key=True)
    eval_set_version: Mapped[str] = mapped_column(String, primary_key=True)
    row_id: Mapped[str] = mapped_column(String, primary_key=True)
    rollout_json: Mapped[dict[str, Any]] = mapped_column(JSON)

    __table_args__ = (
        ForeignKeyConstraint(
            ["eval_set_name", "eval_set_version"],
            ["eval_sets.name", "eval_sets.version"],
            ondelete="CASCADE",
        ),
    )

    eval_set: Mapped[_EvalSetRow] = relationship(back_populates="rows")


class SqlStore:
    """SQLAlchemy-backed :class:`~easytrain_store.EvalSetStore`.

    Construct with a URL:

    >>> store = SqlStore("sqlite:///:memory:")              # tests
    >>> store = SqlStore("sqlite:///./easytrain.db")        # local dev
    >>> store = SqlStore("postgresql+psycopg://u:p@h/db")   # prod

    Tables are created on first use; this is fine for the MVP. Once we
    have multiple processes writing concurrently we'll switch to Alembic
    migrations.
    """

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._engine = create_engine(url, echo=echo, future=True)
        _Base.metadata.create_all(self._engine)

    # ----- Protocol surface -----

    def list_names(self) -> list[str]:
        with Session(self._engine) as session:
            stmt = select(_EvalSetRow.name).distinct().order_by(_EvalSetRow.name)
            return list(session.scalars(stmt))

    def list_versions(self, name: str) -> list[EvalSetVersion]:
        with Session(self._engine) as session:
            stmt = (
                select(_EvalSetRow)
                .where(_EvalSetRow.name == name)
                .order_by(_EvalSetRow.created_at.desc())
            )
            return [_to_version(row) for row in session.scalars(stmt)]

    def get(self, name: str, version: str | None = None) -> EvalSet:
        with Session(self._engine) as session:
            if version is None:
                stmt = (
                    select(_EvalSetRow)
                    .where(_EvalSetRow.name == name)
                    .order_by(_EvalSetRow.created_at.desc())
                    .limit(1)
                )
                eval_set = session.scalars(stmt).first()
                if eval_set is None:
                    raise EvalSetNotFound(f"eval set {name!r} not found")
            else:
                eval_set = session.get(_EvalSetRow, (name, version))
                if eval_set is None:
                    raise EvalSetNotFound(f"eval set {name!r}@{version} not found")

            meta = _to_version(eval_set)
            rows = [
                EvalRow.model_validate(record.rollout_json) for record in eval_set.rows
            ]
            return EvalSet(version=meta, rows=rows)

    def create(
        self, name: str, rows: list[EvalRow], description: str = ""
    ) -> EvalSetVersion:
        version_hash = compute_version(rows)

        with Session(self._engine) as session:
            existing = session.get(_EvalSetRow, (name, version_hash))
            if existing is not None:
                # Idempotent. Return whatever's already stored.
                return _to_version(existing)

            sorted_rows = sorted(rows, key=lambda r: r.id)
            now = datetime.now(UTC)
            eval_set = _EvalSetRow(
                name=name,
                version=version_hash,
                created_at=now,
                description=description,
                rows=[
                    _EvalRowRecord(
                        eval_set_name=name,
                        eval_set_version=version_hash,
                        row_id=row.id,
                        rollout_json=row.model_dump(mode="json"),
                    )
                    for row in sorted_rows
                ],
            )
            session.add(eval_set)
            session.commit()
            session.refresh(eval_set)
            return _to_version(eval_set)


def _to_version(row: _EvalSetRow) -> EvalSetVersion:
    """Translate the ORM row into the public Pydantic type.

    ``created_at`` round-trips through SQLite as a naive datetime (it
    drops tz info on store), so we re-attach UTC on read. Postgres
    preserves tz natively and this is a no-op for tz-aware values.
    """
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return EvalSetVersion(
        name=row.name,
        version=row.version,
        created_at=created_at,
        description=row.description,
        n_rows=len(row.rows),
    )


__all__ = ["SqlStore"]
