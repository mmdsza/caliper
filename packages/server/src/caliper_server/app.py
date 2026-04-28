"""FastAPI app for the Caliper dry-run backend."""

from __future__ import annotations

from typing import Any

from caliper_sdk import GraderLoadError, RunReport
from caliper_store import EvalSetNotFound, EvalSetStore
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from caliper_server._seed import seed_default_eval_sets
from caliper_server._store_factory import build_store_from_env
from caliper_server.executor import (
    GraderExecutor,
    SandboxError,
    build_from_env,
)


class DryRunRequest(BaseModel):
    source: str = Field(..., description="Python source containing one @grader-decorated function")
    eval_set: str = Field(..., description="Name of a registered eval set")
    eval_set_version: str | None = Field(
        default=None,
        description="Optional content-hash pin. Omit for the latest version.",
    )


class EvalRowOut(BaseModel):
    id: str
    prompt: str
    response: str
    gold: str | None


class EvalSetVersionOut(BaseModel):
    name: str
    version: str
    description: str
    n_rows: int


class DryRunResponse(BaseModel):
    grader_name: str
    grader_version: str
    eval_set: str
    eval_set_version: str
    report: RunReport
    rows: list[EvalRowOut]


def _row_to_out(row: Any) -> EvalRowOut:
    return EvalRowOut(
        id=row.id,
        prompt=row.rollout.prompt,
        response=row.rollout.response,
        gold=row.rollout.gold,
    )


def create_app(
    executor: GraderExecutor | None = None,
    store: EvalSetStore | None = None,
    *,
    seed_defaults: bool = True,
) -> FastAPI:
    app = FastAPI(title="Caliper dry-run backend", version="0.0.1")

    # Executor + store are resolved at app-build time so the choice is
    # fixed for the life of the process and clearly visible in logs.
    # Tests inject fakes by passing the arguments or by replacing
    # ``app.state.executor`` / ``app.state.store`` after construction.
    app.state.executor = executor if executor is not None else build_from_env()
    app.state.store = store if store is not None else build_store_from_env()

    if seed_defaults:
        seed_default_eval_sets(app.state.store)

    # Editor talks to us via Next.js rewrites in prod; CORS open in dev
    # because direct fetches from localhost:3456 to localhost:8000 are
    # convenient when probing the API by hand.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3456"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/eval-sets")
    def list_eval_sets(request: Request) -> dict[str, list[str]]:
        store: EvalSetStore = request.app.state.store
        return {"names": store.list_names()}

    @app.get("/eval-sets/{name}")
    def get_eval_set(
        name: str, request: Request, version: str | None = None
    ) -> dict[str, Any]:
        store: EvalSetStore = request.app.state.store
        try:
            eval_set = store.get(name, version=version)
        except EvalSetNotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return {
            "name": eval_set.version.name,
            "version": eval_set.version.version,
            "description": eval_set.version.description,
            "n_rows": eval_set.version.n_rows,
            "rows": [_row_to_out(r) for r in eval_set.rows],
        }

    @app.get("/eval-sets/{name}/versions")
    def list_eval_set_versions(name: str, request: Request) -> dict[str, Any]:
        store: EvalSetStore = request.app.state.store
        versions = store.list_versions(name)
        if not versions:
            raise HTTPException(status_code=404, detail=f"eval set {name!r} not found")
        return {
            "name": name,
            "versions": [
                EvalSetVersionOut(
                    name=v.name,
                    version=v.version,
                    description=v.description,
                    n_rows=v.n_rows,
                ).model_dump()
                for v in versions
            ],
        }

    @app.post("/dry-run", response_model=DryRunResponse)
    def dry_run(req: DryRunRequest, request: Request) -> DryRunResponse:
        store: EvalSetStore = request.app.state.store
        try:
            eval_set = store.get(req.eval_set, version=req.eval_set_version)
        except EvalSetNotFound as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

        executor: GraderExecutor = request.app.state.executor
        try:
            result = executor.execute(req.source, eval_set.rows)
        except GraderLoadError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except SandboxError as e:
            raise HTTPException(status_code=502, detail=f"sandbox error: {e}") from e

        return DryRunResponse(
            grader_name=result.name,
            grader_version=result.version,
            eval_set=eval_set.version.name,
            eval_set_version=eval_set.version.version,
            report=result.report,
            rows=[_row_to_out(r) for r in eval_set.rows],
        )

    return app


app = create_app()
