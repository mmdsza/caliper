"""Load a @grader-decorated callable from user-supplied Python source.

Lives in the SDK so both the server's in-process executor and the sandbox
driver can share the same load semantics. The driver runs in a separate
interpreter inside the E2B sandbox and only has the SDK on its path.
"""

from __future__ import annotations

from dataclasses import dataclass

from caliper_sdk.types import GraderCallable


class GraderLoadError(Exception):
    """Raised when source cannot be parsed, executed, or contains no grader."""


@dataclass
class LoadedGrader:
    grader: GraderCallable
    name: str
    version: str


def _looks_like_grader(obj: object) -> bool:
    """Duck-type a grader: callable with `name`, `version`, and `__wrapped__`.

    The SDK's `@grader` decorator sets all three. Accept any object that
    satisfies the structural shape so users can supply their own wrappers.
    """
    return (
        callable(obj)
        and hasattr(obj, "name")
        and hasattr(obj, "version")
        and hasattr(obj, "__wrapped__")
    )


def load_grader_from_source(source: str) -> LoadedGrader:
    """Compile and execute `source`, return the (single) decorated grader.

    Raises `GraderLoadError` with a useful message on parse failure, exec
    failure, no grader found, or multiple graders found.
    """
    try:
        code = compile(source, "<grader>", "exec")
    except SyntaxError as e:
        raise GraderLoadError(f"syntax error at line {e.lineno}: {e.msg}") from e

    namespace: dict[str, object] = {}
    try:
        exec(code, namespace)  # noqa: S102 - executor decides if this runs in-process or sandboxed
    except Exception as e:
        raise GraderLoadError(f"{type(e).__name__} during import: {e}") from e

    found: list[tuple[str, GraderCallable]] = [
        (name, obj)  # type: ignore[misc]
        for name, obj in namespace.items()
        if not name.startswith("_") and _looks_like_grader(obj)
    ]

    if not found:
        raise GraderLoadError(
            "no @grader-decorated function found — did you import `grader` from `caliper_sdk`?"
        )
    if len(found) > 1:
        names = ", ".join(n for n, _ in found)
        raise GraderLoadError(f"multiple graders defined ({names}); supply only one per source")

    binding_name, grader_obj = found[0]
    return LoadedGrader(
        grader=grader_obj,
        name=getattr(grader_obj, "name", binding_name),
        version=getattr(grader_obj, "version", "0.0.0"),
    )


__all__ = ["GraderLoadError", "LoadedGrader", "load_grader_from_source"]
