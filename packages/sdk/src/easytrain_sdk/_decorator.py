"""The @grader decorator.

Wraps a plain Python function into a typed grader callable that:

* coerces dict inputs to ``Rollout``,
* accepts both ``-> float`` (convenience) and ``-> GraderResult`` (full) return signatures,
* validates the resulting score is in [0.0, 1.0],
* attaches ``name``, ``version``, and ``__wrapped__`` attributes.

Usage::

    @grader
    def my_grader(rollout: Rollout) -> float:
        return 1.0 if rollout.response == rollout.gold else 0.0

    @grader(name="legal_citation", version="1.2.0")
    def legal(rollout: Rollout) -> GraderResult:
        ...
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, overload

from easytrain_sdk.types import GraderResult, Rollout

DEFAULT_VERSION = "0.1.0"


def _coerce_rollout(rollout: Rollout | dict[str, Any]) -> Rollout:
    if isinstance(rollout, Rollout):
        return rollout
    if isinstance(rollout, dict):
        return Rollout.model_validate(rollout)
    raise TypeError(
        f"grader expected Rollout or dict, got {type(rollout).__name__}"
    )


def _coerce_result(value: Any) -> GraderResult:
    """Turn a raw grader return value into a validated GraderResult."""
    if isinstance(value, GraderResult):
        result = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        # bool is a subclass of int — exclude it explicitly so a stray True/False
        # surfaces as a clear error instead of silently scoring 1.0/0.0.
        result = GraderResult(score=float(value))
    else:
        raise TypeError(
            f"grader must return float or GraderResult, got {type(value).__name__}"
        )

    if not (0.0 <= result.score <= 1.0):
        raise ValueError(
            f"grader score {result.score!r} is outside the allowed range [0.0, 1.0]"
        )
    return result


def _wrap(
    func: Callable[[Rollout], Any],
    *,
    name: str | None,
    version: str | None,
) -> Callable[[Rollout | dict[str, Any]], GraderResult]:
    @functools.wraps(func)
    def wrapper(rollout: Rollout | dict[str, Any]) -> GraderResult:
        coerced = _coerce_rollout(rollout)
        raw = func(coerced)
        return _coerce_result(raw)

    wrapper.name = name or func.__name__  # type: ignore[attr-defined]
    wrapper.version = version or DEFAULT_VERSION  # type: ignore[attr-defined]
    # functools.wraps already sets __wrapped__, but be explicit to satisfy the contract.
    wrapper.__wrapped__ = func  # type: ignore[attr-defined]
    return wrapper


@overload
def grader(func: Callable[[Rollout], Any], /) -> Callable[[Rollout | dict[str, Any]], GraderResult]:
    ...


@overload
def grader(
    *,
    name: str | None = None,
    version: str | None = None,
) -> Callable[
    [Callable[[Rollout], Any]],
    Callable[[Rollout | dict[str, Any]], GraderResult],
]:
    ...


def grader(
    func: Callable[[Rollout], Any] | None = None,
    /,
    *,
    name: str | None = None,
    version: str | None = None,
) -> Any:
    """Decorator that turns a function into a typed grader.

    Works as a bare decorator (``@grader``) or with keyword arguments
    (``@grader(name="x", version="1.0.0")``).
    """
    if func is not None:
        # Bare @grader form.
        if not callable(func):
            raise TypeError("@grader must be applied to a callable")
        return _wrap(func, name=name, version=version)

    # Parameterized @grader(...) form.
    def decorator(
        f: Callable[[Rollout], Any],
    ) -> Callable[[Rollout | dict[str, Any]], GraderResult]:
        return _wrap(f, name=name, version=version)

    return decorator
