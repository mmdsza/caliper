"""Compile EasyTrain grader specs to OpenAI RFT JSON.

The spec types live in :mod:`easytrain_rft.spec`. This module turns a typed
``AnyGrader`` into the dict / JSON shape OpenAI's RFT accepts.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from easytrain_rft.spec import AnyGrader, MultiGrader, ScoreModelGrader

# AST node types allowed inside a MultiGrader.calculate_output expression.
_ALLOWED_BINOPS: tuple[type[ast.operator], ...] = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS: tuple[type[ast.unaryop], ...] = (ast.UAdd, ast.USub)


def validate_calculate_output(expr: str, grader_names: set[str]) -> None:
    """Validate a ``MultiGrader.calculate_output`` arithmetic expression.

    Allowed: ``+``, ``-``, ``*``, ``/``, parentheses, numeric literals, and
    identifiers naming sub-graders. Anything else (calls, attribute access,
    ``**`` / ``%``, unknown names, etc.) raises :class:`ValueError`.
    """

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"calculate_output is not a valid expression: {expr!r}") from exc

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            _walk(node.body)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ValueError(
                    f"calculate_output uses unsupported operator "
                    f"{type(node.op).__name__!s} in {expr!r}; only +, -, *, / are allowed"
                )
            _walk(node.left)
            _walk(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise ValueError(
                    f"calculate_output uses unsupported unary operator "
                    f"{type(node.op).__name__!s} in {expr!r}"
                )
            _walk(node.operand)
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(
                    f"calculate_output may only contain numeric literals; "
                    f"found {node.value!r} in {expr!r}"
                )
            return
        if isinstance(node, ast.Name):
            if node.id not in grader_names:
                raise ValueError(
                    f"calculate_output references unknown sub-grader {node.id!r}; "
                    f"known: {sorted(grader_names)}"
                )
            return
        raise ValueError(
            f"calculate_output contains disallowed syntax "
            f"{type(node).__name__!s} in {expr!r}"
        )

    _walk(tree)


def _post_process(value: Any) -> Any:
    """Normalize Pydantic dump output for OpenAI's RFT shape.

    - Tuples become lists (JSON has no tuple).
    - Dicts and lists are walked recursively.
    """

    if isinstance(value, dict):
        return {k: _post_process(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_post_process(v) for v in value]
    return value


def compile_to_rft(grader: AnyGrader) -> dict[str, Any]:
    """Compile a typed grader spec to the dict shape OpenAI's RFT expects."""

    # Pydantic discriminator + Annotated unions: when callers pass a concrete
    # model instance directly, just dump it. ``mode="python"`` keeps tuples
    # so we can normalise them to lists ourselves.
    dumped = grader.model_dump(exclude_none=True, mode="python")
    return _post_process(dumped)


def compile_to_rft_json(grader: AnyGrader, **json_kwargs: Any) -> str:
    """Compile a typed grader spec to a JSON string."""

    return json.dumps(compile_to_rft(grader), **json_kwargs)


__all__ = [
    # Re-exported here for convenience; canonical home is .spec.
    "MultiGrader",
    "ScoreModelGrader",
    "compile_to_rft",
    "compile_to_rft_json",
    "validate_calculate_output",
]
