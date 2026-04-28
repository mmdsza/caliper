"""MultiGrader composition + the docs example."""

from __future__ import annotations

import json

import pytest
from easytrain_rft import (
    MultiGrader,
    PythonGrader,
    ScoreModelGrader,
    StringCheckGrader,
    compile_to_rft,
    compile_to_rft_json,
)

# The exact JSON example from docs/03-graders-and-rewards.md, "OpenAI's RFT
# JSON spec — the de facto interchange format".
DOCS_EXAMPLE_JSON = """
{
  "type": "multi",
  "graders": {
    "format": {"type": "string_check", "operation": "starts_with", "reference": "<think>"},
    "correctness": {"type": "score_model", "model": "gpt-4.1", "prompt": "..."},
    "code_runs": {"type": "python", "source": "..."}
  },
  "calculate_output": "0.3 * format + 0.5 * correctness + 0.2 * code_runs"
}
"""


def _build_docs_example() -> MultiGrader:
    return MultiGrader(
        graders={
            "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            "correctness": ScoreModelGrader(model="gpt-4.1", prompt="..."),
            "code_runs": PythonGrader(source="..."),
        },
        calculate_output="0.3 * format + 0.5 * correctness + 0.2 * code_runs",
    )


def test_multi_three_subgraders_compiles() -> None:
    g = _build_docs_example()
    out = compile_to_rft(g)
    assert out["type"] == "multi"
    assert set(out["graders"].keys()) == {"format", "correctness", "code_runs"}
    assert out["calculate_output"] == "0.3 * format + 0.5 * correctness + 0.2 * code_runs"


def test_docs_example_round_trips_structurally() -> None:
    expected = json.loads(DOCS_EXAMPLE_JSON)
    g = _build_docs_example()
    actual_json = compile_to_rft_json(g)
    actual = json.loads(actual_json)

    # ScoreModelGrader carries a default range; remove it for the structural
    # comparison since the docs example doesn't show one.
    actual["graders"]["correctness"].pop("range", None)

    assert actual == expected


def test_calculate_output_unknown_subgrader_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sub-grader"):
        MultiGrader(
            graders={
                "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            },
            calculate_output="0.5 * format + 0.5 * missing",
        )


def test_calculate_output_attribute_access_rejected() -> None:
    with pytest.raises(ValueError):
        MultiGrader(
            graders={
                "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            },
            calculate_output="os.system('rm -rf /')",
        )


def test_calculate_output_call_rejected() -> None:
    with pytest.raises(ValueError):
        MultiGrader(
            graders={
                "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            },
            calculate_output="__import__('os').system('echo')",
        )


def test_calculate_output_pow_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        MultiGrader(
            graders={
                "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            },
            calculate_output="format ** 2",
        )


def test_calculate_output_modulo_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported operator"):
        MultiGrader(
            graders={
                "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            },
            calculate_output="format % 2",
        )
