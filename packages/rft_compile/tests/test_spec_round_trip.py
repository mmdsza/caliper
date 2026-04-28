"""Each native grader type round-trips through ``compile_to_rft``."""

from __future__ import annotations

from caliper_rft import (
    LabelModelGrader,
    MultiGrader,
    PythonGrader,
    ScoreModelGrader,
    StringCheckGrader,
    TextSimilarityGrader,
    compile_to_rft,
)


def test_string_check_round_trip() -> None:
    g = StringCheckGrader(operation="starts_with", reference="<think>")
    out = compile_to_rft(g)
    assert out["type"] == "string_check"
    assert out["operation"] == "starts_with"
    assert out["reference"] == "<think>"


def test_text_similarity_round_trip() -> None:
    g = TextSimilarityGrader(evaluation_metric="bleu", reference="hello world")
    out = compile_to_rft(g)
    assert out["type"] == "text_similarity"
    assert out["evaluation_metric"] == "bleu"
    assert out["reference"] == "hello world"
    # pass_threshold is None → excluded.
    assert "pass_threshold" not in out


def test_text_similarity_with_threshold() -> None:
    g = TextSimilarityGrader(
        evaluation_metric="cosine", reference="x", pass_threshold=0.8
    )
    out = compile_to_rft(g)
    assert out["pass_threshold"] == 0.8


def test_score_model_round_trip() -> None:
    g = ScoreModelGrader(model="gpt-4.1", prompt="Rate 0-1.")
    out = compile_to_rft(g)
    assert out["type"] == "score_model"
    assert out["model"] == "gpt-4.1"
    assert out["prompt"] == "Rate 0-1."
    # Tuple → list per JSON conventions.
    assert out["range"] == [0.0, 1.0]


def test_score_model_custom_range() -> None:
    g = ScoreModelGrader(model="gpt-4.1", prompt="Rate.", range=(-1.0, 1.0))
    out = compile_to_rft(g)
    assert out["range"] == [-1.0, 1.0]


def test_label_model_round_trip() -> None:
    g = LabelModelGrader(
        model="gpt-4.1",
        prompt="Choose a label.",
        labels=["good", "bad"],
        passing_labels=["good"],
    )
    out = compile_to_rft(g)
    assert out["type"] == "label_model"
    assert out["labels"] == ["good", "bad"]
    assert out["passing_labels"] == ["good"]


def test_python_round_trip() -> None:
    src = "def grade(sample, item):\n    return 1.0\n"
    g = PythonGrader(source=src)
    out = compile_to_rft(g)
    assert out["type"] == "python"
    assert out["source"] == src


def test_multi_round_trip_preserves_type_keys() -> None:
    g = MultiGrader(
        graders={
            "format": StringCheckGrader(operation="starts_with", reference="<think>"),
            "correctness": ScoreModelGrader(model="gpt-4.1", prompt="Score it."),
        },
        calculate_output="0.5 * format + 0.5 * correctness",
    )
    out = compile_to_rft(g)
    assert out["type"] == "multi"
    assert out["graders"]["format"]["type"] == "string_check"
    assert out["graders"]["correctness"]["type"] == "score_model"
    assert out["calculate_output"] == "0.5 * format + 0.5 * correctness"
