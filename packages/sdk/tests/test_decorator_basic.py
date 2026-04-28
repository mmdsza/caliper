"""Tests for the @grader decorator."""

from __future__ import annotations

import pytest
from easytrain_sdk import GraderResult, Rollout, grader


def _rollout() -> Rollout:
    return Rollout(prompt="q", response="a", gold="a")


def test_float_return_is_wrapped_into_grader_result() -> None:
    @grader
    def exact_match(rollout: Rollout) -> float:
        return 1.0 if rollout.response == rollout.gold else 0.0

    result = exact_match(_rollout())
    assert isinstance(result, GraderResult)
    assert result.score == 1.0
    assert result.explanation is None
    assert result.components == {}


def test_grader_result_return_passes_through() -> None:
    @grader
    def with_components(rollout: Rollout) -> GraderResult:
        return GraderResult(
            score=0.5,
            explanation="halfway there",
            components={"format": 1.0, "content": 0.0},
        )

    result = with_components(_rollout())
    assert isinstance(result, GraderResult)
    assert result.score == 0.5
    assert result.explanation == "halfway there"
    assert result.components == {"format": 1.0, "content": 0.0}


def test_default_name_and_version() -> None:
    @grader
    def my_grader(rollout: Rollout) -> float:
        return 0.0

    assert my_grader.name == "my_grader"
    assert my_grader.version == "0.1.0"
    assert my_grader.__wrapped__.__name__ == "my_grader"


def test_parameterized_form_sets_name_and_version() -> None:
    @grader(name="my_grader", version="2.0.0")
    def f(rollout: Rollout) -> float:
        return 0.25

    assert f.name == "my_grader"
    assert f.version == "2.0.0"
    assert f.__wrapped__ is not f
    assert f(_rollout()).score == 0.25


def test_parameterized_form_only_name() -> None:
    @grader(name="custom")
    def f(rollout: Rollout) -> float:
        return 1.0

    assert f.name == "custom"
    assert f.version == "0.1.0"


def test_parameterized_form_only_version() -> None:
    @grader(version="9.9.9")
    def f(rollout: Rollout) -> float:
        return 1.0

    assert f.name == "f"
    assert f.version == "9.9.9"


@pytest.mark.parametrize("bad_score", [-0.1, 1.1, 2.0, -1.0, float("inf")])
def test_score_outside_unit_interval_raises(bad_score: float) -> None:
    @grader
    def naughty(rollout: Rollout) -> float:
        return bad_score

    with pytest.raises(ValueError) as excinfo:
        naughty(_rollout())
    # Offending value should appear in the message.
    assert str(bad_score) in str(excinfo.value) or repr(bad_score) in str(excinfo.value)


def test_grader_result_with_invalid_score_raises_at_construction() -> None:
    # Pydantic itself rejects scores outside [0, 1] inside GraderResult, so
    # constructing the bad result happens before the decorator's check —
    # confirm the error is still raised one way or another.
    @grader
    def via_grader_result(rollout: Rollout) -> GraderResult:
        return GraderResult(score=1.5)  # type: ignore[arg-type]

    with pytest.raises((ValueError, Exception)):
        via_grader_result(_rollout())


def test_dict_input_is_coerced_to_rollout() -> None:
    @grader
    def echo_len(rollout: Rollout) -> float:
        assert isinstance(rollout, Rollout)
        return min(len(rollout.response) / 10.0, 1.0)

    result = echo_len({"prompt": "q", "response": "abc", "gold": "abc"})
    assert isinstance(result, GraderResult)
    assert result.score == pytest.approx(0.3)


def test_invalid_input_type_raises_type_error() -> None:
    @grader
    def f(rollout: Rollout) -> float:
        return 0.0

    with pytest.raises(TypeError):
        f(12345)  # type: ignore[arg-type]


def test_decorator_preserves_wrapped_attribute() -> None:
    def original(rollout: Rollout) -> float:
        return 0.7

    wrapped = grader(original)
    assert wrapped.__wrapped__ is original


def test_non_numeric_return_raises_type_error() -> None:
    @grader
    def bad_return(rollout: Rollout) -> float:
        return "nope"  # type: ignore[return-value]

    with pytest.raises(TypeError):
        bad_return(_rollout())


def test_bool_return_is_rejected() -> None:
    """``True``/``False`` are technically ``int``, but treating them as scores
    is almost always a bug — make it loud."""

    @grader
    def boolish(rollout: Rollout) -> float:
        return True  # type: ignore[return-value]

    with pytest.raises(TypeError):
        boolish(_rollout())
