"""Tests for the built-in mutation catalog.

Each mutation gets:
    - happy path: target structure present, response changes as expected,
      input rollout is not mutated in place
    - no-op path: target structure absent, response is unchanged
    - protocol satisfaction via runtime_checkable Mutation
    - determinism for randomized-looking mutations (whitespace, case)
"""

from __future__ import annotations

from caliper_mutate.mutations import (
    AppendGarbage,
    CasePerturb,
    DropFormatGate,
    HardcodeGold,
    PrependSycophancy,
    TruncateResponse,
    WhitespacePerturb,
    WrongAnswer,
)
from caliper_mutate.types import Mutation, MutationExpectation
from caliper_sdk import Rollout

# --- DropFormatGate ----------------------------------------------------------


def test_drop_format_gate_strips_think_block(formatted_rollout: Rollout) -> None:
    original_response = formatted_rollout.response
    out = DropFormatGate().apply(formatted_rollout)
    assert "<think>" not in out.response
    assert "</think>" not in out.response
    assert out.response == "The answer is 42"
    # Input was not mutated in place.
    assert formatted_rollout.response == original_response
    # The rest of the rollout is preserved.
    assert out.prompt == formatted_rollout.prompt
    assert out.gold == formatted_rollout.gold


def test_drop_format_gate_strips_unclosed_leading_think() -> None:
    rollout = Rollout(
        prompt="p",
        response="<think>still thinking and never closed",
        gold="42",
    )
    out = DropFormatGate().apply(rollout)
    assert "<think>" not in out.response
    assert out.response == ""


def test_drop_format_gate_noop_on_bare(bare_rollout: Rollout) -> None:
    out = DropFormatGate().apply(bare_rollout)
    assert out.response == bare_rollout.response
    # Returned object is a copy, not the same identity.
    assert out is not bare_rollout


def test_drop_format_gate_satisfies_protocol() -> None:
    assert isinstance(DropFormatGate(), Mutation)
    assert DropFormatGate.expected_effect is MutationExpectation.LOWER


# --- WrongAnswer -------------------------------------------------------------


def test_wrong_answer_corrupts_long_gold() -> None:
    rollout = Rollout(
        prompt="capital of France?",
        response="The capital is Paris.",
        gold="Paris",
    )
    out = WrongAnswer().apply(rollout)
    assert out.response != rollout.response
    assert "Paris" not in out.response
    assert "PXXXX" in out.response


def test_wrong_answer_appends_x_for_short_gold(formatted_rollout: Rollout) -> None:
    # gold="42" is shorter than 4 chars; expect "42XXXX" inserted in place of "42".
    out = WrongAnswer().apply(formatted_rollout)
    assert "42XXXX" in out.response
    # Input untouched.
    assert formatted_rollout.response == "<think>Step 1.</think>The answer is 42"


def test_wrong_answer_noop_when_gold_absent() -> None:
    rollout = Rollout(
        prompt="p",
        response="An unrelated answer.",
        gold="Paris",
    )
    out = WrongAnswer().apply(rollout)
    assert out.response == rollout.response


def test_wrong_answer_noop_when_gold_none() -> None:
    rollout = Rollout(prompt="p", response="hello", gold=None)
    out = WrongAnswer().apply(rollout)
    assert out.response == rollout.response


def test_wrong_answer_satisfies_protocol() -> None:
    assert isinstance(WrongAnswer(), Mutation)
    assert WrongAnswer.expected_effect is MutationExpectation.LOWER


# --- TruncateResponse --------------------------------------------------------


def test_truncate_response_drops_trailing_quarter() -> None:
    rollout = Rollout(prompt="p", response="abcdefgh", gold=None)  # 8 chars, drop 2
    out = TruncateResponse().apply(rollout)
    assert out.response == "abcdef"
    # Input untouched.
    assert rollout.response == "abcdefgh"


def test_truncate_response_noop_when_too_short() -> None:
    rollout = Rollout(prompt="p", response="abc", gold=None)
    out = TruncateResponse().apply(rollout)
    assert out.response == "abc"


def test_truncate_response_satisfies_protocol() -> None:
    assert isinstance(TruncateResponse(), Mutation)
    assert TruncateResponse.expected_effect is MutationExpectation.LOWER


# --- WhitespacePerturb -------------------------------------------------------


def test_whitespace_perturb_changes_spacing(formatted_rollout: Rollout) -> None:
    out = WhitespacePerturb().apply(formatted_rollout)
    assert out.response != formatted_rollout.response
    # Original tokens are preserved across whitespace.
    assert out.response.split() == formatted_rollout.response.split()
    # Input untouched.
    assert formatted_rollout.response == "<think>Step 1.</think>The answer is 42"


def test_whitespace_perturb_noop_on_no_whitespace(bare_rollout: Rollout) -> None:
    out = WhitespacePerturb().apply(bare_rollout)
    assert out.response == bare_rollout.response


def test_whitespace_perturb_is_deterministic(formatted_rollout: Rollout) -> None:
    a = WhitespacePerturb().apply(formatted_rollout)
    b = WhitespacePerturb().apply(formatted_rollout)
    assert a.response == b.response


def test_whitespace_perturb_satisfies_protocol() -> None:
    assert isinstance(WhitespacePerturb(), Mutation)
    assert WhitespacePerturb.expected_effect is MutationExpectation.EQUAL


# --- PrependSycophancy -------------------------------------------------------


def test_prepend_sycophancy_prepends(bare_rollout: Rollout) -> None:
    out = PrependSycophancy().apply(bare_rollout)
    assert out.response.startswith("You're absolutely right! ")
    assert out.response.endswith(bare_rollout.response)
    # Input untouched.
    assert bare_rollout.response == "42"


def test_prepend_sycophancy_applies_to_empty_response() -> None:
    rollout = Rollout(prompt="p", response="", gold=None)
    out = PrependSycophancy().apply(rollout)
    # Always applies — response should equal just the prefix.
    assert out.response == "You're absolutely right! "


def test_prepend_sycophancy_satisfies_protocol() -> None:
    assert isinstance(PrependSycophancy(), Mutation)
    assert PrependSycophancy.expected_effect is MutationExpectation.EQUAL


# --- AppendGarbage -----------------------------------------------------------


def test_append_garbage_appends(bare_rollout: Rollout) -> None:
    out = AppendGarbage().apply(bare_rollout)
    assert out.response.endswith(" [DEBUG: tokens=42]")
    assert out.response.startswith(bare_rollout.response)
    # Input untouched.
    assert bare_rollout.response == "42"


def test_append_garbage_applies_to_empty_response() -> None:
    rollout = Rollout(prompt="p", response="", gold=None)
    out = AppendGarbage().apply(rollout)
    # Always applies.
    assert out.response == " [DEBUG: tokens=42]"


def test_append_garbage_satisfies_protocol() -> None:
    assert isinstance(AppendGarbage(), Mutation)
    assert AppendGarbage.expected_effect is MutationExpectation.EQUAL


# --- CasePerturb -------------------------------------------------------------


def test_case_perturb_changes_some_case() -> None:
    rollout = Rollout(
        prompt="p",
        # Plenty of letters so probabilistic toggling almost certainly fires.
        response="The quick brown fox jumps over the lazy dog. " * 3,
        gold=None,
    )
    out = CasePerturb().apply(rollout)
    assert out.response != rollout.response
    # Lowercased forms must match (case-insensitive equality).
    assert out.response.lower() == rollout.response.lower()


def test_case_perturb_noop_on_empty() -> None:
    rollout = Rollout(prompt="p", response="", gold=None)
    out = CasePerturb().apply(rollout)
    assert out.response == ""


def test_case_perturb_noop_on_no_letters() -> None:
    # No alphabetic characters means nothing to flip.
    rollout = Rollout(prompt="p", response="42 + 7 = 49", gold=None)
    out = CasePerturb().apply(rollout)
    assert out.response == rollout.response


def test_case_perturb_is_deterministic() -> None:
    rollout = Rollout(prompt="p", response="The quick brown fox.", gold=None)
    a = CasePerturb().apply(rollout)
    b = CasePerturb().apply(rollout)
    assert a.response == b.response


def test_case_perturb_satisfies_protocol() -> None:
    assert isinstance(CasePerturb(), Mutation)
    assert CasePerturb.expected_effect is MutationExpectation.EQUAL


# --- HardcodeGold ------------------------------------------------------------


def test_hardcode_gold_replaces_response(formatted_rollout: Rollout) -> None:
    out = HardcodeGold().apply(formatted_rollout)
    assert out.response == formatted_rollout.gold == "42"
    # No reasoning trace remains.
    assert "<think>" not in out.response
    # Input untouched.
    assert formatted_rollout.response == "<think>Step 1.</think>The answer is 42"


def test_hardcode_gold_noop_when_gold_none() -> None:
    rollout = Rollout(prompt="p", response="anything", gold=None)
    out = HardcodeGold().apply(rollout)
    assert out.response == rollout.response


def test_hardcode_gold_satisfies_protocol() -> None:
    assert isinstance(HardcodeGold(), Mutation)
    assert HardcodeGold.expected_effect is MutationExpectation.LOWER
