"""Shared fixtures for the monitor test suite."""

from __future__ import annotations


class MockLLMClient:
    """Deterministic LLM client mock for hermetic tests.

    Pass a ``str`` to always return the same canned response, or a ``dict``
    where the first key found anywhere in the prompt selects the response.
    Falls back to ``"FLAGGED: no\\nCONFIDENCE: 0.0"`` so unmatched prompts
    still parse cleanly.
    """

    def __init__(self, responses: dict[str, str] | str):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, model: str, max_tokens: int = 512) -> str:
        self.calls.append((prompt, model))
        if isinstance(self.responses, str):
            return self.responses
        for key, val in self.responses.items():
            if key in prompt:
                return val
        return "FLAGGED: no\nCONFIDENCE: 0.0"
