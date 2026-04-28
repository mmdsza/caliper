"""Shared fixtures for server tests."""

import pytest
from easytrain_server import create_app
from fastapi.testclient import TestClient

SAMPLE_GRADER_SOURCE = '''
from easytrain_sdk import grader, Rollout, GraderResult


@grader(name="legal_citation_grader", version="0.1.0")
def legal_citation_grader(rollout: Rollout) -> GraderResult:
    response = rollout.response
    if not response.startswith("<think>"):
        return GraderResult(score=0.0, explanation="missing <think> gate")
    if rollout.gold is not None and rollout.gold in response:
        return GraderResult(score=1.0, explanation="gold present in response")
    return GraderResult(score=0.0, explanation="gold absent")
'''


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def sample_grader_source() -> str:
    return SAMPLE_GRADER_SOURCE
