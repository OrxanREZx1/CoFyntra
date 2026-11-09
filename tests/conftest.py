"""
Test-suite safety net (Milestone 6.2.3): offline tests must never reach a paid LLM API.

Settings reads OPENAI_API_KEY from the developer's .env, so a test that forgets to mock an LLM node
used to make a real, billed call. Unless a test is marked ``live`` AND STARTARY_RUN_LIVE_TESTS=1 is
set, every test runs with a dummy key and an unroutable OpenAI base URL: an unmocked call now fails
fast with a connection error instead of spending money. LangSmith tracing is disabled as well.
"""

import os

import pytest

_LIVE_TESTS_ENABLED = bool(os.environ.get("STARTARY_RUN_LIVE_TESTS"))


@pytest.fixture(autouse=True)
def _block_paid_llm_calls(request, monkeypatch):
    if _LIVE_TESTS_ENABLED and request.node.get_closest_marker("live"):
        return
    monkeypatch.setenv("OPENAI_API_KEY", "sk-offline-test-dummy")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    # M6.2.6: a developer .env enabling the revision A/B flag must not change the default-mode suite;
    # V2 tests pass Settings(experiment_revision_v2=True) explicitly.
    monkeypatch.setenv("STARTARY_EXPERIMENT_REVISION_V2", "false")
    # M6.2.7: same for the parallel-understanding flag; parallel tests build the graph explicitly.
    monkeypatch.setenv("STARTARY_PARALLEL_UNDERSTANDING", "false")
