"""The manual smoke script must never call the LLM unless --live is passed, and must never print the key."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cofytra_ai import service

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cofytra_chat_smoke.py"


@pytest.fixture()
def smoke(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("service.chat must not be called without --live")
    monkeypatch.setattr(service, "chat", forbidden)
    spec = importlib.util.spec_from_file_location("cofytra_chat_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # import alone must have no side effects beyond disabling tracing
    return module


def test_dry_run_documented_scenario_makes_no_call(smoke, capsys):
    assert smoke.main([]) == 0
    out = capsys.readouterr().out
    assert "FICTIONAL SCENARIO" in out and "DRY RUN: no API call was made" in out
    assert "<current_message>" in out


def test_dry_run_custom_message_without_context(smoke, capsys):
    assert smoke.main(["--message", "How should we price this?", "--no-context"]) == 0
    out = capsys.readouterr().out
    assert "NO project context" in out and "No shared knowledge yet." in out and "How should we price this?" in out


def test_live_without_key_stops_before_calling(smoke, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Nothing was sent"):
        smoke.main(["--live"])


def test_live_path_disables_tracing_and_never_prints_key(smoke, monkeypatch, capsys):
    from cofytra_ai.models import ChatResponse, ResponseMeta

    seen = {}

    def fake_chat(request, *, settings=None, llm_factory=None):
        seen["tracing"] = settings.langsmith_tracing
        return ChatResponse(reply="ok", referenced_knowledge_ids=["kn-201"], meta=ResponseMeta(
            request_id=request.request_id, prompt_version="cofytra.chat.v1", model="m", tier="fast", latency_ms=5, attempts=1))

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-smoke-key-123")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(smoke.service, "chat", fake_chat)
    assert smoke.main(["--live", "--json"]) == 0
    out = capsys.readouterr().out
    assert seen["tracing"] is False
    assert "sk-secret-smoke-key-123" not in out
    assert "kn-201: [founder_hypothesis]" in out and "attempts=1" in out
