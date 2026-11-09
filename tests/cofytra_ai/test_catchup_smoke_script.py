"""Controlled F4 smoke script: real service, fictional inputs, offline providers only."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from cofytra_ai import llm, service
from cofytra_ai.models import contract_violations
from startary.llm import provider
from tests.fixtures.scripted_llm import ScriptedLLM

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cofytra_catchup_smoke.py"
SECRET = "sk-secret-catchup-smoke-key-789"
PRIVATE = "Private conversation must never appear in an error."


def forbidden(*args, **kwargs):
    raise AssertionError("No provider invocation is permitted here")


@pytest.fixture()
def smoke(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    spec = importlib.util.spec_from_file_location("cofytra_catchup_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_base_factory", forbidden)
    monkeypatch.setattr(llm, "get_llm_client", forbidden)
    monkeypatch.setattr(provider, "get_llm_client", forbidden)
    return module


def test_import_disables_tracing(smoke):
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert smoke.service.catch_up is service.catch_up


@pytest.mark.parametrize("args", [[], ["--prompt-key", "--show-system"]])
def test_dry_run_makes_no_service_or_provider_call(smoke, monkeypatch, capsys, args):
    service_spy = Mock(side_effect=forbidden)
    key_spy = Mock(side_effect=forbidden)
    monkeypatch.setattr(service, "catch_up", service_spy)
    monkeypatch.setattr(smoke.getpass, "getpass", key_spy)
    assert smoke.main(args) == 0
    out = capsys.readouterr().out
    service_spy.assert_not_called()
    key_spy.assert_not_called()
    assert "DRY RUN: no API call was made" in out
    assert "FICTIONAL TEST DATA" in out and "invented test data" in out
    assert "<shared_knowledge>" in out and "kn-202" in out
    assert "12,000" in out and "NOT real" in out
    assert "prompt_version=cofytra.catchup.v1" in out and "attempts=0" in out
    for k in smoke.build_request("documented").knowledge:
        if k.source_quote:
            assert k.source_quote not in out


def test_documented_request_is_exact_contract_example(smoke):
    request = smoke.build_request("documented")
    assert request.model_dump(mode="json", exclude_none=True) == type(request).model_validate(
        smoke._doc_example("catch_up.request")
    ).model_dump(mode="json", exclude_none=True)
    assert set(type(request).model_fields).isdisjoint({"messages", "recent_messages"})


def test_missing_key_stops_before_service_or_provider(smoke, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spy = Mock(side_effect=forbidden)
    monkeypatch.setattr(service, "catch_up", spy)
    with pytest.raises(SystemExit, match="Nothing was sent"):
        smoke.main(["--live"])
    spy.assert_not_called()
    assert "LIVE: OpenAI" not in capsys.readouterr().out


@pytest.mark.parametrize("scenario", ["empty", "proposals-only"])
@pytest.mark.parametrize("flags", [[], ["--live", "--prompt-key"]])
def test_deterministic_scenarios_use_real_service_without_llm(smoke, monkeypatch, capsys, scenario, flags):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings_spy = Mock(side_effect=forbidden)
    monkeypatch.setattr(smoke, "load_live_settings", settings_spy)
    monkeypatch.setattr(smoke.getpass, "getpass", forbidden)
    monkeypatch.setattr(llm, "call_structured", forbidden)
    real_service = service.catch_up
    responses = []

    def spy(request, **kwargs):
        response = real_service(request, **kwargs)
        assert contract_violations(request, response) == []
        responses.append(response)
        return response

    monkeypatch.setattr(service, "catch_up", spy)
    assert smoke.main(["--scenario", scenario, "--json", *flags]) == 0
    out = capsys.readouterr().out
    settings_spy.assert_not_called()
    assert len(responses) == 1 and responses[0].meta.attempts == 0
    assert "NO LLM CALL" in out and "model=None" in out and "tier=none" in out
    assert "latency_ms=0" in out and "attempts=0" in out
    if scenario == "proposals-only":
        assert "prop-401" in out and "awaits team approval" in out
        assert not responses[0].discoveries and len(responses[0].pending_decisions) == 1
    else:
        assert not responses[0].pending_decisions


@pytest.mark.parametrize("prompt_key", [False, True])
def test_fake_provider_displays_valid_references_and_never_keys(smoke, monkeypatch, capsys, prompt_key):
    output = smoke._doc_example("catch_up.response")
    output.pop("meta")
    # The real service must reject this invented reference before the script prints its response.
    output["discoveries"].append({"text": "Ben believes a fabricated claim.", "knowledge_ids": ["kn-999"]})
    scripted = ScriptedLLM({"CatchupLLM": lambda messages, recorder: output})
    seen = {}

    def factory(settings, **kwargs):
        seen["tracing"] = settings.langsmith_tracing
        seen["key"] = settings.openai_api_key
        return scripted.get_llm_client(settings, **kwargs)

    monkeypatch.setattr(smoke, "_base_factory", factory)
    if prompt_key:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(smoke.getpass, "getpass", lambda prompt: SECRET)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    assert smoke.main(["--live", "--json", *(["--prompt-key"] if prompt_key else [])]) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert seen == {"tracing": False, "key": SECRET}
    assert len(scripted.calls) == 1 and scripted.calls[0].schema == "CatchupLLM"
    assert SECRET not in out + captured.err
    if prompt_key:
        assert "OPENAI_API_KEY" not in os.environ
    for heading in ("SUMMARY", "DISCOVERIES", "ASSUMPTIONS", "EVIDENCE", "PENDING DECISIONS", "REFERENCED SHARED KNOWLEDGE", "METADATA"):
        assert heading in out
    for kid in ("kn-201", "kn-202", "kn-203", "kn-204"):
        assert f"- {kid}: [" in out
    assert "kn-999" not in out
    assert "founder_hypothesis" in out and "founder_observation" in out and "external_source" in out
    assert "prop-401" in out and "awaits team approval" in out
    assert "model=scripted-fast" in out and "attempts=1" in out and "latency_ms=" in out
    assert "prompt_version=cofytra.catchup.v1" in out
    for k in smoke.build_request("documented").knowledge:
        if k.source_quote:
            assert k.source_quote not in out


def test_credential_echo_is_redacted_in_response_and_json(smoke, monkeypatch, capsys):
    output = smoke._doc_example("catch_up.response")
    output.pop("meta")
    output["summary"] = f"Ben reports a test finding. {SECRET}"
    scripted = ScriptedLLM({"CatchupLLM": lambda messages, recorder: output})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(smoke.getpass, "getpass", lambda prompt: SECRET)
    monkeypatch.setattr(smoke, "_base_factory", scripted.get_llm_client)
    assert smoke.main(["--live", "--prompt-key", "--json"]) == 0
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert "[REDACTED]" in captured.out


def test_provider_error_does_not_print_secret_or_private_text(smoke, monkeypatch, capsys, caplog):
    class APITimeoutError(Exception):
        pass

    def fail(messages, recorder):
        raise APITimeoutError(f"{SECRET} {PRIVATE}")

    scripted = ScriptedLLM({"CatchupLLM": fail})
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(smoke, "_base_factory", scripted.get_llm_client)
    assert smoke.main(["--live"]) == 2
    captured = capsys.readouterr()
    assert "code=provider_unavailable" in captured.out
    assert len(scripted.calls) == 2
    assert SECRET not in captured.out + captured.err + caplog.text
    assert PRIVATE not in captured.out + captured.err + caplog.text


def test_configuration_error_is_suppressed(smoke, monkeypatch, capsys):
    def fail(prompt_key):
        raise ValueError(f"{SECRET} {PRIVATE}")

    monkeypatch.setattr(smoke, "load_live_settings", fail)
    assert smoke.main(["--live"]) == 2
    captured = capsys.readouterr()
    assert "Details suppressed" in captured.out
    assert SECRET not in captured.out + captured.err and PRIVATE not in captured.out + captured.err


def test_hidden_prompt_refuses_visible_fallback(smoke, monkeypatch, capsys):
    def unsupported(prompt):
        import warnings
        warnings.warn("Cannot hide input", smoke.getpass.GetPassWarning)
        raise AssertionError("Visible input must never be reached")

    monkeypatch.setattr(smoke.getpass, "getpass", unsupported)
    assert smoke.main(["--live", "--prompt-key"]) == 2
    assert "Details suppressed" in capsys.readouterr().out
