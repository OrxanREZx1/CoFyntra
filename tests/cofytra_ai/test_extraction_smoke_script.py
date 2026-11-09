"""The extraction smoke script: no LLM call without --live, real service path with a fake provider, no secrets printed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cofytra_ai import service
from tests.fixtures.scripted_llm import ScriptedLLM

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cofytra_extraction_smoke.py"
SECRET = "sk-secret-extraction-smoke-key-456"


@pytest.fixture()
def smoke(monkeypatch):
    spec = importlib.util.spec_from_file_location("cofytra_extraction_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def forbidden(*a, **k):
        raise AssertionError("no provider client may be built without --live")
    monkeypatch.setattr(module, "_base_factory", forbidden)
    return module


@pytest.mark.parametrize("scenario", ["A", "B", "C"])
def test_scenarios_dry_run_without_any_call(smoke, capsys, scenario):
    assert smoke.main(["--scenario", scenario]) == 0
    out = capsys.readouterr().out
    assert "FICTIONAL TEST DATA" in out and "DRY RUN: no API call was made" in out and "<messages>" in out
    assert f"Scenario {scenario}:" in out and "Expected:" in out


def test_custom_message_dry_run(smoke, capsys):
    assert smoke.main(["--message", "We interviewed 3 dentists; all use WhatsApp.", "--with-existing"]) == 0
    out = capsys.readouterr().out
    assert "Custom founder message" in out and "We interviewed 3 dentists" in out and "kn-201" in out


def test_scenario_and_message_are_exclusive(smoke):
    with pytest.raises(SystemExit):
        smoke.main(["--scenario", "A", "--message", "x"])


def test_scenario_requests_are_valid_contract_requests(smoke):
    for name in "ABC":
        req, _, _ = smoke.build_request(name, None, False)
        assert req.messages and req.founder.id == "u-ana"
    assert smoke.build_request("C", None, False)[0].existing_knowledge[0].id == "kn-201"


def test_live_without_key_stops_before_calling(smoke, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="Nothing was sent"):
        smoke.main(["--scenario", "A", "--live"])


def test_live_path_uses_real_service_counts_and_hides_key(smoke, monkeypatch, capsys):
    doc_resp = smoke._doc_example("extract_knowledge.response")
    fabricated = {"category": "evidence", "statement": "All 8 managers would buy now.", "epistemic_status": "founder_observation",
                  "confidence": "high", "source_message_ids": ["msg-101"], "source_quote": "All 8 managers would buy now",
                  "external_reference": None, "duplicate_of_knowledge_id": None}
    repeat = dict(doc_resp["candidates"][1], confidence="low")
    scripted = ScriptedLLM({"ExtractionLLM": lambda m, r: {"candidates": [*doc_resp["candidates"], fabricated, repeat]}})
    seen = {}

    def fake_base(settings, **kwargs):
        seen["tracing"] = settings.langsmith_tracing
        return scripted.get_llm_client(settings, **kwargs)

    monkeypatch.setattr(smoke, "_base_factory", fake_base)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert smoke.main(["--scenario", "A", "--live", "--json"]) == 0
    out = capsys.readouterr().out
    assert seen["tracing"] is False and SECRET not in out
    assert "proposed by model (final attempt, measured by this script) = 5" in out
    assert "accepted = 3   dropped = 1   merged repeats (proposed - accepted - dropped) = 1" in out
    assert "unknown_source=0  quote_not_found=1  invalid_status=0  total=1" in out
    assert "assumption | founder_hypothesis" in out and "duplicate : none (would be shared as new)" in out
    assert "model=scripted-fast" in out and "attempts=1" in out


def test_live_provider_error_is_reported_not_raised(smoke, monkeypatch, capsys):
    class APITimeoutError(Exception):
        pass

    def boom(messages, recorder):
        raise APITimeoutError("Request timed out.")
    scripted = ScriptedLLM({"ExtractionLLM": boom})
    monkeypatch.setattr(smoke, "_base_factory", lambda s, **k: scripted.get_llm_client(s, **k))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    assert smoke.main(["--scenario", "B", "--live"]) == 2
    out = capsys.readouterr().out
    assert "code=provider_unavailable" in out and SECRET not in out


def test_service_function_is_the_real_one(smoke):
    assert smoke.service.extract_knowledge is service.extract_knowledge
