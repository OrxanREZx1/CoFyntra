"""Controlled F5a smoke script: real service, fictional inputs, offline providers only."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from cofytra_ai import llm, service
from cofytra_ai.models import ConflictResponse, contract_violations
from startary.llm import provider
from tests.fixtures.scripted_llm import ScriptedLLM

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "cofytra_conflicts_smoke.py"
SECRET = "sk-secret-conflicts-smoke-key-789"
PRIVATE = "Private conversation must never appear in an error."


def forbidden(*args, **kwargs):
    raise AssertionError("No provider invocation is permitted here")


@pytest.fixture()
def smoke(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    spec = importlib.util.spec_from_file_location("cofytra_conflicts_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_base_factory", forbidden)
    monkeypatch.setattr(llm, "get_llm_client", forbidden)
    monkeypatch.setattr(provider, "get_llm_client", forbidden)
    return module


def _output_for(module, scenario: str) -> dict:
    if scenario == "documented":
        return {"conflicts": module._doc_example("detect_conflicts.response")["conflicts"]}
    return {"conflicts": []}


def _conflict(left, right, *, kind: str = "contradiction", explanation: str = "SEMANTIC_MISS these claims were flagged.") -> dict:
    return {
        "claim_a": {"knowledge_id": left.id, "author_id": left.author_id, "statement": left.statement},
        "claim_b": {"knowledge_id": right.id, "author_id": right.author_id, "statement": right.statement},
        "kind": kind,
        "explanation": explanation,
        "question_to_resolve": "What would show this is a real disagreement?",
    }


def _run_live(smoke, monkeypatch, output: dict, argv: list[str]):
    scripted = ScriptedLLM({"ConflictsLLM": lambda messages, recorder: output})
    seen = {}

    def factory(settings, **kwargs):
        seen["tracing"] = settings.langsmith_tracing
        seen["key"] = settings.openai_api_key
        seen["tier_kwargs"] = kwargs.get("tier")
        return scripted.get_llm_client(settings, **kwargs)

    monkeypatch.setattr(smoke, "_base_factory", factory)
    real = service.detect_conflicts
    captured = []

    def spy(request, **kwargs):
        assert kwargs["llm_factory"] is factory
        response = real(request, **kwargs)
        assert contract_violations(request, response) == []
        captured.append((request, response))
        return response

    monkeypatch.setattr(service, "detect_conflicts", spy)
    code = smoke.main(argv)
    return code, scripted, seen, captured


def test_import_disables_tracing(smoke):
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert smoke.service.detect_conflicts is service.detect_conflicts


def test_documented_rubric_is_the_contract_example(smoke):
    smoke.ensure_documented_rubric()
    assert smoke.DOCUMENTED_PAIRS == (
        (frozenset({"kn-201", "kn-205"}), "contradiction"),
        (frozenset({"kn-203", "kn-206"}), "tension"),
    )


def test_documented_request_is_exact_contract_example(smoke):
    request = smoke.build_request("documented")
    assert request.model_dump(mode="json") == type(request).model_validate(
        smoke._doc_example("detect_conflicts.request")
    ).model_dump(mode="json")
    assert set(type(request).model_fields).isdisjoint({"messages", "recent_messages", "private_messages"})


@pytest.mark.parametrize("scenario", ["compatible", "different-observations"])
def test_non_conflict_scenarios_are_two_founders_and_fictional(smoke, scenario):
    request = smoke.build_request(scenario)
    assert len({item.author_id for item in request.knowledge}) == 2
    statements = [item.statement for item in request.knowledge]
    if scenario == "compatible":
        assert statements == [smoke.COMPATIBLE_ADA, smoke.COMPATIBLE_BO]
        assert all(item.epistemic_status.value == "founder_hypothesis" for item in request.knowledge)
    else:
        assert statements == [smoke.LISBON_OBSERVATION, smoke.PORTO_OBSERVATION]
        assert all(item.epistemic_status.value == "founder_observation" for item in request.knowledge)
        assert "Lisbon" in statements[0] and "Porto" in statements[1]
        assert "March 2026" in statements[0] and "June 2026" in statements[1]


@pytest.mark.parametrize("scenario", ["documented", "compatible", "different-observations"])
@pytest.mark.parametrize("args", [[], ["--prompt-key", "--show-system", "--json"]])
def test_dry_run_makes_no_service_or_provider_call(smoke, monkeypatch, capsys, scenario, args):
    service_spy = Mock(side_effect=forbidden)
    key_spy = Mock(side_effect=forbidden)
    monkeypatch.setattr(service, "detect_conflicts", service_spy)
    monkeypatch.setattr(smoke.getpass, "getpass", key_spy)
    assert smoke.main(["--scenario", scenario, *args]) == 0
    out = capsys.readouterr().out
    service_spy.assert_not_called()
    key_spy.assert_not_called()
    assert "DRY RUN: no API call was made" in out
    assert "FICTIONAL TEST DATA" in out and "invented" in out
    assert "<shared_knowledge>" in out
    assert "RESULT: NOT SCORED (dry run)" in out
    assert "RESULT: PASS" not in out and "RESULT: FAIL" not in out
    assert "prompt_version=cofytra.conflicts.v2" in out and "attempts=0" in out
    assert "source_quote" not in out
    for item in smoke.build_request(scenario).knowledge:
        assert item.statement in out
        if item.source_quote:
            assert item.source_quote not in out


def test_missing_key_stops_before_service_or_provider(smoke, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    spy = Mock(side_effect=forbidden)
    monkeypatch.setattr(service, "detect_conflicts", spy)
    with pytest.raises(SystemExit, match="Nothing was sent"):
        smoke.main(["--live", "--scenario", "compatible"])
    spy.assert_not_called()
    assert "LIVE: OpenAI" not in capsys.readouterr().out


@pytest.mark.parametrize("scenario", ["documented", "compatible", "different-observations"])
@pytest.mark.parametrize("prompt_key", [False, True])
def test_expected_model_output_passes_without_rewriting_it(smoke, monkeypatch, capsys, scenario, prompt_key):
    if prompt_key:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(smoke.getpass, "getpass", lambda prompt: SECRET)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    code, scripted, seen, captured = _run_live(
        smoke, monkeypatch, _output_for(smoke, scenario),
        ["--live", "--json", "--scenario", scenario, *(["--prompt-key"] if prompt_key else [])],
    )
    captured_io = capsys.readouterr()
    out = captured_io.out
    assert code == 0
    assert seen["tracing"] is False and seen["key"] == SECRET and seen["tier_kwargs"] == "fast"
    assert len(scripted.calls) == 1 and scripted.calls[0].schema == "ConflictsLLM" and scripted.calls[0].tier == "fast"
    assert len(captured) == 1
    request, response = captured[0]
    assert isinstance(response, ConflictResponse)
    assert response.meta.prompt_version == "cofytra.conflicts.v2"
    assert response.meta.attempts == 1 and response.meta.tier == "fast"
    assert "Detected conflict count:" in out
    assert "RESULT: PASS" in out and "RESULT: FAIL" not in out
    assert "model=scripted-fast" in out and "attempts=1" in out and "latency_ms=" in out
    assert "prompt_version=cofytra.conflicts.v2" in out
    assert "FULL RESPONSE JSON" in out
    assert SECRET not in out + captured_io.err
    if prompt_key:
        assert "OPENAI_API_KEY" not in os.environ
    if scenario == "documented":
        assert "Detected conflict count: 2" in out
        assert "kn-201 + kn-205 kind=contradiction" in out
        assert "kn-203 + kn-206 kind=tension" in out
        assert "u-ben" in out and "u-ana" in out
        assert "explanation:" in out and "question_to_resolve:" in out
    else:
        assert "Detected conflict count: 0" in out
        assert "(none)" in out


def test_swapped_claim_order_still_matches_pair_identity(smoke, monkeypatch, capsys):
    documented = smoke._doc_example("detect_conflicts.response")["conflicts"]
    swapped = []
    for conflict in documented:
        swapped.append({**conflict, "claim_a": conflict["claim_b"], "claim_b": conflict["claim_a"]})
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    code, _scripted, _seen, captured = _run_live(
        smoke, monkeypatch, {"conflicts": swapped}, ["--live", "--scenario", "documented"],
    )
    out = capsys.readouterr().out
    assert code == 0 and "RESULT: PASS" in out
    assert captured[0][1].conflicts[0].claim_a.knowledge_id == "kn-205"
    assert "kn-205 + kn-201 kind=contradiction" in out


@pytest.mark.parametrize("scenario, mutate", [
    ("documented", "empty"),
    ("documented", "wrong-kind"),
    ("compatible", "false-conflict"),
    ("different-observations", "false-conflict"),
])
def test_semantic_error_fails_and_the_model_output_is_printed_unchanged(smoke, monkeypatch, capsys, scenario, mutate):
    request = smoke.build_request(scenario)
    if mutate == "empty":
        output = {"conflicts": []}
    elif mutate == "wrong-kind":
        output = {"conflicts": smoke._doc_example("detect_conflicts.response")["conflicts"]}
        output["conflicts"][1]["kind"] = "contradiction"
    else:
        output = {"conflicts": [_conflict(request.knowledge[0], request.knowledge[1])]}
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    code, _scripted, _seen, captured = _run_live(
        smoke, monkeypatch, output, ["--live", "--scenario", scenario, "--json"],
    )
    out = capsys.readouterr().out
    response = captured[0][1]
    assert code == 1
    assert "RESULT: FAIL" in out and "RESULT: PASS" not in out
    assert f"Detected conflict count: {len(response.conflicts)}" in out
    if mutate == "empty":
        assert "(none)" in out
        assert "missing pair kn-201 + kn-205 kind=contradiction" in out
        assert "missing pair kn-203 + kn-206 kind=tension" in out
    elif mutate == "wrong-kind":
        assert "missing pair kn-203 + kn-206 kind=tension" in out
        assert "unexpected pair kn-203 + kn-206 kind=contradiction" in out
        assert response.conflicts[1].kind.value == "contradiction"
        assert response.conflicts[1].explanation in out
    else:
        assert "SEMANTIC_MISS these claims were flagged." in out
        assert "unexpected pair" in out
        assert response.conflicts[0].explanation == "SEMANTIC_MISS these claims were flagged."


def test_credential_echo_is_redacted_in_response_and_json(smoke, monkeypatch, capsys):
    documented = {"conflicts": smoke._doc_example("detect_conflicts.response")["conflicts"]}
    documented["conflicts"][0]["explanation"] = f"Ben and Ana disagree. {SECRET}"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(smoke.getpass, "getpass", lambda prompt: SECRET)
    code, _scripted, _seen, _captured = _run_live(
        smoke, monkeypatch, documented, ["--live", "--prompt-key", "--json", "--scenario", "documented"],
    )
    captured_io = capsys.readouterr()
    assert code == 0
    assert SECRET not in captured_io.out + captured_io.err
    assert "[REDACTED]" in captured_io.out


def test_provider_error_does_not_print_secret_or_private_text(smoke, monkeypatch, capsys, caplog):
    class APITimeoutError(Exception):
        pass

    def fail(messages, recorder):
        raise APITimeoutError(f"{SECRET} {PRIVATE}")

    scripted = ScriptedLLM({"ConflictsLLM": fail})
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(smoke, "_base_factory", scripted.get_llm_client)
    assert smoke.main(["--live", "--scenario", "compatible"]) == 2
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
