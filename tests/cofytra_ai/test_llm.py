"""
Offline tests for cofytra_ai.llm (no network, no paid calls).

LLM calls are answered by tests.fixtures.scripted_llm.ScriptedLLM; tests/conftest.py additionally
forces a dummy OPENAI_API_KEY and an unroutable base URL, so an unmocked call cannot reach OpenAI.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

import cofytra_ai.llm as llm_mod
from cofytra_ai.errors import CofytraAIError
from cofytra_ai.llm import MAX_ATTEMPTS, build_run_config, call_structured, load_settings
from cofytra_ai.models import ErrorCode, ResponseMeta
from startary.config import Settings
from startary.observability.langsmith import sanitize_metadata
from tests.fixtures.scripted_llm import ScriptedLLM

REPO_ROOT = Path(__file__).resolve().parents[2]
SECRET_TEXT = "SECRET-founder-text-7f3a: we will pivot to dentists"
SECRET_KEY = "sk-offline-test-dummy"  # set by tests/conftest.py


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    ids: list[str] = Field(default_factory=list)


# Fake provider exceptions, classified by type name / status_code like the real openai ones.
class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class APITimeoutError(Exception):
    pass


class LengthFinishReasonError(Exception):
    pass


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _scripted(*steps) -> ScriptedLLM:
    """Each step answers one call in order: an Exception is raised, anything else is returned."""
    def responder(messages, recorder):
        step = steps[recorder.count("Answer") - 1]
        if isinstance(step, Exception):
            raise step
        return step
    return ScriptedLLM({"Answer": responder})


class RecordingFactory:
    """Wraps ScriptedLLM.get_llm_client and records the client kwargs of every attempt."""

    def __init__(self, scripted: ScriptedLLM):
        self.scripted = scripted
        self.kwargs: list[dict] = []

    def __call__(self, settings, **kwargs):
        self.kwargs.append(kwargs)
        return self.scripted.get_llm_client(settings, **kwargs)


def _call(factory, **kw):
    params = dict(system_prompt="You are Cofytra.", user_prompt=f"Founder said: {SECRET_TEXT}",
                  capability="chat", prompt_version="cofytra.chat.v1", request_id="req-1",
                  settings=_settings(), llm_factory=factory)
    params.update(kw)
    return call_structured(Answer, **params)


# ---------------------------------------------------------------------------
# Success and retry policy
# ---------------------------------------------------------------------------

def test_valid_structured_response():
    scripted = _scripted({"text": "Interview five labs."})
    res = _call(scripted.get_llm_client)
    assert res.output == Answer(text="Interview five labs.")
    assert (res.attempts, res.tier, res.model, res.prompt_version) == (1, "fast", "scripted-fast", "cofytra.chat.v1")
    assert res.latency_ms >= 0
    call = scripted.calls[0]
    assert call.tier == "fast"
    assert [m["role"] for m in call.messages] == ["system", "user"]
    meta = res.meta("req-1")
    assert isinstance(meta, ResponseMeta) and meta.attempts == 1 and meta.model == "scripted-fast"


def test_reasoning_tier_is_passed_through_and_invalid_tier_rejected():
    scripted = _scripted({"text": "ok"})
    assert _call(scripted.get_llm_client, tier="reasoning").tier == "reasoning"
    assert scripted.calls[0].tier == "reasoning"
    with pytest.raises(ValueError):
        _call(scripted.get_llm_client, tier="medium")


def test_invalid_structured_output_then_success_retries_with_correction():
    scripted = _scripted({}, {"text": "fixed"})
    res = _call(scripted.get_llm_client)
    assert res.output.text == "fixed" and res.attempts == 2
    retry_messages = scripted.calls[1].messages
    assert len(retry_messages) == 3 and retry_messages[-1]["role"] == "user"
    assert "text" in retry_messages[-1]["content"]          # names the failing field
    assert SECRET_TEXT not in retry_messages[-1]["content"]  # never echoes founder text
    assert scripted.calls[1].config["metadata"]["retry_reason"] == "invalid_structured_output"


def test_invalid_structured_output_twice_raises_invalid_model_output():
    scripted = _scripted({}, {"text": ""})
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client)
    assert exc_info.value.code == ErrorCode.INVALID_MODEL_OUTPUT and exc_info.value.retryable
    assert scripted.count("Answer") == MAX_ATTEMPTS


def test_validation_error_text_containing_status_codes_is_not_misread_as_auth_failure():
    # extra_forbidden echoes the input "403 forbidden"; it must still be treated as malformed output.
    scripted = _scripted({"text": "x", "unexpected": "403 forbidden unauthorized"}, {"text": "ok"})
    assert _call(scripted.get_llm_client).attempts == 2


def test_unvalidated_model_instance_is_revalidated_never_returned_partially():
    scripted = _scripted(Answer.model_construct(text=""), Answer.model_construct(text=""))
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client)
    assert exc_info.value.code == ErrorCode.INVALID_MODEL_OUTPUT


def test_output_check_violation_is_retried_then_succeeds():
    scripted = _scripted({"text": "a", "ids": ["kn-999"]}, {"text": "b", "ids": ["kn-1"]})
    check = lambda out: [f"unknown {i}" for i in out.ids if i != "kn-1"]  # noqa: E731
    res = _call(scripted.get_llm_client, output_check=check)
    assert res.output.ids == ["kn-1"] and res.attempts == 2
    correction = scripted.calls[1].messages[-1]["content"]
    assert "1 output rule" in correction and "kn-999" not in correction


def test_output_check_violation_twice_raises():
    scripted = _scripted({"text": "a", "ids": ["kn-999"]}, {"text": "a", "ids": ["kn-998"]})
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client, output_check=lambda out: ["bad"] if out.ids else [])
    assert exc_info.value.code == ErrorCode.INVALID_MODEL_OUTPUT


def test_transient_429_then_success_retries_same_request():
    scripted = _scripted(RateLimitError("rate limited"), {"text": "ok"})
    res = _call(scripted.get_llm_client)
    assert res.attempts == 2
    assert scripted.calls[0].messages == scripted.calls[1].messages


def test_permanent_401_is_not_retried():
    scripted = _scripted(AuthenticationError("invalid api key"), {"text": "never reached"})
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client)
    assert exc_info.value.code == ErrorCode.PROVIDER_CONFIG and not exc_info.value.retryable
    assert scripted.count("Answer") == 1


def test_timeout_twice_raises_provider_unavailable():
    scripted = _scripted(APITimeoutError("Request timed out."), APITimeoutError("Request timed out."))
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client)
    assert exc_info.value.code == ErrorCode.PROVIDER_UNAVAILABLE and exc_info.value.retryable
    assert exc_info.value.suggested_http_status == 503
    assert scripted.count("Answer") == MAX_ATTEMPTS


def test_token_budget_exhaustion_retries_once_with_expanded_budget():
    factory = RecordingFactory(_scripted(LengthFinishReasonError("length limit was reached"), {"text": "ok"}))
    res = _call(factory, max_completion_tokens=1000)
    assert res.attempts == 2
    assert factory.kwargs[0]["max_completion_tokens"] == 1000
    assert factory.kwargs[1]["max_completion_tokens"] == 2500  # max(1.6x, +1500)
    assert factory.kwargs[1]["reasoning_effort"] == "low"


def test_token_budget_exhaustion_twice_raises_token_budget():
    factory = RecordingFactory(_scripted(LengthFinishReasonError("length limit was reached"),
                                         LengthFinishReasonError("length limit was reached")))
    with pytest.raises(CofytraAIError) as exc_info:
        _call(factory)
    assert exc_info.value.code == ErrorCode.TOKEN_BUDGET and not exc_info.value.retryable
    assert len(factory.kwargs) == MAX_ATTEMPTS


def test_unknown_error_is_not_retried():
    scripted = _scripted(RuntimeError("boom"), {"text": "never reached"})
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client)
    assert exc_info.value.code == ErrorCode.INTERNAL and scripted.count("Answer") == 1


def test_never_more_than_two_provider_attempts():
    factory = RecordingFactory(_scripted(RateLimitError("x"), RateLimitError("x"), {"text": "third"}))
    with pytest.raises(CofytraAIError):
        _call(factory)
    assert len(factory.kwargs) == 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", [None, "", "sk-...", "PASTE_YOUR_OPENAI_API_KEY_HERE"])
def test_missing_api_key_raises_provider_config_without_calling_provider(monkeypatch, key):
    def exploding_client(*a, **k):
        raise AssertionError("provider must not be constructed without an API key")
    monkeypatch.setattr(llm_mod, "get_llm_client", exploding_client)
    with pytest.raises(CofytraAIError) as exc_info:
        _call(None, settings=_settings(openai_api_key=key))
    assert exc_info.value.code == ErrorCode.PROVIDER_CONFIG


def test_missing_api_key_from_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_settings().openai_api_key is None
    with pytest.raises(CofytraAIError) as exc_info:
        call_structured(Answer, system_prompt="s", user_prompt="u", capability="chat", prompt_version="v1")
    assert exc_info.value.code == ErrorCode.PROVIDER_CONFIG


def test_default_factory_is_the_startary_provider(monkeypatch):
    scripted = _scripted({"text": "ok"})
    monkeypatch.setattr(llm_mod, "get_llm_client", scripted.get_llm_client)
    assert _call(None).output.text == "ok"


def test_settings_come_from_environment_not_working_directory(monkeypatch, tmp_path):
    dotenv_dir = tmp_path / "with_env"
    dotenv_dir.mkdir()
    (dotenv_dir / ".env").write_text(
        "OPENAI_API_KEY=sk-from-dotenv-must-be-ignored\nFAST_MODEL_NAME=dotenv-model\nLANGSMITH_TRACING=true\n",
        encoding="utf-8")
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    for var in ("FAST_MODEL_NAME", "STARTARY_FAST_MODEL", "LANGSMITH_TRACING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("FAST_MODEL_NAME", "env-fast-model")

    results = []
    for cwd in (dotenv_dir, other_dir, REPO_ROOT):  # the repo root has a real .env, which must be ignored too
        monkeypatch.chdir(cwd)
        s = load_settings()
        results.append((s.openai_api_key, s.fast_model_name, s.langsmith_tracing))
    assert results == [(SECRET_KEY, "env-fast-model", False)] * 3


def test_startary_prefixed_model_variable_also_works(monkeypatch):
    monkeypatch.delenv("FAST_MODEL_NAME", raising=False)
    monkeypatch.setenv("STARTARY_FAST_MODEL", "prefixed-fast")
    assert load_settings().fast_model_name == "prefixed-fast"


# ---------------------------------------------------------------------------
# Privacy: LangSmith metadata, logs, errors
# ---------------------------------------------------------------------------

def test_run_config_metadata_is_sanitized_and_complete():
    cfg = build_run_config("extract_knowledge", "cofytra.extract.v1", "fast", 2, "req-9", "transient_api_error")
    md = cfg["metadata"]
    assert sanitize_metadata(md) == md  # nothing in it would be filtered
    assert md["template_version"] == "cofytra.extract.v1" and md["capability"] == "extract_knowledge"
    assert md["attempt"] == 2 and md["request_id"] == "req-9" and md["model_tier"] == "fast"
    assert cfg["run_name"] == "cofytra.extract_knowledge:attempt_2"
    assert "callbacks" not in cfg


def test_recorded_run_config_never_contains_prompts_or_keys():
    scripted = _scripted({}, {"text": "ok"})
    _call(scripted.get_llm_client, settings=_settings(openai_api_key="sk-real-looking-key-123"))
    for call in scripted.calls:
        blob = json.dumps(call.config, default=str)
        assert SECRET_TEXT not in blob and "You are Cofytra." not in blob
        assert "sk-real-looking-key-123" not in blob


def test_tracing_disabled_adds_no_callbacks_and_does_not_touch_environment(monkeypatch):
    monkeypatch.setattr(llm_mod, "get_langsmith_tracer", lambda s: pytest.fail("tracer must not be built"))
    before = {k: v for k, v in os.environ.items() if k.startswith(("LANGSMITH", "LANGCHAIN"))}
    scripted = _scripted({"text": "ok"})
    _call(scripted.get_llm_client, settings=_settings(langsmith_tracing=False))
    assert "callbacks" not in scripted.calls[0].config
    assert {k: v for k, v in os.environ.items() if k.startswith(("LANGSMITH", "LANGCHAIN"))} == before


def test_tracing_enabled_attaches_tracer(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(llm_mod, "get_langsmith_tracer", lambda s: sentinel)
    scripted = _scripted({"text": "ok"})
    _call(scripted.get_llm_client, settings=_settings(langsmith_tracing=True))
    assert scripted.calls[0].config["callbacks"] == [sentinel]


def test_logs_and_errors_never_contain_founder_text_or_keys(caplog):
    caplog.set_level(logging.DEBUG)
    scripted = _scripted(APITimeoutError(f"timeout while sending {SECRET_TEXT} with {SECRET_KEY}"),
                         APITimeoutError(f"timeout while sending {SECRET_TEXT}"))
    with pytest.raises(CofytraAIError) as exc_info:
        _call(scripted.get_llm_client)
    err = exc_info.value
    for text in (caplog.text, str(err), err.info.model_dump_json()):
        assert SECRET_TEXT not in text and SECRET_KEY not in text
    assert err.__cause__ is None and err.__suppress_context__  # raw provider exception not chained
    assert "APITimeoutError" in err.info.message


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------

def test_llm_module_does_not_import_langgraph_or_startary_workflow():
    code = (
        "import sys, cofytra_ai.llm\n"
        "mods = set(sys.modules)\n"
        "bad = sorted(m for m in mods if m.split('.')[0] in ('langgraph', 'django') "
        "or m.startswith(('startary.graph', 'startary.cli', 'startary.llm.prompts', 'startary.llm.schemas')))\n"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
                         env={**os.environ, "LANGSMITH_TRACING": "false"})
    assert out.stdout.strip() == ""
