"""AI adapter: validation pipeline, mocks for all five capabilities, error mapping, isolation from extraction."""

import logging
import sys
from datetime import timedelta

import pytest
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.test import override_settings
from django.utils import timezone

from cofytra_ai import service as ai_service
from cofytra_ai.errors import CofytraAIError
from cofytra_ai.models import (
    CAPABILITIES,
    SUGGESTED_HTTP_STATUS,
    ChatRequest,
    ChatResponse,
    ConflictResponse,
    ErrorCode,
    ExtractResponse,
    contract_violations,
)
from core.api.exceptions import exception_handler
from core.models import DecisionProposal, ProposalVersion
from core.services import ai_adapter, ai_context as ctx
from core.services.ai_adapter import SAFE_ERROR_MESSAGES, AIAdapterError, ServiceBackend, get_backend
from core.services.ai_mock import MockBackend
from core.services.knowledge_sharing import AutoShareBlocked, auto_share_status, ensure_auto_share_allowed

pytestmark = pytest.mark.django_db

SECRET = "FOUNDER-SECRET-7f3a"
T0 = timezone.now() - timedelta(days=1)


@pytest.fixture
def contexts(project, ana, ben, make_message, make_knowledge, commit_revision):
    """One validated request per capability, built from a realistic two-founder project."""
    first = make_message(project, ana, "I talked to 8 clinic managers. 5 said scheduling eats 3 hours a day.",
                         created_at=T0)
    make_message(project, ana, "That is a strong signal.", role="assistant", reply_to=first,
                 created_at=T0 + timedelta(seconds=5))
    current = make_message(project, ana, f"Should we pick clinics or labs? {SECRET}", created_at=T0 + timedelta(minutes=1))
    make_message(project, ben, "Ben's private note", created_at=T0)
    labs = make_knowledge(project, ben, category="target_customer", epistemic_status="founder_hypothesis",
                          statement="Diagnostic labs are the best first customer.")
    clinics = make_knowledge(project, ana, category="target_customer", epistemic_status="founder_hypothesis",
                             statement="Independent clinics are the best first customer.")
    make_knowledge(project, ben, category="evidence", epistemic_status="founder_observation",
                   statement="Two lab managers asked for a demo.")
    commit_revision(project, ana, 1, problem="Providers lose hours to manual scheduling.")
    pending = DecisionProposal.objects.create(project=project, created_by=ben, base_state_revision=1)
    ProposalVersion.create_next(proposal=pending, created_by=ben, title="Target labs first", rationale="r",
                                trade_offs=["t"], next_action="n", referenced_knowledge=[labs])
    return {
        "chat": ctx.build_chat_request(project=project, founder=ana, message=current).request,
        "extract_knowledge": ctx.build_extract_request(project=project, founder=ana, source_message=current).request,
        "catch_up": ctx.build_catchup_request(project=project, viewer=ana).request,
        "detect_conflicts": ctx.build_conflict_request(project=project, requested_by=ana).request,
        "draft_proposal": ctx.build_proposal_request(project=project, requested_by=ana, topic="Choose a first segment",
                                                     conflict_ids=(str(labs.pk), str(clinics.pk))).request,
    }


class StubBackend:
    """Returns (or raises) whatever a test hands it."""

    name = "stub"

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def call(self, capability, request):
        self.calls.append(capability)
        if self.error is not None:
            raise self.error
        return self.result(request) if callable(self.result) else self.result


def run(capability, request, backend):
    return getattr(ai_adapter, capability)(request, backend=backend)


# 1. All five adapters with mocked outputs ------------------------------------------------------

@pytest.mark.parametrize("capability", list(CAPABILITIES))
def test_each_capability_returns_a_validated_mock_response(contexts, capability):
    backend = MockBackend()
    request = contexts[capability]
    response = run(capability, request, backend)
    assert isinstance(response, CAPABILITIES[capability][1])
    assert response.meta.request_id == request.request_id
    assert (response.meta.tier, response.meta.model, response.meta.attempts) == ("none", None, 0)
    assert contract_violations(request, response) == []
    assert backend.calls == [capability]


def test_mock_outputs_are_deterministic_and_meaningful(contexts):
    first = {c: run(c, r, MockBackend()).model_dump(mode="json") for c, r in contexts.items()}
    second = {c: run(c, r, MockBackend()).model_dump(mode="json") for c, r in contexts.items()}
    assert first == second
    assert len(first["detect_conflicts"]["conflicts"]) == 1
    source = contexts["extract_knowledge"].messages[-1]
    candidate = first["extract_knowledge"]["candidates"][0]
    assert candidate["source_message_ids"] == [source.id] and candidate["epistemic_status"] == "founder_hypothesis"
    assert candidate["source_quote"] in source.text
    assert first["catch_up"]["pending_decisions"][0]["proposal_id"] == contexts["catch_up"].pending_proposals[0].id
    assert first["draft_proposal"]["proposal"]["status"] == "draft"
    assert SECRET not in first["chat"]["reply"]


def test_mock_catchup_with_empty_context_is_deterministic(project, ana, ben):
    request = ctx.build_catchup_request(project=project, viewer=ana).request
    response = run("catch_up", request, MockBackend())
    assert response.summary == "ben has not shared new knowledge or proposals since your last catch-up."
    assert response.meta.tier == "none"


# 6 / 9. Reference integrity and invalid AI responses are rejected ---------------------------------

def _chat_response(request, **overrides):
    data = {"reply": "ok", "referenced_knowledge_ids": [], "referenced_decision_ids": [],
            "meta": {"request_id": request.request_id, "prompt_version": "test.v1", "model": "m", "tier": "fast",
                     "latency_ms": 1, "attempts": 1}}
    data.update(overrides)
    return ChatResponse.model_validate(data)


@pytest.mark.parametrize("bad", [
    lambda req: _chat_response(req, referenced_knowledge_ids=["kn-not-in-request"]),
    lambda req: _chat_response(req, referenced_decision_ids=["dec-not-in-request"]),
    lambda req: _chat_response(req, meta={"request_id": "someone-else", "prompt_version": "v", "model": "m",
                                          "tier": "fast", "latency_ms": 1, "attempts": 1}),
    lambda req: ChatResponse.model_construct(reply="", referenced_knowledge_ids=[], referenced_decision_ids=[],
                                             meta=_chat_response(req).meta),
    lambda req: ConflictResponse(conflicts=[], meta=_chat_response(req).meta),
    lambda req: {"reply": "a dict, not a model"},
], ids=["unknown-knowledge", "unknown-decision", "request-id-mismatch", "unvalidated-model", "wrong-model", "dict"])
def test_invalid_ai_output_is_rejected(contexts, bad):
    with pytest.raises(AIAdapterError) as exc:
        run("chat", contexts["chat"], StubBackend(result=bad))
    assert exc.value.code is ErrorCode.INVALID_MODEL_OUTPUT and exc.value.http_status == 502


def test_non_verbatim_extraction_quote_is_rejected(contexts):
    request = contexts["extract_knowledge"]
    founder_message = next(m for m in request.messages if m.role.value == "founder")
    invented = ExtractResponse.model_validate({
        "candidates": [{"category": "problem", "statement": "Invented claim.", "epistemic_status": "founder_hypothesis",
                        "confidence": "high", "source_message_ids": [founder_message.id],
                        "source_quote": "words the founder never wrote"}],
        "meta": {"request_id": request.request_id, "prompt_version": "v", "model": "m", "tier": "fast",
                 "latency_ms": 1, "attempts": 1}})
    with pytest.raises(AIAdapterError) as exc:
        run("extract_knowledge", request, StubBackend(result=invented))
    assert exc.value.code is ErrorCode.INVALID_MODEL_OUTPUT


def test_invalid_requests_never_reach_the_backend(contexts):
    backend = StubBackend(result=None)
    unvalidated = ChatRequest.model_construct(request_id="r-1", project_id="p-1", founder=None, message="")
    for capability, request in (("chat", unvalidated), ("chat", contexts["catch_up"])):
        with pytest.raises(AIAdapterError) as exc:
            run(capability, request, backend)
        assert exc.value.code is ErrorCode.INVALID_REQUEST and exc.value.http_status == 500
    assert backend.calls == []


# 10. Error-code mapping and safe errors --------------------------------------------------------

@pytest.mark.parametrize("code", list(ErrorCode))
def test_cofytra_errors_map_to_safe_adapter_errors(contexts, code):
    request = contexts["chat"]
    backend = StubBackend(error=CofytraAIError(code, f"provider said {SECRET}", request.request_id))
    with pytest.raises(AIAdapterError) as exc:
        run("chat", request, backend)
    err = exc.value
    assert err.code is code and err.http_status == SUGGESTED_HTTP_STATUS[code]
    assert err.retryable == (code in {ErrorCode.PROVIDER_UNAVAILABLE, ErrorCode.INVALID_MODEL_OUTPUT})
    assert err.public_payload() == {"error": {"code": code.value, "message": SAFE_ERROR_MESSAGES[code],
                                              "retryable": err.retryable, "request_id": request.request_id}}
    assert SECRET not in str(err) and SECRET not in str(err.public_payload())
    assert err.__cause__ is None and err.__suppress_context__


def test_unexpected_exceptions_become_internal_without_leaking(contexts, caplog):
    caplog.set_level(logging.DEBUG, logger="cofytra.ai_adapter")
    with pytest.raises(AIAdapterError) as exc:
        run("chat", contexts["chat"], StubBackend(error=RuntimeError(f"boom {SECRET} sk-live-key")))
    assert exc.value.code is ErrorCode.INTERNAL and exc.value.http_status == 500
    assert "RuntimeError" in caplog.text
    assert SECRET not in caplog.text and "sk-live-key" not in caplog.text


def test_successful_calls_log_no_founder_content(contexts, caplog):
    caplog.set_level(logging.DEBUG, logger="cofytra.ai_adapter")
    run("chat", contexts["chat"], MockBackend())
    assert "ai_call_ok capability=chat" in caplog.text and SECRET not in caplog.text


def test_drf_handler_returns_consistent_error_bodies():
    response = exception_handler(AIAdapterError("chat", ErrorCode.PROVIDER_UNAVAILABLE, "req-1"), {})
    assert response.status_code == 503 and response.data["error"]["retryable"] is True
    response = exception_handler(ValidationError({"topic": "Too long."}), {})
    assert response.status_code == 400 and response.data["error"]["fields"] == {"topic": ["Too long."]}


# 12. Extraction is a separate operation; automatic sharing fails closed ----------------------------

def test_chat_never_triggers_extraction(contexts):
    backend = MockBackend()
    run("chat", contexts["chat"], backend)
    assert backend.calls == ["chat"]
    run("extract_knowledge", contexts["extract_knowledge"], backend)
    assert backend.calls == ["chat", "extract_knowledge"]


def test_automatic_sharing_fails_closed_until_the_guard_is_integrated():
    assert auto_share_status() == (False, "sensitive_content_guard_not_integrated")
    with override_settings(COFYTRA_SENSITIVE_CONTENT_GUARD_CONFIRMED=True, DEBUG=True):
        with pytest.raises(AutoShareBlocked):
            ensure_auto_share_allowed()


# 13. No real LLM calls; production backend selection --------------------------------------------

def test_service_backend_calls_cofytra_ai_with_the_request_only(contexts, monkeypatch):
    seen = {}

    def fake_chat(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return _chat_response(args[0])

    monkeypatch.setattr(ai_service, "chat", fake_chat)
    run("chat", contexts["chat"], ServiceBackend())
    assert len(seen["args"]) == 1 and seen["kwargs"] == {}  # no settings=, no llm_factory=


@pytest.mark.parametrize("capability", list(CAPABILITIES))
def test_real_contract_stubs_surface_as_not_implemented(contexts, capability):
    with pytest.raises(AIAdapterError) as exc:
        run(capability, contexts[capability], ServiceBackend())
    assert exc.value.code is ErrorCode.NOT_IMPLEMENTED and exc.value.http_status == 501


def test_no_llm_client_library_is_loaded(contexts):
    for capability, request in contexts.items():
        run(capability, request, MockBackend())
    assert not {"openai", "langchain_openai", "startary"} & set(sys.modules)


def test_backend_selection_is_explicit():
    with override_settings(COFYTRA_AI_MODE="service"):
        assert isinstance(get_backend(), ServiceBackend)
    with override_settings(COFYTRA_AI_MODE="mock", DEBUG=True):
        assert isinstance(get_backend(), MockBackend)
    with override_settings(COFYTRA_AI_MODE="mock", DEBUG=False, COFYTRA_AI_ALLOW_MOCK_WITHOUT_DEBUG=False):
        with pytest.raises(ImproperlyConfigured):
            get_backend()
    with override_settings(COFYTRA_AI_MODE="mock", DEBUG=False, COFYTRA_AI_ALLOW_MOCK_WITHOUT_DEBUG=True):
        assert isinstance(get_backend(), MockBackend)
    with override_settings(COFYTRA_AI_MODE="bogus"):
        with pytest.raises(ImproperlyConfigured):
            get_backend()
