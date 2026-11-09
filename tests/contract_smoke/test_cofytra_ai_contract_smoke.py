"""
Backend-side smoke tests for the imported Cofytra AI contract v1.

These are NOT the AI team's original tests/cofytra_ai/test_models.py (89 tests), which has not
been delivered. They only check, offline and without any LLM call, that:
  - the package imports through the documented paths and the five service signatures are intact;
  - every JSON example in docs/COFYTRA_AI_BACKEND_CONTRACT_V1.md validates against cofytra_ai.models
    and passes contract_violations;
  - a few identity and reference-integrity rules the Django backend relies on hold.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from cofytra_ai import CofytraAIError, models, service
from cofytra_ai.models import ErrorCode

DOC = Path(__file__).resolve().parents[2] / "docs" / "COFYTRA_AI_BACKEND_CONTRACT_V1.md"
_EXAMPLE = re.compile(r"<!-- example: ([\w.]+) -->\s*```json\n(.*?)\n```", re.S)

FUNCTIONS = ["chat", "extract_knowledge", "catch_up", "detect_conflicts", "draft_proposal"]


def _examples() -> dict[str, dict]:
    return {name: json.loads(body) for name, body in _EXAMPLE.findall(DOC.read_text(encoding="utf-8"))}


# example prefix -> capability name in models.CAPABILITIES
_PAIRS = {
    "chat": "chat",
    "extract_knowledge": "extract_knowledge",
    "catch_up": "catch_up",
    "catch_up_empty": "catch_up",
    "detect_conflicts": "detect_conflicts",
    "draft_proposal": "draft_proposal",
}


def test_documented_imports_and_signatures():
    assert set(models.CAPABILITIES) == set(FUNCTIONS)
    for name in FUNCTIONS:
        params = inspect.signature(getattr(service, name)).parameters
        assert list(params) == ["request", "settings", "llm_factory"]
        assert params["settings"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["llm_factory"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["settings"].default is None and params["llm_factory"].default is None


def test_doc_has_all_examples():
    assert set(_examples()) == {f"{p}.{kind}" for p in _PAIRS for kind in ("request", "response")} | {
        "error.provider_unavailable", "error.invalid_request"}


@pytest.mark.parametrize("prefix", sorted(_PAIRS))
def test_doc_example_pair_validates(prefix):
    ex = _examples()
    req_cls, resp_cls = models.CAPABILITIES[_PAIRS[prefix]]
    request = req_cls.model_validate(ex[f"{prefix}.request"])
    response = resp_cls.model_validate(ex[f"{prefix}.response"])
    assert models.contract_violations(request, response) == []


@pytest.mark.parametrize("name", ["error.provider_unavailable", "error.invalid_request"])
def test_doc_error_examples_validate(name):
    info = models.ErrorInfo.model_validate(_examples()[name])
    assert info.retryable is models.ERROR_RETRYABLE[info.code]


@pytest.mark.parametrize("prefix", sorted(_PAIRS))
def test_stubs_raise_not_implemented(prefix):
    capability = _PAIRS[prefix]
    req_cls, _ = models.CAPABILITIES[capability]
    request = req_cls.model_validate(_examples()[f"{prefix}.request"])
    with pytest.raises(CofytraAIError) as exc:
        getattr(service, capability)(request)
    assert exc.value.code is ErrorCode.NOT_IMPLEMENTED
    assert exc.value.info.request_id == request.request_id
    assert exc.value.suggested_http_status == 501


def test_stub_rejects_wrong_request_type():
    request = models.ConflictRequest.model_validate(_examples()["detect_conflicts.request"])
    with pytest.raises(TypeError):
        service.chat(request)


def test_catch_up_rejects_private_messages_without_leaking_them():
    data = dict(_examples()["catch_up_empty.request"])
    data["messages"] = [{"text": "SECRET-PRIVATE-TEXT"}]
    with pytest.raises(ValidationError) as exc:
        models.CatchupRequest.model_validate(data)
    err = CofytraAIError.from_validation_error(exc.value, data["request_id"])
    assert err.code is ErrorCode.INVALID_REQUEST
    assert {"field": "messages", "issue": "extra_forbidden"} in [d.model_dump() for d in err.info.details]
    assert "SECRET-PRIVATE-TEXT" not in err.info.model_dump_json()


def test_chat_rejects_other_founders_private_message():
    data = json.loads(json.dumps(_examples()["chat.request"]))
    data["recent_messages"][0]["author_id"] = "u-ben"
    with pytest.raises(ValidationError):
        models.ChatRequest.model_validate(data)


def test_contract_violations_flags_ids_not_in_request():
    ex = _examples()
    request = models.ChatRequest.model_validate(ex["chat.request"])
    bad = json.loads(json.dumps(ex["chat.response"]))
    bad["referenced_knowledge_ids"] = ["kn-201", "kn-999"]
    bad["meta"]["request_id"] = "req-other"
    violations = models.contract_violations(request, models.ChatResponse.model_validate(bad))
    assert any("kn-999" in v for v in violations)
    assert any("request_id" in v for v in violations)


def test_proposal_draft_cannot_be_approved_by_ai():
    data = json.loads(json.dumps(_examples()["draft_proposal.response"]))
    data["proposal"]["status"] = "approved"
    with pytest.raises(ValidationError):
        models.ProposalResponse.model_validate(data)
