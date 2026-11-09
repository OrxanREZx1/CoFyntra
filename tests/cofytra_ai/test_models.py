"""
Offline contract tests for cofytra_ai v1 (no LLM, no network).

The JSON examples are read straight from docs/COFYTRA_AI_BACKEND_CONTRACT_V1.md, so the published
contract and the Pydantic models cannot drift apart.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from cofytra_ai import service
from cofytra_ai.errors import CofytraAIError
from cofytra_ai.models import (
    CAPABILITIES,
    MAX_KNOWLEDGE_ITEMS,
    MAX_PRIVATE_MESSAGES,
    CatchupRequest,
    ChatRequest,
    ConflictResponse,
    ErrorCode,
    ErrorInfo,
    EpistemicStatus,
    ExtractRequest,
    ExtractResponse,
    KnowledgeCandidate,
    ProposalRequest,
    ProposalResponse,
    ResponseMeta,
    SharedKnowledgeItem,
    contract_violations,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC = REPO_ROOT / "docs" / "COFYTRA_AI_BACKEND_CONTRACT_V1.md"
_EXAMPLE_RE = re.compile(r"<!-- example: (?P<name>[\w.]+) -->\s*```json\n(?P<body>.*?)```", re.DOTALL)

# example base name -> capability key
_BASES = {
    "chat": "chat",
    "extract_knowledge": "extract_knowledge",
    "catch_up": "catch_up",
    "catch_up_empty": "catch_up",
    "detect_conflicts": "detect_conflicts",
    "draft_proposal": "draft_proposal",
}


def _load_examples() -> dict[str, dict]:
    found = {m["name"]: json.loads(m["body"]) for m in _EXAMPLE_RE.finditer(DOC.read_text(encoding="utf-8"))}
    assert found, "no examples found in the contract doc"
    return found


EXAMPLES = _load_examples()
PAIRS = sorted(_BASES)
ERROR_EXAMPLES = sorted(n for n in EXAMPLES if n.startswith("error."))


def _req(base: str):
    return CAPABILITIES[_BASES[base]][0].model_validate(EXAMPLES[f"{base}.request"])


def _resp(base: str):
    return CAPABILITIES[_BASES[base]][1].model_validate(EXAMPLES[f"{base}.response"])


def _ex(name: str) -> dict:
    return copy.deepcopy(EXAMPLES[name])


# ---------------------------------------------------------------------------
# Every documented example validates, round-trips, and passes reference integrity
# ---------------------------------------------------------------------------

def test_doc_contains_exactly_the_expected_examples():
    expected = {f"{b}.{kind}" for b in _BASES for kind in ("request", "response")}
    assert expected <= set(EXAMPLES)
    assert set(EXAMPLES) - expected == set(ERROR_EXAMPLES)
    assert ERROR_EXAMPLES, "doc must include at least one error example"


def test_every_capability_has_a_documented_example():
    assert set(_BASES.values()) == set(CAPABILITIES)


@pytest.mark.parametrize("base", PAIRS)
@pytest.mark.parametrize("kind", ["request", "response"])
def test_example_validates_and_round_trips(base, kind):
    model_cls = CAPABILITIES[_BASES[base]][0 if kind == "request" else 1]
    obj = model_cls.model_validate(EXAMPLES[f"{base}.{kind}"])
    dumped = obj.model_dump(mode="json")
    assert model_cls.model_validate(dumped) == obj
    assert model_cls.model_validate_json(json.dumps(dumped)) == obj


@pytest.mark.parametrize("base", PAIRS)
def test_example_pair_has_no_contract_violations(base):
    assert contract_violations(_req(base), _resp(base)) == []


@pytest.mark.parametrize("name", ERROR_EXAMPLES)
def test_error_example_validates(name):
    ErrorInfo.model_validate(EXAMPLES[name])


# ---------------------------------------------------------------------------
# Strictness and limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base", PAIRS)
def test_unknown_request_fields_are_rejected(base):
    data = _ex(f"{base}.request")
    data["unexpected"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        CAPABILITIES[_BASES[base]][0].model_validate(data)


def test_empty_strings_are_rejected_not_treated_as_absent():
    data = _ex("chat.request")
    data["message"] = "   "
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(data)


def test_naive_timestamps_are_rejected():
    data = _ex("chat.request")
    data["recent_messages"][0]["created_at"] = "2026-11-09T09:00:00"
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(data)


def test_invalid_id_format_is_rejected():
    data = _ex("chat.request")
    data["founder"]["id"] = "u ana"
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(data)


def test_private_message_limit_is_enforced_not_truncated():
    data = _ex("chat.request")
    template = data["recent_messages"][0]
    data["recent_messages"] = [dict(template, id=f"msg-{i}") for i in range(MAX_PRIVATE_MESSAGES + 1)]
    with pytest.raises(ValidationError, match="too_long|at most"):
        ChatRequest.model_validate(data)


def test_knowledge_limit_is_enforced():
    data = _ex("detect_conflicts.request")
    template = data["knowledge"][0]
    data["knowledge"] = [dict(template, id=f"kn-{i}") for i in range(MAX_KNOWLEDGE_ITEMS + 1)]
    with pytest.raises(ValidationError):
        CAPABILITIES["detect_conflicts"][0].model_validate(data)


def test_duplicate_ids_are_rejected():
    data = _ex("detect_conflicts.request")
    data["knowledge"][1]["id"] = data["knowledge"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate knowledge ids"):
        CAPABILITIES["detect_conflicts"][0].model_validate(data)


def test_minimal_chat_request_uses_empty_defaults():
    req = ChatRequest.model_validate({"request_id": "r1", "project_id": "p1", "founder": {"id": "u-ana"}, "message": "Hi"})
    assert req.recent_messages == [] and req.shared_knowledge == [] and req.committed_state is None


def test_committed_state_next_action_is_optional():
    assert _req("chat").committed_state.next_action  # documented example carries it
    assert _req("draft_proposal").committed_state.next_action is None  # older-shape example still valid
    data = _ex("chat.request")
    data["committed_state"].pop("next_action")
    assert ChatRequest.model_validate(data).committed_state.next_action is None


@pytest.mark.parametrize("value", ["", "   ", "x" * 501])
def test_committed_state_next_action_rejects_empty_or_oversized(value):
    data = _ex("chat.request")
    data["committed_state"]["next_action"] = value
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(data)


def test_extract_requires_at_least_one_founder_message():
    data = _ex("extract_knowledge.request")
    data["messages"] = []
    with pytest.raises(ValidationError):
        ExtractRequest.model_validate(data)
    data = _ex("extract_knowledge.request")
    data["messages"] = [m for m in data["messages"] if m["role"] == "assistant"]
    with pytest.raises(ValidationError, match="at least one founder-authored"):
        ExtractRequest.model_validate(data)


def test_messages_must_be_chronological():
    data = _ex("chat.request")
    data["recent_messages"].reverse()
    with pytest.raises(ValidationError, match="oldest to newest"):
        ChatRequest.model_validate(data)


# ---------------------------------------------------------------------------
# Identity boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["messages", "recent_messages", "private_messages"])
def test_catch_up_cannot_receive_private_messages(field):
    data = _ex("catch_up.request")
    data[field] = _ex("chat.request")["recent_messages"]
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        CatchupRequest.model_validate(data)


def test_catch_up_rejects_knowledge_from_the_viewer():
    data = _ex("catch_up.request")
    data["knowledge"][0]["author_id"] = "u-ana"
    with pytest.raises(ValidationError, match="not authored by other_founder"):
        CatchupRequest.model_validate(data)


def test_catch_up_viewer_must_differ_from_other_founder():
    data = _ex("catch_up.request")
    data["other_founder"]["id"] = data["viewer"]["id"]
    with pytest.raises(ValidationError, match="different founders"):
        CatchupRequest.model_validate(data)


def test_catch_up_rejects_items_shared_before_since():
    data = _ex("catch_up.request")
    data["since"] = "2026-11-08T15:15:00Z"
    with pytest.raises(ValidationError, match="before 'since'"):
        CatchupRequest.model_validate(data)


@pytest.mark.parametrize("base,field", [("chat", "recent_messages"), ("extract_knowledge", "messages")])
def test_private_messages_must_belong_to_the_requesting_founder(base, field):
    data = _ex(f"{base}.request")
    data[field][0]["author_id"] = "u-ben"
    with pytest.raises(ValidationError, match="belongs to another founder"):
        CAPABILITIES[_BASES[base]][0].model_validate(data)


def test_assistant_messages_have_no_author():
    data = _ex("chat.request")
    data["recent_messages"][1]["author_id"] = "u-ana"
    with pytest.raises(ValidationError, match="assistant messages must have author_id null"):
        ChatRequest.model_validate(data)


def test_shared_knowledge_cannot_carry_private_message_ids():
    item = _ex("chat.request")["shared_knowledge"][0]
    item["source_message_ids"] = ["msg-101"]
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        SharedKnowledgeItem.model_validate(item)


def test_candidates_have_no_id_until_django_assigns_one():
    cand = _ex("extract_knowledge.response")["candidates"][0]
    cand["id"] = "kn-999"
    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
        KnowledgeCandidate.model_validate(cand)


# ---------------------------------------------------------------------------
# Epistemic status rules
# ---------------------------------------------------------------------------

def test_no_status_implies_verification():
    assert not any("verif" in s.value or "fact" in s.value or "proven" in s.value for s in EpistemicStatus)
    item = _ex("chat.request")["shared_knowledge"][0]
    item["epistemic_status"] = "verified"
    with pytest.raises(ValidationError):
        SharedKnowledgeItem.model_validate(item)


@pytest.mark.parametrize("category,status", [
    ("assumption", "founder_observation"),
    ("assumption", "external_source"),
    ("evidence", "founder_hypothesis"),
    ("evidence", "ai_suggestion"),
])
def test_category_status_mismatches_are_rejected(category, status):
    item = _ex("chat.request")["shared_knowledge"][0]
    item.update(category=category, epistemic_status=status, external_reference="Some report" if status == "external_source" else None)
    with pytest.raises(ValidationError, match="not allowed for category"):
        SharedKnowledgeItem.model_validate(item)


def test_external_source_requires_reference_and_reference_requires_external_source():
    item = _ex("chat.request")["shared_knowledge"][1]
    item["external_reference"] = None
    with pytest.raises(ValidationError, match="require external_reference"):
        SharedKnowledgeItem.model_validate(item)
    item = _ex("chat.request")["shared_knowledge"][0]
    item["external_reference"] = "Some report"
    with pytest.raises(ValidationError, match="only allowed with"):
        SharedKnowledgeItem.model_validate(item)


def test_extracted_candidates_cannot_be_ai_suggestions():
    cand = _ex("extract_knowledge.response")["candidates"][2]  # an assumption
    cand["epistemic_status"] = "ai_suggestion"
    with pytest.raises(ValidationError, match="cannot have epistemic_status 'ai_suggestion'"):
        KnowledgeCandidate.model_validate(cand)


def test_shared_assumption_may_be_an_ai_suggestion():
    item = _ex("detect_conflicts.request")["knowledge"][1]
    item["epistemic_status"] = "ai_suggestion"
    assert SharedKnowledgeItem.model_validate(item).epistemic_status == EpistemicStatus.AI_SUGGESTION


# ---------------------------------------------------------------------------
# Source IDs come from the request, never from the model
# ---------------------------------------------------------------------------

def _violations(base: str, mutate) -> list[str]:
    resp = _ex(f"{base}.response")
    mutate(resp)
    return contract_violations(_req(base), CAPABILITIES[_BASES[base]][1].model_validate(resp))


def test_chat_invented_ids_are_violations():
    v = _violations("chat", lambda r: (r["referenced_knowledge_ids"].append("kn-999"), r["referenced_decision_ids"].append("dec-999")))
    assert any("kn-999" in x for x in v) and any("dec-999" in x for x in v)


def test_request_id_must_be_echoed():
    assert any("request_id" in x for x in _violations("chat", lambda r: r["meta"].update(request_id="req-other")))


def test_extract_unknown_or_assistant_sources_are_violations():
    assert _violations("extract_knowledge", lambda r: r["candidates"][0].update(source_message_ids=["msg-999"]))
    assert _violations("extract_knowledge", lambda r: r["candidates"][0].update(source_message_ids=["msg-102"]))  # assistant msg


def test_extract_fabricated_quote_is_a_violation():
    v = _violations("extract_knowledge", lambda r: r["candidates"][0].update(source_quote="All 8 managers said they would buy today"))
    assert any("not verbatim" in x for x in v)


def test_extract_quote_check_ignores_whitespace_and_case():
    assert _violations("extract_knowledge", lambda r: r["candidates"][1].update(source_quote="  most USE   paper calendars. ")) == []


def test_extract_unknown_duplicate_target_is_a_violation():
    assert _violations("extract_knowledge", lambda r: r["candidates"][0].update(duplicate_of_knowledge_id="kn-999"))
    assert _violations("extract_knowledge", lambda r: r["candidates"][0].update(duplicate_of_knowledge_id="kn-201")) == []


def test_catch_up_wrong_bucket_and_unknown_ids_are_violations():
    assert any("not allowed in evidence" in x for x in _violations("catch_up", lambda r: r["evidence"][0].update(knowledge_ids=["kn-203"])))
    assert _violations("catch_up", lambda r: r["discoveries"][0].update(knowledge_ids=["kn-999"]))
    assert _violations("catch_up", lambda r: r["pending_decisions"][0].update(proposal_id="prop-999"))


def test_conflict_claims_must_be_preserved_verbatim():
    v = _violations("detect_conflicts", lambda r: r["conflicts"][0]["claim_a"].update(statement="Labs are better."))
    assert any("verbatim" in x for x in v)
    v = _violations("detect_conflicts", lambda r: r["conflicts"][0]["claim_b"].update(knowledge_id="kn-999"))
    assert any("kn-999" in x for x in v)


def test_conflict_between_same_author_or_same_item_is_rejected():
    data = _ex("detect_conflicts.response")
    data["conflicts"][0]["claim_b"] = dict(data["conflicts"][0]["claim_a"], knowledge_id="kn-203")
    with pytest.raises(ValidationError, match="different founders"):
        ConflictResponse.model_validate(data)
    data = _ex("detect_conflicts.response")
    data["conflicts"][1] = copy.deepcopy(data["conflicts"][0])
    data["conflicts"][1]["claim_a"], data["conflicts"][1]["claim_b"] = data["conflicts"][0]["claim_b"], data["conflicts"][0]["claim_a"]
    with pytest.raises(ValidationError, match="more than one conflict"):
        ConflictResponse.model_validate(data)


def test_proposal_invented_ids_are_violations():
    assert _violations("draft_proposal", lambda r: r["proposal"]["referenced_knowledge_ids"].append("kn-999"))
    assert _violations("draft_proposal", lambda r: r["warnings"][0]["knowledge_ids"].append("kn-999"))
    assert _violations("draft_proposal", lambda r: r["proposal"]["referenced_decision_ids"].append("dec-999"))


def test_mismatched_request_response_pair_is_reported():
    assert contract_violations(_req("chat"), _resp("draft_proposal"))


# ---------------------------------------------------------------------------
# AI never approves; meta and errors are consistent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["approved", "committed", "pending"])
def test_proposal_status_is_always_draft(status):
    data = _ex("draft_proposal.response")
    data["proposal"]["status"] = status
    with pytest.raises(ValidationError):
        ProposalResponse.model_validate(data)


def test_proposal_conflict_must_reference_request_knowledge():
    data = _ex("draft_proposal.request")
    data["conflict"]["claim_b_id"] = "kn-999"
    with pytest.raises(ValidationError, match="not in the request"):
        ProposalRequest.model_validate(data)


@pytest.mark.parametrize("meta", [
    {"tier": "none", "model": "gpt-x", "attempts": 0},
    {"tier": "none", "model": None, "attempts": 1},
    {"tier": "fast", "model": None, "attempts": 1},
    {"tier": "fast", "model": "gpt-x", "attempts": 0},
])
def test_meta_tier_consistency(meta):
    with pytest.raises(ValidationError):
        ResponseMeta.model_validate({"request_id": "r1", "prompt_version": "v", "latency_ms": 0, **meta})


def test_error_retryable_is_fixed_by_code():
    with pytest.raises(ValidationError, match="retryable must be"):
        ErrorInfo.model_validate({"code": "provider_config", "message": "x", "retryable": True})


def test_error_from_validation_error_never_copies_input_values():
    data = _ex("catch_up.request")
    data["messages"] = [{"text": "SECRET private founder text"}]
    with pytest.raises(ValidationError) as exc_info:
        CatchupRequest.model_validate(data)
    err = CofytraAIError.from_validation_error(exc_info.value, request_id="req-catchup-001")
    assert err.code == ErrorCode.INVALID_REQUEST and err.retryable is False and err.suggested_http_status == 500
    assert "SECRET" not in err.info.model_dump_json()
    assert any(d.field == "messages" and d.issue == "extra_forbidden" for d in err.info.details)


# ---------------------------------------------------------------------------
# Public Python interface
# ---------------------------------------------------------------------------

_EXPECTED_FUNCTIONS = {
    "chat": "ChatRequest",
    "extract_knowledge": "ExtractRequest",
    "catch_up": "CatchupRequest",
    "detect_conflicts": "ConflictRequest",
    "draft_proposal": "ProposalRequest",
}


@pytest.mark.parametrize("name", sorted(_EXPECTED_FUNCTIONS))
def test_service_signature_is_stable(name):
    fn = getattr(service, name)
    params = inspect.signature(fn).parameters
    assert list(params) == ["request", "settings", "llm_factory"]
    assert params["request"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["settings"].kind is params["llm_factory"].kind is inspect.Parameter.KEYWORD_ONLY
    hints = inspect.get_annotations(fn, eval_str=True)
    req_cls, resp_cls = CAPABILITIES[name]
    assert hints["request"] is req_cls and req_cls.__name__ == _EXPECTED_FUNCTIONS[name]
    assert hints["return"] is resp_cls


@pytest.mark.parametrize("base", ["draft_proposal"])  # chat, extract_knowledge, catch_up, detect_conflicts implemented
def test_service_stub_raises_not_implemented(base):
    with pytest.raises(CofytraAIError) as exc_info:
        getattr(service, _BASES[base])(_req(base))
    assert exc_info.value.code == ErrorCode.NOT_IMPLEMENTED
    assert exc_info.value.info.request_id == EXAMPLES[f"{base}.request"]["request_id"]


def test_service_rejects_wrong_request_type():
    with pytest.raises(TypeError):
        service.chat(_req("catch_up"))  # type: ignore[arg-type]


def test_importing_cofytra_ai_has_no_heavy_or_django_imports():
    code = (
        "import sys, cofytra_ai, cofytra_ai.models, cofytra_ai.service, cofytra_ai.errors\n"
        "bad = sorted({m.split('.')[0] for m in sys.modules} & {'langgraph','langchain_openai','langchain_core','openai','django','startary'})\n"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
