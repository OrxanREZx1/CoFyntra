"""
Cofytra AI public interface (contract v1).

Django calls these five functions in-process with a validated request model and receives a
validated response model, or a ``CofytraAIError``. The signatures are the frozen contract.

CONTRACT STUBS: the capability logic is not implemented yet. Each function currently validates
its input type and raises ``CofytraAIError(ErrorCode.NOT_IMPLEMENTED)``.

``llm_factory`` is for offline tests only (inject a scripted fake). Django never passes it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from cofytra_ai.errors import CofytraAIError
from cofytra_ai.models import (
    CatchupRequest,
    CatchupResponse,
    ChatRequest,
    ChatResponse,
    ConflictRequest,
    ConflictResponse,
    ErrorCode,
    ExtractRequest,
    ExtractResponse,
    ProposalRequest,
    ProposalResponse,
)

LLMFactory = Callable[..., Any]  # same call shape as startary.llm.provider.get_llm_client


def _not_implemented(name: str, request: Any, expected: type) -> None:
    if not isinstance(request, expected):
        raise TypeError(f"{name} expects {expected.__name__}, got {type(request).__name__}")
    raise CofytraAIError(ErrorCode.NOT_IMPLEMENTED, f"{name} is not implemented yet.", request.request_id)


def chat(request: ChatRequest, *, settings: Optional[Any] = None, llm_factory: Optional[LLMFactory] = None) -> ChatResponse:
    """Context-aware assistant reply for one founder (own private history + shared knowledge + committed state)."""
    _not_implemented("chat", request, ChatRequest)
    raise AssertionError("unreachable")


def extract_knowledge(request: ExtractRequest, *, settings: Optional[Any] = None,
                      llm_factory: Optional[LLMFactory] = None) -> ExtractResponse:
    """Extract shareable knowledge candidates from the founder's own messages. Nothing is shared by this call."""
    _not_implemented("extract_knowledge", request, ExtractRequest)
    raise AssertionError("unreachable")


def catch_up(request: CatchupRequest, *, settings: Optional[Any] = None, llm_factory: Optional[LLMFactory] = None) -> CatchupResponse:
    """Summarize the other founder's shared knowledge and pending proposals. Never sees private chat."""
    _not_implemented("catch_up", request, CatchupRequest)
    raise AssertionError("unreachable")


def detect_conflicts(request: ConflictRequest, *, settings: Optional[Any] = None,
                     llm_factory: Optional[LLMFactory] = None) -> ConflictResponse:
    """Find contradictory or tense claim pairs across founders, preserving both claims verbatim."""
    _not_implemented("detect_conflicts", request, ConflictRequest)
    raise AssertionError("unreachable")


def draft_proposal(request: ProposalRequest, *, settings: Optional[Any] = None,
                   llm_factory: Optional[LLMFactory] = None) -> ProposalResponse:
    """Draft a decision proposal. Always status 'draft'; approval and commit belong to Django."""
    _not_implemented("draft_proposal", request, ProposalRequest)
    raise AssertionError("unreachable")
