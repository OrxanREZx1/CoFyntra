"""
Cofytra AI public interface (contract v1).

Django calls these five functions in-process with a validated request model and receives a
validated response model, or a ``CofytraAIError``. The signatures are the frozen contract.

Implemented: ``chat`` (``cofytra_ai.assistant``), ``extract_knowledge`` (``cofytra_ai.extraction``),
``catch_up`` (``cofytra_ai.catchup``) and ``detect_conflicts`` (``cofytra_ai.conflicts``). Still a CONTRACT
STUB: ``draft_proposal``; it validates its input type and raises ``CofytraAIError(ErrorCode.NOT_IMPLEMENTED)``.

Capability modules are imported lazily, so ``import cofytra_ai`` stays free of LangChain/OpenAI.

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
    if not isinstance(request, ChatRequest):
        raise TypeError(f"chat expects ChatRequest, got {type(request).__name__}")
    from cofytra_ai.assistant import run_chat  # lazy: keeps package import light

    return run_chat(request, settings=settings, llm_factory=llm_factory)


def extract_knowledge(request: ExtractRequest, *, settings: Optional[Any] = None,
                      llm_factory: Optional[LLMFactory] = None) -> ExtractResponse:
    """Extract shareable knowledge candidates from the founder's own messages. Nothing is shared by this call."""
    if not isinstance(request, ExtractRequest):
        raise TypeError(f"extract_knowledge expects ExtractRequest, got {type(request).__name__}")
    from cofytra_ai.extraction import run_extraction  # lazy: keeps package import light

    return run_extraction(request, settings=settings, llm_factory=llm_factory)


def catch_up(request: CatchupRequest, *, settings: Optional[Any] = None, llm_factory: Optional[LLMFactory] = None) -> CatchupResponse:
    """Summarize the other founder's shared knowledge and pending proposals. Never sees private chat."""
    if not isinstance(request, CatchupRequest):
        raise TypeError(f"catch_up expects CatchupRequest, got {type(request).__name__}")
    from cofytra_ai.catchup import run_catchup  # lazy: keeps package import light

    return run_catchup(request, settings=settings, llm_factory=llm_factory)


def detect_conflicts(request: ConflictRequest, *, settings: Optional[Any] = None,
                     llm_factory: Optional[LLMFactory] = None) -> ConflictResponse:
    """Find contradictory or tense claim pairs across founders, preserving both claims verbatim."""
    if not isinstance(request, ConflictRequest):
        raise TypeError(f"detect_conflicts expects ConflictRequest, got {type(request).__name__}")
    from cofytra_ai.conflicts import run_conflicts  # lazy: keeps package import light

    return run_conflicts(request, settings=settings, llm_factory=llm_factory)


def draft_proposal(request: ProposalRequest, *, settings: Optional[Any] = None,
                   llm_factory: Optional[LLMFactory] = None) -> ProposalResponse:
    """Draft a decision proposal. Always status 'draft'; approval and commit belong to Django."""
    _not_implemented("draft_proposal", request, ProposalRequest)
    raise AssertionError("unreachable")
