"""
Cofytra AI: importable package called in-process by the Django backend.

Public interface: the five functions in ``cofytra_ai.service`` plus the models in
``cofytra_ai.models``. Importing this package has no side effects and does not load
LangChain, LangGraph, Django, or any database code.
"""

from cofytra_ai.errors import CofytraAIError
from cofytra_ai.models import CONTRACT_VERSION
from cofytra_ai.service import catch_up, chat, detect_conflicts, draft_proposal, extract_knowledge

__all__ = [
    "CONTRACT_VERSION",
    "CofytraAIError",
    "catch_up",
    "chat",
    "detect_conflicts",
    "draft_proposal",
    "extract_knowledge",
]
