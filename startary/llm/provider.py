"""
Startary LLM Provider Boundary.

This is the SINGLE place in the codebase that imports from langchain_openai.
All other modules receive an LLM client object — they never import provider
libraries directly.

Configuration:
  OPENAI_API_KEY      — required for live AI execution (set in .env)
  STARTARY_MODEL_NAME — model to use (default: gpt-4o)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from startary.config import Settings

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

PLACEHOLDER_KEYS = frozenset({
    "PASTE_YOUR_OPENAI_API_KEY_HERE",
    "sk-...",
    "",
})


def is_valid_api_key(api_key: str | None) -> bool:
    """
    Return True if an API key is present and not a known template placeholder.
    Does not log or expose the key value.
    """
    if not api_key:
        return False
    stripped = api_key.strip()
    return stripped not in PLACEHOLDER_KEYS


def get_llm_client(settings: Settings) -> "ChatOpenAI":
    """
    Construct and return the LangChain OpenAI chat client.

    This is the only function allowed to import from langchain_openai.
    Call this from nodes; do not import ChatOpenAI elsewhere.

    Raises:
        ValueError: if OPENAI_API_KEY is missing or contains a placeholder.
    """
    # Import here to keep the provider boundary clean and enable mocking
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    api_key = settings.openai_api_key
    if not is_valid_api_key(api_key):
        raise ValueError(
            "OpenAI API key is not configured. "
            "Open the project .env file and replace PASTE_YOUR_OPENAI_API_KEY_HERE with your API key."
        )

    model = settings.model_name
    logger.debug("LLM client | model=%s", model)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0,          # Deterministic; diagnosis requires consistency
        max_retries=0,          # We manage retries ourselves in the node layer
        timeout=60,
    )
