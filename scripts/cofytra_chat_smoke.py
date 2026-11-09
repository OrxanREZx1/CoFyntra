"""
Cofytra Founder Assistant: manual smoke test against the REAL OpenAI API.

SAFE BY DEFAULT: without --live this is a dry run. It builds the exact prompt that would be sent and
prints it, and makes NO network call. Only `--live` calls OpenAI (one fast-tier call, two at most if
the service retries).

  Dry run, documented scenario:     python scripts/cofytra_chat_smoke.py
  Live, documented scenario:        python scripts/cofytra_chat_smoke.py --live
  Live, your own message:           python scripts/cofytra_chat_smoke.py --live --message "How should we price this?"
  Live, own message, no context:    python scripts/cofytra_chat_smoke.py --live --message "..." --no-context

API key: read from the OPENAI_API_KEY environment variable (never from .env), or typed at a hidden
prompt with --prompt-key. It is never printed, logged, or written anywhere.
LangSmith tracing is always disabled for this script.
"""

from __future__ import annotations

import os

# Disable tracing before anything LangChain-related is imported.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cofytra_ai import service  # noqa: E402
from cofytra_ai.errors import CofytraAIError  # noqa: E402
from cofytra_ai.models import ChatRequest, ChatResponse  # noqa: E402

CONTRACT_DOC = REPO_ROOT / "docs" / "COFYTRA_AI_BACKEND_CONTRACT_V1.md"

FICTIONAL_BANNER = (
    "*** FICTIONAL SCENARIO ***\n"
    "The documented scenario (founders 'Ana' and 'Ben', clinics vs. labs, '5 of 8 managers', '12,000 EU labs',\n"
    "'EU Diagnostics Industry Report 2025') is invented test data from the contract doc. It is NOT real\n"
    "customer research, NOT a real market figure, and NOT a real source. Judge the reply on behaviour only."
)
CUSTOM_NO_CONTEXT_BANNER = "Custom message with NO project context (no history, no shared knowledge, no committed state)."
CUSTOM_WITH_CONTEXT_BANNER = (
    "Custom message on top of the FICTIONAL documented context (history, knowledge and committed state are invented)."
)


def load_documented_request() -> dict:
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    match = re.search(r"<!-- example: chat\.request -->\s*```json\n(.*?)```", text, re.DOTALL)
    if not match:
        raise SystemExit(f"Could not find the chat.request example in {CONTRACT_DOC}")
    return json.loads(match.group(1))


def build_request(message: str | None, no_context: bool) -> tuple[ChatRequest, str]:
    data = load_documented_request()
    if message is None:
        if no_context:
            raise SystemExit("--no-context requires --message")
        return ChatRequest.model_validate(data), FICTIONAL_BANNER
    data["request_id"] = "req-smoke-custom"
    data["message"] = message
    if no_context:
        data["recent_messages"] = []
        data["shared_knowledge"] = []
        data["committed_state"] = None
        return ChatRequest.model_validate(data), CUSTOM_NO_CONTEXT_BANNER
    return ChatRequest.model_validate(data), CUSTOM_WITH_CONTEXT_BANNER


def load_live_settings(prompt_key: bool):
    from cofytra_ai.llm import load_settings  # environment variables only; .env is ignored
    from startary.llm.provider import is_valid_api_key

    settings = load_settings()
    updates: dict = {"langsmith_tracing": False}
    if prompt_key:
        key = getpass.getpass("OpenAI API key (input hidden): ").strip()
        if not is_valid_api_key(key):
            raise SystemExit("No valid API key entered. Nothing was sent.")
        updates["openai_api_key"] = key
    elif not is_valid_api_key(settings.openai_api_key):
        raise SystemExit(
            "OPENAI_API_KEY is not set in this shell. Nothing was sent.\n"
            "Easiest: re-run with --prompt-key (hidden input, the key is kept in memory only).\n"
            "Or set it for this PowerShell session only:\n"
            "  $s = Read-Host -AsSecureString 'OpenAI key'\n"
            "  $env:OPENAI_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR("
            "[Runtime.InteropServices.Marshal]::SecureStringToBSTR($s))"
        )
    return settings.model_copy(update=updates)


def print_dry_run(request: ChatRequest) -> None:
    from cofytra_ai import assistant  # local prompt construction only; no network

    ctx = assistant.prepare_context(request)
    print("DRY RUN: no API call was made. Add --live to call OpenAI.\n")
    print("=== SYSTEM PROMPT (fixed) ===")
    print(assistant.SYSTEM_PROMPT)
    print("=== USER PROMPT (what would be sent) ===")
    print(assistant.build_user_prompt(request, ctx))


def print_response(request: ChatRequest, response: ChatResponse) -> None:
    knowledge = {k.id: k for k in request.shared_knowledge}
    decisions = {d.id: d for d in (request.committed_state.decisions if request.committed_state else [])}
    m = response.meta
    print("=== ASSISTANT REPLY ===")
    print(response.reply)
    print("\n=== REFERENCED SHARED KNOWLEDGE ===")
    for kid in response.referenced_knowledge_ids or ["(none)"]:
        k = knowledge.get(kid)
        print(f"- {kid}: [{k.epistemic_status.value}] {k.statement}" if k else f"- {kid}")
    print("\n=== REFERENCED COMMITTED DECISIONS ===")
    for did in response.referenced_decision_ids or ["(none)"]:
        d = decisions.get(did)
        print(f"- {did}: {d.title}" if d else f"- {did}")
    print("\n=== METADATA ===")
    print(f"model={m.model}  tier={m.tier}  latency_ms={m.latency_ms}  attempts={m.attempts}  "
          f"prompt_version={m.prompt_version}  request_id={m.request_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual smoke test for cofytra_ai service.chat() (dry run unless --live).")
    parser.add_argument("--live", action="store_true", help="Actually call the OpenAI API (paid). Without it: dry run.")
    parser.add_argument("--message", help="Custom founder message instead of the documented scenario's message.")
    parser.add_argument("--no-context", action="store_true", help="With --message: send no history, knowledge or committed state.")
    parser.add_argument("--prompt-key", action="store_true", help="Type the API key at a hidden prompt instead of using OPENAI_API_KEY.")
    parser.add_argument("--json", action="store_true", help="Also print the full ChatResponse as JSON.")
    args = parser.parse_args(argv)

    request, banner = build_request(args.message, args.no_context)
    print(banner + "\n")

    if not args.live:
        print_dry_run(request)
        return 0

    settings = load_live_settings(args.prompt_key)
    print(f"LIVE: calling OpenAI ({settings.get_model_for_tier('fast')}, fast tier, LangSmith disabled)...\n")
    try:
        response = service.chat(request, settings=settings)
    except CofytraAIError as exc:
        print(f"CofytraAIError: code={exc.code.value} retryable={exc.retryable} message={exc.info.message}")
        return 2
    print_response(request, response)
    if args.json:
        print("\n=== FULL RESPONSE JSON ===")
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
