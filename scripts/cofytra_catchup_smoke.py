"""Controlled F4 smoke test through the real service.catch_up().

Dry run by default; only --live permits OpenAI calls (fast tier, at most two service attempts).
All inputs are fictional contract examples. No private-message input or file output is supported.
Keys come from the process environment (never .env) or hidden --prompt-key input, in memory only.

  python scripts/cofytra_catchup_smoke.py
  python scripts/cofytra_catchup_smoke.py --scenario empty
  python scripts/cofytra_catchup_smoke.py --scenario proposals-only
  python scripts/cofytra_catchup_smoke.py --live --prompt-key
"""

from __future__ import annotations

import os

# Must precede imports of catchup, LangChain and the provider.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cofytra_ai import service  # noqa: E402
from cofytra_ai.errors import CofytraAIError  # noqa: E402
from cofytra_ai.models import CatchupRequest, CatchupResponse  # noqa: E402

CONTRACT_DOC = REPO_ROOT / "docs" / "COFYTRA_AI_BACKEND_CONTRACT_V1.md"
FICTIONAL_BANNER = (
    "*** FICTIONAL TEST DATA ***\n"
    "Ana and Ben, all research, demo requests, market figures and sources below are invented test data.\n"
    "The two lab managers, 12,000 EU labs and 'EU Diagnostics Industry Report 2025' are NOT real\n"
    "customer research, a real market figure or a real source. Judge summary quality and epistemic accuracy only."
)


def _doc_example(name: str) -> dict:
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    match = re.search(rf"<!-- example: {re.escape(name)} -->\s*```json\n(.*?)```", text, re.DOTALL)
    if not match:
        raise SystemExit(f"Could not find the {name} contract example. Nothing was sent.")
    return json.loads(match.group(1))


def build_request(scenario: str) -> CatchupRequest:
    if scenario == "empty":
        data = _doc_example("catch_up_empty.request")
    else:
        data = _doc_example("catch_up.request")
        if scenario == "proposals-only":
            data["request_id"] = "req-smoke-catchup-proposals"
            data["knowledge"] = []
        elif scenario != "documented":
            raise SystemExit("Unknown scenario. Nothing was sent.")
    return CatchupRequest.model_validate(data)


def _base_factory(settings: Any, **kwargs: Any) -> Any:
    from startary.llm.provider import get_llm_client

    return get_llm_client(settings, **kwargs)


def _forbid_provider(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("This scenario must not invoke a provider.")


def _print_safe(text: str, *secrets: str) -> None:
    # Defence in depth, including if a provider ever echoes a credential in its output.
    values = (*secrets, os.environ.get("OPENAI_API_KEY", ""), os.environ.get("LANGSMITH_API_KEY", ""))
    for value in sorted(set(values), key=len, reverse=True):
        if value:
            text = text.replace(value, "[REDACTED]")
            text = text.replace(json.dumps(value)[1:-1], "[REDACTED]")
    print(text)


def load_live_settings(prompt_key: bool):
    from cofytra_ai.llm import load_settings
    from startary.llm.provider import is_valid_api_key

    settings = load_settings()  # environment only; .env is ignored
    updates: dict = {"langsmith_tracing": False}
    if prompt_key:
        # getpass otherwise warns and falls back to visible input on unsupported terminals.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("OpenAI API key (input hidden): ").strip()
        if not is_valid_api_key(key):
            raise SystemExit("No valid API key entered. Nothing was sent.")
        updates["openai_api_key"] = key
    elif not is_valid_api_key(settings.openai_api_key):
        raise SystemExit("OPENAI_API_KEY is not set in this shell. Nothing was sent.\n"
                         "Re-run with --live --prompt-key (hidden input, kept in memory only).")
    return settings.model_copy(update=updates)


def print_dry_run(request: CatchupRequest, show_system: bool) -> None:
    from cofytra_ai import catchup

    _print_safe("DRY RUN: no API call was made. Add --live to call OpenAI.\n"
                f"prompt_version={catchup.PROMPT_VERSION}  model=not called  latency_ms=0  attempts=0")
    if show_system:
        _print_safe("\n=== SYSTEM PROMPT (fixed) ===\n" + catchup.SYSTEM_PROMPT)
    _print_safe("\n=== USER PROMPT (fictional shared data only) ===\n" +
                catchup.build_user_prompt(request, catchup.prepare_context(request)))


def print_response(request: CatchupRequest, response: CatchupResponse, *, secret: str = "", as_json: bool = False) -> None:
    lines = ["=== SUMMARY ===", response.summary]
    referenced = []
    for bucket in ("discoveries", "assumptions", "evidence"):
        lines.append(f"\n=== {bucket.upper()} ===")
        points = getattr(response, bucket)
        if not points:
            lines.append("(none)")
        for point in points:
            lines.append(f"- {point.text} [knowledge_ids: {', '.join(point.knowledge_ids)}]")
            referenced.extend(point.knowledge_ids)
    lines.append("\n=== PENDING DECISIONS (awaiting team approval) ===")
    for point in response.pending_decisions:
        lines.append(f"- {point.proposal_id}: {point.text}")
    if not response.pending_decisions:
        lines.append("(none)")
    lines.append("\n=== REFERENCED SHARED KNOWLEDGE (invented test data) ===")
    knowledge = {k.id: k for k in request.knowledge}
    for kid in dict.fromkeys(referenced):
        k = knowledge[kid]  # the real service validates reference integrity
        lines.append(f"- {kid}: [{k.epistemic_status.value}] {k.statement}")
        if k.external_reference:
            lines.append(f"  invented source, not checked: {k.external_reference}")
    if not referenced:
        lines.append("(none)")
    m = response.meta
    lines.extend(["\n=== METADATA ===",
                  f"model={m.model}  tier={m.tier}  latency_ms={m.latency_ms}  attempts={m.attempts}  "
                  f"prompt_version={m.prompt_version}  request_id={m.request_id}  contract_version={m.contract_version}"])
    if as_json:
        lines.extend(["\n=== FULL RESPONSE JSON (fictional test output) ===",
                      json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False)])
    _print_safe("\n".join(lines), secret)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled service.catch_up() smoke test (dry run unless --live).")
    parser.add_argument("--scenario", choices=["documented", "empty", "proposals-only"], default="documented",
                        help="Fictional contract scenario (default: documented).")
    parser.add_argument("--live", action="store_true", help="Permit paid OpenAI calls; at most two service attempts.")
    parser.add_argument("--prompt-key", action="store_true", help="Enter a key at a hidden prompt for --live only.")
    parser.add_argument("--show-system", action="store_true", help="Dry run: also print the fixed system prompt.")
    parser.add_argument("--json", action="store_true", help="Also print the CatchupResponse as JSON when available.")
    args = parser.parse_args(argv)
    _print_safe(FICTIONAL_BANNER + f"\n\nScenario: {args.scenario}\n")

    try:
        request = build_request(args.scenario)
        if not request.knowledge:
            _print_safe("NO LLM CALL: deterministic service.catch_up() path; no API key required.")
            response = service.catch_up(request, llm_factory=_forbid_provider)
            print_response(request, response, as_json=args.json)
            return 0
        if not args.live:
            print_dry_run(request, args.show_system)
            return 0

        settings = load_live_settings(args.prompt_key)
        _print_safe(f"LIVE: OpenAI ({settings.get_model_for_tier('fast')}, fast tier, LangSmith disabled).\n",
                    settings.openai_api_key)
        response = service.catch_up(request, settings=settings, llm_factory=_base_factory)
        print_response(request, response, secret=settings.openai_api_key, as_json=args.json)
    except CofytraAIError as exc:
        _print_safe(f"CofytraAIError: code={exc.code.value} retryable={exc.retryable}")
        return 2
    except Exception:
        # No raw exception/traceback: configuration/provider errors can contain credentials or input.
        _print_safe("Smoke test failed. Details suppressed to protect credentials; check configuration and hidden-input support.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
