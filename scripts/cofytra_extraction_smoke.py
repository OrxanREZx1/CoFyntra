"""
Cofytra Knowledge Extraction (F3): manual smoke test against the REAL OpenAI API.

SAFE BY DEFAULT: without --live this is a dry run. It prints the exact prompt that would be sent and
makes NO network call. Only `--live` calls OpenAI (one fast-tier call, two at most if the service retries).

  Dry run, scenario A:           python scripts/cofytra_extraction_smoke.py --scenario A
  Live, scenario A:              python scripts/cofytra_extraction_smoke.py --scenario A --live --prompt-key
  Live, custom message:          python scripts/cofytra_extraction_smoke.py --message "..." --live --prompt-key
  Custom + existing knowledge:   ... --message "..." --with-existing --live --prompt-key

Scenarios (ALL FICTIONAL):
  A  founder-reported interviews + an untested willingness-to-pay guess (the documented contract example)
  B  a message with no startup-relevant knowledge (expected: no candidates)
  C  a founder hypothesis that conflicts with existing shared knowledge (expected: kept, NOT a duplicate)

API key: OPENAI_API_KEY from the environment (never from .env), or a hidden prompt with --prompt-key.
It is never printed, logged or written. LangSmith tracing is always disabled.
"""

from __future__ import annotations

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse  # noqa: E402
import copy  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cofytra_ai import service  # noqa: E402
from cofytra_ai.errors import CofytraAIError  # noqa: E402
from cofytra_ai.models import ExtractRequest, ExtractResponse  # noqa: E402

CONTRACT_DOC = REPO_ROOT / "docs" / "COFYTRA_AI_BACKEND_CONTRACT_V1.md"

FICTIONAL_BANNER = (
    "*** FICTIONAL TEST DATA ***\n"
    "Founders 'Ana' and 'Ben', the clinics/labs story, all numbers and all sources below are invented for testing.\n"
    "They are NOT real customer research. Judge the output on extraction behaviour only."
)
CUSTOM_BANNER = "Custom founder message (your text). Any existing knowledge shown is FICTIONAL test data."

LABS = {
    "id": "kn-201", "author_id": "u-ben", "category": "target_customer",
    "statement": "Diagnostic labs are the best first customer because they handle the highest booking volume.",
    "epistemic_status": "founder_hypothesis", "source_quote": None, "external_reference": None,
    "shared_at": "2026-11-08T15:00:00Z",
}


def _doc_example(name: str) -> dict:
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    match = re.search(rf"<!-- example: {re.escape(name)} -->\s*```json\n(.*?)```", text, re.DOTALL)
    if not match:
        raise SystemExit(f"Could not find the {name} example in {CONTRACT_DOC}")
    return json.loads(match.group(1))


def _single_message_request(request_id: str, text: str, existing: list[dict]) -> dict:
    return {
        "request_id": request_id, "project_id": "proj-smoke",
        "founder": {"id": "u-ana", "display_name": "Ana"},
        "messages": [{"id": "msg-901", "role": "founder", "author_id": "u-ana", "text": text,
                      "created_at": "2026-11-09T10:00:00Z"}],
        "existing_knowledge": existing,
    }


SCENARIOS: dict[str, dict[str, str]] = {
    "A": {"title": "Founder-reported interviews + untested willingness-to-pay guess (documented example)",
          "expect": "evidence/founder_observation for the interviews; the 50 EUR price only as assumption/founder_hypothesis; "
                    "no candidate citing the assistant message msg-102"},
    "B": {"title": "No startup-relevant knowledge",
          "expect": "zero candidates and zero drops"},
    "C": {"title": "Hypothesis that conflicts with existing shared knowledge (kn-201: labs first)",
          "expect": "a target_customer/founder_hypothesis about clinics with duplicate_of_knowledge_id = null"},
}


def scenario_request(name: str) -> dict:
    if name == "A":
        return _doc_example("extract_knowledge.request")
    if name == "B":
        return _single_message_request(
            "req-smoke-b",
            "Hi Ben! Can we move our sync to 3pm tomorrow? I'll bring coffee. Also my laptop is acting up again.",
            [copy.deepcopy(LABS)])
    if name == "C":
        return _single_message_request(
            "req-smoke-c",
            "Honestly, I think independent clinics, not labs, should be our first customer. Labs will move too slowly.",
            [copy.deepcopy(LABS)])
    raise SystemExit(f"Unknown scenario {name!r}")


def build_request(scenario: str | None, message: str | None, with_existing: bool) -> tuple[ExtractRequest, str, str]:
    if message is not None:
        data = _single_message_request("req-smoke-custom", message, [copy.deepcopy(LABS)] if with_existing else [])
        return ExtractRequest.model_validate(data), CUSTOM_BANNER, "custom message"
    name = (scenario or "A").upper()
    info = SCENARIOS.get(name) or {}
    header = f"{FICTIONAL_BANNER}\n\nScenario {name}: {info.get('title')}\nExpected: {info.get('expect')}"
    return ExtractRequest.model_validate(scenario_request(name)), header, f"scenario {name}"


# ---------------------------------------------------------------------------
# Measuring "proposed": wrap the real provider client and count what the model returned
# ---------------------------------------------------------------------------

def _base_factory(settings: Any, **kwargs: Any) -> Any:
    from startary.llm.provider import get_llm_client
    return get_llm_client(settings, **kwargs)


class _CountingStructured:
    def __init__(self, inner: Any, sink: list[int]):
        self._inner, self._sink = inner, sink

    def invoke(self, messages: Any, config: Any = None) -> Any:
        result = self._inner.invoke(messages, config=config)
        candidates = getattr(result, "candidates", None)
        if candidates is None and isinstance(result, dict):
            candidates = result.get("candidates")
        self._sink.append(len(candidates) if isinstance(candidates, list) else -1)
        return result


class _CountingClient:
    """Delegates everything (incl. model-resolution metadata) to the real client."""

    def __init__(self, inner: Any, sink: list[int]):
        self._inner, self._sink = inner, sink

    def with_structured_output(self, schema: Any) -> _CountingStructured:
        return _CountingStructured(self._inner.with_structured_output(schema), self._sink)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def counting_factory(sink: list[int]):
    def factory(settings: Any, **kwargs: Any) -> Any:
        return _CountingClient(_base_factory(settings, **kwargs), sink)
    return factory


# ---------------------------------------------------------------------------
# Settings, dry run, printing
# ---------------------------------------------------------------------------

def load_live_settings(prompt_key: bool):
    from cofytra_ai.llm import load_settings
    from startary.llm.provider import is_valid_api_key

    settings = load_settings()
    updates: dict = {"langsmith_tracing": False}
    if prompt_key:
        key = getpass.getpass("OpenAI API key (input hidden): ").strip()
        if not is_valid_api_key(key):
            raise SystemExit("No valid API key entered. Nothing was sent.")
        updates["openai_api_key"] = key
    elif not is_valid_api_key(settings.openai_api_key):
        raise SystemExit("OPENAI_API_KEY is not set in this shell. Nothing was sent.\n"
                         "Re-run with --prompt-key (hidden input, kept in memory only).")
    return settings.model_copy(update=updates)


def print_dry_run(request: ExtractRequest, show_system: bool) -> None:
    from cofytra_ai import extraction

    print("DRY RUN: no API call was made. Add --live to call OpenAI.\n")
    if show_system:
        print("=== SYSTEM PROMPT (fixed) ===")
        print(extraction.SYSTEM_PROMPT)
    else:
        print(f"(System prompt {extraction.PROMPT_VERSION}: {len(extraction.SYSTEM_PROMPT)} chars; add --show-system to print it.)\n")
    print("=== USER PROMPT (what would be sent) ===")
    print(extraction.build_user_prompt(request))


def print_response(response: ExtractResponse, proposed_per_attempt: list[int]) -> None:
    print("=== EXTRACTED CANDIDATES ===")
    if not response.candidates:
        print("(none)")
    for n, c in enumerate(response.candidates, 1):
        print(f"[{n}] {c.category.value} | {c.epistemic_status.value} | confidence={c.confidence.value}")
        print(f"    statement : {c.statement}")
        print(f"    sources   : {', '.join(c.source_message_ids)}")
        print(f"    quote     : \"{c.source_quote}\"")
        if c.external_reference:
            print(f"    reference : {c.external_reference}")
        print(f"    duplicate : {c.duplicate_of_knowledge_id or 'none (would be shared as new)'}")

    d = response.dropped
    dropped_total = d.unknown_source + d.quote_not_found + d.invalid_status
    accepted = len(response.candidates)
    print("\n=== DROPPED (contract counters) ===")
    print(f"unknown_source={d.unknown_source}  quote_not_found={d.quote_not_found}  invalid_status={d.invalid_status}  total={dropped_total}")

    print("\n=== COUNTS ===")
    final = proposed_per_attempt[-1] if proposed_per_attempt else None
    if final is not None and final >= 0:
        merged = final - accepted - dropped_total
        print(f"proposed by model (final attempt, measured by this script) = {final}")
        print(f"accepted = {accepted}   dropped = {dropped_total}   "
              f"merged repeats (proposed - accepted - dropped) = {merged}")
    else:
        print(f"proposed by model = not measurable   accepted = {accepted}   dropped = {dropped_total}")
    if len(proposed_per_attempt) > 1:
        print(f"note: the service retried; model output per attempt = {proposed_per_attempt} (-1 = unparseable)")

    m = response.meta
    print("\n=== METADATA ===")
    print(f"model={m.model}  tier={m.tier}  latency_ms={m.latency_ms}  attempts={m.attempts}  "
          f"prompt_version={m.prompt_version}  request_id={m.request_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual smoke test for cofytra_ai service.extract_knowledge() (dry run unless --live).")
    parser.add_argument("--scenario", choices=["A", "B", "C", "a", "b", "c"], help="Fictional scenario (default A).")
    parser.add_argument("--message", help="Custom founder message instead of a scenario.")
    parser.add_argument("--with-existing", action="store_true", help="With --message: include fictional kn-201 as existing knowledge.")
    parser.add_argument("--live", action="store_true", help="Actually call the OpenAI API (paid). Without it: dry run.")
    parser.add_argument("--prompt-key", action="store_true", help="Type the API key at a hidden prompt.")
    parser.add_argument("--show-system", action="store_true", help="Dry run: also print the system prompt.")
    parser.add_argument("--json", action="store_true", help="Also print the full ExtractResponse as JSON.")
    args = parser.parse_args(argv)
    if args.message is not None and args.scenario:
        parser.error("use either --scenario or --message, not both")

    request, header, label = build_request(args.scenario, args.message, args.with_existing)
    print(header + "\n")

    if not args.live:
        print_dry_run(request, args.show_system)
        return 0

    settings = load_live_settings(args.prompt_key)
    print(f"LIVE: extracting {label} with OpenAI ({settings.get_model_for_tier('fast')}, fast tier, LangSmith disabled)...\n")
    proposed: list[int] = []
    try:
        response = service.extract_knowledge(request, settings=settings, llm_factory=counting_factory(proposed))
    except CofytraAIError as exc:
        print(f"CofytraAIError: code={exc.code.value} retryable={exc.retryable} message={exc.info.message}")
        return 2
    print_response(response, proposed)
    if args.json:
        print("\n=== FULL RESPONSE JSON ===")
        print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
