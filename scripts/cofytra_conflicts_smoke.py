"""Controlled F5a smoke test through the real service.detect_conflicts().

Dry run by default; only --live permits OpenAI calls (fast tier, at most two service attempts).
All inputs are fictional. No private-message input or file output is supported.
Keys come from the process environment (never .env) or hidden --prompt-key input, in memory only.

The expectation check only reads the service response. It never rewrites conflicts to make a
semantic miss look like a pass.

  python scripts/cofytra_conflicts_smoke.py
  python scripts/cofytra_conflicts_smoke.py --scenario compatible
  python scripts/cofytra_conflicts_smoke.py --scenario different-observations
  python scripts/cofytra_conflicts_smoke.py --scenario documented --live --prompt-key
"""

from __future__ import annotations

import os

# Must precede imports of conflicts, LangChain and the provider.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import argparse  # noqa: E402
import getpass  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cofytra_ai import service  # noqa: E402
from cofytra_ai.errors import CofytraAIError  # noqa: E402
from cofytra_ai.models import ConflictRequest, ConflictResponse  # noqa: E402

CONTRACT_DOC = REPO_ROOT / "docs" / "COFYTRA_AI_BACKEND_CONTRACT_V1.md"
SCENARIOS = ("documented", "compatible", "different-observations")
FICTIONAL_BANNER = (
    "*** FICTIONAL TEST DATA ***\n"
    "Ana, Ben, Ada and Bo, and every clinic, lab, student segment, interview count and schedule below, are invented.\n"
    "None of this is real customer research. Judge whether the model reports a real disagreement, and whether it\n"
    "stays silent when both claims can stand."
)

# Display labels are for the printed report only. They are not added to the model request.
AUTHOR_LABELS = {
    "u-ana": "Ana",
    "u-ben": "Ben",
    "u-ada": "Ada, founder A",
    "u-bo": "Bo, founder B",
}

COMPATIBLE_ADA = "We should interview university students as one possible customer segment."
COMPATIBLE_BO = "We should interview working professionals as another possible customer segment."
LISBON_OBSERVATION = (
    "In March 2026, I interviewed 8 clinic managers in Lisbon. "
    "5 of them said appointment scheduling takes about 3 hours a day."
)
PORTO_OBSERVATION = (
    "In June 2026, I interviewed 6 clinic managers in Porto. "
    "2 of them said appointment scheduling takes about 45 minutes a day."
)


# Each pair is (knowledge ids, kind). Locked to the published contract example.
# main() refuses to run if the doc example differs.
DOCUMENTED_PAIRS = (
    (frozenset({"kn-201", "kn-205"}), "contradiction"),
    (frozenset({"kn-203", "kn-206"}), "tension"),
)

EXPECTATIONS = {
    "documented": {
        "summary": "One contradiction on the exclusive first customer, and one tension on integration effort.",
        "count": 2,
        "pairs": DOCUMENTED_PAIRS,
    },
    "compatible": {
        "summary": "Interviewing students and interviewing professionals are both possible. Expected conflict count: 0.",
        "count": 0,
        "pairs": (),
    },
    "different-observations": {
        "summary": "Lisbon in March and Porto in June are different samples. Expected conflict count: 0.",
        "count": 0,
        "pairs": (),
    },
}


def _doc_example(name: str) -> dict:
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    match = re.search(rf"<!-- example: {re.escape(name)} -->\s*```json\n(.*?)```", text, re.DOTALL)
    if not match:
        raise SystemExit(f"Could not find the {name} contract example. Nothing was sent.")
    return json.loads(match.group(1))


def _contract_pairs() -> list[tuple[frozenset[str], str]]:
    data = _doc_example("detect_conflicts.response")
    return [
        (frozenset((item["claim_a"]["knowledge_id"], item["claim_b"]["knowledge_id"])), item["kind"])
        for item in data["conflicts"]
    ]


def ensure_documented_rubric() -> None:
    if _contract_pairs() != list(DOCUMENTED_PAIRS):
        raise SystemExit(
            "The documented rubric does not match detect_conflicts.response in the contract. Nothing was sent."
        )


def _item(knowledge_id: str, author_id: str, statement: str, category: str, status: str, shared_at: str) -> dict:
    return {
        "id": knowledge_id,
        "author_id": author_id,
        "category": category,
        "statement": statement,
        "epistemic_status": status,
        "source_quote": None,
        "external_reference": None,
        "shared_at": shared_at,
    }


def build_request(scenario: str) -> ConflictRequest:
    if scenario == "documented":
        data = _doc_example("detect_conflicts.request")
    elif scenario == "compatible":
        data = {
            "request_id": "req-smoke-conflicts-compatible",
            "project_id": "proj-smoke",
            "knowledge": [
                _item("kn-stu", "u-ada", COMPATIBLE_ADA, "target_customer", "founder_hypothesis", "2026-11-09T09:00:00Z"),
                _item("kn-pro", "u-bo", COMPATIBLE_BO, "target_customer", "founder_hypothesis", "2026-11-09T09:05:00Z"),
            ],
        }
    elif scenario == "different-observations":
        data = {
            "request_id": "req-smoke-conflicts-observations",
            "project_id": "proj-smoke",
            "knowledge": [
                _item("kn-lisbon", "u-ada", LISBON_OBSERVATION, "evidence", "founder_observation", "2026-11-09T10:00:00Z"),
                _item("kn-porto", "u-bo", PORTO_OBSERVATION, "evidence", "founder_observation", "2026-11-09T10:05:00Z"),
            ],
        }
    else:
        raise SystemExit("Unknown scenario. Nothing was sent.")
    return ConflictRequest.model_validate(data)


def _base_factory(settings: Any, **kwargs: Any) -> Any:
    from startary.llm.provider import get_llm_client

    return get_llm_client(settings, **kwargs)


def _print_safe(text: str, *secrets: str) -> None:
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


def _author(author_id: str) -> str:
    label = AUTHOR_LABELS.get(author_id)
    return f"{label} ({author_id})" if label else author_id


def score_response(scenario: str, response: ConflictResponse) -> tuple[bool, list[str]]:
    """Compare the service response with the scenario rubric. Does not modify the response."""
    expected = EXPECTATIONS[scenario]
    actual = [
        (frozenset((conflict.claim_a.knowledge_id, conflict.claim_b.knowledge_id)), conflict.kind.value)
        for conflict in response.conflicts
    ]
    notes: list[str] = []
    if len(actual) != expected["count"]:
        notes.append(f"count: expected {expected['count']}, actual {len(actual)}")
    expected_pairs = set(expected["pairs"])
    actual_pairs = set(actual)
    for knowledge_ids, kind in sorted(expected_pairs - actual_pairs, key=lambda item: (sorted(item[0]), item[1])):
        notes.append(f"missing pair {' + '.join(sorted(knowledge_ids))} kind={kind}")
    for knowledge_ids, kind in sorted(actual_pairs - expected_pairs, key=lambda item: (sorted(item[0]), item[1])):
        notes.append(f"unexpected pair {' + '.join(sorted(knowledge_ids))} kind={kind}")
    return (not notes, notes)


def _expectation_lines(scenario: str, response: Optional[ConflictResponse]) -> list[str]:
    expected = EXPECTATIONS[scenario]
    lines = [
        "=== EXPECTATION CHECK (fictional rubric; model output was not rewritten) ===",
        expected["summary"],
        "Pair identity ignores which claim is listed first. Kind must match.",
        "Expected pairs:",
    ]
    if expected["pairs"]:
        for knowledge_ids, kind in expected["pairs"]:
            lines.append(f"- {' + '.join(sorted(knowledge_ids))} kind={kind}")
    else:
        lines.append("- (none)")
    lines.append("Actual pairs:")
    if response is None:
        lines.append("- (not scored: dry run, no model output)")
        lines.append("RESULT: NOT SCORED (dry run)")
        return lines
    if response.conflicts:
        for conflict in response.conflicts:
            lines.append(
                f"- {conflict.claim_a.knowledge_id} + {conflict.claim_b.knowledge_id} kind={conflict.kind.value}"
            )
    else:
        lines.append("- (none)")
    passed, notes = score_response(scenario, response)
    lines.append("RESULT: PASS" if passed else "RESULT: FAIL")
    lines.extend(f"- {note}" for note in notes)
    return lines


def print_dry_run(request: ConflictRequest, scenario: str, show_system: bool) -> None:
    from cofytra_ai import conflicts

    _print_safe("DRY RUN: no API call was made. Add --live to call OpenAI.\n"
                f"prompt_version={conflicts.PROMPT_VERSION}  model=not called  latency_ms=0  attempts=0")
    if show_system:
        _print_safe("\n=== SYSTEM PROMPT (fixed) ===\n" + conflicts.SYSTEM_PROMPT)
    _print_safe("\n=== USER PROMPT (fictional shared data only) ===\n" + conflicts.build_user_prompt(request))
    _print_safe("\n" + "\n".join(_expectation_lines(scenario, None)))


def print_response(request: ConflictRequest, response: ConflictResponse, scenario: str, *,
                   secret: str = "", as_json: bool = False) -> bool:
    by_id = {item.id: item for item in request.knowledge}
    lines = [f"Detected conflict count: {len(response.conflicts)}", "", "=== DETECTED CONFLICTS (model output) ==="]
    if not response.conflicts:
        lines.append("(none)")
    for number, conflict in enumerate(response.conflicts, start=1):
        lines.append(f"\n{number}. kind={conflict.kind.value}")
        for side, claim in (("claim_a", conflict.claim_a), ("claim_b", conflict.claim_b)):
            item = by_id.get(claim.knowledge_id)
            status = item.epistemic_status.value if item is not None else "not in request"
            lines.append(f"  {side}: {claim.knowledge_id} | {_author(claim.author_id)} | {status}")
            lines.append(f"  statement: {claim.statement}")
        lines.append(f"  explanation: {conflict.explanation}")
        lines.append(f"  question_to_resolve: {conflict.question_to_resolve}")
    meta = response.meta
    lines.extend([
        "",
        "=== METADATA ===",
        f"model={meta.model}  tier={meta.tier}  latency_ms={meta.latency_ms}  attempts={meta.attempts}  "
        f"prompt_version={meta.prompt_version}  request_id={meta.request_id}  contract_version={meta.contract_version}",
        "",
        *_expectation_lines(scenario, response),
    ])
    if as_json:
        lines.extend([
            "",
            "=== FULL RESPONSE JSON (fictional test output, unchanged) ===",
            json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False),
        ])
    _print_safe("\n".join(lines), secret)
    passed, _notes = score_response(scenario, response)
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Controlled service.detect_conflicts() smoke test (dry run unless --live).")
    parser.add_argument("--scenario", choices=SCENARIOS, default="documented",
                        help="Fictional scenario (default: documented).")
    parser.add_argument("--live", action="store_true", help="Permit paid OpenAI calls; at most two service attempts.")
    parser.add_argument("--prompt-key", action="store_true", help="Enter a key at a hidden prompt for --live only.")
    parser.add_argument("--show-system", action="store_true", help="Dry run: also print the fixed system prompt.")
    parser.add_argument("--json", action="store_true", help="Also print the ConflictResponse as JSON when a call was made.")
    args = parser.parse_args(argv)
    _print_safe(FICTIONAL_BANNER + f"\n\nScenario: {args.scenario}\n")

    try:
        ensure_documented_rubric()
        request = build_request(args.scenario)
        if not args.live:
            print_dry_run(request, args.scenario, args.show_system)
            return 0

        settings = load_live_settings(args.prompt_key)
        _print_safe(f"LIVE: OpenAI ({settings.get_model_for_tier('fast')}, fast tier, LangSmith disabled).\n",
                    settings.openai_api_key)
        response = service.detect_conflicts(request, settings=settings, llm_factory=_base_factory)
        passed = print_response(request, response, args.scenario, secret=settings.openai_api_key, as_json=args.json)
    except CofytraAIError as exc:
        _print_safe(f"CofytraAIError: code={exc.code.value} retryable={exc.retryable}")
        return 2
    except Exception:
        # No raw exception/traceback: configuration/provider errors can contain credentials or input.
        _print_safe("Smoke test failed. Details suppressed to protect credentials; check configuration and hidden-input support.")
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
