"""
M6.2.6.1 — Controlled live experiment-revision probe (development only).

Starts the PRODUCTION experiment subgraph (build_experiment_graph) right after generate_experiment,
with a seeded, deliberately flawed but schema-valid draft, so the real V2 path runs:

    validate_proposed_experiment -> critique_experiment (real Sol critic, V2 full)
        -> ACCEPT: save_final_experiment -> assess_can_execute          (revision path NOT exercised)
        -> REVISE (production strategy choice):
             TARGETED_EDIT     -> revise_experiment -> validate -> critique (FOCUSED or FULL)
             FULL_REGENERATION -> generate_experiment -> validate -> critique (FULL)
        -> second REVISE -> handle_experiment_failure_stop

Nothing is mocked in live mode: no forced verdicts, no prompt changes, no threshold changes. The probe
only seeds state and observes. It never runs on import.

PowerShell (from the worktree root):
    .venv\\Scripts\\python.exe -m startary.devtools.revision_probe --offline-smoke
    $env:STARTARY_EXPERIMENT_REVISION_V2 = "true"
    .venv\\Scripts\\python.exe -m startary.devtools.revision_probe --live --confirm-paid --scenario shiftrelay
    # M6.2.6.2 narrow decision-rule defect (TARGETED_EDIT + FOCUSED critique candidate):
    .venv\\Scripts\\python.exe -m startary.devtools.revision_probe --live --confirm-paid --scenario shiftrelay-c
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from startary.config import Settings, get_settings
from startary.devtools.revision_scenarios import SCENARIOS, scenario_state
from startary.domain.experiment_contract import CRITIC_DIMENSIONS, evaluate_experiment_contract
from startary.domain.experiment_revision import EDITABLE_FIELDS, diff_experiment, select_revision_strategy_with_reason, strict_scope
from startary.graph.builder import build_experiment_graph
from startary.llm.provider import is_valid_api_key
from startary.llm.schemas import ExperimentGeneratorLLMOutput
from startary.observability.langsmith import build_trace_config, emit_decision_span, flush_langsmith_traces, sanitize_metadata

EXPECTED_LANGSMITH_PROJECT = "startary-ai-latency"
DEFAULT_REASONING_MODEL = "gpt-5.6-sol"
DEFAULT_ARTIFACT_DIR = Path(".startary_dev") / "revision_probe"   # git-ignored
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2


class ProbeRefused(RuntimeError):
    """Raised when a live probe is requested without every explicit opt-in and safety precondition."""


# ---------------------------------------------------------------------------
# Live preconditions
# ---------------------------------------------------------------------------

def live_preflight(settings: Settings, expected_reasoning_model: str = DEFAULT_REASONING_MODEL) -> list[str]:
    """Every reason a paid live probe must NOT run. Empty list = all preconditions met."""
    problems: list[str] = []
    if getattr(settings, "experiment_revision_v2", False) is not True:
        problems.append("STARTARY_EXPERIMENT_REVISION_V2 must be true.")
    if not is_valid_api_key(settings.openai_api_key):
        problems.append("OPENAI_API_KEY is missing or a placeholder.")
    if (settings.reasoning_model_name or "") != expected_reasoning_model:
        problems.append(f"Reasoning model is '{settings.reasoning_model_name}', expected '{expected_reasoning_model}'.")
    if not settings.langsmith_tracing:
        problems.append("LANGSMITH_TRACING must be true so the live run is observable.")
    if not settings.langsmith_api_key:
        problems.append("LANGSMITH_API_KEY is not configured.")
    if settings.langsmith_project != EXPECTED_LANGSMITH_PROJECT:
        problems.append(f"LANGSMITH_PROJECT must be '{EXPECTED_LANGSMITH_PROJECT}' (got '{settings.langsmith_project}').")
    if settings.langsmith_hide_inputs is not True or settings.langsmith_hide_outputs is not True:
        problems.append("LANGSMITH_HIDE_INPUTS and LANGSMITH_HIDE_OUTPUTS must both be true.")
    return problems


# ---------------------------------------------------------------------------
# Local input audit (live): confirms what reached the reviser / focused critic, records booleans only
# ---------------------------------------------------------------------------

class PromptInputAuditor(BaseCallbackHandler):
    """Captures chat-model inputs for the revise and critique nodes, LOCALLY ONLY (never sent anywhere)."""

    def __init__(self) -> None:
        self.inputs: list[tuple[str, Optional[int], str]] = []    # (node, revision_round, prompt text)
        self.outputs: list[tuple[str, Optional[int], Any]] = []   # (node, revision_round, parsed structured output)
        self._runs: dict[Any, tuple[str, Optional[int]]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        for node in ("revise_experiment", "critique_experiment"):
            if f"node:{node}" in (tags or []):
                text = "\n".join(str(getattr(m, "content", m)) for batch in messages for m in batch)
                revision_round = (metadata or {}).get("revision_round")
                self.inputs.append((node, revision_round, text))
                self._runs[run_id] = (node, revision_round)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        key = self._runs.get(run_id)
        if key is not None:
            self.outputs.append((key[0], key[1], parse_structured_output(response)))


def parse_structured_output(response: Any) -> Optional[dict[str, Any]]:
    """Best-effort parse of a structured-output chat response (tool call args, parsed object, or JSON text)."""
    try:
        gen = response.generations[0][0]
    except Exception:
        return None
    msg = getattr(gen, "message", None)
    content: Any = getattr(gen, "text", "")
    if msg is not None:
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            return dict(calls[0].get("args") or {})
        parsed = (getattr(msg, "additional_kwargs", None) or {}).get("parsed")
        if parsed is not None:
            return parsed.model_dump(mode="json") if hasattr(parsed, "model_dump") else dict(parsed)
        content = msg.content
    try:
        value = json.loads(content) if isinstance(content, str) else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def first_output(outputs: list[tuple[str, Optional[int], Any]], node: str, revision_round: int) -> Optional[dict[str, Any]]:
    return next((out for n, r, out in outputs if n == node and r == revision_round and out is not None), None)


def first_input(inputs: list[tuple[str, Optional[int], str]], node: str, revision_round: int) -> Optional[str]:
    """First prompt sent by `node` in `revision_round` (token-expansion retries are ignored)."""
    return next((text for n, r, text in inputs if n == node and r == revision_round), None)


def _contains(text: str, value: str) -> bool:
    """Prompts embed drafts as json.dumps output (non-ASCII escaped), so accept either encoding."""
    return value in text or json.dumps(value)[1:-1] in text


def audit_revision_inputs(reviser_inputs: list[str], second_critic_input: Optional[str], second_critic_mode: Optional[str],
                          seeded_draft: dict[str, Any], first_issues: list[str]) -> dict[str, Any]:
    """
    Did the reviser see the original draft and the critic's real feedback, and did a focused critic see
    the original issues? Booleans only; None where the check does not apply to the path taken.
    """
    reviser = reviser_inputs[0] if reviser_inputs else ""
    focused = second_critic_mode == "FOCUSED" and bool(second_critic_input)
    return {
        "reviser_called": bool(reviser_inputs),
        "reviser_saw_original_method": _contains(reviser, seeded_draft["method"]) if reviser else None,
        "reviser_saw_original_support_signal": _contains(reviser, seeded_draft["support_signal"]) if reviser else None,
        "reviser_saw_all_first_critic_issues": (bool(first_issues) and all(_contains(reviser, i) for i in first_issues)) if reviser else None,
        "focused_critic_saw_original_issues": (bool(first_issues) and all(_contains(second_critic_input, i) for i in first_issues)) if focused else None,
    }


_ZERO_QUALIFY = re.compile(
    r"\b(?:0|zero|none|no)\s+(?:of\s+the\s+(?:3|three)\s+)?(?:qualifying\s+)?managers?\b|\bno\s+manager\b|\bnone\s+of\s+the\s+(?:3|three)\b"
    r"|\b0\s*(?:/|of)\s*3\b",
    re.IGNORECASE,
)
_ONE_OR_FEWER = re.compile(
    r"fewer\s+than\s+(?:2|two)|less\s+than\s+(?:2|two)|at\s+most\s+(?:1|one)|\b0\s+or\s+1\b|zero\s+or\s+one|(?:one|1)\s+or\s+fewer",
    re.IGNORECASE,
)
_EXACTLY_ONE = re.compile(r"\b(?:exactly|only)\s+(?:1|one)\b", re.IGNORECASE)
_INCOMPLETE = re.compile(
    r"incomplete|cannot\s+be\s+completed|not\s+(?:all\s+)?completed|unable\s+to\s+complete|fewer\s+than\s+(?:3|three)\s+interviews|missing",
    re.IGNORECASE,
)
_ALL_COMPLETED = re.compile(r"all\s+(?:3|three)\s+interviews|after\s+(?:completing|all)|(?:are|were|is)\s+completed|completing\s+all", re.IGNORECASE)


def decision_rule_check(draft: dict[str, Any]) -> dict[str, Any]:
    """
    HEURISTIC check of the ShiftRelay-C decision rules (text patterns, not proof): does the
    contradiction rule still admit exactly one qualifying manager while the inconclusive rule also
    covers it? A human must still read the criteria.
    """
    support = str(draft.get("support_signal") or "")
    contradiction = str(draft.get("contradiction_signal") or "")
    inconclusive = str(draft.get("inconclusive_signal") or "")
    contradiction_admits_one = bool(_ONE_OR_FEWER.search(contradiction) or _EXACTLY_ONE.search(contradiction))
    inconclusive_covers_one = bool(_EXACTLY_ONE.search(inconclusive))
    return {
        "heuristic": True,
        "support_requires_two_or_more": bool(re.search(r"at\s+least\s+(?:2|two)|(?:2|two)\s+or\s+(?:3|three)|(?:2|two)\s+or\s+more", support, re.IGNORECASE)),
        "contradiction_requires_zero_qualifying": bool(_ZERO_QUALIFY.search(contradiction)),
        "contradiction_requires_completed_interviews": bool(_ALL_COMPLETED.search(contradiction)),
        "contradiction_admits_exactly_one": contradiction_admits_one,
        "inconclusive_covers_exactly_one": inconclusive_covers_one,
        "inconclusive_covers_incomplete_evidence": bool(_INCOMPLETE.search(inconclusive)),
        "overlap_detected": contradiction_admits_one and inconclusive_covers_one,
    }


CORE_FIELDS = ("method", "target_participants_or_scope", "evidence_to_collect", "experiment_type", "hypothesis",
               "assumption_id", "target_sample_size", "estimated_cost", "estimated_time")


def first_rejection(graph: Any, thread_cfg: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The first critique's rejection record, read from the checkpoint taken right before the revision step."""
    for snap in graph.get_state_history(thread_cfg):
        if snap.next in (("revise_experiment",), ("generate_experiment",)):
            return (snap.values.get("experiment_revision") or {}).get("rejected_critique")
    return None


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _records(state: dict[str, Any], node: str) -> list[Any]:
    tel = state.get("turn_telemetry")
    return [r for r in (getattr(tel, "records", None) or []) if r.node_name == node]


def _decision_summary(details: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not details:
        return None
    keys = ("critic_verdict", "critic_model_verdict", "verdict_flag_conflict", "failed_dimensions", "revision_trigger",
            "revision_round", "revision_strategy", "critic_mode", "draft_origin", "issue_count", "issue_codes",
            "contract_violation_codes", "original_failed_dimensions", "remaining_failed_dimensions", "changed_fields",
            "substantial_change", "original_issues_resolved", "new_blocking_issue_count", "new_blocking_issue_codes",
            "remediation_applied", "critic_revision_scope", "next_route")
    out = {k: details.get(k) for k in keys}
    out["dimensions"] = {d: details.get(f"dim_{d}") for d in CRITIC_DIMENSIONS}
    return out


def _strict_rule_respected(decision: dict[str, Any]) -> bool:
    accepted = decision["critic_verdict"] == "ACCEPT"
    all_pass = all(v == "PASS" for v in decision["dimensions"].values())
    model_accept = decision["critic_model_verdict"] == "ACCEPT"
    focused_ok = decision["critic_mode"] != "FOCUSED" or (
        decision["original_issues_resolved"] is True and not decision["new_blocking_issue_count"])
    return accepted == (model_accept and all_pass and focused_ok)


def summarize_probe(final: dict[str, Any], seeded: dict[str, Any], scenario: str, mode: str, thread_id: str,
                    wall_clock_s: float, usage: dict[str, Any], prices: Optional[tuple[float, float]],
                    input_audit: Optional[dict[str, Any]], first_reject: Optional[dict[str, Any]] = None,
                    reviser_patch: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    critiques = [r.details for r in _records(final, "critique_experiment") if r.details]
    first = _decision_summary(critiques[0]) if critiques else None
    second = _decision_summary(critiques[1]) if len(critiques) > 1 else None
    revisions = [r for r in _records(final, "revise_experiment") if r.status == "SUCCESS"]
    regenerations = [r for r in _records(final, "generate_experiment") if r.status == "SUCCESS"]
    critic_llm_calls = [r for r in _records(final, "critique_experiment") if r.status == "SUCCESS"]

    if first is not None and first["critic_verdict"] == "REVISE":
        scope = first.get("critic_revision_scope") or strict_scope((first_reject or {}).get("revision_scope"))
        first["critic_revision_scope"] = scope
        first["strategy_reason"] = select_revision_strategy_with_reason(
            first["revision_trigger"] or "LLM", first["failed_dimensions"] or [], first["contract_violation_codes"] or [], scope)[1]

    if first is None:
        outcome = "NO_CRITIQUE_RECORDED"
    elif first["critic_verdict"] == "ACCEPT":
        outcome = "FIRST_PASS_ACCEPT"            # honest result; the revision path was NOT exercised
    elif first["revision_trigger"] == "DETERMINISTIC":
        outcome = "FIRST_CRITIQUE_DETERMINISTIC"  # fixture defect: it should not trip contract checks
    elif second is None:
        outcome = "REVISION_NOT_COMPLETED"
    else:
        outcome = f"{first['revision_strategy']}_{'ACCEPTED' if second['critic_verdict'] == 'ACCEPT' else 'REJECTED'}"

    final_draft = final.get("proposed_experiment_output") or {}
    changed = diff_experiment(seeded, final_draft) if final_draft else []
    try:
        ExperimentGeneratorLLMOutput(**{k: final_draft[k] for k in ExperimentGeneratorLLMOutput.model_fields if k in final_draft})
        schema_valid = True
    except Exception:
        schema_valid = False
    revision_checks = None
    if first and first["revision_strategy"] == "TARGETED_EDIT" and revisions:
        reported = list((revisions[-1].details or {}).get("changed_fields") or [])
        revision_checks = {
            "changed_fields_reported_by_reviser": reported,
            "changed_fields_vs_seeded_draft": changed,
            "only_reported_fields_changed": set(changed) <= set(reported),
            "unchanged_fields_identical": all(seeded.get(f) == final_draft.get(f) for f in EDITABLE_FIELDS if f not in changed),
            "revised_draft_schema_valid": schema_valid,
            "revised_draft_contract_violations": evaluate_experiment_contract(final_draft, final).violation_codes,
            "expected_second_critic_mode": "FULL" if (revisions[-1].details or {}).get("substantial_change") else "FOCUSED",
            "patch_fields_returned": sorted(k for k, v in (reviser_patch or {}).items() if v is not None and k in EDITABLE_FIELDS),
            "changed_field_count": len(changed),
            "editable_field_count": len(EDITABLE_FIELDS),
            "core_fields_preserved": {f: seeded.get(f) == final_draft.get(f) for f in CORE_FIELDS},
        }
        revision_checks["patch_fields_returned_but_unchanged"] = sorted(set(revision_checks["patch_fields_returned"]) - set(changed))
        if second:
            revision_checks["second_critic_mode_as_expected"] = second["critic_mode"] == revision_checks["expected_second_critic_mode"]
    if second:
        second["all_six_dimensions_rescored"] = all(v in ("PASS", "FAIL") for v in second["dimensions"].values())
        second["strict_acceptance_rule_respected"] = _strict_rule_respected(second)

    stage_latency_ms: dict[str, list[float]] = {}
    for node in ("critique_experiment", "revise_experiment", "generate_experiment"):
        stage_latency_ms[node] = [round(r.latency_ms, 1) for r in _records(final, node) if r.status == "SUCCESS"]
    token_usage = {m: {k: u.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")} for m, u in (usage or {}).items()}
    estimated_cost = None
    if prices and token_usage:
        estimated_cost = round(sum((u["input_tokens"] or 0) * prices[0] / 1e6 + (u["output_tokens"] or 0) * prices[1] / 1e6
                                   for u in token_usage.values()), 4)
    return {
        "probe": "M6.2.6.2" if scenario == "shiftrelay-c" else "M6.2.6.1", "scenario": scenario, "flaw_under_test": SCENARIOS[scenario]["flaw"], "mode": mode,
        "thread_id": thread_id, "completed_at": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "revision_path_exercised": bool(first and first["critic_verdict"] == "REVISE" and first["revision_trigger"] == "LLM"),
        "first_critique": first,
        "revision_strategy": first["revision_strategy"] if first else None,
        "revision_checks": revision_checks,
        "decision_rule_check": {"seeded": decision_rule_check(seeded), "final": decision_rule_check(final_draft)}
        if scenario == "shiftrelay-c" else None,
        "input_audit": input_audit,
        "second_critique": second,
        "final_verdict": (second or first or {}).get("critic_verdict"),
        "final_route": (second or first or {}).get("next_route"),
        "experiment_design_status": final.get("experiment_design_status"),
        "experiment_saved": bool(final.get("active_experiment")),
        "model_calls": {"critique_experiment_llm": len(critic_llm_calls), "revise_experiment": len(revisions),
                        "generate_experiment_regeneration": len(regenerations)},
        "stage_latency_ms": stage_latency_ms,
        "wall_clock_s": round(wall_clock_s, 2),
        "token_usage": token_usage or "not captured (offline smoke); see LangSmith for live usage",
        "estimated_cost_usd": estimated_cost if estimated_cost is not None else "pass --price-input-per-1m and --price-output-per-1m, or read LangSmith",
    }


def langsmith_summary_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Flat, enum/count-only view of the summary for the LangSmith `revision_probe_summary` span."""
    first, second = summary.get("first_critique") or {}, summary.get("second_critique") or {}
    meta = {
        "probe_scenario": summary["scenario"], "probe_mode": summary["mode"], "probe_outcome": summary["outcome"],
        "revision_path_exercised": summary["revision_path_exercised"],
        "first_verdict": first.get("critic_verdict"), "first_failed_dimensions": first.get("failed_dimensions") or [],
        "revision_strategy": summary.get("revision_strategy"),
        "changed_fields": (summary.get("revision_checks") or {}).get("changed_fields_reported_by_reviser") or [],
        "second_critic_mode": second.get("critic_mode"), "original_issues_resolved": second.get("original_issues_resolved"),
        "new_blocking_issue_count": second.get("new_blocking_issue_count"),
        "final_verdict": summary["final_verdict"], "final_route": summary["final_route"],
        "critic_llm_calls": summary["model_calls"]["critique_experiment_llm"],
        "revise_calls": summary["model_calls"]["revise_experiment"],
        "regeneration_calls": summary["model_calls"]["generate_experiment_regeneration"],
        "wall_clock_s": summary["wall_clock_s"],
        "first_revision_scope": first.get("critic_revision_scope"),
        "strategy_reason": first.get("strategy_reason"),
        "patch_fields_returned": (summary.get("revision_checks") or {}).get("patch_fields_returned") or [],
        "changed_field_count": (summary.get("revision_checks") or {}).get("changed_field_count"),
        "criteria_overlap_before": ((summary.get("decision_rule_check") or {}).get("seeded") or {}).get("overlap_detected"),
        "criteria_overlap_after": ((summary.get("decision_rule_check") or {}).get("final") or {}).get("overlap_detected"),
    }
    for d, v in (first.get("dimensions") or {}).items():
        meta[f"first_dim_{d}"] = v
    for d, v in (second.get("dimensions") or {}).items():
        meta[f"second_dim_{d}"] = v
    return sanitize_metadata(meta)


def run_probe(scenario: str, settings: Settings, mode: str, *, extra_callbacks: Optional[list[Any]] = None,
              prices: Optional[tuple[float, float]] = None, artifact_dir: Optional[Path] = None,
              scripted_calls: Optional[list[Any]] = None) -> dict[str, Any]:
    """Seed the production experiment subgraph after generate_experiment and run it to completion."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Known: {sorted(SCENARIOS)}")
    state = scenario_state(scenario)
    seeded = copy.deepcopy(state["proposed_experiment_output"])
    thread_id = f"revision-probe-{scenario}-{uuid.uuid4().hex[:10]}"
    graph = build_experiment_graph(checkpointer=MemorySaver())
    thread_cfg = {"configurable": {"thread_id": thread_id}}
    graph.update_state(thread_cfg, state, as_node="generate_experiment")

    trace_cfg = build_trace_config(
        project_id=thread_id, turn_number=0, phase="TEST", run_type="revision_probe",
        scenario_id=f"m6261-{scenario}", settings=settings,
        extra_metadata={"probe_milestone": "M6.2.6.1", "probe_scenario": scenario, "probe_mode": mode, "revision_mode": "V2"},
    )
    trace_cfg["run_name"] = f"revision_probe_{scenario}"
    trace_cfg["tags"] = list(trace_cfg.get("tags") or []) + ["revision-probe", f"scenario:{scenario}", f"mode:{mode}"]
    usage_cb, auditor = UsageMetadataCallbackHandler(), PromptInputAuditor()
    trace_cfg["callbacks"] = list(trace_cfg.get("callbacks") or []) + [usage_cb, auditor] + list(extra_callbacks or [])

    holder: dict[str, Any] = {}

    def _body(_: Any) -> None:
        start = time.monotonic()
        graph.invoke(None, thread_cfg)
        final = graph.get_state(thread_cfg).values
        if scripted_calls is not None:   # offline: read prompts from the scripted LLM's record
            node_of = {"ExperimentRevisionPatchLLMOutput": "revise_experiment", "ExperimentCriticV2LLMOutput": "critique_experiment",
                       "ExperimentFocusedCriticLLMOutput": "critique_experiment"}
            captured = [(node_of[c.schema], ((c.config or {}).get("metadata") or {}).get("revision_round"),
                         "\n".join(m["content"] for m in c.messages)) for c in scripted_calls if c.schema in node_of]
        else:
            captured = auditor.inputs
        reviser_input = first_input(captured, "revise_experiment", 2)
        if scripted_calls is not None:
            raw_patch = next((c.output for c in scripted_calls if c.schema == "ExperimentRevisionPatchLLMOutput"), None)
            reviser_patch = dict(raw_patch) if isinstance(raw_patch, dict) else None
        else:
            reviser_patch = first_output(auditor.outputs, "revise_experiment", 2)
        second_critic_input = first_input(captured, "critique_experiment", 2)
        first_reject = first_rejection(graph, thread_cfg)
        first_issues = list((first_reject or {}).get("issues") or [])
        critic_records = [r.details for r in _records(final, "critique_experiment") if r.details]
        second_mode = critic_records[1].get("critic_mode") if len(critic_records) > 1 else None
        audit = audit_revision_inputs([reviser_input] if reviser_input else [], second_critic_input, second_mode,
                                      seeded, first_issues) if first_reject else None
        summary = summarize_probe(final, seeded, scenario, mode, thread_id, time.monotonic() - start,
                                  usage_cb.usage_metadata, prices, audit, first_reject=first_reject, reviser_patch=reviser_patch)
        emit_decision_span("revision_probe_summary", langsmith_summary_metadata(summary), tags=["revision-probe", f"scenario:{scenario}"])
        holder.update(summary=summary, final=final, first_reject=first_reject, reviser_patch=reviser_patch)

    RunnableLambda(_body, name=trace_cfg["run_name"]).invoke({}, config=trace_cfg)
    summary = holder["summary"]
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        final = holder["final"]
        artifact = {
            "summary": summary,
            "seeded_draft": seeded,
            "final_draft": final.get("proposed_experiment_output"),
            "first_critique_feedback": holder.get("first_reject"),
            "reviser_patch_returned": holder.get("reviser_patch"),
            "final_critic_output": final.get("experiment_critic_output"),
            "note": "Fictional ShiftRelay scenario. Local development artifact; git-ignored; contains no API keys.",
            "human_evaluation_checklist": [
                "Are the revised support / contradiction / inconclusive rules mutually exclusive and exhaustive?",
                "Do they test recurring absences with material operational disruption (the critical assumption)?",
                "Is the plan still realistic with three managers, within $350 and two weeks?",
                "Is the change genuinely narrower than a full regeneration (method, participants, evidence source kept)?",
            ],
        }
        path = artifact_dir / f"{thread_id}.json"
        path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
        summary["artifact_path"] = str(path)
    return summary


# ---------------------------------------------------------------------------
# Offline smoke (no network): scripted responses through the same production graph
# ---------------------------------------------------------------------------

def offline_smoke_responders(path: str, scenario: str = "shiftrelay") -> dict[str, Any]:
    """Scripted model responses for the offline smoke paths (targeted | redesign | accept | reject)."""
    dims_pass = {d: "PASS" for d in CRITIC_DIMENSIONS}
    if scenario == "shiftrelay-c":
        return _shiftrelay_c_responders(path, dims_pass)
    revise = {**dims_pass, "evidence_validity": "FAIL", "falsifiability": "FAIL", "verdict": "REVISE",
              "issues": ["Support and contradiction signals measure hypothetical interest in an AI assistant, not absence pain.",
                         "No threshold for recurrence or operational disruption."],
              "revision_instructions": "Base the signals on reconstructed incidents: frequency and concrete disruption.",
              "rationale": "Method is sound; decision criteria are not.", "revision_scope": "TARGETED"}
    accept = {**dims_pass, "verdict": "ACCEPT", "issues": [], "revision_instructions": "", "rationale": "Sound.", "revision_scope": "NONE"}
    patch_out = {
        "support_signal": "At least 2 of 3 managers describe 2+ last-minute absences in the past eight weeks that each cost 30+ minutes of coordination or a visible service impact.",
        "contradiction_signal": "At least 2 of 3 managers report at most one absence in eight weeks, or say absences are resolved within minutes without operational impact.",
        "inconclusive_signal": "Managers describe absences but cannot reconstruct any concrete incident, timing, or consequence.",
        "founder_actions": [a for a in SCENARIOS["shiftrelay"]["draft"]["founder_actions"] if "AI scheduling assistant" not in a],
        "evidence_to_collect": "Anonymized notes per incident: date range, how the replacement was found, coordination time, and operational consequence.",
    }
    focused_ok = {**dims_pass, "verdict": "ACCEPT", "issues": [], "revision_instructions": "", "rationale": "Resolved.",
                  "original_issues_resolved": True, "new_blocking_issues": []}
    focused_bad = {**dims_pass, "falsifiability": "FAIL", "verdict": "REVISE", "issues": ["Contradiction threshold still vague."],
                   "revision_instructions": "Tighten it.", "rationale": "Not resolved.", "original_issues_resolved": False, "new_blocking_issues": []}
    if path == "accept":
        return {"ExperimentCriticV2LLMOutput": lambda m, r: accept}
    if path == "redesign":
        regenerated = {k: v for k, v in SCENARIOS["shiftrelay"]["draft"].items()}
        regenerated.update({k: patch_out[k] for k in ("support_signal", "contradiction_signal", "inconclusive_signal", "founder_actions", "evidence_to_collect")})
        calls = iter([dict(revise, revision_scope="REDESIGN"), accept])
        return {"ExperimentCriticV2LLMOutput": lambda m, r: next(calls), "ExperimentGeneratorLLMOutput": lambda m, r: regenerated}
    return {
        "ExperimentCriticV2LLMOutput": lambda m, r: revise,
        "ExperimentRevisionPatchLLMOutput": lambda m, r: patch_out,
        "ExperimentFocusedCriticLLMOutput": lambda m, r: focused_bad if path == "reject" else focused_ok,
    }


SHIFTRELAY_C_REPAIR = {
    "contradiction_signal": "None of the 3 managers meets the qualifying pattern after all three interviews are completed.",
    "inconclusive_signal": "Exactly 1 manager meets the qualifying pattern, or fewer than three interviews are completed.",
}


def _shiftrelay_c_responders(path: str, dims_pass: dict[str, str]) -> dict[str, Any]:
    draft = SCENARIOS["shiftrelay-c"]["draft"]
    revise = {**dims_pass, "falsifiability": "FAIL", "verdict": "REVISE", "revision_scope": "TARGETED",
              "issues": ["Contradiction ('fewer than 2 qualify') and inconclusive ('exactly 1 qualifies') overlap when exactly one manager qualifies.",
                         "Contradiction should only be concluded after all three interviews are completed."],
              "revision_instructions": "Make the rules mutually exclusive: contradiction = 0 qualifying managers after all three interviews; "
                                       "inconclusive = exactly 1 qualifying manager or incomplete interviews.",
              "rationale": "Method, participants and evidence are sound; the decision rules overlap."}
    accept = {**dims_pass, "verdict": "ACCEPT", "issues": [], "revision_instructions": "", "rationale": "Sound.", "revision_scope": "NONE"}
    # The scripted reviser also re-returns the unchanged support signal, as models sometimes do.
    patch_out = {**SHIFTRELAY_C_REPAIR, "support_signal": draft["support_signal"]}
    focused_ok = {**dims_pass, "verdict": "ACCEPT", "issues": [], "revision_instructions": "", "rationale": "Overlap resolved.",
                  "original_issues_resolved": True, "new_blocking_issues": []}
    focused_bad = {**dims_pass, "falsifiability": "FAIL", "verdict": "REVISE", "issues": ["Overlap still present."],
                   "revision_instructions": "Fix the overlap.", "rationale": "Not resolved.", "original_issues_resolved": False, "new_blocking_issues": []}
    if path == "accept":
        return {"ExperimentCriticV2LLMOutput": lambda m, r: accept}
    if path == "redesign":
        calls = iter([dict(revise, revision_scope="REDESIGN"), accept])
        return {"ExperimentCriticV2LLMOutput": lambda m, r: next(calls), "ExperimentGeneratorLLMOutput": lambda m, r: {**draft, **SHIFTRELAY_C_REPAIR}}
    return {
        "ExperimentCriticV2LLMOutput": lambda m, r: revise,
        "ExperimentRevisionPatchLLMOutput": lambda m, r: patch_out,
        "ExperimentFocusedCriticLLMOutput": lambda m, r: focused_bad if path == "reject" else focused_ok,
    }


def offline_smoke_settings() -> Settings:
    return Settings(_env_file=None, openai_api_key="sk-offline-smoke", fast_model_name="scripted-fast",
                    reasoning_model_name="scripted-reasoning", experiment_revision_v2=True)


def run_offline_smoke(scenario: str, path: str, artifact_dir: Optional[Path] = None,
                      extra_callbacks: Optional[list[Any]] = None,
                      responders: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], Any]:
    """Same production graph, scripted model responses, no network. Requires the repo's tests/ fixtures."""
    from tests.fixtures.scripted_llm import ScriptedLLM  # dev-only dependency, imported lazily

    llm = ScriptedLLM(responders if responders is not None else offline_smoke_responders(path, scenario))
    settings = offline_smoke_settings()
    with ExitStack() as stack:
        stack.enter_context(patch("startary.graph.nodes.get_llm_client", side_effect=llm.get_llm_client))
        stack.enter_context(patch("startary.graph.nodes.get_settings", return_value=settings))
        summary = run_probe(scenario, settings, mode=f"offline-smoke-{path}", artifact_dir=artifact_dir,
                            extra_callbacks=extra_callbacks, scripted_calls=llm.calls)
    return summary, llm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m startary.devtools.revision_probe",
                                description="M6.2.6.1 controlled experiment-revision probe (development only).")
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="shiftrelay")
    p.add_argument("--offline-smoke", action="store_true", help="Run the production graph with scripted responses (no network).")
    p.add_argument("--smoke-path", choices=["targeted", "redesign", "accept", "reject"], default="targeted")
    p.add_argument("--live", action="store_true", help="Use the real configured models (PAID). Requires --confirm-paid.")
    p.add_argument("--confirm-paid", action="store_true", help="Explicit acknowledgement that the live run makes paid API calls.")
    p.add_argument("--expected-reasoning-model", default=DEFAULT_REASONING_MODEL)
    p.add_argument("--price-input-per-1m", type=float, default=None, help="USD per 1M input tokens (for a local cost estimate).")
    p.add_argument("--price-output-per-1m", type=float, default=None, help="USD per 1M output tokens (for a local cost estimate).")
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    p.add_argument("--no-artifact", action="store_true")
    return p


def main(argv: Optional[list[str]] = None, settings: Optional[Settings] = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_dir = None if args.no_artifact else args.artifact_dir
    prices = (args.price_input_per_1m, args.price_output_per_1m) if args.price_input_per_1m is not None and args.price_output_per_1m is not None else None

    if args.live and args.offline_smoke:
        print("Refused: choose either --live or --offline-smoke.", file=sys.stderr)
        return EXIT_REFUSED
    if not args.live:
        if not args.offline_smoke:
            print("Nothing run. Use --offline-smoke (free) or --live --confirm-paid (paid).", file=sys.stderr)
            return EXIT_REFUSED
        summary, _ = run_offline_smoke(args.scenario, args.smoke_path, artifact_dir)
        print(json.dumps(summary, indent=2, default=str))
        return EXIT_OK

    if not args.confirm_paid:
        print("Refused: --live makes paid API calls; add --confirm-paid to proceed.", file=sys.stderr)
        return EXIT_REFUSED
    settings = settings or get_settings()
    problems = live_preflight(settings, args.expected_reasoning_model)
    if problems:
        print("Refused: live preconditions not met:\n- " + "\n- ".join(problems), file=sys.stderr)
        return EXIT_REFUSED
    try:
        summary = run_probe(args.scenario, settings, mode="live", prices=prices, artifact_dir=artifact_dir)
    finally:
        flush_langsmith_traces()
    print(json.dumps(summary, indent=2, default=str))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
