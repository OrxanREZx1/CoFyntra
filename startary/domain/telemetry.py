"""
Telemetry and Token Accounting module for Startary (Milestone 4.1.3).

Tracks node-level latency and token usage, and aggregates per-turn metrics
without requiring an external observability service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Milestone 6.2.3: logical pipeline stage per node, so per-turn telemetry can be grouped
# consistently (intake -> diagnosis -> gap planning -> risk -> weak link -> question).
NODE_LOGICAL_STAGES: dict[str, str] = {
    "interpret_founder_turn": "intake",
    "dispatch_founder_turn": "intake",
    "join_founder_turn": "intake",
    "extract_startup_state": "intake",
    "assess_baseline_state_sufficiency": "baseline_gate",
    "plan_baseline_question": "question_planning",
    "diagnose_startup": "diagnosis",
    "plan_information_gap": "gap_planning",
    "assess_startup_areas": "risk_assessment",
    "assess_weak_link_discrimination": "weak_link_selection",
    "select_weak_link_question": "question_planning",
    "compose_advisor_response": "question_composition",
    "handle_conversational_repair": "question_composition",
    "generate_experiment": "experiment_design",
    "revise_experiment": "experiment_design",
    "validate_proposed_experiment": "experiment_validation",
    "critique_experiment": "experiment_critique",
}


def logical_stage_for(node_name: str) -> str:
    """Return the logical pipeline stage for a node (defaults to 'other')."""
    return NODE_LOGICAL_STAGES.get(node_name, "other")


@dataclass
class NodeTelemetryRecord:
    """Telemetry record for a single LLM or reasoning node call."""
    node_name: str
    model_name: str = "unknown"
    configured_model: Optional[str] = None
    resolved_model: str = "unknown"
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    duration_ms: float = 0.0
    latency_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    retry_count: int = 0
    timeout_status: bool = False
    budget_exceeded: bool = False
    schema_name: Optional[str] = None
    founder_turn_id: Optional[str] = None
    project_id: Optional[str] = None
    iteration: int = 1
    execution_path: str = "STRATEGIC_REASONING"
    skipped_by_fast_path: bool = False
    cached: bool = False
    skipped: bool = False
    # Milestone 6.2.3 observability fields
    logical_stage: Optional[str] = None
    model_tier: Optional[str] = None
    reasoning_effort: Optional[str] = None
    status: str = "SUCCESS"
    # Milestone 6.2.5: node-specific structured, non-sensitive metadata (e.g. critic decision codes)
    details: Optional[dict[str, Any]] = None

    def __post_init__(self):
        now = datetime.now(timezone.utc)
        if self.start_time is None:
            self.start_time = now
        if self.end_time is None:
            self.end_time = now
        if self.duration_ms and not self.latency_ms:
            self.latency_ms = self.duration_ms
        if self.resolved_model == "unknown" and self.model_name != "unknown":
            self.resolved_model = self.model_name
        elif self.model_name == "unknown" and self.resolved_model != "unknown":
            self.model_name = self.resolved_model
        if self.configured_model is None:
            self.configured_model = self.resolved_model

        if self.logical_stage is None:
            self.logical_stage = logical_stage_for(self.node_name)
        for field_attr in ("model_name", "configured_model", "resolved_model", "fallback_reason", "model_tier", "reasoning_effort"):
            val = getattr(self, field_attr)
            if val is not None and not isinstance(val, str):
                if hasattr(val, "_mock_name") or "Mock" in type(val).__name__:
                    setattr(self, field_attr, "mock_model" if "model" in field_attr else "mock_str")
                else:
                    setattr(self, field_attr, str(val))

        if self.skipped or self.skipped_by_fast_path:
            self.skipped = True
            self.skipped_by_fast_path = True

    def log_record(self) -> None:
        """Emit structured log for this call."""
        logger.info(
            "node=%s stage=%s status=%s tier=%s reasoning_effort=%s model=%s configured_model=%s resolved_model=%s fallback_used=%s fallback_reason=%s "
            "latency_ms=%.0f prompt_tokens=%s completion_tokens=%s total_tokens=%s retry_count=%d timeout=%s schema=%s path=%s skipped=%s cached=%s",
            self.node_name,
            self.logical_stage,
            self.status,
            self.model_tier or "N/A",
            self.reasoning_effort or "default",
            self.model_name,
            self.configured_model,
            self.resolved_model,
            self.fallback_used,
            self.fallback_reason,
            self.latency_ms,
            self.prompt_tokens if self.prompt_tokens is not None else "N/A",
            self.completion_tokens if self.completion_tokens is not None else "N/A",
            self.total_tokens if self.total_tokens is not None else "N/A",
            self.retry_count,
            self.timeout_status,
            self.schema_name or "None",
            self.execution_path,
            self.skipped_by_fast_path,
            self.cached,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "model_name": self.model_name,
            "configured_model": self.configured_model,
            "resolved_model": self.resolved_model,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "latency_ms": self.latency_ms,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retry_count": self.retry_count,
            "timeout_status": self.timeout_status,
            "budget_exceeded": self.budget_exceeded,
            "schema_name": self.schema_name,
            "founder_turn_id": self.founder_turn_id,
            "project_id": self.project_id,
            "iteration": self.iteration,
            "execution_path": self.execution_path,
            "skipped_by_fast_path": self.skipped_by_fast_path,
            "cached": self.cached,
            "skipped": self.skipped,
            "logical_stage": self.logical_stage,
            "model_tier": self.model_tier,
            "reasoning_effort": self.reasoning_effort,
            "status": self.status,
            "details": dict(self.details) if self.details else None,
        }


DEFAULT_LATENCY_BUDGETS_MS: dict[str, float] = {
    "diagnose_startup": 30000.0,
    "generate_experiment": 45000.0,
    "critique_experiment": 25000.0,
    "extract_experiment_evidence": 15000.0,
    "analyze_evidence": 30000.0,
    "assess_baseline_state_sufficiency": 10000.0,
    "plan_baseline_question": 10000.0,
    "extract_startup_state": 10000.0,
}


def check_latency_budget(node_name: str, latency_ms: float) -> tuple[bool, Optional[str]]:
    """
    Checks whether node latency exceeded the configured or default budget.
    Returns (exceeded: bool, warning_message: Optional[str]).
    """
    budget = DEFAULT_LATENCY_BUDGETS_MS.get(node_name)
    if budget is not None and latency_ms > budget:
        msg = f"LATENCY BUDGET EXCEEDED: node={node_name} took {latency_ms:.0f}ms (budget={budget:.0f}ms)"
        logger.warning(msg)
        return True, msg
    return False, None


@dataclass
class TurnTelemetrySummary:
    """Aggregated telemetry summary for an entire founder turn."""
    turn_id: int = 1
    total_duration_ms: float = 0.0
    turn_total_latency_ms: float = 0.0
    total_llm_calls: int = 0
    llm_call_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    slowest_node: str = "none"
    execution_path: str = "STRATEGIC_REASONING"
    nodes_visited: list[str] = field(default_factory=list)
    nodes_cached: int = 0
    nodes_skipped: int = 0
    records: list[NodeTelemetryRecord] = field(default_factory=list)

    def add_record(self, record: NodeTelemetryRecord) -> None:
        self.records.append(record)
        if record.node_name and record.node_name not in self.nodes_visited:
            self.nodes_visited.append(record.node_name)
        if record.cached:
            self.nodes_cached += 1
        if record.skipped or record.skipped_by_fast_path:
            self.nodes_skipped += 1
        if not record.skipped_by_fast_path and not record.cached and not record.skipped:
            self.llm_call_count += 1
            self.total_llm_calls += 1
            lat = record.latency_ms or record.duration_ms or 0.0
            self.turn_total_latency_ms += lat
            self.total_duration_ms += lat
            if record.prompt_tokens:
                self.total_prompt_tokens += record.prompt_tokens
            if record.completion_tokens:
                self.total_completion_tokens += record.completion_tokens
            if record.total_tokens:
                self.total_tokens += record.total_tokens
            elif record.prompt_tokens or record.completion_tokens:
                self.total_tokens += (record.prompt_tokens or 0) + (record.completion_tokens or 0)
            if lat > (max((r.latency_ms for r in self.records[:-1]), default=0.0)):
                self.slowest_node = record.node_name

    def log_summary(self) -> None:
        """Emit per-founder-turn aggregate log."""
        logger.info(
            "TURN_TELEMETRY | TURN_TOTAL_LATENCY=%.0fms LLM_CALL_COUNT=%d "
            "TOTAL_PROMPT_TOKENS=%d TOTAL_COMPLETION_TOKENS=%d SLOWEST_NODE=%s PATH=%s",
            self.turn_total_latency_ms,
            self.llm_call_count,
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.slowest_node,
            self.execution_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "total_duration_ms": self.total_duration_ms,
            "turn_total_latency_ms": self.turn_total_latency_ms,
            "total_llm_calls": self.total_llm_calls,
            "llm_call_count": self.llm_call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "slowest_node": self.slowest_node,
            "execution_path": self.execution_path,
            "nodes_visited": list(self.nodes_visited),
            "nodes_cached": self.nodes_cached,
            "nodes_skipped": self.nodes_skipped,
            "records": [r.to_dict() for r in self.records],
        }


def record_telemetry_event(
    state: dict[str, Any],
    node_name: str,
    model_name: str = "unknown",
    duration_ms: float = 0.0,
    latency_ms: Optional[float] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cached: bool = False,
    skipped: bool = False,
    skipped_by_fast_path: bool = False,
    execution_path: Optional[str] = None,
    retry_count: int = 0,
    timeout_status: bool = False,
    schema_name: Optional[str] = None,
    configured_model: Optional[str] = None,
    resolved_model: Optional[str] = None,
    fallback_used: bool = False,
    fallback_reason: Optional[str] = None,
    logical_stage: Optional[str] = None,
    model_tier: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    status: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> TurnTelemetrySummary:
    """Helper to construct, log, and register a NodeTelemetryRecord onto state['turn_telemetry']."""
    project_id = state.get("project_id", "<unknown>")
    iteration = state.get("iteration", 1)
    turn_id = state.get("answered_question_id") or state.get("pending_question_id")
    path = execution_path or state.get("execution_path") or "STRATEGIC_REASONING"
    lat = latency_ms if latency_ms is not None else duration_ms

    now = datetime.now(timezone.utc)
    st = start_time or now
    et = end_time or now

    budget_exceeded, _ = check_latency_budget(node_name, lat)

    record = NodeTelemetryRecord(
        node_name=node_name,
        model_name=model_name,
        configured_model=configured_model or kwargs.get("configured_model"),
        resolved_model=resolved_model or model_name or kwargs.get("resolved_model", "unknown"),
        fallback_used=fallback_used or kwargs.get("fallback_used", False),
        fallback_reason=fallback_reason or kwargs.get("fallback_reason"),
        start_time=st,
        end_time=et,
        latency_ms=lat,
        duration_ms=lat,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        retry_count=retry_count,
        timeout_status=timeout_status,
        budget_exceeded=budget_exceeded,
        schema_name=schema_name,
        founder_turn_id=turn_id,
        project_id=project_id,
        iteration=iteration,
        execution_path=path,
        skipped_by_fast_path=skipped or skipped_by_fast_path,
        cached=cached,
        skipped=skipped or skipped_by_fast_path,
        logical_stage=logical_stage,
        model_tier=model_tier,
        reasoning_effort=reasoning_effort,
        status=status or ("CACHED" if cached else "SKIPPED" if (skipped or skipped_by_fast_path) else "SUCCESS"),
        details=details,
    )
    record.log_record()

    # Aggregate onto state['turn_telemetry']
    summary = state.get("turn_telemetry")
    if not isinstance(summary, TurnTelemetrySummary):
        summary = TurnTelemetrySummary(execution_path=path)
        state["turn_telemetry"] = summary

    summary.add_record(record)
    return summary
