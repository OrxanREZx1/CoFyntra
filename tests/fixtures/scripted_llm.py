"""
Scripted offline LLM for full-graph tests (Milestone 6.2.3).

Replaces ``startary.graph.nodes.get_llm_client`` so the real LangGraph topology, interrupt/resume
mechanics, and deterministic nodes run unchanged while every LLM call is answered by a test
responder and recorded (schema, tier, messages, run config). No network access is possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

Responder = Callable[[list[dict[str, Any]], "ScriptedLLM"], Any]


@dataclass
class LLMCall:
    schema: str
    tier: str
    messages: list[dict[str, Any]]
    config: dict[str, Any] | None
    output: Any = None


class ScriptedLLM:
    """Records every LLM call and answers it with the responder registered for its schema."""

    def __init__(self, responders: dict[str, Responder]):
        self.responders = dict(responders)
        self.calls: list[LLMCall] = []
        self.unscripted: list[str] = []

    def count(self, schema_name: str) -> int:
        return sum(1 for c in self.calls if c.schema == schema_name)

    def calls_for(self, schema_name: str) -> list[LLMCall]:
        return [c for c in self.calls if c.schema == schema_name]

    def get_llm_client(
        self,
        settings: Any,
        tier: str = "reasoning",
        model: str | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> "_ScriptedChatModel":
        # Mirrors provider defaults: fast tier runs at low effort, reasoning tier at the API default.
        effort = reasoning_effort or ("low" if tier == "fast" else None)
        return _ScriptedChatModel(self, tier, effort)

    def _answer(self, schema_name: str, tier: str, messages: list[dict[str, Any]], config: dict | None) -> Any:
        call = LLMCall(schema_name, tier, list(messages), config)
        self.calls.append(call)
        responder = self.responders.get(schema_name)
        if responder is None:
            self.unscripted.append(schema_name)
            raise AssertionError(f"Unscripted LLM call for schema {schema_name}")
        call.output = responder(messages, self)
        return call.output


class _ScriptedChatModel:
    def __init__(self, recorder: ScriptedLLM, tier: str, effort: str | None):
        self._recorder = recorder
        self._tier = tier
        self._startary_configured_model = f"scripted-{tier}"
        self._startary_resolved_model = f"scripted-{tier}"
        self._startary_fallback_used = False
        self._startary_fallback_reason = None
        self._startary_reasoning_effort = effort
        self._startary_model_tier = tier

    def with_structured_output(self, schema: type) -> "_ScriptedStructured":
        return _ScriptedStructured(self._recorder, self._tier, schema)

    def invoke(self, messages: list[dict[str, Any]], config: dict | None = None) -> Any:
        text = self._recorder._answer("TEXT", self._tier, messages, config)
        return SimpleNamespace(content=text)


class _ScriptedStructured:
    def __init__(self, recorder: ScriptedLLM, tier: str, schema: type):
        self._recorder = recorder
        self._tier = tier
        self._schema = schema

    def invoke(self, messages: list[dict[str, Any]], config: dict | None = None) -> Any:
        out = self._recorder._answer(self._schema.__name__, self._tier, messages, config)
        return out if isinstance(out, self._schema) else self._schema(**out)


def _field(messages: list[dict[str, Any]], label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", messages[-1]["content"], re.MULTILINE)
    return match.group(1).strip() if match else ""


def composer_echo(messages: list[dict[str, Any]], recorder: ScriptedLLM) -> dict[str, Any]:
    """
    Advisor composer responder that honours the planner's question_id / target (so validation
    passes) and stamps each response with its call number, making any regeneration visible.
    """
    call_no = recorder.count("AdvisorResponseLLMOutput")
    canonical = _field(messages, "Canonical Question")
    return {
        "question_id": _field(messages, "Question ID"),
        "next_question_target": _field(messages, "Information Target"),
        "next_question_rephrased": canonical,
        "next_question": canonical,
        "acknowledge": "Thanks, that is a clear picture.",
        "answer_to_founder": None,
        "diagnosis_delta": None,
        "should_ask_next_question": True,
        "response_text": f"[composer call {call_no}] Thanks, that is a clear picture. {canonical}",
    }
