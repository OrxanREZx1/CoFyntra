# Startary M6.2.3 — State Integrity, Interrupt Safety, and Observability

**Scope.** This milestone fixes correctness in the existing sequential LangGraph workflow. It does not add parallel execution or change any prompt template, model assignment or Sol reasoning effort. No paid API calls were made, nothing was committed or pushed, and `deploy-alpha` was not touched.

**Outcome.**
- The founder-facing question is now composed exactly once, persisted before the interrupt, and displayed, resumed and recorded as that same text.
- Founder facts reach the risk evaluator.
- All eight first-turn LLM nodes emit consistent telemetry.
- The offline suite runs green (854 passed) and can no longer make paid calls by accident.

**Labels used in this document:**
- **Confirmed:** reproduced offline or proven by a test.
- **Projected:** estimated, not measured.

---

## 1. Issues, root causes, and fixes at a glance

| # | Issue | Root cause | Fix | Proof |
|---|---|---|---|---|
| 1 | The question composer ran twice (once before the founder saw the question, once on resume), and the transcript could store a different wording | The composer LLM call sat **inside** the interrupt node, before `interrupt()`. LangGraph re-executes an interrupted node from the top on resume and never commits what the node computed before `interrupt()` | New `compose_founder_question` node composes and persists the exact prompt **before** the interrupt. `ask_and_wait_for_founder` no longer calls any LLM | Confirmed: the old node reproduces the bug offline; the new tests prove 1 call before the interrupt, 0 on resume, and byte-identical text |
| 2 | `assess_startup_areas` saw `UNKNOWN` for facts already extracted | `merge_diagnosis` rebuilt `StartupState(raw_context, areas)` from scratch. Its `execution_context` and `progress` layers were reset to defaults on every turn | Field-wise fact preservation in `merge_diagnosis`, plus a read-time overlay in `assess_startup_areas` from the canonical fact stores | Confirmed by graph tests; mutation check: 5 tests fail if the defect is reintroduced |
| 3 | `answered_question_id` was silently lost (found during this milestone) | Written by `merge_founder_answer` and read by telemetry (`founder_turn_id`), but not declared in `StartaryGraphState`, so LangGraph dropped it | Declared the key | Confirmed by tests |
| 4 | Tests could not run, and 7 "offline" tests made live LLM calls | `pytest` (the `dev` extra) was not installed in the worktree `.venv`. Seven tests left an LLM node unmocked, and `.env` supplied a real key | Installed the dev extra into `.venv`. Fixed the isolation of the 7 tests. Added an autouse `tests/conftest.py` guard | Confirmed: 854 passed, 0 network attempts |
| 5 | Five first-turn nodes had no Startary telemetry | `record_telemetry_event` was never called in `plan_information_gap`, `assess_startup_areas`, `assess_weak_link_discrimination`, `select_weak_link_question` or `compose_advisor_response` | Uniform `_record_llm_node_telemetry` helper; new record fields: `logical_stage`, `model_tier`, `reasoning_effort` and `status` | Confirmed by tests over a real graph run |
| 6 | "zero paying customers" counted as paying traction | Plain substring keyword checks (`"paying" in text`) | Negation-aware `contains_unnegated_keyword`, applied at both affected sites | Confirmed: mutation check makes 2 tests fail |
| 7 | `can_formulate_testable_uncertainty` read a dead key | It read `startup_state["area_knowledge"]`, but diagnosis persists `areas` (`area_knowledge` exists only after an evidence cycle) | Reads `areas`, falls back to `area_knowledge`, and ignores areas diagnosed `UNKNOWN` | Confirmed by tests |

---

## 2. Root causes in detail

### 2.1 Duplicate question composition

The old flow was `select_weak_link_question → ask_and_wait_for_founder`. Inside `ask_and_wait_for_founder`, three things happened in order:

1. `compose_advisor_response(state)`, a fast-tier LLM call;
2. `display_question = advisor_resp.response_text`;
3. `interrupt({...display_question...})`.

`interrupt()` raises, so the node's return value (including the composed `latest_advisor_response`) was never committed. On `Command(resume=...)` LangGraph re-ran the node from the top. `latest_advisor_response` was still `None` (cleared by the previous `merge_founder_answer`), so the composer ran again. That second, regenerated response was the one returned and committed. `merge_founder_answer` then wrote it into `founder_messages` and the clarification histories.

**Offline reproduction (confirmed).** I ran the `HEAD` version of `ask_and_wait_for_founder` against the scripted PitchVision flow. Results:
- composer calls before the interrupt: 1;
- LLM calls on resume: `['AdvisorResponseLLMOutput']`;
- displayed text: `[composer call 1] …`; saved text: `[composer call 2] …`; match: **False**.

The real composer runs at temperature 0, but its output is not guaranteed to be identical across calls. The transcript could therefore differ from what the founder actually read.

A smaller related defect: the legacy fallback question id was `f"q_legacy_{int(time.time())}"`, which is not deterministic across re-execution.

### 2.2 Founder facts reset before risk evaluation

- **Ownership.** `merge_execution_state` owns the canonical fact stores: the top-level `execution_context` and `startup_progress`, each merged under provenance rules.
- **The reset.** `merge_diagnosis` built `StartupState(raw_context=raw_context, areas=areas)`. That dropped the existing `execution_context` and `progress` layers and replaced them with all-`UNKNOWN` defaults, on every turn, including turns where the diagnosis was served from cache.
- **Who read the reset layers.** `assess_startup_areas` serializes the whole `startup_state` into its prompt, so Sol rated decision risk without the team, budget, stage or traction facts. A second consumer, `domain/experiment.py:1096` (the founder experiment summary), read the same placeholder layers.

### 2.3 Test environment

- **Interpreter.** The worktree's `.venv` is correct: Python 3.11.5, with `startary` installed in editable mode from this worktree. Only the `dev` extra was missing.
- **Baseline.** Before any M6.2.3 change, run with the network blocked by hand: **801 passed, 7 failed**. All 7 failures were blocked live calls:
  - `test_m4122…::test_24` and `::test_25` patched `diagnose_startup` but not `assess_startup_areas`, which then called Sol.
  - `test_m414…::test_assess_weak_link_discrimination_determines_clear_winner_without_clarification`, ×5 fixtures: the test state omitted `weak_link_details`, so `max_score` defaulted to 1. That skipped the documented "without LLM call" fast path and called Sol.
- **Exposure.** With the real `.env` key, a normal `pytest` run billed these calls.

---

## 3. Implementation changes

| File | Change |
|---|---|
| `startary/graph/nodes.py` | **New:** `_resolve_active_question`, `_advisor_response_is_bound`, `_build_founder_prompt`, `_prompt_matches_active_question`, and the node `compose_founder_question`. **`ask_and_wait_for_founder`:** no LLM call; displays the persisted `pending_founder_prompt` (deterministic fallback: committed advisor text, else planner text); learning-input branch unchanged. **`merge_founder_answer`:** records the persisted prompt verbatim; consumes it (`pending_founder_prompt=None`) on all three branches. **`handle_conversational_repair`:** also persists its re-asked prompt. **`merge_diagnosis`:** `_preserve_fact_layer` keeps known facts (canonical known > previously mirrored known > default) and never downgrades to `UNKNOWN`. **`assess_startup_areas`:** reads `_startup_state_with_canonical_facts(state)`; traction adjustment uses `contains_unnegated_keyword`. **`assess_build_readiness`:** live-traction shortcut uses `contains_unnegated_keyword`. **Telemetry:** `_record_llm_node_telemetry`, called by the five uninstrumented nodes (success, failure and fallback). Discrimination's deterministic exits record `SKIPPED`. `interpret_founder_turn`, `extract_startup_state` and `diagnose_startup` records now carry tier, effort and status. `logical_stage` added to the token-retry wrapper's LangSmith metadata. Composer calls carry `run_name` and metadata (`node_name`, `logical_stage`, `attempt`, `model_tier`, `question_id`). |
| `startary/graph/builder.py` | Registered `compose_founder_question`. `plan_baseline_question → compose_founder_question`, `select_weak_link_question → compose_founder_question`, and `route_on_gap_plan`'s `"ask_and_wait_for_founder"` key now maps to `compose_founder_question` (the route key is unchanged so existing routing tests still apply). Added `compose_founder_question → ask_and_wait_for_founder`. Repair and experiment-failure edges still go straight to the interrupt node, because their text is already committed by the preceding node. |
| `startary/graph/graph_state.py` | Added `pending_founder_prompt` and `answered_question_id` (the latter was already written and read, but dropped because it was undeclared). |
| `startary/domain/telemetry.py` | `NODE_LOGICAL_STAGES` and `logical_stage_for`. New `NodeTelemetryRecord` fields: `logical_stage`, `model_tier`, `reasoning_effort`, `status` (included in logs and `to_dict`). `record_telemetry_event` accepts them and derives `status` (`SUCCESS`, `CACHED` or `SKIPPED`) by default. |
| `startary/llm/provider.py` | `get_llm_client_resolution_info` now also returns `reasoning_effort` and `model_tier`; the client records `_startary_model_tier`. Model and effort selection are unchanged. |
| `startary/domain/evidence.py` | `contains_unnegated_keyword`: negation is checked within the same sub-clause, so "zero revenue and 8 paying teams" still counts. A trailing zero is also treated as negation ("paying customers: 0"). |
| `startary/domain/gap_plan.py` | `can_formulate_testable_uncertainty` reads `areas`, falls back to `area_knowledge`, and ignores areas explicitly diagnosed `UNKNOWN`. |
| `tests/conftest.py` (new) | Autouse guard: unless a test is marked `live` **and** `STARTARY_RUN_LIVE_TESTS=1` is set, it uses a dummy OpenAI key, an unroutable base URL and LangSmith tracing off. |
| `tests/fixtures/scripted_llm.py`, `tests/fixtures/pitchvision_scenario.py` (new) | Offline scripted LLM that records every call (schema, tier, messages, run config), plus the PitchVision AI first-turn script used by the graph-level tests. |
| `tests/test_m623_*.py` (new) | 46 tests (Section 5). |
| `tests/test_m4122…`, `tests/test_m414…` | Isolation fixes only: the missing `assess_startup_areas` mock, and the `weak_link_details` that `select_weak_link` always provides in the graph. Assertions are unchanged. |
| `.venv` (not tracked) | Installed pytest 9.2.0, pytest-asyncio 1.4.1.post1 and pytest-mock 3.17.0 (the declared `dev` extra). Production packages and `pyproject.toml` are unchanged by M6.2.3. |

---

## 4. Question composition: before and after

**Before (M6.2.2 and earlier)**

```
TURN N (first pass)
  select_weak_link_question ──► ask_and_wait_for_founder
                                  ├─ compose_advisor_response()   LLM call #1 (fast)   ── NOT committed
                                  └─ interrupt(display = call #1 text)        ◄── founder sees call #1

RESUME (Command(resume=answer))
  ask_and_wait_for_founder  (re-executed from the top)
    ├─ compose_advisor_response()   LLM call #2 (fast)   ── regenerated wording
    └─ interrupt() returns answer  → returns latest_advisor_response = call #2
  merge_founder_answer
    └─ transcript records call #2 text   ✗ can differ from what the founder saw
```

**After (M6.2.3)**

```
TURN N
  select_weak_link_question / plan_information_gap / plan_baseline_question
      ▼
  compose_founder_question
    ├─ compose_advisor_response()   LLM call #1 (fast)   (skipped if a response is already bound to this question_id)
    └─ pending_founder_prompt = {question_id, question (exact display text), source, …}
      ▼   ── checkpoint committed here ──
  ask_and_wait_for_founder   (LLM-free; a pure function of committed state)
    └─ interrupt(display = pending_founder_prompt.question)   ◄── founder sees call #1

RESUME / RETRY / CHECKPOINT RESTORE / DOUBLE SUBMIT
  ask_and_wait_for_founder  (re-executed: reads the same committed prompt, 0 LLM calls)
      ▼
  merge_founder_answer
    └─ transcript records pending_founder_prompt.question   ✓ byte-identical to what was displayed
       prompt consumed (pending_founder_prompt = None)

CONVERSATIONAL REPAIR ("I don't understand")
  merge_founder_answer (repair) → interpret → handle_conversational_repair (1 rephrase call; persists its prompt)
      → ask_and_wait_for_founder (0 LLM calls) → the answer is attributed to the same question_id
```

---

## 5. Test results

**Command.** No external environment overrides are needed; the conftest guard provides the isolation.

```
.venv/Scripts/python.exe -m pytest -m "not live"
```

| Run | Result |
|---|---|
| Baseline, before M6.2.3 (network blocked by hand) | 801 passed, 7 failed (all 7 were blocked live LLM calls; see 2.3) |
| After M6.2.3 | **854 passed, 0 failed**, 5 deselected (opt-in `live` tests). No test attempted a network call. |

**Required tests, mapped to the code.** Classes in the first two rows are in `tests/test_m623_question_composition_integrity.py`; the rest are in `tests/test_m623_state_integrity_and_observability.py`.

| Required behaviour | Test(s) |
|---|---|
| 1. One question-generation call before the interrupt | `TestSingleCompositionAndExactTranscript::test_one_question_generation_call_before_interrupt`; `test_every_question_path_composes_once_and_never_on_resume[baseline\|strategic\|weak_link]` |
| 2. Zero regeneration calls on resume | `::test_resume_makes_zero_question_regeneration_calls` (zero LLM calls of any kind between resume and `merge_founder_answer`); `test_every_question_path…`; `TestNodeContracts::test_ask_node_never_calls_the_composer` |
| 3. Saved question equals displayed question | `::test_saved_question_exactly_matches_displayed_question`; `TestNodeContracts::test_merge_records_persisted_prompt_not_stale_advisor_text` |
| 4. Answer attaches to the correct question | `::test_founder_response_attaches_to_correct_question`; `TestCheckpointResumeSafety::test_conversational_repair_reasks_without_composer_and_attributes_answer` |
| 5. Checkpoint/resume never duplicates transcript entries | `TestCheckpointResumeSafety::test_checkpoint_restoration_resumes_without_composer_or_duplicates` (fresh graph, same checkpointer), `::test_retry_after_downstream_failure_does_not_duplicate_transcript` (simulated outage, then `/retry`), `::test_repeated_resume_of_same_checkpoint_is_idempotent` |
| 6. Extracted facts survive diagnosis merging | `TestPitchVisionFactPropagation::test_extracted_facts_survive_diagnosis_merging`, `::test_facts_persist_into_follow_up_turn_with_empty_extraction`; `TestMergeDiagnosisNeverDowngradesFacts` (3 tests) |
| 7. Risk assessment receives accurate known facts | `TestPitchVisionFactPropagation::test_risk_assessment_receives_accurate_known_facts` (inspects the exact JSON embedded in the prompt) |
| 8. "Zero paying customers" stays negative evidence | `test_paying_keyword_detection_respects_negation` (9 cases); `TestZeroPayingCustomersRemainsNegativeEvidence` (4 tests, each with a positive control) |
| 9. Unknown information is not invented | `TestPitchVisionFactPropagation::test_unknown_information_is_not_invented`; `TestMergeDiagnosisNeverDowngradesFacts::test_no_canonical_stores_means_no_injected_placeholders` |
| 10. Existing offline tests still pass | Full suite: 854 passed |
| Observability (Task 4) | `TestObservabilityCoverage` (5 tests: per-node fields across all 8 first-turn LLM nodes, no founder text in records, LangSmith metadata survives `sanitize_metadata`, `FAILED` status, fast-path `SKIPPED`) |
| Dead `areas` key (Task 5) | `TestTestableUncertaintyReadsAreas` (2 tests) |

**The graph-level tests are not source inspection.** They run the real compiled graph, MemorySaver checkpointer, `interrupt()` / `Command(resume=…)` and all deterministic nodes. Only `get_llm_client` is replaced, by a scripted LLM. The PitchVision script reproduces the live trace's 8 first-turn LLM stages in the same order.

**The tests detect the defects (confirmed by mutation checks):**
- Old interrupt node from `HEAD`: composer called on resume, and the saved text differs from the displayed text.
- Facts reset in `merge_diagnosis`: 5 tests fail.
- Plain substring traction keywords: 2 tests fail.

**PitchVision facts verified end to end**, in both `startup_state` and the risk-evaluator prompt, with provenance:

| Fact | Value | Provenance |
|---|---|---|
| Founders / active team | 2 | EXPLICIT |
| Hours per founder per week | 15 | EXPLICIT |
| Total weekly capacity | 30 | INFERRED |
| Budget | 1,500 | EXPLICIT |
| Customer access | Two amateur teams and three coaches | EXPLICIT |
| Development stage | Idea | INFERRED |
| MVP | No | EXPLICIT |
| Structured interviews / validation activity | None | EXPLICIT |
| Paying customers | 0 | EXPLICIT |
| Revenue | 0 | EXPLICIT |

Fields the founder never stated stay `UNKNOWN` with value `None`: funding, geography, audience, active users, launch, prototype and product status. In the areas, founder claims (hypotheses) and evidence remain separate lists. Raw founder text is retained in `raw_context`; in the weak-link loop it is the channel that carries the founder's answer to the evaluator.

---

## 6. Latency and cost effects

No live trace was run, so nothing below is a measured speed-up.

- **Follow-up turns (projected improvement).** Each answered question saves one fast-tier composer call, because the resume path no longer composes. In the M6.2.2 PitchVision trace, `ask_and_wait_for_founder` took 4.70 s on turn 1, and that node contained exactly one composer call. The removed resume call uses the same prompt template and similar inputs, so expect a saving of the same order at the start of every follow-up turn. That is projected, not measured. Each removed call also saves the composer's input tokens (about 2.2K tokens of fixed system prompt and schema, plus context).
- **First turn (no expected change in LLM work).**
  - Still 8 LLM calls. The composer moves from `ask_and_wait_for_founder` to `compose_founder_question`; in LangSmith the ~4.7 s now appears under the new node and the interrupt node drops to about 0.
  - The Task 2 fix grows the PitchVision `assess_startup_areas` input by **121 characters (about 30 tokens, +1.2%)**. That was measured offline on prompt size, not latency. It is small because the placeholder fields were already being sent; real values replace nulls.
  - One extra node and checkpoint write: milliseconds.
- **Indirect effect (unknown).** Sol now rates risk with real facts. That can change the selected weak link and therefore the number of clarification turns. Only the A/B planned in M6.2.2 §7 can measure this.

---

## 7. Remaining risks

1. **Risk-evaluator behaviour changes (intended, not yet validated).** `assess_startup_areas` now sees real team, budget, stage and traction facts. For PitchVision and the canonical fixtures, the selected weak link and discrimination outcome may differ from before. Run the M6.2.2 A/B (original PitchVision message, Sol vs. Sol before and after) before optimizing further.
2. **Founder experiment summary output changes.** `domain/experiment.py:1096` reads the same facts layer and will now include execution facts such as "team can build the prototype". This is correct, but the founder will see the change.
3. **Routing can change for founders without matching keywords.** `can_formulate_testable_uncertainty` now uses diagnosed areas, so a founder with KNOWN or PARTIALLY_KNOWN Target User and Problem areas can route to risk assessment instead of the insufficient-context stop, when turns run out or the gap planner declines to ask. This is the intended design. Areas diagnosed `UNKNOWN` are excluded.
4. **The negation helper is a heuristic.** It handles same-clause negation and trailing zeros. It can miss long-distance or past-tense constructions ("we used to have paying customers"). It is used only in two deterministic adjustments, and each has a positive-control test.
5. **In-flight threads at upgrade time.** A thread interrupted inside the old node and resumed by the new code cannot recover the composed text the founder saw, because it was never persisted. It records the planner question instead. Today's checkpointer is in-process MemorySaver, so this only matters within one running process.
6. **Two copies of the facts.** `startup_state.execution_context` and `.progress` are now a faithful mirror of the canonical stores, refreshed at `merge_diagnosis` and overlaid at read time in `assess_startup_areas`. Any new consumer should read the canonical top-level stores or `_startup_state_with_canonical_facts`.
7. **Retries hidden by the token wrapper.** Telemetry `retry_count` counts node-level attempts. Token-budget expansion retries inside `_invoke_structured_with_token_retry` show only in LangSmith (`:retry_token_expansion` spans and tags).
8. **Pre-existing serialization warnings (not changed).** LangGraph warns when deserializing custom enums and dataclasses from checkpoints (`WorkflowPhase`, `TurnTelemetrySummary`, and others), and a future strict-msgpack default would block them. `pending_founder_prompt` holds only plain types. Address this before adopting a persistent checkpointer for the web app.
9. **`pending_information_target` is also undeclared** in the state schema, so it is dropped as well. Nothing reads it today; it was left unchanged.

---

## 8. Verifying correctness in LangSmith

1. **Enable tracing locally.** In `.env`, set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=…` and `LANGSMITH_PROJECT=startary-ai-latency`. Keep `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true`.
2. **Run the original PitchVision first message** through `startary-dev` and answer the first question. These are paid calls; run them only when you choose to.
3. **In the `startary_turn_N` trace that ends at the first question:**
   - The node sequence ends `select_weak_link_question → compose_founder_question → ask_and_wait_for_founder`.
   - `compose_founder_question` has exactly one ChatOpenAI child, named `compose_advisor_response:call_1`, with metadata `node_name=compose_advisor_response`, `logical_stage=question_composition`, `model_tier=fast` and `question_id=q_wl_…`.
   - `ask_and_wait_for_founder` has **no** LLM child, and its duration is about 0.
4. **In the resume trace (the founder's answer):**
   - There is no ChatOpenAI run between `ask_and_wait_for_founder` and `interpret_founder_turn`.
   - Every `compose_advisor_response:*` run in this trace carries a `metadata.question_id` **different** from the answered one; it is the next question, if any.
5. **Count check.** Filter runs by `metadata.node_name = compose_advisor_response` and group by trace. The count should equal the number of questions displayed (1 per question, not 2).
6. **Stage coverage.** The LLM runs of `plan_information_gap`, `assess_startup_areas`, `assess_weak_link_discrimination` and `select_weak_link_question` carry `metadata.logical_stage` (`gap_planning`, `risk_assessment`, `weak_link_selection`, `question_planning`) together with `model_tier`, `resolved_model` and `reasoning_effort`. Sol shows the provider default, which is unchanged.
7. **Transcript check.** After the run, `founder_messages[1].content` in the final state must equal the text the CLI printed for the first question.
8. **Privacy check.** Inputs and outputs remain hidden. Run metadata contains only operational fields; `question_id` is an opaque identifier, not founder text.

Startary's own telemetry carries the same fields offline, without LangSmith. Each `turn_telemetry.records[i]` has `node_name`, `logical_stage`, `model_tier`, `resolved_model`, `reasoning_effort`, `latency_ms`, `retry_count` and `status`, and founder text is never recorded.
