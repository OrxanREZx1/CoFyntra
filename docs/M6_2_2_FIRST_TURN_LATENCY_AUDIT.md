# M6.2.2 — First-Turn Latency Architecture Audit

**Branch / worktree:** `ai-latency-langsmith` @ `a4db52e` + uncommitted M6.1/M6.2.1 changes (`Startary_AI_Latency`)
**Scope:** Code inspection only. No implementation changes, no paid API calls, nothing committed.
**Benchmark under study (measured, M6.2.1 LangSmith trace, PitchVision AI):** 77.46 s, ~24.48K tokens, $0.105, 8 LLM-bearing stages before the first clarification question.

**Research question:** How can Startary ask its first strategically useful clarification question much faster without weakening founder understanding or the quality of the eventual diagnosis?

**Short answer:** Most of the 77 s comes from five `gpt-5.6-sol` calls and three `gpt-5.6-luna` calls running **strictly one after another**. The code shows that several of them **do not depend on each other**. Running those calls concurrently, with **byte-identical prompts**, should remove roughly 29 s from the critical path (projected, not measured) without changing any model input. That makes this lever quality-neutral by design. Merging nodes or adding a "lightweight first-turn path" would save less time with more risk. The lightweight path as proposed would also drop the weak-link reasoning that makes the first question strategic.

Every statement below is labelled **[CONFIRMED]** (verified in code, or by an offline run with no API access) or **[HYPOTHESIS]** (plausible, needs measurement). Projections are labelled **[PROJECTED]**. They are arithmetic on the measured per-node times and are not measurements.

---

## 1. Current first-turn execution dependency diagram

The path the PitchVision trace actually took. Wiring is in `startary/graph/builder.py:486-565`.

```
founder message
  │
initialize_state ─ verify_state                                   (deterministic, ~0 s)
  │
interpret_founder_turn ............ LUNA (low)   6.97 s           nodes.py:5245
  │  route_after_interpret_founder_turn  (deterministic: repair text / phase)  builder.py:320
extract_startup_state ............. LUNA (low)  14.93 s           nodes.py:1590
  │
merge_execution_state                (deterministic provenance merge)        nodes.py:1843
assess_baseline_state_sufficiency    (deterministic fast path: all 5 dims known)  nodes.py:3515
  │  route_on_baseline → diagnose_startup                                    builder.py:129
diagnose_startup .................. SOL (API-default effort) 18.65 s      nodes.py:624
  │
merge_diagnosis                      (deterministic)                         nodes.py:859
  │  route_after_merge_diagnosis → plan_information_gap                      builder.py:148
plan_information_gap .............. SOL  7.16 s                            nodes.py:2025
  │  route_on_gap_plan → assess_startup_areas (should_ask=False or EXTERNAL_EVIDENCE)  builder.py:164
assess_startup_areas .............. SOL  8.68 s                            nodes.py:2233
  │
select_weak_link                     (deterministic Decision-Risk argmax)    nodes.py:2353
assess_weak_link_discrimination ... SOL  7.86 s  (fast path NOT taken → tie / gap<10)  nodes.py:3862
  │  route_on_weak_link → select_weak_link_question (should_clarify=True)    builder.py:251
select_weak_link_question ......... SOL  8.48 s                            nodes.py:4024
  │
ask_and_wait_for_founder .......... LUNA 4.70 s  (= compose_advisor_response inside the node)  nodes.py:1153/1205
  │  interrupt()  → founder sees the question
```

**Actual data dependencies (who reads whose output on turn 1)** [CONFIRMED]:

```
                 founder_messages / raw_context
        ┌──────────────┬─────────────┴──────────────┐
        ▼              ▼                            ▼
   interpret      extract ─► merge_exec ─► baseline gate          diagnose
        │                                    │ (routes only)        │
        │                                    └──────── gate ───────►│
        │                                                           ▼
        │                                                     merge_diagnosis
        │                                         ┌─────────────────┴────────────────┐
        │                                         ▼                                  ▼
        │                                plan_information_gap                 assess_startup_areas
        │                                  (routing bits only)                       │
        │                                                                   select_weak_link
        │                                                                            │
        │                                                                   discrimination
        │                                                                            │
        │                                                                   select_weak_link_question
        │                                                                            │
        └─────────────────────────────────────────────────────────────────► compose_advisor_response
```

Observations behind the diagram:

- `diagnose_startup` reads `raw_context`, `founder_messages`, `startup_state.areas` (empty on turn 1) and `founder_evidence`. It does **not** read `execution_context` or `startup_progress`, the outputs of extraction (nodes.py:698-759). Its only dependency on extraction is the baseline gate's **routing decision**.
- `interpret_founder_turn`'s LLM output does not affect routing. `route_after_interpret_founder_turn` reads `is_conversational_repair` (set from a deterministic text check, nodes.py:5495), `execution_path` and `phase`. The interpretation is consumed only by `compose_advisor_response` at the very end.
- `assess_startup_areas` reads `startup_state`, `diagnosis`, `diagnosis_output`, `raw_context` and `founder_messages`. It does **not** read `gap_plan`. `plan_information_gap` and `assess_startup_areas` depend only on diagnosis, not on each other.

Critical-path arithmetic from the trace: Sol calls = 50.83 s (65.6%); Luna calls = 26.60 s (34.4%). Node sum = 77.43 s ≈ 77.46 s wall clock, so the graph is fully serial.

---

## 2. Node-by-node responsibility and dependency table

| Node | Exact responsibility | Required inputs | Output (schema → state keys) | Downstream consumers (turn 1) | Model / effort / max tokens | Persisted? | Overlapping analysis | Needed before first question? |
|---|---|---|---|---|---|---|---|---|
| `interpret_founder_turn` | Classify intent of latest founder turn (question, concern, correction, goal) plus deterministic keyword overrides | `founder_messages[-6:]`, `pending_question`, `phase`, `raw_context[:300]`, `weak_area` | `FounderTurnInterpretationLLMOutput` (14 fields) → `latest_founder_turn_interpretation`, `founder_current_goal`, `founder_current_concern`, `recent_corrections` | `compose_advisor_response` (response mode, acknowledgment). **Not** routing | Luna, `reasoning_effort=low` (provider default for fast tier, provider.py:476), 800 tokens | Yes (checkpointed state) | None with strategy nodes. Repair detection is duplicated deterministically | Yes, but only by the time the composer runs. **Not** on the critical path logically |
| `extract_startup_state` | Extract operational facts (team, capacity, budget, access, product stage, users, revenue) with provenance | founder messages (first + last 2), current `execution_context`, `startup_progress`, `pending_question` | `StartupStateDeltaLLMOutput` (29 fields, 8.2K-char schema) → `startup_state_delta`, `extraction_status` | `merge_execution_state` → baseline gate, gap planner, composer fallback | Luna, low, 2500 tokens + 1.6× retry | Yes, via deterministic merge | `diagnose_startup` also reads the same founder text but outputs *strategic* areas, not operational facts | **Yes.** It is the fact-grounding and readiness-gating source |
| `merge_execution_state` / baseline gate | Provenance merge; 5-dimension readiness check | delta, existing state | `execution_context`, `startup_progress`, `fact_conflicts`, revision bump; `is_baseline_sufficient` | `route_on_baseline` | Deterministic (LLM only if a dimension is missing) | Yes | — | Yes (cheap) |
| `diagnose_startup` | 10-area epistemic status: understanding, claims, evidence, missing info, stage, summary | `raw_context`/`founder_messages`, known areas, evidence summary | `DiagnosisLLMOutput` → `diagnosis_output`; `merge_diagnosis` → `startup_state.areas`, `diagnosis` | Gap planner (summary), assess (areas + summary + stage), discrimination (summary), WL question (summary), composer (summary[:300]) | **Sol, effort not set → API default**, 3500 tokens | Yes; **cached** across turns by `strategic_state_revision == diagnosis_revision` (nodes.py:645) | Partial overlap with `assess_startup_areas` (both reason over all 10 areas) | Yes, for a *weak-link* question (see §4) |
| `plan_information_gap` | Decide: ask ONE strategic founder-knowledge question now, or proceed or hand off to TEST | diagnosis summary, full founder history, exec/progress JSON, previous questions | `InformationGapPlanLLMOutput` (13 fields) → `gap_plan`, `context_sufficiency`, `is_sufficient`, question/decision if asking | **Only `route_on_gap_plan`** on the proceed path. `recommended_area`, `missing_information` and `why_this_matters` are read by nothing downstream (only a composer fallback, nodes.py:5696) | Sol, API default, 2000 tokens | Yes | `resolution_mode` FOUNDER_KNOWLEDGE/EXTERNAL_EVIDENCE is judged again by discrimination. Its "primary uncertainty" is re-derived by assess + select | Its *routing decision* is needed; its analysis is discarded when proceeding |
| `assess_startup_areas` | Qualitative Importance / Uncertainty / Evidence-gap / Consequence for 10 areas | `json.dumps(startup_state)` (raw_context + areas + **default-UNKNOWN exec/progress**), diagnosis summary, stage | `StartupAreaAssessmentBatchLLMOutput` → `area_assessments_output` | `select_weak_link` (deterministic scoring) | Sol, API default, 2500 tokens | Yes | Uncertainty / evidence gap partly re-derive diagnose's status / evidence / missing info | Yes, for a weak-link question |
| `select_weak_link` | Decision Risk = I×U×G×C, deterministic tie-break | `area_assessments_output`, `startup_state.areas` | `weak_area`, `weak_link_details` (+ saturation telemetry), `area_assessments` | discrimination | Deterministic | Yes | — | Yes (cheap) |
| `assess_weak_link_discrimination` | Is the winner decisive? If not, can founder knowledge separate the tied areas? | score table, telemetry, diagnosis summary, WL Q&A history | `WeakLinkDiscriminationLLMOutput` → `selection_confidence`, `competing_weak_areas`, `should_clarify`, `recommended_information_goal` | `route_on_weak_link`, WL question (`recommended_information_goal`). **`missing_discriminating_information` is never read** (schemas.py:721) | Deterministic fast path (single winner, gap ≥ 10); otherwise Sol, API default, 2000 tokens | Yes | `resolution_mode` judgment duplicates the gap planner's in kind | Yes when scores tie (WL-001 saturation makes this common) |
| `select_weak_link_question` | Phrase ONE question that separates the competing areas | competing areas, info goal, diagnosis summary, full history | `WeakLinkClarifyingQuestionLLMOutput` → `active_next_turn_decision`, `current_weak_link_question` | composer (canonical question) | Sol, API default, 2000 tokens | Yes | The *what to ask* is already in discrimination's `recommended_information_goal`. The *wording* is rewritten again by the composer | Partly. See consolidation candidate O5 |
| `ask_and_wait_for_founder` → `compose_advisor_response` | Turn the planner's canonical question into the founder-facing message (acknowledge, intro summary, answer founder's question). Planner is the "sole authority" on content | interpretation, recent history, canonical question / target / qid, goal, `raw_context[:300]`, summary[:300] | `AdvisorResponseLLMOutput` → `latest_advisor_response` | Interrupt payload, `merge_founder_answer` (transcript text) | Luna, low, 1000 tokens; qid/target validation + 1 retry + deterministic fallback | **No on turn 1.** Runs before `interrupt()`, so the result is lost and recomputed on resume (§3, D1) | Re-phrases the Sol question | Yes (conversational layer) |

---

## 3. Confirmed duplicated reasoning and related defects

### D1 — `compose_advisor_response` runs twice per question [CONFIRMED, offline run]
`ask_and_wait_for_founder` calls the composer **before** `interrupt()` (nodes.py:1200-1219). LangGraph re-executes an interrupted node from the top on `Command(resume=...)` (cli.py:971), and the first result was never committed, so the composer runs again. Offline harness (stub composer, `MemorySaver`, langgraph 1.2.15): **1 call before resume, 2 after resume.**

- Latency: the extra call lands at the start of every follow-up turn. It is not part of the 77.46 s, but [PROJECTED] adds ≈ the measured 4.70 s to each next turn.
- Integrity: `merge_founder_answer` writes the *second* composition into `founder_messages` (nodes.py:1346-1373). The transcript can therefore record wording the founder never saw. Whether the two wordings actually differ is [HYPOTHESIS]; that they are generated independently is confirmed.

### D2 — The gap planner's analysis is discarded on the proceed path [CONFIRMED]
On PitchVision, `plan_information_gap` (7.16 s) routed to `assess_startup_areas`. When it proceeds, only `should_ask`, `strategic_sufficient`, `resolution_mode` and `can_formulate_testable_uncertainty` are used (builder.py:198-216). Its `recommended_area`, `missing_information` and `why_this_matters`, which together are its identification of the primary uncertainty, are **not passed** to `assess_startup_areas`, `assess_weak_link_discrimination` or `select_weak_link_question`. The weak link is re-derived from scratch.

### D3 — "Can founder knowledge resolve this?" is judged by two Sol calls [CONFIRMED in schema; partial overlap]
Both `InformationGapPlanLLMOutput` and `WeakLinkDiscriminationLLMOutput` carry `resolution_mode ∈ {FOUNDER_KNOWLEDGE, EXTERNAL_EVIDENCE, …}` and a should-ask flag. The scopes differ: strategic understanding versus tie-breaking between areas. Both read the same diagnosis summary. On PitchVision the first judged "no founder question needed now" and the second judged "ask the founder". [HYPOTHESIS] These are not contradictory, because different question types are involved, but it is the same kind of judgment made twice.

### D4 — Question content is produced in three stages [CONFIRMED]
`assess_weak_link_discrimination` decides *what information would discriminate* (`recommended_information_goal`, plus an unused `missing_discriminating_information`). Next, `select_weak_link_question` (Sol, 8.48 s) writes the canonical question. Finally, `compose_advisor_response` (Luna) rewrites it into the text the founder actually sees (`display_question = advisor_resp.response_text`, nodes.py:1210).

### D5 — The risk evaluator receives placeholder facts instead of extracted facts [CONFIRMED, offline run]
`merge_diagnosis` rebuilds `StartupState(raw_context=..., areas=...)` (nodes.py:922), so `startup_state.execution_context` and `.progress` are **default all-UNKNOWN objects**. The real extracted facts live in the top-level `execution_context` / `startup_progress` keys. `assess_startup_areas` serialises the whole `startup_state` (nodes.py:2247). Offline check: 0 non-UNKNOWN operational fields reach the prompt, about 4.2K chars (~1K tokens) of placeholders are sent, and the prompt also contains the verbatim `raw_context`.
- This is a grounding defect: the evaluator is told "MVP / users / revenue / budget: UNKNOWN" for a founder who stated them explicitly.
- [HYPOTHESIS] It may push Uncertainty / Evidence-gap toward HIGH and worsen WL-001 score saturation. That would make the discrimination fast path rarer and trigger the two extra Sol calls more often.
- Caution [CONFIRMED]: the verbatim `raw_context` inside this payload is **load-bearing**. During the weak-link loop, diagnosis is usually cache-reused, and the founder's answer reaches `assess_startup_areas` only through `raw_context` (merge_founder_answer appends it, nodes.py:1466-1470). Do not strip it without replacing that channel.

### D6 — Founder text is repeated across calls [CONFIRMED]
On turn 1 the founder message is sent to 7 of the 8 calls. It appears three times inside the interpret prompt (latest message, history, `startup_summary` slice) and up to three times in the composer prompt. Turn-1 extraction also sends about 4.2K chars of empty exec/progress JSON. Fixed overhead (system prompt + JSON schema) per call, as approximate tokens at chars/4: interpret 1754, extract 3111, diagnose 1379, gap 1916, assess 1017, discrimination 955, WL question 621, composer 2181, ≈ 12.9K in total. [HYPOTHESIS] Input tokens are a cost lever more than a latency lever for reasoning models, where decode and reasoning dominate. Their latency effect must be measured.

### Other confirmed findings that matter for any fix
- **D7** `can_formulate_testable_uncertainty` reads `startup_state["area_knowledge"]` (gap_plan.py:159), but `merge_diagnosis` writes `areas`. The area-based checks are therefore dead code and the helper falls back to keyword lists only. Any "deterministic readiness evaluation" that relied on this helper would be weaker than it appears.
- **D8** Startary's own `turn_telemetry` covers only 3 of 8 first-turn calls. `plan_information_gap`, `assess_startup_areas`, `assess_weak_link_discrimination`, `select_weak_link_question` and `compose_advisor_response` never call `record_telemetry_event`. LangSmith is currently the only complete per-node latency source.
- **D9** Reasoning-tier nodes never set `reasoning_effort`, so they run at the API default (provider.py:474-477, 509). Startary has no explicit effort control for Sol nodes except in the token-retry path, which forces `low`.
- **D10** (latent, not hit by PitchVision) The traction adjustment in `assess_startup_areas` matches the bare keyword `"paying"` (nodes.py:2283), so "**zero** paying customers" counts as paying evidence. It is harmless for PitchVision only because no "live" keyword is present.
- **D11** No reducers on `traces` / `turn_telemetry` (graph_state.py:107-108, 173). Every node does read-modify-write. Naive LangGraph fan-out would raise `InvalidUpdateError` or lose entries. This constrains *how* to parallelise (§5).

---

## 4. Nodes that must remain separate, and why

| Keep separate | Reason grounded in code |
|---|---|
| `extract_startup_state` ≠ `diagnose_startup` | Different targets: operational facts with EXPLICIT/INFERRED provenance merged deterministically (`validate_state_grounding`, conflict detection) versus strategic area status. The deterministic baseline gate and the "LLM never mutates durable state" contract (state.py docstring) depend on the extractor's narrow schema. Merging them would put fact grounding inside a strategic reasoning call. |
| `interpret_founder_turn` ≠ `extract_startup_state` | Intent is explicitly modelled "separately from factual state extraction" (nodes.py:5248). Intake-is-not-evidence sanitisation (nodes.py:5372-5388) and repair routing rely on it. They should run **concurrently**, not merge. |
| `diagnose_startup` ≠ `assess_startup_areas` | Cache granularity. During the weak-link loop `route_after_merge_diagnosis` skips the gap planner, diagnosis is normally reused (revision unchanged), and only `assess_startup_areas` re-runs with the new answer. A merged call would force a full 10-area re-diagnosis on every loop turn and move latency into turn 2. The split also keeps "LLM gives qualitative levels, Python computes Decision Risk". |
| `select_weak_link` (deterministic) | Core invariant. Already poison-pill tested (`test_m4141_deterministic_nodes_audit.py`). |
| `assess_weak_link_discrimination` gate | Its deterministic fast path (nodes.py:3897) is the only thing that can skip two Sol calls. It must stay as a separate gate, though it can absorb question wording (O5). |
| `compose_advisor_response` | "Planner is sole authority, composer only phrases" (qid/target validation, nodes.py:5784-5817). Keep it, and fix D1. |
| `plan_information_gap` | Needed for founders whose context is NOT_YET_FORMULATABLE or who have founder-knowledge gaps. Its latency can be hidden by concurrency (O3) without changing its decision. Skipping it conditionally (O9) is strictly worse than O3. |

---

## 5. Proposed optimizations, ranked

"Input-identical" means every LLM call receives exactly the same messages as today. Quality cannot change by construction; only plumbing can break. Savings are **[PROJECTED]** from measured node times and assume per-call latency is unchanged under concurrency, which is itself an assumption to verify.

| Rank | Change | Projected turn-1 saving | Quality risk | Effort | Notes |
|---|---|---|---|---|---|
| **O1** | **Parallel intake: `interpret_founder_turn` ∥ `extract_startup_state`** (UNDERSTAND phase, non-repair turns) | −6.97 s → ~70.5 s | **None** (input-identical) | M | Run both inside one wrapper node using `langchain_core`'s `ContextThreadPoolExecutor`, so LangSmith child spans stay attached. Merge `traces` / `turn_telemetry` deterministically (D11). Repair detection stays a deterministic pre-check. Also benefits every later UNDERSTAND turn. |
| **O2** | **Speculative `diagnose_startup` ∥ extraction** when no diagnosis exists yet (turn 1) | intake becomes max(6.97, 14.93, 18.65) = 18.65 s instead of 40.55 s → ~55.5 s cumulative | **None** when committed (input-identical) | M | Commit only if the baseline gate passes **and** extraction ≠ FAILED. Otherwise discard: a baseline answer can change *Founder Constraints*, and committing would poison the revision-based cache. Set `diagnosis_revision` to the **post-merge** `strategic_state_revision`: PitchVision's explicit "zero revenue" bumps it to 2. Cost: one wasted Sol call on baseline-incomplete turns. |
| **O3** | **Speculative `assess_startup_areas` ∥ `plan_information_gap`** | −7.16 s on the proceed path → ~48.4 s cumulative | **None** (input-identical; neither reads the other) | M | If the gap planner asks, discard the assessment (the strategic answer will trigger re-diagnosis). Cost: one wasted Sol call when the planner asks. Dominates O9. |
| **O4** | **Fix D1: compose once, outside the interrupt node** (split into a compose node before `ask_and_wait_for_founder`, or persist before `interrupt()`) | 0 s on turn 1; [PROJECTED] ≈ −4.7 s on every follow-up turn | None; *improves* transcript integrity | S | All edges into `ask_and_wait_for_founder` (baseline, gap, weak-link, repair, experiment failure) need the same treatment. |
| O5 | Let discrimination emit the discriminating question when `should_clarify=True` (it already produces `recommended_information_goal`); drop `select_weak_link_question` | −8.48 s when clarifying → ~39.9 s cumulative | **Medium**: schema + prompt change on a Sol node | M | Justification: D4. Keep the current node as fallback if the field is empty. Needs quality A/B. |
| O6 | Fix D5: give `assess_startup_areas` the real `execution_context` / `startup_progress` (compact) instead of placeholders, keeping `raw_context` | Unknown. [HYPOTHESIS] fewer saturated ties → more discrimination fast paths (−16.3 s when it hits) | **Medium-high**: changes weak-link inputs | S code / L validation | This is a grounding *fix* first and a latency lever second. Must be A/B'd on weak-link stability. |
| O7 | Explicit `reasoning_effort` for Sol nodes (A/B `low` vs default on discrimination, WL question, gap planner) | Unknown | Medium | S / M | The prior Sol-vs-Luna test covered **model** substitution, not effort level. No evidence either way. Do **not** move `assess_startup_areas` or `plan_information_gap` to Luna: the M4.1.4.1 live test showed 60% primary-weak-link divergence. |
| O8 | Prompt slimming: no empty JSON on turn-1 extraction, de-duplicate founder text in interpret / composer | Small. [HYPOTHESIS] mostly cost | Low | S | Do not touch the `raw_context` channel (D5 caution). |
| O9 | Conditionally skip `plan_information_gap` on rich first turns | −7.16 s | Medium-high | M | **Not recommended**: O3 gives the same saving at zero decision risk. |
| O10 | Merge diagnose + assess | ≈ −8 s on turn 1, slower turn 2 | High | L | **Not recommended** (§4). |

Cumulative O1+O2+O3 [PROJECTED]: 77.46 s → ~48.4 s (−37%) with no change to any prompt, model, schema or routing decision. Adding O5 brings it to ~39.9 s, at a quality risk that needs A/B testing.

### C. Evaluation of the proposed lightweight first-turn pathway
*Founder message → understand/update state → deterministic readiness → highest-priority missing evidence → ask.*

- **[CONFIRMED] It already exists for incomplete founders.** Extraction → deterministic baseline gate → `plan_baseline_question` → ask. No diagnosis runs. The 77 s path occurs precisely *because* PitchVision's founder supplied all five baseline dimensions, so the remaining gap is strategic.
- **For rich founders it would change the product's behaviour.** The PitchVision question is a *weak-link discriminating* question. It exists because 10-area risk scoring produced a tie, and it is designed to separate competing bottlenecks. A deterministic readiness evaluator can rank missing **fields**, not strategic **risk**. The gap-planner rules explicitly forbid asking about fields "merely because they are UNKNOWN" (prompts.py:672), and the deterministic helper it would lean on is partly dead code (D7). The pathway would preserve goals 1 (facts) and 2 (missing evidence) but weaken goals 3–5 (stage, weak link, strategic value of the question).
- **It would mostly move latency, not remove it.** After the founder answers, the loop re-runs `assess_startup_areas` and discrimination anyway (builder.py:156), and diagnosis would then run for the first time. Turn 2 would absorb the deferred ~45 s of Sol work.
- **Verdict:** do not adopt it as the default. Keep full strategic reasoning before the first weak-link question, and restructure it for concurrency (O1–O3), which achieves most of the latency goal without changing any decision. [HYPOTHESIS, optional UX] A deterministic "here is what I captured" acknowledgment after extraction (~15 s) could cut *perceived* wait without touching the reasoning chain. That is a product decision, not an architecture one.

---

## 6. Recommended first implementation change

**O1+O2 as one "parallel intake" change, plus O4 as a separate small bug fix.**

1. **Commit A (O1):** a wrapper intake step for UNDERSTAND-phase, non-repair turns that runs `interpret_founder_turn` and `extract_startup_state` concurrently via `ContextThreadPoolExecutor`. Node functions, prompts and routers stay unchanged. `traces` are merged in a fixed order (interpret, extract) and telemetry events are merged explicitly.
2. **Commit B (O2), behind a setting such as `STARTARY_SPECULATIVE_DIAGNOSIS`, default off until A/B'd:** add `diagnose_startup` to the same pool when `diagnosis_output` is absent. Commit it only after `merge_execution_state` and a passing baseline gate, with the revision aligned to the post-merge value. Otherwise discard it.
3. **Commit C (O4):** compose the advisor response before the interrupt node so it runs once and the transcript matches what was displayed.

Why this first: it is the largest critical-path reduction available ([PROJECTED] ~22 s on turn 1) that changes **no model input**. Its correctness can therefore be proven offline with prompt-hash equality tests before spending anything on live runs. O3 follows naturally with the same pattern.

Prerequisites, both small and both instrumentation or correctness only: add `record_telemetry_event` to the five uninstrumented nodes (D8), so A/B results don't depend on LangSmith alone, and fix the D7 key mismatch.

---

## 7. Safe A/B benchmarking plan (original PitchVision context)

1. **Freeze the input.** Use the *exact* first message from the M6.2.1 live run. LangSmith hides inputs (`LANGSMITH_HIDE_INPUTS=true`), so recover it from your local CLI session or notes. The fixture in `tests/test_pitchvision_state_integrity.py` is a fallback but is **not guaranteed identical** to the live text. Store it as a versioned fixture file.
2. **Arms.** A = current HEAD + M6.2.1 working tree. B = exactly **one** change (O1, then O1+O2, then O3…). Never bundle untested changes into one arm.
3. **Offline gate before any paid run.** With a recording mock LLM, assert that arm B sends byte-identical `messages` per node to arm A (SHA-256 of serialized messages) for O1–O4. If hashes match, quality equivalence for those changes is proven without live calls.
4. **Live runs (needs your explicit approval, since they are paid).** At ≥5 paired runs per arm, interleaved A,B,A,B to cancel API-latency drift, the cost is [PROJECTED] ~$1 at the measured $0.105 per first turn. Use the same models, token caps and `.env`. Tag each run via `build_trace_config(extra_metadata={"ab_arm": "A"|"B", "scenario": "pitchvision_v1"})`, which goes through the existing sanitiser.
5. **Metrics.** Wall-clock to first interrupt (primary), per-node span duration, number of LLM calls, total and reasoning tokens, cost, wasted speculative calls. Report **median and range**. With N=5, do not claim statistical significance.
6. **Quality comparison (required for O5–O7; a sanity check for O1–O4).** EXPLICIT fact set after turn 1, `startup_stage`, the 10 area statuses, `weak_area`, top-3 risk set, `selection_confidence`, `should_clarify`, the question's target area, and a blind human rubric (strategic value, grounding, CV-001 compliance, no premature recommendation).
7. **Cost-of-speculation scenarios.** Add one baseline-incomplete fixture (measures O2 waste) and one fixture where the gap planner asks (measures O3 waste) from `tests/fixtures/scenarios.py`.
8. **Turn-2 check.** Answer the first question with a fixed reply in both arms to confirm D1/O4 removes the resume-time composer call and that diagnosis cache behaviour is unchanged.

---

## 8. Quality regression tests needed

Offline (mocked LLM, no network):

1. `test_parallel_intake_prompts_identical`: interpret / extract / diagnose receive byte-identical messages in serial and parallel modes.
2. `test_parallel_intake_state_merge`: all three traces present in deterministic order, telemetry events for all nodes, no `InvalidUpdateError`.
3. `test_parallel_intake_langsmith_parent_linkage`: child LLM runs keep the parent run context (callback-handler spy).
4. `test_repair_turn_skips_parallel_intake` and `test_waiting_for_evidence_not_parallelised`: existing routes unchanged.
5. `test_speculative_diagnosis_discarded_when_baseline_insufficient`: no `diagnosis_output` / `diagnosis_revision` committed. The next turn re-diagnoses *with* the baseline answer.
6. `test_speculative_diagnosis_discarded_when_extraction_failed`: facts not mutated (existing invariant), diagnosis not committed.
7. `test_speculative_diagnosis_revision_alignment`: PitchVision's explicit zero revenue bumps the revision. The committed `diagnosis_revision` equals the post-merge value, with no redundant re-diagnosis and no stale reuse next turn.
8. `test_speculative_assess_discarded_when_gap_planner_asks` / `test_precomputed_assess_used_on_proceed`.
9. `test_composer_called_once_per_question`: interrupt + `Command(resume=...)` → exactly one composer call (the offline harness used in this audit, turned into a test).
10. `test_transcript_records_displayed_question`: the `startary` message appended by `merge_founder_answer` equals the interrupt payload's `question`.
11. `test_pitchvision_explicit_facts_preserved`: 2 co-founders, 15 h each, $1,500, 2 teams + 3 coaches access, no MVP, 0 paying, 0 revenue, no interviews, all EXPLICIT after turn 1 in both modes.
12. `test_pitchvision_stage_purity`: PitchVision is never treated as traction stage. No "Adjusted for verified paying customers" rationale, so D10 cannot fire.
13. Existing suites must stay green: `test_weak_link_routing`, `test_weak_link_discrimination`, `test_information_gap_planner`, `test_uncertainty_handoff`, `test_m413_latency_and_efficiency` (diagnosis cache test 42), `test_m4141_deterministic_nodes_audit`, `test_langsmith_observability`, `test_token_budget_recovery`.

Only if O5/O6 are pursued:

14. `test_discrimination_question_present_when_clarifying` (one question, CV-001 compliant), with fallback to `select_weak_link_question` when empty.
15. `test_assess_areas_receives_extracted_facts`: no all-UNKNOWN placeholder block, and `raw_context` (the weak-link answer channel) is still present.
16. Live, gated by `STARTARY_RUN_LIVE_TESTS=1`: weak-link top-3 containment and primary-match rate versus arm A across the 5 canonical fixtures, using the same harness as `test_m414_sol_vs_fast_equivalence.py`.

> Note: the existing suite was **not** run during this audit. `pytest` is not installed in this worktree's `.venv` (it is only in the `dev` extra), and the environment was left unmodified.

---

## Confirmed findings vs hypotheses (summary)

**Confirmed:**
- The first-turn graph is fully serial (node sum 77.43 s ≈ wall 77.46 s).
- interpret ⟂ extract, diagnose ⟂ extraction outputs (only the baseline route links them), and gap planner ⟂ assess.
- D1 composer double call (offline run).
- D2 gap analysis discarded on the proceed path.
- D3 / D4 duplicated judgments and question wording.
- D5 placeholder facts in the risk-evaluator prompt (offline run).
- D6 repeated payloads.
- D7 dead `area_knowledge` key; D8 telemetry gaps; D9 no effort control on Sol; D10 negation-blind "paying" keyword; D11 no state reducers.

**Hypotheses (need measurement):**
- Concurrency leaves per-call latency unchanged.
- D5 contributes to WL-001 saturation.
- `reasoning_effort=low` on selected Sol nodes preserves decisions.
- Input-token slimming has a material latency effect.
- The two composer wordings differ in practice.
- The perceived-latency UX acknowledgment.

**Projected, not measured:** every seconds figure attributed to a change in §5–§6.
