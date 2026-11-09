# Startary M6.2.5 — Experiment Loop Correctness and Observability

**Scope.** This milestone fixes correctness, state-persistence and observability defects in the existing experiment loop.

**Unchanged:** the loop architecture (one generation, one critique, at most one regeneration, then the failure stop); model assignments; Sol reasoning effort. No parallelism was added.

**Not done:** no paid API calls, no commit or push. `deploy-alpha` was not touched.

**Not claimed:** any latency improvement. That needs a new live run (Section 7).

**Result.** **953 offline tests pass** (baseline 854), and no test attempted a network call. For every defect fixed here, putting the old behaviour back in place makes the new tests fail (Section 2).

---

## 1. Root cause and fix for each confirmed defect

| # | Defect | Root cause | Fix (files) |
|---|---|---|---|
| 1 | Critic decisions invisible in LangSmith and in local telemetry | Inputs and outputs are hidden; no verdict, dimension flags or issue data were recorded anywhere | Structured decision record (Section 3) attached to the `critique_experiment` telemetry record and emitted as a sanitized LangSmith child span `experiment_critic_decision` under the critic node. `revision_round` is added to the generator's and critic's LLM run metadata (`nodes.py`, `domain/telemetry.py`, `observability/langsmith.py`) |
| 2 | Verdict read by substring | `"ACCEPT" in verdict` accepted "ACCEPTED" or "accept with changes"; `verdict` was a free `str`, and the six flags were ignored | Schema uses the strict enums `CriticVerdict` / `CriticDimensionResult` (case and whitespace are normalized, nothing else is accepted). Fail-closed interpretation: **ACCEPT only when verdict == ACCEPT and all six dimensions PASS.** One routing rule, `experiment_critique_route`, is shared by the router and telemetry (`llm/schemas.py`, `domain/experiment_contract.py`, `graph/builder.py`) |
| 3 | `experiment_design_status`, `active_execution_handoff`, `execution_handoff_text` and `waiting_response` dropped | Written and read, but not declared in `StartaryGraphState`, so LangGraph silently drops them | Declared as typed fields (`graph/graph_state.py`) |
| 4 | `customer_access_assessment` written but never read | A derived value recomputed by `resolve_customer_access`; it has no readers | Write removed rather than adding a redundant state field (`nodes.py: assess_can_execute`) |
| 5 | Dead budget pre-check | The check read `exec_ctx["monthly_budget"]`; the fact store uses `available_budget_amount` | Shared `compare_cost_to_budget` reads `available_budget_amount` (and `budget_currency`). Money parsing is anchored to currency markers: "$1,500" is 1500, and in "$50 over 2 weeks" the 2 is ignored. **An unknown budget stays unknown.** A cost range exceeds the budget only when its lower bound does. Different currencies give an advisory, not a rejection. `assess_can_execute` uses the same parser; its old `\$?\b(\d+)\b` regex read "$1,500" as 1 |
| 6 | Dead assumption-ID check | The generator schema has no `assumption_id` | `generate_experiment` now stamps each draft with the active assumption's ID. Because code, not the model, writes this stamp, a mismatch means the draft is stale and gives `ASSUMPTION_MISMATCH`. The validator still binds `hypothesis` to the active statement |
| 7 | Validator errors made `/retry` a dead end | `validate_proposed_experiment` raised `ValueError`; `/retry` re-validated the same committed draft | The validator only normalizes and never raises for draft defects. The critic's contract check turns them into a **deterministic REVISE** (no LLM call) inside the bounded loop |
| 8 | BUILD/MVP contradiction | BUILD forced `PROTOTYPE_TEST`, and the pre-check rejected every `PROTOTYPE_TEST` without an MVP. It also checked only `mvp_exists` and listed the non-existent types `MVP_TEST` and `USABILITY_TEST` | `assess_prototype_requirement` separates testing an **existing** prototype, MVP or launched product from **building** a bounded one. Building is allowed only when build readiness is `BUILD_MINIMAL_PROTOTYPE` **and** `minimum_build_scope` is not empty; the scope and must-not-build list are passed to the critic. `assess_can_execute` uses the same rule |
| 9 | Critic could make 4 Sol calls on token exhaustion | The outer loop broke only on permanent errors, so the second outer attempt repeated the base call and its expansion | Failures are classified with `classify_llm_failure` and handled per kind (Section 5). The critic and the generator now make at most **2** calls when tokens are persistently exhausted |
| 10 | Invalid structured output was retried with an identical prompt | No feedback on the retry | The retry adds a corrective message that names **only the failing schema fields** (`schema_correction_message`) |
| 11 | No privacy or consent safeguard | Nothing checked recording or personal data | A deterministic check (`check_privacy_consent`), plus the requirement made explicit to the generator and the critic (Section 6) |

**Related small fixes:**
- **Sample size field:** `assess_can_execute` read `sample_size`, which doesn't exist, instead of `target_sample_size`.
- **Empty revision instructions:** when the critic's REVISE carried no issues or instructions, the regeneration received no revision section at all. It now falls back to the failed dimensions, then to the critic's rationale.
- **Logical stages:** `generate_experiment` and `critique_experiment` now have telemetry stages (`experiment_design` and `experiment_critique`).

**New shared module:** `startary/domain/experiment_contract.py` holds every deterministic rule (contract checks, budget, prototype, privacy, verdict interpretation, routing, metadata), so the critic node, the validator, the router, `assess_can_execute` and telemetry can't disagree.

**Violations vs. advisories:**
- **Violations** (objective, reliable): `REQUIRED_FIELD_EMPTY`, `ASSUMPTION_MISMATCH`, `INVALID_SAMPLE_SIZE` (≤ 0 only; no upper limits), `COST_EXCEEDS_BUDGET`, `PROTOTYPE_NOT_AVAILABLE`, `BUILD_SCOPE_UNDEFINED`, `CONSENT_STEP_MISSING`, `GUARDIAN_CONSENT_MISSING`. These produce a deterministic REVISE.
- **Advisories** (heuristic or contextual): `BUDGET_UNKNOWN`, `COST_NOT_COMPARABLE`, `BUILD_AUTHORIZED_WITHIN_SCOPE`, `POSSIBLE_MINORS`, `PERSONAL_DATA_PRIVACY`. These never block; they go to the LLM critic as "DETERMINISTIC CHECK NOTES".

**Targeted prompt changes** (versions bumped to `experiment_generator_v2` and `experiment_critic_v2`):
- a PRIVACY & CONSENT section in both system prompts;
- critic guidance on deterministic notes and authorized building;
- an explicit strict verdict rule;
- an optional notes section in the critic's message, added only when there are advisories.

The generator's system prompt grows by about 169 tokens and the critic's by about 216. The strict-enum schema adds about 45 tokens.

---

## 2. Tests added or updated

**New: `tests/test_m625_experiment_loop_correctness.py` (98 tests).** The graph tests run the real compiled graph and MemorySaver. Each is seeded with `update_state(..., as_node="decide_next_strategic_action")`, so the actual generate → validate → critique → route → save → can_execute → handoff / failure-stop path runs, with only the LLM scripted.

| Required area | Tests |
|---|---|
| 1. Graph-state persistence | Every key written by the loop's nodes is declared. Handoff and waiting reply survive transitions, a checkpoint restore into a fresh graph, and resume; the interrupt shows the composed handoff, then the advisor's reply. `can_render_waiting_for_evidence` and terminal status `WAITING_FOR_EVIDENCE` are now reachable. Failure status persists; the terminal shows `EXPERIMENT_DESIGN_NEEDS_REVISION`; founder learning input clears the status and the counter |
| 2. ACCEPT / REVISE routing | Route table, including "ACCEPTED", "accept with minor changes" and None. Graph bounds for accept-first (1+1 calls), revise-then-accept (2+2) and revise-twice (2+2, then failure stop). Round 2 receives the critic's instructions with `revision_round = 2` |
| 3. Six-dimension interpretation | ACCEPT with all PASS; ACCEPT with a FAIL becomes REVISE (conflict); REVISE with all PASS stays REVISE (conflict); a missing flag never accepts; the schema rejects non-token values; node-level actionable instructions when flags conflict |
| 4. BUILD/MVP | 24-case matrix: {BUILD, BUILD without scope, CONTINUE_MANUAL, none} × {no artifact, prototype, MVP} × {PROTOTYPE_TEST, PROBLEM_INTERVIEW}. Graph: authorized BUILD reaches the LLM critic with scope notes, executes, and can_execute raises no false "no prototype" requirement. Prototype test with no artifact under manual mode gives a deterministic REVISE |
| 5. Budget | Known (within / over), zero budget, zero cost, unknown, `UNKNOWN` provenance, currency mismatch, range lower bound, "$50 over 2 weeks", unparseable text; thousands separators; violation vs. advisory; can_execute parsing |
| 6. Assumption alignment | Draft stamped with the assumption ID; a stale stamp gives a deterministic REVISE with no LLM call; hypothesis bound to the statement |
| 7. Validator / `/retry` | A whitespace draft is a bounded REVISE, not a graph error; a blank draft followed by a good one recovers, and the revision names the defect; invalid structured output retries once with schema feedback |
| 8. Retry bounds | Failure classification table; critic 2 calls (was 4) and generator 2 under token exhaustion; transient error retried once unchanged; permanent error not retried; deterministic violation never reaches the LLM |
| 9. Telemetry and sanitization | Decision records for both rounds; `sanitize_metadata` keeps every field unchanged; a sentinel founder phrase and critic prose are absent from telemetry and spans; the LangSmith span is a child of `critique_experiment`; deterministic decisions are observable |
| 10. Privacy and consent | An ordinary interview (including "video interviews" and "record whether…") has no findings. Filming, uploads, screenshots and audio recordings without consent are violations. Consent present leaves only the minors advisory. Known minors require guardian authorization. Personal data is advisory only. The critic's instruction is specific. Both prompts carry the requirement without claiming compliance |
| 11. PitchVision grounding | Generator and critic payloads contain the PitchVision facts (2 founders, $1,500, two teams and three coaches, no MVP, 0 paying, 0 revenue, 0 interviews); unknown facts are absent; the $1,500 budget is enforced |
| 12. M6.2.3 protections | The full suite, including `test_m623_*`, still passes |

**Updated existing tests (inputs or expectations only, with reasons in comments):**
- **`test_experiment_generator.py`:** three tests that expected `ValueError` are replaced by one parametrized test (4 cases). It asserts the defect becomes a deterministic REVISE with no critic LLM call (defect 7).
- **`test_m413_latency_and_efficiency.py::test_44`:** now uses a complete draft. Its partial draft is now, correctly, a deterministic REVISE before the critic LLM; the test's purpose (summary slimming) is unchanged.

**Mutation checks.** Each defect was put back in place in turn, and the new tests failed every time:

| Defect reintroduced | Tests failing |
|---|---|
| Undeclared state keys | 5 |
| Substring verdict routing | 2 |
| BUILD/MVP contradiction | 6 |
| Dead budget check | 4 |
| Critic 4× token retry | 2 |
| Verdict ignores flags | 2 |
| No privacy safeguard | 1 |

## 3. Full offline test results

```
.venv/Scripts/python.exe -m pytest -m "not live"
Baseline (before M6.2.5): 854 passed, 5 deselected
After M6.2.5:             953 passed, 5 deselected, 0 failed, 0 network attempts
```

The 99 additional tests are the 98 new ones plus one net from parametrizing the validator test. The 5 deselected tests are the opt-in live tests.

---

## 4. Critic metadata schema

The same dictionary appears in two places:
- `turn_telemetry.records[i].details` for `node_name == "critique_experiment"`;
- the metadata of the LangSmith span `experiment_critic_decision` (tags `node:critique_experiment` and `decision:llm` or `decision:deterministic`).

**What it contains.** Only enums, codes, counts and flags: no founder text, no experiment text, and no critic prose. `sanitize_metadata` returns it unchanged, which a test verifies.

| Field | Type | Meaning |
|---|---|---|
| `critic_verdict` | `ACCEPT` \| `REVISE` | Effective verdict used for routing |
| `critic_model_verdict` | `ACCEPT` \| `REVISE` \| null | Verdict as emitted (null for a deterministic decision) |
| `verdict_flag_conflict` | bool | Verdict disagreed with the six flags |
| `dim_assumption_alignment` … `dim_scope` | `PASS` \| `FAIL` \| `NOT_EVALUATED` | Six dimension flags (`NOT_EVALUATED` on a deterministic REVISE) |
| `failed_dimensions` | list[str] | Names of the failing dimensions |
| `revision_trigger` | `LLM` \| `DETERMINISTIC` \| null | What caused the REVISE (null on ACCEPT) |
| `revision_round` | int | 1 = first critique, 2 = critique of the regeneration |
| `issue_count` | int | Number of critic issues |
| `issue_codes` | list[str] | Categories from a fixed set: BUILD_FIRST, CONTAMINATION, FALSIFIABILITY, EVIDENCE_VALIDITY, SAMPLE, BUDGET_COST, TIMELINE, ACCESS, PROTOTYPE, OVERPOWERED_CRITERIA, PRIVACY_CONSENT, SCOPE, ASSUMPTION_ALIGNMENT, OTHER |
| `contract_violation_codes` / `contract_advisory_codes` | list[str] | Deterministic findings (Section 1) |
| `critic_attempts` | int | Outer attempts used (0 = deterministic) |
| `llm_failure_kinds` | list[str] | Failed attempts before success, by kind |
| `next_route` | `save_final_experiment` \| `generate_experiment` \| `handle_experiment_failure_stop` | Where the graph goes next |

**The record itself also carries:**
- `node_name` and `logical_stage = experiment_critique`;
- `model_tier`, `resolved_model` and `reasoning_effort` (null means the provider default; Sol is unchanged);
- `latency_ms` and `retry_count`;
- estimated `prompt_tokens`, `completion_tokens` and `total_tokens`;
- `status`: SUCCESS, SKIPPED (deterministic) or FAILED.

LangSmith's ChatOpenAI child run shows real token usage.

**Generator records:** `details = {revision_round, is_revision, llm_failure_kinds}` with the same model fields.

**Verdict versus dimension flags (fail-closed):**

| Model verdict | Flags | Effective | Conflict |
|---|---|---|---|
| ACCEPT | all PASS | ACCEPT | no |
| ACCEPT | any FAIL or missing | REVISE (instructions name the failed dimensions) | yes |
| REVISE | any FAIL | REVISE | no |
| REVISE | all PASS | REVISE (instructions: issues, then failed dimensions, then rationale) | yes |
| invalid token | n/a | rejected by the schema, so treated as an invalid-output retry | n/a |

---

## 5. Routing before and after, for the corrected edge cases

| Case | Before | After |
|---|---|---|
| Verdict "ACCEPTED" / "accept with changes" | Accepted by substring | Schema rejects it; one corrective retry; anything other than an exact ACCEPT routes as REVISE |
| ACCEPT with a FAIL flag | Saved and handed to the founder | REVISE; conflict recorded |
| REVISE, all PASS, no issues or instructions | Regeneration with **no** revision section (identical prompt) | REVISE with the rationale as instruction |
| BUILD_MINIMAL_PROTOTYPE, no prototype | 2 deterministic REVISEs, then failure stop every time | LLM critic judges the bounded build against `minimum_build_scope`; on ACCEPT, executes without a false "no prototype" requirement |
| PROTOTYPE_TEST, prototype exists but no MVP | Deterministic REVISE (checked only `mvp_exists`) | Allowed (testing an existing prototype) |
| PROTOTYPE_TEST, no artifact, not authorized | Deterministic REVISE | Deterministic REVISE (`PROTOTYPE_NOT_AVAILABLE`) |
| Cost over a known budget | Not checked before critique (dead key); can_execute misparsed "$1,500" | Deterministic REVISE before any LLM critique |
| Unknown budget | Not checked | Advisory to the critic; never treated as zero |
| Whitespace-only required field | `ValueError`, graph error; `/retry` failed forever | Deterministic REVISE inside the bounded loop |
| Draft stamped for another assumption | Check never fired | Deterministic REVISE (`ASSUMPTION_MISMATCH`) |
| Persistent token exhaustion in the critic | 4 Sol calls | 2 calls, then `TokenBudgetExhaustedError` |
| Invalid structured output | Identical retry | One retry with schema-field feedback |
| Recording without consent | Not checked | Deterministic REVISE with a specific consent instruction |
| Second REVISE (failure stop) | `experiment_design_status` dropped | Persisted as `QUALITY_BAR_NOT_MET`; terminal shows `EXPERIMENT_DESIGN_NEEDS_REVISION` |
| Accepted experiment, evidence wait | Handoff dropped; interrupt re-rendered it; terminal could never report `WAITING_FOR_EVIDENCE` | Composed handoff displayed and persisted; status reachable |
| Founder asks a question while waiting | Advisor reply (`waiting_response`) dropped; handoff shown again | Reply displayed at the next wait |

---

## 6. Remaining quality and latency risks

1. **Stricter acceptance may add revisions.** ACCEPT with a FAIL flag now revises, and a deterministic consent or budget violation uses up the single revision. The generator is now told about consent up front to reduce first-draft rejections. Whether revisions become more or less frequent is unmeasured.
2. **Deterministic REVISEs share the one revision with the LLM critic.** If a regenerated draft trips a violation again, the cycle ends in the failure stop. This is unchanged architecture, kept per scope.
3. **Heuristic detectors.**
   - The recording, consent and minors patterns can miss unusual wording, or match consent words used in another sense.
   - Personal-data collection is advisory only, by design.
   - Issue codes are coarse categories for telemetry; they never drive routing.
   - A minor is "known" only if the founder's text or the experiment text mentions youth or under-18 participants. Otherwise the critic gets a POSSIBLE_MINORS note.
4. **Prompt changes (v2).** About +169 tokens for the generator and +216 for the critic, plus a stricter schema. This can change model behaviour slightly. The live benchmark below is the check.
5. **Budget comparison is single-currency.** Costs in another currency are advisory only. Budget period (monthly vs. one-off) is not modelled.
6. **Not changed (out of scope):**
   - the dead duplicate `assess_can_execute` definition (`nodes.py` ~4952, shadowed by ~5117);
   - other undeclared state keys from the M6.2.4 scan (`active_critical_assumption`, `evidence_clarification_question`, `previous_weak_area`);
   - the conversational-repair detector, which treats any message starting with "what…" as a repair, including during the evidence wait. This was observed while testing.
7. **Latency.** No speed-up is claimed.
   - The added deterministic work takes milliseconds.
   - A deterministic REVISE now skips one Sol critique when it fires.
   - A conflict REVISE or a consent REVISE can add a cycle.
   - Larger prompts add a few hundred input tokens per call.

   Only a live trace can show the net effect.

---

## 7. One live PitchVision benchmark (paid, about $0.35; run only with your approval)

1. **Environment** (in `.env` of this worktree):
   - `LANGSMITH_TRACING=true`
   - `LANGSMITH_API_KEY=…`
   - `LANGSMITH_PROJECT=startary-ai-latency`
   - keep `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true`
   - leave models, effort and `STARTARY_*` settings unchanged
2. **Start:** `.venv/Scripts/startary-dev.exe`. Paste the **exact** original PitchVision first message used for the 199.35 s trace.
3. **Answer each clarification question the same way as in the baseline run** (reuse those answers if you have them). Let the run continue to the evidence handoff or the fallback learning action, then exit.
4. **In LangSmith, record for that run:**
   - total duration, tokens and cost;
   - durations of each `generate_experiment` and `critique_experiment` span;
   - for each `experiment_critic_decision` span: `critic_verdict`, `critic_model_verdict`, `verdict_flag_conflict`, `failed_dimensions`, `revision_trigger`, `revision_round`, `issue_codes`, `contract_violation_codes`, `contract_advisory_codes`, `next_route`;
   - any `:retry_token_expansion` run;
   - whether the run ended at `wait_for_evidence` (the founder sees the handoff) or at the fallback.
5. **Compare with the baseline:** 199.35 s total, 50.08K tokens, $0.3483, and an experiment loop of 108.61 s (42.34 + 20.75 + 26.52 + 19.00). **Report the difference as one measurement, not a trend.**
6. **Quality check:** read the accepted experiment. It should:
   - target the active assumption;
   - fit within the $1,500 budget;
   - use the two teams and three coaches;
   - have concrete founder actions;
   - define support, contradiction and inconclusive signals;
   - include a permission step if it records matches.

---

## 8. Files changed in M6.2.5

| Status | Files |
|---|---|
| New | `startary/domain/experiment_contract.py`; `tests/test_m625_experiment_loop_correctness.py`; this report |
| Modified | `startary/graph/nodes.py` (generate / validate / critique / `assess_can_execute`, wrapper `run_metadata`, telemetry `details`); `startary/graph/builder.py` (strict route); `startary/graph/graph_state.py` (4 declared keys); `startary/domain/telemetry.py` (`details`, stages); `startary/llm/provider.py` (`classify_llm_failure`, `schema_correction_message`); `startary/llm/schemas.py` (strict critic enums); `startary/llm/prompts.py` (v2 prompts, notes section); `startary/observability/langsmith.py` (`emit_decision_span`); `tests/test_experiment_generator.py`; `tests/test_m413_latency_and_efficiency.py` |
