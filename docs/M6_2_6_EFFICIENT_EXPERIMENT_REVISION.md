# Startary M6.2.6: Efficient Experiment Revision Architecture

**Status:** implemented behind `STARTARY_EXPERIMENT_REVISION_V2` (default **false**) and verified offline: **1013 tests pass**, 0 fail, no network attempts.

**Not done:** no live benchmark, no paid calls, no commit or push, no deployment. `deploy-alpha`, model assignments and Sol reasoning effort are unchanged.

**No latency improvement is claimed** (Section 11).

**Principle:** V2 does not make experiments easier to accept. The acceptance rule is unchanged (an exact `ACCEPT` and all six dimensions `PASS`), and the focused re-critique is stricter than the full one. V2 changes only *how* a rejected draft is repaired:
- keep what was valid;
- fix only what was flagged;
- repair consent-class defects in code;
- re-check independently;
- never spend more expensive calls than the current loop.

---

## 1. Experiment-loop diagrams

**Before, and still the behaviour with the flag OFF (M6.2.5.1):**

```
generate_experiment ─► validate_proposed_experiment ─► critique_experiment (deterministic contract check → full critic)
                                                          │
           ┌──────────────────────────────────────────────┼─────────────────────────────┐
         ACCEPT                                 REVISE, count ≤ 1                 REVISE, count ≥ 2
           ▼                                              ▼                                ▼
  save_final_experiment                 generate_experiment (new draft;        handle_experiment_failure_stop
           ▼                             only the critic's instruction text)     (fallback; founder learning input
  assess_can_execute ─► handoff                    ▼                              restarts the understand pipeline)
                                       validate ─► full critique
```

**After, with the flag ON (V2):**

```
generate_experiment (draft_origin = INITIAL)
     ▼
validate_proposed_experiment ── V2: CONSENT/GUARDIAN-only violations → deterministic remediation (no LLM call,
     │                               no revision allowance used) → contract re-run; tag required_authorizations=PENDING
     ▼
critique_experiment
     │  contract violation, or a targeted edit that changed nothing ─► deterministic REVISE (no LLM call)
     │  otherwise FULL critic (V2 schema adds revision_scope), or FOCUSED critic for a light targeted edit
     ▼  route_on_experiment_critique(verdict, count, strategy)
 ┌───────────────┬───────────────────────────────────────────┬─────────────────────────────────────────────┬──────────────────┐
 ACCEPT          REVISE, count ≤ 1, strategy=TARGETED_EDIT     REVISE, count ≤ 1, strategy=FULL_REGENERATION   REVISE, count ≥ 2
 ▼               ▼                                             ▼                                             ▼
 save ─► can_execute ─► handoff        revise_experiment (patch of the preserved           generate_experiment              failure stop
 (pending permissions shown           rejected draft; only changed fields)                (draft_origin = FULL_REGEN.)     (unchanged)
  "not yet obtained")                  ▼                                                   ▼
                                       validate (remediation again if needed)             validate ─► FULL critique
                                       ▼
                                       critique: FOCUSED (light edit) or FULL (substantial edit)
```

---

## 2. Implementation changes and modified files

| File | Change |
|---|---|
| `startary/config.py` | `experiment_revision_v2: bool = False`, alias `STARTARY_EXPERIMENT_REVISION_V2` |
| `startary/domain/experiment_revision.py` (new) | Pure V2 rules: strategy selection, consent remediation, patch application, field diff, substantial-change test, critic-mode selection, focused acceptance rule, revision records, V2 metadata, bounds |
| `startary/domain/experiment_contract.py` | `PrivacySignals` / `privacy_signals()` extracted (same rules) and shared by the contract and authorization tagging; `Authorization` codes; `required_authorizations()`, `pending_authorizations()`, `pending_authorization_notice()`; `experiment_critique_route(..., revision_strategy=None)`; `critic_decision_metadata(..., revision_strategy=None, extra=None)`. With the defaults (flag OFF), every function behaves as before |
| `startary/llm/schemas.py` | `ExperimentCriticV2LLMOutput` (adds `revision_scope`: NONE / TARGETED / REDESIGN); `ExperimentFocusedCriticLLMOutput` (adds `original_issues_resolved`, `new_blocking_issues`); `ExperimentRevisionPatchLLMOutput` (all editable fields optional). The flag-OFF schemas are untouched |
| `startary/llm/prompts.py` | `EXPERIMENT_CRITIC_REVISION_SCOPE_ADDENDUM` (appended only in V2); `EXPERIMENT_REVISION_SYSTEM_PROMPT` and its message builder; `EXPERIMENT_FOCUSED_CRITIC_SYSTEM_PROMPT` and its message builder. The flag-OFF prompts are untouched |
| `startary/graph/nodes.py` | **New `revise_experiment` node.** **`validate_proposed_experiment`:** V2 remediation and authorization tagging. **`critique_experiment`:** V2 full or focused critic, strategy selection, rejection record, no-change guard, V2 metadata. **`generate_experiment`:** V2 records the draft's origin. **`save_final_experiment`:** carries pending authorizations, which exist only on V2 drafts. **`assess_can_execute` / `compose_execution_handoff`:** state pending permissions. **Reset:** `experiment_revision` is reset at `increment_iteration` and at the learning-input merge |
| `startary/graph/builder.py` | Adds the `revise_experiment` node (main graph and experiment subgraph), the edge `revise_experiment → validate_proposed_experiment`, and strategy-aware routing |
| `startary/graph/graph_state.py` | One new declared key: `experiment_revision: Optional[dict[str, Any]]` |
| `startary/domain/telemetry.py` | Logical stages for `revise_experiment` (experiment_design) and `validate_proposed_experiment` (experiment_validation) |
| `.env.example` | `STARTARY_EXPERIMENT_REVISION_V2=false` |
| `tests/conftest.py` | Pins the flag to false for the default suite, so a developer `.env` with V2 enabled can't change existing tests. V2 tests pass the setting explicitly |
| `tests/test_m626_efficient_experiment_revision.py` (new) | 41 tests (Section 8) |

**No existing test assertion was changed.**

---

## 3. Choosing a targeted edit or a full regeneration

`select_revision_strategy` applies these rules; the first match wins.

| Trigger | Condition | Strategy |
|---|---|---|
| Deterministic REVISE | Any of `PROTOTYPE_NOT_AVAILABLE`, `BUILD_SCOPE_UNDEFINED`, `ASSUMPTION_MISMATCH`, or an unremediable `CONSENT_STEP_MISSING` / `GUARDIAN_CONSENT_MISSING` | FULL_REGENERATION (the method must change) |
| Deterministic REVISE | Only `COST_EXCEEDS_BUDGET`, `REQUIRED_FIELD_EMPTY`, `INVALID_SAMPLE_SIZE` | TARGETED_EDIT (a local fix) |
| Deterministic REVISE | Anything else | FULL_REGENERATION |
| LLM REVISE | `assumption_alignment` FAIL | FULL_REGENERATION (tests the wrong thing) |
| LLM REVISE | Critic `revision_scope` is not exactly `TARGETED` (REDESIGN, or missing) | FULL_REGENERATION |
| LLM REVISE | More than 3 of the six dimensions FAIL | FULL_REGENERATION |
| LLM REVISE | Otherwise | TARGETED_EDIT |

**How the critic chooses `revision_scope`.** The full critic in V2 declares it:
- **TARGETED:** editing specific fields fixes the issues while keeping the method, participants and evidence source.
- **REDESIGN:** a new method, population or evidence source is needed, for example "opinion interviews cannot test the claim".

This is how an evidence-validity failure that needs a new method leads to a full redesign, while one fixable by sharper signals leads to an edit.

**How the targeted edit works.**
- The reviser receives:
  - the preserved rejected draft;
  - the active assumption;
  - the failed dimensions, the critic's issues and instructions;
  - every deterministic finding;
  - the known founder facts (UNKNOWN placeholders omitted);
  - the build-readiness guidance and the revision round.
- **It returns only the fields that must change.** The patch is applied to the rejected draft, so untouched fields can't regress.
- **Code-owned fields are re-imposed after patching:**
  - the hypothesis (= the assumption statement);
  - the assumption stamp;
  - the BUILD type forcing.
- **Derived safety fields are recomputed** by validation: remediation codes and authorizations.
- **The merged draft must still pass `ExperimentGeneratorLLMOutput`**; otherwise there is one corrective retry naming the failing fields.

---

## 4. Deterministic remediation: rules and safety limits

Implemented in `remediate_consent_violations` and called from `validate_proposed_experiment`, V2 only.

| Rule | Implementation |
|---|---|
| Codes eligible | Only `CONSENT_STEP_MISSING` and `GUARDIAN_CONSENT_MISSING` |
| All-or-nothing | Refused when **any** other violation is present (no partial repair: budget, prototype, assumption, and other substantive violations are never repaired automatically) |
| Method must not change | Refused when the method itself is the problem: secretly / covertly / hidden camera / undercover / scraping / "without their knowledge/consent" |
| Placement | Inserted immediately **before** the first sensitive founder action (recording, data collection, minors, interviewing/surveying/observing/testing); all other actions keep their order |
| Idempotent | An identical step is never inserted twice (checked against the action text, so replays, re-validation and resume are safe) |
| Verified | The full contract is re-run on the repaired draft; if any violation remains, the repair is discarded and the normal revision or failure path continues (`REMEDIATION_INSUFFICIENT`) |
| Still reviewed | The repaired draft always goes to an LLM critic for semantic evaluation |
| Free | No LLM call, and no use of the semantic revision allowance |

**The steps (code-authored, with no claim of legal compliance):**
- **Consent:** "Before any recording or data collection, explain the purpose to everyone involved and obtain their informed permission; record nothing without it. Keep recordings private to the founding team and delete them once the analysis is complete."
- **Guardian:** "Before recording, interviewing, or collecting information about any participant under 18, obtain written parent/guardian authorization and the club's permission; involve no minor without both."

**A permission step is not permission.**
- Every V2 draft is tagged with `required_authorizations` (RECORDING_PERMISSION, GUARDIAN_AUTHORIZATION, PERSONAL_DATA_NOTICE), using the same signals as the contract, and with `authorization_status = "PENDING"`.
- **Nothing in Startary ever sets it to OBTAINED.**
- `assess_can_execute` states the experiment "may start only after these permissions are obtained (not yet obtained)".
- The handoff begins: "BEFORE YOU START: these permissions are required and have NOT been obtained yet…".

**Guardian requirements still follow M6.2.5.1.** They come from the experiment itself, never from youth words in conversation alone. Unknown ages remain an advisory to the critic; they are never turned into a "minor" or "adult" fact.

---

## 5. Focused or full critique

| Situation | Critic |
|---|---|
| First draft; any full regeneration | FULL (V2 schema with `revision_scope`) |
| Targeted edit changing `experiment_type`, or rewriting `method`, `target_participants_or_scope` or `evidence_to_collect` (word-set similarity < 0.5) | FULL |
| Targeted edit that is light, with the prior critique recorded | FOCUSED |
| Targeted edit that changed nothing | No LLM call: deterministic REVISE (`NO_EFFECTIVE_CHANGE`) |
| Any contract violation (after remediation) | No LLM call: deterministic REVISE |

**What the focused critic receives:**
- the complete revised draft;
- a structured change list with the original and revised value of each changed field (every other field is identical to the original, so the original is fully determined without sending it twice);
- the original verdict, all six original dimension results and the original issues;
- the founder facts;
- the deterministic notes.

**What it must do:**
- re-score **all six** dimensions, with no carry-over from the earlier pass;
- say whether every original issue is resolved;
- list any new blocking issue.

**Focused acceptance is stricter.** It requires an exact `ACCEPT`, all six `PASS`, `original_issues_resolved = true` and no `new_blocking_issues`; otherwise the result is REVISE. The generator never evaluates its own work.

---

## 6. Revision and retry state machine, with bounds

| Mechanism | Bound per experiment cycle | Uses the semantic allowance? |
|---|---|---|
| Semantic revision (TARGETED_EDIT **or** FULL_REGENERATION) | **1** (`MAX_SEMANTIC_REVISIONS`, unchanged) | Yes |
| Deterministic contract REVISE | Within the same allowance | Yes (unchanged) |
| Deterministic consent remediation | At most 1 per draft (at most 2 per cycle), idempotent | **No** |
| LLM retries per call (transient, or invalid output with schema feedback) | 2 attempts | n/a |
| Token-budget recovery per attempt | 1 expansion; exhaustion stops (M6.2.5) | n/a |
| Generation-class LLM calls (`generate_experiment` + `revise_experiment`) | **≤ 2**, the same as the flag-OFF loop | n/a |
| Critic LLM calls | **≤ 2**, the same as the flag-OFF loop | n/a |

**States:** INITIAL draft → critique → {ACCEPT → save once → can_execute → handoff | REVISE(round 1) → TARGETED_EDIT or FULL_REGENERATION → critique → {ACCEPT | REVISE → failure stop}}.

**Every path ends** at the evidence handoff, `pause_needs_help`, or `handle_experiment_failure_stop`. An always-rejecting critic ends at the failure stop after 1 generation, 1 edit, 1 full and 1 focused critique (verified).

**State:**
- `experiment_revision` holds the rejected draft, a summary of its critique, the chosen strategy, how the active draft was produced, and the changed fields.
- The rejected draft is preserved until the revision completes.
- The revised draft becomes `proposed_experiment_output` only through `revise_experiment` or `generate_experiment`.
- The key is reset with `critic_revision_count`.

---

## 7. Feature flag

**Setting:** `Settings.experiment_revision_v2`, from env or `.env` as `STARTARY_EXPERIMENT_REVISION_V2`, default `false`. Nodes read it at execution, and only an explicit `True` enables V2.

**Flag OFF (exact M6.2.5.1 behaviour):**
- the original critic schema and prompt;
- no remediation;
- no `experiment_revision` state;
- no authorization keys;
- critic metadata keys unchanged;
- REVISE always goes to `generate_experiment`;
- `revise_experiment` exists in the graph but is unreachable.

**Flag ON:** everything in Sections 3 to 6.

**One code path.** Both modes reuse `experiment_contract.py` for every validation rule. V2 adds strategy and remediation rules on top in `experiment_revision.py`, without duplicating the business logic.

**For A/B runs:**
- Change only this one variable; models, effort, prompts and inputs stay the same.
- Restart the CLI between arms, because the flag is read per node and switching mid-cycle would mix modes.

---

## 8. Offline test results

```
.venv/Scripts/python.exe -m pytest -m "not live"
Before M6.2.6: 972 passed
After  M6.2.6: 1013 passed, 5 deselected (live, opt-in), 0 failed, 0 network attempts
```

`tests/test_m626_efficient_experiment_revision.py` holds 41 tests. They use the real compiled graph, MemorySaver and the scripted LLM. The schema of each call identifies its kind, so the tests count generations, edits, full critiques and focused critiques exactly.

| # | Requirement | Test |
|---|---|---|
| 1 | First-draft ACCEPT: 1 generation and 1 full critique | `test_first_draft_accept_uses_one_generation_and_one_full_critique` |
| 2, 3 | Targeted edit preserves passing parts and fixes falsifiability | `test_targeted_edit_fixes_falsifiability_preserves_other_fields_and_uses_focused_critic` |
| 4, 6 | Evidence-validity redesign: full regeneration, then a full critique | `test_evidence_validity_redesign_triggers_full_regeneration_and_full_critique`; `test_substantial_targeted_edit_gets_a_full_critique` |
| 5 | Focused critique catches regressions | `test_focused_critique_rejects_regressions_and_loop_ends_bounded` (new blocking issue / regressed dimension / unresolved issue) |
| 7 | Remediation inserts the step once | `test_consent_remediation_inserts_step_once_before_recording_and_keeps_permission_pending`; `test_remediation_is_idempotent` |
| 8 | Guardian safeguard for actual minors | `test_guardian_safeguard_for_actual_minors_is_remediated_and_stays_pending`; `test_covert_recording_of_minors_is_not_remediated` |
| 9 | Adult coaches: no false guardian requirement | `test_adult_coach_interviews_do_not_trigger_guardian_requirements` |
| 10 | Unknown ages are not invented | `test_unknown_ages_are_not_turned_into_facts` |
| 11 | Remediation is not permission | Asserted in the consent test: `PENDING`, "not yet obtained", handoff notice |
| 12 | Budget and prototype not auto-repaired | `test_budget_and_prototype_violations_are_never_auto_repaired`; `test_mixed_consent_and_budget_violations_are_not_partially_repaired`; `test_budget_violation_gets_a_targeted_edit_with_the_finding` |
| 13, 14 | Bounded; fallback reachable | `test_always_rejecting_critic_is_bounded_and_falls_back`; `test_targeted_edit_with_no_change_is_rejected_without_a_critic_call` |
| 15 | State survives checkpoint restoration; saved once | `test_revision_state_survives_checkpoint_restoration_and_saves_once` |
| 16 | Duplicate resume duplicates nothing | `test_duplicate_resume_does_not_duplicate_remediation_or_llm_calls` |
| 17 | Flag OFF preserves behaviour | `test_flag_off_preserves_m6251_behaviour`; `test_flag_off_revise_still_regenerates_with_the_original_critic` |
| 18 | Flag ON activates V2 | `test_flag_on_uses_v2_schemas_and_records_v2_metadata` |
| 19 | No sensitive text in metadata | `test_v2_metadata_has_no_sensitive_text_and_survives_sanitization` |
| 20 | Prior milestones intact | Full suite (M6.2.3, M6.2.5, M6.2.5.1 all pass) |
| n/a | Grounding | `test_revision_prompt_stays_grounded_in_pitchvision_facts` |
| n/a | Rules | `TestStrategyRules` (11-case strategy table, substantial-change rules, critic mode, focused acceptance, bounds) |

**Mutation checks.** Each V2 mechanism was broken in turn, and the tests failed every time:

| Mechanism broken | Tests failing |
|---|---|
| Targeted edits never chosen | 11 |
| Focused acceptance rule off | 2 |
| Focused critic never used | 8 |
| No remediation | 9 |
| Non-idempotent remediation | 7 |
| Permissions treated as granted | 4 |
| Router ignores strategy | 11 |
| No-change guard removed | 9 |

---

## 9. Remaining risks and edge cases

1. **The critic's `revision_scope` drives targeted versus redesign.** A wrong TARGETED call could produce a superficial fix. Mitigations:
   - the focused critic re-scores all six dimensions, and its acceptance rule is stricter;
   - `assumption_alignment` FAIL, or more than 3 failed dimensions, forces a redesign;
   - substantial edits get a full critic.
2. **The substantial-change test is a word-overlap heuristic** (Jaccard < 0.5 on the method, population or evidence source). A reworded but materially different method could get a focused critique. That critique still re-evaluates every dimension.
3. **The focused critic gets a compressed original:** the full revised draft plus before and after values of the changed fields, not two full drafts. The information is equivalent and the input is smaller, but it relies on the change list being complete. The change list is computed by code, not the model.
4. **A reviser can return a full list that drops the remediation step.** Validation then remediates again (idempotent, still free).
5. **Pending authorizations are stated but not tracked to completion.** No founder flow marks them OBTAINED yet, and evidence analysis doesn't check that permission was obtained. Readiness stays `CAN_EXECUTE`, with the reason "may start only after these permissions are obtained". `NEEDS_HELP` would end the session without a handoff.
6. **Remediation text is generic and code-authored.** It's a practical safeguard, not legal advice.
7. **The no-change guard fails a cycle without a critic call** when a targeted edit changes nothing. The outcome matches what the flag-OFF loop would reach, at lower cost.
8. **Live behaviour is unknown.** Real revision scopes, edit quality, and focused-critic strictness can only be measured live (Section 10).
9. **Unchanged from before:** the msgpack deserialization warnings, the duplicate dead `assess_can_execute`, the "what…"-prefix repair detection, and the undeclared keys listed in M6.2.5.

---

## 10. Manual A/B benchmark (paid; run only with approval)

**Setup:**
- Same worktree, same `.env` (models, effort, LangSmith with hidden inputs and outputs).
- **Control:** `STARTARY_EXPERIMENT_REVISION_V2=false`. **Treatment:** `true`.
- Restart `startary-dev` after each change.
- Use the original PitchVision first message and the **same clarification answers** in every run.
- Run at least 5 per arm, alternating arms to spread out time-of-day API variance.
- Revision cases happen naturally when the first critique rejects. To guarantee revision coverage, also run the canonical fixtures that previously drew a REVISE.

**Collect per run from LangSmith:**

| Metric | Source |
|---|---|
| Total and experiment-loop latency | Root `startary_turn_*` span; sum of `generate_experiment`, `revise_experiment` and `critique_experiment` spans |
| Generation calls / targeted edits | Count of `generate_experiment` / `revise_experiment` node runs |
| Full / focused critiques | `experiment_critic_decision` spans by `critic_mode` |
| Deterministic remediations | `experiment_remediation` spans with `remediation_applied = true` |
| Semantic revisions | Decision spans where `revision_strategy` is not null and `critic_verdict = REVISE` |
| Handoff vs. fallback | Run ends at `wait_for_evidence` vs. `handle_experiment_failure_stop` |
| Tokens and cost | LangSmith run totals (ChatOpenAI child runs) |
| Six dimension results | `dim_*` fields on the final decision span |
| Experiment usefulness | Human rubric, 1–5 each: tests the assumption, fits the $1,500 budget and the founders' access, concrete founder actions, falsifiable thresholds, permission steps where people are recorded |

**Report:** the median and the range per arm, plus the handoff rate. A single run shows variance, not an improvement.

---

## 11. Expected latency impact (estimates only; nothing measured)

The reference is your trace: gen 1 38.36 s, critique 1 24.54 s, gen 2 25.54 s, critique 2 0.00 s (deterministic GUARDIAN), then the failure stop.

**Measured offline (prompt size, not time):**

| Call | Input (approx.) | Output (approx.) |
|---|---|---|
| Full generation (PitchVision) | 4.1K tokens | 530 tokens (whole draft) |
| Targeted edit | 2.4K tokens | 60 tokens (patch fixing two signals) |
| Full critic (V2) | 2.9K tokens | n/a |
| Focused critic | 2.1K tokens | n/a |

**Call count:** the worst case is unchanged (≤ 2 generation-class calls, ≤ 2 critic calls), and full generations never increase.

**Expected effects (estimates):**
- **Targeted path.** Smaller inputs and much smaller outputs should shorten the revision call. Sol's reasoning time dominates, so the size of the saving is unknown until measured.
- **Your trace's ending.** If its round-2 guardian violation was genuine, V2 repairs it in code and spends **one more critic call** (critique 1 took about 24.5 s). The loop can then end in a handoff instead of the failure stop and the full restart that follows it. The experiment loop may get *longer* in that case, while end-to-end time to a usable experiment should fall.
- **First-pass ACCEPT runs:** no change expected.

Only the A/B in Section 10 can confirm any of this.
