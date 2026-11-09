# Startary M6.2.4: Experiment Generation and Critic Loop Investigation

**Status:** read-only investigation. No implementation changes, no paid API calls, nothing committed. `deploy-alpha` was not touched.

**Labels used in this document:**
- **Confirmed (code)**: read directly in the source.
- **Confirmed (offline run)**: reproduced by running the real nodes or the experiment subgraph with a scripted LLM and no network.
- **Hypothesis**: plausible, not yet verified.
- **Estimate**: a projected number, not a measurement.

**Trace under study (PitchVision AI, full workflow):** 200 s, 50.08K tokens, $0.3483. The four experiment stages were:

| Stage | Duration |
|---|---|
| Experiment generation 1 | 42.34 s |
| Critique 1 | 20.75 s |
| Experiment generation 2 | 26.52 s |
| Critique 2 | 19.00 s |
| **Total** | **108.61 s (about 54% of the run)** |

---

## 1. The experiment loop as it runs today

```
decide_next_strategic_action
   │
   ▼
generate_experiment ─ LLM: Sol (reasoning tier), max_completion_tokens=3500, reasoning_effort = API default,
   │                  client timeout 60 s. Outer attempts ≤ 2; each attempt may add 1 token-expansion retry
   │                  (1.6×, effort=low). Stops early on permanent errors and on TokenBudgetExhausted.
   │                  Revision input: the SAME full context plus the critic's instruction text only
   │                  (the rejected draft is NOT included).
   ▼
validate_proposed_experiment ─ deterministic. Checks text fields ≥ 5 characters and lists non-empty
   │                           (already enforced by the Pydantic schema). Overwrites hypothesis with the
   │                           assumption statement. Its fake-precision check only logs a warning.
   │                           A failure raises ValueError: a graph error, not a revision.
   ▼
critique_experiment
   ├─ deterministic pre-checks (0 LLM calls). Any hit → REVISE and critic_revision_count += 1:
   │     • target_sample_size ≤ 0
   │     • experiment_type ∈ {PROTOTYPE_TEST, MVP_TEST, USABILITY_TEST} and mvp_exists is not true
   │     • budget: reads exec_ctx["monthly_budget"]        ← DEAD KEY (the store uses available_budget_amount)
   │     • exp.assumption_id ≠ active id                   ← DEAD (the generator schema has no assumption_id)
   └─ LLM: Sol, max_tokens=2500, effort = API default. Outer attempts ≤ 2, each with a token-expansion retry.
         Does NOT stop on TokenBudgetExhausted, so up to 4 Sol calls.
         Verdict: contains "ACCEPT" → ACCEPT; anything else → REVISE.
         On REVISE: count += 1, critic_revision_instructions = revision_instructions or "; ".join(issues)
   │
   ▼ route_on_experiment_critique (builder.py:291)
   ├─ ACCEPT ─────────────────────► save_final_experiment ─► assess_can_execute (deterministic)
   │                                    ├─ CAN_EXECUTE ─► compose_execution_handoff ─► wait_for_evidence
   │                                    ├─ NEEDS_HELP ──► pause_needs_help ─► END
   │                                    └─ BLOCKED ─────► handle_experiment_failure_stop   (no revision attempt)
   ├─ REVISE and count ≤ 1 ─────────► generate_experiment   (the single allowed regeneration)
   └─ REVISE and count ≥ 2 ─────────► handle_experiment_failure_stop
                                       ─► ask_and_wait_for_founder (waiting_for_learning_input)
```

- **Revision bound (confirmed, code):** at most **2 generations and 2 critic evaluations** per experiment cycle. One evaluation can be the deterministic pre-check.
- **Counter reset (confirmed, code):** `critic_revision_count` and `critic_revision_instructions` reset only in `increment_iteration` and in `merge_founder_answer` (learning-input branch). `save_final_experiment` does not reset them, but every new cycle passes through `increment_iteration` first.
- **Termination:** `handle_experiment_failure_stop` records the rejected draft as `QUALITY_BAR_NOT_MET` in `previous_experiments`, builds a `FallbackLearningAction`, and pauses for founder input.
- **The status meant for the terminal renderer is lost (confirmed, offline run).** The node also sets `experiment_design_status = QUALITY_BAR_NOT_MET` for `cli.py:1047` and `domain/terminal_state.py`. That key is **not declared** in `StartaryGraphState`, so LangGraph drops it. A subgraph run ending in failure returned `experiment_design_status = None`. The same node's `active_execution_handoff`, `execution_handoff_text` and `waiting_response` are also undeclared, as is `customer_access_assessment` from `assess_can_execute`. This is the same defect class M6.2.3 fixed for `answered_question_id`.

**Normal-case LLM calls per cycle:** 1 generation + 1 critique (accepted first time), or 2 + 2 (one revision, as in the PitchVision trace). The worst case with persistent token exhaustion is 8 + 8.

---

## 2. Why the loop repeats (confirmed)

The **only** edge back to `generate_experiment` is a **REVISE verdict from the first critique**. The REVISE can come from two places:
1. the deterministic pre-checks (0 ms, no LLM call), or
2. the Sol critic.

**In the PitchVision trace, critique 1 took 20.75 s, so it was a Sol call:** the loop repeated because of a **semantic revision request** from the LLM critic. The rejection reason itself is **not observable**. Inputs and outputs are hidden in LangSmith, and neither Startary telemetry nor run metadata records the critic's verdict, its six pass/fail flags, or its issues. This is the main observability gap for this milestone.

**The second outcome is not visible from the stage times either.** If `save_final_experiment` follows critique 2 in the trace, the revision was accepted. If `handle_experiment_failure_stop` follows, the cycle ended in fallback.

**Hypotheses for the PitchVision rejection (none verified):**
- **Contamination.** The founder said coaches "agreed to test our prototype once ready". The critic's contamination rule ("no pitching or demoing before validating the problem") **is not in the generator prompt**, so a first draft built around those coaches could be rejected on it.
- **Prototype test without an MVP.** The generator prompt forbids it, but the model can still produce one.
- **Other critic dimensions:** falsifiability, or criteria stronger than the assumption itself ("overpowered" criteria).

**Gen 2 started from scratch (confirmed, code).** `build_experiment_generator_user_message` adds a "CRITIC REVISION INSTRUCTIONS" section to the same full context. The rejected draft and the critic's per-dimension results are not passed, so the generator can't patch only the failing parts.

**Critique 2 judged without memory (confirmed, code).** `build_experiment_critic_user_message` has no parameter for the previous critique. Critique 2 is therefore a complete, from-scratch review of a completely new draft, and it can raise issues that critique 1 never mentioned. With only one regeneration allowed, that sends the cycle to the failure stop.

**A deterministic dead end for BUILD_MINIMAL_PROTOTYPE (confirmed, offline run).**
- When `build_readiness.decision == BUILD_MINIMAL_PROTOTYPE`, `generate_experiment` forces `experiment_type = PROTOTYPE_TEST`.
- When `mvp_exists` is not true, the critic's pre-check rejects every PROTOTYPE_TEST.
- Running the experiment subgraph offline: both generations were forced to PROTOTYPE_TEST, both were rejected by the pre-check (0 critic LLM calls), and the cycle hit the revision cap with "no prototype exists yet".
- Build readiness exists to say "build the minimal prototype now", so this combination can never succeed. It does not affect PitchVision's normal path (build readiness keeps idea-stage startups on CONTINUE_MANUAL), but it costs two Sol generations whenever it fires.

---

## 3. When the critic rejects

**Deterministic pre-checks:** see the diagram. Two of the four checks can never fire (dead budget key; no `assumption_id` in the schema).

**LLM critic**, as written in its system prompt:
- **ACCEPT** when all six dimensions pass: `assumption_alignment`, `evidence_validity`, `feasibility`, `cost_efficiency`, `falsifiability` and `scope`.
- **REVISE** when any dimension fails, with emphasis on:
  - **build-first bias:** building apps, roadmaps or launches;
  - **missing falsification thresholds;**
  - **work the founder can't actually execute;**
  - **contamination:** pitching or demoing before the problem is validated;
  - **prototype-standard violations:** scope, feature creep, more than 2 weeks;
  - **criteria stronger than the assumption;**
  - **treating recruitment difficulty as contradiction.**

**The code trusts only the verdict string (confirmed, code).** The six flags are never checked against it. Because `verdict` is a free string, any wording without "ACCEPT" counts as REVISE, even when all six flags pass.

### Three kinds of retry

| Kind | Where | Bound | Effect |
|---|---|---|---|
| **Correctness retry: schema** | Generator and critic outer loops; Pydantic failures raise inside the structured call | 2 attempts, identical prompt, no error feedback | Most of `validate_proposed_experiment` duplicates the schema (`min_length=5`, lists `min_length=1`). Offline: a draft with a 3-character support signal failed both generator attempts on every `/retry` (6 calls over 3 tries). |
| **Correctness retry: validator** | `validate_proposed_experiment` | 0 retries; raises | A graph error. `/retry` re-runs validation on the same committed draft, so it fails again. Today only whitespace-only fields can reach it. |
| **Correctness retry: token budget** | `_invoke_structured_with_token_retry` | 1 expanded retry per attempt | Generator: at most 2 calls. **Critic: at most 4 Sol calls** because it doesn't stop on `TokenBudgetExhaustedError` (confirmed, offline: 4 calls vs 2). |
| **API failure: transient** (timeout, connection) | Outer loops | 2 attempts | Doubles latency and cost on failure. **Gen 1 took 42.34 s against a 60 s reasoning-tier timeout**, so a slow tail can time out and repeat the whole call. |
| **API failure: permanent** (401/403/404, model not found) | Outer loops | Stops immediately | Graph error; the CLI offers `/retry`. |
| **Semantic revision** | Critic REVISE, from the pre-check or the LLM | 1 regeneration per cycle | Deterministic and LLM rejections **share** this single revision. A second REVISE goes to the failure stop. |
| **Post-acceptance gate** | `assess_can_execute` | none | BLOCKED goes straight to the failure stop without a revision attempt. NEEDS_HELP pauses. |

---

## 4. Where the generator, validator and critic overlap

| Requirement | Generator prompt | Schema / validator | Critic pre-check | LLM critic | Post-acceptance gate |
|---|---|---|---|---|---|
| Critical-assumption grounding | ✔ | hypothesis overwritten with the statement | assumption_id (dead) | assumption_alignment | — |
| Founder constraints and budget | ✔ | — | monthly_budget (dead) | feasibility | `assess_can_execute` budget (live key) |
| Customer-access realism | ✔ | — | — | feasibility | access status |
| Concrete execution steps | implied | founder_actions ≥ 1 (twice) | — | scope (indirect) | — |
| Evidence collection | ✔ | evidence_to_collect ≥ 5 characters (twice) | — | evidence_validity | — |
| Falsifiable support / contradiction / inconclusive signals | ✔ | each ≥ 5 characters (twice) | — | falsifiability | — |
| No prototype test without an MVP | ✔ | — | ✔ (conflicts with BUILD forcing) | prototype standards | ✔ |
| Contamination (no pitching first) | **✘ missing** | — | — | ✔ | — |
| Criteria not stronger than the assumption | ✔ | — | — | ✔ | — |
| Recruitment difficulty is not contradiction | ✔ | — | — | ✔ | — |
| No fake precision | ✔ | logged only, not normalized | — | — | — |
| **Ethics, privacy, consent** | **✘** | **✘** | **✘** | **✘** | **✘** |

**Ethics and privacy (confirmed, code).** The brief lists ethical and privacy constraints as a quality requirement to preserve, but **nothing in the loop enforces them today**: no prompt, schema or check mentions ethics, privacy or consent. PitchVision records amateur football matches, which may include minors, so participant consent is a real consideration for its experiments.

**Fact payloads (measured offline on prompt size, PitchVision facts after M6.2.3):**
- **Generator:** about 4.0K input tokens. System prompt 5.6K characters, schema 4.8K characters, and a user message of 5.7K characters, of which **4.3K is full execution/progress JSON, mostly UNKNOWN placeholders**.
- **Critic:** about 2.4K input tokens. It already receives a slimmed fact summary (0.5K characters).
- Sending the generator the same slimmed facts would remove **about 950 input tokens per generation (about 24%)**.
- A sample draft is about 530 output tokens. The 42 s and 20 s stage times are therefore dominated by Sol's reasoning and output, not by input size.

---

## 5. Ways to reduce redundant reasoning

This covers research questions 7 and 8.

**Can quality be enforced before the first critique, with no extra model call?** Partly. A deterministic **experiment contract gate** can check:
- budget against estimated cost, using `available_budget_amount`;
- prototype type consistent with both the MVP status and the build-readiness decision;
- target sample size against the founder's stated reachable contacts (a heuristic);
- support and contradiction signals not identical, and the contradiction signal not about recruitment;
- estimated time within 1 to 2 weeks;
- hypothesis equal to the assumption statement;
- fake-precision normalization;
- the experiment's `required_access` stays within the access the founder actually stated (a heuristic).

Semantic qualities can't be checked deterministically: alignment, evidence validity, contamination, cost efficiency, and whether the criteria are stronger than the assumption. These still need a model judgment. They can be **put in front of the generator** as an explicit checklist that mirrors the critic's six dimensions, including contamination and consent. That is a prompt change, so it needs an A/B test.

**Is a second full critique always needed?** Today, **yes**. Gen 2 regenerates the whole draft without seeing gen 1, so every dimension can change and a full review is the only safe check. Once revisions edit the rejected draft instead of starting over, a **focused re-check (delta critique)** is enough. It confirms each listed issue is resolved and that dimensions which passed before haven't regressed. The second generation itself stays necessary whenever the critic's rejection is legitimate.

---

## 6. Proposed architecture: validate first, edit the draft, re-check only the changes

```
generate_experiment (Sol; generator receives slimmed facts + the experiment contract checklist)
   ▼
experiment_contract_gate (deterministic; merges validate + pre-checks + can_execute's hard blockers; dead keys fixed;
   │                      BUILD/MVP conflict resolved)
   ├─ contract violations → targeted revision request (no critic LLM call; its own small budget)
   ▼
critique_experiment (Sol; full critique, first pass only; records 6 dimension flags + issue categories in telemetry)
   ├─ ACCEPT → save_final_experiment → assess_can_execute (now only NEEDS_HELP / access, since blockers ran earlier)
   └─ REVISE (round 1) → revise_experiment (Sol; input = rejected draft + failed dimensions + issues;
                                             "change only what is needed")
                           ▼
                        experiment_contract_gate
                           ▼
                        delta_critique (Sol; input = round-1 issues + revised draft + diff;
                                        verify resolution, guard PASS dimensions, no new non-blocking issues)
                           ├─ ACCEPT → save
                           └─ REVISE → handle_experiment_failure_stop (unchanged termination)
```

**Supporting fixes, each with no expected quality cost:**
- Declare the dropped state keys.
- Stop the critic's token-retry amplification: at most 2 calls instead of 4.
- Log critic dimension flags, the revision trigger (deterministic or LLM) and token-retry use in telemetry and LangSmith metadata. These are non-sensitive operational fields.

**Not recommended:**
- **Merging generator and critic into one self-critiquing call.** It removes the independent check.
- **Skipping the critic after a revision.** It would accept an unreviewed, fully regenerated draft.
- **Moving the critic to the fast tier, or lowering Sol's reasoning effort, without an A/B test.** M4.1.4.1 mandated Sol for `critique_experiment`, and M6.2.3 kept effort unchanged.

## 7. Quality risks of each change

| Change | Quality risk | Mitigation |
|---|---|---|
| Contract gate before the critic | Heuristics (sample vs. contacts, access) can reject good drafts, using up a revision on noise | Only hard violations block; heuristics become advisory notes passed to the critic; give the gate its own revision budget |
| Checklist in the generator prompt | Prompt change can alter experiment style; the checklist may make drafts longer | A/B against canonical fixtures plus PitchVision; compare the critic's dimension flags |
| Revising the existing draft | The generator may hold on to a flawed design instead of rethinking it; small edits may hide a wrong method choice | When `assumption_alignment` or `evidence_validity` fails, allow full regeneration instead of an edit |
| Delta critique | May miss regressions in dimensions it didn't re-examine; "no new non-blocking issues" could let a real new defect through | Always re-check all six flags; allow new **blocking** issues; full critique whenever the revision was a full regeneration |
| Slimmed facts for the generator | UNKNOWN fields vanish, so the generator might assume something unknown is absent (e.g., no budget) | Add a short "unknown fields" list (field names only) |
| Resolving the BUILD/MVP conflict | Might let real build-first tests through | Allow a prototype test without an MVP **only** when build readiness says BUILD and the founder actions stay within `minimum_build_scope` |
| Ethics/privacy criterion | Could over-reject routine interviews | Scope it to recording, personal data and minors; require a consent step rather than rejecting the experiment |
| Capping critic token retries | A long critique that used to recover on its 3rd or 4th call now fails earlier | Failure is already a graph error with `/retry`; measure the token-retry rate first |

---

## 8. Offline regression tests needed before implementation

All of these can run with `tests/fixtures/scripted_llm.py` (no network; the `tests/conftest.py` guard applies).

1. **Loop routing table:** ACCEPT → save; REVISE 1 → generate; REVISE 2 → failure stop; at most 2 generations and 2 critiques per cycle; counter reset at `increment_iteration` and the learning-input merge. Use the graph-level run that counts calls by schema.
2. **Termination state persists in graph runs:** `experiment_design_status = QUALITY_BAR_NOT_MET`, `fallback_learning_action`, and the waiting_for_learning_input interrupt payload. This **fails today**, because the status key is dropped.
3. **Revision payload:** round 2 receives the critic's instructions (exists today). After the change, it also receives the rejected draft and the failed dimensions.
4. **Delta critique payload** (after the change): it receives the round-1 issues, and dimensions that passed are still re-checked.
5. **Contract gate units:** budget via `available_budget_amount` (PitchVision $1,500); a $0 budget with paid ads; prototype × MVP × build-readiness truth table (including the BUILD dead end, which fails today); sample size; signal distinctness; hypothesis overwrite; fake precision.
6. **Retry taxonomy:**
   - schema-invalid output (2 attempts);
   - transient error retried once, then a graph error;
   - permanent error not retried;
   - persistent token exhaustion: generator 2 calls, critic 4 calls today (2 after the fix);
   - validator failure on `/retry` does not dead-end.
7. **Accepted-experiment invariants:** hypothesis equals the assumption; three signals present; non-empty founder actions; valid and invalid inferences present; parameters marked AI_PROPOSED; `save_final_experiment` field mapping.
8. **Fact grounding in the loop:** the PitchVision facts (2 founders, 15 h each, $1,500, two teams and three coaches, no MVP, 0 paying, 0 revenue, no interviews) appear in both the generator payload and the critic payload; unknown facts are not presented as known.
9. **Telemetry:** the critic records dimension flags, issue count, revision trigger and revision round, with no founder text; LangSmith metadata survives `sanitize_metadata`.
10. **Ethics/consent** (if adopted): experiments that involve recording or personal data must carry a consent step in their founder actions, checked in the gate and the critic payload.
11. **M6.2.3 protection:** the full suite (854) passes, including question composition once, fact propagation and the observability tests.

**Paid, user-approved tests, after the offline tests pass:** a live A/B on the canonical fixtures plus the original PitchVision message. Compare the first-pass ACCEPT rate, the critic's six flags, the revision count, a human rubric score for the accepted experiment, and stage latencies.

---

## 9. Projected savings (all estimates, none measured)

The baseline is the trace's 108.61 s experiment loop within a 200 s run.

| Change | Estimated effect on the PitchVision trace | Basis |
|---|---|---|
| First draft accepted without revision | **−45.52 s** (gen 2 + critique 2), loop 108.61 → 63.09 s, about −23% of the run | Only to the extent first-pass quality improves; the probability is unknown until the A/B |
| Edit-the-draft revision + delta critique (revision still needed) | Unknown. If gen 2 and critique 2 each shrank by 30–50% (assumed), roughly −14 to −23 s | Gen 2 was already 26.5 s vs 42.3 s for gen 1; there is no data on edit-style revisions |
| Contract gate catches a deterministic defect before the critic | About −19 to −21 s (one Sol critique) **per caught defect**; **0 s for this trace**, because both critiques were LLM calls | Stage times from the trace |
| Slimmed facts for the generator | About −950 input tokens per generation (measured on prompt size); latency effect expected small | Output and reasoning dominate the stage time |
| Critic token-retry cap | 0 s in the normal case; avoids up to 2 extra Sol calls under persistent token exhaustion | Offline run |
| Timeout headroom | No saving; gen 1 at 42.34 s vs the 60 s timeout is a tail risk of a full repeated call | Configuration vs. trace |

**Recommended order (each step needs your approval):**
1. **Make the cause visible:** log critic dimension flags, revision trigger and token-retry use, then re-run PitchVision once to see *why* critique 1 rejected.
2. **Fix the confirmed defects:** undeclared state keys, dead pre-check keys, the BUILD/MVP dead end, and critic retry amplification.
3. **Edit-the-draft revision + delta critique** behind a flag, with the offline tests in Section 8 first.
4. **The contract gate and the generator checklist,** decided by the A/B.
