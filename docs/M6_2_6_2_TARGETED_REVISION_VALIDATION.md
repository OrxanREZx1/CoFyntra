# Startary M6.2.6.2: Controlled TARGETED_EDIT and FOCUSED Critic Validation

**Status:** implemented and verified offline (**1075 tests pass**, 0 fail, no network attempts).

**Not done:** **no paid live call was made**, nothing was committed, pushed or deployed, and `deploy-alpha` was not touched.

**Unchanged:** production prompts, routing rules, acceptance criteria, models and Sol effort.

**Starting point (your M6.2.6.1 live run, confirmed from its local artifact):**

| Item | Value |
|---|---|
| Outcome | `FULL_REGENERATION_ACCEPTED` |
| Calls / tokens / time | 3 Sol calls; 10,703 in + 4,288 out = 14,991 tokens; 59.03 s |
| First critique | REVISE on assumption_alignment, evidence_validity and falsifiability |
| Critic's own scope | TARGETED (6 issues) |
| Strategy actually used | FULL_REGENERATION, by the production rule "assumption_alignment FAIL → redesign" |

The targeted-edit path and the focused critic are still unvalidated live. ShiftRelay-C is designed to give Sol a defect it can reasonably call narrow.

## 1. The ShiftRelay-C fixture (`startary/devtools/revision_scenarios.py`)

**Scope.** Same fictional startup, critical assumption, founder facts and build readiness (`CONTINUE_MANUAL`) as `shiftrelay`. The method text is **identical** to the primary fixture's.

**Founder facts preserved:**
- 2 founders × 8 h/week;
- a **$350** budget;
- **at most two weeks**;
- access to **3 managers** (and 4 part-time workers);
- **no prototype or MVP**;
- 0 paying customers, 0 revenue;
- no access to schedules, payroll or KakaoTalk;
- anonymized notes only.

| Field | Value |
|---|---|
| experiment_type | `PROBLEM_INTERVIEW` |
| objective | Determine whether last-minute absences recur and cause concrete operational disruption for these café and restaurant managers. |
| method | 30–45 min retrospective interviews with the three managers, reconstructing every real last-minute absence of the previous eight weeks (replacement, coordination time, consequences); anonymized written notes only; no schedules, payroll data or private messages requested. |
| target_participants_or_scope | The three accessible managers (8–25 staff each); the four part-time workers as an optional second perspective. |
| target_sample_size | 3 |
| founder_actions | Schedule the 3 interviews in week 1 · walk through every absence of the past 8 weeks · per incident: how it was communicated, how the gap was filled, coordination time, consequence · decide per manager whether the qualifying pattern is met · optionally ask the part-time workers · anonymized notes, no records requested |
| evidence_to_collect | Anonymized notes per incident (week, how communicated, how replaced, coordination minutes, consequence) and, per manager, whether the qualifying pattern is met. |
| **support_signal** | At least 2 of the 3 managers meet the qualifying pattern: 2 or more last-minute absences in the past eight weeks, each with a concrete operational consequence (30+ minutes of replacement coordination, a short-staffed shift, delayed service, or unplanned overtime). |
| **contradiction_signal** | **Fewer than 2 of the 3 managers meet the qualifying pattern.** |
| **inconclusive_signal** | **Exactly 1 manager meets the qualifying pattern, or one or more of the three interviews cannot be completed.** |
| valid / invalid inferences | Recurrence and disruption for these three managers; most common consequences / *not* prevalence across Korea; *not* product adoption or payment |
| estimated_time / cost | 1–2 weeks / $30 (coffee for interviewees) |
| parameters | `qualifying_pattern` (as above), `support_threshold` (at least 2 of 3) |

**The one deliberate defect.** With exactly one qualifying manager, both the contradiction rule ("fewer than 2") and the inconclusive rule ("exactly 1") apply. The contradiction rule can also fire before all interviews are completed.

**The ideal narrow repair** changes only the contradiction and inconclusive signals:
- contradiction: 0 qualifying managers, after all three interviews are completed;
- inconclusive: exactly 1 qualifying manager, or incomplete interviews;
- support: unchanged (it already requires 2 or 3).

**Verified offline:**
- valid under `ExperimentGeneratorLLMOutput`;
- **zero contract violations, zero advisories, zero required authorizations**;
- within budget; no prototype;
- no AI-assistant question, which is the likely trigger of the `assumption_alignment` failure in M6.2.6.1.

## 2. Implementation changes

| File | Change |
|---|---|
| `startary/devtools/revision_scenarios.py` | `SHIFTRELAY_C_DRAFT`, and the `shiftrelay-c` entry in `SCENARIOS` |
| `startary/devtools/revision_probe.py` | Scenario-aware offline scripts (`offline_smoke_responders(path, scenario)`). **Reviser patch capture:** `PromptInputAuditor.on_llm_end` + `parse_structured_output` locally in live runs; the scripted LLM's recorded output offline. **New summary fields:** first-critique `critic_revision_scope` and `strategy_reason`; `revision_checks.patch_fields_returned`, `patch_fields_returned_but_unchanged`, `changed_field_count`, `core_fields_preserved` (method, participants, evidence source, type, hypothesis, assumption ID, sample, cost, time); `decision_rule_check` (ShiftRelay-C heuristic, seeded vs. final). **Artifact** additions: the raw patch and a human-evaluation checklist. **LangSmith summary span:** `first_revision_scope`, `strategy_reason`, `patch_fields_returned`, `changed_field_count`, `criteria_overlap_before`, `criteria_overlap_after` |
| `startary/domain/experiment_revision.py` (observability only) | `select_revision_strategy_with_reason()` returns the strategy plus the rule that fired. `select_revision_strategy()` delegates to it, so routing is identical (tested). The V2 critic decision span now also carries `critic_revision_scope` (an enum). |
| `tests/fixtures/scripted_llm.py` | `LLMCall.output` records what each scripted call returned (additive) |
| `tests/test_m6261_revision_probe.py` | The completeness test's exact key set now includes the six new summary fields; the assertion stays exact equality |
| `tests/test_m6262_targeted_revision_probe.py` (new) | 26 tests (Section 3) |

No production prompt, routing rule, threshold, model or effort setting was changed.

## 3. Offline test results

```
.venv/Scripts/python.exe -m pytest -m "not live"
Before: 1047 passed     After: 1075 passed, 5 deselected (live), 0 failed, 0 network attempts
```

The 28 additional tests are 26 in the new file plus 2 from existing M6.2.6.1 fixture tests that are parametrized over every scenario and now include ShiftRelay-C.

| Area | Tests |
|---|---|
| Fixture | Setup preserved (same method, 3 managers, $350, two weeks, no MVP, no AI-assistant question); contract-clean; exactly the intended overlap present; the reference repair removes it; heuristic phrasing table (6 cases) |
| Targeted path (scripted) | Call order critic → `revise_experiment` → focused critic. Scope TARGETED with reason `CRITIC_SCOPE_TARGETED`. Only `contradiction_signal` and `inconclusive_signal` changed; a returned-but-identical `support_signal` is reported separately. All core fields preserved; schema and contract valid. FOCUSED mode as expected; all six dimensions re-scored; strict rule respected; overlap gone; input audit all true |
| Real feedback reaches the next step | The reviser prompt contains the critic's actual issues, instructions and failed dimensions. The focused prompt contains both changed fields with their original "before" values, plus the original issues |
| Every possible critic outcome, with the production rule shown | REDESIGN → FULL_REGENERATION (`CRITIC_SCOPE_NOT_TARGETED`); assumption_alignment FAIL → FULL (`ASSUMPTION_ALIGNMENT_FAILED`); >3 failed dimensions → FULL (`TOO_MANY_FAILED_DIMENSIONS`); 2 failed with TARGETED → TARGETED_EDIT. First-pass ACCEPT reported honestly (overlap still present). A substantial edit goes to the FULL critic and is reported (`core_fields_preserved.method = false`), not overridden. A new blocking issue in focused mode → REVISE → failure stop. A rejected edit reaches the fallback |
| Strategy reason | Never changes the strategy (6-case equivalence) |
| Patch inspection | The live auditor parses Sol-style JSON content and tool-call arguments through real LangChain callbacks |
| Observability | Summary metadata is sanitized and contains scope, strategy, reason, patch fields, change count and overlap before/after. The production decision span carries `critic_revision_scope`. No fixture or founder text in any span |
| Artifact and CLI | Artifact keeps the raw patch and the checklist locally; CLI offline smoke for `shiftrelay-c`; `--live` without `--confirm-paid` is refused |

## 4. Manual paid live test (PowerShell; run only when you choose to)

```powershell
cd "C:\Users\orxan\OneDrive\Masaüstü\Startary_AI_Latency"

# Optional free check first:
.\.venv\Scripts\python.exe -m startary.devtools.revision_probe --offline-smoke --scenario shiftrelay-c --no-artifact

# One paid live run:
$env:STARTARY_EXPERIMENT_REVISION_V2 = "true"
.\.venv\Scripts\python.exe -m startary.devtools.revision_probe --live --confirm-paid --scenario shiftrelay-c
Remove-Item Env:STARTARY_EXPERIMENT_REVISION_V2
```

**Preconditions.** The same preflight as M6.2.6.1 applies: V2 flag, valid key, `gpt-5.6-sol`, LangSmith tracing to `startary-ai-latency`, hidden inputs and outputs. Without them the probe refuses (exit 2).

**Expected size.** Typically 3 Sol calls (critic → reviser → focused critic), or 1 if Sol accepts. The upper bound is unchanged.

## 5. Expected LangSmith trace

```
revision_probe_shiftrelay-c                          (root; run_type=revision_probe, scenario_id=m6261-shiftrelay-c)
└─ LangGraph
   ├─ validate_proposed_experiment
   ├─ critique_experiment ─ ChatOpenAI gpt-5.6-sol   (revision_round=1, critic_mode=FULL)
   │  └─ experiment_critic_decision                    critic_verdict=REVISE, dim_falsifiability=FAIL,
   │                                                   critic_revision_scope=TARGETED, revision_strategy=TARGETED_EDIT,
   │                                                   next_route=revise_experiment
   ├─ revise_experiment ─ ChatOpenAI gpt-5.6-sol     (revision_round=2, revision_strategy=TARGETED_EDIT)
   │  └─ experiment_targeted_revision                  changed_fields=[contradiction_signal, inconclusive_signal],
   │                                                   substantial_change=false, next_critic_mode=FOCUSED
   ├─ validate_proposed_experiment
   ├─ critique_experiment ─ ChatOpenAI gpt-5.6-sol   (revision_round=2, critic_mode=FOCUSED)
   │  └─ experiment_critic_decision                    critic_mode=FOCUSED, original_issues_resolved, new_blocking_issue_count,
   │                                                   remaining_failed_dimensions, dim_* (all six), next_route
   └─ save_final_experiment ─ assess_can_execute    |  handle_experiment_failure_stop
└─ revision_probe_summary                            probe_outcome, first_revision_scope, strategy_reason, changed_fields,
                                                     patch_fields_returned, second_critic_mode, criteria_overlap_before/after, …
```

The values above are **the hypothesis under test**. Sol may decide differently; whatever happens is reported as is.

## 6. Criteria for confirming TARGETED_EDIT and FOCUSED critique

**Path actually exercised.** All of these must hold:

| Where | What to check |
|---|---|
| stdout summary | `outcome` is `TARGETED_EDIT_ACCEPTED` or `TARGETED_EDIT_REJECTED`; `revision_path_exercised = true` |
| stdout summary | `first_critique.critic_revision_scope = TARGETED` and `strategy_reason = CRITIC_SCOPE_TARGETED` |
| stdout summary | `model_calls.revise_experiment = 1` and `generate_experiment_regeneration = 0` |
| stdout summary | `second_critique.critic_mode = FOCUSED` and `revision_checks.second_critic_mode_as_expected = true` |
| stdout summary | `input_audit`: `reviser_saw_original_method`, `reviser_saw_all_first_critic_issues` and `focused_critic_saw_original_issues` are all `true` |
| LangSmith | A `revise_experiment` node with a ChatOpenAI child; a second `experiment_critic_decision` with `critic_mode=FOCUSED` |

**Repair narrow and correct.** Don't rely only on the critic's ACCEPT:
- **Narrowness:** `changed_field_count` is small (ideally 2, `contradiction_signal` and `inconclusive_signal`), and `core_fields_preserved` is all `true` (method, participants, evidence source, type, hypothesis, sample, cost, time).
- **Overlap:** `decision_rule_check.final.overlap_detected = false` and `contradiction_requires_zero_qualifying = true`. This is a **text heuristic**; confirm by reading the revised rules in the artifact.
- **Schema and contract:** `revised_draft_schema_valid = true` and `revised_draft_contract_violations = []`.
- **Second critique:** `all_six_dimensions_rescored = true` and `strict_acceptance_rule_respected = true`; on ACCEPT, `original_issues_resolved = true` and `new_blocking_issue_count = 0`.
- **Human review** of the artifact's `human_evaluation_checklist`:
  - Are the rules mutually exclusive?
  - Do they test recurring, material disruption?
  - Is the plan still realistic with 3 managers, $350 and two weeks?
  - Is the change narrower than a regeneration?

**Other outcomes:**
- **`FIRST_PASS_ACCEPT`:** Sol accepted the overlapping rules. That is a critic-quality observation, and the revision path remains untested.
- **`FULL_REGENERATION_*`:** read `strategy_reason` to see which production rule chose the redesign.
- **A FULL second critique after a targeted edit:** `revision_checks.expected_second_critic_mode = FULL` means Sol's edit was substantial.

**Please share:**
- the stdout JSON;
- the artifact (`.startary_dev/revision_probe/<thread_id>.json`), which contains the seeded and revised drafts and Sol's raw patch, all fictional;
- LangSmith totals and the run tree.

## 7. Remaining limitations

- **One live run is one sample.** Sol may accept, redesign, or edit more broadly than the ideal two fields. None of these is a harness failure, and none is overridden.
- **The overlap check is a text heuristic** written for this fixture's rules; human reading is required.
- **Live reviser capture is untested live.** Capturing the reviser's prompt and patch through LangChain callbacks is tested offline with real LangChain chat-model callbacks, but your earlier live run never reached the reviser. If the structured-output format differs, `patch_fields_returned` may be empty; `changed_fields` from production telemetry is unaffected.
- **Seeding skips the first generation.** The probe stops at `assess_can_execute`.
- **Cost:** shown in LangSmith; a local estimate needs `--price-input-per-1m` and `--price-output-per-1m`.
