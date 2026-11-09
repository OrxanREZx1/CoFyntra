# Startary M6.2.6.1: Controlled Live Experiment-Revision Test Harness

**Status:** the harness is implemented and verified offline (**1047 tests pass**, 0 fail, no network attempts).

**Not done:** **no live or paid call was made**, nothing was committed, pushed or deployed, and `deploy-alpha` was not touched. The live run is for you to start manually (Section 7).

## 1. Design and entry point

**Entry point:** `python -m startary.devtools.revision_probe`. The module (`startary/devtools/revision_probe.py`) runs only when invoked; importing it has no side effects, which a test verifies.

**What it drives.** The harness uses the **production experiment subgraph** (`build_experiment_graph`) with a MemorySaver checkpointer and a fresh thread. It seeds the graph state as if `generate_experiment` had just produced the flawed draft:

```python
graph.update_state(thread_cfg, scenario_state("shiftrelay"), as_node="generate_experiment")
graph.invoke(None, thread_cfg)
```

**Why seeding is legitimate.** It skips only the first generation and the expensive understand/diagnose pipeline. Every node after it is the real code, routed by the real `route_on_experiment_critique`:

```
validate_proposed_experiment ─► critique_experiment (real Sol, V2 FULL critic)
   ├─ ACCEPT ─► save_final_experiment ─► assess_can_execute ─► END        (revision path NOT exercised)
   ├─ REVISE + TARGETED_EDIT ─► revise_experiment (real Sol) ─► validate ─► critique (FOCUSED or FULL) ─► save | failure stop
   └─ REVISE + FULL_REGENERATION ─► generate_experiment (real Sol) ─► validate ─► critique (FULL) ─► save | failure stop
```

**What it does not change:**
- No mocked critic in live mode.
- No forced verdict, and no hard-coded failed dimensions.
- No changes to prompts, thresholds or model output.

The probe only seeds state and observes. If Sol accepts, the result is reported as `FIRST_PASS_ACCEPT`, an inconclusive test of the revision path.

**What the probe adds (all observation):**
- **A root LangSmith run** `revision_probe_<scenario>` wrapping the graph run.
- **A `revision_probe_summary` span** with flat metrics (Section 6).
- **`UsageMetadataCallbackHandler`**, which captures real token usage per model.
- **`PromptInputAuditor`, a local-only callback.** It checks that the reviser received the original draft and the critic's real feedback, and that a focused critic received the original issues. Prompt text never leaves the machine; only booleans are recorded. Captured inputs are keyed by node and by `revision_round`, from production run metadata, so token-expansion retries don't confuse the audit.
- **First-critique feedback read from the checkpoint history** (the snapshot taken right before the revision step), not from a re-implementation.
- **An optional local artifact:** `.startary_dev/revision_probe/<thread_id>.json` (git-ignored). It holds the seeded draft, the final draft, the first critique's feedback, the final critic output and the summary. The scenario is fictional, and the artifact contains no keys.

**Offline smoke mode** (`--offline-smoke`) runs the same production graph with scripted responses from `tests/fixtures/scripted_llm.py`. It makes no network calls and supports four paths: `targeted`, `redesign`, `accept` and `reject`.

## 2. Test fixture (`startary/devtools/revision_scenarios.py`)

**ShiftRelay is fictional.**

**Critical assumption:** "In independent cafés and small restaurants in South Korea employing around 8–25 workers, last-minute employee absences recur and create material operational disruption for managers." Area: Pain / Urgency.

**Founder facts (EXPLICIT unless noted):**
- 2 founders; 8 h/week each (16 h/week total, INFERRED);
- **$350** USD budget;
- access to **3 managers and 4 part-time workers** willing to discuss past incidents;
- market: South Korea;
- stage IDEA (INFERRED); **no prototype, no MVP**, not launched;
- **0 paying customers, 0 revenue**;
- no structured interviews yet, though interviews about real past incidents are possible.

The founder message also states the two-week maximum and the lack of access to private schedules, payroll or KakaoTalk messages. Build readiness is `CONTINUE_MANUAL`.

**Primary draft (`--scenario shiftrelay`).** A `PROBLEM_INTERVIEW` with a sound method:
- 30–45 min retrospective interviews with the 3 managers;
- reconstructing each real last-minute absence from the past 8 weeks: replacement workflow, coordination time, operational consequences;
- anonymized notes only, with no schedules, payroll or private messages requested;
- target sample 3, cost "$30 (coffee for interviewees)", 1–2 weeks.

**The deliberate semantic flaw is in the decision criteria:**
- Support: "At least two of the three managers say an AI scheduling assistant sounds useful."
- Contradiction: "None of the three managers expresses interest in an AI scheduling assistant."
- Inconclusive: "Only one manager expresses interest in an AI scheduling assistant."

To make those criteria observable, the last interview action asks whether the AI assistant sounds useful. These criteria measure hypothetical product interest, not recurring, material absence pain.

**Verified by tests:**
- valid under `ExperimentGeneratorLLMOutput`;
- **zero contract violations, zero advisories, zero required authorizations** (no recording or personal data);
- within the $350 budget; no prototype;
- hypothesis equals the active assumption, with the assumption stamp present.

**Second predefined fixture (`--scenario shiftrelay-b`), only if Sol accepts the primary one.** It keeps the same method and removes the AI-assistant question. Its criteria cannot meaningfully fail, and they ignore recurrence and disruption:
- Support: "At least one manager recalls at least one last-minute absence in the past eight weeks."
- Contradiction: "No manager has ever experienced a last-minute absence."

It is also schema-valid and has no deterministic findings.

## 3. Real production nodes exercised

- `validate_proposed_experiment` (V2 remediation and authorization tagging; inactive for these clean fixtures)
- `critique_experiment` (V2 full critic, then the focused or full critic)
- `route_on_experiment_critique` (strict verdict; `TARGETED_EDIT` → `revise_experiment`)
- `revise_experiment` (targeted patch of the preserved rejected draft) **or** `generate_experiment` (full regeneration)
- `save_final_experiment` → `assess_can_execute`, **or** `handle_experiment_failure_stop`

All V2 rules come from `experiment_revision.py` and `experiment_contract.py`, unchanged. No production code was modified in M6.2.6.1.

## 4. Isolation and cost safeguards

| Safeguard | How |
|---|---|
| No automatic execution | Runs only via `python -m …`; no import-time work; not called by tests or installation |
| Explicit paid opt-in | `--live` **and** `--confirm-paid` are both required; `--live` alone, no flags, or `--live` together with `--offline-smoke` all refuse with exit code 2 before any client is built |
| Live preflight (all must pass) | `STARTARY_EXPERIMENT_REVISION_V2=true`; a valid `OPENAI_API_KEY`; reasoning model == `gpt-5.6-sol` (override: `--expected-reasoning-model`); `LANGSMITH_TRACING=true` with an API key; `LANGSMITH_PROJECT=startary-ai-latency`; `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` |
| Unique run | Thread `revision-probe-<scenario>-<10 hex>`; scenario ID `m6261-<scenario>`; tags `revision-probe`, `scenario:*`, `mode:*` |
| Bounded calls | Production bounds: at most 2 critic LLM calls and at most 1 revision or regeneration call, each with at most 2 attempts plus 1 token expansion. Typical live run: 2–3 Sol calls, or 1 if Sol accepts |
| No secrets | The summary, artifact and LangSmith metadata contain no keys (tested); prompt text stays local |
| Offline tests | `tests/conftest.py` blocks network access; offline smoke patches only `get_llm_client` / `get_settings` |
| Worktree only | Artifacts go under the worktree's git-ignored `.startary_dev/` |

## 5. Offline test results

```
.venv/Scripts/python.exe -m pytest -m "not live"
Before: 1013 passed      After: 1047 passed, 5 deselected (live), 0 failed, 0 network attempts
```

The new file is `tests/test_m6261_revision_probe.py` (34 tests).

| # | Requirement | Coverage |
|---|---|---|
| 1 | Seeded draft satisfies the schema | Both scenarios |
| 2 | No unintended deterministic violations | Zero violations, advisories and authorizations; budget within; no prototype; exact founder facts; the flaw is in the criteria, not the method |
| 3 | The probe invokes the production critic | First call is the V2 critic with the exact production system prompt plus the scope addendum; seeded draft in its input; no first generation; production node spans nested under the probe root |
| 4 | Scripted REVISE reaches the correct node | Call order critic → `revise_experiment` → focused critic; outcome `TARGETED_EDIT_ACCEPTED`; input audit all true |
| 5 | Scripted ACCEPT ends without revision | `FIRST_PASS_ACCEPT`, 1 call, experiment saved |
| 6 | Targeted edit changes only returned fields | Changed fields equal the patch keys; unchanged fields identical; schema valid; no contract violations |
| 7 | Focused re-critique selected | `critic_mode = FOCUSED`, as expected for a light edit |
| 8 | Full regeneration gets a full critique | Critic → generator → full critic |
| 9 | Failure fallback reachable | Focused REVISE → `handle_experiment_failure_stop`, `QUALITY_BAR_NOT_MET`, nothing saved |
| 10 | Bounded | A reviser that returns no changes → no focused call, failure stop, at most 4 total calls |
| 11 | No live calls without opt-in | 4 refused argument combinations; 8 preflight refusals (each condition); passes only with every condition; import has no side effects; the real `ChatOpenAI` is never constructed offline; the live auditor works with real LangChain callbacks |
| 12 | Metadata safe and complete | All required summary keys present; sanitized unchanged; no founder message, draft text or keys in any span; artifact and summary contain no `sk-` / `lsv2_`; artifact directory is git-ignored |
| 13 | Existing tests still pass | Full suite |

## 6. LangSmith metadata

**Existing M6.2.6 spans (unchanged):**
- **`experiment_critic_decision`:** verdict, `dim_*`, failed dimensions, revision trigger and strategy, round, critic mode, issue codes, remaining and new blocking counts, next route.
- **`experiment_targeted_revision`:** changed field names, `substantial_change`, `next_critic_mode`.
- **`experiment_remediation`:** only on consent violations.

**The new root run** `revision_probe_<scenario>` carries `run_type=revision_probe`, `scenario_id=m6261-<scenario>`, `probe_mode`, `probe_scenario`, `probe_milestone=M6.2.6.1` and `revision_mode=V2`.

**The new `revision_probe_summary` span** (flat, codes and counts only):

| Group | Fields |
|---|---|
| Probe | `probe_outcome`, `revision_path_exercised` |
| First critique | `first_verdict`, `first_dim_<6 dims>`, `first_failed_dimensions` |
| Revision | `revision_strategy`, `changed_fields` |
| Second critique | `second_critic_mode`, `second_dim_<6 dims>`, `original_issues_resolved`, `new_blocking_issue_count` |
| Result | `final_verdict`, `final_route` |
| Calls and time | `critic_llm_calls`, `revise_calls`, `regeneration_calls`, `wall_clock_s` |

**Where the remaining metrics live:**
- **Token usage, latency and cost:** LangSmith's ChatOpenAI child runs (real values).
- **In the stdout summary:** real token usage per model (via `UsageMetadataCallbackHandler`), per-node LLM latency, and a cost estimate only if you pass `--price-input-per-1m` and `--price-output-per-1m`. No prices are invented.

## 7. Exact manual commands (PowerShell, from the worktree root)

```powershell
cd "C:\Users\orxan\OneDrive\Masaüstü\Startary_AI_Latency"
```

**Offline smoke test (free, no network):**

```powershell
.\.venv\Scripts\python.exe -m startary.devtools.revision_probe --offline-smoke --scenario shiftrelay --no-artifact
# other offline paths: --smoke-path redesign | accept | reject
```

**One paid live revision-path test.** Your `.env` must already contain the LangSmith settings checked in Section 4.

```powershell
$env:STARTARY_EXPERIMENT_REVISION_V2 = "true"
.\.venv\Scripts\python.exe -m startary.devtools.revision_probe --live --confirm-paid --scenario shiftrelay
Remove-Item Env:STARTARY_EXPERIMENT_REVISION_V2
```

**Optional:** add `--price-input-per-1m <USD> --price-output-per-1m <USD>` with your current `gpt-5.6-sol` prices for a local cost estimate.

**If Sol accepts the primary draft** (`"outcome": "FIRST_PASS_ACCEPT"`), run the second fixture once. Do not re-run the primary draft repeatedly hoping for a rejection.

```powershell
$env:STARTARY_EXPERIMENT_REVISION_V2 = "true"
.\.venv\Scripts\python.exe -m startary.devtools.revision_probe --live --confirm-paid --scenario shiftrelay-b
Remove-Item Env:STARTARY_EXPERIMENT_REVISION_V2
```

**Exit codes:** 0 means the run completed (whatever the verdict), 2 means refused, 1 means error.

**Expected LangSmith trace (project `startary-ai-latency`):**

```
revision_probe_shiftrelay                       (root; run_type=revision_probe)
└─ LangGraph
   ├─ validate_proposed_experiment
   ├─ critique_experiment
   │  ├─ critique_experiment:attempt_1 ─ ChatOpenAI gpt-5.6-sol   (revision_round=1, critic_mode=FULL)
   │  └─ experiment_critic_decision                                (critic_verdict, dim_*, revision_strategy, next_route)
   ├─ revise_experiment                                            (only if REVISE + TARGETED_EDIT)
   │  ├─ revise_experiment:attempt_1 ─ ChatOpenAI gpt-5.6-sol     (revision_round=2, revision_strategy=TARGETED_EDIT)
   │  └─ experiment_targeted_revision                              (changed_fields, substantial_change, next_critic_mode)
   │     ─ or ─ generate_experiment ─ ChatOpenAI                   (only if REVISE + FULL_REGENERATION)
   ├─ validate_proposed_experiment
   ├─ critique_experiment ─ ChatOpenAI + experiment_critic_decision (critic_mode=FOCUSED|FULL, original_issues_resolved, ...)
   └─ save_final_experiment ─ assess_can_execute   |   handle_experiment_failure_stop
└─ revision_probe_summary
```

**What to share for evaluation:**
1. The JSON summary printed to stdout. It contains no secrets.
2. The artifact at the printed `artifact_path` (`.startary_dev/revision_probe/<thread_id>.json`). It holds the drafts and the critic's feedback text, all fictional.
3. From LangSmith: the root run's total latency, tokens and cost, and a screenshot of the run tree.

**How to tell whether TARGETED_EDIT and FOCUSED actually ran.** All of these must hold:
- `outcome` is `TARGETED_EDIT_ACCEPTED` or `TARGETED_EDIT_REJECTED`, and `revision_path_exercised` is `true`;
- `model_calls.revise_experiment` is 1, and LangSmith shows a `revise_experiment` node with a ChatOpenAI child (`revision_strategy=TARGETED_EDIT`);
- `second_critique.critic_mode` is `FOCUSED`, `revision_checks.second_critic_mode_as_expected` is `true`, and the second `experiment_critic_decision` span has `critic_mode=FOCUSED`;
- `input_audit.reviser_saw_original_method`, `reviser_saw_all_first_critic_issues` and `focused_critic_saw_original_issues` are all `true`.

A `FULL` second critique after a targeted edit means the edit was substantial; `revision_checks.expected_second_critic_mode` shows why. `FULL_REGENERATION_*` means Sol asked for a redesign, which is legitimate. That path exercises regeneration and the full re-critique instead of the targeted edit.

## 8. Remaining limitations

- **One live run is anecdotal.** A REVISE or ACCEPT reflects one sample of Sol's behaviour; run more before drawing conclusions.
- **The first generation is skipped by design.** First-draft quality isn't tested here. The redesign path does call the real generator.
- **Sol may choose differently than intended:**
  - It may judge the AI-assistant question as contamination, or the criteria as a method problem, and pick REDESIGN. That's a valid production decision, not a harness failure.
  - It may also accept. In that case use `shiftrelay-b`, once.
- **The probe stops at `assess_can_execute`.** The handoff composition and evidence wait aren't part of the subgraph; offline tests cover them.
- **Cost.** A dollar estimate needs your current prices; LangSmith shows the actual cost.
- **Prerequisites.** Offline smoke needs the repo's `tests/` package, so run it from the worktree root.
- **Token counts.** Startary's own telemetry estimates them; real usage comes from LangSmith and the usage callback, live only.
