# Startary M6.2.7: Safe First-Turn LangGraph Parallelization

**Status:** implemented behind `STARTARY_PARALLEL_UNDERSTANDING` (default `false`) and verified offline. **1131 tests pass** (1075 existing + 56 new), 0 fail, 0 non-loopback network connections.

**Not done:** **no paid live call was made**, nothing was committed, pushed or deployed, and `deploy-alpha` was not touched.

**Unchanged:** prompts, models, Sol reasoning effort, routing rules, the experiment system and the experiment revision V2 flag. LangSmith `HIDE_INPUTS` / `HIDE_OUTPUTS` remain `true`.

**Production latency is not claimed.** Every latency figure below is either taken from the M6.2.2 trace (measured) or projected from it. The only new timings are offline simulations with scripted delays.

**One-line result:** with the flag on, `interpret_founder_turn` and `extract_startup_state` run in the same LangGraph superstep on separate threads, but only on turns where the sequential graph would extract anyway. A new barrier node, `join_founder_turn`, merges their shared bookkeeping in sequential order before any downstream node runs. Repair and evidence turns stay strictly sequential and make no extraction call.

---

## 1. Dependency analysis (Task 1)

Read and write sets come from the current code (AST of each node's return dicts and its `state.get(...)` / `state[...]` reads). `turn_telemetry` is also mutated in place through `record_telemetry_event`.

| | `interpret_founder_turn` | `extract_startup_state` |
|---|---|---|
| **Reads** | `active_experiment`, `active_interrupt_type`, `clarification_turn_count`, `evidence_submissions`, `execution_path`, `founder_current_goal`, `founder_messages`, `is_conversational_repair`, `pending_question`, `phase`, `project_id`, `raw_context`, `recent_corrections`, `traces`, `weak_area`, `turn_telemetry` | `clarification_turn_count`, `execution_context`, `execution_path`, `founder_messages`, `pending_question`, `project_id`, `startup_progress`, `startup_state_delta`, `traces`, `turn_telemetry` |
| **Writes** | `latest_founder_turn_interpretation`, `founder_current_concern`, `founder_current_goal`, `recent_corrections`, `cannot_act_now_context`, `is_conversational_repair`\*, `execution_path`\*, `traces`, `last_trace`, `turn_telemetry` | `startup_state_delta`, `extraction_status`, `extraction_error`, `strategic_state_changed`, `traces`, `last_trace`, `turn_telemetry` |

\* Written only on conversational-repair turns.

**Cross-dependencies:**

| Check | Finding | Consequence |
|---|---|---|
| Does extraction read anything interpretation writes? | Only `execution_path` and `traces`. Interpretation writes `execution_path` only on repair turns, and repair turns are never parallel. `traces` is append-only bookkeeping. | No data dependency on a parallel turn |
| Does interpretation read anything extraction writes? | Only `traces` | No data dependency |
| Keys written by both | `traces`, `last_trace`, `turn_telemetry` | **The only hazard.** Each branch is isolated and the barrier merges them deterministically (§4) |
| In-place mutation | `record_telemetry_event` appends to the `TurnTelemetrySummary` object in state. Both nodes copy the `traces` list before appending. | Each branch gets a deep copy of the telemetry summary |
| Routing | `route_after_interpret_founder_turn` reads `is_conversational_repair`, `execution_path` and `phase`. Interpretation sets the first two only if they are already set or `is_conversational_repair_text(latest founder message)` is true. That check is deterministic, not LLM output. Interpretation never writes `phase`. | The route can be computed before interpretation (`founder_turn_route`), so no speculative extraction is ever needed |
| Order dependence of prompts | Neither node's prompt includes the other's output | Prompts are identical in both modes. A test asserts this for every LLM call over two turns |
| Shared module state | No module-level mutable state on either node's path. The LangSmith client cache is created before `invoke`. | Thread-safe |
| Node names used elsewhere | `cli.py` maps timeout errors by node name (`extract_startup_state`) | The parallel graph keeps both node names |

**Verdict: safe.** No latent state dependency was found, so there was no need to stop and report.

**One design change from the M6.2.2 proposal.** The audit suggested a single wrapper node running both calls in a `ContextThreadPoolExecutor`. Native LangGraph fan-out was used instead, because it gives three properties for free:
- each node keeps its own LangSmith run, so the Waterfall shows the overlap;
- LangGraph checkpoints a successful branch's writes as pending writes, so a retry re-runs only the failed branch;
- there is no hand-written thread pool.

---

## 2. Before / after diagrams

**Flag OFF (default; topology byte-for-byte unchanged):**

```
verify_state / merge_founder_answer / wait_for_evidence
        │
interpret_founder_turn                      (Luna, 6.97 s in the M6.2.2 trace)
        │  route_after_interpret_founder_turn
        ├── repair ───────────────► handle_conversational_repair ─► ask_and_wait_for_founder
        ├── WAITING_FOR_EVIDENCE ─► classify_waiting_for_evidence_turn
        └── otherwise ────────────► extract_startup_state   (Luna, 14.93 s)
                                          │
                                    merge_execution_state ─► assess_baseline_state_sufficiency ─► …
```

**Flag ON:**

```
verify_state / merge_founder_answer / wait_for_evidence
        │
dispatch_founder_turn   (deterministic; founder_turn_route = same predicate as the router)
        │
        ├── PARALLEL (route would be extract_startup_state) ── one superstep, two threads ──┐
        │        ├─► interpret_founder_turn ──┐                                             │
        │        └─► extract_startup_state ───┤                                             │
        │                                     ▼                                             │
        │                      join_founder_turn   (barrier: runs once, after BOTH)  ◄──────┘
        │                         ├── merge_execution_state ─► … (unchanged downstream)
        │                         └── handle_conversational_repair   (route-mismatch guard only)
        │
        └── SEQUENTIAL (repair / WAITING_FOR_EVIDENCE: these turns never extract)
                 └─► interpret_founder_turn ─► route_after_interpret_founder_turn (unchanged)
                                                ├── handle_conversational_repair
                                                └── classify_waiting_for_evidence_turn
```

---

## 3. Exact changes

| File | Change |
|---|---|
| `startary/config.py` | `parallel_understanding: bool = Field(default=False, alias="startary_parallel_understanding")` |
| `.env.example` | `STARTARY_PARALLEL_UNDERSTANDING=false`, with a comment |
| `startary/graph/graph_state.py` | Reducer `merge_understanding_branches` (merge by branch name; `{"__reset__": True}` replaces the value). New keys `founder_turn_dispatch` and `understanding_branches: Annotated[dict, merge_understanding_branches]` |
| `startary/domain/telemetry.py` | Logical stage `intake` for `dispatch_founder_turn` and `join_founder_turn` |
| `startary/graph/nodes.py` | `import copy as _copy`. `interpret_founder_turn` records `retry_count=max(0, attempt - 1)` in telemetry (previously always 0; no behaviour change). New functions: `founder_turn_route`, `dispatch_founder_turn`, `_run_understanding_branch`, `parallel_interpret_founder_turn`, `parallel_extract_startup_state`, `join_founder_turn`. **The two original node functions are not modified.** |
| `startary/graph/builder.py` | `build_graph(checkpointer=None, parallel_understanding=None)`. `None` reads the setting through `parallel_understanding_enabled()`, where only an explicit `True` enables it. New routers: `route_on_founder_turn_dispatch`, `route_after_interpret_parallel_aware`, `route_after_join_founder_turn`. The ON topology is wired only when the flag is true. |
| `startary/observability/langsmith.py` | `build_trace_config` adds root-run metadata `parallel_understanding: true/false` and the tag `understanding:parallel` / `understanding:sequential`, for A/B filtering |
| `tests/conftest.py` | Pins `STARTARY_PARALLEL_UNDERSTANDING=false` so a developer `.env` cannot change the default-mode suite |
| `tests/test_m627_parallel_understanding.py` | New file with 56 tests (§7) |

**What each new node does:**
- **`dispatch_founder_turn`:** computes `founder_turn_route(state)`. It writes `founder_turn_dispatch = {mode: PARALLEL|SEQUENTIAL, reason: UNDERSTAND_TURN|CONVERSATIONAL_REPAIR|WAITING_FOR_EVIDENCE, dispatched_at, founder_turn_count}` and resets `understanding_branches`.
- **Branch wrappers:** on a SEQUENTIAL dispatch they simply call the original node. On a PARALLEL dispatch they run the original node on a shallow state copy that holds a deep copy of `turn_telemetry`. They then move the node's `traces`, `last_trace` and `turn_telemetry` output into `understanding_branches[interpret|extract]`, as new traces, new telemetry records, `last_trace` and start/end time. All other writes pass through untouched.
- **`join_founder_turn`:**
  - raises `RuntimeError` if either branch payload is missing;
  - rebuilds `traces` and the telemetry summary in sequential order and adds its own deterministic telemetry record;
  - emits the region span and clears `understanding_branches`;
  - on a SEQUENTIAL dispatch it is unreachable, and it returns `{}` if ever called.

### LangSmith metadata (Task 7)

All three carriers below contain only numbers, booleans and enum strings. Every key passes `sanitize_metadata` unchanged, and no founder text is included (both are tested).

| Where | Fields |
|---|---|
| Root run (every turn, both modes) | `parallel_understanding`, tag `understanding:<mode>` |
| `understanding_dispatch` span (ON) | `parallel_understanding`, `dispatch_mode`, `dispatch_reason`, `fallback_to_sequential` (true on repair and evidence turns) |
| `understanding_parallel_region` span (ON, nested under the `join_founder_turn` run); the same dict is also stored as the join telemetry record's `details` | `interpretation_ms`, `extraction_ms`, `sum_of_branches_ms`, `branch_overlap_ms`, `parallel_region_ms` (dispatch → barrier wall clock), `sync_wait_ms` (fast branch waiting for slow branch), `interpretation_retry_count`, `extraction_retry_count`, `interpretation_status`, `extraction_status`, `sync_status` (`SUCCESS` / `ROUTE_MISMATCH`), `next_route`, `fallback_to_sequential` |

**Waterfall:** interpretation and extraction are separate node runs with the same parent and overlapping start/end times. An offline test captures exactly the run tree LangSmith receives and asserts the overlap (§7).

---

## 4. Synchronization and reducer strategy (Tasks 3 and 5)

- **Barrier:**
  - Both branches are tasks of one LangGraph superstep, and a superstep commits only after all its tasks finish.
  - `join_founder_turn` is reached by a conditional edge from interpretation and a plain edge from extraction, and runs **once**, in the next superstep (tested with debug-stream step numbers).
  - It also verifies that both branch results exist and fails loudly if not.
  - `merge_execution_state` runs only in the superstep after the barrier.
- **No last-writer-wins on authoritative state:**
  - Every authoritative key is written by exactly one branch, and all of them remain plain LastValue channels.
  - If a future change made both branches write the same key, LangGraph would raise `InvalidUpdateError` instead of silently keeping one value.
- **Shared keys:**
  - In parallel mode the branches never write `traces`, `last_trace` or `turn_telemetry`. They write only their own slot of `understanding_branches`, so concurrent updates cannot overwrite each other.
  - The barrier then merges:
    - `traces` as base + interpretation's + extraction's (sequential order);
    - telemetry records replayed in the same order onto a copy of the turn summary, so LLM call count, tokens and slowest node are identical to sequential (tested), then the barrier's deterministic record;
    - `last_trace` as extraction's, as in sequential.
- **No lost updates, duplicates or stale data:**
  - Neither node writes `founder_messages`.
  - Both branches read the same checkpointed pre-turn state (same founder message and turn), so turn attribution cannot diverge.
  - The scratch slot is reset by dispatch and again by the barrier.
- **No duplicate LLM calls:** each node runs once per turn. An ON-vs-OFF test compares the call multiset, the per-call prompts and the per-call tiers.

---

## 5. Flag behavior (Task 2)

**Configuration:**

| Setting | Default | Read when |
|---|---|---|
| `STARTARY_PARALLEL_UNDERSTANDING` | `false` | When `build_graph()` compiles the graph. The CLI builds once at start-up, so **restart `startary-dev` to switch.** |

**OFF:**
- Identical topology and the original node callables (asserted by test).
- No new spans and no new state keys are written.
- Two telemetry-only differences remain visible:
  - interpretation's telemetry record now carries its real `retry_count`;
  - the root run carries `parallel_understanding: false` / `understanding:sequential`.

**ON:**
- Parallel only when `founder_turn_route` returns `extract_startup_state`.
- Repair and evidence turns are dispatched SEQUENTIAL and follow the exact original path. No extraction call is made, which matches sequential behaviour.

**Independence:**
- `STARTARY_EXPERIMENT_REVISION_V2` is a separate field.
- The experiment subgraph and the experiment edges of the main graph are identical in both modes (tested).
- All V2 tests pass in both modes.

---

## 6. Checkpoint and failure handling

- **Checkpoints:**
  - ON adds two supersteps per founder turn (dispatch and barrier). Each costs a checkpoint write; with MemorySaver that is negligible.
  - No interrupt happens inside the region. Resume enters through `ask_and_wait_for_founder → merge_founder_answer → dispatch_founder_turn`.
- **Branch crash plus restart:**
  - When one branch raises, LangGraph keeps the other branch's completed writes as pending writes.
  - A freshly compiled graph on the same checkpointer (simulated restart) then resumes with `invoke(None)`, the CLI `/retry` path, and re-runs **only the failed branch**.
  - This was tested in both directions: the completed branch's LLM call count stays 1, the barrier runs once, and the final state equals an uninterrupted run.
- **Bounded retries (existing, unchanged):**
  - Interpretation makes at most 2 attempts, then uses its existing logged deterministic fallback. The retry count and `FALLBACK` status now appear in the region metadata.
  - Extraction keeps its existing token-budget retry and `FAILED` status, and propagates exactly as in sequential mode (tested: same error/next node/state/calls).
- **No new exception handling:** the wrappers and the barrier contain no `try/except`, and nothing silently falls back to another path.
- **Duplicate resume:** replaying the same interrupted checkpoint twice forks it; each fork understands the single stored answer once (tested). The existing M6.2.3 duplicate-resume and checkpoint tests also pass under ON (§7).
- **Route-mismatch guard:**
  - Impossible by construction, since the barrier re-checks the same repair flags that the dispatch predicate used.
  - If it ever fires, the barrier logs an error, sets `sync_status=ROUTE_MISMATCH`, discards `startup_state_delta` and routes to repair, so an extraction never merges on a turn the sequential graph would not have extracted.

---

## 7. Offline test results

```
.\.venv\Scripts\python.exe -m pytest -m "not live"
Before M6.2.7: 1075 passed
After  M6.2.7: 1131 passed, 5 deselected (live, opt-in), 0 failed
Socket audit (scratch plugin wrapping socket.connect): 0 non-loopback connections
tests/test_m627_parallel_understanding.py: 56 tests; 5 repeated runs, all green (timing tests use 0.3–0.6 s sleeps with 40 % slack)
```

**Whole suite with the flag forced ON** (scratch plugin, not committed): 1128 passed, 3 failed. All three failures are expected:

| Test | Why it fails under forced ON | Product behaviour |
|---|---|---|
| `test_m623_question_composition_integrity::test_resume_makes_zero_question_regeneration_calls` | Asserts that the node after `merge_founder_answer` is named `interpret_founder_turn`. In the ON graph it is `dispatch_founder_turn`, a scheduling change. | Same: zero LLM calls on resume, one composer call. The ON counterpart in the new file asserts these with the new node name. **The existing assertion was not changed.** |
| `test_m627…::test_flag_defaults_false_and_only_explicit_true_enables`, `test_flag_off_builds_the_unchanged_sequential_topology` | They test the default-OFF path, which the plugin overrides | Not applicable |

**Requirement coverage (brief Task 8):**

| # | Requirement | Test(s) |
|---|---|---|
| 1 | Flag OFF | `test_flag_defaults_false_and_only_explicit_true_enables`, `test_flag_off_builds_the_unchanged_sequential_topology`, `test_flag_off_runs_interpretation_strictly_before_extraction` |
| 2 | Flag ON | `test_flag_on_inserts_dispatch_and_barrier_only_around_understanding` |
| 3 | Concurrency with controlled delays | `test_flag_on_runs_both_branches_concurrently` (different threads, overlap ≥ 60 % of the delay, region < sum); `test_run_tree_shows_overlapping_understanding_runs_under_one_parent` |
| 4 | Sync before routing | `test_barrier_waits_for_the_slower_branch_before_routing` (same superstep, barrier once at +1, merge at +2, `sync_wait_ms` ≥ 400) |
| 5 | State preserved | `test_on_matches_off_state_prompts_and_calls_over_two_turns`: all state except bookkeeping keys, interrupt payloads, call multiset, per-call prompts and tiers equal after removing uuids, timestamps and random question-id suffixes |
| 6 | Shared keys safe | `test_shared_keys_merge_in_sequential_order_without_loss_or_duplicates`, `test_reducer_merges_branches_and_reset_replaces` |
| 7 | Provenance | `test_founder_fact_provenance_is_preserved`, plus the M6.2.3 fact-propagation class re-run under ON |
| 8 | Corrections | `test_correction_and_goal_from_interpretation_survive_alongside_extraction` |
| 9 | Negative evidence | `test_negative_evidence_and_unknowns_are_preserved` |
| 10–11 | Initial and clarification routes | `test_initial_and_clarification_turns_dispatch_parallel`; `test_dispatch_prediction_equals_post_interpretation_route` (10 cases: plain, repair phrase, repair starter, third-party question, flag, path, evidence phase as enum and as string, evidence + repair, no message) |
| 12 | Evidence route | `test_evidence_turn_stays_sequential_and_never_extracts` |
| 13 | Interrupt/resume, no regeneration | `test_resume_reaches_dispatch_with_zero_llm_calls_and_no_regeneration`; M6.2.3 composition class under ON |
| 14 | Duplicate resume | `test_duplicate_resume_of_the_same_checkpoint_is_idempotent`; M6.2.3 checkpoint/resume class under ON |
| 15 | Bounded failure | `test_interpretation_retry_is_bounded_and_reported`, `test_interpretation_exhaustion_uses_the_existing_fallback_visibly_and_identically`, `test_extraction_failure_propagates_exactly_as_in_sequential_mode`, `test_join_fails_loudly_when_a_branch_result_is_missing`, `test_join_discards_extraction_on_route_mismatch` |
| 16 | Checkpoint restore | `test_branch_failure_and_checkpoint_restore_rerun_only_the_failed_branch[interpret / extract]` |
| 17 | Diagnosis / readiness | Covered by the ON-vs-OFF equivalence over two turns: diagnosis, baseline gate, area assessments, weak-link selection and the question are all equal |
| 18 | V2 tests pass | `test_parallel_flag_is_independent_of_experiment_revision_v2`; all `test_m626*` tests pass in both modes |
| 19 | Privacy | `test_span_metadata_is_complete_and_private`, `test_root_run_carries_the_understanding_mode_for_ab_filtering`; the M6.2.3 observability class (no founder text in telemetry) under ON |
| 20 | No extra LLM calls on skip paths | Repair and evidence tests (0 extraction calls; identical call lists ON vs OFF) |

**Re-run classes:** M6.2.3's `TestSingleCompositionAndExactTranscript`, `TestCheckpointResumeSafety`, `TestPitchVisionFactPropagation` and `TestObservabilityCoverage` are subclassed with the flag ON and inherit their assertions unchanged. The one exception is the node-name test, which has an ON counterpart.

---

## 8. Risks and edge cases

1. **Provider concurrency is unmeasured.** Two simultaneous calls on one OpenAI key may run slower per call (rate limits, shared quota), which would reduce the saving. The A/B records `interpretation_ms` and `extraction_ms` per arm to detect this.
2. **Work after a crash.** If one branch crashes, the other still runs to completion (threads cannot be cancelled). Its result is kept, so `/retry` does not pay for it twice. Apart from that, there are no speculative calls.
3. **Drift of the route prediction.** `founder_turn_route` mirrors `route_after_interpret_founder_turn` and interpretation's repair check. If either changes, the parametrized property test fails. In production the `ROUTE_MISMATCH` guard discards the extraction and logs an error.
4. **New shared keys.** A future key written by both nodes fails loudly (`InvalidUpdateError`). A future *append-only* shared key must be added to the branch isolation in `_run_understanding_branch`.
5. **Interrupts inside a branch.** Neither node calls `interrupt()`. If one ever does, parallel resume semantics need a review.
6. **Persistent checkpointers and mode switching.**
   - `understanding_branches` holds telemetry dataclasses between two supersteps. `turn_telemetry` already requires these to be serializable.
   - A thread interrupted inside the parallel region and resumed by a graph built with the other flag value is unsupported. Today the CLI uses an in-process MemorySaver, so a restart cannot resume across modes.
7. **Timing precision.** Region timings use wall-clock `time.time()`, because the dispatch time must survive a checkpoint. On Windows they are accurate to a few milliseconds.
8. **Logs interleave.** In ON mode, interpretation and extraction log lines are mixed.
9. **Live non-determinism.** The equivalence proof covers the scripted PitchVision flows. Live outputs differ run to run in *both* arms, so quality must be compared at the fact and question level, not byte-for-byte.

---

## 9. Manual A/B benchmark (paid; run only with your approval)

**Pre-flight (free):**

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not live"
# Confirms the arm without printing any secret:
.\.venv\Scripts\python.exe -c "from startary.config import get_settings as g; s=g(); print('parallel', s.parallel_understanding, 'revision_v2', s.experiment_revision_v2)"
```

**Control (sequential):**

```powershell
$env:STARTARY_PARALLEL_UNDERSTANDING = "false"
.\.venv\Scripts\startary-dev.exe
```

**Treatment (parallel):**

```powershell
$env:STARTARY_PARALLEL_UNDERSTANDING = "true"
.\.venv\Scripts\startary-dev.exe
```

**Afterwards:**

```powershell
Remove-Item Env:STARTARY_PARALLEL_UNDERSTANDING
```

**Protocol:**
- Same worktree and same `.env` (models, effort, LangSmith with hidden inputs and outputs).
- Keep `STARTARY_EXPERIMENT_REVISION_V2` at the same value in both arms.
- Restart the CLI between arms.
- Use the original PitchVision first message and the **same clarification answer** in every run.
- Run at least 5 runs per arm, alternating arms.

**Collect from LangSmith (filter by the tag `understanding:parallel` / `understanding:sequential`):**

| Metric | Source |
|---|---|
| Time to first question | Root `startary_turn_*` run latency, turn 1 (turn 2 separately) |
| Understanding stage | Control: `interpret_founder_turn` + `extract_startup_state` node runs. Treatment: `parallel_region_ms` on `understanding_parallel_region` |
| Per-call slowdown under concurrency | `interpretation_ms` / `extraction_ms`, compared across arms |
| Overlap and wait | `branch_overlap_ms`, `sync_wait_ms`; the Waterfall shows the two node runs side by side |
| Calls, tokens, cost | LangSmith run totals (should be equal across arms) |
| Correctness | Extracted facts and provenance, the chosen weak link and the first question, compared across arms |
| Failures | `sync_status`, `interpretation_status`, `extraction_status`, retry counts |

**Report** the median and the range per arm. A single run shows variance, not an improvement.

---

## 10. Projected latency impact (not measured)

| Scope | Sequential (M6.2.2 trace, measured) | Parallel [PROJECTED] |
|---|---|---|
| Understanding stage, PitchVision turn 1 | 6.97 + 14.93 = 21.90 s | ≈ max(6.97, 14.93) = 14.93 s (barrier overhead ≤ 0.01 s) |
| First turn end to end | 77.46 s | ≈ 70.5 s (−6.97 s, ≈ −9 %) |
| Each later clarification turn | interpretation + extraction | ≈ the shorter of the two saved |
| Repair / evidence turns | unchanged | unchanged (sequential by design) |
| LLM calls, tokens, cost | — | unchanged (no speculative calls) |

**Caveats:**
- The 77.46 s trace predates the M6.2.3–M6.2.6 changes.
- The projection assumes per-call latency is unchanged under concurrency (risk 1).

**Offline simulation (scripted delays only, 5 runs per arm).** Delays were scaled to the measured ratio: interpretation 0.35 s, extraction 0.75 s.
- First turn: median 1.275 s sequential vs 0.926 s parallel. The saving of 0.349 s equals the shorter branch.
- Barrier overhead: median 0 ms, max 10 ms.

This shows the plumbing achieves max() instead of sum(). It says nothing about OpenAI behaviour.

---

## 11. Next parallelization candidates (Task 10, read-only)

Read and write sets come from the current code (post-M6.2.3).

| Candidate | Dependency status | Gate required | Projected saving (M6.2.2 numbers) | Cost of a miss | Assessment |
|---|---|---|---|---|---|
| **O3: `assess_startup_areas` ∥ `plan_information_gap`** | **Input-independent.** Assess reads `diagnosis`, `diagnosis_output`, `founder_messages`, `raw_context` and canonical facts (`startup_state`, `execution_context`, `startup_progress`). The planner writes `gap_plan`, `active_next_turn_decision`, `active_interrupt_type`, `context_sufficiency*`, `current_clarifying_question`, `is_sufficient` and trace keys. The only overlap is the trace keys. | Speculative: `route_on_gap_plan` runs assess only if the planner does not ask. When the planner asks, discard `area_assessments_output`. | −7.16 s on the proceed path | One wasted Sol call (~8.7 s compute and its tokens) whenever the planner asks | **Recommended next.** Reuse the M6.2.7 dispatch/barrier pattern and add an explicit discard path. Measure the planner's ask rate first. |
| **O2: speculative `diagnose_startup` during intake (turn 1)** | The prompt (raw context, founder messages, known areas, evidence) does not use extraction output, so it is input-identical on turn 1. **But** its cache/revision bookkeeping reads `extraction_status`, `strategic_state_changed` and `strategic_state_revision`, all extraction/merge outputs, and it **writes `phase`**, which interpretation reads. | Commit only if the baseline gate passes and extraction ≠ FAILED. Recompute `diagnosis_revision` after merge. Isolate `phase`. | Turn 1: up to ≈ −14.9 s more (intake + diagnosis overlap) | One wasted Sol call (~18.7 s) on baseline-incomplete turns | Larger saving, highest complexity. Needs its own flag and A/B, **after** O3. |
| `plan_information_gap` earlier | Reads `diagnosis` (the `merge_diagnosis` output) | — | — | — | Cannot move before diagnosis. It is only parallelizable against assessment (O3). |

**Recommended order:**
1. Live A/B of M6.2.7, with your approval.
2. O3 behind its own flag.
3. O2.
